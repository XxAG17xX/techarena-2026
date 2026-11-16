"""One sampler run ("chain") of the main model family (ta/gibbsc.GibbsM with the ta/gibbsn layers) per (cutoff, seed).

    python run_chain.py <cutoffs> <seeds> <part>/<parts> <sweeps> <burn> <every> <options> <prefix>

Writes cache/<prefix>_<cutoff>_s<seed>.npy: the target events' click probabilities [targets, 10], averaged over the
saved sweeps, plus the chain's progress log in runs/<prefix>_chains/ and one journal record (with the options) in
runs/journal.jsonl. Finished (cutoff, seed) pairs are skipped, and <part>/<parts> splits the task list between
parallel workers. Needs a CUDA GPU, and the item groupings first (make_groups.py:
cache/cats3_<cutoff>_k<K>[_s<seed>].npz).

What one run does:
  1. load the events, compute the frozen inputs: the category click rate, the cleaned shown counts, the item's
     history-click rate, the momentum, newest-3 and as-of terms;
  2. find the starting point (ta/gibbs_fit.point_start: BPR, then the point-estimate choice model);
  3. build the sampler class and run it for <sweeps> sweeps, discarding the first <burn> and averaging every
     <every>-th sweep after that into the prediction.

The final file uses ten families of this script, all at the evaluation cutoff 1737936000 with 1500 sweeps, 400 burn-in
and every 2 (the 11th family comes from run_chain_exposure.py). Option sets:

    Q1 = clean,pre,floor=0.7,user,fbk=20,asof,lam=0.4,pa=8,fit,abs=1.5,absw=0.7
    Q3 = clean,pre,user,fbk=20,asof,lam=0.4,pa=8,fit,floor=-1,abs=1.5,absw=1,norel
    PT = clean,pre,floor=0.7,user,fbk=20,lam=0.4,pa=8,fit,abs=1.5,absw=0.7

    prefix (family)  options
    gibbsq1          Q1                  the main family
    gibbsq1c1        Q1,catseed=1        the same on a second, independent estimate of the item groupings
    gibbsq1c2        Q1,catseed=2        ... and a third
    gibbsq1n3        Q1,new3             + the extra pull of the newest 3 clicks
    gibbsq3          Q3                  only the absolute-share category term, hinge form for the fine terms
    gibbsq3c1        Q3,catseed=1
    gibbsq3c2        Q3,catseed=2
    gibbsq3n3        Q3,new3
    gibbsqi          PT,k=20             the user's mix sampled over 20 groups instead of 10, no as-of term
    gibbsqin3        PT,k=20,new3

Options:
  clean      shown counts cleaned of the click feedback loop (ta/gibbsc.clean_counts)
  pre        term: the item's history clicks against what its showings lead to expect (ta/gibbsc.item_pre_rate)
  floor=F    category term log(mix / share + F); F < 0 gives the hinge log(max(mix / share, -F))
  user       the no-click prior's mean depends on facts about the user (ta/gibbsc.user_facts)
  fbk=K      momentum: boost of the fine groups (K groups) of the user's recent clicks (ta/gibbsc.feedback_term)
  asof       term: how much the user's rows so far favour the fine group over its category (ta/gibbsc.asof_fine)
  lam=L,pa=A the mix = L * floor + (1 - L) * a personal part with a Dirichlet(A * share) prior
  fit        the floor of the mix is each category's share of items: the uniform draws over the catalogue
  abs=H      term on the absolute share of the category, abs_w * log(max(K * mix, H)); absw=W sets abs_w
  norel      drop the relative term log(mix / share ...), keep only the absolute one
  catseed=S  use the item groupings built with seed S (independent estimates of the same structure)
  new3       term: how many of the newest 3 clicks share the row's fine group (ta/gibbsc.recent_match)
  k=K        number of groups of the sampled mix (default 10)
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from ta import cf, data, features, gibbs_fit, gibbsc, gibbsn, mnl  # noqa: E402
from ta.data import N_ITEMS, T0  # noqa: E402
from ta.harness import auc_report, fold, log_run  # noqa: E402
from ta.paths import CACHE  # noqa: E402

# ---- command line and options (see the module docstring)
if len(sys.argv) != 9:
    sys.exit("usage: python run_chain.py <cutoffs> <seeds> <part>/<parts> <sweeps> <burn> <every> <options> <prefix>")
cuts, seeds, part = [int(c) for c in sys.argv[1].split(",")], [int(s) for s in sys.argv[2].split(",")], sys.argv[3]
sweeps, burn, every = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
k, parts = map(int, part.split("/"))
opts, prefix = sys.argv[7].split(","), sys.argv[8]
clean, pre_rate, user_prior = "clean" in opts, "pre" in opts, "user" in opts
floor = float(next((x[6:] for x in opts if x.startswith("floor=")), 0.0))
fbk = int(next((x[4:] for x in opts if x.startswith("fbk=")), 0))
catseed = next((x[8:] for x in opts if x.startswith("catseed=")), None)
smooth = 20.0  # smoothing of the user mixes towards the overall shares
asof = "asof" in opts
# prior of the mix: floor lam plus a personal part Dirichlet(pa * overall)
lam = float(next((x[4:] for x in opts if x.startswith("lam=")), 0.0))
pa = float(next((x[3:] for x in opts if x.startswith("pa=")), 8.0))
# term on the absolute share of the category, bend at abs / K
abs_h = float(next((x[4:] for x in opts if x.startswith("abs=")), 0.0))
abs_w = float(next((x[5:] for x in opts if x.startswith("absw=")), 0.7))
new3 = "new3" in opts  # extra pull of the fine groups of the newest 3 clicks, with its own sampled weight
# norel: no relative category term; fit: the floor of the mix goes with the share of items (uniform draws)
no_rel, f_items = "norel" in opts, "fit" in opts
mixk = int(next((x[2:] for x in opts if x.startswith("k=")), 10))  # groups of the sampled mix
cfile = lambda c, k: CACHE / (f"cats3_{c}_k{k}_s{catseed}.npz" if catseed else f"cats3_{c}_k{k}.npz")
todo = [(c, s) for c in cuts for s in seeds][k::parts]

# ---- data, once per worker: event arrays, as-of features, and the same on the GPU
A = data.load()
EF, CFa = features.load(A)
nu = int(A["user"].max()) + 1
T = mnl.Tensors(A, EF, CFa, torch.device("cuda"))
(ROOT / "runs" / f"{prefix}_chains").mkdir(parents=True, exist_ok=True)
# ---- one sampler run per (cutoff, seed)
for c, s in todo:
    f = CACHE / f"{prefix}_{c}_s{s}.npy"
    if f.exists():
        continue
    t0 = time.time()
    np.random.seed(s)
    fit, tgt = fold(A, c)
    # Item groups of the sampled mix. xe / xg hold the frozen extra terms of fit / target rows [rows, 10, terms] and
    # w0 their starting weights. Column 0 is a placeholder that GibbsM fills with the live category term of the
    # sampled mix; column 1 is the category's click rate.
    z = np.load(cfile(c, mixk))["z"]
    K = int(z.max()) + 1
    xe, xg = gibbsc.terms(A, fit, tgt, z, K)
    C0 = gibbsc.shown_counts(A, fit, nu, z, K) + gibbsc.shown_counts(A, tgt, nu, z, K)
    g = C0.sum(0) / C0.sum()
    if clean:  # category counts with the click feedback loop taken out
        C0, g, _ = gibbsc.clean_counts(A, np.r_[fit, tgt], nu, z, K, a=smooth)
    # Uniform draws: the floor of the mix is each group's share of the catalogue, and the personal part's prior is
    # centred on what is left of the overall shares.
    f_share = np.bincount(z, minlength=K) / N_ITEMS if f_items else None
    h_share = (
        None if f_share is None else np.maximum(g - lam * f_share, 1e-3) / np.maximum(g - lam * f_share, 1e-3).sum()
    )
    st = gibbs_fit.point_start(A, EF, CFa, c, nu, fit, tgt, s, T)  # BPR, then the point-estimate choice model
    au, ai, at = cf.known_clicks(A, fit, tgt)
    pre = at < T0  # clicks before the training window: known only from histories
    w0 = (1.0 if floor else 0.7, 0.1)
    if pre_rate:
        x_item = gibbsc.item_pre_rate(A, fit, ai, pre)
        xe, xg = (
            np.concatenate([xe, x_item[A["cand"][fit]][..., None]], -1),
            np.concatenate([xg, x_item[A["cand"][tgt]][..., None]], -1),
        )
        w0 = w0 + (0.25,)
    if fbk:  # momentum on the fine groups, from the user's recent clicks
        zf = np.load(cfile(c, fbk))["z"]
        Kf = int(zf.max()) + 1
        Cf, gf, wff = gibbsc.clean_counts(A, np.r_[fit, tgt], nu, zf, Kf, a=smooth)
        thf = (Cf + smooth * gf) / (Cf.sum(1, keepdims=True) + smooth)
        xe, xg = (
            np.concatenate([xe, gibbsc.feedback_term(A, fit, zf, Kf, thf, gf, wff, floor or 0.7)[..., None]], -1),
            np.concatenate([xg, gibbsc.feedback_term(A, tgt, zf, Kf, thf, gf, wff, floor or 0.7)[..., None]], -1),
        )
        w0 = w0 + (0.5,)
        if asof:  # fine group vs category in the user's rows so far
            fe_, ft_ = gibbsc.asof_fine(A, fit, tgt, zf, Kf, gf, z, K, g, floor or 0.7, smooth)
            xe, xg = np.concatenate([xe, fe_[..., None]], -1), np.concatenate([xg, ft_[..., None]], -1)
            w0 = w0 + (0.1,)
        if new3:  # newest-3 clicks in the row's fine group
            xe, xg = (
                np.concatenate([xe, gibbsc.recent_match(A, fit, zf)[..., None]], -1),
                np.concatenate([xg, gibbsc.recent_match(A, tgt, zf)[..., None]], -1),
            )
            w0 = w0 + (0.05,)
    # Start values: the repeat-click offsets are centred on their median and the item biases on their mean. The no-click
    # utilities shift by the same amounts and absorb the mean no-click offset of the event features (the sampler keeps
    # no per-event no-click offset), so the start stays close to the point estimate.
    q = np.bincount(A["cand"][fit].ravel(), minlength=N_ITEMS) + 1.0
    g0, bm = float(np.median(st["ge"])), float(st["b"].mean())
    ev = dict(u=A["user"][fit], c=A["cand"][fit], y=A["label"][fit], o=st["ge"] - g0)
    tg = dict(u=A["user"][tgt], c=A["cand"][tgt], o=st["gt"] - g0)
    # The sampler: GibbsM with the four gibbsn layers, starting from the point estimate, sharpness 7.
    G = gibbsn.make(base=gibbsc.GibbsM, outside=True, slate_b=True, item_pre=pre_rate, user_prior=user_prior)(
        T.u.device,
        nu,
        N_ITEMS,
        ev,
        (au[pre], ai[pre]),
        tg,
        st["U"],
        st["V"],
        st["b"] - bm,
        st["c"] - bm - g0 + st["n0"],
        7.0,
        q=q,
        pre_outside=True,
        chunk=150000,
        fit_scales=True,
        seed=s,
        xe=xe,
        xt=xg,
        w0=w0,
        z=z,
        pop=gibbsc.popularity(A, fit),
        C0=C0,
        g=g,
        a=smooth,
        K=K,
        floor=floor,
        lam=lam,
        a_p=pa,
        abs_h=abs_h,
        abs_w=abs_w,
        no_rel=no_rel,
        f_share=f_share,
        h_share=h_share,
    )
    if pre_rate:
        G.set_item_pre(x_item)
    if user_prior:
        G.set_user_prior(gibbsc.user_facts(A, fit, au, pre, nu))
    # the repeat-click offset starts switched off; its multiplier is sampled
    G._set_scales(0.0, 1.0)
    lab = A["label"][tgt]
    # TA_GIBBS_PAUSE: seconds of idle time per sweep (keeps a laptop GPU cool)
    hist = G.run(
        sweeps,
        burn,
        every=every,
        label=lab if (lab >= 0).all() else None,
        log_every=100,
        pause=float(os.environ.get("TA_GIBBS_PAUSE", "0")),
    )
    p = G.mean_prediction()
    tmp = f.with_name(f"{f.stem}.tmp{os.getpid()}.npy")  # write, then rename: parallel workers never see half a file
    np.save(tmp, p)
    os.replace(tmp, f)
    (ROOT / "runs" / f"{prefix}_chains" / f"{c}_s{s}.json").write_text(json.dumps(hist, default=float))
    scores = dict(auc_report(p, lab), start=auc_report(st["p"], lab)["auc"]) if (lab >= 0).all() else {}
    log_run(
        f"{prefix}_chain",
        cutoff=c,
        seed=s,
        opts=sys.argv[7],
        sweeps=sweeps,
        burn=burn,
        every=every,
        sec=round(time.time() - t0),
        alpha=hist[-1]["alpha"],
        sg=hist[-1]["sg"],
        w=hist[-1]["w"],
        s_c=hist[-1]["s_c"],
        s_b=hist[-1]["s_b"],
        **scores,
    )
    del G
    torch.cuda.empty_cache()
print("worker done", part, flush=True)
