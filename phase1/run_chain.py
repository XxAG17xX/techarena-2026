"""One sampler run ("chain") of the main model family (ta/gibbsc.GibbsM with the ta/gibbsn layers) per (cutoff, seed).

    python run_chain.py <cutoffs> <seeds> <part>/<parts> <sweeps> <burn> <every> [options] [prefix]

Writes cache/<prefix>_<cutoff>_s<seed>.npy: the target events' click probabilities [targets, 10], averaged over the
saved sweeps, plus the chain's progress log in runs/<prefix>_chains/. Finished (cutoff, seed) pairs are skipped, and
<part>/<parts> splits the task list between parallel workers. Needs a CUDA GPU, and the item groupings first
(make_groups.py: cache/cats3_<cutoff>_k<K>[_s<seed>].npz).

What one run does:
  1. load the events, compute the frozen inputs: the user-by-category term, the cleaned shown counts, the item's
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

Options used by the final families:
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
Options explored and left off in the final model: fb, a=, hinge=, taue=, catw, ctxc, bpr, pilog=, lin=, span, fbabs,
few, new3a, new3m=, mixll, state (see the comments where each is parsed).
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
from ta import cf, data, expo, features, gibbs_fit, gibbsc, gibbsn, mnl  # noqa: E402
from ta.data import N_ITEMS, T0  # noqa: E402
from ta.harness import auc_report, fold, log_run  # noqa: E402
from ta.paths import CACHE  # noqa: E402

# ---- command line and options (see the module docstring; options not listed there were explored and are off)
cuts, seeds, part = [int(c) for c in sys.argv[1].split(",")], [int(s) for s in sys.argv[2].split(",")], sys.argv[3]
sweeps, burn, every = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
k, parts = map(int, part.split("/"))
opts = sys.argv[7].split(",") if len(sys.argv) > 7 else []
# flags (fb: momentum on the 10 categories instead of the fine groups)
clean, pre_rate, user_prior, use_fb = "clean" in opts, "pre" in opts, "user" in opts, "fb" in opts
floor = float(next((x[6:] for x in opts if x.startswith("floor=")), 0.0))
fbk = int(next((x[4:] for x in opts if x.startswith("fbk=")), 0))
catseed = next((x[8:] for x in opts if x.startswith("catseed=")), None)
# smoothing of the user mixes towards the overall shares
smooth = float(next((x[2:] for x in opts if x.startswith("a=")), 20.0))
asof = "asof" in opts
# prior of the mix: floor lam plus a personal part Dirichlet(pa * overall)
lam = float(next((x[4:] for x in opts if x.startswith("lam=")), 0.0))
pa = float(next((x[3:] for x in opts if x.startswith("pa=")), 8.0))
# term on the absolute share of the category, bend at abs / K
abs_h = float(next((x[4:] for x in opts if x.startswith("abs=")), 0.0))
abs_w = float(next((x[5:] for x in opts if x.startswith("absw=")), 0.7))
# personal liking of a category beyond the mix
tau_l = float(next((x[5:] for x in opts if x.startswith("taue=")), 0.0))
save_state = "state" in opts  # also save the posterior means of the vectors, biases and mixes
# strength of the category term per category; the time-of-day table inside the chain
cat_w, ctx_chain = "catw" in opts, "ctxc" in opts
use_bpr = "bpr" in opts  # BPR exposure scores as fixed terms, finer than the groups
# term on the log of the personal part of the mix
pilog = float(next((x[6:] for x in opts if x.startswith("pilog=")), 0.0))
lin = float(next((x[4:] for x in opts if x.startswith("lin=")), 0.0))  # term straight in the share of the category
# span: how fast the user reached 20 warm-up clicks, and users with no events in the fit period, in the no-click prior
use_span = "span" in opts
# fbabs: the pull of recent clicks on the absolute share; few: prior bands for users with almost no known clicks
fb_abs, few = "fbabs" in opts, "few" in opts
new3 = "new3" in opts  # extra pull of the fine groups of the newest 3 clicks, with its own sampled weight
# the same pull, matches averaged over two independent fine-group estimates (seed-0 and seed-1 k=fbk files)
new3a = "new3a" in opts
# the same pull averaged over this many fine-group estimates (seeds 0 ... n-1)
new3m = int(next((x[6:] for x in opts if x.startswith("new3m=")), 0))
consistent_mix_ll = "mixll" in opts  # preserve history item offsets in the mix-update MH ratio
# fit: the floor of the mix goes with the share of items, uniform draws over the catalogue
no_rel, f_items = "norel" in opts, "fit" in opts
mixk = int(next((x[2:] for x in opts if x.startswith("k=")), 10))  # groups of the sampled mix
# bend of the category term; floor=-1 puts the same bend in the momentum and fine terms too
hinge = float(next((x[6:] for x in opts if x.startswith("hinge=")), 0.0))
cfile = lambda c, k: CACHE / (f"cats3_{c}_k{k}_s{catseed}.npz" if catseed else f"cats3_{c}_k{k}.npz")
prefix = sys.argv[8] if len(sys.argv) > 8 else ("gibbsf" if clean else "gibbsm")
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
    # Item groups of the sampled mix, and the first two utility terms: the user's category term and the category's
    # click rate. xe / xg hold the frozen extra terms of fit / target rows [rows, 10, terms]; w0 their starting weights.
    z = np.load(cfile(c, mixk))["z"]
    K = int(z.max()) + 1
    xe, xg, table, mix = gibbsc.terms(A, fit, tgt, nu, z, K)
    C0 = gibbsc.shown_counts(A, fit, nu, z, K) + gibbsc.shown_counts(A, tgt, nu, z, K)
    g = C0.sum(0) / C0.sum()
    if clean:  # category counts with the click feedback loop taken out
        C0, g, wf = gibbsc.clean_counts(A, np.r_[fit, tgt], nu, z, K, a=smooth)
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
    if use_fb:
        th_ = (C0 + smooth * g) / (C0.sum(1, keepdims=True) + smooth)
        xe, xg = (
            np.concatenate([xe, gibbsc.feedback_term(A, fit, z, K, th_, g, wf, floor or 0.7)[..., None]], -1),
            np.concatenate([xg, gibbsc.feedback_term(A, tgt, z, K, th_, g, wf, floor or 0.7)[..., None]], -1),
        )
        w0 = w0 + (0.3,)
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
        if new3a or new3m:
            zfs = [np.load(CACHE / f"cats3_{c}_k{fbk}.npz")["z"]] + [
                np.load(CACHE / f"cats3_{c}_k{fbk}_s{j}.npz")["z"] for j in range(1, max(new3m, 2))
            ]
            ra = lambda rows: np.mean([gibbsc.recent_match(A, rows, zz) for zz in zfs], 0).astype(np.float32)
            xe, xg = np.concatenate([xe, ra(fit)[..., None]], -1), np.concatenate([xg, ra(tgt)[..., None]], -1)
            w0 = w0 + (0.05,)
    if fb_abs:
        th_ = (C0 + smooth * g) / (C0.sum(1, keepdims=True) + smooth)
        gu_ = np.ones(K) / K
        xe, xg = (
            np.concatenate([xe, gibbsc.feedback_term(A, fit, z, K, th_, gu_, wf, -abs_h)[..., None]], -1),
            np.concatenate([xg, gibbsc.feedback_term(A, tgt, z, K, th_, gu_, wf, -abs_h)[..., None]], -1),
        )
        w0 = w0 + (0.2,)
    if ctx_chain:
        xe, xg = (
            np.concatenate([xe, gibbsc.context_shift(A, fit, fit, z, K)[..., None]], -1),
            np.concatenate([xg, gibbsc.context_shift(A, fit, tgt, z, K)[..., None]], -1),
        )
        w0 = w0 + (1.0,)
    if use_bpr:
        be_, bt_ = expo.standard_scores(c)
        xe, xg = np.concatenate([xe, be_], -1), np.concatenate([xg, bt_], -1)
        w0 = w0 + (0.05, 0.05, 0.02)
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
        slate_mh=True,
        fit_scales=True,
        seed=s,
        xe=xe,
        xt=xg,
        w0=w0,
        z=z,
        table=table,
        pi=mix,
        pop=gibbsc.popularity(A, fit),
        C0=C0,
        g=g,
        a=smooth,
        K=K,
        floor=-hinge if hinge else floor,
        lam=lam,
        a_p=pa,
        abs_h=abs_h,
        abs_w=abs_w,
        no_rel=no_rel,
        f_share=f_share,
        h_share=h_share,
        lin=lin,
        pilog=pilog,
        tau_l=tau_l,
        cat_w=cat_w,
    )
    if pre_rate:
        G.set_item_pre(x_item)
    G.consistent_mix_ll = consistent_mix_ll
    if user_prior:
        facts_ = gibbsc.user_facts(A, fit, au, pre, nu)
        if few:
            # warm-up clicks only: clicks in the window are in the likelihood already
            k_ = np.bincount(au[pre], minlength=nu)
            facts_ = np.c_[facts_, k_ == 0, (k_ >= 1) & (k_ < 3)].astype(np.float32)
        if use_span:
            k_, first_ = np.bincount(au[pre], minlength=nu), np.full(nu, float(T0))
            np.minimum.at(first_, au[pre], at[pre].astype(float))
            span_ = (k_ >= 20) * np.log((T0 - first_) / 86400.0 + 0.1)
            facts_ = np.c_[facts_, span_, np.bincount(A["user"][fit], minlength=nu) == 0].astype(np.float32)
        G.set_user_prior(facts_)
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
    if save_state:
        np.savez(
            CACHE / f"state_{prefix}_{c}_s{s}.npz",
            P=(G.Pm / G.n).cpu().numpy(),
            Q=(G.Qm / G.n).cpu().numpy(),
            b=G.b.cpu().numpy(),
            c=G.c.cpu().numpy(),
            mix=G.mix.cpu().numpy(),
            eps=G.eps.cpu().numpy(),
        )
    tmp = f.with_name(f"{f.stem}.tmp{os.getpid()}.npy")  # write, then rename: parallel workers never see half a file
    np.save(tmp, p)
    os.replace(tmp, f)
    (ROOT / "runs" / f"{prefix}_chains" / f"{c}_s{s}.json").write_text(json.dumps(hist, default=float))
    scores = dict(auc_report(p, lab), start=auc_report(st["p"], lab)["auc"]) if (lab >= 0).all() else {}
    log_run(
        f"{prefix}_chain",
        cutoff=c,
        seed=s,
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
