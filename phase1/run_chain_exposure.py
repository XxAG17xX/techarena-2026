"""One sampler run of the exposure family (ta/gibbsx.GibbsX) per (cutoff, seed): the choice model plus the three
exposure scores of ta/expo.py as frozen utility terms with sampled weights.

    python run_chain_exposure.py <cutoffs> <seeds> <part>/<parts> <sweeps> <burn> <every> [new3] [prefix]

Writes cache/<prefix>_<cutoff>_s<seed>.npy (default prefix gibbsx). Needs a CUDA GPU and the exposure scores first
(make_exposure.py: cache/expo_mean_<cutoff>.npz). The final file uses this family as "gibbsx": evaluation cutoff
1737936000, 2000 sweeps, 500 burn-in, every 2, without the new3 option.

new3 adds the newest-3 fine-group matches (seed-0 groupings with 20 groups) as a 4th term with its own sampled weight
(prefix gibbsxn3; tested, not in the final file).
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
from ta import cf, data, expo, features, gibbs_fit, gibbsc, gibbsx, mnl  # noqa: E402
from ta.data import N_ITEMS, T0  # noqa: E402
from ta.harness import auc_report, fold, log_run  # noqa: E402
from ta.paths import CACHE  # noqa: E402

cuts, seeds, part = [int(c) for c in sys.argv[1].split(",")], [int(s) for s in sys.argv[2].split(",")], sys.argv[3]
sweeps, burn, every = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
k, parts = map(int, part.split("/"))
opts = sys.argv[7].split(",") if len(sys.argv) > 7 else []
prefix = sys.argv[8] if len(sys.argv) > 8 else "gibbsx"
todo = [(c, s) for c in cuts for s in seeds][k::parts]
A = data.load()
EF, CFa = features.load(A)
nu = int(A["user"].max()) + 1
T = mnl.Tensors(A, EF, CFa, torch.device("cuda"))
(ROOT / "runs" / f"{prefix}_chains").mkdir(parents=True, exist_ok=True)
for c, s in todo:
    f = CACHE / f"{prefix}_{c}_s{s}.npy"
    if f.exists():
        continue
    t0 = time.time()
    np.random.seed(s)
    fit, tgt = fold(A, c)
    xe, xg = expo.standard_scores(c)  # user-item exposure affinity, history-profile affinity, item bias
    w0 = (0.18, 0.17, 0.04)  # their starting weights
    if "new3" in opts:
        zf = np.load(CACHE / f"cats3_{c}_k20.npz")["z"]
        xe = np.concatenate([xe, gibbsc.recent_match(A, fit, zf)[..., None]], -1)
        xg = np.concatenate([xg, gibbsc.recent_match(A, tgt, zf)[..., None]], -1)
        w0 = w0 + (0.05,)
    st = gibbs_fit.point_start(A, EF, CFa, c, nu, fit, tgt, s, T)
    au, ai, at = cf.known_clicks(A, fit, tgt)
    pre = at < T0
    q = np.bincount(A["cand"][fit].ravel(), minlength=N_ITEMS) + 1.0
    g0, bm = float(np.median(st["ge"])), float(st["b"].mean())
    ev = dict(u=A["user"][fit], c=A["cand"][fit], y=A["label"][fit], o=st["ge"] - g0)
    tg = dict(u=A["user"][tgt], c=A["cand"][tgt], o=st["gt"] - g0)
    # History clicks' unknown slates are drawn by how often items are shown (q), and kept by a Metropolis step.
    G = gibbsx.GibbsX(
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
        pre_outside=False,
        chunk=150000,
        slate_mh=True,
        fit_scales=True,
        seed=s,
        xe=xe,
        xt=xg,
        w0=w0,
    )
    # the repeat-click offset starts switched off; its multiplier is sampled
    G._set_scales(0.0, 1.0)
    lab = A["label"][tgt]
    hist = G.run(
        sweeps,
        burn,
        every=every,
        label=lab if (lab >= 0).all() else None,
        log_every=100,
        pause=float(os.environ.get("TA_GIBBS_PAUSE", "0")),
    )
    p = G.mean_prediction()
    tmp = f.with_name(f"{f.stem}.tmp{os.getpid()}.npy")
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
