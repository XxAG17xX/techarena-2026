"""Settings and BPR factors for the sampler's starting point.

The name is historical: these pieces came from the earlier gradient-boosted feature stack, which the final model no
longer uses. What is left is what ta/gibbs_fit.py needs to start a sampler run: the BPR settings and cached BPR fits
(the 8-dim starting vectors come from them) and the settings of the point-estimate choice model (ta/mnl.py) that is
trained from those vectors first.
"""

import os
import numpy as np

from ta import cf
from ta.harness import fold
from ta.paths import CACHE

BPR = dict(k=128, lr=0.02, reg=0.01, iters=400, seeds=(0,))
# The point-estimate choice model is built up in steps; each line adds to the one before.
# 8-dim cosine choice model on user vectors only, plus a sampled-softmax loss on clicks known only from histories
MNL8A_CFG = dict(
    use_P=True,
    use_Y=False,
    init_Y=True,
    cosine=True,
    alpha0=5.0,
    aux_w=1.0,
    l2_item=0.03,
    l2_bias=0.03,
    lr=5e-4,
    epochs=18,
)
# + confidence shrinkage of user and item directions by their known clicks
MNL8C_CFG = {**MNL8A_CFG, "conf": True, "conf_item": True, "epochs": 24}
# + history-click competitors drawn in proportion to item exposure, and settings from a hyperparameter search:
# stronger user and item penalties, sharpness start 3, 32 epochs, batch 4096
MNL8H_CFG = {
    **MNL8C_CFG,
    "aux_q": True,
    "epochs": 32,
    "l2_item": 0.1,
    "l2_bias": 0.03,
    "l2_user": 0.1,
    "batch": 4096,
    "alpha0": 3.0,
}
# The final start: those settings, 40 epochs, started from an aligned average of 5 BPR fits. The seed picks which
# 5 BPR fits are averaged (0: 0-4, 1: 5-9, 2: 10-14, 3: 15-19, then a random 5 of 0-19), so runs differ in start.
MNL8T_CFG = {**MNL8H_CFG, "epochs": 40}


def avg_bpr(seed):
    """The BPR seeds whose fits are averaged into the start of a sampler run with this seed."""
    if seed < 4:
        return tuple(range(5 * seed, 5 * seed + 5))
    return tuple(sorted(int(k) for k in np.random.default_rng(seed).choice(20, 5, replace=False)))


def _factors(A, c, n_users, kind, seed=0):
    """BPR (or SVD) factors of the known clicks before cutoff c, cached in cache/<kind>_<c>[_s<seed>].npz."""
    f = CACHE / f"{kind}_{c}{f'_s{seed}' if seed else ''}.npz"
    if f.exists():
        z = np.load(f)
        return z["U"], z["V"]
    fit, tgt = fold(A, c)
    X = cf.click_matrix(A, fit, tgt, n_users)
    if kind == "bpr":
        U, V = cf.bpr_factors(X, BPR["k"], lr=BPR["lr"], reg=BPR["reg"], iters=BPR["iters"], seed=seed)
    elif kind == "bpr256":  # bigger BPR: slightly better 8-dim starts (jan24 +0.0015 single)
        U, V = cf.bpr_factors(X, 256, lr=BPR["lr"], reg=BPR["reg"], iters=BPR["iters"], seed=seed)
    else:
        U, V = cf.svd_factors(X, 64)
    tmp = f.with_name(f"{f.stem}.tmp{os.getpid()}.npz")  # atomic: parallel workers poll for these files
    np.savez(tmp, U=U, V=V)
    os.replace(tmp, f)
    return U, V
