"""Exposure model: BPR on what each user was SHOWN (fit events, rows outside the user's history, no click labels).

What the platform shows a user is not random: it follows a personal mix of item categories. That exposure affinity is
unrelated to the 8-dim taste score, yet at a fixed taste score it moves the click rate strongly, and shown rows hold ten
times more data than clicks. These scores are the extra terms of the exposure sampler family (run_chain_exposure.py).

For a fit event the score must not have seen that event's own slate, so fit events are split in two and scored by the
model of the other half. Targets are scored by the mean of both half models, which keeps one scale everywhere.
"""

import numpy as np
import scipy.sparse as sp

from ta import cf
from ta.data import N_ITEMS
from ta.paths import CACHE


def shown_matrix(A, ev, nu):
    """Sparse user-by-item count of rows shown in events `ev`, leaving out items already in the user's click history
    (those slots replay a past click and say nothing about the user's mix)."""
    cand = A["cand"][ev]
    inh = (A["hist"][ev][:, None, :] == cand[:, :, None]).any(-1)
    X = sp.csr_matrix(
        (
            np.ones(int((~inh).sum()), np.float32),
            (np.repeat(A["user"][ev], 10)[~inh.ravel()], cand.ravel()[~inh.ravel()]),
        ),
        shape=(nu, N_ITEMS),
    )
    X.sum_duplicates()
    return X


def _scores(A, ev, U, V):
    """[n, 10, 3]: user-item affinity, history-profile affinity, item bias (BPR's last column is the bias)."""
    uu, cc = A["user"][ev], A["cand"][ev]
    out = np.empty((len(ev), 10, 3), np.float32)
    for s in range(0, len(ev), 100000):
        j = slice(s, s + 100000)
        Vc = V[cc[j]]
        h = A["hist"][ev[j]]
        w = (h >= 0)[..., None].astype(np.float32)
        prof = (V[np.maximum(h, 0), :-1] * w).sum(1) / np.maximum(w.sum(1), 1)
        out[j, :, 0] = np.einsum("ed,ejd->ej", U[uu[j], :-1], Vc[:, :, :-1])
        out[j, :, 1] = np.einsum("ed,ejd->ej", prof, Vc[:, :, :-1])
        out[j, :, 2] = Vc[:, :, -1]
    return out


def fit_scores(A, c, fit, tgt, nu, k=16, seed=0, iters=80, cache=True, keep=None):
    """(scores of fit events from the other half's model [fit, 10, 3], scores of targets [tgt, 10, 3]); cached per
    cutoff."""
    f = CACHE / f"expo_k{k}_{c}_s{seed}.npz"
    if cache and f.exists():
        z = np.load(f)
        return z["xf"], z["xt"]
    part = np.random.default_rng(seed + 991).integers(2, size=len(fit)).astype(bool)
    xf, xt = np.empty((len(fit), 10, 3), np.float32), 0
    for side in (False, True):
        U, V = cf.bpr_factors(
            shown_matrix(A, fit[part != side], nu), k, lr=0.02, reg=0.01, iters=iters, seed=seed + 2 * side
        )
        if keep:  # keep the strongest item directions of a large fit
            U, V = cf.project(U, V, keep)
        xf[part == side] = _scores(A, fit[part == side], U, V)
        xt = xt + _scores(A, tgt, U, V) / 2
    if cache:
        tmp = f.with_name(f"{f.stem}.tmp.npz")
        np.savez(tmp, xf=xf, xt=xt)
        tmp.replace(f)
    return xf, xt


def standard_scores(c, tag=""):
    """Seed-averaged scores of a cutoff (make_exposure.py), standardised on the fit events: (fit events, targets)."""
    z = np.load(CACHE / f"expo_mean{tag}_{c}.npz")
    mu, sd = z["xf"].reshape(-1, 3).mean(0), z["xf"].reshape(-1, 3).std(0)
    return (z["xf"] - mu) / sd, (z["xt"] - mu) / sd
