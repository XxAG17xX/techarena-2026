"""Collaborative-filtering helpers: the known clicks before a cutoff, BPR factorisation, and the projection of BPR
factors onto 8 unit-length dimensions.

Known clicks = history entries of the fit and target events (all strictly before the target events, and a target is
the user's first event after the cutoff) plus the clicks inside fit events, de-duplicated on (user, item, minute).

In the final model BPR serves three purposes: the 8-dim starting vectors of the sampler (ta/stack.py, ta/gibbs_fit.py),
the item groupings found in what users are shown (ta/cats.py), and the exposure scores (ta/expo.py).
"""

import numpy as np
import scipy.sparse as sp

from ta.data import H, N_ITEMS


def known_clicks(A, fit, tgt):
    """(users, items, times) of every click known before the target events, one entry per (user, item, minute)."""
    ev = np.r_[fit, tgt]
    m = A["hist"][ev] >= 0
    u = np.broadcast_to(A["user"][ev][:, None], (len(ev), H))[m]
    i, t = A["hist"][ev][m], A["hist_ts"][ev][m]
    lab = A["label"][fit]
    c = lab < 10
    u = np.r_[u, A["user"][fit][c]]
    i = np.r_[i, A["cand"][fit][c, lab[c]]]
    t = np.r_[t, A["ts"][fit][c]]
    key = ((u.astype(np.int64) * N_ITEMS + i) << 26) + (t - t.min()) // 60
    _, first = np.unique(key, return_index=True)
    return u[first], i[first], t[first]


def click_matrix(A, fit, tgt, n_users, cutoff=None, half_life_days=None):
    """Sparse user-by-item matrix of known clicks (optionally down-weighted by age)."""
    u, i, t = known_clicks(A, fit, tgt)
    w = np.ones(len(u), np.float32)
    if half_life_days:
        w = np.exp2(-(cutoff - t) / (86400.0 * half_life_days)).astype(np.float32)
    X = sp.csr_matrix((w, (u, i)), shape=(n_users, N_ITEMS))
    X.sum_duplicates()
    return X


def svd_factors(X, k=64):
    """Truncated SVD of a popularity-normalised click matrix (an alternative start; the final model uses BPR)."""
    from scipy.sparse.linalg import svds

    pop = np.asarray((X > 0).sum(0)).ravel()
    Xs = X.multiply(1 / np.sqrt(pop + 1)).tocsr()
    U, s, Vt = svds(Xs, k=k)
    return U * np.sqrt(s), Vt.T * np.sqrt(s)


def bpr_factors(X, k=64, lr=0.05, reg=0.01, iters=100, seed=0):
    """Bayesian Personalised Ranking (the `implicit` library) on a binary user-by-item matrix.
    Returns (user factors, item factors); the last item column is the item bias and the last user column is ones.
    TA_BPR_THREADS caps the CPU threads (0 = all)."""
    from implicit.bpr import BayesianPersonalizedRanking
    import os

    m = BayesianPersonalizedRanking(
        factors=k,
        learning_rate=lr,
        regularization=reg,
        iterations=iters,
        random_state=seed,
        num_threads=int(os.environ.get("TA_BPR_THREADS", "0")),
    )
    m.fit((X > 0).astype(np.float32).tocsr(), show_progress=False)
    return np.asarray(m.user_factors), np.asarray(m.item_factors)


def project(U, V, r=8):
    """Denoise implicit-BPR factors (last column = item bias, user column of ones): keep the item vectors' top-r
    principal directions. The data looks generated from 8-dim latent vectors, and the remaining directions of a
    k=128 fit are noise (single-model AUC 0.762 -> 0.806 on the jan26 fold)."""
    Vf, Uf = V[:, :-1], U[:, :-1]
    _, _, Wt = np.linalg.svd(Vf, full_matrices=False)
    P = Wt[:r].T
    return np.c_[Uf @ P, U[:, -1]].astype(np.float32), np.c_[Vf @ P, V[:, -1]].astype(np.float32)


def unit(U, V):
    """Drop the bias column of projected factors and put every vector on the unit sphere: true vectors look unit-norm
    (estimated norms are far more concentrated than Gaussian vectors), and cosine beats the dot product (0.811 vs
    0.808)."""
    nz = lambda M: (M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)).astype(np.float32)
    return nz(U[:, :-1]), nz(V[:, :-1])


def aligned_mean(UV):
    """Average several unit 8-dim factorisations (list of (U, V)) after mapping each onto the first one's coordinates
    with the least-squares linear map of the item vectors (users get the inverse transpose, so scores are kept)."""
    nz = lambda M: (M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)).astype(np.float32)
    U0, V0 = UV[0]
    Us, Vs = [U0], [V0]
    for U, V in UV[1:]:
        # Each BPR fit finds the same 8-dim space in its own rotation: R maps this fit's item vectors onto the first
        # fit's, and users get inv(R).T so that every user-item dot product is unchanged.
        R, *_ = np.linalg.lstsq(V, V0, rcond=None)
        Vs.append(V @ R)
        Us.append(U @ np.linalg.inv(R).T)
    return nz(np.mean(Us, 0)), nz(np.mean(Vs, 0))
