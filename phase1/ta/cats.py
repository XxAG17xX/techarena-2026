"""Item groups ("categories") found in what users are SHOWN, without click labels.

The simulator gives every item exactly one category, and each user a mix over categories from which their slates are
drawn. This module recovers that structure by co-clustering the shown rows: each item gets one group, each user a mix
over groups, and P(item | user) = mix[user, group of item] * share of the item within its group.

fit_merge over-splits the items into 40 groups (k-means on 16-dim BPR vectors of the shown rows, refined by EM) and then
repeatedly merges the pair of groups whose users' excess counts agree most, down to K groups. The final model uses
K = 10 (the categories) and K = 20 (finer sub-groups), each built several times with different seeds: two estimates of
the same grouping agree on only part of the items, so terms built on one grouping copy its errors, and averaging a
term over several independent groupings removes some of that noise.
"""

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans

from ta import cf, expo
from ta.data import N_ITEMS


def mix(X, z, K, a=20.0):
    """(user mix over groups [users, K], smoothed towards the overall shares with weight a; overall shares [K])."""
    Z = sp.csr_matrix((np.ones(N_ITEMS), (np.arange(N_ITEMS), z)), shape=(N_ITEMS, K))
    C = (X @ Z).toarray()
    g = np.maximum(C.sum(0) / C.sum(), 1e-5)
    g /= g.sum()
    return (C + a * g) / (C.sum(1, keepdims=True) + a), g


def _agreement(X, z, K):
    """Correlation over users with enough rows of the excess counts of each pair of groups."""
    Z = sp.csr_matrix((np.ones(N_ITEMS), (np.arange(N_ITEMS), z)), shape=(N_ITEMS, K))
    C = (X @ Z).toarray()
    N = C.sum(1, keepdims=True)
    g = C.sum(0) / C.sum()
    keep = N.ravel() >= 100
    r = np.corrcoef(((C[keep] - N[keep] * g) / np.sqrt(N[keep] * g * (1 - g))).T)
    np.fill_diagonal(r, -1)
    return r


def _em(X, z, K, iters, a=20.0):
    """Hard EM: given the user mixes, move every item to the group that best explains who was shown it."""
    for _ in range(iters):
        pi, g = mix(X, z, K, a)
        new = ((X.T @ (np.log(pi) - np.log(g))) + np.log(g)).argmax(1)
        if (new == z).all():
            break
        z = new
    return z


def fit_merge(A, rows, nu, K=10, K0=40, seed=0):
    """Over-split, then merge the pair of groups whose users' excess counts agree most. K-means starts alone get
    stuck: big groups split while small ones stay merged. Agreement stays above .26 down to 10 groups, then falls
    to about -.05. rows may include target slates. Returns (category per item, user mix, overall shares, agreement
    at the next merge)."""
    X = expo.shown_matrix(A, rows, nu)
    U, V = cf.bpr_factors(X, 128, lr=0.02, reg=0.01, iters=80, seed=seed)
    Vu = cf.unit(*cf.project(U, V, 16))[1]
    imp = np.asarray(X.sum(0)).ravel()
    z = _em(X, KMeans(K0, n_init=3, random_state=seed).fit(Vu[imp >= 100]).predict(Vu), K0, 6)
    k = K0
    while k > K:
        r = _agreement(X, z, k)
        a, b = np.unravel_index(r.argmax(), r.shape)
        z[z == max(a, b)] = min(a, b)
        z[z > max(a, b)] -= 1
        z = np.unique(_em(X, z, k - 1, 3), return_inverse=True)[1]
        k = int(z.max()) + 1
    pi, g = mix(X, z, K)
    return z, pi, g, float(_agreement(X, z, K).max())
