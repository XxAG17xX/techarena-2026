"""Recovery test of the base sampler (ta/gibbs.Gibbs) on a simulated world; needs no organiser data.

The world follows the README's choice model with known values: 2,000 users and 1,000 items with unit taste vectors in
8 dimensions, sharpness 6, item biases, no-click utilities, 50 slate events per user (10 random items each), 5 clicks
per user known only from histories, and one target event per user. The sampler starts from a rough guess (vectors
with noise of sd 0.3 per coordinate, zero biases, sharpness 7 as in the run scripts) and runs 200 sweeps on the CPU
(about a minute). Checks:
  - the sharpness, averaged over the last 100 sweeps, is within 5% of the truth (seeds 0-2 gave 6.01-6.08);
  - the target AUC is within 0.04 of the AUC of the true probabilities (seeds 0-2: 0.021-0.023 below; it cannot reach
    it, since 50 events per user leave each taste vector uncertain) and at least 0.1 above the start (seeds: +0.14).

    python phase1/test_sampler.py        or        pytest phase1/test_sampler.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ta.gibbs import Gibbs  # noqa: E402
from ta.harness import auc_report  # noqa: E402

N_USERS, N_ITEMS, DIM, EVENTS, HISTORY, ALPHA = 2000, 1000, 8, 50, 5, 6.0


def simulate(seed=0):
    rng = np.random.default_rng(seed)
    unit = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
    theta, phi = unit(rng.normal(size=(N_USERS, DIM))), unit(rng.normal(size=(N_ITEMS, DIM)))
    b, c = rng.normal(0.0, 0.3, N_ITEMS), rng.normal(3.0, 0.5, N_USERS)

    def events(per_user, no_click=True):
        u = np.repeat(np.arange(N_USERS), per_user)
        cand = rng.integers(N_ITEMS, size=(len(u), 10))
        util = ALPHA * np.einsum("ed,ejd->ej", theta[u], phi[cand]) + b[cand]
        lg = np.concatenate([util, c[u][:, None]], 1) if no_click else util
        p = np.exp(lg - lg.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        y = (p.cumsum(1) > rng.random((len(u), 1))).argmax(1)
        return u, cand, y, p[:, :10]

    fit = events(EVENTS)
    tgt = events(1)
    hu, hc, hy, _ = events(HISTORY, no_click=False)  # history clicks: the item is known, the slate is not
    return dict(theta=theta, phi=phi, fit=fit, tgt=tgt, pre=(hu, hc[np.arange(len(hu)), hy]), rng=rng)


def test_sampler_recovers_sharpness_and_auc(sweeps=200, burn=100):
    t0 = time.time()
    w = simulate()
    rng = w["rng"]
    noisy = lambda x: (lambda y: y / np.linalg.norm(y, axis=1, keepdims=True))(x + 0.3 * rng.normal(size=x.shape))
    (fu, fc, fy, _), (tu, tc, ty, p_true) = w["fit"], w["tgt"]
    G = Gibbs(
        torch.device("cpu"),
        N_USERS,
        N_ITEMS,
        dict(u=fu, c=fc, y=fy),
        w["pre"],
        dict(u=tu, c=tc),
        noisy(w["theta"]),
        noisy(w["phi"]),
        np.zeros(N_ITEMS),
        np.full(N_USERS, 2.0),
        7.0,
        seed=0,
    )
    np.random.seed(0)
    start_auc = auc_report(G.predict().numpy(), ty)["auc"]
    hist = G.run(sweeps, burn, every=2, log_every=10)
    alpha = np.mean([r["alpha"] for r in hist if r["sweep"] >= burn])
    auc, true_auc = auc_report(G.mean_prediction(), ty)["auc"], auc_report(p_true, ty)["auc"]
    print(
        f"sharpness {alpha:.3f} (true {ALPHA}); target AUC {auc:.4f}, true probabilities {true_auc:.4f}, "
        f"start {start_auc:.4f}; {time.time() - t0:.0f} s"
    )
    assert abs(alpha - ALPHA) < 0.05 * ALPHA
    assert auc > true_auc - 0.04 and auc > start_auc + 0.1


if __name__ == "__main__":
    test_sampler_recovers_sharpness_and_auc()
    print("ok")
