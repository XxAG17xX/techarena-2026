"""Point-estimate choice model: a factorised multinomial logit with a no-click option, trained by gradient descent
on whole events (11-way softmax). In the final model it gives each sampler run its starting point (ta/gibbs_fit.py).

utility(candidate i) = alpha * cos(P_u, Q_i) * conf_u * conf_i + b_i + g(repeat features of i)
utility(no click)    = c0 + c_u + linear(event features)

conf_u = n_u / (n_u + kappa) shrinks the taste term of a user with few known clicks n_u (conf_i the same for items).
A second, sampled-softmax loss covers clicks known only from histories (before the training window, slates unknown):
each such click competes with aux_neg items drawn in proportion to how often items are shown.

This module once had many switches; only the configuration the final start uses (ta.stack.MNL8T_CFG with d = 8) is
left. Two random draws of removed parts are kept, so that results stay bit-identical to the published runs.
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ta import cf
from ta.data import N_ITEMS, T0

DEFAULT = dict(
    d=64,
    lr=2e-3,
    batch=2048,
    epochs=30,
    l2_user=1e-3,
    l2_item=1e-4,
    l2_bias=1e-4,
    hist_drop=0.1,
    alpha0=1.0,
    aux_w=1.0,
    aux_neg=50,
    seed=0,
)


class MNL(nn.Module):
    """The choice model's parameters and forward pass: logits [batch, 11] for 10 shown items plus "no click"."""

    def __init__(self, n_users, d, n_ef, n_cf, alpha0, n_clicks, m_clicks):
        super().__init__()
        # affinity scaled by n / (n + kappa), n = the user's (item's) known clicks: directions estimated from little
        # evidence are trusted less (a shrinkage that cosine affinity otherwise lacks)
        self.register_buffer("ncl", torch.as_tensor(n_clicks, dtype=torch.float32))
        self.kappa = nn.Parameter(torch.tensor(3.0))
        self.register_buffer("mcl", torch.as_tensor(m_clicks, dtype=torch.float32))
        self.kappa_i = nn.Parameter(torch.tensor(3.0))
        # The published runs also created a per-user temperature table and a history-pooling table here, which the
        # final model never uses. Their random initialisations are still drawn, so the random stream stays the same.
        burn = lambda *shape: torch.empty(*shape).normal_()
        burn(n_users, 1)
        self.P = nn.Embedding(n_users, d)
        self.cu = nn.Embedding(n_users, 1)
        self.Q = nn.Embedding(N_ITEMS, d)
        burn(N_ITEMS + 1, d)
        self.b = nn.Embedding(N_ITEMS, 1)
        for e in (self.P, self.cu, self.b):
            nn.init.zeros_(e.weight)
        nn.init.normal_(self.Q.weight, std=0.05)
        burn(N_ITEMS + 1, d)
        self.g = nn.Sequential(nn.Linear(n_cf, 32), nn.ReLU(), nn.Linear(32, 1))
        self.out = nn.Linear(n_ef, 1)
        self.c0 = nn.Parameter(torch.tensor(1.77))  # e^c0 / (e^c0 + 10) ~ 0.37 = no-click share
        self.alpha = nn.Parameter(torch.tensor(float(alpha0)))  # scale on the affinity term

    def forward(self, u, hist, hlen, cand, ef, cf, hist_drop=0.0):
        """(logits [B, 11], (user, item, bias) penalty terms) for a batch of events."""
        if hist_drop > 0:  # a draw the published runs made for history dropout; kept for the random stream
            torch.rand(hist.shape, device=hist.device)
        pu = self.P(u)
        qc = self.Q(cand)
        util = self.affinity(pu, qc) * self._temp(u) * self._iconf(cand) + self.item_bias(cand) + self.g(cf).squeeze(-1)
        nc = self.c0 + self.cu(u).squeeze(-1) + self.out(ef).squeeze(-1)
        reg = (
            pu.pow(2).sum(1).mean(),
            qc.pow(2).sum(-1).mean(),
            self.b(cand).pow(2).mean() + self.cu(u).pow(2).mean(),
        )
        return torch.cat([util, nc[:, None]], 1), reg

    def _temp(self, u):
        n = self.ncl[u][:, None]
        return n / (n + F.softplus(self.kappa))

    def _iconf(self, items):
        m = self.mcl[items]
        return m / (m + F.softplus(self.kappa_i))

    def affinity(self, z, q):
        """alpha * cos(q, z) on [B, K, d] items and [B, d] users."""
        zm, qm = F.normalize(z, dim=-1), F.normalize(q, dim=-1)
        return self.alpha * (qm * zm[:, None, :]).sum(-1)

    def aff(self, u, items):
        """[B, K] user-vector utility of items (for the history-click loss)."""
        return self.affinity(self.P(u), self.Q(items)) * self._temp(u) * self._iconf(items) + self.item_bias(items)

    def item_bias(self, items):
        """Item popularity bias b_i."""
        return self.b(items).squeeze(-1)


def _click_counts(A, fit, tgt, items=False):
    """Known clicks (before the cutoff) per user, or per item with items=True."""
    u, i, _ = cf.known_clicks(A, fit, tgt)
    return np.bincount(i, minlength=N_ITEMS) if items else np.bincount(u, minlength=int(A["user"].max()) + 1)


class Tensors:
    """All model inputs resident on the GPU; batches are gathered by event index."""

    def __init__(self, A, EF, CF, dev):
        t = lambda x, dt: torch.as_tensor(np.ascontiguousarray(x)).to(dev, dt)
        self.u, self.hist, self.hlen = t(A["user"], torch.long), t(A["hist"], torch.long), t(A["hlen"], torch.long)
        self.cand, self.ef, self.cf = t(A["cand"], torch.long), t(EF, torch.float32), t(CF, torch.float32)
        self.label = t(np.maximum(A["label"], 0), torch.long)

    def batch(self, idx):
        return self.u[idx], self.hist[idx], self.hlen[idx], self.cand[idx], self.ef[idx], self.cf[idx]


@torch.no_grad()
def predict(model, T, idx, bs=16384):
    """Click probabilities [len(idx), 10] of the shown items of events idx."""
    model.eval()
    out = []
    for s in range(0, len(idx), bs):
        logits, _ = model(*T.batch(idx[s : s + bs]))
        out.append(torch.softmax(logits, 1)[:, :10].float().cpu())
    return torch.cat(out).numpy()


def train(A, EF, CF, fit, tgt, cfg, T, init):
    """Fit on events `fit` for cfg['epochs'] epochs, starting P and Q from init = (user factors, item factors).
    Returns (model, target probabilities [len(tgt), 10], per-epoch log)."""
    cfg = {**DEFAULT, **cfg}
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nu = int(A["user"].max()) + 1
    model = MNL(
        nu,
        cfg["d"],
        EF.shape[1],
        CF.shape[2],
        cfg["alpha0"],
        _click_counts(A, fit, tgt),
        _click_counts(A, fit, tgt, items=True),
    ).to(dev)
    # clicks seen only in histories (before the training window); their competitors are drawn like real slates, in
    # proportion to how often each item is shown in fit events, so exposure does not leak into the item biases
    au, ai, at = cf.known_clicks(A, fit, tgt)
    pre = at < T0
    au, ai = torch.as_tensor(au[pre], device=dev).long(), torch.as_tensor(ai[pre], device=dev).long()
    q = torch.as_tensor(np.bincount(A["cand"][fit].ravel(), minlength=N_ITEMS) + 1.0, device=dev, dtype=torch.float32)
    U, V = init
    with torch.no_grad():
        model.P.weight[: len(U)] = torch.as_tensor(U, dtype=torch.float32, device=dev)
        model.Q.weight[:] = torch.as_tensor(V, dtype=torch.float32, device=dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    fit_t, tgt_t = torch.as_tensor(fit, device=dev), torch.as_tensor(tgt, device=dev)
    hist = []
    for ep in range(cfg["epochs"]):
        model.train()
        t0, tot = time.time(), 0.0
        perm = fit_t[torch.randperm(len(fit_t), device=dev)]
        for s in range(0, len(perm), cfg["batch"]):
            idx = perm[s : s + cfg["batch"]]
            logits, (ru, ri, rb) = model(*T.batch(idx), hist_drop=cfg["hist_drop"])
            loss = F.cross_entropy(logits, T.label[idx])
            loss = loss + cfg["l2_user"] * ru + cfg["l2_item"] * ri + cfg["l2_bias"] * rb
            j = torch.randint(len(au), (len(idx),), device=dev)
            neg = torch.multinomial(q, len(idx) * cfg["aux_neg"], replacement=True).view(len(idx), -1)
            items = torch.cat([ai[j, None], neg], 1)
            aux = F.cross_entropy(model.aff(au[j], items), torch.zeros(len(idx), dtype=torch.long, device=dev))
            loss = loss + cfg["aux_w"] * aux
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        hist.append({"epoch": ep + 1, "loss": round(tot / len(perm), 5), "sec": round(time.time() - t0, 1)})
        if os.environ.get("TA_EPOCH_LOG"):  # one short line per epoch, for progress monitoring
            print(f"epoch {ep + 1}/{cfg['epochs']}", flush=True)
    return model, predict(model, T, tgt_t), hist
