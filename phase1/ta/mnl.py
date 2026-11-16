"""Point-estimate choice model: a factorised multinomial logit with a no-click option, trained by gradient descent
on whole events (11-way softmax). In the final model it gives each sampler run its starting point (ta/gibbs_fit.py).

utility(candidate i) = b_i + Q_i . (P_u + pooled history Y) + g(repeat features of i)
utility(no click)    = c0 + c_u + linear(event features)

aux_w > 0 adds a sampled-softmax loss on clicks known only from histories (before the train window, slates unknown):
each such click competes with aux_neg random catalogue items under the user-vector utility b_i + Q_i . P_u.

The model grew many switches while it was the main model (the DEFAULT dict below). The final start uses
ta.stack.MNL8T_CFG with d = 8: unit (cosine) user and item vectors, no history pooling (use_Y False), confidence
shrinkage by known clicks (conf, conf_item), the history-click loss with exposure-weighted competitors (aux_w, aux_q),
40 epochs. Every other switch keeps its default, which turns it off.
"""

import copy
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ta.data import H, N_ITEMS
from ta.harness import auc_report

DEFAULT = dict(
    d=64,
    lr=2e-3,
    batch=2048,
    epochs=30,
    patience=3,
    l2_user=1e-3,
    l2_item=1e-4,
    l2_bias=1e-4,
    hist_drop=0.1,
    use_P=True,
    freeze=False,
    init_Y=False,
    attn=False,
    cosine=False,
    alpha0=1.0,
    use_Y=True,
    aux_w=0.0,
    aux_neg=50,
    aux_all=False,
    d_rest=0,
    ema=0.0,
    user_temp=False,
    l2_temp=0.1,
    conf=False,
    conf_item=False,
    conf_pow=False,
    freeze_P=False,
    freeze_Q=False,
    init_b=None,
    aux_q=False,
    aux_qu=False,
    kappa_u=None,
    kappa_i=None,
    fshape=0,
    boot=None,
    avg_last=0,
    rp_u=0.0,
    rp_b=0.0,
    frob_u=0.0,
    frob_i=0.0,
    frob_b=0.0,
    fix_alpha=False,
    init_scale=0.0,
    anchor_u=0.0,
    anchor_i=0.0,
    aux_marginal=0,
    aux_clickrate=0.63,
    pop_prior=False,
    pop_l2=0.0,
    freeze_b=False,
    click_power=None,
    direction_noise=0.0,
    unit_factors=False,
    factor_lr=None,
    factor_lr_final_ratio=1.0,
    geometry_stats=False,
    aux_personal_weight=0.0,
    auc_weight=0.0,
    auc_pairs=16,
    seed=0,
)


class MNL(nn.Module):
    """The choice model's parameters and forward pass: logits [batch, 11] for 10 shown items plus "no click"."""

    def __init__(
        self,
        n_users,
        d,
        n_ef,
        n_cf,
        use_P=True,
        attn=False,
        cosine=False,
        alpha0=1.0,
        use_Y=True,
        d_rest=0,
        user_temp=False,
        temp_reg=1.0,
        n_clicks=None,
        m_clicks=None,
        conf_pow=False,
        fshape=0,
        popularity=None,
        click_power=None,
        direction_noise=0.0,
        noise_counts=None,
    ):
        super().__init__()
        # fshape > 0 (cosine only): utility shape f(cos) = cos + sum_k w_k relu(cos - knot_k), fshape interior knots on
        # [-1, 1], identity at start; tests whether the generator's utility is linear in the cosine
        self.fshape = fshape
        if fshape:
            self.register_buffer("knots", torch.linspace(-1, 1, fshape + 2)[1:-1])
            self.fk = nn.Parameter(torch.zeros(fshape))
        # conf_pow: learn exponents for the confidence curves, (n / (n + kappa)) ** gamma
        self.conf_pow = conf_pow
        self.lg_u = nn.Parameter(torch.tensor(0.0))
        self.lg_i = nn.Parameter(torch.tensor(0.0))
        # n_clicks given: affinity is scaled by n_u / (n_u + kappa), n_u = the user's known clicks, so directions
        # estimated from little evidence are trusted less (a shrinkage that cosine affinity otherwise lacks)
        self.conf = n_clicks is not None
        if self.conf:
            self.register_buffer("ncl", torch.as_tensor(n_clicks, dtype=torch.float32))
            self.kappa = nn.Parameter(torch.tensor(3.0))
        # m_clicks given: the same shrinkage for items, m_i = the item's known clicks
        self.conf_item = m_clicks is not None
        if self.conf_item:
            self.register_buffer("mcl", torch.as_tensor(m_clicks, dtype=torch.float32))
            self.kappa_i = nn.Parameter(torch.tensor(3.0))
        # user_temp: each user's affinity is scaled by exp(t_u) (some users choose sharply, others nearly at random);
        # t_u starts at 0 and its L2 penalty is temp_reg times the bias penalty
        self.user_temp, self.temp_reg = user_temp, temp_reg
        self.lt = nn.Embedding(n_users, 1)
        nn.init.zeros_(self.lt.weight)
        # d_rest > 0 (with cosine): the last d_rest dimensions enter as a plain dot product with their own scale
        self.d_rest = d_rest
        self.alpha_r = nn.Parameter(torch.tensor(float(alpha0)))
        self.use_P = use_P
        self.use_Y = use_Y
        self.attn = attn
        self.cosine = cosine  # affinity = alpha * cos(user, item): true vectors look unit-norm
        if attn:  # candidate-aware pooling: softmax over history items of (beta * similarity + recency bias)
            self.beta = nn.Parameter(torch.tensor(1.0))
            self.gamma = nn.Parameter(torch.tensor(0.0))
            self.apos = nn.Parameter(torch.zeros(H))
        self.P = nn.Embedding(n_users, d)
        self.cu = nn.Embedding(n_users, 1)
        self.Q = nn.Embedding(N_ITEMS, d)
        self.Y = nn.Embedding(N_ITEMS + 1, d, padding_idx=N_ITEMS)
        self.b = nn.Embedding(N_ITEMS, 1)
        for e in (self.P, self.cu, self.b):
            nn.init.zeros_(e.weight)
        nn.init.normal_(self.Q.weight, std=0.05)
        nn.init.normal_(self.Y.weight, std=0.05)
        with torch.no_grad():
            self.Y.weight[N_ITEMS].zero_()
        self.wpos = nn.Parameter(torch.zeros(H))
        self.g = nn.Sequential(nn.Linear(n_cf, 32), nn.ReLU(), nn.Linear(32, 1))
        self.out = nn.Linear(n_ef, 1)
        self.c0 = nn.Parameter(torch.tensor(1.77))  # e^c0 / (e^c0 + 10) ~ 0.37 = no-click share
        self.alpha = nn.Parameter(torch.tensor(float(alpha0)))  # scale on the affinity term
        if popularity is not None:
            self.register_buffer("popularity", torch.as_tensor(popularity, dtype=torch.float32))
            self.pop_slope = nn.Parameter(torch.tensor(0.0))
        if click_power is not None:
            # Negative means learn the coefficient from a unit start; nonnegative fixes it.
            self.click_power = nn.Parameter(
                torch.tensor(1.0 if click_power < 0 else float(click_power)), requires_grad=click_power < 0
            )
        self.direction_noise = direction_noise
        if direction_noise:
            self.register_buffer("noise_user_count", torch.as_tensor(noise_counts[0], dtype=torch.float32))
            self.register_buffer("noise_item_count", torch.as_tensor(noise_counts[1], dtype=torch.float32))

    def forward(self, u, hist, hlen, cand, ef, cf, hist_drop=0.0):
        """(logits [B, 11], (user, item, bias) penalty terms) for a batch of events."""
        valid = hist >= 0
        if hist_drop > 0:
            valid = valid & (torch.rand(valid.shape, device=hist.device) >= hist_drop)
        fe = (hlen[:, None] - 1 - torch.arange(H, device=hist.device)[None, :]).clamp(0, H - 1)
        w = F.softplus(self.wpos)[fe] * valid
        hv = self.Y(torch.where(hist >= 0, hist, N_ITEMS))
        pooled = (w[..., None] * hv).sum(1) / (w.sum(1, keepdim=True) + 1e-6)
        if not self.use_Y:
            pooled = torch.zeros_like(pooled)
        pu = self.P(u) if self.use_P else torch.zeros_like(pooled)
        z = pu + pooled
        qc = self.Q(cand)
        util = (
            self.affinity(self.noisy_direction(z, u, "user"), self.noisy_direction(qc, cand, "item"))
            * self._temp(u)
            * self._iconf(cand)
            + self.item_bias(cand)
            + self.g(cf).squeeze(-1)
        )
        if self.attn:
            s = torch.einsum("ejd,ehd->ejh", qc, hv)  # [B,10,H]
            logit = (self.beta * s + self.apos[fe][:, None, :]).masked_fill(~valid[:, None, :], -1e4)
            a = torch.softmax(logit, -1) * valid.any(1)[:, None, None]
            util = util + self.gamma * (a * s).sum(-1)
        nc = self.c0 + self.cu(u).squeeze(-1) + self.out(ef).squeeze(-1)
        if hasattr(self, "click_power"):
            # Preserve the conditional item softmax, while relaxing how attractiveness
            # changes P(any click): logit(P(any)) = power * logsumexp(utility) - nc.
            nc = nc + (1 - self.click_power) * torch.logsumexp(util, dim=1)
        reg = (
            pu.pow(2).sum(1).mean(),
            qc.pow(2).sum(-1).mean() + (w[..., None] * hv).pow(2).sum(-1).mean(),
            self.b(cand).pow(2).mean()
            + self.cu(u).pow(2).mean()
            + (self.temp_reg * self.lt(u).pow(2).mean() if self.user_temp else 0),
        )
        return torch.cat([util, nc[:, None]], 1), reg

    def _temp(self, u):
        t = torch.exp(self.lt(u)) if self.user_temp else 1.0
        if self.conf:
            n = self.ncl[u][:, None]
            r = n / (n + F.softplus(self.kappa))
            t = t * (r.clamp_min(1e-6) ** torch.exp(self.lg_u) if self.conf_pow else r)
        return t

    def _iconf(self, items):
        if not self.conf_item:
            return 1.0
        m = self.mcl[items]
        r = m / (m + F.softplus(self.kappa_i))
        return r.clamp_min(1e-6) ** torch.exp(self.lg_i) if self.conf_pow else r

    def affinity(self, z, q):
        """alpha * <q, z> on [B, K, d] items and [B, d] users; cosine on the main dimensions when enabled."""
        k = q.shape[-1] - self.d_rest
        zm, qm = z[..., :k], q[..., :k]
        if self.cosine:
            zm, qm = F.normalize(zm, dim=-1), F.normalize(qm, dim=-1)
        c = (qm * zm[:, None, :]).sum(-1)
        if self.fshape:
            c = c + (self.fk * F.relu(c[..., None] - self.knots)).sum(-1)
        out = self.alpha * c
        if self.d_rest:
            out = out + self.alpha_r * (q[..., k:] * z[:, None, k:]).sum(-1)
        return out

    def aff(self, u, items):
        """[B, K] user-vector affinity b_i + alpha * Q_i . P_u (for the history-click loss)."""
        return self.affinity(
            self.noisy_direction(self.P(u), u, "user"), self.noisy_direction(self.Q(items), items, "item")
        ) * self._temp(u) * self._iconf(items) + self.item_bias(items)

    def noisy_direction(self, x, ids, side):
        """Training-time noise on unit directions, smaller for well-observed users and items (off by default)."""
        if not self.training or not self.direction_noise:
            return x
        counts = getattr(self, "noise_" + side + "_count")[ids]
        sd = self.direction_noise / torch.sqrt(counts + 2)
        return F.normalize(x, dim=-1) + torch.randn_like(x) * sd[..., None]

    def item_bias(self, items):
        """Item popularity bias b_i (plus an optional popularity slope)."""
        b = self.b(items).squeeze(-1)
        if hasattr(self, "popularity"):
            b = b + self.pop_slope * self.popularity[items]
        return b


def _click_counts(A, fit, tgt, items=False):
    """Known clicks (before the cutoff) per user, or per item with items=True."""
    from ta import cf

    u, i, _ = cf.known_clicks(A, fit, np.asarray([], np.int64) if tgt is None else tgt)
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


@torch.no_grad()
def project_factors(model):
    """Fix the redundant radial degree of freedom of a pure cosine model."""
    for embedding in (model.P, model.Q):
        embedding.weight.copy_(F.normalize(embedding.weight, dim=1))


def train(
    A,
    EF,
    CF,
    fit,
    tgt=None,
    cfg=None,
    T=None,
    verbose=True,
    init=None,
    early_stop=True,
    epoch_callback=None,
    aux_pool=None,
):
    """Fit on events `fit`. With labelled `tgt`, early-stop on its global AUC; otherwise run cfg['epochs'].
    init = (user factors, item factors) from a CF model (e.g. BPR) to start P and Q from; cfg freeze keeps them fixed.
    Returns (model, target probabilities [len(tgt),10] or None, history list)."""
    cfg = {**DEFAULT, **(cfg or {})}
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = T or Tensors(A, EF, CF, dev)
    assert 0 <= cfg["auc_weight"] <= 1
    pair_rng = torch.Generator(device=dev).manual_seed(cfg["seed"] + 12345) if cfg["auc_weight"] else None
    if cfg["aux_personal_weight"]:
        assert aux_pool is not None and cfg["aux_q"] and not cfg["aux_qu"] and not cfg["aux_marginal"]
        assert 0 < cfg["aux_personal_weight"] <= 1
        aux_pool = torch.as_tensor(aux_pool, device=dev).long()
    popularity = None
    if cfg["pop_prior"]:
        popularity = np.log1p(np.bincount(A["cand"][fit].ravel(), minlength=N_ITEMS)).astype(np.float32)
        popularity = (popularity - popularity.mean()) / max(popularity.std(), 1e-6)
    noise_counts = None
    if cfg["direction_noise"]:
        assert cfg["cosine"] and not cfg["use_Y"], "direction noise is defined for unit-factor models"
        noise_counts = (_click_counts(A, fit, tgt), _click_counts(A, fit, tgt, items=True))
    model = MNL(
        int(A["user"].max()) + 1,
        cfg["d"],
        EF.shape[1],
        CF.shape[2],
        cfg["use_P"],
        cfg["attn"],
        cfg["cosine"],
        cfg["alpha0"],
        cfg["use_Y"],
        cfg["d_rest"],
        cfg["user_temp"],
        cfg["l2_temp"] / max(cfg["l2_bias"], 1e-9),
        _click_counts(A, fit, tgt) if cfg["conf"] else None,
        _click_counts(A, fit, tgt, items=True) if cfg["conf_item"] else None,
        cfg["conf_pow"],
        cfg["fshape"],
        popularity,
        cfg["click_power"],
        cfg["direction_noise"],
        noise_counts,
    ).to(dev)
    if cfg["aux_w"] > 0:  # clicks seen only in histories (before the train window)
        from ta import cf as cfm
        from ta.data import T0

        au, ai, at = cfm.known_clicks(A, fit, fit if tgt is None else tgt)
        pre = at < T0 if not cfg["aux_all"] else np.ones(len(at), bool)  # aux_all: in-window clicks too
        au, ai = torch.as_tensor(au[pre], device=dev).long(), torch.as_tensor(ai[pre], device=dev).long()
        # aux_q: draw the competing items like real slates, in proportion to how often each item is shown (fit
        # events), so exposure does not leak into the item biases (sampling-of-alternatives / imputed slates)
        q = torch.as_tensor(
            np.bincount(A["cand"][fit].ravel(), minlength=N_ITEMS) + 1.0, device=dev, dtype=torch.float32
        )
        if cfg["aux_qu"]:  # aux_qu: draw from the items that user was shown in fit events (personalised exposure)
            su = np.repeat(A["user"][fit], 10)
            o = np.argsort(su, kind="stable")
            shown = torch.as_tensor(A["cand"][fit].ravel()[o], device=dev).long()
            nus = int(A["user"].max()) + 1
            cnt = np.bincount(su, minlength=nus)
            sh_start = torch.as_tensor(np.r_[0, np.cumsum(cnt)[:-1]], device=dev).long()
            sh_len = torch.as_tensor(cnt, device=dev).long()
    if init is not None:
        U, V = init
        if cfg["init_scale"]:  # dot-product models: start with utilities alpha0-sized
            U, V = U * cfg["init_scale"], V * cfg["init_scale"]
        with torch.no_grad():
            model.P.weight[: len(U)] = torch.as_tensor(U, dtype=torch.float32, device=dev)
            model.Q.weight[:] = torch.as_tensor(V, dtype=torch.float32, device=dev)
            if cfg["init_Y"]:  # history items start in the same space as candidates
                model.Y.weight[:N_ITEMS] = torch.as_tensor(V, dtype=torch.float32, device=dev)
            if cfg["init_b"] is not None:  # item biases (e.g. known true values in a synthetic test)
                model.b.weight[:, 0] = torch.as_tensor(cfg["init_b"], dtype=torch.float32, device=dev)
        if cfg["freeze"] or cfg["freeze_P"]:
            model.P.weight.requires_grad_(False)
        if cfg["freeze"] or cfg["freeze_Q"]:
            model.Q.weight.requires_grad_(False)
            if cfg["init_b"] is not None:
                model.b.weight.requires_grad_(False)
    for name, val in (("kappa", cfg["kappa_u"]), ("kappa_i", cfg["kappa_i"])):  # fixed confidence strengths
        if val is not None and hasattr(model, name):
            par = getattr(model, name)
            with torch.no_grad():
                par.fill_(float(np.log(np.expm1(val))))  # softplus(par) = val
            par.requires_grad_(False)
    if cfg["freeze_b"]:
        model.b.weight.requires_grad_(False)
    if cfg["anchor_u"] or cfg["anchor_i"]:  # directions of the starting point, for the anchor penalty
        P0n = F.normalize(model.P.weight.detach().clone(), dim=1)
        Q0n = F.normalize(model.Q.weight.detach().clone(), dim=1)
    if cfg["fix_alpha"]:  # keep the utility scale fixed (penalised vectors carry it)
        model.alpha.requires_grad_(False)
    if cfg["unit_factors"]:
        assert cfg["cosine"] and cfg["use_P"] and not cfg["use_Y"] and not cfg["d_rest"]
        assert not cfg["ema"], "averaging raw projected factors needs separate handling"
        project_factors(model)
    trainable = [q for q in model.parameters() if q.requires_grad]
    if cfg["factor_lr"] is None:
        opt = torch.optim.Adam(trainable, lr=cfg["lr"])
    else:
        factor_ids = {id(model.P.weight), id(model.Q.weight)}
        opt = torch.optim.Adam(
            [
                {"params": [q for q in trainable if id(q) in factor_ids], "lr": cfg["factor_lr"], "factors": True},
                {"params": [q for q in trainable if id(q) not in factor_ids], "lr": cfg["lr"]},
            ],
            lr=cfg["lr"],
        )
    diagnostic_rows = (
        [torch.linspace(0, e.weight.shape[0] - 1, 128, device=dev).long() for e in (model.P, model.Q)]
        if cfg["geometry_stats"]
        else []
    )
    if cfg["boot"] is not None:  # bootstrap resample of events and history clicks (bagging)
        rb_ = np.random.default_rng(cfg["boot"])
        fit = rb_.choice(fit, size=len(fit), replace=True)
        if cfg["aux_w"] > 0:
            k_ = torch.as_tensor(rb_.integers(len(au), size=len(au)), device=dev)
            au, ai = au[k_], ai[k_]
    fit_t = torch.as_tensor(fit, device=dev)
    tgt_t = None if tgt is None else torch.as_tensor(tgt, device=dev)
    # rp_u / rp_b > 0: per-user / per-item priors on the no-click and item biases (each row gets the same prior
    # strength, in units of events, instead of one penalty per event, which barely shrinks light users and rare items)
    n_ev_u = torch.as_tensor(
        np.bincount(A["user"][fit], minlength=model.cu.weight.shape[0]) + 1.0, device=dev, dtype=torch.float32
    )
    n_imp_i = torch.as_tensor(
        np.bincount(A["cand"][fit].ravel(), minlength=N_ITEMS) + 1.0, device=dev, dtype=torch.float32
    )
    acc = 0  # avg_last > 0: average target predictions over the last epochs
    labelled = early_stop and tgt is not None and (A["label"][tgt] >= 0).all()
    best, best_state, hist, bad = -1.0, None, [], 0
    # ema > 0: evaluate and return an exponential moving average of the weights (decay per step)
    ema = [q.detach().clone() for q in model.parameters()] if cfg["ema"] else None

    def swap():  # exchange live and averaged weights
        if ema is not None:
            with torch.no_grad():
                for q, e in zip(model.parameters(), ema):
                    tmp = q.detach().clone()
                    q.copy_(e)
                    e.copy_(tmp)

    for ep in range(cfg["epochs"]):
        model.train()
        for group in opt.param_groups:
            if group.get("factors"):
                phase = ep / max(1, cfg["epochs"] - 1)
                ratio = cfg["factor_lr_final_ratio"]
                group["lr"] = cfg["factor_lr"] * (ratio + (1 - ratio) * (1 + np.cos(np.pi * phase)) / 2)
        t0, tot = time.time(), 0.0
        angular_steps = []
        perm = fit_t[torch.randperm(len(fit_t), device=dev)]
        for s in range(0, len(perm), cfg["batch"]):
            idx = perm[s : s + cfg["batch"]]
            logits, (ru, ri, rb) = model(*T.batch(idx), hist_drop=cfg["hist_drop"])
            loss = F.cross_entropy(logits, T.label[idx])
            if cfg["auc_weight"]:
                from ta.ranking import global_pair_loss

                rank_loss = global_pair_loss(logits, T.label[idx], cfg["auc_pairs"], pair_rng)
                loss = (1 - cfg["auc_weight"]) * loss + cfg["auc_weight"] * rank_loss
            loss = loss + cfg["l2_user"] * ru + cfg["l2_item"] * ri + cfg["l2_bias"] * rb
            if cfg["pop_l2"]:
                loss = loss + cfg["pop_l2"] * model.b(T.cand[idx]).pow(2).mean()
            if cfg["anchor_u"] or cfg["anchor_i"]:
                # prior centred on the starting directions (e.g. the averaged BPR estimate), per user / item, not per
                # event: lets the cosine model train to convergence without drifting into noise
                loss = loss + (
                    cfg["anchor_u"] * (F.normalize(model.P.weight, dim=1) - P0n).pow(2).sum()
                    + cfg["anchor_i"] * (F.normalize(model.Q.weight, dim=1) - Q0n).pow(2).sum()
                ) / len(fit)
            if cfg["frob_u"] or cfg["frob_i"] or cfg["frob_b"]:
                # penalties on the whole user / item matrices (factored nuclear norm, Kallus & Udell 2016), not per
                # event: the full objective is sum of event losses + frob * ||P||^2 + ..., so per batch-mean / len(fit)
                loss = loss + (
                    cfg["frob_u"] * model.P.weight.pow(2).sum()
                    + cfg["frob_i"] * model.Q.weight.pow(2).sum()
                    + cfg["frob_b"] * (model.b.weight.pow(2).sum() + model.cu.weight.pow(2).sum())
                ) / len(fit)
            if cfg["rp_u"] or cfg["rp_b"]:
                ub, cb = T.u[idx], T.cand[idx]
                loss = loss + (
                    cfg["rp_u"] * (model.cu(ub).squeeze(-1) ** 2 / n_ev_u[ub]).mean()
                    + cfg["rp_b"] * (model.b(cb).squeeze(-1) ** 2 / n_imp_i[cb]).mean()
                )
            if cfg["aux_w"] > 0:
                j = torch.randint(len(au), (len(idx),), device=dev)
                if cfg["aux_qu"]:
                    ln = sh_len[au[j]][:, None]
                    k = (torch.rand(len(idx), cfg["aux_neg"], device=dev) * ln.clamp_min(1)).long()
                    glob = torch.multinomial(q, len(idx) * cfg["aux_neg"], replacement=True).view(len(idx), -1)
                    neg = torch.where(ln > 0, shown[(sh_start[au[j]][:, None] + k).clamp_max(len(shown) - 1)], glob)
                else:
                    neg = None
                neg = (
                    neg
                    if neg is not None
                    else (
                        torch.multinomial(q, len(idx) * cfg["aux_neg"], replacement=True).view(len(idx), -1)
                        if cfg["aux_q"]
                        else torch.randint(N_ITEMS, (len(idx), cfg["aux_neg"]), device=dev)
                    )
                )
                if cfg["aux_personal_weight"]:
                    cols = torch.randint(aux_pool.shape[1], neg.shape, device=dev)
                    personal = aux_pool[au[j, None], cols]
                    neg = torch.where(
                        torch.rand_like(neg, dtype=torch.float32) < cfg["aux_personal_weight"], personal, neg
                    )
                items = torch.cat([ai[j, None], neg], 1)
                aux_logits = model.aff(au[j], items)
                if cfg["aux_marginal"]:
                    from ta.history_likelihood import marginal_logits

                    draws = cfg["aux_marginal"] * 9
                    competitors = torch.multinomial(q, len(idx) * draws, replacement=True).view(len(idx), draws)
                    comp = model.aff(au[j], competitors).view(len(idx), cfg["aux_marginal"], 9)
                    aux_logits = marginal_logits(aux_logits, comp, cfg["aux_clickrate"])
                aux = F.cross_entropy(aux_logits, torch.zeros(len(idx), dtype=torch.long, device=dev))
                loss = loss + cfg["aux_w"] * aux
            opt.zero_grad(set_to_none=True)
            loss.backward()
            measure = cfg["geometry_stats"] and (s // cfg["batch"]) % 16 == 0
            if measure:
                before = [
                    F.normalize(e.weight.detach()[rows], dim=1).clone()
                    for e, rows in zip((model.P, model.Q), diagnostic_rows)
                ]
            opt.step()
            if cfg["unit_factors"]:
                project_factors(model)
            if measure:
                with torch.no_grad():
                    # Chord-to-angle avoids arccos roundoff for very small updates.
                    angular_steps.append(
                        torch.stack(
                            [
                                (
                                    2
                                    * torch.asin(
                                        ((F.normalize(e.weight[rows], dim=1) - old).norm(dim=1) / 2).clamp(max=1)
                                    )
                                ).mean()
                                for e, rows, old in zip((model.P, model.Q), diagnostic_rows, before)
                            ]
                        )
                    )
            if ema is not None:
                with torch.no_grad():
                    for q, e in zip(model.parameters(), ema):
                        e.mul_(cfg["ema"]).add_(q.detach(), alpha=1 - cfg["ema"])
            tot += loss.item() * len(idx)
        rec = {"epoch": ep + 1, "loss": round(tot / len(perm), 5), "sec": round(time.time() - t0, 1)}
        if cfg["geometry_stats"]:
            with torch.no_grad():
                for key, e in (("P", model.P), ("Q", model.Q)):
                    norms = e.weight.norm(dim=1)
                    rec[key + "_norm_mean"] = round(norms.mean().item(), 6)
                    rec[key + "_norm_min"] = round(norms.min().item(), 6)
                angles = torch.stack(angular_steps).mean(0).tolist()
                rec.update(P_step_radians=angles[0], Q_step_radians=angles[1], factor_lr=opt.param_groups[0]["lr"])
        if labelled:
            swap()
            rec.update(auc_report(predict(model, T, tgt_t), A["label"][tgt]))
            if rec["auc"] > best:
                best, best_state, bad = rec["auc"], copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
            swap()
        if cfg["avg_last"] and tgt_t is not None and ep >= cfg["epochs"] - cfg["avg_last"]:
            acc = acc + predict(model, T, tgt_t) / cfg["avg_last"]
        if epoch_callback is not None:
            epoch_callback(model, T, tgt_t, rec)
        hist.append(rec)
        if verbose:
            print(rec, flush=True)
        elif os.environ.get("TA_EPOCH_LOG"):  # one short line per epoch, for progress monitoring
            print(f"epoch {ep + 1}/{cfg['epochs']}", flush=True)
        if labelled and bad >= cfg["patience"]:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    elif ema is not None:
        swap()
    probs = None if tgt is None else (acc if cfg["avg_last"] and best_state is None else predict(model, T, tgt_t))
    return model, probs, hist
