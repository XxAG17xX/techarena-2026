"""The main sampler family: the choice model plus what users are shown (exposure), and the terms built from it.

How the simulator shows items: every item has one of 10 categories, and each user has a personal mix over them. A
slate row is either a uniform draw over the whole catalogue (about 40% of rows), a row that follows the categories of
the user's recent clicks (the click feedback loop), a replay of a past click, or a draw from the personal mix. What
is shown says nothing about taste, but the user's mix also enters the click utility: a user clicks more readily in
categories that dominate their mix.

GibbsC  puts the user's category term in the utility, also on clicks known only from histories, whose unknown
        slates are drawn the way real slates are: category from the user's mix, item from that category by
        popularity. Without that the term ran away.
GibbsM  samples each user's mix (the final model) and builds the category term from it. Each sweep proposes a new
        mix from the Dirichlet posterior given the categories of the user's shown rows, and keeps it with the click
        likelihood ratio (Metropolis-Hastings; see GibbsM._update_mix). Main settings of the final families:
          lam = 0.4, f_share = item share per category: 40% of the mix is the uniform draw over the catalogue;
          C0 from clean_counts: shown rows explained by the user's recent clicks are taken out of the counts first;
          floor = 0.7, abs_h = 1.5, abs_w = 0.7: the category term (see GibbsM._tab).

The functions after the classes build the frozen inputs (counts, terms) and the end terms of build_ensemble.py.
"""

import numpy as np
import torch

from ta.data import N_ITEMS, T0
from ta.gibbsx import GibbsX


def terms(A, fit, tgt, z, K):
    """The first two extra terms of fit rows [fit, 10, 2] and target rows [tgt, 10, 2]. Column 0 is a placeholder of
    zeros: GibbsM fills it with the live category term of the sampled mix. Column 1 is the log of the category's
    overall click rate, standardised over the fit rows."""
    kf, kt = z[A["cand"][fit]], z[A["cand"][tgt]]
    lab = A["label"][fit]
    hit = lab < 10
    rate = np.bincount(kf[hit, lab[hit]], minlength=K) / np.maximum(np.bincount(kf.ravel(), minlength=K), 1)
    lr = np.log(rate / rate.mean())
    lr = (lr - lr[kf].mean()) / max(lr[kf].std(), 1e-6)
    xe = np.stack([np.zeros(kf.shape), lr[kf]], -1).astype(np.float32)
    xt = np.stack([np.zeros(kt.shape), lr[kt]], -1).astype(np.float32)
    return xe, xt


class GibbsC(GibbsX):
    """GibbsX with the user's category term on history clicks too, and history-click slates drawn from the user's
    mix. z: category per item; pop: item popularity. The term table and the mix are set by the subclass (GibbsM)."""

    def __init__(self, *args, z, pop, K=11, personal=True, on_history=True, **kw):
        super().__init__(*args, **kw)
        t = lambda x, ty=torch.float32: torch.as_tensor(np.asarray(x), device=self.dev, dtype=ty)
        self.z = t(z, torch.long)
        self.K, self.personal, self.on_history = K, personal, on_history
        pop = t(pop)
        self.members = [torch.nonzero(self.z == k).squeeze(1) for k in range(K)]
        self.weights = [pop[m].clamp_min(1e-3) for m in self.members]

    def _draw_personal(self):
        """Nine slate items per history click: category from the user's mix, item from that category by popularity."""
        k = torch.multinomial(self.mix[self.pu], 9, replacement=True)
        out = torch.empty_like(k)
        for c in range(self.K):
            m = k == c
            n = int(m.sum())
            if n:
                out[m] = self.members[c][torch.multinomial(self.weights[c], n, replacement=True)]
        return out

    def _new_negatives(self):
        """Propose fresh guessed slates for all history clicks (9 items each from the user's mix) and keep each one by
        an independence Metropolis step, as in ta/gibbs.Gibbs._new_negatives."""
        self.logw = 0.0
        if len(self.pi) == 0:
            self.pa = self.pi[:, None]
            return
        neg = (
            self._draw_personal()
            if self.personal
            else torch.multinomial(self.q, len(self.pi) * 9, replacement=True).view(-1, 9)
        )
        prop = torch.cat([self.pi[:, None], neg], 1)
        if getattr(self, "pa", None) is not None:
            ok = torch.log(torch.rand(len(self.pi), device=self.dev)) < self._pre_den_all(self.pa) - self._pre_den_all(
                prop
            )
            self.pa = torch.where(ok[:, None], prop, self.pa)
            self.rates["s"] = ok.float().mean().item()
        else:
            self.pa = prop

    def _pre_x(self, u, c):
        """The category term on history clicks and their guessed slates, with the same weight w[0] as on slate rows."""
        if not getattr(self, "on_history", False):
            return 0.0
        k = self.z[c]
        return self.w[0] * self.table[u[:, None], k]


def popularity(A, fit):
    """How often each item was shown in the fit events, plus one [items]."""
    return np.bincount(A["cand"][fit].ravel(), minlength=N_ITEMS).astype(np.float32) + 1.0


def shown_counts(A, rows, nu, z, K):
    """Category counts per user over the slates of these events, leaving out candidates already in the user's history
    (replay slots are not drawn from the mix)."""
    cand = A["cand"][rows]
    keep = ~(A["hist"][rows][:, None, :] == cand[:, :, None]).any(-1).ravel()
    C = np.zeros((nu, K), np.float32)
    np.add.at(C, (np.repeat(A["user"][rows], 10)[keep], z[cand.ravel()[keep]]), 1)
    return C


class GibbsM(GibbsC):
    """The user's mix is sampled. Generative story: mix -> slate rows -> click, and the utility has w0 times the
    category term of the mix (see _tab). Given the slates (known ones, target slates too, and the current fill-ins
    of history clicks) the mix has a Dirichlet posterior, which serves as the proposal of a Metropolis-Hastings
    step decided by the clicks (see _update_mix). The first extra term is the live category term, unscaled."""

    def __init__(
        self,
        *args,
        C0,
        g,
        a=20.0,
        floor=0.0,
        lam=0.0,
        a_p=8.0,
        abs_h=0.0,
        abs_w=0.7,
        no_rel=False,
        f_share=None,
        h_share=None,
        **kw,
    ):
        # term on the absolute share: abs_w * log(max(K * mix, abs_h))
        self.abs_h, self.abs_w, self.no_rel = float(abs_h), float(abs_w), bool(no_rel)
        # mix = lam * overall + (1 - lam) * personal part, personal part Dirichlet(a_p * overall)
        self.lam, self.a_p = float(lam), float(a_p)
        # utility term log(mix / share + floor): flat where a category suits the user poorly
        self.floor = float(floor)
        super().__init__(*args, **kw)
        t = lambda x: torch.as_tensor(np.asarray(x), device=self.dev, dtype=torch.float32)
        self.C0, self.g, self.a = t(C0), t(g), float(a)
        self.f_s = self.g if f_share is None else t(f_share)  # floor of the mix
        self.h_s = self.g if h_share is None else t(h_share)  # base of the personal part
        self.ze, self.zt = self.z[self.ec], self.z[self.tc]
        self._set_mix((self.C0 + self.a * self.g) / (self.C0.sum(1, keepdim=True) + self.a))
        if self.lam:
            self.pi_p = ((self.mix - self.lam * self.f_s) / (1 - self.lam)).clamp_min(1e-4)
            self.pi_p = self.pi_p / self.pi_p.sum(1, keepdim=True)
            self._set_mix(self.lam * self.f_s + (1 - self.lam) * self.pi_p)

    def _tab(self, m):
        """The category term for mix m [users, K], with share g = overall share of the category:

            log(m / g + floor) - log(1 + floor)       relative term: how much the user's mix favours the category
          + abs_w * log(max(K * m, abs_h))            absolute term: the category's own share in the user's mix

        floor < 0 gives the hinge log(max(m / g, -floor)) instead; no_rel keeps only the absolute term."""
        ab = self.abs_w * torch.log((m * m.shape[1]).clamp_min(self.abs_h)) if self.abs_h else 0.0
        if self.no_rel:
            return ab
        if self.floor < 0:  # hinge: no effect below the bend, log(mix / share) above it
            return torch.log((m / self.g).clamp_min(-self.floor)) + ab
        return torch.log(m / self.g + self.floor) - float(np.log(1.0 + self.floor)) + ab

    def _set_mix(self, m):
        """Install a new mix: rebuild the category table and the first extra term of every slate row."""
        self.mix, self.table = m, self._tab(m)
        self.xe0[..., 0] = self.table[self.eu[:, None], self.ze]
        self.xt0[..., 0] = self.table[self.tu[:, None], self.zt]
        self._set_scales(self.sg, self.sn)

    def _click_ll(self, L):
        """Log-likelihood of each user's clicks [users] if their category table were L (slate events and history
        clicks). Only the category term changes between the current and a proposed mix, so only that is swapped.
        The history clicks' item term (ItemPreMixin, w[2] * x_item) is left out of this test, as in the published runs,
        so the mix step uses a slightly simpler likelihood than the rest of the sampler (see the README)."""
        w = self.w[0]
        ll = torch.zeros(self.nu, device=self.dev)
        for j in self._chunks(len(self.eu)):
            u, y, c = self.eu[j], self.ey[j], self.ec[j]
            util = (
                self.alpha * torch.einsum("ed,ejd->ej", self.P[u], self.Q[c])
                + self.b[c]
                + self.eo[j]
                + w * (L[u[:, None], self.ze[j]] - self.xe0[j][..., 0])
            )
            lg = torch.cat([util, (self.c[u] + self.en[j])[:, None]], 1)
            ll.index_add_(0, u, lg.gather(1, y[:, None]).squeeze(1) - torch.logsumexp(lg, 1))
        for j in self._chunks(len(self.pu)):
            u, c = self.pu[j], self.pa[j]
            ua = (
                self._a_pre() * torch.einsum("nd,nkd->nk", self.P[u], self.Q[c])
                + self.b[c]
                + w * L[u[:, None], self.z[c]]
            )
            ll.index_add_(0, u, ua[:, 0] - self._pre_den(ua, u))
        return ll

    def _update_mix(self):
        """One Metropolis-Hastings step on every user's mix at once.

        The categories of the user's shown rows (the cleaned counts C0, plus the user's history clicks with their
        current guessed slates) are multinomial draws from the mix, so given them the mix has a Dirichlet posterior.
        A draw from that posterior is the proposal, and the acceptance test compares the click likelihoods under the
        new and the old mix. With lam > 0 the mix is lam * floor + (1 - lam) * personal part: each row is first
        assigned to the floor or to the personal part (binomial, given the current mix), and only the personal part
        gets the Dirichlet draw. So the proposal depends on the current mix; it is not an independence proposal. The
        step stays valid because that split is reversible."""
        C = self.C0.clone()
        k, who = self.z[self.pa], self.pu[:, None].expand_as(self.pa)
        C.index_put_((who.ravel(), k.ravel()), torch.ones(k.numel(), device=self.dev), accumulate=True)
        # which rows came from the personal part, then that part given them: reversible for the slates,
        # so the clicks still decide
        if self.lam:
            floor_ = self.lam * self.f_s
            own = (1 - self.lam) * self.pi_p
            Cp = torch.binomial(C.round(), own / (floor_ + own))
            pi_new = torch.distributions.Gamma(Cp + self.a_p * self.h_s, 1.0).sample().clamp_min(1e-12)
            pi_new = pi_new / pi_new.sum(1, keepdim=True)
            prop = floor_ + (1 - self.lam) * pi_new
        else:
            prop = torch.distributions.Gamma(C + self.a * self.g, 1.0).sample().clamp_min(1e-12)
            prop = prop / prop.sum(1, keepdim=True)
        new_table = self._tab(prop.clamp_min(1e-8))
        ok = torch.log(torch.rand(self.nu, device=self.dev)) < self._click_ll(new_table) - self._click_ll(self.table)
        if self.lam:
            self.pi_p = torch.where(ok[:, None], pi_new, self.pi_p)
        self._set_mix(torch.where(ok[:, None], prop, self.mix))
        self.rates["m"] = ok.float().mean().item()

    def _after_sweep(self, adapt):
        """After the taste updates of each sweep: update the mixes."""
        super()._after_sweep(adapt)
        self._update_mix()


def clean_counts(A, rows, nu, z, K, a=20.0, iters=30):
    """Slate rows come from the user's long-run mix or from the categories of their own recent clicks, with a weight per
    click age (EM). Returns the category counts credited to the long-run mix, the overall share of those and the
    weights, newest click first.

    This is the cleaned feedback loop: a user's recent clicks pull later slates towards the same categories, so raw
    shown counts overstate the categories the user clicked, and the clicks would inflate their own category term. EM
    splits every row's category between the two sources: the E-step credits each row to the long-run mix or to each
    recent click in proportion to (weight of that click age x its category match); the M-step re-estimates the mixes
    and the weight per click age (index 0 = newest click)."""
    cand, hist = A["cand"][rows], A["hist"][rows]
    keep = ~(hist[:, None, :] == cand[:, :, None]).any(-1)
    Ce = np.zeros((len(rows), K), np.float32)
    np.add.at(Ce, (np.repeat(np.arange(len(rows)), 10)[keep.ravel()], z[cand.ravel()[keep.ravel()]]), 1)
    ue = A["user"][rows]
    o = np.argsort(-np.where(hist >= 0, A["hist_ts"][rows], -1), axis=1, kind="stable")
    hs = np.take_along_axis(hist, o, 1)
    zh, valid = z[np.maximum(hs, 0)], hs >= 0
    w = np.full(hist.shape[1], 0.01)
    C = np.zeros((nu, K))
    np.add.at(C, ue, Ce)
    g = C.sum(0) / C.sum()
    th = (C + a * g) / (C.sum(1, keepdims=True) + a)
    n_rows = Ce.sum(1)
    for _ in range(iters):
        F = np.zeros(Ce.shape)
        for l in range(len(w)):
            v = np.nonzero(valid[:, l])[0]
            np.add.at(F, (v, zh[v, l]), w[l])
        be = 1 - (valid * w[None, :]).sum(1, keepdims=True)
        ratio = Ce / np.maximum(be * th[ue] + F, 1e-9)
        C = np.zeros((nu, K))
        np.add.at(C, ue, ratio * be * th[ue])
        g = C.sum(0) / C.sum()
        th = (C + a * g) / (C.sum(1, keepdims=True) + a)
        for l in range(len(w)):
            v = np.nonzero(valid[:, l])[0]
            w[l] *= ratio[v, zh[v, l]].sum() / max(n_rows[v].sum(), 1)
    return C.astype(np.float32), g.astype(np.float32), w


def context_shift(A, fit, tgt, z, K, a=500.0):
    """Click rates of a category move with the time bucket and the weekend, the same for everyone. log rate(category,
    bucket) / rate(category), plus the weekend ratio when the fit period holds a weekend, as [targets, 10]. Rates
    are smoothed towards the product of the category and bucket margins with weight a; an end term of the
    ensemble."""
    cand, lab = A["cand"][fit], A["label"][fit]
    keep = ~(A["hist"][fit][:, None, :] == cand[:, :, None]).any(-1)
    hit = np.zeros(cand.shape, bool)
    h = lab < 10
    hit[np.nonzero(h)[0], lab[h]] = True
    R = np.repeat(np.arange(len(fit)), 10).reshape(cand.shape)[keep]
    k, Y = z[cand[keep]], hit[keep].astype(float)
    tb, dt = A["tb"][fit][R].astype(int), A["dt"][fit][R].astype(int)
    S, C = np.zeros((K, 4)), np.zeros((K, 4))
    np.add.at(S, (k, tb), 1)
    np.add.at(C, (k, tb), Y)
    rk = C.sum(1, keepdims=True) / S.sum(1, keepdims=True)
    cb = C.sum(0, keepdims=True) / S.sum(0, keepdims=True) / (C.sum() / S.sum())
    tab = np.log((C + a * rk * cb) / (S + a) / (rk * cb))
    W = np.zeros(K)
    if dt.max() > dt.min():
        S2, C2 = np.zeros((K, 2)), np.zeros((K, 2))
        np.add.at(S2, (k, dt), 1)
        np.add.at(C2, (k, dt), Y)
        r2 = C2.sum(0) / S2.sum(0)
        W = np.log(
            ((C2[:, 1] + a * r2[1] * rk[:, 0] / rk.mean()) / (S2[:, 1] + a))
            / ((C2[:, 0] + a * r2[0] * rk[:, 0] / rk.mean()) / (S2[:, 0] + a))
            / (r2[1] / r2[0])
        )
    kt = z[A["cand"][tgt]]
    return tab[kt, A["tb"][tgt].astype(int)[:, None]] + W[kt] * A["dt"][tgt].astype(int)[:, None]


def item_pre_rate(A, fit, ai, pre, a=10.0):
    """log of an item's history clicks against what its shows in the window lead to expect, smoothed, [items]."""
    cand = A["cand"][fit]
    keep = ~(A["hist"][fit][:, None, :] == cand[:, :, None]).any(-1)
    shows = np.bincount(cand[keep], minlength=N_ITEMS).astype(np.float64)
    n = np.bincount(ai[pre], minlength=N_ITEMS).astype(np.float64)
    return np.log((n + a) / (n.sum() * shows / shows.sum() + a)).astype(np.float32)


def user_facts(A, fit, au, pre, nu, at=None, more=False):
    """Facts about the user that the event likelihood does not see: bands of history clicks, log of events in the fit
    period, [users, 5]. Users with few known clicks click much less, which the prior of the no-click utility then
    learns (ta/gibbsn.UserPriorMixin)."""
    n = np.bincount(au[pre], minlength=nu)
    ev = np.bincount(A["user"][fit], minlength=nu)
    x = [n < 5, (n >= 5) & (n < 10), (n >= 10) & (n < 20), n >= 40, np.log1p(ev) / 3.0]
    if more:
        first = np.full(nu, float(T0))
        np.minimum.at(first, au[pre], at[pre].astype(float))
        x += [n == 0, ev == 0, np.log((n + 0.5) / ((T0 - first) / 86400.0 + 1.0))]
    return np.stack(x, 1).astype(np.float32)


def curve(r, floor):
    """The shape of an exposure term in r = user's share / overall share: log(r + floor), or with a negative floor
    the hinge log(max(r, -floor)), flat below the bend."""
    return np.log(np.maximum(r, -floor)) if floor < 0 else np.log(r + floor)


def feedback_term(A, rows, z, K, th, g, wf, floor=0.7):
    """Boost of a row whose category the user clicked recently, from clicks before the event only: log(share at the
    event / overall + floor) - log(long-run share / overall + floor), [rows, 10]. The "share at the event" mixes
    the long-run share th with the user's recent clicks, weighted by click order (wf from clean_counts). On the 20
    fine groups this is the momentum term of the final model."""
    hist, cand, ue = A["hist"][rows], A["cand"][rows], A["user"][rows]
    o = np.argsort(-np.where(hist >= 0, A["hist_ts"][rows], -1), axis=1, kind="stable")
    hs = np.take_along_axis(hist, o, 1)
    F = np.zeros((len(rows), K), np.float32)
    for l in range(hs.shape[1]):
        v = np.nonzero(hs[:, l] >= 0)[0]
        np.add.at(F, (v, z[hs[v, l]]), float(wf[l]))
    be = 1 - ((hs >= 0) * wf[None, :]).sum(1, keepdims=True)
    k = z[cand]
    idx = np.arange(len(rows))[:, None]
    now = be * th[ue[:, None], k] + F[idx, k]
    return (curve(now / g[k], floor) - curve(th[ue[:, None], k] / g[k], floor)).astype(np.float32)


def recent_match(A, rows, zf, n=3):
    """How many of the user's newest n clicks before the event fall in the row's fine group, [rows, 10]. The momentum
    term's click-order profile is too flat for the newest 2-3, so the newest 3 clicks get this extra pull with its
    own weight."""
    hist = A["hist"][rows]
    o = np.argsort(-np.where(hist >= 0, A["hist_ts"][rows], -1), axis=1, kind="stable")
    hs = np.take_along_axis(hist, o, 1)[:, :n]
    hz = np.where(hs >= 0, zf[np.maximum(hs, 0)], -1)
    return (zf[A["cand"][rows]][:, :, None] == hz[:, None, :]).sum(-1).astype(np.float32)


def asof_fine(A, fit, tgt, zf, Kf, gf, z, K, g, floor=0.7, a=20.0):
    """How much more the user's rows so far favour the row's fine group than its category, from events strictly
    before each event, ([fit, 10], [targets, 10]). Nothing from later events, so clicks cannot leak back into it."""

    def asof(ev_z, gg, n):
        rows = np.r_[fit, tgt]
        o = np.lexsort((A["ts"][rows], A["user"][rows]))
        r = rows[o]
        cand, ue = A["cand"][r], A["user"][r]
        keep = ~(A["hist"][r][:, None, :] == cand[:, :, None]).any(-1)
        C = np.zeros((len(r), n), np.float32)
        np.add.at(C, (np.repeat(np.arange(len(r)), 10)[keep.ravel()], ev_z[cand.ravel()[keep.ravel()]]), 1)
        cum = np.cumsum(C, 0)
        start = np.searchsorted(ue, ue, side="left")
        before = np.where((start > 0)[:, None], cum - C - cum[np.maximum(start - 1, 0)], cum - C)
        k = ev_z[cand]
        idx = np.arange(len(r))[:, None]
        out = np.empty(cand.shape, np.float32)
        val = np.exp(np.log(before[idx, k] + a * gg[k]) - np.log(before.sum(1, keepdims=True) + a) - np.log(gg[k]))
        out[o] = 0.0
        res = np.empty(cand.shape, np.float32)
        res[o] = val
        return res

    x = curve(asof(zf, gf, Kf), floor) - curve(asof(z, g, K), floor)
    return x[: len(fit)], x[len(fit) :]
