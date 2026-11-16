"""Two-sided posterior sampling for the 8-dim cosine choice model: Metropolis-within-Gibbs on the unit sphere.

Model: utility(i) = alpha * theta_u . phi_i + b_i + o_ei, utility(no click) = c_u + n_e; theta and phi uniform on the
sphere, c_u ~ N(mu_c, s_c^2), b_i ~ N(0, s_b^2). Given the items, users are conditionally independent, so all users
are updated in parallel. Items sharing a slate are coupled through the softmax denominator D; one exponential
auxiliary variable per event removes that (1/D = int exp(-lam D) dlam, lam | rest ~ Exp(D)), so all items are updated
in parallel too, exactly. Clicks known only from histories (slates unknown) compete with K fresh draws from the
exposure distribution, standing in for the 9 unknown slate items. Target probabilities are averaged over samples.

In plain words, one sweep (`run`) does, in order:
  1. refresh the guessed slates of history clicks (`_new_negatives`);
  2. move every user's taste vector theta_u (`_update_users`): propose a small random step on the sphere and keep it
     with the Metropolis rule, i.e. with probability min(1, likelihood ratio); the step size adapts during burn-in so
     that about 30% of proposals are kept;
  3. move every user's no-click utility c_u the same way, then redraw the prior mean and spread of c (`_update_c`);
  4. move every item's vector phi_i and bias b_i (`_update_items`), all items at once thanks to the auxiliary variables;
  5. every second sweep, move the shared sharpness alpha (`_update_alpha`), and every fourth sweep the multipliers of
     the frozen offsets (`_update_scales`);
  6. after burn-in, add the current click probabilities of the target events to a running average (`predict`).

The vectors are only identified up to a common rotation, but predictions are not, so averaging predictions over
sweeps (rather than vectors) is what the run returns. Subclasses (ta/gibbsx.py, ta/gibbsc.py, ta/gibbsn.py) add terms
to the utility by overriding the hooks `_pre_x`, `_a_pre`, `_set_scales`, `_update_scales` and `_after_sweep`.
"""

import time

import numpy as np
import torch
import torch.nn.functional as F

from ta.harness import auc_report


class Gibbs:
    """The base sampler: taste vectors, item biases, no-click utilities and the sharpness, on one device."""

    def __init__(
        self,
        dev,
        n_users,
        n_items,
        ev,
        pre,
        tg,
        P,
        Q,
        b,
        c,
        alpha,
        q=None,
        K=9,
        pre_outside=False,
        sample_b=True,
        s_b=0.3,
        chunk=200000,
        seed=0,
        slate_mh=False,
        fit_scales=False,
    ):
        """ev / tg = dicts u, c [n,10], y (ev only), o [n,10] or None, n [n] or None; pre = (users, items).

        ev: the fit events (user, shown items, label, frozen utility offsets); tg: the target events to predict;
        pre: clicks known only from histories (user, item), whose slates are unknown. P, Q, b, c, alpha: the start
        (user vectors, item vectors, item biases, no-click utilities, sharpness). q: how often each item is shown, the
        proposal for guessed slates. chunk: events per GPU batch."""
        self.dev, self.nu, self.ni, self.K, self.pre_outside = dev, n_users, n_items, K, pre_outside
        self.chunk, self.sample_b, self.s_b, self.slate_mh, self.fit_scales = chunk, sample_b, s_b, slate_mh, fit_scales
        torch.manual_seed(seed)
        t = lambda x, ty: torch.as_tensor(np.asarray(x), device=dev, dtype=ty)
        f, l = torch.float32, torch.long
        self.eu, self.ec, self.ey = t(ev["u"], l), t(ev["c"], l), t(ev["y"], l)
        self.tu, self.tc = t(tg["u"], l), t(tg["c"], l)
        zr = lambda n, m=None: torch.zeros((n, m) if m else (n,), device=dev)
        self.eo = t(ev["o"], f) if ev.get("o") is not None else zr(len(self.eu), 10)
        self.en = t(ev["n"], f) if ev.get("n") is not None else zr(len(self.eu))
        self.to = t(tg["o"], f) if tg.get("o") is not None else zr(len(self.tu), 10)
        self.tn = t(tg["n"], f) if tg.get("n") is not None else zr(len(self.tu))
        self.eo0, self.en0, self.to0, self.tn0 = self.eo, self.en, self.to, self.tn
        self.sg, self.sn = 1.0, 1.0  # how much the frozen offsets are trusted (fit_scales)
        self.pu, self.pi = t(pre[0], l), t(pre[1], l)
        self.q = t(q, f) if q is not None else torch.ones(n_items, device=dev)
        self.P, self.Q = F.normalize(t(P, f), dim=1), F.normalize(t(Q, f), dim=1)
        self.b, self.c = t(b, f).clone(), t(c, f).clone()
        self.alpha = torch.tensor(float(alpha), device=dev)
        # Starting Metropolis step sizes, smaller for users and items with more data (their posteriors are narrower).
        n_u = torch.bincount(self.eu, minlength=n_users).float() + torch.bincount(self.pu, minlength=n_users).float()
        n_i = (
            torch.bincount(self.ec.ravel(), minlength=n_items).float() * 0.1
            + torch.bincount(self.pi, minlength=n_items).float()
        )
        self.sig_u = 0.6 / torch.sqrt(n_u + 1.0)
        self.sig_i = 0.6 / torch.sqrt(n_i + 1.0)
        self.sig_c = 0.5 / torch.sqrt(torch.bincount(self.eu, minlength=n_users).float() + 1.0)
        self.sig_b = 0.5 / torch.sqrt(torch.bincount(self.ec.ravel(), minlength=n_items).float() * 0.1 + 1.0)
        self.sig_a = 0.005
        self.mu_c, self.s_c = c.mean().item(), max(c.std().item(), 0.3)
        self.acc, self.n = None, 0
        self.Pm, self.Qm = torch.zeros_like(self.P), torch.zeros_like(self.Q)
        self.rates = {}

    def _chunks(self, n):
        """Slices of at most `chunk` rows, to bound GPU memory."""
        return (slice(s, s + self.chunk) for s in range(0, n, self.chunk))

    def _pre_den_all(self, pa):
        """Log denominator of every history click given guessed slates pa [clicks, 1 + K]."""
        den = torch.empty(len(self.pu), device=self.dev)
        for j in self._chunks(len(self.pu)):
            u, c, self._j = self.pu[j], pa[j], j
            ua = self._a_pre() * torch.einsum("nd,nkd->nk", self.P[u], self.Q[c]) + self.b[c] + self._pre_x(u, c)
            den[j] = self._pre_den(ua, u)
        return den

    def _new_negatives(self):
        """Refresh the unknown slates of history clicks. slate_mh: Metropolis on P(slate | click) with the exposure
        distribution as proposal (accept with P(click | new) / P(click | old)); otherwise plain draws from exposure."""
        self.logw = float(np.log(9.0 / self.K))
        if len(self.pi) == 0:
            self.pa = self.pi[:, None]
            return
        neg = torch.multinomial(self.q, len(self.pi) * self.K, replacement=True).view(-1, self.K)
        prop = torch.cat([self.pi[:, None], neg], 1)
        if self.slate_mh and getattr(self, "pa", None) is not None:
            # Independence Metropolis: the proposal is the prior of slates, so the acceptance ratio is just the ratio of
            # P(click | slate) = exp(u_click) / denominator, i.e. old denominator / new denominator.
            ok = torch.log(torch.rand(len(self.pi), device=self.dev)) < self._pre_den_all(self.pa) - self._pre_den_all(
                prop
            )
            self.pa = torch.where(ok[:, None], prop, self.pa)
            self.rates["s"] = ok.float().mean().item()
        else:
            self.pa = prop

    def _pre_den(self, ua, u):
        """Log denominator of a history click: itself, 9 unknown slate items (estimated), and the outside option."""
        den = torch.logaddexp(ua[:, 0], self.logw + torch.logsumexp(ua[:, 1:], 1))
        return torch.logaddexp(den, self.c[u] + self.en_bar) if self.pre_outside else den

    # ---- user block
    def _user_ll(self, V, Qc, Qa, bo, ba):
        """Log-likelihood of each user's data [users] with user vectors V: their slate events (11-way softmax) plus
        their history clicks (against the guessed slates)."""
        ll = torch.zeros(self.nu, device=self.dev)
        for j in self._chunks(len(self.eu)):
            u, y = self.eu[j], self.ey[j]
            util = self.alpha * torch.einsum("ed,ejd->ej", V[u], Qc[j]) + bo[j]
            lg = torch.cat([util, (self.c[u] + self.en[j])[:, None]], 1)
            ll.index_add_(0, u, self.tau_e * (lg.gather(1, y[:, None]).squeeze(1) - torch.logsumexp(lg, 1)))
        for j in self._chunks(len(self.pu)):
            u = self.pu[j]
            ua = self._a_pre() * torch.einsum("nd,nkd->nk", V[u], Qa[j]) + ba[j]
            ll.index_add_(0, u, self.tau_h * (ua[:, 0] - self._pre_den(ua, u)))
        return ll

    _j = slice(None)  # which history clicks the current _pre_x call covers
    # same for slate events: both below 1 give a warm posterior for the vectors
    tau_e = 1.0
    # weight of the likelihood of history clicks, 1 is the model as it stands
    tau_h = 1.0

    def _update_users(self, steps, adapt):
        """Random-walk Metropolis on the sphere for every user vector at once. The prior is uniform on the sphere, so
        a proposal (current vector plus Gaussian noise, renormalised) is kept with probability
        min(1, exp(new log-likelihood - old log-likelihood)). The step is symmetric, so no proposal correction."""
        Qc, bo = self.Q[self.ec], self.b[self.ec] + self.eo
        self._j = slice(None)
        Qa, ba = self.Q[self.pa], self.b[self.pa] + self._pre_x(self.pu, self.pa)
        ll = self._user_ll(self.P, Qc, Qa, bo, ba)
        rate = 0.0
        for _ in range(steps):
            prop = F.normalize(self.P + self.sig_u[:, None] * torch.randn_like(self.P), dim=1)
            llp = self._user_ll(prop, Qc, Qa, bo, ba)
            ok = torch.log(torch.rand(self.nu, device=self.dev)) < llp - ll
            self.P = torch.where(ok[:, None], prop, self.P)
            ll = torch.where(ok, llp, ll)
            if adapt:  # burn-in only: aim for about 30% accepted proposals per user
                self.sig_u = (self.sig_u * torch.exp(0.05 * (ok.float() - 0.3))).clamp(1e-3, 2.0)
            rate += ok.float().mean().item() / steps
        self.rates["u"] = rate
        del Qc, Qa, bo, ba

    # ---- no-click utilities: only the events' log-sum-exp is needed, so several steps are cheap
    def _update_c(self, steps, adapt):
        """Random-walk Metropolis on every user's no-click utility c_u under its Normal prior. The items' part of each
        event's softmax does not change while c moves, so it is computed once."""
        lse = torch.empty(len(self.eu), device=self.dev)
        for j in self._chunks(len(self.eu)):
            u = self.eu[j]
            util = (
                self.alpha * torch.einsum("ed,ejd->ej", self.P[u], self.Q[self.ec[j]]) + self.b[self.ec[j]] + self.eo[j]
            )
            lse[j] = torch.logsumexp(util, 1)
        none = (self.ey == 10).float()

        def ll(c):
            nc = c[self.eu] + self.en
            return torch.zeros(self.nu, device=self.dev).index_add_(0, self.eu, none * nc - torch.logaddexp(lse, nc))

        prior = lambda c: -0.5 * ((c - self._c_mean()) / self.s_c) ** 2
        cur = ll(self.c) + prior(self.c)
        rate = 0.0
        for _ in range(steps):
            prop = self.c + self.sig_c * torch.randn_like(self.c)
            new = ll(prop) + prior(prop)
            ok = torch.log(torch.rand(self.nu, device=self.dev)) < new - cur
            self.c = torch.where(ok, prop, self.c)
            cur = torch.where(ok, new, cur)
            if adapt:
                self.sig_c = (self.sig_c * torch.exp(0.05 * (ok.float() - 0.3))).clamp(1e-3, 5.0)
            rate += ok.float().mean().item() / steps
        self.rates["c"] = rate
        self._update_c_prior()

    def _c_mean(self):
        """Prior mean of c_u (ta/gibbsn.UserPriorMixin makes it depend on facts about the user)."""
        return self.mu_c

    def _update_c_prior(self):
        # hyperparameters of the no-click prior (conjugate; flat prior on mu, weak inverse-gamma on the variance)
        n = self.nu
        rss = ((self.c - self.c.mean()) ** 2).sum().item()
        var = (1.0 + 0.5 * rss) / max(np.random.gamma(1.0 + 0.5 * (n - 1)), 1e-6)
        self.s_c = float(np.sqrt(var))
        self.mu_c = self.c.mean().item() + float(np.random.normal()) * self.s_c / np.sqrt(n)

    # ---- item block, conditionally independent given one exponential variable per event
    def _item_aux(self):
        """Draw lam_e ~ Exp(D_e) for every event (and every history click), where D_e is the event's softmax
        denominator. Given lam, the likelihood of event e is exp(u_click) * exp(-lam_e * D_e), and D_e is a SUM over
        the shown items (plus the no-click term, which no item moves), so the likelihood factorises into one term per
        item. Returned as log lam."""
        loglam = torch.empty(len(self.eu), device=self.dev)
        for j in self._chunks(len(self.eu)):
            u = self.eu[j]
            util = (
                self.alpha * torch.einsum("ed,ejd->ej", self.P[u], self.Q[self.ec[j]]) + self.b[self.ec[j]] + self.eo[j]
            )
            lg = torch.cat([util, (self.c[u] + self.en[j])[:, None]], 1)
            loglam[j] = torch.log(torch.empty_like(lg[:, 0]).exponential_()) - torch.logsumexp(lg, 1)
        loglam_a = torch.empty(len(self.pu), device=self.dev)
        for j in self._chunks(len(self.pu)):
            u, self._j = self.pu[j], j
            ua = (
                self._a_pre() * torch.einsum("nd,nkd->nk", self.P[u], self.Q[self.pa[j]])
                + self.b[self.pa[j]]
                + self._pre_x(u, self.pa[j])
            )
            loglam_a[j] = torch.log(torch.empty_like(ua[:, 0]).exponential_()) - self._pre_den(ua, u)
        return loglam, loglam_a

    def _item_ll(self, Q, b, loglam, loglam_a):
        """Per-item log-likelihood [items] given the auxiliary variables: every shown row adds -lam_e * exp(u), every
        click adds its utility u. Each item only sees its own rows, so all items can be updated in parallel."""
        ll = torch.zeros(self.ni, device=self.dev)
        for j in self._chunks(len(self.eu)):
            c, y = self.ec[j], self.ey[j]
            util = self.alpha * torch.einsum("ed,ejd->ej", self.P[self.eu[j]], Q[c]) + b[c] + self.eo[j]
            ll.index_add_(0, c.ravel(), self.tau_e * (-torch.exp(loglam[j][:, None] + util)).ravel())
            k = (y < 10).nonzero().squeeze(1)
            ll.index_add_(0, c[k, y[k]], self.tau_e * util[k, y[k]])
        return self._item_ll_pre(ll, Q, b, loglam_a)

    def _item_ll_pre(self, ll, Q, b, loglam_a):
        """The same for history clicks and their guessed slates (fill-in items weighted by exp(logw))."""
        for j in self._chunks(len(self.pu)):
            c, self._j = self.pa[j], j
            ua = (
                self._a_pre() * torch.einsum("nd,nkd->nk", self.P[self.pu[j]], Q[c]) + b[c] + self._pre_x(self.pu[j], c)
            )
            term = -torch.exp(loglam_a[j][:, None] + ua)
            term[:, 1:] = term[:, 1:] * np.exp(self.logw)
            ll.index_add_(0, c.ravel(), self.tau_h * term.ravel())
            ll.index_add_(0, c[:, 0], self.tau_h * ua[:, 0])
        return ll

    def _update_items(self, steps, adapt):
        """Random-walk Metropolis on every item vector at once (on the sphere, like the users), then one step on every
        item bias under its Normal prior, then a fresh draw of the prior spread s_b (conjugate inverse-gamma)."""
        loglam, loglam_a = self._item_aux()
        ll = self._item_ll(self.Q, self.b, loglam, loglam_a)
        rate = 0.0
        for _ in range(steps):
            prop = F.normalize(self.Q + self.sig_i[:, None] * torch.randn_like(self.Q), dim=1)
            llp = self._item_ll(prop, self.b, loglam, loglam_a)
            ok = torch.log(torch.rand(self.ni, device=self.dev)) < llp - ll
            self.Q = torch.where(ok[:, None], prop, self.Q)
            ll = torch.where(ok, llp, ll)
            if adapt:
                self.sig_i = (self.sig_i * torch.exp(0.05 * (ok.float() - 0.3))).clamp(1e-3, 2.0)
            rate += ok.float().mean().item() / steps
        self.rates["i"] = rate
        if self.sample_b:
            prior = lambda b: -0.5 * (b / self.s_b) ** 2
            cur = ll + prior(self.b)
            prop = self.b + self.sig_b * torch.randn_like(self.b)
            new = self._item_ll(self.Q, prop, loglam, loglam_a) + prior(prop)
            ok = torch.log(torch.rand(self.ni, device=self.dev)) < new - cur
            self.b = torch.where(ok, prop, self.b)
            if adapt:
                self.sig_b = (self.sig_b * torch.exp(0.05 * (ok.float() - 0.3))).clamp(1e-3, 2.0)
            self.rates["b"] = ok.float().mean().item()
            self.s_b = float(
                np.sqrt((1.0 + 0.5 * (self.b**2).sum().item()) / max(np.random.gamma(1.0 + 0.5 * self.ni), 1e-6))
            )

    # ---- sharpness (global, shared proposal)
    def _total_ll(self, alpha):
        """Total log-likelihood of all events and history clicks at sharpness alpha (a scalar)."""
        keep, self.alpha = self.alpha, alpha
        total = 0.0
        for j in self._chunks(len(self.eu)):
            u, y, c = self.eu[j], self.ey[j], self.ec[j]
            util = self.alpha * torch.einsum("ed,ejd->ej", self.P[u], self.Q[c]) + self.b[c] + self.eo[j]
            lg = torch.cat([util, (self.c[u] + self.en[j])[:, None]], 1)
            total += (lg.gather(1, y[:, None]).squeeze(1) - torch.logsumexp(lg, 1)).sum().item()
        for j in self._chunks(len(self.pu)):
            u, c, self._j = self.pu[j], self.pa[j], j
            ua = self._a_pre() * torch.einsum("nd,nkd->nk", self.P[u], self.Q[c]) + self.b[c] + self._pre_x(u, c)
            total += (ua[:, 0] - self._pre_den(ua, u)).sum().item()
        self.alpha = keep
        return total

    def _update_alpha(self, adapt):
        """One Metropolis step on the shared sharpness alpha, proposed on the log scale. Returns the log-likelihood."""
        cur = self._total_ll(self.alpha)
        prop = self.alpha * torch.exp(torch.randn(1, device=self.dev)[0] * self.sig_a)
        ok = np.log(np.random.random()) < self._total_ll(prop) - cur
        if ok:
            self.alpha = prop
        if adapt:
            self.sig_a = float(np.clip(self.sig_a * np.exp(0.05 * (float(ok) - 0.3)), 1e-4, 0.1))
        self.rates["a"] = float(ok)
        return cur

    def _set_scales(self, sg, sn):
        """Rebuild the offsets from their frozen values and multipliers (sg: item offsets, sn: no-click offsets)."""
        self.sg, self.sn = sg, sn
        self.eo, self.to, self.en, self.tn = sg * self.eo0, sg * self.to0, sn * self.en0, sn * self.tn0
        self.en_bar = self.en.mean()

    def _update_scales(self):
        """Offsets come from a point estimate: sample one global multiplier for each (repeat-candidate, event terms)."""
        for k in (0, 1):
            old = (self.sg, self.sn)
            cur = self._total_ll(self.alpha)
            prop = list(old)
            prop[k] = old[k] + 0.02 * float(np.random.normal())
            self._set_scales(*prop)
            if np.log(np.random.random()) >= self._total_ll(self.alpha) - cur:
                self._set_scales(*old)

    def _pre_x(self, u, c):
        """Hook: extra utility of history clicks and their guessed slates (none in the base model)."""
        return 0.0

    def _a_pre(self):
        """Sharpness of history clicks."""
        return self.alpha

    def _after_sweep(self, adapt):
        """Hook: extra updates at the end of each sweep."""
        pass

    def _rec_extra(self):
        """Hook: extra fields for the progress log."""
        return {}

    # ---- predictions
    def predict(self):
        """Click probabilities [targets, 10] of the target events at the current state (softmax over 10 items + no
        click)."""
        out = []
        for j in self._chunks(len(self.tu)):
            u, c = self.tu[j], self.tc[j]
            util = self.alpha * torch.einsum("ed,ejd->ej", self.P[u], self.Q[c]) + self.b[c] + self.to[j]
            lg = torch.cat([util, (self.c[u] + self.tn[j])[:, None]], 1)
            out.append(torch.softmax(lg, 1)[:, :10])
        return torch.cat(out)

    def mean_prediction(self):
        """The posterior predictive: target click probabilities averaged over the saved sweeps."""
        return (self.acc / self.n).cpu().numpy()

    @torch.no_grad()
    def run(
        self,
        sweeps,
        burn,
        every=5,
        user_steps=2,
        item_steps=2,
        c_steps=4,
        alpha_every=2,
        label=None,
        log_every=25,
        quality=None,
        callback=None,
        pause=0.0,
        extra_draws=0,
    ):
        """Run the chain: `sweeps` sweeps, the first `burn` adapt step sizes and are discarded, then every `every`-th
        sweep is averaged into the prediction. label (targets' clicks, when known) adds AUC to the progress log.
        Returns the list of progress records."""
        self.en_bar = self.en.mean() if len(self.en) else 0.0
        hist = []
        for s in range(sweeps):
            adapt = s < burn
            time.sleep(pause)  # laptop: leave the GPU some idle time to stay cool
            self._new_negatives()
            self._update_users(user_steps, adapt)
            self._update_c(c_steps, adapt)
            self._update_items(item_steps, adapt)
            ll = self._update_alpha(adapt) if s % alpha_every == 0 else None
            if self.fit_scales and s % 4 == 0:
                self._update_scales()
            self._after_sweep(adapt)
            # the likelihood cannot see a shift of all b and all c together: pin it
            if self.sample_b:
                d = self.b.mean()
                self.b, self.c = self.b - d, self.c - d
            if s >= burn and (s - burn) % every == 0:
                p = self.predict()
                self.acc = p if self.acc is None else self.acc + p
                self.n += 1
                self.Pm += self.P
                self.Qm += self.Q
                # users and c given the items mix fast: more of them per saved sample
                for _ in range(extra_draws):
                    self._update_users(1, False)
                    self._update_c(1, False)
                    self.acc = self.acc + self.predict()
                    self.n += 1
            if s % log_every == 0 or s == sweeps - 1:
                rec = {
                    "sweep": s,
                    "alpha": round(self.alpha.item(), 3),
                    "mu_c": round(self.mu_c, 3),
                    "s_c": round(self.s_c, 3),
                    "s_b": round(self.s_b, 3),
                    "ll": None if ll is None else round(ll / 1e6, 4),
                    "sg": round(self.sg, 3),
                    "sn": round(self.sn, 3),
                    "c_sd": round(self.c.std().item(), 3),
                    **self._rec_extra(),
                    "n": self.n,
                    **{f"r_{k}": round(v, 2) for k, v in self.rates.items()},
                }
                if label is not None:
                    rec["single"] = auc_report(self.predict().cpu().numpy(), label)["auc"]
                    if self.n:
                        rec.update({("avg_" + k): v for k, v in auc_report(self.mean_prediction(), label).items()})
                if quality is not None:
                    rec["q_now"] = quality(self.P.cpu().numpy(), self.Q.cpu().numpy())
                    if self.n:
                        rec["q_mean"] = quality(self.Pm.cpu().numpy(), self.Qm.cpu().numpy())
                print(rec, flush=True)
                hist.append(rec)
                if callback is not None:
                    callback(self, rec)
        return hist
