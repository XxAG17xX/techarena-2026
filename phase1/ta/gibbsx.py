"""Sampler with frozen extra terms in the utility, each with its own sampled multiplier:

    utility(i) = alpha * theta_u . phi_i + b_i + sg * repeat term + sum_k w_k * x_kei

The x terms are computed once, before the run (exposure scores, category terms, recent-click momentum ...). Only their
weights w_k and the repeat-term multiplier sg are sampled, by one random-walk Metropolis step each, every fourth sweep.
On its own (run_chain_exposure.py) the x terms are the three exposure scores of ta/expo.py; ta/gibbsc.py builds on it.
"""

import numpy as np
import torch

from ta.gibbs import Gibbs


class GibbsX(Gibbs):
    """Gibbs plus frozen extra utility terms with sampled weights."""

    def __init__(self, *args, xe, xt, w0, **kw):
        """xe [fit, 10, m], xt [targets, 10, m], standardised outside; w0 = starting multipliers."""
        super().__init__(*args, **kw)
        t = lambda x: torch.as_tensor(np.asarray(x), device=self.dev, dtype=torch.float32)
        self.xe0, self.xt0 = t(xe), t(xt)
        self.w = [float(x) for x in w0]
        self.sig_w = [0.03] * len(self.w)
        self._burning = True
        self._set_scales(self.sg, self.sn)

    def _set_scales(self, sg, sn):
        """Offsets = sg * repeat term + sum_k w_k * x_k (rebuilt whenever a multiplier or weight moves)."""
        self.sg, self.sn = sg, sn
        w = torch.as_tensor(getattr(self, "w", []), device=self.dev, dtype=torch.float32)
        extra_e = (self.xe0 * w).sum(-1) if hasattr(self, "xe0") else 0.0
        extra_t = (self.xt0 * w).sum(-1) if hasattr(self, "xt0") else 0.0
        self.eo, self.to = sg * self.eo0 + extra_e, sg * self.to0 + extra_t
        self.en, self.tn = sn * self.en0, sn * self.tn0
        self.en_bar = self.en.mean()

    def _update_scales(self):
        """One Metropolis step on sg, then one on each weight w_k in turn (flat priors), each judged on the total
        log-likelihood of all events."""
        cur = self._total_ll(self.alpha)
        prop = self.sg + 0.02 * float(np.random.normal())
        old = self.sg
        self._set_scales(prop, self.sn)
        new = self._total_ll(self.alpha)
        if np.log(np.random.random()) < new - cur:
            cur = new
        else:
            self._set_scales(old, self.sn)
        for k in range(len(self.w)):
            old = self.w[k]
            self.w[k] = old + self.sig_w[k] * float(np.random.normal())
            self._set_scales(self.sg, self.sn)
            new = self._total_ll(self.alpha)
            ok = np.log(np.random.random()) < new - cur
            if ok:
                cur = new
            else:
                self.w[k] = old
                self._set_scales(self.sg, self.sn)
            if self._burning:
                self.sig_w[k] = float(np.clip(self.sig_w[k] * np.exp(0.05 * (float(ok) - 0.3)), 1e-3, 0.1))

    def _after_sweep(self, adapt):
        """Remember whether we are still in burn-in, when the weights' step sizes adapt."""
        self._burning = adapt

    def _rec_extra(self):
        """Log the current weights."""
        return {"w": [round(x, 3) for x in self.w]}
