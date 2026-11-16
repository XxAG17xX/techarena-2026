"""Optional layers on top of a sampler class, combined by make().

The final model builds every main-family run as make(base=GibbsM, outside=True, slate_b=True, item_pre=True,
user_prior=True), i.e. all four layers below on top of ta/gibbsc.GibbsM:

  OutsideHistMixin  history clicks keep the no-click option in their likelihood, and the sharpness and the term
                    weights are learned from slate events only;
  SlateBiasMixin    item biases are learned from slate events only;
  ItemPreMixin      an item's history clicks against what its showings lead to expect, as a weighted utility term;
  UserPriorMixin    the prior mean of a user's no-click utility depends on simple facts about the user.
"""

import numpy as np
import torch

from ta.gibbsc import GibbsM
from ta.gibbsx import GibbsX


class OutsideHistMixin:
    """A history click is one the user made instead of clicking nothing, on a slate more attractive than chance, so
    its likelihood keeps the no-click option: exp(u) / (D + exp(c)). That is P(click, item | slate); dividing by
    P(click) is what makes it conditional. Slates are unrelated to taste, so P(click) does not depend on the
    user's direction or on one item, but it does depend on the sharpness and on c. So directions, biases and
    slates see the history clicks, while the sharpness and c are learned from the slate events alone."""

    def _total_ll(self, alpha):
        """The total log-likelihood (used for the sharpness and the term weights) over slate events only."""
        pu = self.pu
        self.pu = pu[:0]
        r = super()._total_ll(alpha)
        self.pu = pu
        return r


class SlateBiasMixin:
    """Item biases learned from slate events only: a history click count also carries the item's popularity before
    the window, which fill-in slates only approximate, and it inflates the spread."""

    def _item_ll_pre(self, ll, Q, b, loglam_a):
        """History clicks are scored with the current biases, so they cancel out of the bias update."""
        return super()._item_ll_pre(ll, Q, self.b, loglam_a)


class ItemPreMixin:
    """History clicks of an item against what its shows lead to expect, as a fixed term of the utility with a sampled
    weight. Extra terms: lift, category rate, this one. It applies to history clicks and their slates too."""

    x_item = None

    def set_item_pre(self, x):
        """x: the term per item [items] (ta/gibbsc.item_pre_rate)."""
        self.x_item = torch.as_tensor(np.asarray(x), device=self.dev, dtype=torch.float32)

    def _pre_x(self, u, c):
        return super()._pre_x(u, c) + self.w[2] * self.x_item[c]


class UserPriorMixin:
    """The no-click prior depends on facts about the user that the event likelihood does not see: c_u ~ N(X gamma,
    s_c^2), conjugate regression. X holds no intercept, mu_c does that."""

    c_x = None

    def set_user_prior(self, x):
        """x: facts per user [users, m] (ta/gibbsc.user_facts), centred here."""
        self.c_x = torch.as_tensor(np.asarray(x), device=self.dev, dtype=torch.float32)
        self.c_x = self.c_x - self.c_x.mean(0, keepdim=True)
        self.c_g = torch.zeros(self.c_x.shape[1], device=self.dev)

    def _c_mean(self):
        return self.mu_c if self.c_x is None else self.mu_c + self.c_x @ self.c_g

    def _update_c_prior(self):
        """Draw the regression of c on the facts from its conditional posterior (Bayesian linear regression with a
        weak inverse-gamma prior on the variance): first s_c, then gamma given s_c, then the intercept mu_c."""
        if self.c_x is None:
            return super()._update_c_prior()
        n = self.nu
        X, y = self.c_x, self.c - self.c.mean()
        A_ = X.T @ X + 1e-3 * torch.eye(X.shape[1], device=self.dev)
        L = torch.linalg.cholesky(A_)
        hat = torch.cholesky_solve((X.T @ y)[:, None], L).squeeze(1)
        rss = ((y - X @ hat) ** 2).sum().item()
        var = (1.0 + 0.5 * rss) / max(np.random.gamma(1.0 + 0.5 * (n - 1 - X.shape[1])), 1e-6)
        self.s_c = float(np.sqrt(var))
        self.c_g = hat + self.s_c * torch.linalg.solve_triangular(
            L.T, torch.randn_like(hat)[:, None], upper=True
        ).squeeze(1)
        self.mu_c = self.c.mean().item() + float(np.random.normal()) * self.s_c / np.sqrt(n)


def make(base=GibbsX, outside=False, slate_b=False, item_pre=False, user_prior=False):
    """A sampler class with the chosen variants on top of base."""
    mix = (
        ([UserPriorMixin] if user_prior else [])
        + ([ItemPreMixin] if item_pre else [])
        + ([SlateBiasMixin] if slate_b else [])
        + ([OutsideHistMixin] if outside else [])
    )

    class Sampler(*mix, base):
        pass

    return Sampler
