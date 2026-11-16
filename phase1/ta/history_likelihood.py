"""Approximate marginal choice likelihood for clicks whose original slate is missing.

An observed history item was chosen from a small slate, not from the whole
catalogue. Its marginal selection rate saturates as its utility increases.
Integrate over independent nine-item slates before a sampled classification loss.

An optional loss for the point-estimate choice model (ta/mnl.py, cfg "aux_marginal"). The final model leaves it off;
the sampler handles the missing slates directly (ta/gibbs.py).
"""

import math

import torch
import torch.nn.functional as F


def marginal_logits(utility, competitor_utility, click_rate=0.63):
    """Log mean P(item selected | nine competitors); tensors [B,K], [B,M,9].

    The outside option is approximated from the mean competing utility mass and
    the population click rate. It is detached: histories contain no nonclicks
    with which to estimate that rate. Constants common to items cancel in NCE.
    """
    log_mass = torch.logsumexp(competitor_utility, dim=-1)
    if click_rate < 1:
        log_outside = (
            torch.logsumexp(competitor_utility.flatten(1), dim=1)
            - math.log(competitor_utility.shape[1] * 9)
            + math.log(10 * (1 - click_rate) / click_rate)
        ).detach()
        log_mass = torch.logaddexp(log_mass, log_outside[:, None])
    log_win = F.logsigmoid(utility[:, :, None] - log_mass[:, None, :])
    return torch.logsumexp(log_win, dim=-1) - math.log(competitor_utility.shape[1])
