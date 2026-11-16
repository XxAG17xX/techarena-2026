"""Global row-ranking surrogate; negative pairs span the whole event minibatch.

An optional extra loss for the point-estimate choice model (ta/mnl.py, cfg "auc_weight"). The final model leaves it off.
"""

import torch
import torch.nn.functional as F


def row_log_odds(logits):
    """logit P(click candidate j), including all rivals and the outside option.

    Directly summing rivals avoids log(1-exp(logp)) cancellation at extreme logits.
    The result is invariant to an event-specific additive logit shift.
    """
    k = logits.shape[1] - 1
    mask = torch.eye(k, k + 1, device=logits.device, dtype=torch.bool)
    rivals = logits[:, None, :].expand(-1, k, -1).masked_fill(mask[None], -torch.inf)
    return logits[:, :k] - torch.logsumexp(rivals, dim=-1)


def global_pair_loss(logits, labels, negatives=16, generator=None):
    """Mean logistic pair loss; negatives=None enumerates all pairs for tiny checks.

    Uniform random negatives include rows from no-click events and other users.
    This is a minibatch estimate of a pairwise AUC surrogate, not AUC itself.
    """
    score = row_log_odds(logits)
    y = labels[:, None] == torch.arange(score.shape[1], device=labels.device)[None]
    pos, neg = score[y], score[~y]
    if not pos.numel() or not neg.numel():
        return logits.sum() * 0
    if negatives is None:
        selected = neg[None, :]
    else:
        idx = torch.randint(len(neg), (len(pos), negatives), device=logits.device, generator=generator)
        selected = neg[idx]
    return F.softplus(selected - pos[:, None]).mean()
