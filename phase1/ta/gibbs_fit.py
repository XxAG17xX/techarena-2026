"""The starting point of a sampler run: a point estimate of the choice model.

For one cutoff and seed: fit BPR five times on the known clicks (seeds chosen by ta.stack.avg_bpr), project each fit
onto its top 8 directions and the unit sphere, average the five after rotating them onto each other, and train the
point-estimate choice model (ta/mnl.py, ta.stack.MNL8T_CFG) from those vectors. The sampler starts from that model's
unit user and item vectors, item biases and no-click utilities. Its small repeat-click network (on ta/features.py's
candidate features) is kept as a frozen offset per shown row, whose multiplier the sampler learns.
"""

import torch
import torch.nn.functional as F

from ta import cf, mnl, stack


def point_start(A, EF, CFa, c, nu, fit, tgt, seed, T, epochs=None):
    """dict: U, V (unit vectors), b (item biases), c (no-click utilities), a (sharpness), p (the point estimate's
    target probabilities), ge / gt (repeat-click offsets of fit / target rows), n0 (mean no-click offset of fit
    events)."""
    U, V = cf.aligned_mean([cf.unit(*cf.project(*stack._factors(A, c, nu, "bpr", s), 8)) for s in stack.avg_bpr(seed)])
    cfg = {**stack.MNL8T_CFG, "d": 8, "seed": seed}
    if epochs:
        cfg["epochs"] = epochs
    model, p, _ = mnl.train(A, EF, CFa, fit, tgt, cfg, T=T, verbose=False, init=(U, V), early_stop=False)
    dev = T.u.device
    with torch.no_grad():
        fe, te = torch.as_tensor(fit, device=dev), torch.as_tensor(tgt, device=dev)
        g = (
            lambda idx: torch.cat([model.g(T.cf[idx[s : s + 65536]]).squeeze(-1) for s in range(0, len(idx), 65536)])
            .cpu()
            .numpy()
        )
        r = dict(
            U=F.normalize(model.P.weight, dim=1).cpu().numpy(),
            V=F.normalize(model.Q.weight, dim=1).cpu().numpy(),
            b=model.b.weight[:, 0].cpu().numpy(),
            c=(model.c0 + model.cu.weight[:, 0]).cpu().numpy(),
            a=float(model.alpha),
            p=p,
            ge=g(fe),
            gt=g(te),
            n0=model.out(T.ef[fe]).mean().item(),
        )
    del model
    torch.cuda.empty_cache()
    return r
