"""Leak-free "as-of" features: every value uses only events strictly before the event it describes.

Because a validation or evaluation target is a user's FIRST event after a cutoff, its as-of features use exactly the
events before the cutoff, so one global computation serves every fold and the real evaluation file.

In the final model these feed only the point-estimate choice model (ta/mnl.py) that starts the sampler: the event
features shift the no-click utility, and the candidate features (has the user clicked this item before, how recently,
how often was it shown to them) feed a small repeat-click network. The sampler keeps that network's output as a frozen
offset with a sampled multiplier.
"""

import os

import numpy as np

from ta.data import H
from ta.paths import CACHE

# Column names, in order (documentation only).
EF_NAMES = [
    "log_hlen",
    "empty_hist",
    "log_h_since_click",
    "log_prior_events",
    "prior_click_rate",
    "log_h_since_prev_event",
    "no_prior_event",
    "tb0",
    "tb1",
    "tb2",
    "tb3",
    "weekend",
]
CF_NAMES = [
    "in_hist_cnt",
    "in_hist",
    "last_click_recency",
    "log_h_since_item_click",
    "log_prior_shows",
    "prior_noclick_shows",
    "is_last_hist",
]


def load(A):
    """(EF [E, len(EF_NAMES)], CF [E, 10, len(CF_NAMES)]), cached in cache/."""
    fe, fc = CACHE / "EF.npy", CACHE / "CF.npy"
    if not (fe.exists() and fc.exists()):
        for f, build in ((fe, event_features), (fc, cand_features)):
            tmp = f.with_name(f"{f.stem}.tmp{os.getpid()}.npy")  # write, then rename: readers never see half a file
            np.save(tmp, build(A))
            os.replace(tmp, f)
    return np.load(fe), np.load(fc)


def _group_cumcount(keys_sorted_equal_prev):
    """keys_sorted_equal_prev[i] is True when row i is in the same group as row i-1."""
    n = len(keys_sorted_equal_prev)
    start = ~keys_sorted_equal_prev
    idx = np.arange(n)
    grp_start = np.maximum.accumulate(np.where(start, idx, 0))
    return idx - grp_start, grp_start


def event_features(A):
    """Per-event features [E, 12]: history length, time since the last click, the user's earlier events and click
    rate (smoothed), time since their previous event, time-of-day bucket and weekend."""
    E = len(A["user"])
    ts, user, hlen = A["ts"], A["user"], A["hlen"].astype(np.int64)
    last_ts = A["hist_ts"][np.arange(E), np.maximum(hlen - 1, 0)]
    h_since_click = np.where(hlen > 0, (ts - last_ts) / 3600.0, 0.0)

    order = np.lexsort((ts, user))
    same = np.r_[False, user[order][1:] == user[order][:-1]]
    prior, gstart = _group_cumcount(same)
    clicked = (A["label"][order] >= 0) & (A["label"][order] < 10)
    csum = np.cumsum(clicked)
    prior_clicks = csum - clicked - np.where(gstart > 0, csum[gstart - 1], 0)
    prev_ts = np.r_[0, ts[order][:-1]]
    h_prev = np.where(same, (ts[order] - prev_ts) / 3600.0, 0.0)

    pe, pc, hp = (np.empty(E) for _ in range(3))
    pe[order], pc[order], hp[order] = prior, prior_clicks, h_prev
    rate = (pc + 0.63 * 2) / (pe + 2)
    tb = np.eye(4)[A["tb"]]
    ef = np.column_stack(
        [np.log1p(hlen), hlen == 0, np.log1p(h_since_click), np.log1p(pe), rate, np.log1p(hp), pe == 0, tb, A["dt"]]
    ).astype(np.float32)
    return ef


def cand_features(A):
    """Per-candidate features [E, 10, 7]: is the item in the user's click history (how often, how recently, is it the
    last click), and how often it was shown to this user before, with and without a click."""
    E = len(A["user"])
    cand, hist, hlen, ts = A["cand"], A["hist"], A["hlen"].astype(np.int64), A["ts"]
    match = hist[:, None, :] == cand[:, :, None]  # [E,10,20]
    cnt = match.sum(-1)
    # last match or -1
    pos = np.where(match, np.arange(H, dtype=np.int8)[None, None, :], np.int8(-1)).max(-1).astype(np.int64)
    inh = pos >= 0
    from_end = np.where(inh, hlen[:, None] - 1 - pos, H)
    t_last = np.take_along_axis(A["hist_ts"], np.maximum(pos, 0), axis=1)
    h_item = np.where(inh, (ts[:, None] - t_last) / 3600.0, 0.0)
    is_last = inh & (from_end == 0)

    # prior shows of this (user, item) and how many of them were not clicked
    u = np.repeat(A["user"], 10)
    it = cand.ravel()
    t = np.repeat(ts, 10)
    lab = np.repeat(A["label"], 10)
    slot = np.tile(np.arange(10), E)
    clk = (lab == slot).astype(np.int64)
    order = np.lexsort((t, it, u))
    same = np.r_[False, (u[order][1:] == u[order][:-1]) & (it[order][1:] == it[order][:-1])]
    prior, gstart = _group_cumcount(same)
    cs = np.cumsum(clk[order])
    prior_clk = cs - clk[order] - np.where(gstart > 0, cs[gstart - 1], 0)
    shows, noclk = np.empty(len(u)), np.empty(len(u))
    shows[order], noclk[order] = prior, prior - prior_clk

    cf = np.stack(
        [
            cnt,
            inh,
            np.where(inh, 1.0 / (1 + from_end), 0.0),
            np.log1p(h_item),
            np.log1p(shows.reshape(E, 10)),
            np.log1p(noclk.reshape(E, 10)),
            is_last,
        ],
        -1,
    )
    return cf.astype(np.float32)
