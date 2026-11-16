"""Load the organisers' Parquet files into dense per-event NumPy arrays, cached in cache/arrays.npz.

One row per event, training events first, then evaluation events:

    user, ts            user id and Unix time of the event
    tb, dt              time-of-day bucket (0-3) and day type (weekday / weekend)
    is_eval             True for the events to predict
    hlen, hist [E, 20]  the user's last clicks before the event, oldest to newest, left-aligned, padded with -1
    hist_ts [E, 20]     their times (padding 0)
    cand [E, 10]        the 10 items shown
    label               clicked slot 0-9, 10 = no click, -1 = unknown (evaluation events)

Every other module works on this dict of arrays (called `A` throughout).
"""

import numpy as np
import polars as pl

from ta.paths import CACHE, DATA

H = 20  # length of the click history attached to each event
CUT = 1737936000  # 2025-01-27 00:00 UTC, end of the training window (evaluation events come after it)
# 2025-01-20 00:00 UTC, start of the training window; history clicks before it have no known slate
T0 = 1737331200
DAY = 86400
N_ITEMS = 50000


def _pad(lists, dtype, fill):
    """A Polars list column -> left-aligned [rows, H] array padded with `fill`, plus the list lengths."""
    lens = lists.list.len().to_numpy().astype(np.int64)
    flat = lists.explode().drop_nulls().to_numpy().astype(dtype)
    out = np.full((len(lens), H), fill, dtype)
    row = np.repeat(np.arange(len(lens)), lens)
    pos = np.arange(len(flat)) - np.repeat(np.cumsum(lens) - lens, lens)
    out[row, pos] = flat
    return out, lens


def _build():
    """Read the four Parquet files and assemble the arrays described in the module docstring."""
    parts = []
    for split in ["train", "eval"]:
        e = pl.read_parquet(DATA / f"{split}_events.parquet").sort("event")
        r = pl.read_parquet(DATA / f"{split}_rows.parquet")
        hist, hlen = _pad(e["hist_items"], np.int32, -1)
        hist_ts, _ = _pad(e["hist_ts"], np.int64, 0)
        cand = r["item"].to_numpy().astype(np.int32).reshape(-1, 10)
        if split == "train":
            clk = r["clicked"].to_numpy().reshape(-1, 10)
            label = np.where(clk.any(1), clk.argmax(1), 10).astype(np.int8)
        else:
            label = np.full(len(cand), -1, np.int8)
        parts.append(
            dict(
                user=e["user"].to_numpy().astype(np.int32),
                ts=e["ts"].to_numpy().astype(np.int64),
                tb=e["time_bucket"].to_numpy().astype(np.int8),
                dt=e["day_type"].to_numpy().astype(np.int8),
                is_eval=np.full(len(cand), split == "eval"),
                hlen=hlen.astype(np.int8),
                hist=hist,
                hist_ts=hist_ts,
                cand=cand,
                label=label,
            )
        )
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def load():
    """The event arrays as a dict; built from data/ on the first call, then read from the cache."""
    f = CACHE / "arrays.npz"
    if not f.exists():
        CACHE.mkdir(exist_ok=True)
        np.savez(f, **_build())
    z = np.load(f)
    return {k: z[k] for k in z.files}


def eval_rowids():
    """rowid strings for eval events, shape [E_eval, 10], aligned with the eval part of load()."""
    return pl.read_parquet(DATA / "eval_rows.parquet")["rowid"].to_numpy().reshape(-1, 10)
