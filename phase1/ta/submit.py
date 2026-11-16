"""Write a submission CSV (rowid, predicted_probability) and check it against the official format rules."""

import numpy as np
import polars as pl

from ta.data import eval_rowids
from ta.paths import SUBS

N_EVAL_ROWS = 529390


def write(A, tgt, probs, name):
    """tgt: indices into A of eval events; probs: [len(tgt), 10]."""
    n_train = int((~A["is_eval"]).sum())
    assert A["is_eval"][tgt].all() and len(np.unique(tgt)) == len(tgt)
    rowid = eval_rowids()[tgt - n_train].ravel()
    p = np.clip(np.asarray(probs, np.float64).ravel(), 0.0, 1.0)
    df = pl.DataFrame({"rowid": rowid, "predicted_probability": p})
    assert df.height == N_EVAL_ROWS, df.height
    assert df["rowid"].n_unique() == N_EVAL_ROWS
    assert np.isfinite(p).all()
    SUBS.mkdir(exist_ok=True)
    out = SUBS / f"{name}.csv"
    df.write_csv(out, float_precision=8)
    print(f"wrote {out}  rows {df.height:,}  mean {p.mean():.5f}", flush=True)
    return out
