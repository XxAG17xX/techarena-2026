"""Validation folds shaped like the evaluation file, the competition metric (global ROC AUC), and the run journal.

The evaluation file holds each user's first event after the training cutoff. A fold copies that shape: pick a
pseudo-cutoff inside the training window, fit on everything before it, and predict each user's first event after it.
"""

import json
import time

import numpy as np
from sklearn.metrics import roc_auc_score

from ta.data import CUT, DAY
from ta.paths import RUNS

# Pseudo-cutoffs at 00:00 UTC, like the real one. Passing CUT itself means "fit on all training data and predict the
# evaluation file". Most comparisons used jan24 and jan26.
FOLDS = {"jan24": CUT - 3 * DAY, "jan25": CUT - 2 * DAY, "jan26": CUT - DAY}


def fold(A, cutoff):
    """(fit event indices, target event indices). Targets = each user's first event at/after the cutoff."""
    train = ~A["is_eval"]
    fit = np.flatnonzero(train & (A["ts"] < cutoff))
    pool = A["is_eval"] if cutoff >= CUT else train & (A["ts"] >= cutoff)
    idx = np.flatnonzero(pool)
    idx = idx[np.lexsort((A["ts"][idx], A["user"][idx]))]
    first = np.r_[True, A["user"][idx][1:] != A["user"][idx][:-1]]
    return fit, idx[first]


def auc_report(p, label):
    """p: [n,10] scores, label: [n] (0..9 clicked slot, 10 none). Global AUC plus its two components:
    auc_noclick = positives vs rows of no-click events, auc_click = positives vs other rows of click events."""
    y = label[:, None] == np.arange(10)[None, :]
    anyc = label < 10
    pos = p[y]
    neg0 = p[~anyc].ravel()
    neg1 = p[anyc][~y[anyc]]
    two = lambda a, b: roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b])
    return {
        "auc": round(roc_auc_score(y.ravel(), p.ravel()), 5),
        "auc_noclick": round(two(pos, neg0), 5),
        "auc_click": round(two(pos, neg1), 5),
    }


def log_run(name, **info):
    """Append one JSON line to runs/journal.jsonl and print it."""
    RUNS.mkdir(exist_ok=True)
    rec = {"time": time.strftime("%Y-%m-%d %H:%M"), "name": name, **info}
    with open(RUNS / "journal.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)
