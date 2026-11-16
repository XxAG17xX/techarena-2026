"""Self-check for both round entries, with no organiser data: build a tiny made-up round, run each
entry's predict() the way the scorer does, and check the contract.

    python phase2/test_entries.py        (or: python -m pytest phase2)

Contract: every request is answered exactly once, with string IDs, by an item from the catalogue that is
available at the request time. Row IDs carry leading zeros on purpose: read as numbers they would break.
The made-up round is too small for round 4's ranker to train, so round 4 runs its rule fallback here; the
ranker path was checked against the submitted picks on the real round 4 data (see round4/README.md).
"""
import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
T0 = 1_736_000_000                      # history starts here; requests start one day later


def make_round(folder):
    """30 users with 10 three-item events each, 5 brand-new users, 40 items, a few expiring mid-round."""
    rng = np.random.default_rng(0)
    items = [f"itm_{k:02d}" for k in range(40)]
    ends = [T0 + 86_400 + 3_600 * k if k < 4 else 2**40 for k in range(40)]   # items 0-3 expire during the round
    pd.DataFrame({"itemID": items, "valid_from": T0, "valid_to": ends}).to_csv(folder / "catalog.csv", index=False)
    rows = []
    for u in range(30):
        for e in range(10):
            shown = rng.choice(40, 3, replace=False)
            click = rng.integers(-1, 3)                                   # -1: no click
            for j, i in enumerate(shown):
                rows.append((f"e{u}_{e}", f"u{u}", T0 + 600 * (10 * u + e), items[i], int(j == click)))
    pd.DataFrame(rows, columns=["event_id", "user_id", "timestamp", "itemID", "clicked"]).to_csv(
        folder / "feedback_long.csv", index=False)
    users = [f"u{u}" for u in range(35)]                                  # u30-u34 are brand-new
    requests = [(f"{n:06d}", users[n % 35], T0 + 86_400 + 900 * n) for n in range(105)]
    pd.DataFrame(requests, columns=["rowid", "user_id", "timestamp"]).to_csv(folder / "x_pred.csv", index=False)


def check_entry(entry):
    spec = importlib.util.spec_from_file_location(f"{entry}_solution", HERE / entry / "solution.py")
    solution = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(solution)
    with tempfile.TemporaryDirectory() as temp:
        folder = Path(temp)
        make_round(folder)
        answers = [pair for batch in solution.predict(str(folder), 16) for pair in batch]
        requests = pd.read_csv(folder / "x_pred.csv", dtype={"rowid": str, "user_id": str})
        catalog = pd.read_csv(folder / "catalog.csv", dtype={"itemID": str}).set_index("itemID")
    assert all(isinstance(r, str) and isinstance(i, str) for r, i in answers), "IDs must be strings"
    rowids = [r for r, _ in answers]
    assert len(rowids) == len(set(rowids)) == len(requests), "every request answered exactly once"
    assert set(rowids) == set(requests.rowid), "answers use the requests' own row IDs"
    t = requests.set_index("rowid").timestamp
    for rowid, item in answers:
        assert item in catalog.index, f"{item} is not in the catalogue"
        assert catalog.valid_from[item] <= t[rowid] < catalog.valid_to[item], f"{item} unavailable for {rowid}"


def test_round3():
    check_entry("round3")


def test_round4():
    check_entry("round4")


if __name__ == "__main__":
    for entry in ("round3", "round4"):
        check_entry(entry)
        print(f"{entry}: ok")
