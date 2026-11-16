"""Round 3 entry: recommend each user the most credible item they clicked before.

Team NextToken, Huawei Ireland TechArena 2026, Phase 2 on-site final (Dublin, 15 November 2026).
Written with Claude Code (Anthropic) during the round, then tidied for publication without changing
a single prediction (checked row by row on the round 3 data, see README.md).

There is no training and no weight file: everything is computed from the data folder when
predict() runs.

1. Every item a user clicked gets a score:
   - when the click happened in the user's history. Early clicks (the user's first events, when they
     click almost anything) count far less than late ones (when they rarely click);
   - a bonus if they clicked it in two or more separate events, and a penalty if it was shown to them
     again after their last click on it and they ignored it;
   - a small tie-break in favour of the more recent click.
   Each request gets the user's best-scoring clicked item that is available at the request time.
2. Brand-new users (no history at all) all get the same item: the most-clicked item that stays
   available for the whole round.
3. Everyone else (history, but no clicked item available) gets the most-clicked item available at
   the request time.

If step 1 fails for any reason, its requests are answered by steps 2 and 3 instead.
Requests are yielded in batches as soon as they are ready, the most confident first.
"""
import time

# Taken before the heavy imports: the organisers' 240 s limit counts from Python start.
START_TIME = time.time()

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------------------------
# Settings. Measured on the round 3 history (20,000 users, 100 random 3-item slates each).
# ---------------------------------------------------------------------------------------------

# Score by the user's event number at the time of the click (0 = their first event). The value is
# the log of the measured re-click rate when the item was shown to the user again later:
# 3.5% for clicks in events 0-4, 6.3% in events 5-19, 9.8% from event 20 on.
EVENT_NUMBER_EDGES = [5, 20]
EVENT_NUMBER_SCORES = [np.log(0.035), np.log(0.063), np.log(0.098)]

# Bonus for an item clicked in two or more separate events. Rare in the round 3 history (a few
# hundred user-item pairs), but those items were re-clicked far more often than any other group.
CLICKED_TWICE_BONUS = 1.6

# Penalty for an item shown again after the user's last click on it and not clicked.
# Measured re-click rate 1.54% for such items against 7.14% for the others.
IGNORED_PENALTY = -1.6

# Tie-break: up to 0.3 points for a later click. Smaller than the smallest gap between the other
# score levels (about 0.44), so it only orders items that are otherwise equal.
RECENCY_TIE_BREAK = 0.3


def log(message):
    """Print a progress line with the seconds since start (print() output goes to the run log)."""
    print("[round3 %6.1fs] %s" % (time.time() - START_TIME, message), flush=True)


def to_times(column, missing):
    """Turn a timestamp column into float seconds; a missing or unreadable value becomes `missing`."""
    values = pd.to_numeric(column, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
    return np.where(np.isnan(values), missing, values)


def in_batches(rowids, items, batch_size):
    """Yield lists of (rowid, itemID) pairs, at most batch_size long."""
    for start in range(0, len(rowids), batch_size):
        stop = start + batch_size
        yield list(zip(rowids[start:stop], items[start:stop]))


def read_requests(data):
    """Read x_pred.csv: one request per row (rowid, user_id, timestamp), plus the request time t."""
    requests = pd.read_csv(data / "x_pred.csv", dtype={"rowid": str, "user_id": str})
    requests = requests.drop_duplicates("rowid")
    requests["rowid"] = requests["rowid"].astype(str)
    requests["user_id"] = requests["user_id"].astype(str)
    # A request without a readable time counts as "very late": only items with no end date fit it.
    requests["t"] = to_times(requests["timestamp"], np.inf)
    return requests


def read_catalogue(data):
    """Read catalog.csv: every item of the round and its window [vf, vt) of availability."""
    catalog = pd.read_csv(data / "catalog.csv", dtype={"itemID": str}).drop_duplicates("itemID")
    return pd.DataFrame({
        "itemID": catalog["itemID"].astype(str),
        "vf": to_times(catalog["valid_from"], -np.inf),   # no start date: available from the beginning
        "vt": to_times(catalog["valid_to"], np.inf),      # no end date: never expires
    })


def read_feedback(data):
    """Read feedback_long: one row per item shown (event_id, user_id, timestamp, itemID, clicked)."""
    columns = ["event_id", "user_id", "timestamp", "itemID", "clicked"]
    try:
        feedback = pd.read_parquet(data / "feedback_long.parquet", columns=columns)
    except Exception as error:
        # The CSV holds the same data; it is only slower to read.
        log("feedback_long.parquet not usable (%r), reading the CSV" % (error,))
        feedback = pd.read_csv(data / "feedback_long.csv", usecols=columns)
    for column in ("event_id", "user_id", "itemID"):
        feedback[column] = feedback[column].astype(str)
    feedback["timestamp"] = pd.to_numeric(feedback["timestamp"], errors="coerce").fillna(0).astype(np.int64)
    feedback["clicked"] = pd.to_numeric(feedback["clicked"], errors="coerce").fillna(0).astype(np.int8)
    return feedback


def score_clicked_items(feedback):
    """Score every (user, item the user clicked) pair.

    Returns a table with one row per pair: kc (the user's event number at the last click on the
    item), ncl (number of events in which it was clicked), ign (times shown again after the last
    click) and score. Also returns the feedback with each row's event number k added.
    """
    # Number each user's events in time order: k = 0, 1, 2, ...
    events = feedback.drop_duplicates("event_id")[["event_id", "user_id", "timestamp"]]
    events = events.sort_values(["user_id", "timestamp", "event_id"])
    events["k"] = events.groupby("user_id").cumcount()
    feedback = feedback.merge(events[["event_id", "k"]], on="event_id")

    clicks = feedback[feedback["clicked"] == 1]
    pairs = clicks.groupby(["user_id", "itemID"]).agg(kc=("k", "max"), ncl=("k", "nunique")).reset_index()

    # Showings of a clicked item after the user's last click on it. None of these was clicked,
    # since kc is the last click.
    shown = feedback.merge(pairs[["user_id", "itemID", "kc"]], on=["user_id", "itemID"])
    ignored = shown[shown["k"] > shown["kc"]].groupby(["user_id", "itemID"]).size().rename("ign").reset_index()
    pairs = pairs.merge(ignored, on=["user_id", "itemID"], how="left").fillna({"ign": 0})

    bucket = np.searchsorted(EVENT_NUMBER_EDGES, pairs["kc"].to_numpy(), side="right")
    last_event = max(float(events["k"].max()), 1.0)
    pairs["score"] = (np.array(EVENT_NUMBER_SCORES)[bucket]
                      + CLICKED_TWICE_BONUS * (pairs["ncl"].to_numpy() >= 2)
                      + IGNORED_PENALTY * (pairs["ign"].to_numpy() >= 1)
                      + RECENCY_TIE_BREAK * pairs["kc"].to_numpy() / last_event)
    return pairs, feedback


def best_own_items(requests, catalogue, pairs):
    """For each request whose user has a clicked item available at that time, the best-scoring one.

    Returns rowid, itemID and score, sorted with the most confident requests first.
    """
    pairs = pairs.merge(catalogue, on="itemID")   # items no longer in the catalogue drop out
    options = requests[["rowid", "user_id", "t"]].merge(pairs[["user_id", "itemID", "score", "vf", "vt"]],
                                                        on="user_id")
    options = options[(options["vf"] <= options["t"]) & (options["t"] < options["vt"])]
    # Highest score per request. Equal scores go to the smaller itemID, so the pick never depends
    # on row order.
    best = options.sort_values(["rowid", "score", "itemID"], ascending=[True, False, True])
    best = best.drop_duplicates("rowid")
    # Most confident first: if the run were cut short, the best rows would already be saved.
    return best.sort_values("score", ascending=False)


def first_available(valid_from, valid_to, times, block=256):
    """For each request time, the position of the first item (in the given order) available then.

    Returns -1 where no item is available. Items are checked 256 at a time, and a request stops
    being checked once it has its item, so a popularity order usually settles in the first block.
    """
    pick = np.full(len(times), -1, dtype=np.int64)
    todo = np.arange(len(times))
    for start in range(0, len(valid_from), block):
        if len(todo) == 0:
            break
        request_times = times[todo][:, None]
        available = ((valid_from[None, start:start + block] <= request_times)
                     & (request_times < valid_to[None, start:start + block]))
        found = available.any(axis=1)
        pick[todo[found]] = start + available[found].argmax(axis=1)
        todo = todo[~found]
    return pick


def predict(data_dir, batch_size=256):
    """Yield lists of (rowid, itemID) string pairs: one pair per request, a batch at a time."""
    data = Path(data_dir)
    batch_size = max(1, int(batch_size or 256))
    requests = read_requests(data)
    catalogue = read_catalogue(data)
    log("x_pred rows %d, catalogue items %d" % (len(requests), len(catalogue)))

    # Step 1: each user's best clicked item.
    feedback = None
    answered = []
    try:
        feedback = read_feedback(data)
        pairs, feedback = score_clicked_items(feedback)
        log("clicked pairs %d of %d users" % (len(pairs), pairs["user_id"].nunique()))
        best = best_own_items(requests, catalogue, pairs)
        rowids = best["rowid"].tolist()
        items = [str(item) for item in best["itemID"].tolist()]
        log("rows with an own click: %d (%.1f%%), %d distinct items"
            % (len(rowids), 100.0 * len(rowids) / max(len(requests), 1), len(set(items))))
        for batch in in_batches(rowids, items, batch_size):
            yield batch
        answered = rowids
    except Exception as error:
        log("PERSONAL PART FAILED (%r): most-clicked item for the remaining rows" % (error,))

    rest = requests[~requests["rowid"].isin(answered)]
    if len(rest):
        # Popularity: number of clicks on each catalogue item in the whole feedback.
        try:
            if feedback is None:
                feedback = read_feedback(data)
            click_counts = feedback.loc[feedback["clicked"].to_numpy() == 1, "itemID"].astype(str).value_counts()
        except Exception as error:
            log("no feedback readable (%r)" % (error,))
            click_counts = pd.Series(dtype=np.float64)
        n_clicks = catalogue["itemID"].map(click_counts).fillna(0).to_numpy(dtype=np.float64)
        by_popularity = np.argsort(-n_clicks, kind="stable")   # catalogue positions, most clicked first

        # Step 2: brand-new users (no history at all) all get the most-clicked item that is
        # available for the whole round, from the first request to the last.
        new_rowids = []
        try:
            known_users = set(feedback["user_id"].unique()) if feedback is not None else set()
            new_requests = rest[~rest["user_id"].isin(known_users)]
            if len(new_requests):
                first_time = float(requests["t"].min())
                last_time = float(requests["t"].max())
                whole_round = ((catalogue["vf"].to_numpy()[by_popularity] <= first_time)
                               & (catalogue["vt"].to_numpy()[by_popularity] > last_time))
                candidates = by_popularity[whole_round]
                if len(candidates):
                    item = str(catalogue["itemID"].iloc[candidates[0]])
                    new_rowids = new_requests["rowid"].tolist()
                    new_items = [item] * len(new_rowids)
                    log("brand-new users: %s (%d clicks), available for the whole round"
                        % (item, int(n_clicks[candidates[0]])))
                    for batch in in_batches(new_rowids, new_items, batch_size):
                        yield batch
                    log("rows of brand-new users: %d" % len(new_rowids))
        except Exception as error:
            log("brand-new-user pick failed (%r): they get the most-clicked item" % (error,))

        # Step 3: everyone left gets the most-clicked item available at the request time. If no item
        # is available at all, the most-clicked item of the round is used anyway.
        rest = rest[~rest["rowid"].isin(set(new_rowids))]
        items_by_popularity = catalogue["itemID"].to_numpy(dtype=object)[by_popularity]
        valid_from = catalogue["vf"].to_numpy()[by_popularity]
        valid_to = catalogue["vt"].to_numpy()[by_popularity]
        rest_rowids = rest["rowid"].tolist()
        rest_times = rest["t"].to_numpy()
        chunk = max(batch_size, 8192)
        for start in range(0, len(rest_rowids), chunk):
            pick = first_available(valid_from, valid_to, rest_times[start:start + chunk])
            chosen = [str(item) for item in items_by_popularity[np.maximum(pick, 0)].tolist()]
            for batch in in_batches(rest_rowids[start:start + chunk], chosen, batch_size):
                yield batch
        log("rows without an own click: %d, given the most-clicked available item" % len(rest_rowids))
    log("done: %d rows" % len(requests))
