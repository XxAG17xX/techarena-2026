"""Round 4 final entry: rank each user's own past clicks with a small LightGBM model.

Team NextToken, Huawei Ireland TechArena 2026, Phase 2 on-site final (Dublin, 15 November 2026).

Authorship: written by OpenAI Codex during the event, as the team's second AI assistant, and tidied
for publication by Claude Code (Anthropic). The tidy-up renamed variables, split long lines, added
comments and removed two test-only switches. It changes no prediction: the original and this
version give the same item for all 400,000 round 4 requests (see README.md).

How it works
------------
Each request asks for one item for one user at one moment. In this competition a user's own past
clicks were by far the strongest candidates, so the question is which of them to recommend.

1. This process (the parent) first computes a simple rule and yields straight away every request a
   model cannot improve: brand-new users, users with no clicked item available, and users with only
   one clicked item available at the request time.
2. A worker subprocess builds a strictly-past index of the feedback, trains a LightGBM lambdarank
   model on round 3's pooled slates (lists in which a user saw two or more of their own past clicks
   and picked one of them), then scores every request's candidates. It writes its picks to a
   temporary folder in chunks; the parent checks each chunk and yields it.
3. If the worker crashes, writes anything invalid, or is not finished 95 s after Python start, the
   parent stops it and answers the remaining requests with the rule. The organisers' limit is 240 s.

The training happens inside the scoring run, so the submission holds no weights: just this file.
The worker is a separate process so that it can be stopped at the deadline, and so that a crash in
it cannot take down the parent, which has already yielded part of the answer.
"""
import time

# Taken before the heavy imports: the organisers' time limit counts from Python start.
START_TIME = time.monotonic()

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------------------------

# Start of round 3's request window (9 Jan 2025, 00:00 UTC). Feedback from this time on is round 3's
# pooled slates: every team's pick for a request shown together, and which one the user clicked.
ROUND3_START = 1736380800

# Search keys combine a (user, item) pair number and a time as pair * KEY_STRIDE + time. Any stride
# larger than every timestamp (about 1.74e9) keeps the keys in (pair, time) order. In round 4 the
# largest key is about 1.6e8 pairs x 4e9 = 6.5e17, far below the 64-bit limit of about 9.2e18.
KEY_STRIDE = 4_000_000_000

# Seconds after Python start at which the parent stops waiting for the worker and uses the rule.
MODEL_BUDGET = 95.0

# Requests are handled in chunks of this many rows, which keeps memory low and lets results stream.
CHUNK_ROWS = 20000

# With fewer training lists than this, the feedback is not what the model expects: use the rule.
MIN_TRAINING_LISTS = 5000

# Command-line flag that makes this file run as the worker (see the end of the file).
WORKER_FLAG = "--worker"

# Threads for LightGBM training and scoring.
THREADS = 8

# LightGBM lambdarank settings. Codex validated a tree with these settings on held-out users
# (train on half the users, test on the other half; see README.md). deterministic=True with a fixed
# thread count makes the trained model repeatable. lambdarank_truncation_level=1: only the top pick
# of each list matters, as in the scoring.
RANKER_SETTINGS = dict(
    objective="lambdarank",
    n_estimators=160,
    learning_rate=0.035,
    num_leaves=7,
    min_child_samples=300,
    reg_lambda=10,
    n_jobs=THREADS,
    verbosity=-1,
    random_state=2301,
    deterministic=True,
    force_col_wise=True,
    lambdarank_truncation_level=1,
)

# Time constants (hours) of the exponentially decayed click counts.
DECAY_HOURS = [3, 8, 24, 72]

# The rule (fallback): score = clicks on the item in separate events, minus this penalty if it was
# shown again after the last click and ignored. Measured on round 3's pooled slates (see README.md).
RULE_IGNORED_PENALTY = 0.5
# Rule tie-break: up to 0.3 points for a later click, less than the 0.5 step, so it only breaks ties.
RULE_RECENCY_TIE_BREAK = 0.3

# The 38 model inputs, in the order the model sees them. A candidate is one item the user clicked
# before. Every value uses only feedback from strictly before the request (or training slate) time.
# "Rejected" = shown in a slate where the user clicked another item; "silent" = shown in a slate
# where the user clicked nothing. The "slate_" inputs compare a candidate with the others in its list.
FEATURES = [
    "log_age",                 # log(1 + hours since the user's last click on this item)
    "log_newer",               # log(1 + the user's clicks on anything since then)
    "slate_latest",            # 1 if it is the most recently clicked candidate
    "user_latest",             # 1 if it is the user's most recent click of all
    "age3",                    # exp(-hours since last click / 3)
    "age24",                   # exp(-hours since last click / 24)
    "log_first_age",           # log(1 + hours since the first click on it)
    "slate_log_age_gap",       # log_age minus the smallest log_age among the candidates
    "log_clicks",              # log(1 + clicks on it)
    "log_r3clicks",            # log(1 + clicks on it in round 3)
    "twice",                   # 1 if clicked at least twice
    "log_shows",               # log(1 + times shown)
    "empirical_ctr",           # clicks / (times shown + 1)
    "log_r3shows",             # log(1 + times shown in round 3)
    "slate_log_count_ratio",   # log((clicks + 0.5) / (most clicks of any candidate + 0.5))
    "log_ignored",             # log(1 + times shown since the last click)
    "log_rejected",            # log(1 + of those, times rejected)
    "log_silent",              # log(1 + of those, times silent)
    "log_show_age",            # log(1 + hours since last shown)
    "log_r3rejects",           # log(1 + times rejected in round 3 after the first click)
    "log_r3silent",            # log(1 + times silent in round 3 after the first click)
    "slate_rejection_diff",    # rejections since the last click minus the fewest among the candidates
    "last_click_r3",           # 1 if the last click was in round 3
    "last_click_event",        # the user's event number at the last click / 120
    "first_click_event",       # the user's event number at the first click / 120
    "last_click_competitors",  # number of other items in the slate of the last click
    "hour_distance",           # hours between the request's and the last click's time of day (0-12)
    "same_weekday",            # 1 if the request is on the same weekday as the last click
    "same_weekend",            # 1 if both are at the weekend, or both on weekdays
    "same_6h",                 # 1 if both fall in the same 6-hour block of the day
    "click_hour_alignment",    # mean cosine between the request's hour and each past click's hour
    "ewma_clicks_3h",          # log(1 + sum over past clicks of exp(-hours ago / 3))
    "ewma_clicks_8h",          # the same with 8 hours
    "ewma_clicks_24h",         # the same with 24 hours
    "ewma_clicks_72h",         # the same with 72 hours
    "r3_ctr_logit",            # log((round 3 clicks + 1) / (round 3 shows without a click + 2))
    "r3_choice_logit",         # log((round 3 clicks + 1) / (round 3 rejections + 1))
    "personal_ctr_logit",      # log((clicks + 1) / (shows without a click + 2))
]


# ---------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------

def log(message):
    """Print a progress line with the seconds since start (print() output goes to the run log)."""
    print("[round4 %5.1fs] %s" % (time.monotonic() - START_TIME, message), flush=True)


def read_requests_and_catalogue(data):
    """Read x_pred.csv (rowid, user_id, timestamp) and catalog.csv (itemID, valid_from, valid_to)."""
    requests = pd.read_csv(data / "x_pred.csv").drop_duplicates("rowid").reset_index(drop=True)
    catalogue = pd.read_csv(data / "catalog.csv").drop_duplicates("itemID").reset_index(drop=True)
    return requests, catalogue


def read_feedback(data, cutoff):
    """Read feedback_long (one row per item shown) and keep only the rows before `cutoff`.

    The cut at the first request time is a precaution: if a data snapshot ever overlapped the
    requests, no outcome from the period being predicted could leak into the features.
    """
    columns = ["event_id", "user_id", "timestamp", "itemID", "clicked"]
    try:
        feedback = pd.read_parquet(data / "feedback_long.parquet", columns=columns)
    except Exception:
        feedback = pd.read_csv(data / "feedback_long.csv", usecols=columns)
    return feedback[feedback.timestamp < cutoff].reset_index(drop=True)


def run_starts(labels):
    """For an array in which equal labels sit next to each other: where each run starts, and its length."""
    starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
    lengths = np.diff(np.r_[starts, len(labels)])
    return starts, lengths


def best_in_each_run(scores, starts, lengths):
    """Position of the highest score in each run; on a tie, the first such position."""
    run_of_row = np.repeat(np.arange(len(starts)), lengths)
    run_max = np.maximum.reduceat(scores, starts)
    positions = np.where(scores == run_max[run_of_row], np.arange(len(scores)), len(scores))
    return np.minimum.reduceat(positions, starts)


def first_available(order, valid_from, valid_to, times):
    """For each request time, the first catalogue position in `order` whose item is available then.

    Items are checked 256 at a time, and a request stops being checked once it has its item, so a
    popularity order usually settles in the first block. Raises an error if a request has no
    available item at all.
    """
    picked = np.full(len(times), -1, np.int32)
    todo = np.arange(len(times))
    for offset in range(0, len(order), 256):
        if not len(todo):
            break
        block = order[offset:offset + 256]
        request_times = times[todo, None]
        available = (valid_from[block][None, :] <= request_times) & (request_times < valid_to[block][None, :])
        found = available.any(axis=1)
        picked[todo[found]] = block[available[found].argmax(axis=1)]
        todo = todo[~found]
    if len(todo):
        raise ValueError("No catalogue item available for %d requests" % len(todo))
    return picked


def save_atomically(path, **arrays):
    """Write an .npz file under a temporary name, then rename it, so the parent never reads half a file."""
    temp = path.with_suffix(".partial")
    with temp.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temp, path)


# ---------------------------------------------------------------------------------------------
# The rule: the fallback for every request, and the answer wherever the model has no choice to make
# ---------------------------------------------------------------------------------------------

def popularity_order(feedback, catalogue):
    """Catalogue positions sorted by number of clicks in the feedback, most clicked first."""
    clicks = feedback.loc[feedback.clicked == 1, "itemID"].value_counts()
    n_clicks = catalogue.itemID.map(clicks).fillna(0).to_numpy()
    return np.argsort(-n_clicks, kind="stable")


def rule_scores(feedback):
    """One row per (user, item the user clicked), with the rule's score.

    kc = the user's event number at the last click on the item, ncl = number of events in which it
    was clicked, ign = times shown again after the last click.
    """
    # Number each user's events in time order: k = 0, 1, 2, ...
    events = feedback.drop_duplicates("event_id")[["event_id", "user_id", "timestamp"]]
    events = events.sort_values(["user_id", "timestamp", "event_id"])
    events["k"] = events.groupby("user_id").cumcount()
    shown = feedback.merge(events[["event_id", "k"]], on="event_id")

    clicks = shown[shown.clicked == 1]
    pairs = clicks.groupby(["user_id", "itemID"]).agg(kc=("k", "max"), ncl=("k", "nunique")).reset_index()

    # Showings of a clicked item after the last click on it (none of them was clicked).
    again = shown.merge(pairs[["user_id", "itemID", "kc"]], on=["user_id", "itemID"])
    ignored = again[again.k > again.kc].groupby(["user_id", "itemID"]).size().rename("ign").reset_index()
    pairs = pairs.merge(ignored, on=["user_id", "itemID"], how="left").fillna({"ign": 0})

    last_event = max(float(events.k.max()), 1.0)
    pairs["score"] = (pairs.ncl
                      - RULE_IGNORED_PENALTY * (pairs.ign >= 1)
                      + RULE_RECENCY_TIE_BREAK * pairs.kc / last_event)
    return pairs


def rule_choices(feedback, requests, catalogue):
    """The rule's pick for every request (a catalogue position), and each request's own-item count.

    - a user with clicked items available at the request time: the best by rule_scores();
    - a brand-new user: the most-clicked item available for the whole request period;
    - anyone else: the most-clicked item available at the request time.
    own_count = how many of the user's clicked items are available at the request time.
    """
    times = requests.timestamp.to_numpy()
    valid_from = catalogue.valid_from.to_numpy()
    valid_to = catalogue.valid_to.to_numpy()

    order = popularity_order(feedback, catalogue)
    choice = first_available(order, valid_from, valid_to, times)

    # Brand-new users: one item for all their requests, so it must be available for the whole period.
    is_new = ~requests.user_id.isin(set(feedback.user_id.unique())).to_numpy()
    whole_period = order[(valid_from[order] <= times.min()) & (valid_to[order] > times.max())]
    if len(whole_period):
        choice[is_new] = whole_period[0]

    own_count = np.zeros(len(requests), np.int16)
    if not len(feedback) or not int(feedback.clicked.sum()):
        return choice, own_count

    pairs = rule_scores(feedback)
    pairs = pairs.merge(catalogue[["itemID", "valid_from", "valid_to"]], on="itemID")   # listed items only
    pairs["ci"] = pd.Index(catalogue.itemID.to_numpy()).get_indexer(pairs.itemID)
    for start in range(0, len(requests), CHUNK_ROWS):
        chunk = requests.iloc[start:start + CHUNK_ROWS][["user_id", "timestamp"]].copy()
        chunk["rid"] = np.arange(start, start + len(chunk))
        options = chunk.merge(pairs, on="user_id")
        options = options[(options.valid_from <= options.timestamp) & (options.timestamp < options.valid_to)]
        if not len(options):
            continue
        counts = options.groupby("rid").size()
        own_count[counts.index] = counts.to_numpy()
        # Highest score per request; equal scores go to the smaller itemID.
        best = options.sort_values(["rid", "score", "itemID"], ascending=[True, False, True])
        best = best.drop_duplicates("rid")
        choice[best.rid.to_numpy()] = best.ci.to_numpy()
    return choice, own_count


# ---------------------------------------------------------------------------------------------
# The model's view of the history
# ---------------------------------------------------------------------------------------------

class ClickHistory:
    """Answers "what happened between this user and this item before time t?" for many rows at once.

    Every feedback row (one item shown in one slate) is sorted by (user-item pair, time, event) and
    gets the search key pair * KEY_STRIDE + time. A binary search on the keys then finds, for any
    (user, item, t), the block of that pair's rows strictly before t, and running totals over the
    sorted rows turn counts inside the block into two look-ups. User and item IDs are only used to
    find the history; they are never model inputs.
    """

    def __init__(self, feedback):
        self.users = np.array(sorted(feedback.user_id.unique()))
        self.items = np.array(sorted(feedback.itemID.unique()))
        self.n_items = len(self.items)

        # One entry per feedback row, in file order.
        self.raw_user = pd.Index(self.users).get_indexer(feedback.user_id).astype(np.int64)
        self.raw_item = pd.Index(self.items).get_indexer(feedback.itemID).astype(np.int64)
        self.raw_time = feedback.timestamp.to_numpy(np.int64)
        self.raw_click = feedback.clicked.to_numpy(np.int8)

        # Slates (events), numbered in (user, time, event_id) order; k = the user's event number.
        events = feedback.drop_duplicates("event_id")[["event_id", "user_id", "timestamp"]]
        events = events.sort_values(["user_id", "timestamp", "event_id"])
        events["k"] = events.groupby("user_id").cumcount()
        self.raw_event = pd.Index(events.event_id).get_indexer(feedback.event_id)
        slate_size = np.bincount(self.raw_event)
        clicks_in_slate = np.bincount(self.raw_event, weights=self.raw_click)
        self.raw_slate_clicked = clicks_in_slate[self.raw_event] > 0
        event_number = events.k.to_numpy()[self.raw_event]
        competitors = slate_size[self.raw_event] - 1

        # Sort the rows by (pair, time, event) and build the search keys.
        raw_pair = self.raw_user * self.n_items + self.raw_item
        order = np.lexsort((self.raw_event, self.raw_time, raw_pair))
        self.pair = raw_pair[order]
        self.time = self.raw_time[order]
        self.click = self.raw_click[order]
        self.event_number = event_number[order]
        self.competitors = competitors[order]
        self.keys = self.pair * KEY_STRIDE + self.time

        # Running totals over the sorted rows: rejected_before[p] = rejected rows before position p
        # (another item in the slate was clicked), silent_before[p] = silent rows (nothing clicked).
        rejected = (self.raw_slate_clicked & ~(self.raw_click == 1))[order]
        silent = (~self.raw_slate_clicked)[order]
        self.rejected_before = np.r_[0, np.cumsum(rejected, dtype=np.int64)]
        self.silent_before = np.r_[0, np.cumsum(silent, dtype=np.int64)]

        # Sorted positions of the clicks, and every user's click times (for "clicks since then").
        self.click_rows = np.flatnonzero(self.click == 1)
        if not len(self.click_rows):
            raise ValueError("No click history for the ranker")
        clicked = self.raw_click == 1
        self.user_click_keys = np.sort(self.raw_user[clicked] * KEY_STRIDE + self.raw_time[clicked])

        # Running sums over each pair's clicks in time order: time of day as a point on a 24-hour
        # circle (cosine and sine), and decayed click weights. The weights are taken relative to
        # ROUND3_START so that exp() stays in range; the features undo that shift.
        click_pair = self.pair[self.click_rows]
        click_time = self.time[self.click_rows]
        pair_run = np.cumsum(np.r_[True, click_pair[1:] != click_pair[:-1]]) - 1
        angle = 2 * np.pi * (click_time % 86400) / 86400
        self.cos_sum = pd.Series(np.cos(angle)).groupby(pair_run, sort=False).cumsum().to_numpy()
        self.sin_sum = pd.Series(np.sin(angle)).groupby(pair_run, sort=False).cumsum().to_numpy()
        self.decayed_sum = {}
        for tau in DECAY_HOURS:
            weight = np.exp((click_time - ROUND3_START) / (3600 * tau))
            self.decayed_sum[tau] = pd.Series(weight).groupby(pair_run, sort=False).cumsum().to_numpy()

    def locate(self, user, item, t):
        """Find each (user, item) pair's rows strictly before time t.

        Returns pair; lo and hi (the rows are sorted positions lo to hi - 1); first_click and
        last_click (indexes into click_rows of the first and last click in the block); n_clicks.
        """
        pair = user.astype(np.int64) * self.n_items + item
        lo = np.searchsorted(self.keys, pair * KEY_STRIDE, "left")
        hi = np.searchsorted(self.keys, pair * KEY_STRIDE + t, "left")
        first_click = np.searchsorted(self.click_rows, lo, "left")
        last_click = np.searchsorted(self.click_rows, hi, "left") - 1
        n_clicks = np.maximum(last_click - first_click + 1, 0)
        return pair, lo, hi, first_click, last_click, n_clicks

    def features(self, user, item, t, group):
        """Model inputs for candidate rows: a float32 matrix with the columns in FEATURES order.

        One candidate per row (user, item, t). `group` names the list each row belongs to (a training
        slate or a request), with each list's rows next to each other. Every candidate must be an item
        the user clicked before t.
        """
        pair, lo, hi, first_click, last_click, n = self.locate(user, item, t)
        if not np.all(n > 0):
            raise ValueError("Non-own item passed to own-item model")
        last_row = self.click_rows[last_click]   # sorted position of the last click before t
        first_row = self.click_rows[first_click]
        last_time = self.time[last_row]
        first_time = self.time[first_row]

        # Boundaries inside the pair's block of rows.
        after_last = np.searchsorted(self.keys, pair * KEY_STRIDE + last_time, "right")
        after_first = np.searchsorted(self.keys, pair * KEY_STRIDE + first_time, "right")
        r3_start = np.searchsorted(self.keys, pair * KEY_STRIDE + ROUND3_START, "left")
        r3_start = np.minimum(r3_start, hi)
        r3_after_first = np.maximum(r3_start, after_first)
        r3_after_first = np.minimum(r3_after_first, hi)

        # Counts inside the block.
        r3_clicks = np.maximum(last_click - np.searchsorted(self.click_rows, r3_start, "left") + 1, 0)
        shows = hi - lo
        r3_shows = hi - r3_start
        ignored = hi - after_last
        rejected = self.rejected_before[hi] - self.rejected_before[after_last]
        silent = self.silent_before[hi] - self.silent_before[after_last]
        r3_rejected = self.rejected_before[hi] - self.rejected_before[r3_after_first]
        r3_silent = self.silent_before[hi] - self.silent_before[r3_after_first]
        newer = (np.searchsorted(self.user_click_keys, user * KEY_STRIDE + t, "left")
                 - np.searchsorted(self.user_click_keys, user * KEY_STRIDE + last_time, "right"))
        age = (t - last_time) / 3600
        first_age = (t - first_time) / 3600

        # Per list: the smallest age, the most clicks and the fewest rejections among its candidates.
        starts, lengths = run_starts(group)
        list_of_row = np.repeat(np.arange(len(starts)), lengths)
        min_age = np.minimum.reduceat(age, starts)[list_of_row]
        max_clicks = np.maximum.reduceat(n, starts)[list_of_row]
        min_rejected = np.minimum.reduceat(rejected, starts)[list_of_row]

        # Clock: hour of the day (UTC) and weekday (0 = Monday; 1 January 1970 was a Thursday, hence + 3).
        hour = (t % 86400) / 3600
        last_hour = (last_time % 86400) / 3600
        hour_distance = np.abs(hour - last_hour)
        hour_distance = np.minimum(hour_distance, 24 - hour_distance)
        weekday = (t // 86400 + 3) % 7
        last_weekday = (last_time // 86400 + 3) % 7
        alignment = (self.cos_sum[last_click] * np.cos(hour * np.pi / 12)
                     + self.sin_sum[last_click] * np.sin(hour * np.pi / 12)) / n

        columns = {
            "log_age": np.log1p(age),
            "log_newer": np.log1p(newer),
            "slate_latest": age == min_age,
            "user_latest": newer == 0,
            "age3": np.exp(-age / 3),
            "age24": np.exp(-age / 24),
            "log_first_age": np.log1p(first_age),
            "log_clicks": np.log1p(n),
            "log_r3clicks": np.log1p(r3_clicks),
            "twice": n >= 2,
            "log_shows": np.log1p(shows),
            "empirical_ctr": n / (shows + 1),
            "log_ignored": np.log1p(ignored),
            "log_rejected": np.log1p(rejected),
            "log_silent": np.log1p(silent),
            "log_show_age": np.log1p((t - self.time[hi - 1]) / 3600),
            "log_r3shows": np.log1p(r3_shows),
            "log_r3rejects": np.log1p(r3_rejected),
            "log_r3silent": np.log1p(r3_silent),
            "last_click_r3": last_time >= ROUND3_START,
            "last_click_event": self.event_number[last_row] / 120,
            "first_click_event": self.event_number[first_row] / 120,
            "last_click_competitors": self.competitors[last_row],
            "hour_distance": hour_distance,
            "same_weekday": weekday == last_weekday,
            "same_weekend": (weekday >= 5) == (last_weekday >= 5),
            "same_6h": hour // 6 == last_hour // 6,
            "click_hour_alignment": alignment,
            "slate_log_age_gap": np.log1p(age) - np.log1p(min_age),
            "slate_log_count_ratio": np.log((n + 0.5) / (max_clicks + 0.5)),
            "slate_rejection_diff": rejected - min_rejected,
            "r3_ctr_logit": np.log((r3_clicks + 1) / (np.maximum(r3_shows - r3_clicks, 0) + 2)),
            "r3_choice_logit": np.log((r3_clicks + 1) / (r3_rejected + 1)),
            "personal_ctr_logit": np.log((n + 1) / (shows - n + 2)),
        }
        for tau in DECAY_HOURS:
            # sum over past clicks of exp(-(t - click time) / tau hours)
            decayed = self.decayed_sum[tau][last_click] * np.exp(-(t - ROUND3_START) / (3600 * tau))
            columns["ewma_clicks_%dh" % tau] = np.log1p(decayed)
        return np.column_stack([columns[name] for name in FEATURES]).astype(np.float32)

    def training_lists(self):
        """Training data from round 3's pooled slates.

        A list = one slate in which the user saw two or more items they had clicked before and
        clicked one of them. Returns the inputs, the labels (1 = the clicked item) and the list sizes.
        """
        rows = np.flatnonzero((self.raw_time >= ROUND3_START) & self.raw_slate_clicked)
        n_clicks = self.locate(self.raw_user[rows], self.raw_item[rows], self.raw_time[rows])[-1]
        rows = rows[n_clicks > 0]   # own items only: clicked before this slate
        slate = self.raw_event[rows]
        own_per_slate = np.bincount(slate, minlength=int(self.raw_event.max()) + 1)
        own_wins_per_slate = np.bincount(slate, weights=self.raw_click[rows], minlength=len(own_per_slate))
        rows = rows[(own_per_slate[slate] >= 2) & (own_wins_per_slate[slate] > 0)]
        rows = rows[np.argsort(self.raw_event[rows], kind="stable")]   # each slate's rows together
        X = self.features(self.raw_user[rows], self.raw_item[rows], self.raw_time[rows], self.raw_event[rows])
        _, list_sizes = run_starts(self.raw_event[rows])
        return X, self.raw_click[rows], list_sizes

    def clicked_pairs(self):
        """Every (user, item) pair with at least one click, as user index u and item index i."""
        pairs = np.unique(self.pair[self.click_rows])
        return pd.DataFrame({"u": pairs // self.n_items, "i": pairs % self.n_items})


# ---------------------------------------------------------------------------------------------
# The worker subprocess: train, then score every request's candidates
# ---------------------------------------------------------------------------------------------

def train_and_score(data_dir, out_dir):
    """Train the ranker, then write each request's best own item to chunk files in out_dir."""
    data = Path(data_dir)
    out = Path(out_dir)
    requests, catalogue = read_requests_and_catalogue(data)
    feedback = read_feedback(data, int(requests.timestamp.min()))
    history = ClickHistory(feedback)
    del feedback
    log("causal history index ready")

    import lightgbm as lgb   # only the worker needs LightGBM

    X, labels, list_sizes = history.training_lists()
    if len(list_sizes) < MIN_TRAINING_LISTS:
        raise ValueError("Too few own-click training lists")
    ranker = lgb.LGBMRanker(**RANKER_SETTINGS)
    ranker.fit(X, labels, group=list_sizes)
    model = ranker.booster_
    log("trained on %d lists / %d candidates" % (len(list_sizes), len(labels)))
    del X, labels, list_sizes

    # Candidates: every (user, item) pair with a click, with the item's catalogue position and window.
    pairs = history.clicked_pairs()
    pairs["itemID"] = history.items[pairs.i.to_numpy()]
    catalogue["ci"] = np.arange(len(catalogue))
    pairs = pairs.merge(catalogue[["itemID", "ci", "valid_from", "valid_to"]], on="itemID")
    request_user = pd.Index(history.users).get_indexer(requests.user_id)   # -1 for a brand-new user

    for start in range(0, len(requests), CHUNK_ROWS):
        stop = min(start + CHUNK_ROWS, len(requests))
        chunk = pd.DataFrame({"rid": np.arange(start, stop), "u": request_user[start:stop],
                              "t": requests.timestamp.to_numpy()[start:stop]})
        options = chunk.merge(pairs, on="u")
        options = options[(options.valid_from <= options.t) & (options.t < options.valid_to)]
        options = options.sort_values(["rid", "i"])
        path = out / ("chunk_%06d.npz" % start)
        if len(options):
            rid = options.rid.to_numpy()
            starts, lengths = run_starts(rid)
            X = history.features(options.u.to_numpy(), options.i.to_numpy(), options.t.to_numpy(), rid)
            scores = model.predict(X, num_threads=THREADS)
            best = best_in_each_run(scores, starts, lengths)
            save_atomically(path, rid=rid[best].astype(np.int32), ci=options.ci.to_numpy()[best].astype(np.int32))
        else:
            save_atomically(path, rid=np.array([], np.int32), ci=np.array([], np.int32))
    (out / "DONE").write_text("complete", encoding="utf-8")


# ---------------------------------------------------------------------------------------------
# The parent: the entry point the scorer calls
# ---------------------------------------------------------------------------------------------

def start_worker(data, work_dir, log_file):
    """Start this same file as the worker subprocess, its output going to log_file."""
    env = os.environ.copy()
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = str(THREADS)
    command = [sys.executable, str(Path(__file__).resolve()), WORKER_FLAG, str(data), str(work_dir)]
    return subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT, env=env)


def read_worker_chunk(path, n_requests, n_items, times, valid_from, valid_to):
    """Load one chunk of the worker's picks and check it. Returns request and catalogue positions."""
    with np.load(path) as chunk:
        rows = chunk["rid"]
        choices = chunk["ci"]
    if len(rows) != len(choices) or len(np.unique(rows)) != len(rows):
        raise ValueError("Invalid worker chunk")
    if np.any(rows < 0) or np.any(rows >= n_requests) or np.any(choices < 0) or np.any(choices >= n_items):
        raise ValueError("Out-of-range worker choice")
    if not np.all((valid_from[choices] <= times[rows]) & (times[rows] < valid_to[choices])):
        raise ValueError("Unavailable worker item")
    return rows, choices


def predict(data_dir, batch_size=256):
    """Yield lists of (rowid, itemID) string pairs, answering every request exactly once."""
    data = Path(data_dir).resolve()
    batch_size = max(1, int(batch_size or 256))
    requests, catalogue = read_requests_and_catalogue(data)
    if not len(requests):
        return
    rowids = requests.rowid.astype(str).to_numpy()
    item_ids = catalogue.itemID.astype(str).to_numpy()
    times = requests.timestamp.to_numpy()
    valid_from = catalogue.valid_from.to_numpy()
    valid_to = catalogue.valid_to.to_numpy()
    answered = np.zeros(len(requests), bool)
    deadline = START_TIME + MODEL_BUDGET

    with tempfile.TemporaryDirectory(prefix="round4_") as temp_dir:
        work_dir = Path(temp_dir)
        worker = None
        worker_log = (work_dir / "worker.log").open("w", encoding="utf-8")
        try:
            worker = start_worker(data, work_dir, worker_log)
        except Exception as error:
            log("worker launch failed; using rule: %r" % error)
        try:
            # 1. The rule, while the worker trains.
            try:
                feedback = read_feedback(data, int(times.min()))
            except Exception as error:
                log("feedback unavailable; source-rule population fallback: %r" % error)
                feedback = pd.DataFrame(columns=["event_id", "user_id", "timestamp", "itemID", "clicked"])
            try:
                fallback, own_count = rule_choices(feedback, requests, catalogue)
            except Exception as error:
                log("personal rule failed; population fallback: %r" % error)
                fallback = first_available(popularity_order(feedback, catalogue), valid_from, valid_to, times)
                own_count = np.zeros(len(requests), np.int16)
            del feedback

            # 2. Requests with at most one own item available: the model has nothing to choose.
            early = np.flatnonzero(own_count <= 1)
            log("rule ready; yielding %d fallback/single-candidate rows" % len(early))
            for offset in range(0, len(early), batch_size):
                rows = early[offset:offset + batch_size]
                answered[rows] = True
                yield [(str(rowids[j]), str(item_ids[fallback[j]])) for j in rows]

            # 3. The worker's picks, chunk by chunk as they appear, until all are in, the worker
            #    fails, or the deadline passes.
            consumed = set()
            worker_failed = False
            while worker is not None and not answered.all() and time.monotonic() < deadline:
                for path in sorted(work_dir.glob("chunk_*.npz")):
                    if path.name in consumed:
                        continue
                    try:
                        rows, choices = read_worker_chunk(path, len(requests), len(catalogue), times,
                                                          valid_from, valid_to)
                        keep = ~answered[rows]
                        rows = rows[keep]
                        choices = choices[keep]
                        for offset in range(0, len(rows), batch_size):
                            batch_rows = rows[offset:offset + batch_size]
                            batch_choices = choices[offset:offset + batch_size]
                            answered[batch_rows] = True
                            yield [(str(rowids[j]), str(item_ids[k])) for j, k in zip(batch_rows, batch_choices)]
                        consumed.add(path.name)
                    except Exception as error:
                        log("worker output failed; remaining rows use rule: %r" % error)
                        worker_failed = True
                        break
                if worker_failed:
                    break
                if (work_dir / "DONE").exists() or worker.poll() is not None:
                    # The worker may write its last chunk between our listing and this check:
                    # read any chunk still waiting before giving up on it.
                    pending = any(p.name not in consumed for p in work_dir.glob("chunk_*.npz"))
                    if not pending:
                        break
                    continue
                time.sleep(0.05)
            if worker is not None and worker.poll() is None:
                worker.kill()
                worker.wait(timeout=5)

            # 4. Anything the worker did not answer gets the rule's pick.
            rest = np.flatnonzero(~answered)
            if len(rest):
                log("worker failed/timed out; %d remaining rows use the rule" % len(rest))
            for offset in range(0, len(rest), batch_size):
                rows = rest[offset:offset + batch_size]
                answered[rows] = True
                yield [(str(rowids[j]), str(item_ids[fallback[j]])) for j in rows]
        finally:
            if worker is not None and worker.poll() is None:
                worker.kill()
                worker.wait(timeout=5)
            worker_log.close()
            # Copy the end of the worker's log into the run log.
            try:
                tail = (work_dir / "worker.log").read_text(encoding="utf-8", errors="replace").splitlines()[-8:]
                for line in tail:
                    log("worker: " + line)
            except Exception:
                pass
    log("complete: %d/%d rows" % (answered.sum(), len(requests)))


if __name__ == "__main__" and len(sys.argv) >= 4 and sys.argv[1] == WORKER_FLAG:
    train_and_score(sys.argv[2], sys.argv[3])
