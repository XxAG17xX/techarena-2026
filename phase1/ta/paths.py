"""Folder layout, relative to this repository folder (`phase1/`).

data/          the organisers' Parquet files (not included; see the README)
cache/         arrays, features, item groupings, exposure scores and sampler runs, all rebuilt from data/
submissions/   the assembled submission files
runs/          the run journal (journal.jsonl) and per-run sampler histories
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = ROOT / "cache"
SUBS = ROOT / "submissions"
RUNS = ROOT / "runs"
