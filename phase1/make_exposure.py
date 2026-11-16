"""Exposure scores per cutoff (CPU only), averaged over BPR seeds: cache/expo_mean[<tag>]_<cutoff>.npz.

    python make_exposure.py <cutoffs> <k> <seeds> <part>/<parts> [keep tag]

Each seed fits BPR with k factors on what users were shown (ta/expo.fit_scores) and scores every shown row; the scores
are averaged over seeds. The exposure family (run_chain_exposure.py) of the final file used

    python make_exposure.py 1737936000 10 0,1,2 0/1
"""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from ta import data, expo  # noqa: E402
from ta.harness import fold  # noqa: E402
from ta.paths import CACHE  # noqa: E402

cuts, k, seeds, part = (
    [int(c) for c in sys.argv[1].split(",")],
    int(sys.argv[2]),
    [int(s) for s in sys.argv[3].split(",")],
    sys.argv[4],
)
i, parts = map(int, part.split("/"))
keep, tag = (int(sys.argv[5]), sys.argv[6]) if len(sys.argv) > 6 else (None, "")
A = data.load()
nu = int(A["user"].max()) + 1
for c in cuts[i::parts]:
    f = CACHE / f"expo_mean{tag}_{c}.npz"
    if f.exists():
        continue
    t0 = time.time()
    fit, tgt = fold(A, c)
    parts_xf, parts_xt = zip(*[expo.fit_scores(A, c, fit, tgt, nu, k=k, seed=s, cache=False, keep=keep) for s in seeds])
    xf, xt = np.mean(parts_xf, 0), np.mean(parts_xt, 0)
    tmp = f.with_name(f"{f.stem}.tmp.npz")
    np.savez(tmp, xf=xf, xt=xt, k=k, seeds=np.array(seeds))
    tmp.replace(f)
    print(f"cutoff {c} done in {time.time() - t0:.0f}s", flush=True)
print("precompute done", part, flush=True)
