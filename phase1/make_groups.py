"""Item groupings found in what users are shown (ta/cats.fit_merge), per cutoff: cache/cats3_<cutoff>_k<K>.npz.

    python make_groups.py <cutoffs> <part>/<parts> [K, default 10] [seed, default 0]

A seed other than 0 writes cache/cats3_<cutoff>_k<K>_s<seed>.npz: an independent estimate of the same grouping.
Built on the fit events and the target slates (no click labels). The final file needs, at the evaluation cutoff
1737936000: K = 10 with seeds 0, 1, 2 (the main and side families' mixes) and K = 20 with seeds 0 to 7 (the fine groups
of the momentum, newest-3 and as-of terms, and the 20-group families), e.g.

    python make_groups.py 1737936000 0/1 10 1
    python make_groups.py 1737936000 0/1 20 7

CPU only. TA_BPR_THREADS caps the BPR threads.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from ta import cats, data  # noqa: E402
from ta.harness import fold  # noqa: E402
from ta.paths import CACHE  # noqa: E402

cuts = [int(c) for c in sys.argv[1].split(",")]
k, parts = map(int, sys.argv[2].split("/"))
K = int(sys.argv[3]) if len(sys.argv) > 3 else 10
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
A = data.load()
nu = int(A["user"].max()) + 1
for c in cuts[k::parts]:
    f = CACHE / (f"cats3_{c}_k{K}_s{seed}.npz" if seed else f"cats3_{c}_k{K}.npz")
    if f.exists():
        continue
    t0 = time.time()
    fit, t = fold(A, c)
    z, pi, g, next_agree = cats.fit_merge(A, np.r_[fit, t], nu, K=K, seed=seed)
    tmp = f.with_name(f"{f.stem}.tmp{os.getpid()}.npz")  # write, then rename: readers never see half a file
    np.savez(tmp, z=z, pi=pi, g=g)
    os.replace(tmp, f)
    print(
        c,
        "sizes",
        np.sort(np.bincount(z, minlength=10))[::-1],
        "agreement at the next merge",
        round(next_agree, 3),
        round(time.time() - t0),
        "s",
        flush=True,
    )
