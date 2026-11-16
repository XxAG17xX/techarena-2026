"""Assemble the submission from all sampler runs: a weighted mix of the model families on the log-odds scale, plus a few
end terms computed directly from the data.

    python build_ensemble.py <name> <family weights> [end terms] [seeds] [manifest]

For each family (a prefix of run_chain.py or run_chain_exposure.py), its runs at the evaluation cutoff are averaged as
probabilities, then turned into log-odds; the families are mixed with the given weights (normalised to sum to 1). The
end terms are added on the log-odds scale, the result is turned back into probabilities and written as
submissions/<name>.csv and .csv.zip through ta/submit.py. seeds lists the run seeds to look for (default 0-5); manifest
is an optional JSON file listing each family's run files with their SHA-256 hashes, to freeze the set of runs.

The final upload (public AUC 0.834618) was built with these four arguments (each one string, no spaces):

    name     chains_r29
    weights  gibbsq1=0.085,gibbsq1c1=0.085,gibbsq1c2=0.085,gibbsq1n3=0.255,gibbsq3=0.0283,gibbsq3c1=0.0283,
             gibbsq3c2=0.0284,gibbsq3n3=0.085,gibbsx=0.17,gibbsqi=0.075,gibbsqin3=0.075
    end      ctx=1.0,lift2=0.05,abs=0.2,abend=0.18,oldf=-0.11,n3o=0.1,n3u=-0.05,n3k=7,afo=0.1,afu=-0.05,afk=7
    seeds    0,1,2,...,1300 (every seed from 0 to 1300; families use the seeds whose run files exist)

End terms of the final file (each computed from the data, on top of what the runs already hold):
  ctx               time-of-day and weekend shift of each category's click rate (ta/gibbsc.context_shift)
  lift2             the square of the log of the user's cleaned category lift r (share of the category in the
                    user's cleaned mix over its overall share)
  abs, abend, oldf  abs * log(max(a, abend)) + oldf * log(r + 0.7), a = the user's cleaned share of the category:
                    corrects the form of the category term the runs hold
  n3o, n3u, n3k     newest-3 fine-group matches averaged over n3k other estimates of the 20 groups (weight n3o),
                    minus a little of the estimate the runs used (weight n3u): the runs lean on the noise in their
                    own grouping
  afo, afu, afk     the same for the as-of fine-group term
Other end terms were explored during the competition and are not included.
"""

import sys
import zipfile
import json
import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from ta import data, gibbsc, submit  # noqa: E402
from ta.data import CUT  # noqa: E402
from ta.harness import fold, log_run  # noqa: E402
from ta.paths import CACHE, SUBS  # noqa: E402

name = sys.argv[1]
parse = lambda x: {k: float(v) for k, v in (kv.split("=") for kv in x.split(",") if kv)}
w, add = parse(sys.argv[2]), parse(sys.argv[3]) if len(sys.argv) > 3 else {}
seeds = [int(s) for s in sys.argv[4].split(",")] if len(sys.argv) > 4 else list(range(6))
# Optional frozen source manifest: concurrent workers cannot change this build's chain set.
manifest = json.loads(Path(sys.argv[5]).read_text()) if len(sys.argv) > 5 else None
actual_seeds = {}
A = data.load()
nu = int(A["user"].max()) + 1
K = 10
fit, t = fold(A, CUT)
lg = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))

# ---- families: average each family's runs as probabilities, then mix the families' log-odds
tot = sum(w.values())
z_ = np.zeros((len(t), 10))
for n, v in w.items():  # a family always comes from its run files
    if manifest is not None:
        entries = manifest["families"][n]
        files = []
        for entry in entries:
            assert entry["name"] == f"{n}_{CUT}_s{int(entry['seed'])}.npy"
            f = CACHE / entry["name"]
            assert hashlib.sha256(f.read_bytes()).hexdigest() == entry["sha256"], f
            files.append(f)
        actual_seeds[n] = [entry["seed"] for entry in entries]
    else:  # every listed seed whose run file exists
        files = [CACHE / f"{n}_{CUT}_s{s}.npy" for s in seeds if (CACHE / f"{n}_{CUT}_s{s}.npy").exists()]
        actual_seeds[n] = [int(f.stem.rsplit("_s", 1)[1]) for f in files]
    chains = [np.load(f, allow_pickle=False) for f in files]
    assert chains and all(c.shape == (len(t), 10) and np.isfinite(c).all() for c in chains), n
    print(n, "chains", len(chains), "weight", round(v / tot, 3), flush=True)
    z_ += v / tot * lg(np.mean(chains, 0))

# ---- end terms, on the log-odds scale (k = the category of each shown row, u = its user)
if add:
    z = np.load(CACHE / f"cats3_{CUT}_k10.npz")["z"]
    k, u = z[A["cand"][t]], A["user"][t]
    if "lift2" in add or "abs" in add:
        Cc, gc, _ = gibbsc.clean_counts(A, np.r_[fit, t], nu, z, K)
        r_ = ((Cc + 20 * gc) / (Cc.sum(1, keepdims=True) + 20) / gc)[u[:, None], k]
        lift = np.log(r_)
        z_ += add.get("lift2", 0.0) * lift**2
        if "abs" in add:  # what is left of the term on the absolute share after the chains
            a_ = ((Cc + 20 * gc) / (Cc.sum(1, keepdims=True) + 20))[u[:, None], k]
            z_ += add["abs"] * np.log(np.maximum(a_, add.get("abend", 0.18))) + add.get("oldf", 0.0) * np.log(r_ + 0.7)
    # newest-3 matches on fine groups the newest-3 runs did not use (seeds 1 ... n3k),
    # minus a little of the seed-0 ones they did
    if "n3o" in add:
        # average the matches over this many other estimates; 7 was best on the validation folds
        n_other = int(add.get("n3k", 1))
        m1 = np.mean(
            [
                gibbsc.recent_match(A, t, np.load(CACHE / f"cats3_{CUT}_k20_s{j}.npz")["z"])
                for j in range(1, n_other + 1)
            ],
            0,
        )
        m0 = gibbsc.recent_match(A, t, np.load(CACHE / f"cats3_{CUT}_k20.npz")["z"])
        print("seed-0 / seed-1 newest-3 matches differ on", round(float(np.mean(m1 != m0)), 4), "of rows", flush=True)
        z_ += add["n3o"] * m1 + add.get("n3u", 0.0) * m0
    # the as-of fine term (every run with asof uses seed-0 k20 groups) on other k20 estimates (seeds 1 ... afk),
    # minus a little of seed 0
    if "afo" in add:

        def asof_t(zf):
            Kf = int(zf.max()) + 1
            _, gf, _ = gibbsc.clean_counts(A, np.r_[fit, t], nu, zf, Kf)
            return gibbsc.asof_fine(A, fit, t, zf, Kf, gf, z, K, gc, 0.7, 20.0)[1]

        a1 = np.mean(
            [asof_t(np.load(CACHE / f"cats3_{CUT}_k20_s{j}.npz")["z"]) for j in range(1, int(add.get("afk", 3)) + 1)], 0
        )
        z_ += add["afo"] * a1 + add.get("afu", 0.0) * asof_t(np.load(CACHE / f"cats3_{CUT}_k20.npz")["z"])
    if "ctx" in add:
        x = gibbsc.context_shift(A, fit, t, z, K)
        print("context table sd on eval rows", round(float(x.std()), 4), flush=True)
        z_ += add["ctx"] * x
p = 1 / (1 + np.exp(-z_))
assert np.isfinite(p).all() and p.shape == (len(t), 10)

# ---- write and log
np.save(CACHE / f"pred_{name}_eval.npy", p)
out = submit.write(A, t, p, name)
with zipfile.ZipFile(SUBS / f"{name}.csv.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(out, f"{name}.csv")
log_run(
    name,
    fold="eval",
    w=w,
    add=add,
    seeds=seeds,
    actual_seeds=actual_seeds,
    manifest=sys.argv[5] if manifest is not None else None,
    note="mix recipe",
)
print("built", SUBS / f"{name}.csv.zip", flush=True)
