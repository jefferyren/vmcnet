"""Iterations-to-target from the slurm .out files -- no new GPU time, no wandb.

WHY THIS EXISTS. Final eval energy answers "how accurate does this arm get"; it does
not answer "how long does it take to get somewhere useful", which is the practically
relevant question once every arm is inside chemical accuracy anyway. Every training
line in the .out files carries the UNCLIPPED energy, so this is recoverable for every
run already on disk.

METRIC. Trailing-W-epoch mean of energy_noclip against a target energy:
  --target chem   reference + 1.5936 mHa (1 kcal/mol)
  --target prime  the level PRIME-SR reaches at the end of training in that same
                  condition (median over its seeds) -- use this where chemical accuracy
                  is out of reach, i.e. N2 and CO
Two crossing definitions are reported because they can differ by 4x:
  first  = first epoch the trailing mean drops below target
  stable = first epoch after which it NEVER returns above target  (the headline one)
A run that never crosses is censored and printed as >Nk -- censored runs are excluded
from the median and the count is printed, so a censored arm cannot silently look fast.

CAVEATS, all of which the reader should be told:
  - this is the TRAINING energy, a proxy for the eval-phase number. Directionally fine
    for ranking arms, not the same estimator as the published eval energy.
  - iterations are not wall-clock: SS-SPRING pays for its probe every step. Measure that
    separately (see --walls) before quoting a speedup as a cost saving.
  - H4 has no literature reference, so it has no chemical-accuracy target.

Usage, from the repo root (where the slurm-*.out files live):

    python slurm/time_to_target.py                      # both targets, W=1000
    python slurm/time_to_target.py --window 500 2000    # sensitivity to the window
    python slurm/time_to_target.py --run slurm-e9-atoms-37270317_32.out   # one run
"""

import argparse
import re
from collections import defaultdict
from datetime import datetime
from glob import glob

import numpy as np

CA = 1.5936e-3  # Ha, 1 kcal/mol

# Same references as slurm/parse_eval_energies.py -- keep the two in sync.
REF = {"N": -54.5892, "O": -75.0673, "carbon": -37.8450,
       "N2_eq": -109.5423, "CO": -113.3255}

# Training lines carry the noclip energy in parentheses; eval lines do not, so this
# regex also acts as the training/eval filter.
TRAIN = re.compile(r"Epoch\s+(\d+), Energy: -?[\d.]+e[+-]\d+ \((-?[\d.]+e[+-]\d+)\)")
HEAD = re.compile(r"^E(\d+) arm=")
DONE = re.compile(r"^E\d+ done\s+(\d{4}-\d\d-\d\dT[\d:+-]+)")

# E7 prints tuned_mu on every line regardless of arm, so map by TAG, not by tuned_mu.
LABEL = {"ssu_defaults": "SS-SPRING", "prime_sr": "PRIME-SR",
         "spring_default": "SPRING(0.99)", "spring_tuned": "SPRING(tuned)",
         "spring_mu0.9": "SPRING(0.9)", "spring_mu0.95": "SPRING(0.95)",
         "spring_mu0.99": "SPRING(0.99)", "spring_mu0.995": "SPRING(0.995)"}
ORDER = ["SS-SPRING", "SPRING(tuned)", "SPRING(0.995)", "SPRING(0.99)",
         "SPRING(0.95)", "SPRING(0.9)", "PRIME-SR"]
DEFAULT_ETA = {"7": "0.02", "9": "0.02", "10": "0.002", "11": "0.02"}
DEFAULT_PRESET = {"11": "carbon"}


def load(paths, experiments):
    """Parse (system, eta, arm, seed) -> training curve, keeping the longest duplicate.

    Only the production experiments are included by default. E6/E8 ran 25k epochs with
    the eval phase disabled and would otherwise merge into the carbon eta=0.02 cell and
    corrupt it.
    """
    runs = {}
    for path in sorted(paths):
        lines = open(path, errors="ignore").readlines()
        head = next((ln.strip() for ln in lines[:6] if HEAD.match(ln)), None)
        if head is None:
            continue  # phase-A scripts predate the header line
        exp = HEAD.match(head).group(1)
        if exp not in experiments:
            continue
        arm = re.search(r"arm=(\S+)", head).group(1)
        seed = re.search(r"seed=(\d+)", head)
        if arm not in LABEL or seed is None:
            continue
        preset = re.search(r"preset=(\S+)", head)
        preset = preset.group(1) if preset else DEFAULT_PRESET.get(exp)
        eta = re.search(r"\beta=([\d.]+)", head)
        eta = eta.group(1) if eta else DEFAULT_ETA.get(exp, "?")
        if preset not in REF:
            continue
        ep, en, end = [], [], None
        for ln in lines:
            m = TRAIN.search(ln)
            if m:
                ep.append(int(m.group(1)))
                en.append(float(m.group(2)))
            elif DONE.match(ln):
                end = datetime.fromisoformat(DONE.match(ln).group(1))
        if not ep:
            continue
        ep, en = np.array(ep), np.array(en)
        ep, en = ep[np.r_[True, np.diff(ep) > 0]], en[np.r_[True, np.diff(ep) > 0]]
        wall = ((end - datetime.fromisoformat(head.split()[-1])).total_seconds() / 3600
                if end else np.nan)
        key = (preset, eta, LABEL[arm], int(seed.group(1)))
        if key not in runs or len(ep) > len(runs[key]["ep"]):
            runs[key] = dict(ep=ep, en=en, wall=wall, path=path)
    return runs


def trailing(r, w):
    if len(r["en"]) < w:
        return None, None
    return r["ep"][w - 1:], np.convolve(r["en"], np.ones(w) / w, mode="valid")


def cross(r, thr, w):
    """(first, stable) crossing epochs; inf where the run never gets below thr."""
    ep, k = trailing(r, w)
    if ep is None or not (k < thr).any():
        return np.inf, np.inf
    below = k < thr
    bad = np.where(~below)[0]
    stable = np.inf if bad[-1] == len(below) - 1 else ep[bad[-1] + 1]
    return ep[np.argmax(below)], stable


def show(x, cap):
    return f">{cap // 1000}k" if not np.isfinite(x) else f"{int(round(x / 100) * 100):,}"


def median(v):
    fin = [x for x in v if np.isfinite(x)]
    return np.median(fin) if fin else np.inf


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--window", type=int, nargs="+", default=[1000],
                   help="trailing-mean window(s) in epochs; pass several to see the "
                        "sensitivity of the ordering to this choice")
    p.add_argument("--target", choices=["chem", "prime", "both"], default="both")
    p.add_argument("--glob", default="slurm-*.out")
    p.add_argument("--run", help="inspect a single .out file instead")
    p.add_argument("--experiments", nargs="+", default=["7", "9", "10", "11"],
                   help="experiment numbers to include (default: the 50k+eval ones)")
    p.add_argument("--walls", action="store_true",
                   help="also print mean wall-clock hours per arm (see the caveat: the "
                        "scatter between nominally identical arms is ~10%%)")
    args = p.parse_args()

    runs = load([args.run] if args.run else glob(args.glob), set(args.experiments))
    if not runs:
        raise SystemExit(f"no parseable runs matched {args.run or args.glob}")

    groups = defaultdict(lambda: defaultdict(dict))
    for (sysm, eta, arm, seed), r in runs.items():
        groups[(sysm, eta)][arm][seed] = r

    for w in args.window:
        print(f"\n{'#' * 78}\n# trailing-mean window W = {w} epochs\n{'#' * 78}")
        for sysm, eta in sorted(groups, key=lambda k: (k[0], float(k[1]) if k[1] != "?" else 0)):
            arms = groups[(sysm, eta)]
            cap = max(len(r["ep"]) for a in arms.values() for r in a.values())
            # PRIME-SR's end-of-training level in this condition, median over its seeds
            lv = [trailing(r, w)[1][-1] for r in arms.get("PRIME-SR", {}).values()
                  if trailing(r, w)[1] is not None]
            par = np.median(lv) if lv else None
            hdr = f"\n=== {sysm}, eta={eta}   chem target = ref + 1.594 mHa"
            if par is not None:
                hdr += f"   |   PRIME-SR level = ref + {(par - REF[sysm]) * 1000:.2f} mHa"
            print(hdr)
            cols = f"{'arm':15s} {'n':>2s}"
            if args.target in ("chem", "both"):
                cols += f" {'chem:stable':>12s} {'chem:first':>11s}"
            if args.target in ("prime", "both") and par is not None:
                cols += f" {'prime:stable':>13s}"
            print(cols + (f" {'wall h':>7s}" if args.walls else "") + "   per-seed (chem:stable)")
            for arm in ORDER:
                if arm not in arms:
                    continue
                rs = [arms[arm][s] for s in sorted(arms[arm])]
                ca = [cross(r, REF[sysm] + CA, w) for r in rs]
                line = f"{arm:15s} {len(rs):2d}"
                if args.target in ("chem", "both"):
                    line += (f" {show(median([c[1] for c in ca]), cap):>12s}"
                             f" {show(median([c[0] for c in ca]), cap):>11s}")
                if args.target in ("prime", "both") and par is not None:
                    pa = [cross(r, par, w)[1] for r in rs]
                    line += f" {show(median(pa), cap):>13s}"
                if args.walls:
                    wl = [r["wall"] for r in rs if np.isfinite(r["wall"])]
                    line += f" {np.mean(wl) if wl else float('nan'):7.2f}"
                cens = sum(1 for c in ca if not np.isfinite(c[1]))
                note = f"  [{cens}/{len(rs)} censored]" if cens else ""
                print(line + "   " + ",".join(show(c[1], cap) for c in ca) + note)


if __name__ == "__main__":
    main()
