"""Report which runs diverged, at what epoch, and how healthy they were beforehand.

WHY THIS EXISTS, SEPARATELY FROM parse_eval_energies.py. That script answers "what
energy did this cell reach", and it deliberately refuses to emit a number for a run
that died -- a tail average over a crashed run is a pre-crash average, not a result.
This script answers the opposite question: *when and how* did it die. For a stability
experiment like E15 that is the entire output, and there is no eval energy at all.

The failure epoch is read from the training log, not from wandb, for two reasons: wandb
downsamples history, so the terminal NaN row is frequently missing and a divergence
check against history undercounts; and a run killed by check_for_nans never writes an
eval phase, which is the part wandb drops anyway.

Usage:
    python slurm/divergence_check.py --experiment E14 E15
    python slurm/divergence_check.py --experiment E14 --window 5000
"""

import argparse
import glob
import os
import re
import sys

import numpy as np

# "Epoch  1234, Energy: -1.09e+02 (-1.09e+02), Variance: ..." -- the parenthetical is
# the unclipped value, and it is the one to trust (the clipped energy is biased and the
# bias depends on walker count).
#
# *** The exponent sign is why this is a full float pattern and not a character class.
# An earlier version used [\d.eE+] and so could not match "9.74596e-01". Variance falls
# below 1.0 early in every healthy run, so that version silently stopped matching at
# ~epoch 1800 and reported a 100k run as ending at epoch 2807 -- with no error, and with
# the divergence verdicts still correct, which is exactly what makes it dangerous. ***
_FLOAT = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
EPOCH_RE = re.compile(
    rf"Epoch\s+(\d+), Energy: ({_FLOAT}) \(({_FLOAT})\), "
    rf"Variance: ({_FLOAT}) \(({_FLOAT})\)"
)
NAN_MARKER = "VMC terminated due to Nans! Aborting."

# Job-name globs, matching the sbatch --job-name of each experiment. Keep these
# prefix-disjoint: a pattern that also matches another experiment's files merges the
# two silently, with no error.
EXPERIMENTS = {
    "E14": dict(pattern="slurm-e14-n2-stretched-*_{idx}.out", ntasks=9, nepochs=100000,
                cell=lambda i: (["spring_mu0.99", "prime_sr", "ssu_defaults"][i // 3],
                                i % 3)),
    "E15": dict(pattern="slurm-e15-n2-seedcheck-*_{idx}.out", ntasks=5, nepochs=100000,
                cell=lambda i: ("ssu_defaults", i + 3)),
    # SPRING proper is not the arm that blew up on this system -- SS-SPRING was -- but
    # mu=0.995 is a longer momentum tail than anything E14 ran here, so check rather
    # than assume. Index is the seed; seeds 0-2, shared with E14's SPRING arm.
    "E16": dict(pattern="slurm-e16-n2-mu0995-*_{idx}.out", ntasks=3, nepochs=100000,
                cell=lambda i: ("spring_mu0.995", i)),
    # E17 varies eta, not the arm, so the "arm" label has to carry the eta -- otherwise
    # the per-arm verdict at the bottom of the report pools the two learning rates and
    # prints one meaningless fraction across both.
    "E17": dict(pattern="slurm-e17-n2-eta-*_{idx}.out", ntasks=6, nepochs=100000,
                cell=lambda i: (f"ssu_eta{('0.001', '0.0005')[i // 3]}", i % 3)),
    # Same "arm label carries the eta" convention as E17, so E14/E17/E18 can be parsed
    # in one invocation and read as a single eta axis.
    "E18": dict(pattern="slurm-e18-n2-eta0015-*_{idx}.out", ntasks=3, nepochs=100000,
                cell=lambda i: ("ssu_eta0.0015", i)),
    # E19 varies the norm constraint at fixed eta, so the arm label carries the cap
    # setting. Parse with E14 (`--experiment E14 E19`): E14's 3/3 divergence at
    # C=1e-3 on these same three seeds IS the control.
    "E19": dict(pattern="slurm-e19-n2-normcap-*_{idx}.out", ntasks=6, nepochs=100000,
                cell=lambda i: (("ssu_cap0.01", "ssu_nocap")[i // 3], i % 3)),
}


def parse(path, window):
    """Return what the log says about how this run ended."""
    text = open(path, errors="ignore").read()
    rows = [(int(m.group(1)), float(m.group(3)), float(m.group(5)))
            for m in EPOCH_RE.finditer(text)]
    if not rows:
        return None
    epochs = np.array([r[0] for r in rows])
    energy = np.array([r[1] for r in rows])
    variance = np.array([r[2] for r in rows])

    # Block-average before judging health. A single epoch's unclipped energy swings by
    # ~1 Ha on these systems, so a per-epoch test reports a 0.25 Ha "excursion" on a
    # perfectly healthy run. Blocks of 100 reduce that to the level the divergence
    # itself (4-9 Ha) sits far above.
    nb = max(1, len(energy) // 100)
    usable = nb * 100
    blk_e = energy[:usable].reshape(nb, 100).mean(axis=1)
    blk_ep = epochs[:usable].reshape(nb, 100)[:, -1]

    out = {
        "last_epoch": int(epochs[-1]),
        "diverged": NAN_MARKER in text,
        "final_energy": float(blk_e[-1]),
        "best_energy": float(np.nanmin(blk_e)),
        "best_epoch": int(blk_ep[int(np.nanargmin(blk_e))]),
    }
    # The last block that still sat near the run's own best, and how far it ran away
    # afterwards. A run that ends healthy has healthy_until == last_epoch and an
    # excursion near zero; a cliff shows up as a large gap between the two.
    healthy = blk_e <= out["best_energy"] + 0.5
    if healthy.any():
        out["healthy_until"] = int(blk_ep[int(np.nonzero(healthy)[0][-1])])
        out["excursion_Ha"] = float(blk_e[-1] - out["best_energy"])
    tail = epochs >= max(0, out["last_epoch"] - window)
    out["tail_variance"] = float(np.nanmean(variance[tail]))
    return out


def run(experiment, repo, window):
    spec = EXPERIMENTS[experiment]
    rows = []
    for idx in range(spec["ntasks"]):
        paths = glob.glob(os.path.join(repo, spec["pattern"].format(idx=idx)))
        parsed = [(p, parse(p, window)) for p in paths]
        parsed = [(p, s) for p, s in parsed if s]
        if not parsed:
            rows.append((idx, None, None))
            continue
        # Prefer the most recent file when an array index was resubmitted.
        path, stats = max(parsed, key=lambda ps: os.path.getmtime(ps[0]))
        rows.append((idx, stats, os.path.basename(path)))

    print(f"\n=== {experiment}  ({spec['nepochs']} epochs planned)")
    hdr = (f"{'arm':<16}{'seed':>5}{'diverged':>10}{'last epoch':>12}"
           f"{'healthy until':>15}{'best E [Ha]':>14}{'excursion':>11}")
    print(hdr)
    print("-" * len(hdr))
    by_arm = {}
    for idx, stats, _ in rows:
        arm, seed = spec["cell"](idx)
        if stats is None:
            print(f"{arm:<16}{seed:>5}{'NO LOG':>10}")
            continue
        by_arm.setdefault(arm, []).append(stats["diverged"])
        div = "YES" if stats["diverged"] else "no"
        hu = stats.get("healthy_until", "-")
        exc = stats.get("excursion_Ha")
        print(f"{arm:<16}{seed:>5}{div:>10}{stats['last_epoch']:>12}"
              f"{hu:>15}{stats['best_energy']:>14.4f}"
              f"{(f'{exc:+.3f}' if exc is not None else '-'):>11}")

    print()
    for arm, flags in sorted(by_arm.items()):
        n, d = len(flags), sum(flags)
        verdict = ("consistent divergence" if d == n else
                   "no divergence" if d == 0 else
                   "SEED-DEPENDENT -- report the fraction, not a mean")
        print(f"  {arm:<16} {d}/{n} diverged   {verdict}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", nargs="+", required=True,
                    choices=sorted(EXPERIMENTS))
    ap.add_argument("--repo", default=".", help="directory holding the slurm .out files")
    ap.add_argument("--window", type=int, default=2000,
                    help="epochs to average the tail variance over")
    args = ap.parse_args()
    for experiment in args.experiment:
        run(experiment, args.repo, args.window)
    print()


if __name__ == "__main__":
    sys.exit(main())
