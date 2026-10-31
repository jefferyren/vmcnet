#!/usr/bin/env python3
"""Stamp divergence and norm-cap fields onto the Phase G runs (and the E14/E15 N2_4.0
runs they are read against), so the Phase G report can plot them.

parse_eval_energies.py --backfill writes the eval energies but not whether a run
diverged, and wandb's sampled history cannot tell (the terminal NaN row is often
dropped). So this reads divergence from the slurm .out files, the source of truth, via
divergence_check.parse, and the pre-registered readout from wandb history:

    x_diverged            1/0, from the "VMC terminated due to Nans" marker
    x_healthy_until       last 100-epoch block within 0.5 Ha of the run's best
    x_cap_frac_0_12k      mean norm_cap_applied over epochs 0-12k (Campaign log 4.15)
    x_cap_frac_8_10k      the same over 8-10k (late cap binding)

Run after `parse_eval_energies.py --experiment G1 G2 G3 G4 G5 --backfill`, from
the repo root with the .out files present:
    python slurm/g_backfill_divergence.py            # write
    python slurm/g_backfill_divergence.py --dry-run  # print only
Re-run it after any re-sync: a sync resets server-side summary fields.
"""

import argparse
import glob
import os

import numpy as np
import wandb

import divergence_check as dc
from parse_eval_energies import EXPERIMENTS as PARSE_EXPERIMENTS

ENTITY = "ren27-university-of-california-berkeley"
# experiment -> wandb project. E14/E15 are the N2_4.0 end of the G1 bond scan.
PROJECTS = {"G1": "vmcnet-phase-g", "G2": "vmcnet-phase-g", "G3": "vmcnet-phase-g",
            "G4": "vmcnet-phase-g", "G5": "vmcnet-phase-g",
            "E14": "vmcnet-phase-e", "E15": "vmcnet-phase-e"}
ARMS = ("ssu_defaults", "spring_mu0.99")


def out_stats(experiment, repo):
    """Run name -> divergence_check.parse() stats, newest .out per array index."""
    spec = dc.EXPERIMENTS[experiment]
    cell = PARSE_EXPERIMENTS[experiment]["cell"]
    stats = {}
    for idx in range(spec["ntasks"]):
        paths = glob.glob(os.path.join(repo, spec["pattern"].format(idx=idx)))
        if not paths:
            continue
        s = dc.parse(max(paths, key=os.path.getmtime), window=2000)
        if s:
            stats[cell(idx)[0]] = s
    return stats


def cap_fractions(run):
    rows = run.scan_history(keys=["_step", "norm_cap_applied"], max_step=12001)
    steps = np.array([r["_step"] for r in rows], float)
    cap = np.array([r["norm_cap_applied"] for r in rows], float)

    def frac(lo, hi):
        m = (steps >= lo) & (steps < hi) & np.isfinite(cap)
        return float(cap[m].mean()) if m.any() else None
    return frac(0, 12000), frac(8000, 10000)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", default=".", help="directory holding the slurm .out files")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    api = wandb.Api(timeout=180)
    for experiment, project in PROJECTS.items():
        stats = out_stats(experiment, args.repo)
        runs = [r for r in api.runs(f"{ENTITY}/{project}",
                                    filters={"config.x_experiment": experiment})
                if r.config.get("x_arm") in ARMS and r.name in stats]
        print(f"{experiment}: {len(runs)} wandb run(s), {len(stats)} .out file(s)")
        for run in runs:
            s = stats[run.name]
            cap012, cap810 = cap_fractions(run)
            fields = {"x_diverged": int(s["diverged"]),
                      "x_healthy_until": s.get("healthy_until"),
                      "x_cap_frac_0_12k": cap012, "x_cap_frac_8_10k": cap810}
            print(f"  {run.name:34} " + " ".join(f"{k}={v}" for k, v in fields.items()))
            if not args.dry_run:
                for k, v in fields.items():
                    if v is not None:
                        run.summary[k] = v
                run.summary.update()


if __name__ == "__main__":
    main()
