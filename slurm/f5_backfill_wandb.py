#!/usr/bin/env python3
"""Stamp the F5 replay runs in wandb with the x_* config fields the reports filter on.

The F5 replays (slurm/f5_beta_star_guard.sbatch) ran with wandb offline and log only
the vmcnet config, so after `wandb sync --legacy` they have no x_experiment / x_system /
x_arm / x_seed. slurm/report_f5_guard.py filters on those. Everything is derived from
the run name f5_<PRESET>_<TAG>_guard, TAG = e14s0 / d7s0 / ... (the replayed state).

Config fields survive a later re-sync (summary fields do not; see CAMPAIGN_LOG section 7),
but re-run this after any sync anyway: it is idempotent.

    python slurm/f5_backfill_wandb.py --dry-run
    python slurm/f5_backfill_wandb.py
"""

import argparse
import collections
import re

import wandb

ENTITY = "ren27-university-of-california-berkeley"
PROJECT = "vmcnet-phase-f"
NAME = re.compile(r"^f5_(?P<preset>.+)_(?P<tag>(?P<exp>[de]\d+)s(?P<seed>\d+))_guard$")
# learning rate and checkpoint epoch of each replayed state, as in the sbatch
ETA = {"N2_4.0": 0.002, "N2_eq": 0.002, "CO": 0.002}
FROM = {"e15s6": 60000}


def fields(name):
    m = NAME.match(name)
    if m is None:
        return None
    preset, tag, exp, seed = m["preset"], m["tag"], m["exp"].upper(), int(m["seed"])
    eta = 0.0015 if exp == "E18" else ETA.get(preset, 0.02)
    start = FROM.get(tag, 10000 if preset == "N2_4.0" else 50000)
    return {
        "x_experiment": "F5",
        "x_system": preset,
        "x_arm": "f5_guard",
        "x_method": "SS-SPRING + beta* guard",
        "x_seed": seed,
        "x_eta": eta,
        "x_adaptive": True,
        "x_replay_of": f"{exp} s{seed}",
        "x_replay_from": start,
        "x_replay_to": start + (10000 if preset == "N2_4.0" else 5000),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    runs = list(wandb.Api(timeout=120).runs(f"{ENTITY}/{PROJECT}"))
    f5 = [r for r in runs if r.name.startswith("f5_")]
    counts = collections.Counter(r.name for r in f5)
    dupes = sorted(n for n, c in counts.items() if c > 1)
    if dupes:
        raise SystemExit(f"duplicate run names (re-synced twice?), fix first: {dupes}")
    print(f"{len(f5)} F5 run(s) in {PROJECT} (expected 12)")
    for r in sorted(f5, key=lambda r: r.name):
        new = fields(r.name)
        if new is None:
            print(f"  skip {r.name}: name does not parse")
            continue
        print(f"  {r.name}: last step {r.lastHistoryStep}; {new}")
        if not args.dry_run:
            r.config.update(new)
            r.update()
    if args.dry_run:
        print("dry run: nothing written")


if __name__ == "__main__":
    main()
