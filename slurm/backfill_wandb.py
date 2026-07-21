"""Backfill wandb from the metric files of already-completed vmcnet runs.

vmcnet writes per-epoch metrics to plain text files in each logdir (one value
per line, via utils.io.write_metric_to_file) independently of wandb. That means
a run launched with wandb disabled can still be replayed into wandb afterward
without recomputing anything.

Subsampling matches vmcnet's own live behavior: train.vmc caps wandb at
MAX_WANDB_LOGS = 10000 points via `wandb_freq = nepochs // min(nepochs,
MAX_WANDB_LOGS)`, so the resulting charts look the same as a live run's.

Run this from a LOGIN node (compute nodes have no reliable outbound network),
using the env's interpreter:

    /global/scratch/users/$USER/envs/vmcnet/bin/python slurm/backfill_wandb.py \
        /global/scratch/users/$USER/vmcnet_logs \
        --runs carbon_spring carbon_prime_sr carbon_same_sampled_spring_unified \
        --project vmcnet-carbon --group carbon

Requires `wandb login` to have been run once (writes ~/.netrc).
"""

import argparse
import json
import os
from glob import glob

import numpy as np
import wandb

# Matches train.vmc.MAX_WANDB_LOGS.
MAX_WANDB_LOGS = 10000

# Written by _save_git_hash, not a metric.
NON_METRIC_TXT = {"git_hash"}


def _load_metrics(run_dir):
    """Load every per-epoch metric file in run_dir as {name: array}."""
    metrics = {}
    for path in sorted(glob(os.path.join(run_dir, "*.txt"))):
        name = os.path.splitext(os.path.basename(path))[0]
        if name in NON_METRIC_TXT:
            continue
        values = np.loadtxt(path)
        if values.ndim != 1 or values.size == 0:
            print(f"  skipping {name}: unexpected shape {values.shape}")
            continue
        metrics[name] = values
    return metrics


def backfill_run(run_dir, run_name, project, group, dry_run):
    """Replay one run's metric files into a single wandb run."""
    metrics = _load_metrics(run_dir)
    if not metrics:
        print(f"  no metric files found in {run_dir}, skipping")
        return

    # Files can differ in length by an epoch or two depending on when the job
    # ended; truncate to the shortest so every step has a complete row.
    nepochs = min(len(v) for v in metrics.values())
    stride = max(1, nepochs // min(max(nepochs, 1), MAX_WANDB_LOGS))

    summary = ", ".join(f"{k} ({len(v)})" for k, v in sorted(metrics.items()))
    print(f"  metrics: {summary}")
    print(f"  logging {len(range(0, nepochs, stride))} points (stride {stride})")

    if dry_run:
        return

    config = {}
    config_path = os.path.join(run_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)

    run = wandb.init(
        project=project,
        group=group,
        name=run_name,
        config=config,
        reinit=True,
        notes=f"Backfilled from {run_dir} (metrics replayed, not recomputed)",
    )
    for epoch in range(0, nepochs, stride):
        run.log({k: float(v[epoch]) for k, v in metrics.items()}, step=epoch)
    run.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logroot", help="Directory holding the run subfolders.")
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Run subfolder names; each becomes one wandb run of the same name.",
    )
    parser.add_argument("--project", default="vmcnet-carbon")
    parser.add_argument("--group", default="carbon")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be logged without contacting wandb.",
    )
    args = parser.parse_args()

    for run_name in args.runs:
        run_dir = os.path.join(args.logroot, run_name)
        print(f"{run_name}:")
        if not os.path.isdir(run_dir):
            print(f"  {run_dir} does not exist, skipping")
            continue
        backfill_run(run_dir, run_name, args.project, args.group, args.dry_run)


if __name__ == "__main__":
    main()
