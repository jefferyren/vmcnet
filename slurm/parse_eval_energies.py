"""Parse eval-phase energies from slurm .out files and backfill them into wandb.

WHY THIS EXISTS. vmcnet's eval phase restarts its step counter, so wandb drops it
entirely -- yet the eval energy is the publishable number (campaign log s6: "final
energies come from the eval phase, not a training tail"). The only surviving record is
the slurm .out file. Eval lines are distinguishable from training lines because they
lack the noclip parenthetical:

    training:  Epoch 49999, Energy: -5.45888e+01 (-5.45886e+01), Variance: ...
    eval:      Epoch     1, Energy: -5.45920e+01, Variance: ...

WHY IT GLOBS JOB IDS. An array that loses tasks to a node fault gets resubmitted under a
NEW slurm job id, so a single experiment ends up split across several
`slurm-<name>-<JOBID>_<IDX>.out` files. This walks every job id, keys results by array
index, and reports the reconciliation explicitly -- which cells are covered, which are
still missing, and (loudly) any index that completed twice, since a duplicate would
silently double-count in the arm means.

Usage, from the repo root:

    # what do we have?
    python slurm/parse_eval_energies.py --experiment E9 E11

    # write x_eval_* and tail-averaged x_final_* fields into wandb
    python slurm/parse_eval_energies.py --experiment E9 E11 --backfill

Safe to re-run: the backfill is idempotent, recomputing the same values in place.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from glob import glob

import numpy as np

EVAL_RE = re.compile(
    r"Epoch\s+(\d+), Energy: (-?[\d.]+e[+-]\d+), "
    r"Variance: ([\d.]+e[+-]\d+), Accept ratio: ([\d.]+)"
)
# Training lines carry the noclip value in parentheses; always score on noclip
# (campaign log s6 -- the clipped energy is biased and the bias depends on nchains).
TRAIN_RE = re.compile(
    r"Epoch\s+(\d+), Energy: -?[\d.]+e[+-]\d+ \((-?[\d.]+e[+-]\d+)\), "
    r"Variance: [\d.]+e[+-]\d+ \(([\d.]+e[+-]\d+)\)"
)
EVAL_MARKER = "Completed VMC! Evaluating"
NO_GPU = "No visible GPU devices"
OOM = "RESOURCE_EXHAUSTED"

# Reference energies (Ha). Carbon: Chakravorty 1993. N/O: preset_configs/
# reference_energies.json. H4 has no literature benchmark -- compare arms only.
REFERENCES = {"N": -54.5892, "O": -75.0673, "carbon": -37.8450,
              "N2_eq": -109.5423, "CO": -113.3255}

E9_ARMS = ["spring_mu0.95", "spring_mu0.99", "spring_mu0.995", "prime_sr", "ssu_defaults"]
E9_SYSTEMS = ["N", "O"]
E11_ARMS = ["spring_mu0.99", "spring_mu0.995", "prime_sr", "ssu_defaults"]
E11_ETAS = ["0.005", "0.05"]
E10_ARMS = ["spring_mu0.9", "spring_mu0.95", "prime_sr", "ssu_defaults",
            "spring_mu0.99", "spring_mu0.995"]
E10_SYSTEMS = ["N2_eq", "CO"]

METHOD = {
    "spring_mu0.9": "SPRING",
    "spring_mu0.95": "SPRING",
    "spring_mu0.99": "SPRING",
    "spring_mu0.995": "SPRING",
    "spring_mu0.999": "SPRING",
    "prime_sr": "PRIME-SR",
    "ssu_defaults": "SS-SPRING",
}
MU_NOMINAL = {"spring_mu0.9": 0.9, "spring_mu0.95": 0.95, "spring_mu0.99": 0.99,
              "spring_mu0.995": 0.995, "spring_mu0.999": 0.999}

# Tail-averaged training metrics, for the summary panels. Keys that a given arm does
# not log (probe_* on the non-SS-SPRING arms) are skipped silently.
TAIL_METRICS = ["energy_noclip", "variance_noclip", "mu", "norm_cap_applied",
                "update_sq_norm_preclip", "probe_r_ip", "probe_res_norm", "accept_ratio"]
TAIL_FRAC = 0.12


def _e9_cell(idx):
    """Array index -> cell, mirroring e9_atoms_headtohead.sbatch exactly."""
    arm, system, seed = E9_ARMS[idx // 10], E9_SYSTEMS[(idx % 10) // 5], idx % 5
    return f"e9_{system}_{arm}_s{seed}", dict(
        x_experiment="E9", x_system=system, x_arm=arm, x_seed=seed, x_eta=0.02)


def _e11_cell(idx):
    """Array index -> cell, mirroring e11_eta_robustness_carbon.sbatch exactly."""
    arm, eta, seed = E11_ARMS[idx // 6], E11_ETAS[(idx % 6) // 3], idx % 3
    return f"e11_carbon_{arm}_eta{eta}_s{seed}", dict(
        x_experiment="E11", x_system="carbon", x_arm=arm, x_seed=seed, x_eta=float(eta))


def _e10_cell(idx):
    """Array index -> cell, mirroring e10_molecules_hometurf.sbatch exactly."""
    arm, system, seed = E10_ARMS[idx // 6], E10_SYSTEMS[(idx % 6) // 3], idx % 3
    return f"e10_{system}_{arm}_s{seed}", dict(
        x_experiment="E10", x_system=system, x_arm=arm, x_seed=seed, x_eta=0.002)


EXPERIMENTS = {
    "E9": dict(pattern="slurm-e9-atoms-*_{idx}.out", ntasks=50, cell=_e9_cell,
               nepochs=50000),
    "E11": dict(pattern="slurm-e11-eta-robust-*_{idx}.out", ntasks=24, cell=_e11_cell,
                nepochs=50000),
    "E10": dict(pattern="slurm-e10-hometurf-*_{idx}.out", ntasks=36, cell=_e10_cell,
                nepochs=50000),
}


def blocked_error(values, nblocks=50):
    """Standard error of the mean via blocking, to absorb autocorrelation."""
    values = np.asarray(values)
    usable = len(values) // nblocks * nblocks
    if usable == 0:
        return float("nan")
    means = values[:usable].reshape(nblocks, -1).mean(axis=1)
    return float(means.std(ddof=1) / np.sqrt(nblocks))


def parse_out_file(path):
    """Statistics from one .out file.

    Returns a dict describing how far the run got. `eval_energy` is present only when
    the eval phase actually produced epochs -- that is the publishable number. When
    training finishes but eval dies (e.g. the eval walker count OOMs on a bigger
    system), the training tail is reported instead and flagged, because a training tail
    is NOT interchangeable with an eval energy (campaign log s6).
    """
    with open(path, errors="replace") as handle:
        text = handle.read()

    train_epochs, train_e, train_v = [], [], []
    for match in TRAIN_RE.finditer(text):
        train_epochs.append(int(match.group(1)))
        train_e.append(float(match.group(2)))
        train_v.append(float(match.group(3)))

    out = dict(source=os.path.basename(path))
    if train_epochs:
        out["last_train_epoch"] = max(train_epochs)
        tail = max(1, int(len(train_e) * TAIL_FRAC))
        out["train_tail_energy"] = float(np.mean(train_e[-tail:]))
        out["train_tail_variance"] = float(np.mean(train_v[-tail:]))

    if EVAL_MARKER not in text:
        # No training lines and no eval marker -> this file has nothing for us. An
        # eval-only recovery run (vmc.nepochs=0 off a checkpoint) legitimately has no
        # training lines, so absence of training alone must not disqualify a file.
        if not train_epochs:
            return None
        out["reached_eval"] = False
        return out
    out["reached_eval"] = True

    energies, variances, accepts = [], [], []
    for line in text.split(EVAL_MARKER, 1)[1].splitlines():
        if "(" in line:  # a training line; eval lines have no noclip parenthetical
            continue
        match = EVAL_RE.search(line)
        if match:
            energies.append(float(match.group(2)))
            variances.append(float(match.group(3)))
            accepts.append(float(match.group(4)))
    if not energies:
        out["eval_failed"] = "OOM" if OOM in text else "NO_EVAL_EPOCHS"
        return out

    energies = np.array(energies)
    out.update(
        n_eval=len(energies),
        eval_energy=float(energies.mean()),
        eval_energy_err=blocked_error(energies),
        eval_variance=float(np.mean(variances)),
        eval_accept=float(np.mean(accepts)),
    )
    return out


def collect(experiment, repo):
    """One row per array index, preferring the .out that got furthest."""
    spec = EXPERIMENTS[experiment]
    nepochs = spec.get("nepochs")
    rows, duplicates = [], []
    for idx in range(spec["ntasks"]):
        name, meta = spec["cell"](idx)
        paths = sorted(glob(os.path.join(repo, spec["pattern"].format(idx=idx))))
        parsed = [(p, s) for p in paths for s in [parse_out_file(p)] if s]
        with_eval = [(p, s) for p, s in parsed if "eval_energy" in s]
        if len(with_eval) > 1:
            duplicates.append((name, [os.path.basename(p) for p, _ in with_eval]))

        if with_eval:
            path, stats = max(with_eval, key=lambda ps: os.path.getmtime(ps[0]))
            # An eval-only recovery run carries no training lines. Merge the training
            # fields back in from whichever file actually did the training, so the
            # tail-vs-eval comparison stays available after a salvage.
            if "train_tail_energy" not in stats:
                trained = [s for _, s in parsed if "train_tail_energy" in s]
                if trained:
                    best = max(trained, key=lambda s: s["last_train_epoch"])
                    stats = {**{k: v for k, v in best.items() if k != "source"}, **stats}
            rows.append(dict(name=name, idx=idx, status="OK", **meta, **stats))
            continue
        if parsed:
            # Training data exists but no eval energy. Distinguish "training finished,
            # eval blew up" (salvageable from a checkpoint) from "training died".
            path, stats = max(parsed, key=lambda ps: ps[1]["last_train_epoch"])
            done = nepochs is None or stats["last_train_epoch"] >= nepochs - 10
            if stats.get("eval_failed") == "OOM":
                status = "EVAL_OOM" if done else "EVAL_OOM_TRAIN_SHORT"
            elif not done:
                status = "TRAIN_INCOMPLETE"
            else:
                status = "NO_EVAL"
            rows.append(dict(name=name, idx=idx, status=status, **meta, **stats))
            continue

        if not paths:
            why = "NO_OUT_FILE"
        elif any(NO_GPU in open(p, errors="replace").read(4000) for p in paths):
            why = "CUDA_NO_DEVICE"
        else:
            why = "NO_EVAL"
        rows.append(dict(name=name, idx=idx, status=why, **meta))
    return rows, duplicates


def compress(indices):
    """[7,10,16,17] -> '7,10,16-17', ready to paste after sbatch --array=."""
    if not indices:
        return ""
    parts, start, prev = [], indices[0], indices[0]
    for i in indices[1:] + [None]:
        if i != prev + 1:
            parts.append(str(start) if start == prev else f"{start}-{prev}")
            start = i
        prev = i
    return ",".join(parts)


SCRIPTS = {"E9": "e9_atoms_headtohead", "E10": "e10_molecules_hometurf",
           "E11": "e11_eta_robustness_carbon"}


def report(experiment, rows, duplicates):
    ok = [r for r in rows if r["status"] == "OK"]
    counts = Counter(r["status"] for r in rows)
    print(f"=== {experiment}: {len(ok)}/{len(rows)} runs with an EVAL energy")
    if set(counts) - {"OK"}:
        print(f"    statuses: {dict(counts)}")

    if duplicates:
        print("  !! DUPLICATE COMPLETIONS -- the same cell ran more than once. wandb "
              "allows duplicate run names, so these would double-count in arm means:")
        for name, files in duplicates:
            print(f"     {name}: {files}")

    # Fall back to training tails only when no eval energy exists anywhere, and say so
    # loudly -- a training tail is not comparable to an eval energy.
    using_tail = not ok and any("train_tail_energy" in r for r in rows)
    scored = [r for r in rows if ("train_tail_energy" in r if using_tail else r in ok)]
    if using_tail:
        scored = [r for r in scored
                  if r["status"] in ("EVAL_OOM", "NO_EVAL")]  # exclude short training
        print("  !! NO EVAL ENERGIES EXIST. Falling back to the TRAINING TAIL (last "
              f"{TAIL_FRAC:.0%} of noclip training energy). These are NOT comparable to "
              "the eval numbers quoted for other experiments -- see campaign log s6.")

    key_field = "train_tail_energy" if using_tail else "eval_energy"
    by_arm = defaultdict(list)
    for r in scored:
        key = ((r["x_system"], r["x_arm"]) if experiment in ("E9", "E10")
               else (r["x_eta"], r["x_arm"]))
        by_arm[key].append(r)
    if by_arm:
        label = "mean mHa (TAIL)" if using_tail else "mean mHa"
        print(f"  {'cell':<30} {'n':>2}  {'seeds':<12} {label:>15} {'sem':>7} {'worst':>8}")
        for key in sorted(by_arm):
            rs = sorted(by_arm[key], key=lambda r: r["x_seed"])
            ref = REFERENCES[rs[0]["x_system"]]
            d = np.array([(r[key_field] - ref) * 1000 for r in rs])
            sem = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
            print(f"  {str(key):<30} {len(d):>2}  {str([r['x_seed'] for r in rs]):<12} "
                  f"{d.mean():>15.3f} {sem:>7.3f} {d.max():>8.3f}")

    oom = sorted(r["idx"] for r in rows if r["status"] == "EVAL_OOM")
    if oom:
        print(f"\n  {len(oom)} run(s) FINISHED TRAINING and then ran out of memory "
              f"entering eval. Do NOT retrain these -- the 50k training is done and "
              f"checkpointed. Re-run the eval phase alone from the final regular "
              f"checkpoint (`checkpoints/50000.npz`, NOT `best_checkpoint.npz`, which "
              f"is selected on best running energy and would bias each arm differently) "
              f"with a smaller `--config.eval.nchains`.")
    resubmit = sorted(r["idx"] for r in rows
                      if r["status"] in ("NO_OUT_FILE", "CUDA_NO_DEVICE",
                                         "TRAIN_INCOMPLETE", "EVAL_OOM_TRAIN_SHORT"))
    if resubmit:
        print(f"\n  {len(resubmit)} cell(s) need a real rerun (no usable training). "
              f"Resubmit exactly these:")
        print(f"    sbatch --array={compress(resubmit)} "
              f"slurm/{SCRIPTS[experiment]}.sbatch")
    if not oom and not resubmit:
        print("\n  Complete -- every cell has an eval energy.")
    print()


def backfill(rows, entity, project):
    """Write x_* config and summary fields onto the matching wandb runs."""
    import wandb

    parsed = {r["name"]: r for r in rows}
    api = wandb.Api(timeout=120)
    runs = [r for r in api.runs(f"{entity}/{project}") if r.name in parsed]
    print(f"backfilling {len(runs)} wandb run(s)")

    seen = defaultdict(int)
    for i, run in enumerate(runs):
        seen[run.name] += 1
        row = parsed[run.name]
        config = dict(x_experiment=row["x_experiment"], x_system=row["x_system"],
                      x_arm=row["x_arm"], x_seed=row["x_seed"], x_eta=row["x_eta"],
                      x_method=METHOD[row["x_arm"]],
                      x_adaptive=row["x_arm"] in ("prime_sr", "ssu_defaults"))
        if row["x_arm"] in MU_NOMINAL:
            config["x_mu_nominal"] = MU_NOMINAL[row["x_arm"]]

        summary = {}
        ref = REFERENCES[row["x_system"]]
        if row["status"] == "OK":
            summary["x_eval_energy"] = row["eval_energy"]
            summary["x_eval_energy_err"] = row["eval_energy_err"]
            summary["x_eval_variance"] = row["eval_variance"]
            summary["x_eval_mHa"] = (row["eval_energy"] - ref) * 1000.0
            summary["x_eval_nepochs"] = row["n_eval"]
        # Deliberately a DIFFERENT field name from x_eval_mHa. A training tail is not
        # an eval energy and must never silently land in the same panel as one.
        if "train_tail_energy" in row:
            summary["x_train_tail_mHa"] = (row["train_tail_energy"] - ref) * 1000.0
            summary["x_train_tail_variance"] = row["train_tail_variance"]
            summary["x_last_train_epoch"] = row["last_train_epoch"]
        summary["x_status"] = row["status"]

        keys = [k for k in TAIL_METRICS if k in run.summary]
        if keys:
            history = run.history(keys=keys, samples=4000, pandas=False)
            for key in keys:
                vals = np.array([h.get(key) for h in history
                                 if isinstance(h.get(key), (int, float))], float)
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    n = max(1, int(vals.size * TAIL_FRAC))
                    summary[f"x_final_{key}"] = float(vals[-n:].mean())
        summary["x_last_step"] = int(run.summary.get("_step", 0))

        run.config.update(config)
        run.update()
        for key, value in summary.items():
            run.summary[key] = value
        run.summary.update()
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(runs)}", flush=True)

    dupes = {n: c for n, c in seen.items() if c > 1}
    if dupes:
        print(f"\n!! {len(dupes)} run name(s) appear more than once in wandb -- a "
              f"resubmitted cell was synced alongside its original. Delete the stale "
              f"run before reporting, or arm means double-count: {sorted(dupes)}")
    missing_in_wandb = sorted(
        r["name"] for r in rows
        if r["status"] == "OK" and r["name"] not in {x.name for x in runs})
    if missing_in_wandb:
        print(f"\n!! {len(missing_in_wandb)} completed run(s) have a .out file but no "
              f"wandb run -- sync them from a login node with `bash slurm/sync_wandb.sh` "
              f"(unfiltered): {missing_in_wandb}")
    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", nargs="+", default=["E9", "E11"],
                        choices=sorted(EXPERIMENTS))
    parser.add_argument("--repo", default=".", help="directory holding the .out files")
    parser.add_argument("--json", default=None, help="write parsed rows here")
    parser.add_argument("--backfill", action="store_true",
                        help="write x_* fields into wandb (idempotent)")
    parser.add_argument("--entity", default="ren27-university-of-california-berkeley")
    parser.add_argument("--project", default="vmcnet-phase-c")
    args = parser.parse_args()

    all_rows = []
    for experiment in args.experiment:
        rows, duplicates = collect(experiment, args.repo)
        report(experiment, rows, duplicates)
        all_rows += rows

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(all_rows, handle, indent=1)
        print(f"wrote {args.json}")

    if args.backfill:
        backfill([r for r in all_rows if "train_tail_energy" in r or r["status"] == "OK"],
                 args.entity, args.project)


if __name__ == "__main__":
    sys.exit(main())
