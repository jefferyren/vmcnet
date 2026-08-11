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
from collections import defaultdict
from glob import glob

import numpy as np

EVAL_RE = re.compile(
    r"Epoch\s+(\d+), Energy: (-?[\d.]+e[+-]\d+), "
    r"Variance: ([\d.]+e[+-]\d+), Accept ratio: ([\d.]+)"
)
EVAL_MARKER = "Completed VMC! Evaluating"
NO_GPU = "No visible GPU devices"

# Reference energies (Ha). Carbon: Chakravorty 1993. N/O: preset_configs/
# reference_energies.json. H4 has no literature benchmark -- compare arms only.
REFERENCES = {"N": -54.5892, "O": -75.0673, "carbon": -37.8450}

E9_ARMS = ["spring_mu0.95", "spring_mu0.99", "spring_mu0.995", "prime_sr", "ssu_defaults"]
E9_SYSTEMS = ["N", "O"]
E11_ARMS = ["spring_mu0.99", "spring_mu0.995", "prime_sr", "ssu_defaults"]
E11_ETAS = ["0.005", "0.05"]

METHOD = {
    "spring_mu0.95": "SPRING",
    "spring_mu0.99": "SPRING",
    "spring_mu0.995": "SPRING",
    "spring_mu0.999": "SPRING",
    "prime_sr": "PRIME-SR",
    "ssu_defaults": "SS-SPRING",
}
MU_NOMINAL = {"spring_mu0.95": 0.95, "spring_mu0.99": 0.99,
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


EXPERIMENTS = {
    "E9": dict(pattern="slurm-e9-atoms-*_{idx}.out", ntasks=50, cell=_e9_cell),
    "E11": dict(pattern="slurm-e11-eta-robust-*_{idx}.out", ntasks=24, cell=_e11_cell),
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
    """Eval-phase statistics from one .out file, or None if it never reached eval."""
    with open(path, errors="replace") as handle:
        text = handle.read()
    if EVAL_MARKER not in text:
        return None
    energies, variances, accepts = [], [], []
    for line in text.split(EVAL_MARKER, 1)[1].splitlines():
        if "(" in line:  # a training line that followed a restart; never an eval line
            continue
        match = EVAL_RE.search(line)
        if match:
            energies.append(float(match.group(2)))
            variances.append(float(match.group(3)))
            accepts.append(float(match.group(4)))
    if not energies:
        return None
    energies = np.array(energies)
    return dict(
        n_eval=len(energies),
        eval_energy=float(energies.mean()),
        eval_energy_err=blocked_error(energies),
        eval_variance=float(np.mean(variances)),
        eval_accept=float(np.mean(accepts)),
        source=os.path.basename(path),
    )


def collect(experiment, repo):
    """One row per array index, choosing the .out that reached eval."""
    spec = EXPERIMENTS[experiment]
    rows, duplicates = [], []
    for idx in range(spec["ntasks"]):
        name, meta = spec["cell"](idx)
        paths = sorted(glob(os.path.join(repo, spec["pattern"].format(idx=idx))))
        completed = [(p, s) for p in paths for s in [parse_out_file(p)] if s]
        if len(completed) > 1:
            duplicates.append((name, [os.path.basename(p) for p, _ in completed]))
        if completed:
            # Newest wins; a rerun of an already-good cell is reported above.
            path, stats = max(completed, key=lambda ps: os.path.getmtime(ps[0]))
            rows.append(dict(name=name, idx=idx, status="OK", **meta, **stats))
        else:
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


def report(experiment, rows, duplicates):
    ok = [r for r in rows if r["status"] == "OK"]
    missing = sorted(r["idx"] for r in rows if r["status"] != "OK")
    print(f"=== {experiment}: {len(ok)}/{len(rows)} runs complete")

    if duplicates:
        print("  !! DUPLICATE COMPLETIONS -- the same cell ran more than once. wandb "
              "allows duplicate run names, so these would double-count in arm means:")
        for name, files in duplicates:
            print(f"     {name}: {files}")

    by_arm = defaultdict(list)
    for r in ok:
        key = (r["x_system"], r["x_arm"]) if experiment == "E9" else (r["x_eta"], r["x_arm"])
        by_arm[key].append(r)
    print(f"  {'cell':<28} {'n':>2}  {'seeds':<12} {'mean mHa':>9} {'sem':>7} {'worst':>8}")
    for key in sorted(by_arm):
        rs = sorted(by_arm[key], key=lambda r: r["x_seed"])
        ref = REFERENCES[rs[0]["x_system"]]
        d = np.array([(r["eval_energy"] - ref) * 1000 for r in rs])
        sem = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        seeds = [r["x_seed"] for r in rs]
        print(f"  {str(key):<28} {len(d):>2}  {str(seeds):<12} "
              f"{d.mean():>9.3f} {sem:>7.3f} {d.max():>8.3f}")

    if missing:
        script = ("e9_atoms_headtohead" if experiment == "E9"
                  else "e11_eta_robustness_carbon")
        print(f"\n  {len(missing)} cell(s) still missing. Resubmit exactly these:")
        print(f"    sbatch --array={compress(missing)} slurm/{script}.sbatch")
    else:
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
        if row["status"] == "OK":
            ref = REFERENCES[row["x_system"]]
            summary["x_eval_energy"] = row["eval_energy"]
            summary["x_eval_energy_err"] = row["eval_energy_err"]
            summary["x_eval_variance"] = row["eval_variance"]
            summary["x_eval_mHa"] = (row["eval_energy"] - ref) * 1000.0
            summary["x_eval_nepochs"] = row["n_eval"]

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
        backfill([r for r in all_rows if r["status"] == "OK"], args.entity, args.project)


if __name__ == "__main__":
    sys.exit(main())
