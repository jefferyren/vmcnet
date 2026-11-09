"""H1: put the frozen-theta probe results into wandb next to the replay they came from.

Two things, both in project vmcnet-phase-h:
  1. Tag the dense-replay runs (synced from offline first) with queryable config:
     x_experiment=H1, x_kind=replay, x_run=e14s0 / e15s7. vmcnet logs its config as a
     ConfigDict blob, so without these a report runset cannot select the runs.
  2. Log one wandb run per frozen snapshot (.npz from h1_frozen_probe.py), named
     h1_frozen_<run>_<snapshot>_e<epoch>, holding
       - a freeze marker: two rows at step = epoch and epoch + 1 whose freeze/* values
         span the replay's own energy, variance, mu and probe_r_ip ranges, so on a
         panel against Step each snapshot draws a vertical line on the replay's curve;
       - the probe curves, at step = epoch + 2 + t with t logged as frozen_step (plot
         them against frozen_step): per arm the held-out and training-block residuals
         relative to step 0 and r_ip, geometric means over the --nxstar targets; the
         adaptive arms' beta; the Gram spectrum and MCMC acceptance;
       - summary x_g_* / x_rip_* / x_drop_* (h1_frozen_probe.arm_stats, 2nd half) and
         the run's own probe over the same epochs (x_inrun_rip, x_inrun_mu).
     The marker ranges are stored as config x_marker_<metric>_lo / _hi; the report
     (slurm/report_h1_frozen.py) uses them as the y-ranges.

Run on a Savio login node after syncing the replays, from the vmcnet repo root:
    grep -h "^wandb: wandb sync" slurm-h1-n2-dense-replay-*.out | awk '{print $4}' \
        | xargs -n1 wandb sync
    python slurm/h1_log_to_wandb.py
Re-running skips snapshots already logged; --replace deletes and re-logs them.
"""

import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from h1_frozen_probe import DEFAULT_OUT, arm_stats, in_run, load_txt  # noqa: E402

ENTITY = "ren27-university-of-california-berkeley"
PROJECT = "vmcnet-phase-h"
REPLAY_NAME = re.compile(r"^h1_N2_4\.0_(?P<tag>\w+?)_dense$")
# replay metric -> (marker key, log-scale panel?)
MARKERS = {
    "energy_noclip": ("energy", False),
    "variance_noclip": ("variance", True),
    "mu": ("mu", False),
    "probe_r_ip": ("r_ip", True),
}


def arm_key(label):
    """'beta=0.99' -> 'b0990', 'beta=run' -> 'brun', 'adaptive' -> 'adaptive'."""
    if label.startswith("beta="):
        v = label[len("beta=") :]
        return "brun" if v == "run" else "b" + f"{float(v):.3f}".replace(".", "")
    return label


def run_tag(logdir):
    m = REPLAY_NAME.match(os.path.basename(str(logdir).rstrip("/")))
    return m.group("tag") if m else os.path.basename(str(logdir).rstrip("/"))


def marker_range(logdir, name, log_scale, first_epoch):
    """(lo, hi) spanning the replay's trace of `name` from `first_epoch` on (the .txt
    files are indexed by epoch and include the original run before the reload)."""
    x = load_txt(logdir, name)
    if x is None:
        return None
    x = x[first_epoch:]
    x = x[np.isfinite(x)]
    if log_scale:
        x = x[x > 0]
    if not len(x):
        return None
    lo, hi = np.percentile(x, [0.5, 99.5])
    if log_scale:
        return float(lo / 2), float(hi * 2)
    pad = 0.05 * (hi - lo) if hi > lo else 1e-3
    return float(lo - pad), float(hi + pad)


def gmean(a, axis=1):
    """Geometric mean over the targets; nan where any target is non-finite or <= 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.exp(np.mean(np.log(a), axis=axis))


def snapshot_rows(d):
    """Per-step probe rows (dicts) and the per-arm summary of one frozen snapshot."""
    labels = [str(x) for x in d["arm_labels"]]
    groups = {lab: [i for i, x in enumerate(labels) if x == lab] for lab in labels}
    p = int(d["p"])
    nsteps = d["res"].shape[0]
    series, summary = {}, {}
    for lab, idx in groups.items():
        k = arm_key(lab)
        for name in ("res_hold", "res"):
            a = d[name][:, idx]
            series[f"probe/rel_{name}_{k}"] = gmean(a / a[0])
        rip = gmean(d["r_ip"][:, idx])
        rip[: 2 * p] = np.nan  # gated to 1 by the controller before 2p steps
        series[f"probe/r_ip_{k}"] = rip
        if lab == "adaptive":
            series[f"probe/beta_{k}"] = np.mean(d["beta"][:, idx], axis=1)
        stats = np.array([arm_stats(d, i, p) for i in idx], dtype=float)
        summary[f"x_rip_{k}"] = float(np.exp(np.nanmean(np.log(stats[:, 0]))))
        summary[f"x_g_{k}"] = float(np.exp(np.nanmean(np.log(stats[:, 1]))))
        summary[f"x_drop_{k}"] = float(np.nanmean(stats[:, 2]))
        summary[f"x_beta_{k}"] = float(np.nanmean(stats[:, 3]))
    series["frozen/gram_min"] = d["gram_min"]
    series["frozen/gram_max"] = d["gram_max"]
    series["frozen/gram_n_below_damping"] = d["gram_n_below_damping"]
    series["frozen/accept_ratio"] = d["accept"]

    rows = []
    for t in range(nsteps):
        row = {k: float(v[t]) for k, v in series.items() if np.isfinite(v[t])}
        if not any(k.startswith("probe/rel_res_hold") for k in row):
            continue  # all probes non-finite: the run stopped early
        row["frozen_step"] = t
        rows.append((t, row))
    return rows, summary


def tag_replays(api, dry_run):
    try:
        runs = list(api.runs(f"{ENTITY}/{PROJECT}"))
    except Exception as err:  # project missing until the first sync
        print(f"WARNING: cannot list {PROJECT} ({err}); sync the replays first")
        return
    found = False
    for r in runs:
        m = REPLAY_NAME.match(r.name or "")
        if not m:
            continue
        found = True
        tag = m.group("tag")
        new = {
            "x_experiment": "H1",
            "x_kind": "replay",
            "x_run": tag,
            "x_snapshot": f"{tag} replay (the VMC run)",
        }
        if all(r.config.get(k) == v for k, v in new.items()):
            print(f"replay {r.name}: already tagged")
            continue
        print(f"replay {r.name}: tagging {new}")
        if not dry_run:
            r.config.update(new)
            r.update()
    if not found:
        print(f"WARNING: no h1_N2_4.0_*_dense run in {PROJECT}; sync the replays first")


def existing_runs(api):
    try:
        return {r.name: r for r in api.runs(f"{ENTITY}/{PROJECT}")}
    except Exception:
        return {}


def log_snapshot(path, existing, args):
    import wandb

    d = np.load(path, allow_pickle=False)
    logdir, label, epoch = str(d["logdir"]), str(d["label"]), int(d["epoch"])
    event, tag = int(d["event"]), run_tag(d["logdir"])
    name = f"h1_frozen_{tag}_{label}_e{epoch}"
    if name in existing:
        if not args.replace:
            print(f"{name}: already logged, skipping (--replace to re-log)")
            return
        print(f"{name}: deleting the existing run")
        if not args.dry_run:
            existing[name].delete()

    rows, summary = snapshot_rows(d)
    nsteps = d["res"].shape[0]
    rip, mu, n_in = in_run(d, nsteps)
    summary.update(
        x_inrun_rip=rip, x_inrun_mu=mu, x_inrun_n=n_in, x_frozen_steps_finite=len(rows)
    )
    config = {
        "x_experiment": "H1",
        "x_kind": "frozen",
        "x_run": tag,
        "x_snapshot": label,
        "x_epoch": epoch,
        "x_event": event,
        "x_before_event": event - epoch if event >= 0 else None,
        "x_run_beta": float(d["run_beta"]),
        "x_arms": sorted(set(map(str, d["arm_labels"]))),
        "x_nxstar": int(np.sum(d["arm_labels"] == d["arm_labels"][0])),
        "x_nsteps": nsteps,
        "x_p": int(d["p"]),
        "x_probe_lr": float(d["probe_lr"]),
        "x_probe_damping": float(d["probe_damping"]),
        "x_source": path,
        "x_replay_logdir": logdir,
    }
    lo_rows, hi_rows = {}, {}
    for metric, (key, log_scale) in MARKERS.items():
        rng = marker_range(logdir, metric, log_scale, args.marker_from)
        if rng is None:
            print(f"  (no {metric}.txt in {logdir}: no {key} marker)")
            continue
        config[f"x_marker_{key}_lo"], config[f"x_marker_{key}_hi"] = rng
        lo_rows[f"freeze/{key}"], hi_rows[f"freeze/{key}"] = rng

    print(
        f"{name}: {len(rows)} finite frozen steps of {nsteps}; "
        + ", ".join(f"{k} {v:.4g}" for k, v in summary.items() if k.startswith("x_g_"))
    )
    if args.dry_run:
        print(f"  config {config}\n  first row {rows[0][1] if rows else None}")
        return
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=name,
        group="h1-frozen",
        job_type="frozen-probe",
        config=config,
        reinit=True,
        tags=["h1", "phase-h", "frozen-probe", tag, label],
        notes=f"H1 frozen-theta Kaczmarz++ probe at epoch {epoch} of {tag} "
        f"(first event {event}). Source {path}.",
    )
    wandb.define_metric("frozen_step")
    for k in rows[0][1] if rows else []:
        if k != "frozen_step":
            wandb.define_metric(k, step_metric="frozen_step")
    if lo_rows:
        run.log(lo_rows, step=epoch)
        run.log(hi_rows, step=epoch + 1)
    for t, row in rows:
        run.log(row, step=epoch + 2 + t)
    run.summary.update(summary)
    run.finish()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--frozen", default=DEFAULT_OUT, help="h1_frozen_probe.py --out dir"
    )
    ap.add_argument("--replace", action="store_true", help="re-log logged snapshots")
    ap.add_argument(
        "--marker-from",
        type=int,
        default=9999,
        help="marker ranges span the replay's metrics from this epoch on",
    )
    ap.add_argument("--dry-run", action="store_true", help="print, write nothing")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.frozen, "*.npz")))
    if not paths:
        raise SystemExit(f"no .npz in {args.frozen}")
    if args.dry_run:
        api, existing = None, {}
    else:
        import wandb

        api = wandb.Api(timeout=60)
        tag_replays(api, args.dry_run)
        existing = existing_runs(api)
    for path in paths:
        log_snapshot(path, existing, args)


if __name__ == "__main__":
    main()
