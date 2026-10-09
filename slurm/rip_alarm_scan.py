"""Step 0 of the momentum-feedback route: the probe's r_ip on every SS-SPRING run.

WHY. Phase F's F4 route makes the beta controller respond to growth or noise instead of
clipping r_ip to 1 (CAMPAIGN_LOG Phase F, F4d RESULT). Before any controller change, two
numbers are needed from runs that already exist, at zero GPU cost:
  1. False alarms: on the systems where SS-SPRING is stable (carbon, N, O, H4, N2-eq,
     CO), how often does the raw window ratio r_ip rise above 1, and would any
     threshold rule fire? A rule that fires there changes the method on stable systems.
  2. Lead: on the N2-4.0 runs that failed, how many epochs before the first
     catastrophe does each rule fire?
probe_r_ip and mu (beta) have been logged per epoch since Phase A, so every original
run can answer both. Caveat: these runs predate F4d (gram_center_first), which changes
r_ip near the N2 events; calibrate the F4d replays with
`f2_replay_summary.py --rip --logroot .../phase_f/f4d_center_first`.

For each SS-SPRING logdir matching --logdirs it prints one line:
  rows, first catastrophe (same detector as f2_replay_summary.py), median beta over the
  healthy rows past --beta-from, the r_ip distribution over the healthy rows, and for
  each rule in RIP_RULES "healthy firings/lead" (lead = epochs from the first firing to
  the catastrophe; "-" when the run never fails).
then a per-group total (group = run name without its experiment prefix and seed).
"Healthy" = from --start to 300 epochs before the first catastrophe, or to the end.

Usage (Savio login node; no jax needed):
    python slurm/rip_alarm_scan.py --logdirs \\
        "/global/scratch/users/$USER/vmcnet_logs/phase_[cde]/*/*"
    python slurm/rip_alarm_scan.py --logdirs \\
        "/global/scratch/users/$USER/vmcnet_logs/phase_e/e1[4-9]*/*N2_4.0*"
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from f2_replay_summary import (  # noqa: E402
    RIP_RULES,
    first_event,
    load,
    rip_rules,
    rip_summary,
)

SSU = "same_sampled_spring_unified"


def group_of(name):
    """'d10_N2_eq_ssu_defaults_s0' -> 'N2_eq_ssu_defaults'."""
    return re.sub(r"_s\d+$", "", re.sub(r"^[a-z]\d+[a-z]?_", "", name))


def is_ssu(d):
    path = os.path.join(d, "config.json")
    if not os.path.exists(path):
        return False
    return json.load(open(path))["vmc"]["optimizer_type"] == SSU


def fmt_rule(n_healthy, lead):
    return f"{n_healthy}/{'-' if lead is None else lead}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--logdirs", required=True, help="logdir glob (quote it)")
    ap.add_argument(
        "--start",
        type=int,
        default=1000,
        help="first epoch counted (skips the init transient, as safeguard_start_step)",
    )
    ap.add_argument(
        "--beta-from",
        type=int,
        default=10000,
        help="median beta is taken over healthy epochs from here on",
    )
    args = ap.parse_args()

    rule_hdr = "".join(f"{f'>{t} {m}/{w}':>11}" for t, w, m in RIP_RULES)
    print(
        f"{'run':<44}{'rows':>7}{'event':>7}{'beta':>8}{'med':>6}{'p99.9':>7}"
        f"{'max':>9}{'%>1':>6}{'run>1':>6}" + rule_hdr
    )
    print(
        "  rule columns: 'healthy firings/lead'; rule '>T M/W' = r_ip > T on >= M of "
        "the last W rows"
    )
    groups = defaultdict(lambda: dict(runs=0, failed=0, rows=0, fires=None, betas=[]))
    n_seen = 0
    for d in sorted(glob.glob(args.logdirs)):
        if not os.path.isdir(d) or not is_ssu(d):
            continue
        r = load(d, "probe_r_ip")
        e, v = load(d, "energy_noclip"), load(d, "variance_noclip")
        if r is None or e is None or v is None:
            continue
        n_seen += 1
        name = os.path.basename(os.path.normpath(d))
        n = min(len(r), len(e), len(v))
        r, e, v = r[:n], e[:n], v[:n]
        last = n - 1
        ev = first_event(e, v, args.start)
        hi = (ev - 300) if ev is not None else last + 1
        st = rip_summary(r, args.start, hi)
        rules = rip_rules(r, args.start, hi, ev)
        mu = load(d, "mu")
        beta = np.nan
        if mu is not None and hi > args.beta_from:
            beta = np.nanmedian(mu[args.beta_from : min(hi, len(mu))])
        dist = (
            f"{st['median']:>6.2f}{st['p999']:>7.2f}{st['max']:>9.3g}"
            f"{100 * st['frac_above_1']:>6.1f}{st['longest_above_1']:>6}"
            if st is not None
            else f"{'(no healthy rows)':>34}"
        )
        print(
            f"{name[:43]:<44}{n:>7}{'-' if ev is None else ev:>7}{beta:>8.4f}"
            + dist
            + "".join(f"{fmt_rule(h, lead):>11}" for h, _, lead in rules)
        )
        g = groups[group_of(name)]
        g["runs"] += 1
        g["failed"] += ev is not None
        g["rows"] += max(0, hi - args.start)
        fires = np.array([h for h, _, _ in rules])
        g["fires"] = fires if g["fires"] is None else g["fires"] + fires
        if np.isfinite(beta):
            g["betas"].append(beta)

    if not n_seen:
        print("no SS-SPRING logdirs with probe_r_ip.txt matched", args.logdirs)
        return
    print(
        "\nper group: healthy firings summed over runs (a firing on a never-failing "
        "group is a false alarm)"
    )
    print(
        f"{'group':<44}{'runs':>5}{'fail':>5}{'healthy rows':>13}{'beta range':>16}"
        + "".join(f"{f'>{t} {m}/{w}':>11}" for t, w, m in RIP_RULES)
    )
    for name, g in sorted(groups.items()):
        b = g["betas"]
        br = f"{min(b):.4f}-{max(b):.4f}" if b else "-"
        print(
            f"{name[:43]:<44}{g['runs']:>5}{g['failed']:>5}{g['rows']:>13}{br:>16}"
            + "".join(f"{x:>11}" for x in g["fires"])
        )


if __name__ == "__main__":
    main()
