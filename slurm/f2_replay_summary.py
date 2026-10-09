"""Read out Phase F checkpoint replays (f2_n2_replay / f3b_n2_carried_cap sbatch).

For each replay logdir under LOGROOT it prints:
  1. Reproduction check: max |E_replay - E_original| over epochs 10000-10500. If the
     safeguard-off arm does not track the original closely, the diagnostics changed the
     float32 rounding and the trajectory wandered -- read event timings, not epochs.
  2. Outcome: last epoch reached, first catastrophe (same detector as
     n2_failure_anatomy.py), and the mean noclip energy of the last 500 epochs, with the
     original run's level over 9500-10000 for comparison.
  3. Safeguard: number of triggers / rewinds and the rows at which they fired.
  Rows are line numbers in the replay logdir's .txt files, which equal epochs to
  within one (reload.append copies the history and the replay re-logs the checkpoint
  epoch); the reproduction check reports the shift.
  4. Onset anatomy (safeguard-off arms): every logged quantity, epoch by epoch, over the
     30 epochs before the first catastrophe -- which one moves first.

Also reports, for runs with carried_cap on, how often the cap was active, and for
safeguarded runs which rewinds the F3c carried trigger and the probe guard caused. The
original run is read from each replay's reload_config.json. `--sustained` calibrates
the F3c trigger (K hits in a trailing window) on any replays with diagnostics on, and
`--rip` does the same for the probe's raw window ratio r_ip. slurm/rip_alarm_scan.py
runs the r_ip calibration over original (non-replay) runs on every system.

Usage (Savio login node, after the array finishes):
    python slurm/f2_replay_summary.py                       # F2/F3 replays
    python slurm/f2_replay_summary.py --precursors --carried
    python slurm/f2_replay_summary.py \
        --logroot /global/scratch/users/$USER/vmcnet_logs/phase_f/f3b_carried_cap
    python slurm/f2_replay_summary.py --sustained --logroot ...   # F3c calibration
    python slurm/f2_replay_summary.py --rip --logroot ...         # r_ip alarm calib.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from n2_failure_anatomy import EVENT_DE, EVENT_VAR, trailing_median  # noqa: E402

USER = os.environ.get("USER", "")
DEFAULT_ROOT = f"/global/scratch/users/{USER}/vmcnet_logs/phase_f/f2_replay"
PHASE_E = f"/global/scratch/users/{USER}/vmcnet_logs/phase_e"
ORIGINALS = {
    "e14s1": f"{PHASE_E}/e14_n2_stretched_100k/e14_N2_4.0_ssu_defaults_s1",
    "e18s2": f"{PHASE_E}/e18_n2_stretched_eta0015/e18_N2_4.0_ssu_defaults_eta0.0015_s2",
}
ONSET_KEYS = [
    "energy_noclip",
    "mu",
    "variance_noclip",
    "update_sq_norm_preclip",
    "probe_r_ip",
    "diag_bound_ratio",
    "diag_equation_residual",
    "diag_carried_over_eps",
    "diag_gram_lam_max",
    "diag_gram_min_eig_preclip",
    "diag_walker_rownorm_max_over_median",
    "diag_mean_jac_sq_over_trace",
]


def load(logdir, key, pad=0):
    """Metric as a row-indexed array. diag_*/sg_* files exist only for the replay, so
    they are front-padded with `pad` NaNs to line up with the appended history."""
    path = os.path.join(logdir, key + ".txt")
    if not os.path.exists(path):
        return None
    x = np.atleast_1d(np.loadtxt(path))
    if key.startswith(("diag_", "sg_")):
        x = np.concatenate([np.full(pad, np.nan), x])
    return x


def first_event(e, v, start):
    """First epoch >= start where E > trailing median + 1 Ha and var > 20x median."""
    em, vm = trailing_median(e[::10], 50), trailing_median(v[::10], 50)
    idx = np.nonzero((e[::10] > em + EVENT_DE) & (v[::10] > EVENT_VAR * vm))[0]
    idx = idx[idx * 10 >= start]
    return int(idx[0] * 10) if len(idx) else None


def original_logdir(d, name):
    """The run a replay was reloaded from: reload_config.json's logdir, if present."""
    path = os.path.join(d, "reload_config.json")
    if os.path.exists(path):
        logdir = json.load(open(path)).get("logdir")
        if logdir and logdir != "NONE":
            return logdir
    return ORIGINALS["e14s1" if "e14s1" in name else "e18s2"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--logroot", default=DEFAULT_ROOT)
    ap.add_argument("--pattern", default="f*_N2_4.0_*", help="replay logdir glob")
    ap.add_argument(
        "--precursors",
        action="store_true",
        help="print binned medians of the diag_* metrics from the replay start to the "
        "first catastrophe (or the end): is there a slow precursor?",
    )
    ap.add_argument(
        "--sustained",
        action="store_true",
        help="calibrate the F3c trigger: for each (window W, count M), how often "
        "carried/eps > K on >= M of the last W rows fires in healthy rows, and its "
        "lead",
    )
    ap.add_argument(
        "--carried",
        action="store_true",
        help="per-row calibration of diag_carried_over_eps = ||A(beta phi)||/||eps||: "
        "distribution in healthy rows, and first row above each threshold",
    )
    ap.add_argument(
        "--rip",
        action="store_true",
        help="per-row calibration of the probe's raw window ratio probe_r_ip: "
        "distribution in healthy rows, and for each (T, W, M) rule (r_ip > T on >= M "
        "of the last W rows) its healthy firings and lead",
    )
    args = ap.parse_args()

    for d in sorted(glob.glob(os.path.join(args.logroot, args.pattern))):
        name = os.path.basename(d)
        original = original_logdir(d, name)
        e, v = load(d, "energy_noclip"), load(d, "variance_noclip")
        orig_e = load(original, "energy_noclip")
        # a file only the replay writes: sg_* (safeguard on) or diag_* (originals ran
        # without diagnostics)
        new = load(d, "sg_trigger")
        if new is None:
            new = load(d, "diag_bound_ratio")
        print(f"\n=== {name}  (original: {original})")
        if e is None or new is None:
            print("  no replay epochs logged yet")
            continue
        # reload.append copies the original history and the replay re-logs from the
        # checkpoint epoch, so line index = epoch only to within one; find the shift.
        start = len(e) - len(new)
        last = len(e) - 1
        seg = e[start : start + 500]
        devs = {}
        for off in (-1, 0, 1):
            ref = orig_e[max(0, start + off) :]
            n = min(len(seg), len(ref))
            if n > 0:
                devs[off] = (np.nanmax(np.abs(seg[:n] - ref[:n])), n)
        if not devs:
            print("  reproduction: no overlap with the original file")
            off = 0
            devs[0] = (np.nan, 0)
        off = min(devs, key=lambda k: devs[k][0])
        print(
            f"  reproduction: max |E - E_orig| over the first {devs[off][1]} replay rows: "
            f"{devs[off][0]:.2e} Ha (row shift {off:+d} vs the original file)"
        )
        ev = first_event(e, v, start)
        tail = np.nanmean(e[max(start, last - 500) : last + 1])
        before = np.nanmean(orig_e[start - 500 : start])
        print(
            f"  last epoch {last}; first catastrophe {ev}; mean E last 500 {tail:.4f} "
            f"(original, 500 rows before the replay: {before:.4f})"
        )

        scale = load(d, "diag_carried_scale", start)
        if scale is not None:
            capped = np.nonzero(scale[start:] < 1.0)[0]
            first_cap = int(start + capped[0]) if len(capped) else None
            print(
                f"  carried cap: active on {len(capped)} of {last + 1 - start} rows "
                f"({100.0 * len(capped) / max(1, last + 1 - start):.2f}%); first at "
                f"{first_cap}; min scale {np.nanmin(scale[start:]):.3g}"
            )
        trig, rew = load(d, "sg_trigger", start), load(d, "sg_rewind", start)
        if trig is not None:
            t_ep = np.nonzero(trig > 0)[0]
            r_ep = np.nonzero(rew > 0)[0]
            print(
                f"  safeguard: {len(t_ep)} triggers at {t_ep[:20].tolist()}"
                f"{' ...' if len(t_ep) > 20 else ''}; {len(r_ep)} rewinds at "
                f"{r_ep[:20].tolist()}"
            )
            for key, label in (
                ("sg_trigger_carried", "carried-trigger rewinds"),
                ("sg_probe_bad", "non-finite probe rows"),
            ):
                x = load(d, key, start)
                if x is not None:
                    rows = np.nonzero(x > 0)[0]
                    print(f"    {label}: {len(rows)} at {rows[:20].tolist()}")
        if ev is not None:
            cols = {k: load(d, k, start) for k in ONSET_KEYS}
            cols = {k: x for k, x in cols.items() if x is not None}
            short = {
                k: k.replace("diag_", "").replace("update_sq_norm_", "")[:12]
                for k in cols
            }
            print("  onset anatomy, epochs", ev - 30, "to", ev + 2)
            print("  " + f"{'epoch':>6}" + "".join(f"{short[k]:>13}" for k in cols))
            for ep in range(max(start, ev - 30), min(last, ev + 2) + 1):
                print(
                    "  " + f"{ep:>6}" + "".join(f"{cols[k][ep]:>13.4g}" for k in cols)
                )
        if args.precursors:
            precursor_table(d, start, ev if ev is not None else last)
        if args.carried:
            carried_table(d, start, ev, last)
        if args.sustained:
            sustained_table(d, start, ev, last)
        if args.rip:
            rip_table(load(d, "probe_r_ip"), start, ev, last)


PRECURSOR_KEYS = [
    "diag_carried_over_eps",
    "diag_equation_residual",
    "diag_gram_min_eig_preclip",
    "diag_gram_lam_max",
    "diag_bound_ratio",
    "probe_r_ip",
    "update_sq_norm_preclip",
    "variance_noclip",
]


CARRIED_THRESHOLDS = [5, 10, 20, 50]


def carried_table(d, start, ev, last):
    """Per-row ||A(beta phi)||/||eps||. "Healthy" = from the replay start to 300 rows
    before the first catastrophe (or the end). For each threshold K: rows above K in the
    healthy span (false alarms if the run survives), and the first row above K overall
    together with its lead over the catastrophe."""
    c = load(d, "diag_carried_over_eps", start)
    if c is None:
        return
    hi = (ev - 300) if ev is not None else last + 1
    h = c[start:hi]
    h = h[np.isfinite(h)]
    if len(h):
        print(
            f"  carried/eps, healthy rows {start}-{hi - 1}: median {np.median(h):.3g}, "
            f"p99 {np.percentile(h, 99):.3g}, p99.9 {np.percentile(h, 99.9):.3g}, "
            f"max {h.max():.3g}"
        )
    for k in CARRIED_THRESHOLDS:
        above = np.nonzero(c[start:] > k)[0]
        first = int(start + above[0]) if len(above) else None
        n_healthy = int(np.sum(h > k)) if len(h) else 0
        lead = (ev - first) if (first is not None and ev is not None) else None
        print(
            f"    K={k:<3} healthy rows above: {n_healthy:<6} first row above: "
            f"{first}  lead over catastrophe: {lead}"
        )


SUSTAINED_K = 10
SUSTAINED_GRID = [(20, 3), (50, 3), (50, 5), (50, 10), (100, 5), (100, 10), (100, 20)]


def sustained_table(d, start, ev, last):
    """Calibrate the F3c trigger: carried/eps > K on >= M of the last W rows. Healthy
    span as in carried_table. Firings are counted as rising edges (the real trigger
    clears its window on a rewind). Rows logged under a binding cap are pre-cap
    values, so the F3b cap arms are valid input. In a failing run the "healthy" span
    can include the start of the approach, so read false alarms off the surviving
    runs (the F2 safeguard-on replays, E18 s2) and lead times off the failing ones."""
    c = load(d, "diag_carried_over_eps", start)
    if c is None:
        return
    hit = ~(c[start:] <= SUSTAINED_K)  # NaN counts as a hit, as in the code
    hi = (ev - 300) if ev is not None else last + 1
    print(
        f"  sustained trigger (carried/eps > {SUSTAINED_K}), healthy rows "
        f"{start}-{hi - 1}:"
    )
    for w, m in SUSTAINED_GRID:
        edges = firing_edges(hit, w, m) + start
        n_healthy = int(np.sum(edges < hi))
        first = int(edges[0]) if len(edges) else None
        lead = (ev - first) if (first is not None and ev is not None) else None
        print(
            f"    W={w:<4} M={m:<3} healthy firings: {n_healthy:<4} first firing: "
            f"{first}  lead over catastrophe: {lead}"
        )


def firing_edges(hit, w, m):
    """Rows (indices into `hit`) where ">= m hits in the last w rows" switches on.
    Counted as rising edges: a sustained episode fires once."""
    count = np.convolve(hit.astype(int), np.ones(w, int))[: len(hit)]
    on = count >= m
    return np.nonzero(on & ~np.concatenate([[False], on[:-1]]))[0]


# (T, W, M): r_ip > T on >= M of the last W rows. (T, 1, 1) is the bare threshold.
# r_ip is itself a ratio over two adjacent p=30 windows, so consecutive rows are
# strongly correlated and M counts rows, not independent tests.
RIP_RULES = [
    (1.5, 1, 1),
    (1.5, 30, 20),
    (2, 1, 1),
    (2, 10, 5),
    (2, 30, 10),
    (3, 1, 1),
    (3, 10, 5),
    (5, 1, 1),
]


def rip_rules(r, start, hi, ev):
    """For each RIP_RULES entry: (firings in the healthy rows [start, hi), first firing
    at or after start, lead of that first firing over the catastrophe `ev`). A
    non-finite r_ip (the probe overflowed) counts as a hit."""
    out = []
    for t, w, m in RIP_RULES:
        edges = firing_edges(~(r[start:] <= t), w, m) + start
        first = int(edges[0]) if len(edges) else None
        lead = (ev - first) if (first is not None and ev is not None) else None
        out.append((int(np.sum(edges < hi)), first, lead))
    return out


def rip_summary(r, start, hi):
    """Distribution of r_ip over the healthy rows [start, hi): median, p99, p99.9, max,
    the fraction of rows above 1 (probe residual growing over the last window) and the
    longest unbroken run of such rows."""
    h = r[start:hi]
    h = h[np.isfinite(h)]
    if not len(h):
        return None
    above = h > 1.0
    longest, cur = 0, 0
    for a in above:
        cur = cur + 1 if a else 0
        longest = max(longest, cur)
    return dict(
        median=np.median(h),
        p99=np.percentile(h, 99),
        p999=np.percentile(h, 99.9),
        max=h.max(),
        frac_above_1=above.mean(),
        longest_above_1=longest,
    )


def rip_table(r, start, ev, last):
    """Per-row probe_r_ip. Healthy span as in carried_table. Read false alarms off the
    surviving runs and lead times off the failing ones."""
    if r is None:
        return
    hi = (ev - 300) if ev is not None else last + 1
    st = rip_summary(r, start, hi)
    if st is not None:
        print(
            f"  r_ip, healthy rows {start}-{hi - 1}: median {st['median']:.3g}, "
            f"p99 {st['p99']:.3g}, p99.9 {st['p999']:.3g}, max {st['max']:.3g}; "
            f"rows > 1: {100 * st['frac_above_1']:.1f}%, longest run > 1: "
            f"{st['longest_above_1']}"
        )
    for (t, w, m), (n_healthy, first, lead) in zip(
        RIP_RULES, rip_rules(r, start, hi, ev)
    ):
        print(
            f"    r_ip > {t:<3} on >= {m:<2} of {w:<2} rows  healthy firings: "
            f"{n_healthy:<4} first firing: {first}  lead over catastrophe: {lead}"
        )


def precursor_table(d, start, end):
    """Medians in 500-row bins from the replay start to `end`, then 50-row bins over
    the last 500 rows before it. A slow precursor shows up as a trend in the coarse
    bins; the fine bins show the final approach."""
    cols = {k: load(d, k, start) for k in PRECURSOR_KEYS}
    cols = {k: x for k, x in cols.items() if x is not None}
    short = {
        k: k.replace("diag_", "").replace("update_sq_norm_", "")[:12] for k in cols
    }
    edges = list(range(start, max(start, end - 500), 500)) + list(
        range(max(start, end - 500), end, 50)
    )
    print(f"  precursors: bin medians up to row {end}")
    print("  " + f"{'rows':>13}" + "".join(f"{short[k]:>13}" for k in cols))
    for lo, hi in zip(edges, edges[1:] + [end]):
        if hi <= lo:
            continue
        cells = "".join(f"{np.nanmedian(cols[k][lo:hi]):>13.4g}" for k in cols)
        print("  " + f"{lo:>6}-{hi:<6}" + cells)


if __name__ == "__main__":
    main()
