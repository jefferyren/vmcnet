"""How the stretched-N2 runs fail: first catastrophe, SPRING's exact bound, triggers.

WHY THIS EXISTS, SEPARATELY FROM divergence_check.py. That script answers *when* a run
died -- the NaN epoch and the last block within 0.5 Ha of the best energy. This one
answers *how*. Every diverged N2-4.0 run has a FIRST catastrophe, partly recovers, and
only NaNs later (divergence_check's "healthy until" marks the second event, not the
first). After the first catastrophe the SPRING momentum buffer grows far beyond anything
the algorithm can produce in exact arithmetic.

THE BOUND. SPRING's step (spring.py; same_sampled_spring_unified.py; prime_sr.py) is

    phi_k = beta * (I - P_k) phi_{k-1} + A^T (T + lam)^-1 eps_k,      0 <= P_k < I,

so in exact arithmetic

    ||phi_k|| <= beta * ||phi_{k-1}|| + ||eps_k|| / (2 sqrt(lam)),

with ||eps_k||^2 = (clipped variance) * (N - 1) / N and ||phi_k|| =
sqrt(update_sq_norm_preclip) / eta_k. A ratio above 1 cannot come from the algorithm: it
is floating-point error in the Gram solve. With wandb history (every 10th step) the
bound is chained over the 10-step gap using the larger of the two bracketing variances,
which is generous. With --logdirs (per-epoch .txt files on Savio) it is exact per step.

Usage:
    python slurm/n2_failure_anatomy.py                     # wandb, vmcnet-phase-e
    python slurm/n2_failure_anatomy.py --cache /tmp/n2.pkl # reuse a previous pull
    python slurm/n2_failure_anatomy.py --logdirs \\
        "/global/scratch/users/$USER/vmcnet_logs/phase_e/e1[4-9]*/*N2_4.0*"
"""

import argparse
import glob
import json
import os
import pickle
import re

import numpy as np

PROJECT = "ren27-university-of-california-berkeley/vmcnet-phase-e"
NAME_RE = re.compile(r"^e1[4-9]_N2_4\.0_")
BASE_KEYS = [
    "_step",
    "energy_noclip",
    "variance",
    "variance_noclip",
    "update_sq_norm_preclip",
]
OPTIONAL_KEYS = ["mu", "probe_res_norm", "probe_r_ip"]
TXT_KEYS = BASE_KEYS[1:] + OPTIONAL_KEYS
WINDOW_STEPS = 500  # trailing window for medians
EVENT_DE = 1.0  # Ha above the trailing median energy ...
EVENT_VAR = 20.0  # ... AND variance above this multiple of its trailing median
TRIGGER = 3.0  # pre-clip step above this multiple of its trailing median
COMPLETE_STEP = 99000  # a run that logged past this finished its 100k


def arm_of(name):
    """Short arm label from a run or logdir name."""
    for tag, label in [
        ("prime_sr", "PRIME-SR"),
        ("spring_mu0.995", "SPRING 0.995"),
        ("spring_mu0.99", "SPRING 0.99"),
        ("ssu_cap0.01", "SS-SPRING C=1e-2"),
        ("ssu_nocap", "SS-SPRING no cap"),
        ("eta0.0005", "SS-SPRING eta 5e-4"),
        ("eta0.0015", "SS-SPRING eta 1.5e-3"),
        ("eta0.001", "SS-SPRING eta 1e-3"),
        ("ssu_defaults", "SS-SPRING"),
    ]:
        if tag in name:
            return label
    return "?"


def hyperparams_from_name(name):
    """wandb stores the config as one YAML string, so read eta / mu from the run name."""
    m = re.search(r"eta([\d.]+)_s\d", name)
    eta0 = float(m.group(1)) if m else 0.002
    mu = 0.995 if "mu0.995" in name else (0.99 if "mu0.99" in name else None)
    return dict(eta0=eta0, decay=1e-4, lam=1e-3, nchains=1000, mu=mu)


def load_wandb(cache):
    if cache and os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    import wandb

    api = wandb.Api(timeout=120)
    runs = {}
    for r in api.runs(PROJECT):
        if not NAME_RE.match(r.name):
            continue
        rows = []
        for keys in (BASE_KEYS + OPTIONAL_KEYS, BASE_KEYS + ["mu"], BASE_KEYS):
            rows = list(r.scan_history(keys=keys, page_size=5000))
            if rows:
                break
        arr = {
            k: np.array([np.nan if row.get(k) is None else row[k] for row in rows], float)
            for k in keys
        }
        arr["step"] = arr.pop("_step")
        if "mu" in arr:
            arr["mu_hist"] = arr.pop("mu")
        arr.update(hyperparams_from_name(r.name), name=r.name)
        runs[r.name] = arr
        print(f"  pulled {r.name}: {len(rows)} rows", flush=True)
    if cache:
        pickle.dump(runs, open(cache, "wb"))
    return runs


def load_logdirs(pattern):
    runs = {}
    for d in sorted(glob.glob(pattern)):
        cfg_path = os.path.join(d, "config.json")
        if not os.path.exists(os.path.join(d, "update_sq_norm_preclip.txt")):
            continue
        vmc = json.load(open(cfg_path))["vmc"]
        opt = vmc["optimizer"][vmc["optimizer_type"]]
        arr = {}
        for k in TXT_KEYS:
            path = os.path.join(d, k + ".txt")
            if os.path.exists(path):
                arr[k] = np.atleast_1d(np.loadtxt(path))
        n = min(len(v) for v in arr.values())
        arr = {k: v[:n] for k, v in arr.items()}
        arr["step"] = np.arange(n, dtype=float)
        if "mu" in arr:
            arr["mu_hist"] = arr.pop("mu")
        name = os.path.basename(os.path.normpath(d))
        mu = None if vmc["optimizer_type"] != "spring" else float(opt["mu"])
        arr.update(
            eta0=float(opt["learning_rate"]),
            decay=float(opt["learning_decay_rate"]),
            lam=float(opt["damping"]),
            nchains=int(vmc["nchains"]),
            mu=mu,
            name=name,
        )
        runs[name] = arr
    return runs


def trailing_median(x, w):
    out = np.full(len(x), np.nan)
    for i in range(w, len(x)):
        out[i] = np.nanmedian(x[i - w : i])
    return out


def decimate(run, stride):
    """Every `stride`-th row, so event detection sees the same cadence for both sources."""
    if stride <= 1:
        return run
    keep = (run["step"] % stride) == 0
    return {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in run.items()}


def first_event(run, w):
    e, v = run["energy_noclip"], run["variance_noclip"]
    em, vm = trailing_median(e, w), trailing_median(v, w)
    idx = np.nonzero((e > em + EVENT_DE) & (v > EVENT_VAR * vm))[0]
    return int(idx[0]) if len(idx) else None


def bound_ratio(run):
    """||phi_k|| / (exact bound), chained over the gap since the previous logged row."""
    s = run["step"]
    eta = run["eta0"] / (1.0 + run["decay"] * s)
    phi = np.sqrt(run["update_sq_norm_preclip"]) / eta
    n_ch = run["nchains"]
    eps = np.sqrt(run["variance"] * (n_ch - 1) / n_ch)
    beta = run.get("mu_hist")
    if beta is None or not np.isfinite(beta).any():
        beta = np.full(len(s), np.nan if run["mu"] is None else run["mu"])
    ratio = np.full(len(s), np.nan)
    for i in range(1, len(s)):
        gap = int(s[i] - s[i - 1])
        if gap <= 0 or gap > 20:
            continue
        b = beta[i]
        eps_max = np.nanmax([eps[i], eps[i - 1]])
        geo = sum(b**j for j in range(gap))  # sum of beta^j over the gap
        bnd = b**gap * phi[i - 1] + geo * eps_max / (2.0 * np.sqrt(run["lam"]))
        with np.errstate(invalid="ignore", over="ignore"):
            ratio[i] = phi[i] / bnd
    return ratio


def episodes(steps, flag, gap=WINDOW_STEPS):
    out = []
    for i in np.nonzero(flag)[0]:
        if not out or steps[i] - out[-1] > gap:
            out.append(steps[i])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--logdirs", help="glob of per-run logdirs (per-epoch .txt files)")
    ap.add_argument("--cache", help="pickle cache for the wandb pull")
    args = ap.parse_args()

    runs = load_logdirs(args.logdirs) if args.logdirs else load_wandb(args.cache)

    print(f"\n{len(runs)} runs. Bound ratio = ||phi_k|| / exact-arithmetic bound "
          "(<= 1 always, for exact SPRING).\n")
    hdr = (f"{'run':<42}{'arm':<20}{'1st event':>10}{'last step':>10}"
           f"{'max ratio healthy':>18}{'first >1':>9}{'at event':>10}{'max after':>11}")
    print(hdr)
    survivors_trig, survivor_steps, healthy_ratios = 0, 0, []
    leads_trig, leads_rip, healthy_rip = [], [], []
    for name in sorted(runs):
        full = runs[name]
        ratio_full = bound_ratio(full)
        stride = 1 if np.nanmedian(np.diff(full["step"])) > 1 else 10
        run = decimate(dict(full, ratio=ratio_full), stride)
        s = run["step"]
        w = max(1, WINDOW_STEPS // int(np.nanmedian(np.diff(s))))
        i_ev = first_event(run, w)
        last = int(full["step"][-1])
        survived = last >= COMPLETE_STEP and i_ev is None
        lo = np.searchsorted(full["step"], 1000)
        hi = np.searchsorted(full["step"], s[i_ev]) if i_ev is not None else len(full["step"])
        healthy = ratio_full[lo:hi]
        over = np.nonzero(healthy > 1)[0]
        first_over = int(full["step"][lo + over[0]]) if len(over) else None
        j_ev = np.searchsorted(full["step"], s[i_ev]) if i_ev is not None else None
        at_ev = ratio_full[j_ev] if j_ev is not None else np.nan
        after = np.nan
        if j_ev is not None:
            tail = ratio_full[j_ev:]
            fin = tail[np.isfinite(tail)]
            after = fin.max() if len(fin) else np.inf
        print(f"{name:<42}{arm_of(name):<20}{(int(s[i_ev]) if i_ev is not None else '-'):>10}"
              f"{last:>10}{np.nanmax(healthy):>18.3g}{(first_over or '-'):>9}"
              f"{at_ev:>10.3g}{after:>11.3g}")
        if survived:
            healthy_ratios.append(np.nanmax(healthy))

        # trigger: pre-clip step above TRIGGER x its trailing median
        u = run["update_sq_norm_preclip"]
        flag = u > TRIGGER * trailing_median(u, w)
        if survived:
            survivors_trig += len(episodes(s, flag))
            survivor_steps += last
        elif i_ev is not None:
            win = [j for j in range(max(0, i_ev - 40), i_ev + 1) if flag[j]]
            leads_trig.append((name, int(s[i_ev] - s[win[0]]) if win else None))
        if "probe_r_ip" in run:
            r = run["probe_r_ip"]
            if i_ev is not None:
                win = [j for j in range(max(0, i_ev - 30), i_ev + 1) if r[j] > 2]
                leads_rip.append((name, int(s[i_ev] - s[win[0]]) if win else None))
                healthy_rip.append(r[np.searchsorted(s, 2000): max(0, i_ev - 50)])
            elif survived:
                healthy_rip.append(r[np.searchsorted(s, 2000):])

    if healthy_ratios:
        print(f"\nSurvivors: max healthy bound ratio {max(healthy_ratios):.3g} "
              f"(median over runs {np.median(healthy_ratios):.3g}).")
    print(f"Trigger 'pre-clip step > {TRIGGER:g}x trailing {WINDOW_STEPS}-step median': "
          f"{survivors_trig} alarm episodes in {survivor_steps / 1e6:.2f}M survivor steps.")
    print("  steps between first alarm and the first event (0 = fires at the event row):")
    for name, lead in leads_trig:
        print(f"    {name:<42} {lead}")
    if leads_rip:
        h = np.concatenate([x[np.isfinite(x)] for x in healthy_rip])
        print(f"Probe r_ip, healthy: median {np.median(h):.3f}, p99 {np.percentile(h, 99):.3f}, "
              f"p99.9 {np.percentile(h, 99.9):.3f}")
        print("  steps before the first event at which r_ip first exceeds 2 (0 = only at it):")
        for name, lead in leads_rip:
            print(f"    {name:<42} {lead}")
    spike_response(runs)


def spike_response(runs, horizon_steps=600, spike=10.0):
    """Does a local-energy outlier kick the update? Event-triggered median of the pre-clip
    step (relative to its trailing median) after unclipped-variance spikes, in healthy
    stretches only (survivors, or more than 1000 steps before a run's first event)."""
    by_arm = {}
    for name, full in runs.items():
        if "C=1e-2" in arm_of(name) or "no cap" in arm_of(name):
            continue
        stride = 1 if np.nanmedian(np.diff(full["step"])) > 1 else 10
        run = decimate(full, stride)
        s = run["step"]
        dt = int(np.nanmedian(np.diff(s)))
        w, hz = max(1, WINDOW_STEPS // dt), horizon_steps // dt
        i_ev = first_event(run, w)
        stop = (i_ev - 1000 // dt) if i_ev is not None else len(s) - hz - 1
        v, u = run["variance_noclip"], run["update_sq_norm_preclip"]
        vm, um = trailing_median(v, w), trailing_median(u, w)
        kept = []
        for i in range(np.searchsorted(s, 2000), max(0, stop)):
            if v[i] > spike * vm[i] and (not kept or i - kept[-1] > hz):
                kept.append(i)
        for i in kept:
            if um[i] > 0 and i + hz < len(u):
                by_arm.setdefault(arm_of(name), []).append(u[i : i + hz + 1] / um[i])
    print(f"\nPre-clip step / trailing median after unclipped-variance spikes (>{spike:g}x),"
          " healthy stretches; median over spikes at +0/+50/+200/+600 steps:")
    for a, lst in sorted(by_arm.items()):
        m = np.nanmedian(np.array(lst), axis=0)
        idx = [0, 50 // dt, 200 // dt, len(m) - 1]
        print(f"  {a:<22} n={len(lst):<4} " + "  ".join(f"{m[j]:.2f}" for j in idx))


if __name__ == "__main__":
    main()
