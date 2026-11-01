"""H1 step 3: freeze theta at a snapshot and run Kaczmarz++ on the frozen linear system.

THE QUESTION. In SS-SPRING the probe runs the SPRING/Kaczmarz++ recursion with momentum
beta on the SAME sampled operator A_k as the main update, against a target A_k x* with
a known solution, and the adaptive controller sets beta from how fast the probe
residual falls. In a VMC run A_k changes every step because theta moves. Hypothesis:
that change slows the probe, the controller reads a slow probe as "needs more
momentum" and pushes beta up, and the high beta diverges. If so, Kaczmarz++ on the
system frozen at a snapshot of a failing run should converge fine at the run's beta,
while the run itself (same snapshot onward) does not.

WHAT THIS SCRIPT DOES, for each snapshot (a checkpoint of a failing run):
  1. Load theta, the walkers and the SS-SPRING state. theta is never updated again.
  2. For --nsteps steps: move the walkers with the run's own MCMC walker_fn (so each
     step sees a fresh block of rows of the same frozen system), build A_t and its
     Gram eigendecomposition once with the module's own _build_operators, and advance
     every probe arm one step on that A_t with exactly the module's probe recursion:
         phi <- A^T (A A^T + lam_p)^-1 A (x* - z - beta phi) + beta phi
         z   <- z + probe_lr * phi
     then record the residual ||A_t (z - x*)|| (what the controller sees) and the
     same residual on a fixed held-out walker set (a cleaner convergence measure).
  3. Arms, each with --nxstar independent targets x* (the same x* at every snapshot):
       - beta fixed at each value in --betas ("run" = the checkpoint's beta, i.e. what
         the run was using at that point);
       - "adaptive": a FRESH controller (beta starts at the config's mu, r_hat = 1,
         empty buffer), updated with the module's own _adaptive_beta_update. This is
         the beta the controller would choose on the frozen system.
  4. Save everything to <out>/<run>_<snapshot>_e<epoch>.npz and print a summary next
     to the run's own probe (probe_r_ip.txt, mu.txt) over the same epochs.

SNAPSHOTS. By default: the earliest checkpoint in the logdir ("healthy") plus the
checkpoint at or before event - offset for each --offsets value, where the event is the
run's FIRST catastrophe as defined in n2_failure_anatomy.py (energy > trailing median +
1 Ha AND unclipped variance > 20x its trailing median, 500-epoch window), or the first
non-finite energy if earlier. Checkpoint file K.npz holds the state at the START of
epoch K-1 (vmcnet saves before the update), and the snapshot's epoch is that K-1.

READING IT. r_ip is the module's window ratio: sum of squared probe residuals over the
last p steps / over the p before (r_ip < 1 = shrinking). On the real runs the healthy
in-run probe sits at r_ip ~ 1.00 (the residual plateaus). Per-step factor g = exp(slope
of log held-out residual) over the second half of the frozen run.
  - frozen arm at the run's beta contracts (r_ip, g < 1) near the event while the run's
    own probe stalls or grows -> the changing system is the problem (hypothesis holds);
  - frozen arm at the run's beta also stalls or grows near the event -> the system
    itself became too hard for that beta;
  - adaptive beta settles below the run's beta -> without drift the controller would
    have chosen less momentum.
Caveat: the residual eventually hits the float32 floor, after which r_ip -> 1; check
the total drop (log10 res) before reading a late r_ip ~ 1 as a stall.

Usage (Savio GPU node; slurm/h1_frozen_probe.sbatch runs one snapshot per array task):
    python slurm/h1_frozen_probe.py --logdir <replay logdir> --list
    python slurm/h1_frozen_probe.py --logdir <replay logdir> --snapshot-index 0
    python slurm/h1_frozen_probe.py --logdir <dir> --ckpts 10050.npz,11200.npz
    python slurm/h1_frozen_probe.py --summary <out dir>
"""

import argparse
import glob
import os
import re
from typing import NamedTuple

import numpy as np

USER = os.environ.get("USER", "")
DEFAULT_OUT = f"/global/scratch/users/{USER}/vmcnet_logs/phase_h/h1_frozen"


def checkpoint_epochs(logdir):
    """{epoch: path} of the regular checkpoints; K.npz holds the start of epoch K-1."""
    out = {}
    for path in glob.glob(os.path.join(logdir, "checkpoints*", "*.npz")):
        m = re.fullmatch(r"(\d+)\.npz", os.path.basename(path))
        if m:
            out[int(m.group(1)) - 1] = path
    return dict(sorted(out.items()))


def load_txt(logdir, name):
    path = os.path.join(logdir, name + ".txt")
    return np.atleast_1d(np.loadtxt(path)) if os.path.exists(path) else None


def find_event(logdir, search_from):
    """First catastrophe at or after `search_from` (n2_failure_anatomy's definition)."""
    from n2_failure_anatomy import EVENT_DE, EVENT_VAR, WINDOW_STEPS, trailing_median

    e, v = load_txt(logdir, "energy_noclip"), load_txt(logdir, "variance_noclip")
    n = min(len(e), len(v))
    e, v = e[:n], v[:n]
    em, vm = trailing_median(e, WINDOW_STEPS), trailing_median(v, WINDOW_STEPS)
    hit = (e > em + EVENT_DE) & (v > EVENT_VAR * vm)
    hit |= ~np.isfinite(e)
    hit[: max(search_from, 0)] = False
    idx = np.nonzero(hit)[0]
    return int(idx[0]) if len(idx) else None


def pick_snapshots(logdir, offsets, ckpts):
    """[(label, epoch, path)] and the event epoch (None if not used / not found)."""
    epochs = checkpoint_epochs(logdir)
    if not epochs:
        raise SystemExit(f"no checkpoints under {logdir}/checkpoints*")
    if ckpts:
        by_name = {os.path.basename(p): (e, p) for e, p in epochs.items()}
        snaps = []
        for name in ckpts:
            if name not in by_name:
                raise SystemExit(f"{name} not in {logdir}/checkpoints*")
            e, p = by_name[name]
            snaps.append((f"ckpt{e}", e, p))
        return snaps, None

    first = min(epochs)
    event = find_event(logdir, first)
    if event is None:
        raise SystemExit(
            f"no first event found in {logdir} after epoch {first}; "
            "pass --ckpts explicitly"
        )
    snaps = [("healthy", first, epochs[first])]
    for off in offsets:
        before = [e for e in epochs if e <= event - off]
        if not before:
            print(f"  (event-{off}: no checkpoint at or before {event - off}; skipped)")
            continue
        e = max(before)
        if e in [s[1] for s in snaps]:
            print(f"  (event-{off}: same checkpoint as an earlier snapshot; skipped)")
            continue
        snaps.append((f"event-{off}", e, epochs[e]))
    return snaps, event


class ProbeArm(NamedTuple):
    """One probe's state; stacked over arms along a leading axis."""

    z: object
    phi: object
    x_star: object
    residual_buffer: object
    r_hat: object
    beta: object
    checkpoint_idx: object
    adaptive: object


def make_frozen_step(log_psi_apply, kernel_fn, probe_lr, probe_damping, p):
    """Jitted (params, positions, hold_pos, arms, t) -> (arms, records, spectrum).

    One Gram eigendecomposition per call, shared by all arms; the per-arm recursion is
    line for line the probe block of get_same_sampled_spring_unified_step, and the
    controller is the module's _adaptive_beta_update (its beta is kept only for
    adaptive arms).
    """
    import jax
    import jax.numpy as jnp

    from vmcnet.updates import same_sampled_spring_unified as ssu

    tree_map = jax.tree_util.tree_map

    @jax.jit
    def frozen_step(params, positions, hold_pos, arms, t):
        apply_A, apply_AT, tvals, tvecs = ssu._build_operators(
            kernel_fn, log_psi_apply, params, positions
        )
        tvals_clipped = jnp.maximum(tvals, 0.0)
        # Same operation order as the module's solve(): (tvecs @ D) @ tvecs.T @ rhs.
        solve_mat = tvecs @ jnp.diag(1.0 / (tvals_clipped + probe_damping)) @ tvecs.T

        n_hold = hold_pos.shape[0]
        _, jvp_hold = jax.linearize(log_psi_apply, params, hold_pos)
        zeros_hold = jnp.zeros_like(hold_pos)

        def apply_A_hold(vec):
            o = jvp_hold(vec, zeros_hold) / jnp.sqrt(n_hold)
            return o - jnp.mean(o, axis=0, keepdims=True)

        def one_arm(arm):
            beta = arm.beta
            probe_in = tree_map(
                lambda xs, z, ph: xs - z - beta * ph, arm.x_star, arm.z, arm.phi
            )
            v = solve_mat @ apply_A(probe_in)
            w = apply_AT(v)
            phi_new = tree_map(lambda w_, ph: w_ + beta * ph, w, arm.phi)
            z_new = tree_map(lambda z, ph: z + probe_lr * ph, arm.z, phi_new)
            resid_in = tree_map(lambda z, xs: z - xs, z_new, arm.x_star)
            res = jnp.linalg.norm(apply_A(resid_in))
            res_hold = jnp.linalg.norm(apply_A_hold(resid_in))
            buf, r_hat, beta_new, ckpt, r_ip = ssu._adaptive_beta_update(
                arm.residual_buffer, res, arm.r_hat, beta, arm.checkpoint_idx, t, p
            )
            new_arm = ProbeArm(
                z=z_new,
                phi=phi_new,
                x_star=arm.x_star,
                residual_buffer=buf,
                r_hat=r_hat,
                beta=jnp.where(arm.adaptive, beta_new, beta),
                checkpoint_idx=ckpt,
                adaptive=arm.adaptive,
            )
            return new_arm, jnp.stack([res, res_hold, beta, r_ip])

        # lax.map, not vmap: one arm's jvp/vjp live at a time (2080 Ti memory).
        arms, records = jax.lax.map(one_arm, arms)
        spectrum = jnp.stack(
            [
                tvals.min(),
                tvals.max(),
                jnp.sum(tvals < probe_damping).astype(tvals.dtype),
            ]
        )
        return arms, records, spectrum

    return frozen_step


def init_arms(params, arm_specs, nxstar, seed, p):
    """Stacked fresh probes: z = phi = 0, one x* per (arm, draw), same draws per arm."""
    import jax
    import jax.numpy as jnp

    from vmcnet.updates import same_sampled_spring_unified as ssu

    keys = jax.random.split(jax.random.PRNGKey(seed), nxstar)
    x_draws = [ssu._draw_unit_norm_like(k, params) for k in keys]
    xs, betas, adaptive = [], [], []
    for _, beta0, is_adaptive in arm_specs:
        for x in x_draws:
            xs.append(x)
            betas.append(beta0)
            adaptive.append(is_adaptive)
    n = len(xs)
    stacked_x = jax.tree_util.tree_map(lambda *a: jnp.stack(a), *xs)
    zeros = jax.tree_util.tree_map(jnp.zeros_like, stacked_x)
    return ProbeArm(
        z=zeros,
        phi=zeros,
        x_star=stacked_x,
        residual_buffer=jnp.zeros((n, 2 * p)),
        r_hat=jnp.ones(n),
        beta=jnp.asarray(betas, jnp.float32),
        checkpoint_idx=jnp.ones(n, jnp.int32),
        adaptive=jnp.asarray(adaptive),
    )


def run_snapshot(logdir, label, epoch, path, event, args):
    import jax
    import jax.numpy as jnp

    import vmcnet.models as models
    import vmcnet.utils as utils
    from vmcnet.train import runners

    import neural_tangents as nt  # after vmcnet, see the SS-SPRING unit tests

    config = utils.io.load_config_dict(logdir, "config.json")
    dtype = runners._get_dtype(config)
    opt_cfg = config.vmc.optimizer.same_sampled_spring_unified
    p = int(opt_cfg.lb_window)
    probe_damping = float(opt_cfg.probe_damping)
    probe_lr = (
        float(opt_cfg.learning_rate)
        if opt_cfg.probe_lr < 0
        else float(opt_cfg.probe_lr)
    )

    directory, name = os.path.split(path)
    ck_epoch, data, params, opt, _ = utils.io.reload_vmc_state(directory, name)
    core = opt.core if hasattr(opt, "core") else opt
    run_beta = float(core.beta)
    if np.asarray(data["walker_data"]["position"]).ndim == 4:  # saved per device
        data = jax.tree_util.tree_map(lambda x: np.asarray(x)[0], data)
    data = jax.tree_util.tree_map(jnp.asarray, data)
    params = jax.tree_util.tree_map(jnp.asarray, params)

    ion_pos, ion_charges, nelec = runners._get_electron_ion_config_as_arrays(
        config, dtype=dtype
    )
    slog_psi = models.construct.get_model_from_config(
        config.model, nelec, ion_pos, ion_charges, dtype=dtype
    )
    log_psi_apply = models.construct.slog_psi_to_log_psi_apply(slog_psi.apply)
    _, walker_fn = runners._get_mcmc_fns(config.vmc, log_psi_apply, apply_pmap=False)
    kernel_fn = nt.empirical_kernel_fn(log_psi_apply, vmap_axes=0, trace_axes=())
    frozen_step = make_frozen_step(log_psi_apply, kernel_fn, probe_lr, probe_damping, p)

    arm_specs = []
    for b in args.betas.split(","):
        b = b.strip()
        beta0 = run_beta if b == "run" else float(b)
        arm_specs.append((f"beta={b}", beta0, False))
    arm_specs.append(("adaptive", float(opt_cfg.mu), True))
    arms = init_arms(params, arm_specs, args.nxstar, args.seed, p)
    narms = len(arm_specs) * args.nxstar
    arm_labels = [s[0] for s in arm_specs for _ in range(args.nxstar)]

    positions = data["walker_data"]["position"]
    hold_pos = positions[: args.nhold]
    print(
        f"\n=== {os.path.basename(os.path.normpath(logdir))} {label}: epoch {ck_epoch}"
        f" ({os.path.basename(path)}), event {event}, run beta {run_beta:.6f}, "
        f"probe_lr {probe_lr:g}, probe_damping {probe_damping:g}, p {p}, "
        f"{positions.shape[0]} walkers, {hold_pos.shape[0]} held out, "
        f"{narms} probes, {args.nsteps} frozen steps",
        flush=True,
    )

    rec = np.full((args.nsteps, narms, 4), np.nan, np.float64)
    spec = np.full((args.nsteps, 3), np.nan, np.float64)
    accept = np.full(args.nsteps, np.nan, np.float64)
    key = jax.random.PRNGKey(args.seed + 1)
    for t in range(args.nsteps):
        acc, data, key = walker_fn(params, data, key)
        arms, r, s = frozen_step(
            params, data["walker_data"]["position"], hold_pos, arms, jnp.int32(t)
        )
        rec[t], spec[t], accept[t] = np.asarray(r), np.asarray(s), float(acc)
        if (t + 1) % 500 == 0 or t == 0:
            msg = "  ".join(
                f"{lab} {rec[t, i, 1]:.3g}"
                for i, lab in enumerate(arm_labels)
                if i % args.nxstar == 0
            )
            print(f"  step {t + 1:5d}  held-out residual: {msg}", flush=True)
        if not np.isfinite(rec[t, :, 0]).any():
            print(f"  all probes non-finite at step {t + 1}; stopping")
            break

    os.makedirs(args.out, exist_ok=True)
    run_name = os.path.basename(os.path.normpath(logdir))
    out_path = os.path.join(args.out, f"{run_name}_{label}_e{ck_epoch}.npz")
    np.savez(
        out_path,
        res=rec[:, :, 0],
        res_hold=rec[:, :, 1],
        beta=rec[:, :, 2],
        r_ip=rec[:, :, 3],
        gram_min=spec[:, 0],
        gram_max=spec[:, 1],
        gram_n_below_damping=spec[:, 2],
        accept=accept,
        arm_labels=np.array(arm_labels),
        epoch=ck_epoch,
        event=-1 if event is None else event,
        label=label,
        logdir=logdir,
        run_beta=run_beta,
        p=p,
        probe_lr=probe_lr,
        probe_damping=probe_damping,
    )
    print(f"  saved {out_path}")
    summarize([out_path])


def arm_stats(d, i, p):
    """Median r_ip, per-step factor g, log10 drop and beta (2nd half) of probe i."""
    res_hold, r_ip = d["res_hold"][:, i], d["r_ip"][:, i]
    n = int(np.sum(np.isfinite(d["res"][:, i])))
    half = slice(max(n // 2, 2 * p), n)
    steps = np.arange(len(res_hold))[half]
    y = np.log(res_hold[half])
    ok = np.isfinite(y)
    g = float(np.exp(np.polyfit(steps[ok], y[ok], 1)[0])) if ok.sum() > 2 else np.nan
    rip = float(np.nanmedian(r_ip[half])) if n > 2 * p else np.nan
    drop = float(np.log10(res_hold[n - 1] / res_hold[min(p, n - 1)])) if n else np.nan
    return rip, g, drop, float(np.nanmedian(d["beta"][half, i])) if n else np.nan


def in_run(d, nsteps):
    """The run's own probe over the same epochs: median r_ip and mu (to the event)."""
    logdir, e = str(d["logdir"]), int(d["epoch"])
    end = e + nsteps
    if int(d["event"]) >= 0:
        end = min(end, int(d["event"]))
    out, n = [], 0
    for name in ("probe_r_ip", "mu"):
        x = load_txt(logdir, name)
        x = x[e:end] if x is not None else np.array([])
        n = len(x)
        out.append(float(np.nanmedian(x)) if n else np.nan)
    return out[0], out[1], n


def summarize(paths):
    rows = {}
    for path in sorted(paths):
        d = np.load(path, allow_pickle=False)
        run = os.path.basename(str(d["logdir"]).rstrip("/"))
        rows.setdefault(run, []).append((int(d["epoch"]), path, d))
    for run, items in rows.items():
        items.sort()
        d0 = items[0][2]
        labels = [str(x) for x in d0["arm_labels"]]
        groups = list(dict.fromkeys(labels))
        print(f"\n--- {run}: first event {int(d0['event'])} ---")
        print(
            "    r_ip = median probe window ratio over the 2nd half (<1 shrinking); "
            "g = per-step factor of the held-out residual; drop = log10 total"
        )
        head = (
            f"{'snapshot':>11}{'epoch':>7}{'run beta':>10}{'in-run r_ip / mu (n)':>29}"
        )
        for gname in groups:
            head += f" | {gname + ' r_ip g drop':>28}"
        print(head)
        for epoch, _, d in items:
            p = int(d["p"])
            nsteps = d["res"].shape[0]
            rip_run, mu_run, n_run = in_run(d, nsteps)
            line = (
                f"{str(d['label']):>11}{epoch:>7}{float(d['run_beta']):>10.5f}"
                f"{rip_run:>12.3f} / {mu_run:.5f} ({n_run:>4})"
            )
            for gname in groups:
                cols = [i for i, lab in enumerate(labels) if lab == gname]
                st = np.array([arm_stats(d, i, p) for i in cols])
                rip, g, drop, beta = np.nanmedian(st, axis=0)
                cell = f"{rip:7.3f} {g:9.6f} {drop:6.2f}"
                if gname == "adaptive":
                    cell = f"{rip:5.2f} {g:8.5f} b={beta:.4f}"
                line += f" | {cell:>28}"
            print(line)
        print(
            "    adaptive column: r_ip, g and the median beta the FRESH controller "
            "chose over the 2nd half"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--logdir", help="replay logdir with dense checkpoints")
    ap.add_argument("--offsets", default="500,200,50,20", help="epochs before event")
    ap.add_argument("--ckpts", default="", help="explicit checkpoint files (comma)")
    ap.add_argument("--snapshot-index", type=int, default=-1, help="-1 = all")
    ap.add_argument("--list", action="store_true", help="print snapshots and exit")
    ap.add_argument("--betas", default="0.99,0.995,run", help="fixed-beta arms")
    ap.add_argument("--nxstar", type=int, default=2, help="targets x* per arm")
    ap.add_argument("--nsteps", type=int, default=3000)
    ap.add_argument("--nhold", type=int, default=1000, help="held-out walkers")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--summary", default=None, help="summarize the .npz in this dir")
    args = ap.parse_args()

    if args.summary:
        paths = glob.glob(os.path.join(args.summary, "*.npz"))
        if not paths:
            raise SystemExit(f"no .npz in {args.summary}")
        summarize(paths)
        return
    if not args.logdir:
        ap.error("--logdir is required unless --summary is given")

    offsets = [int(x) for x in args.offsets.split(",") if x.strip()]
    ckpts = [x.strip() for x in args.ckpts.split(",") if x.strip()]
    snaps, event = pick_snapshots(args.logdir, offsets, ckpts)
    print(f"{args.logdir}: first event {event}")
    for i, (label, e, path) in enumerate(snaps):
        print(f"  [{i}] {label:>11}  epoch {e:6d}  {os.path.basename(path)}")
    if args.list:
        return
    if args.snapshot_index >= 0:
        if args.snapshot_index >= len(snaps):
            print(f"snapshot index {args.snapshot_index} out of range; nothing to do")
            return
        snaps = [snaps[args.snapshot_index]]
    for label, e, path in snaps:
        run_snapshot(args.logdir, label, e, path, event, args)


if __name__ == "__main__":
    main()
