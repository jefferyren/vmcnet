"""One SS-SPRING step from a checkpoint: how much of phi is float32 noise? (Phase F.)

The E15 s6 F3b arms reloaded the same 60000.npz and agreed on every input of their
first step, yet produced ||phi_new|| = 21.1 (no cap) and 45.4 (cap path, inactive),
i.e. the compiled program alone doubled the momentum. The hypothesis: Gram eigenvalues
below the float32 noise floor (min eig ~ -0.005, clipped to 0) are inverted at
1/damping = 1000, so the sub-floor eigendirections inject a phi component of order
||eps|| sqrt(floor) / damping ~ 28, which is rounding noise, and which does not carry
over to fresh walkers (the carried/eps precursor). This script measures that directly
on one state, with no training:

  1. Reproduce: the module's own step, cap off vs cap path at scale 1.
  2. Decompose phi_new = beta phi + phi_sub + phi_hi, where phi_sub comes from Gram
     eigendirections with eigenvalue < floor (default |min eig|) and phi_hi from the
     rest.
  3. Rounding sensitivity: re-run the step with the walkers permuted (identical in
     exact arithmetic). Hypothesis: phi_sub changes a lot, phi_hi barely.
  4. Float64 eigensolve of the float32 Gram (host numpy): is the floor the eigh's
     rounding or the kernel's accumulation?
  5. Held-out walkers: solve on half the walkers, apply the pieces to the other half.
     Hypothesis: ||A_heldout phi_sub|| >> ||A_fit phi_sub||, unlike phi_hi.
  4b. Gram rebuilt from explicit float32 per-walker gradient rows, three ways:
     float32 centred AFTER the product (as the module does: the uncentered kernel is
     dominated by the mean gradient, ~15-20x the centred part, so centring cancels),
     float32 centred FIRST, and float64 (the reference). For each: the spectrum, the
     step, and the equation residual ||A A^T d - (rhs - damping d - mean d)|| / ||eps||
     that an exact solve makes 0. Host numpy; needs ~4 GB of host RAM (-c 4).
  F4d: section 1 also runs the module step with gram_center_first, and 4b adds the
     F4d kernel (sr_kernel.py), which should match "rows f32, centre first".
  6. With --dtype float64 (second run): the whole step in float64, compared with the
     float32 run's saved vectors (--compare). Does NOT fit on a 2080 Ti (local
     energies alone need 9 GB); run it on CPU if at all. 4b answers the same question.

Usage (Savio GPU node; see the srun line in docs/CAMPAIGN_LOG.md, F3b RESULT):
    python slurm/f4_one_step_check.py --dtype float32 --out f4_e15s6_f32.npz
    python slurm/f4_one_step_check.py --dtype float64 --out f4_e15s6_f64.npz \
        --compare f4_e15s6_f32.npz
"""

import argparse
import os

USER = os.environ.get("USER", "")
DEFAULT_LOGDIR = (
    f"/global/scratch/users/{USER}/vmcnet_logs/phase_e/e15_n2_seedcheck/"
    "e15_N2_4.0_ssu_defaults_s6"
)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--logdir", default=DEFAULT_LOGDIR, help="original run (config)")
    ap.add_argument("--ckpt", default="checkpoints/60000.npz")
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--nperm", type=int, default=3, help="walker permutations")
    ap.add_argument(
        "--floor", type=float, default=0.0, help="sub-floor threshold; 0 = |min eig|"
    )
    ap.add_argument("--out", default=None, help="save vectors here (.npz)")
    ap.add_argument("--compare", default=None, help="float32 .npz to compare against")
    ap.add_argument("--no-jacobian", action="store_true", help="skip section 4b")
    ap.add_argument("--chunk", type=int, default=20, help="walkers per gradient batch")
    ap.add_argument("--block", type=int, default=20000, help="params per Gram block")
    args = ap.parse_args()

    import jax

    jax.config.update("jax_enable_x64", args.dtype == "float64")
    import jax.numpy as jnp
    import numpy as np
    from jax.flatten_util import ravel_pytree

    import vmcnet.models as models
    import vmcnet.physics as physics
    import vmcnet.utils as utils
    from vmcnet.train import runners
    from vmcnet.updates import same_sampled_spring_unified as ssu
    from vmcnet.updates.sr_kernel import get_sr_kernel_fn

    import neural_tangents as nt  # after vmcnet, see the SS-SPRING unit tests

    dtype = jnp.float64 if args.dtype == "float64" else jnp.float32
    config = utils.io.load_config_dict(args.logdir, "config.json")
    opt_cfg = config.vmc.optimizer.same_sampled_spring_unified
    damping = float(opt_cfg.damping)

    directory, name = os.path.split(os.path.join(args.logdir, args.ckpt))
    epoch, data, params, opt, _ = utils.io.reload_vmc_state(directory, name)
    core = opt.core if hasattr(opt, "core") else opt

    def cast(tree):
        return jax.tree_util.tree_map(
            lambda x: (
                jnp.asarray(x, dtype)
                if jnp.issubdtype(jnp.asarray(x).dtype, jnp.floating)
                else jnp.asarray(x)
            ),
            tree,
        )

    params, core = cast(params), cast(core)
    pos_raw = np.asarray(data["walker_data"]["position"])
    pos = jnp.asarray(pos_raw.reshape((-1,) + pos_raw.shape[-2:]), dtype)
    n = pos.shape[0]
    beta = core.beta
    print(
        f"checkpoint {args.logdir}/{args.ckpt}: epoch {epoch}, {n} walkers, "
        f"dtype {args.dtype}, beta {float(beta):.6f}, damping {damping:g}"
    )

    ion_pos, ion_charges, nelec = runners._get_electron_ion_config_as_arrays(
        config, dtype=dtype
    )
    slog_psi = models.construct.get_model_from_config(
        config.model, nelec, ion_pos, ion_charges, dtype=dtype
    )
    log_psi_apply = models.construct.slog_psi_to_log_psi_apply(slog_psi.apply)
    local_energy_fn = runners._assemble_mol_local_energy_fn(
        ion_pos,
        ion_charges,
        config.problem.ei_softening,
        config.problem.ee_softening,
        log_psi_apply,
    )
    energy_fn = physics.core.create_energy_and_statistics_fn(
        local_energy_fn, n, runners._get_clipping_fn(config.vmc), config.vmc.nan_safe
    )
    energy, local_e, stats = jax.jit(energy_fn)(params, pos)
    print(
        f"energy {float(energy):.4f} (noclip {float(stats['energy_noclip']):.4f}), "
        f"variance noclip {float(stats['variance_noclip']):.4g}"
    )
    centered = local_e - energy

    kernel_fn = nt.empirical_kernel_fn(log_psi_apply, vmap_axes=0, trace_axes=())
    f4d_kernel_fn = get_sr_kernel_fn(log_psi_apply, center_first=True)
    phi_flat, unravel = ravel_pytree(core.phi)

    # ---- 1. the module's own step, cap off vs cap path at scale 1 ----
    lr = float(opt_cfg.learning_rate)
    probe_lr = lr if opt_cfg.probe_lr < 0 else float(opt_cfg.probe_lr)
    print("\n[1] module step from the checkpoint state")
    for label, cap, cf in (
        ("cap off", 0.0, False),
        ("cap path, K=1e30 (inactive)", 1e30, False),
        ("F4d gram_center_first", 0.0, True),
    ):
        step_fn = ssu.get_same_sampled_spring_unified_step(
            log_psi_apply,
            lambda t: lr,
            damping,
            opt_cfg.probe_damping,
            int(opt_cfg.lb_window),
            probe_lr,
            bool(opt_cfg.adaptive_eta),
            bool(opt_cfg.adaptive_probe),
            return_diagnostics=True,
            carried_cap=cap,
            gram_center_first=cf,
        )
        _, _, diag = jax.jit(step_fn)(centered, params, pos, core)
        print(
            f"    {label:30s} ||phi_new|| {float(diag['diag_phi_norm']):9.4g}  "
            f"bound ratio {float(diag['diag_bound_ratio']):.4g}  carried/eps "
            f"{float(diag['diag_carried_over_eps']):.4g}  min eig "
            f"{float(diag['diag_gram_min_eig_preclip']):.3g}"
        )

    # ---- shared pieces ----
    def gram(params_, pos_, kfn=kernel_fn):
        m = pos_.shape[0]
        t = kfn(pos_, pos_, "ntk", params_) / m
        t = t - jnp.mean(t, axis=0, keepdims=True)
        t = t - jnp.mean(t, axis=1, keepdims=True)
        t = t + jnp.ones((m, m), t.dtype) / m
        return (t + t.T) / 2

    @jax.jit
    def ops(params_, pos_, centered_, phi_):
        sqrt_m = jnp.sqrt(pos_.shape[0])
        apply_A, _, tvals, tvecs, _ = ssu._build_operators_with_stats(
            kernel_fn, log_psi_apply, params_, pos_
        )
        eps = centered_ / sqrt_m
        carried = apply_A(unravel(beta * phi_))
        rhs = eps - carried
        tv = jnp.maximum(tvals, 0.0)
        dual_ref = tvecs @ jnp.diag(1.0 / (tv + damping)) @ tvecs.T @ rhs
        return dict(
            tvals=tvals,
            tvecs=tvecs,
            rhs=rhs,
            dual_ref=dual_ref,
            eps_norm=jnp.linalg.norm(eps),
            carried_norm=jnp.linalg.norm(carried),
            gram=gram(params_, pos_),
        )

    @jax.jit
    def apply_AT_flat(params_, pos_, v):
        _, vjp_fn = jax.vjp(log_psi_apply, params_, pos_)
        dtheta = vjp_fn(v - jnp.mean(v))[0]
        return ravel_pytree(dtheta)[0] / jnp.sqrt(pos_.shape[0])

    @jax.jit
    def apply_A_flat(params_, pos_, vec):
        _, jvp_fn = jax.linearize(log_psi_apply, params_, pos_)
        o = jvp_fn(unravel(vec), jnp.zeros_like(pos_)) / jnp.sqrt(pos_.shape[0])
        return o - jnp.mean(o)

    def split_solve(o, floor, eig=None):
        """Duals of the sub-floor and above-floor parts (host float64 arithmetic)."""
        tvals, tvecs = eig if eig is not None else (o["tvals"], o["tvecs"])
        tvals = np.asarray(tvals, np.float64)
        tvecs = np.asarray(tvecs, np.float64)
        rhs = np.asarray(o["rhs"], np.float64)
        w = (tvecs.T @ rhs) / (np.maximum(tvals, 0.0) + damping)
        sub = tvals < floor
        return (
            tvecs @ np.where(sub, w, 0.0),
            tvecs @ np.where(sub, 0.0, w),
            int(sub.sum()),
        )

    def pieces(params_, pos_, o, floor, eig=None):
        d_sub, d_hi, n_sub = split_solve(o, floor, eig)
        phi_sub = apply_AT_flat(params_, pos_, jnp.asarray(d_sub, dtype))
        phi_hi = apply_AT_flat(params_, pos_, jnp.asarray(d_hi, dtype))
        return np.asarray(phi_sub), np.asarray(phi_hi), n_sub

    nrm = np.linalg.norm
    beta_phi = float(beta) * np.asarray(phi_flat)

    # ---- 2. decomposition at the reference walker order ----
    ref = ops(params, pos, centered, phi_flat)
    tvals = np.asarray(ref["tvals"])
    floor = args.floor if args.floor > 0 else abs(float(tvals.min()))
    if args.compare and args.floor == 0:
        floor = float(np.load(args.compare)["floor"])  # same split as the f32 run
    phi_new = np.asarray(apply_AT_flat(params, pos, ref["dual_ref"])) + beta_phi
    phi_sub, phi_hi, n_sub = pieces(params, pos, ref, floor)
    eps_norm = float(ref["eps_norm"])
    print(
        f"\n[2] Gram spectrum: max {tvals.max():.4g}, min {tvals.min():.3g}, "
        f"# < damping {int((tvals < damping).sum())}, # < 0 {int((tvals < 0).sum())}"
    )
    carried_over_eps = float(ref["carried_norm"]) / eps_norm
    print(
        f"    floor = {floor:.3g}: {n_sub} of {n} eigendirections are sub-floor; "
        f"||eps|| {eps_norm:.4g}, carried/eps {carried_over_eps:.4g}"
    )
    print(
        f"    ||beta phi_old|| {nrm(beta_phi):.4g}   ||phi_sub|| {nrm(phi_sub):.4g}   "
        f"||phi_hi|| {nrm(phi_hi):.4g}   ||phi_new|| {nrm(phi_new):.4g}"
    )
    gap = nrm(beta_phi + phi_sub + phi_hi - phi_new) / max(nrm(phi_new), 1e-30)
    print(f"    check: |beta phi + phi_sub + phi_hi - phi_new| / |phi_new| = {gap:.2g}")
    print(
        f"    estimate ||eps|| sqrt(floor) / damping = "
        f"{eps_norm * np.sqrt(floor) / damping:.4g}"
    )

    # ---- 3. walker permutations (identical in exact arithmetic) ----
    print(f"\n[3] {args.nperm} walker permutations vs the reference order")
    rng = np.random.default_rng(0)
    for i in range(args.nperm):
        perm = jnp.asarray(rng.permutation(n))
        o = ops(params, pos[perm], centered[perm], phi_flat)
        pn = np.asarray(apply_AT_flat(params, pos[perm], o["dual_ref"])) + beta_phi
        ps, ph, _ = pieces(params, pos[perm], o, floor)
        print(
            f"    perm {i}: ||phi_new|| {nrm(pn):9.4g}  ||d phi_new|| "
            f"{nrm(pn - phi_new):9.4g}  ||d phi_sub|| {nrm(ps - phi_sub):9.4g}  "
            f"||d phi_hi|| {nrm(ph - phi_hi):9.4g}  min eig "
            f"{float(o['tvals'].min()):.3g}"
        )

    # ---- 4. float64 eigensolve of this run's Gram ----
    t64 = np.asarray(ref["gram"], np.float64)
    vals64, vecs64 = np.linalg.eigh(t64)
    ps64, ph64, n_sub64 = pieces(params, pos, ref, floor, eig=(vals64, vecs64))
    w64 = (vecs64.T @ np.asarray(ref["rhs"], np.float64)) / (
        np.maximum(vals64, 0.0) + damping
    )
    pn64 = (
        np.asarray(apply_AT_flat(params, pos, jnp.asarray(vecs64 @ w64, dtype)))
        + beta_phi
    )
    print(
        f"\n[4] float64 eigh of this {args.dtype} Gram: min eig {vals64.min():.3g}, "
        f"# < 0 {int((vals64 < 0).sum())}, # < floor {n_sub64}"
    )
    print(
        f"    ||phi_new|| {nrm(pn64):.4g} (vs {nrm(phi_new):.4g})  ||phi_sub|| "
        f"{nrm(ps64):.4g}  ||phi_hi|| {nrm(ph64):.4g}  ||d phi_hi|| vs [2] "
        f"{nrm(ph64 - phi_hi):.4g}"
    )

    # ---- 4b. Gram from explicit gradient rows: centring order and precision ----
    if not args.no_jacobian:
        rhs64 = np.asarray(ref["rhs"], np.float64)

        def eq_resid(d):
            img = apply_A_flat(params, pos, apply_AT_flat(params, pos, jnp.asarray(d)))
            d = np.asarray(d, np.float64)
            target = rhs64 - damping * d - d.mean()
            return nrm(np.asarray(img, np.float64) - target) / eps_norm

        def solve_with(t):
            vals, vecs = np.linalg.eigh(np.asarray(t, np.float64))
            w = (vecs.T @ rhs64) / (np.maximum(vals, 0.0) + damping)
            return vals, vecs @ w

        grad_rows = jax.jit(
            jax.vmap(
                lambda x, p_: ravel_pytree(
                    jax.grad(lambda q: log_psi_apply(q, x[None])[0])(p_)
                )[0],
                in_axes=(0, None),
            )
        )
        rows = np.concatenate(
            [
                np.asarray(grad_rows(pos[i : i + args.chunk], params))
                for i in range(0, n, args.chunk)
            ]
        )
        mean64 = rows.mean(axis=0, dtype=np.float64)

        def gram_from_rows(center_first, dt):
            t = np.zeros((n, n), dt)
            for k in range(0, rows.shape[1], args.block):
                b = rows[:, k : k + args.block].astype(dt)
                if center_first:
                    b = b - mean64[k : k + args.block].astype(dt)
                t += b @ b.T
            if not center_first:
                t = t - t.mean(axis=0, keepdims=True)
                t = t - t.mean(axis=1, keepdims=True)
            return t / dt(n) + dt(1.0) / dt(n)

        variants = [
            ("module kernel (f32)", np.asarray(ref["gram"])),
            (
                "F4d kernel (f32)",
                np.asarray(
                    jax.jit(lambda q, x: gram(q, x, f4d_kernel_fn))(params, pos)
                ),
            ),
            ("rows f32, centre after", gram_from_rows(False, np.float32)),
            ("rows f32, centre first", gram_from_rows(True, np.float32)),
            ("rows f64 (reference)", gram_from_rows(True, np.float64)),
        ]
        del rows
        _, d_ref = solve_with(variants[-1][1])
        step_ref = np.asarray(apply_AT_flat(params, pos, jnp.asarray(d_ref, dtype)))
        print(
            f"\n[4b] Gram from explicit gradient rows ({len(mean64)} params); "
            f"step and equation residual per variant (residual 0 = exact solve):"
        )
        for label, t in variants:
            vals, d = solve_with(t)
            step = np.asarray(apply_AT_flat(params, pos, jnp.asarray(d, dtype)))
            print(
                f"    {label:24s} min eig {vals.min():10.3g}  # < 0 "
                f"{int((vals < 0).sum()):4d}  # < damping "
                f"{int((vals < damping).sum()):4d}  ||phi_new|| "
                f"{nrm(step + beta_phi):8.4g}  ||d step vs f64|| "
                f"{nrm(step - step_ref):8.4g}  eq. residual "
                f"{eq_resid(jnp.asarray(d, dtype)):8.4g}"
            )
        print(f"    (||step|| of the f64 reference: {nrm(step_ref):.4g})")

    # ---- 5. held-out walkers ----
    h = n // 2
    fit, held = pos[:h], pos[h:]
    c_fit = local_e[:h] - jnp.mean(local_e[:h])
    c_held = local_e[h:] - jnp.mean(local_e[h:])
    o = ops(params, fit, c_fit, phi_flat)
    ps_f, ph_f, n_sub_f = pieces(params, fit, o, floor)
    pn_f = np.asarray(apply_AT_flat(params, fit, o["dual_ref"])) + beta_phi
    eps_held = float(jnp.linalg.norm(c_held / jnp.sqrt(n - h)))
    print(
        f"\n[5] solve on walkers 0-{h - 1} ({n_sub_f} sub-floor dirs), apply to "
        f"walkers {h}-{n - 1}:"
    )
    for label, vec in (("phi_sub", ps_f), ("phi_hi", ph_f)):
        a_fit = float(jnp.linalg.norm(apply_A_flat(params, fit, jnp.asarray(vec))))
        a_held = float(jnp.linalg.norm(apply_A_flat(params, held, jnp.asarray(vec))))
        print(
            f"    {label:8s} ||vec|| {nrm(vec):9.4g}  ||A_fit vec|| {a_fit:9.4g}  "
            f"||A_held vec|| {a_held:9.4g}  ratio {a_held / max(a_fit, 1e-30):.3g}"
        )
    carried_next = float(
        jnp.linalg.norm(apply_A_flat(params, held, jnp.asarray(float(beta) * pn_f)))
    )
    print(
        f"    next-step carried/eps on the held-out walkers: "
        f"{carried_next / eps_held:.4g}"
    )

    # ---- 6. save / compare ----
    if args.out:
        np.savez(
            args.out,
            floor=floor,
            tvals=tvals,
            phi_new=phi_new,
            phi_sub=phi_sub,
            phi_hi=phi_hi,
        )
        print(f"\nsaved {args.out}")
    if args.compare:
        other = np.load(args.compare)
        print(f"\n[6] this {args.dtype} run vs {args.compare}:")
        mine = dict(phi_new=phi_new, phi_sub=phi_sub, phi_hi=phi_hi)
        for k, a in mine.items():
            b = other[k]
            print(
                f"    {k:8s} ||this|| {nrm(a):9.4g}  ||other|| {nrm(b):9.4g}  "
                f"||diff|| {nrm(a - b):9.4g}"
            )


if __name__ == "__main__":
    main()
