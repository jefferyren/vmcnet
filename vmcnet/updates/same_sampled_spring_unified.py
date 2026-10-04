"""Same-Sampled SPRING with an interleaved same-sampled SPRING probe.

JAX/vmcnet port of the PyTorch reference ``same_sampled_spring_unified``. The main
update is ordinary SPRING (see ``vmcnet/updates/spring.py``) with a dynamic decay
factor ``beta``; a probe runs a second SPRING solve on the *same* sampled Jacobian
against a fixed synthetic target ``b = A @ x_star`` and its residual history drives an
adaptive-``beta`` schedule. See
``docs/superpowers/specs/2026-07-02-same-sampled-spring-unified-design.md``.
"""

from typing import Callable, NamedTuple

import chex
import jax
import jax.flatten_util
import jax.numpy as jnp
import neural_tangents as nt  # type: ignore
import optax
from ml_collections import ConfigDict

import vmcnet.utils as utils
from vmcnet.updates.update_param_fns import (
    UpdateParamFn,
    get_update_norm_diagnostics,
    make_traced_fn_with_single_metrics,
    update_metrics_with_noclip,
)
from vmcnet.utils.distribute import pmean_if_pmap
from vmcnet.utils.pytree_helpers import (
    multiply_tree_by_scalar,
    tree_inner_product,
    tree_reduce_l1,
)
from vmcnet.utils.typing import (
    Array,
    D,
    GetPositionFromData,
    LearningRateSchedule,
    ModelApply,
    P,
    PRNGKey,
    PyTree,
    S,
    Tuple,
    UpdateDataFn,
)


class SameSampledSPRINGUnifiedState(NamedTuple):
    """Optimizer state for `same_sampled_spring_unified`.

    Attributes:
        phi: main SPRING momentum (params-shaped pytree).
        z_probe: probe iterate (params-shaped pytree).
        phi_probe: probe SPRING momentum (params-shaped pytree).
        x_star: fixed random unit-norm probe target (params-shaped pytree).
        residual_buffer: chronological sliding buffer (float[2p]) of probe residuals.
        r_hat: running convergence-rate estimate (scalar).
        beta: current decay factor / momentum (scalar).
        checkpoint_idx: adaptive-beta checkpoint counter (int scalar, starts at 1).
        step: step/epoch counter (int scalar, starts at 0).
        r_ip: raw (pre-clip) windowed residual-contraction ratio from the most
            recent step (scalar, diagnostic only). Values > 1 mean the probe
            residual is growing — the regime where min(1, r_ip) clipping can
            pin r_hat at 1 and drive beta toward 1.
    """

    phi: PyTree
    z_probe: PyTree
    phi_probe: PyTree
    x_star: PyTree
    residual_buffer: Array
    r_hat: Array
    beta: Array
    checkpoint_idx: Array
    step: Array
    r_ip: Array


def _draw_unit_norm_like(key: PRNGKey, params: P) -> P:
    """Draw a random pytree like `params`, normalized to global unit L2 norm."""
    flat, unravel = jax.flatten_util.ravel_pytree(params)
    noise = jax.random.normal(key, flat.shape, dtype=flat.dtype)
    noise = noise / jnp.linalg.norm(noise)
    return unravel(noise)


def _build_operators(
    kernel_fn: Callable,
    log_psi_apply: ModelApply[P],
    params: P,
    positions: Array,
) -> Tuple[Callable[[P], Array], Callable[[Array], P], Array, Array]:
    """Build the shared SPRING linear operators (see _build_operators_with_stats)."""
    apply_A, apply_AT, tvals, tvecs, _ = _build_operators_with_stats(
        kernel_fn, log_psi_apply, params, positions
    )
    return apply_A, apply_AT, tvals, tvecs


def _build_operators_with_stats(
    kernel_fn: Callable,
    log_psi_apply: ModelApply[P],
    params: P,
    positions: Array,
) -> Tuple[Callable[[P], Array], Callable[[Array], P], Array, Array, dict]:
    """Build the shared SPRING linear operators at the current (params, positions).

    The main solve and the probe solve must use identical operators, so they are built
    once here and reused. `apply_A` centers its output and `apply_AT` centers its input
    (they are transposes of one another); the constant-mode term `(1/N) ones onesᵀ` is
    included in `T`, but the negative-eigenvalue clip and damping are deferred to the
    consumer of the returned eigendecomposition (not applied in this function).

    Args:
        kernel_fn: neural-tangents empirical kernel, built once from `log_psi_apply`.
        log_psi_apply: maps (params, positions) -> log|psi| of shape (nchains,).
        params: current model parameters.
        positions: current walker positions, shape (nchains, ...).

    Returns:
        Tuple (apply_A, apply_AT, Tvals, Tvecs, stats) where apply_A:
        P -> Array[nchains] (mean-zero), apply_AT: Array[nchains] -> P, Tvals/Tvecs are the
        eigendecomposition of T = A Aᵀ + (1/N) ones onesᵀ, and stats holds two
        diagnostics of the kernel (unused, hence compiled away, unless logged):
        "diag_mean_jac_sq_over_trace" = ||mean_i O_i||^2 / tr(centered kernel),
        the size of the cancellation when the uncentered kernel is centered, and
        "diag_walker_rownorm_max_over_median" = max/median of the per-walker
        centered squared Jacobian norms (the diagonal of the centered kernel).
    """
    nchains = positions.shape[0]
    sqrt_n = jnp.sqrt(nchains)
    ones = jnp.ones((nchains, 1))

    # Reuse a single linearization / vjp so the multiple probe + main calls are cheap.
    _, jvp_fn = jax.linearize(log_psi_apply, params, positions)
    zeros_positions = jnp.zeros_like(positions)

    def apply_A(vec: P) -> Array:
        o = jvp_fn(vec, zeros_positions) / sqrt_n
        return o - jnp.mean(o, axis=0, keepdims=True)

    _, vjp_fn = jax.vjp(log_psi_apply, params, positions)

    def apply_AT(v: Array) -> P:
        v_hat = v - jnp.mean(v)
        dtheta = vjp_fn(v_hat)[0]
        return multiply_tree_by_scalar(dtheta, 1.0 / sqrt_n)

    t = kernel_fn(positions, positions, "ntk", params) / nchains
    uncentered_trace = jnp.trace(t)
    t = t - jnp.mean(t, axis=0, keepdims=True)
    t = t - jnp.mean(t, axis=1, keepdims=True)
    row_sq = jnp.diag(t)
    centered_trace = jnp.sum(row_sq)
    stats = {
        "diag_mean_jac_sq_over_trace": (uncentered_trace - centered_trace)
        / centered_trace,
        "diag_walker_rownorm_max_over_median": jnp.max(row_sq) / jnp.median(row_sq),
    }
    t = t + ones @ ones.T / nchains
    t = (t + t.T) / 2
    tvals, tvecs = jnp.linalg.eigh(t)
    return apply_A, apply_AT, tvals, tvecs, stats


def _adaptive_beta_update(
    residual_buffer: Array,
    probe_res_norm: Array,
    r_hat: Array,
    beta: Array,
    checkpoint_idx: Array,
    step: Array,
    p: int,
) -> Tuple[Array, Array, Array, Array, Array]:
    """Slide the probe residual into the buffer and update beta on schedule.

    The residual buffer is chronological (oldest first), length 2p. Every p steps once
    `step >= 2p`, beta is updated from the ratio of squared-residual sums over the two
    adjacent length-p windows (Appendix D of the reference). Off-schedule steps only
    slide the buffer. All arithmetic runs every step; writes are gated with `jnp.where`
    (a Python branch would be illegal under jit).

    Args:
        residual_buffer: chronological float buffer of length 2p.
        probe_res_norm: the current step's probe residual norm (scalar).
        r_hat: running convergence-rate estimate (scalar).
        beta: current decay factor (scalar).
        checkpoint_idx: adaptive-beta checkpoint counter (int scalar, >= 1).
        step: current step index (int scalar, pre-increment).
        p: lookback window length.

    Returns:
        Tuple (new_buffer, new_r_hat, new_beta, new_checkpoint_idx, r_ip), where
        r_ip is the RAW (pre-min-clip) window ratio computed this step. It is
        returned every step as a diagnostic (only trigger steps feed it into
        r_hat); values > 1 flag a growing probe residual. While step < 2p the
        buffer is partially zero and the ratio is meaningless (up to res^2/1e-30),
        so the returned diagnostic is gated to 1.0 there; the internal r_hat
        update is unaffected (it is already gated to trigger steps >= 2p).
    """
    new_buffer = jnp.concatenate([residual_buffer[1:], probe_res_norm[None]])

    window_tp = new_buffer[0:p]
    window_t = new_buffer[p : 2 * p]
    # small epsilon guards the early (partially-zero) buffer; gated out anyway.
    r_ip = jnp.sum(window_t**2) / (jnp.sum(window_tp**2) + 1e-30)

    n_old = checkpoint_idx.astype(r_hat.dtype)
    n_new = n_old + 1.0
    alph = jnp.power(n_old, jnp.log(n_old)) / jnp.power(n_new, jnp.log(n_new))

    r_hat_new = alph * r_hat + (1.0 - alph) * jnp.minimum(1.0, r_ip)
    rho = jnp.clip(1.0 - jnp.power(r_hat_new, 1.0 / p), min=0.0)
    beta_new = (1.0 - rho) / (1.0 + rho)

    do_update = (step % p == 0) & (step >= 2 * p)
    new_beta = jnp.where(do_update, beta_new, beta)
    new_r_hat = jnp.where(do_update, r_hat_new, r_hat)
    new_checkpoint_idx = jnp.where(do_update, checkpoint_idx + 1, checkpoint_idx)
    r_ip_diag = jnp.where(step >= 2 * p, r_ip, jnp.ones_like(r_ip))
    return new_buffer, new_r_hat, new_beta, new_checkpoint_idx, r_ip_diag


def _adaptive_eta_main(
    lr: Array, base_lr: float, beta: Array, adaptive_eta: bool
) -> Array:
    """Main step size, applying the schedule's decay as an OUTER factor.

    When ``adaptive_eta`` is False this is just ``lr`` (== base SPRING). When True it is
    ``decay(t) * (1 - beta * (1 - eta0))`` where ``eta0 = base_lr`` is the base learning
    rate (``lr`` at ``t == 0``) and ``decay(t) = lr / base_lr`` is the schedule's decay
    factor. Applying the decay OUTSIDE the ``1 - beta * (1 - .)`` term keeps the step
    annealing toward 0 under a decaying (e.g. inverse_time) schedule, and -- once beta
    stabilizes -- decaying in step with the schedule. Plugging the already-decayed
    ``lr`` INSIDE the term instead would plateau at ``1 - beta`` as ``lr -> 0``.
    Under a constant schedule ``decay(t) == 1``, so this reduces to
    ``1 - beta * (1 - eta0)`` (the PINN reference's adaptive-eta form).

    Args:
        lr: the scheduled learning rate at the current step, ``lr(t)``.
        base_lr: the base learning rate ``eta0 == lr(0)`` (a static scalar).
        beta: the current (dynamic) decay factor / momentum.
        adaptive_eta: if True, use the outer-decay adaptive form; else return ``lr``.

    Returns:
        The main step size ``eta_main``.
    """
    if not adaptive_eta:
        return lr
    decay_factor = lr / base_lr
    return decay_factor * (1.0 - beta * (1.0 - base_lr))


def get_same_sampled_spring_unified_step(
    log_psi_apply: ModelApply[P],
    learning_rate_schedule: LearningRateSchedule,
    damping: chex.Scalar,
    probe_damping: chex.Scalar,
    p: int,
    probe_lr: float,
    adaptive_eta: bool,
    adaptive_probe: bool,
    return_diagnostics: bool = False,
    carried_cap: float = 0.0,
) -> Callable[
    [Array, P, Array, SameSampledSPRINGUnifiedState],
    Tuple,
]:
    """Get the same_sampled_spring_unified step kernel.

    Returns a pure function `step(centered_local_energies, params, positions, state)`
    that returns `(updates, new_state)`, where `updates` is the (unconstrained)
    parameter delta already scaled by `-eta_main`. The main SPRING solve and the probe
    solve reuse the same operators and single eigendecomposition; only their additive
    damping differs.

    With `return_diagnostics=True` it returns `(updates, new_state, diag)`, where diag
    is a dict of scalar diagnostics of the main solve (all prefixed "diag_"):

    - bound_ratio: ||phi_k|| / (beta ||phi_{k-1}|| + ||eps|| / (2 sqrt(damping))). In
      exact arithmetic this is <= 1 (0 <= P_k < I); larger values are float32 error.
    - equation_residual: ||A phi_k - (eps - damping * dual)|| / ||eps||, the residual of
      the identity A phi_k = eps - damping (T + damping)^-1 rhs (one extra jvp).
    - carried_over_eps: ||A (beta phi_{k-1})|| / ||eps||, the size of the carried
      momentum on this step's (fresh) walkers.
    - gram_lam_max, gram_min_eig_preclip, gram_trace, gram_n_eig_below_damping.
    - phi_norm, eps_norm, and the two kernel stats from _build_operators_with_stats.
    - carried_scale (only with carried_cap > 0): the factor applied to the carried
      momentum this step (1 = cap inactive). carried_over_eps is reported BEFORE it.

    With `carried_cap = K > 0` (Phase F step F3b), whenever the carried momentum seen
    on this step's walkers exceeds K times the target, ||A(beta phi)|| > K ||eps||, the
    momentum buffer entering the step is shrunk so that ||A(beta phi)|| = K ||eps||.
    Because A is linear this costs nothing extra; it is equivalent to using momentum
    beta * scale for this one step. K = 0 disables it (the default, unchanged path).

    Args:
        log_psi_apply: maps (params, positions) -> log|psi|, shape (nchains,).
        learning_rate_schedule: step -> base learning rate.
        damping: additive damping for the main normal equation.
        probe_damping: additive damping for the probe normal equation.
        p: adaptive-beta lookback window.
        probe_lr: base probe step (used when adaptive_probe is False).
        adaptive_eta: if True, eta_main = decay(t) * (1 - beta*(1 - eta0)) with
            eta0 = lr(0) and decay(t) = lr(t)/eta0 (see _adaptive_eta_main); else lr(t).
        adaptive_probe: if True, the probe step uses eta_main instead of probe_lr.
        return_diagnostics: if True, also return the diag dict described above.
        carried_cap: K for the carried-momentum cap; 0 disables it.

    Returns:
        The step kernel described above.
    """
    kernel_fn = nt.empirical_kernel_fn(log_psi_apply, vmap_axes=0, trace_axes=())
    # eta0: the base learning rate (lr at t=0), used by the adaptive-eta scheme to
    # apply the schedule's decay as an outer factor. Evaluated once, eagerly.
    base_lr = float(learning_rate_schedule(jnp.array(0)))

    def step(
        centered_local_energies: Array,
        params: P,
        positions: Array,
        state: SameSampledSPRINGUnifiedState,
    ) -> Tuple:
        nchains = positions.shape[0]
        sqrt_n = jnp.sqrt(nchains)
        beta = state.beta

        apply_A, apply_AT, tvals, tvecs, kernel_stats = _build_operators_with_stats(
            kernel_fn, log_psi_apply, params, positions
        )
        tvals_clipped = jnp.maximum(tvals, 0.0)

        def solve(rhs: Array, damp: chex.Scalar) -> Array:
            # Mirrors the exact operation order of base SPRING's
            # `Tvecs @ jnp.diag(1 / Tvals) @ Tvecs.T @ epsilon_tilde` (spring.py:175):
            # the mathematically-equivalent (Tvecs.T @ rhs) / Tvals two-matvec form
            # takes a different float32 rounding path and can drift from base SPRING
            # by more than this module's equivalence tolerance on ill-conditioned T.
            return tvecs @ jnp.diag(1.0 / (tvals_clipped + damp)) @ tvecs.T @ rhs

        # ---- main SPRING (identical to base SPRING; beta is dynamic, from state) ----
        epsilon_bar = centered_local_energies / sqrt_n
        carried = apply_A(multiply_tree_by_scalar(state.phi, beta))
        carried_raw_norm = jnp.linalg.norm(carried)
        if carried_cap > 0:
            # F3b: shrink the carried momentum to at most K x the target on fresh walkers
            carried_scale = jnp.minimum(
                1.0,
                carried_cap * jnp.linalg.norm(epsilon_bar) / (carried_raw_norm + 1e-30),
            )
            carried = carried * carried_scale
            beta_main = beta * carried_scale
        else:
            beta_main = beta
        rhs_main = epsilon_bar - carried
        dual_main = solve(rhs_main, damping)
        step_primal = apply_AT(dual_main)
        phi_new = jax.tree_map(lambda s, ph: s + beta_main * ph, step_primal, state.phi)

        # eta_main: with adaptive_eta, the schedule's decay is applied as an OUTER
        # factor -- eta_main = decay(t) * (1 - beta*(1 - eta0)) -- so the step still
        # anneals toward 0 under a decaying schedule (rather than plateauing at 1-beta).
        # Without adaptive_eta this is just lr(step), i.e. base SPRING. See
        # _adaptive_eta_main for the full rationale.
        lr = learning_rate_schedule(state.step)
        eta_main = _adaptive_eta_main(lr, base_lr, beta, adaptive_eta)
        updates = multiply_tree_by_scalar(phi_new, -eta_main)

        # ---- probe: self-contained synthetic solve on the SAME operators ----
        # probe_lr defaults to the base learning_rate value (resolved in the
        # initializer); when adaptive_probe is set, the probe uses eta_main instead.
        # When adaptive_probe is False, eta_probe == probe_lr is a FIXED scalar for
        # every step (not re-evaluated against the schedule), so under an
        # inverse_time schedule the main step's eta_main decays over training while
        # the probe step stays constant -- intended, matching the reference.
        eta_probe = eta_main if adaptive_probe else probe_lr

        # zeta_probe = (b - A z_probe) - beta*(A phi_probe)
        #            = A(x_star - z_probe - beta*phi_probe)
        probe_in = jax.tree_map(
            lambda xs, z, ph: xs - z - beta * ph,
            state.x_star,
            state.z_probe,
            state.phi_probe,
        )
        v = solve(apply_A(probe_in), probe_damping)
        w = apply_AT(v)
        phi_probe_new = jax.tree_map(lambda w_, ph: w_ + beta * ph, w, state.phi_probe)
        z_probe_new = jax.tree_map(
            lambda z, ph: z + eta_probe * ph, state.z_probe, phi_probe_new
        )

        # probe residual on the UPDATED iterate:
        # ||A z_probe_new - b|| = ||A(z_probe_new - x_star)||
        resid_in = jax.tree_map(lambda z, xs: z - xs, z_probe_new, state.x_star)
        probe_res_norm = jnp.linalg.norm(apply_A(resid_in))

        # ---- adaptive-beta ----
        new_buffer, new_r_hat, new_beta, new_ckpt, r_ip = _adaptive_beta_update(
            state.residual_buffer,
            probe_res_norm,
            state.r_hat,
            beta,
            state.checkpoint_idx,
            state.step,
            p,
        )

        new_state = SameSampledSPRINGUnifiedState(
            phi=phi_new,
            z_probe=z_probe_new,
            phi_probe=phi_probe_new,
            x_star=state.x_star,
            residual_buffer=new_buffer,
            r_hat=new_r_hat,
            beta=new_beta,
            checkpoint_idx=new_ckpt,
            step=state.step + 1,
            r_ip=r_ip,
        )
        if not return_diagnostics:
            return updates, new_state

        eps_norm = jnp.linalg.norm(epsilon_bar)
        phi_prev_norm = jnp.sqrt(tree_inner_product(state.phi, state.phi))
        phi_new_norm = jnp.sqrt(tree_inner_product(phi_new, phi_new))
        bound = beta_main * phi_prev_norm + eps_norm / (2.0 * jnp.sqrt(damping))
        residual = apply_A(phi_new) - (epsilon_bar - damping * dual_main)
        diag = {
            "diag_bound_ratio": phi_new_norm / bound,
            "diag_equation_residual": jnp.linalg.norm(residual) / eps_norm,
            "diag_carried_over_eps": carried_raw_norm / eps_norm,
            "diag_phi_norm": phi_new_norm,
            "diag_eps_norm": eps_norm,
            "diag_gram_lam_max": jnp.max(tvals),
            "diag_gram_min_eig_preclip": jnp.min(tvals),
            "diag_gram_trace": jnp.sum(tvals),
            "diag_gram_n_eig_below_damping": jnp.sum(tvals < damping).astype(
                tvals.dtype
            ),
            **kernel_stats,
        }
        if carried_cap > 0:
            diag["diag_carried_scale"] = carried_scale
        return updates, new_state, diag

    return step


def constrain_norm(grad: P, norm_constraint: chex.Numeric = 0.001) -> P:
    """Euclidean norm constraint on the update (matches spring.py).

    Args:
        grad: the (params-shaped) update to constrain.
        norm_constraint: the maximum allowed squared L2 norm.

    Returns:
        `grad`, rescaled down (never up) so its global squared L2 norm is at most
        `norm_constraint`.
    """
    sq_norm_scaled_grads = tree_inner_product(grad, grad)
    sq_norm_scaled_grads = pmean_if_pmap(sq_norm_scaled_grads)
    norm_scale_factor = jnp.sqrt(norm_constraint / sq_norm_scaled_grads)
    coefficient = jnp.minimum(norm_scale_factor, 1)
    return multiply_tree_by_scalar(grad, coefficient)


class SafeguardState(NamedTuple):
    """State of the optional detect-skip-rewind safeguard (Phase F step F3).

    Attributes:
        log_step_buffer: ring buffer (float[window]) of log pre-clip squared step
            norms from accepted (non-triggered) steps.
        n_logged: number of entries ever pushed into the buffer (int scalar).
        last_trigger: step index of the most recent trigger (int scalar).
        hold_until: beta is capped at `beta_cap` while step < hold_until (int scalar).
        n_triggers: total triggers (skips + rewinds) so far (int scalar).
        n_rewinds: total rewinds so far (int scalar).
        snap_a_params, snap_b_params: two alternating parameter snapshots.
        snap_a_z_probe, snap_b_z_probe: the probe iterate at those snapshots.
        snap_a_step, snap_b_step: the step index at which each snapshot was taken.
    """

    log_step_buffer: Array
    n_logged: Array
    last_trigger: Array
    hold_until: Array
    n_triggers: Array
    n_rewinds: Array
    snap_a_params: PyTree
    snap_b_params: PyTree
    snap_a_z_probe: PyTree
    snap_b_z_probe: PyTree
    snap_a_step: Array
    snap_b_step: Array


class SafeguardedState(NamedTuple):
    """Optimizer state when the safeguard is on: the unchanged core state + guard."""

    core: SameSampledSPRINGUnifiedState
    guard: SafeguardState


class SafeguardConfig(NamedTuple):
    """Static safeguard settings, read from the optimizer config."""

    step_ratio: float  # trigger if pre-clip sq step > step_ratio * trailing median
    bound_ratio: float  # trigger if diag_bound_ratio > bound_ratio
    window: int  # trailing-median window (accepted steps)
    min_fill: int  # trailing-median criterion is active once this many are logged
    start_step: int  # no triggers before this step (init chaos)
    rewind_within: int  # a second trigger within this many steps rewinds
    snapshot_every: int  # snapshot interval; rewind target is 1-2 intervals old
    beta_cap: float  # beta cap held after a rewind
    hold_steps: int  # how long the cap is held


def _tree_where(cond: Array, a: PyTree, b: PyTree) -> PyTree:
    return jax.tree_util.tree_map(lambda x, y: jnp.where(cond, x, y), a, b)


def _init_safeguard_state(
    params: P, core: SameSampledSPRINGUnifiedState, window: int
) -> SafeguardState:
    """Fresh guard state; both snapshots start at the current params and step."""
    int_zero = jnp.zeros((), jnp.int32)
    return SafeguardState(
        log_step_buffer=jnp.zeros((window,), jnp.float32),
        n_logged=int_zero,
        last_trigger=jnp.array(-(10**9), jnp.int32),
        hold_until=jnp.array(-1, jnp.int32),
        n_triggers=int_zero,
        n_rewinds=int_zero,
        snap_a_params=params,
        snap_b_params=params,
        snap_a_z_probe=core.z_probe,
        snap_b_z_probe=core.z_probe,
        snap_a_step=core.step.astype(jnp.int32),
        snap_b_step=core.step.astype(jnp.int32),
    )


def _safeguarded_apply(
    step_fn: Callable,
    cfg: SafeguardConfig,
    centered_local_energies: Array,
    params: P,
    positions: Array,
    state: SafeguardedState,
    constrain: bool,
    norm_constraint: chex.Numeric,
) -> Tuple[P, SafeguardedState, dict]:
    """One SS-SPRING step behind the detect-skip-rewind safeguard (Phase F, F3).

    Trigger (from `start_step` on): the pre-clip squared step exceeds `step_ratio` x
    its trailing median over accepted steps, OR the exact-bound ratio exceeds
    `bound_ratio`, OR either is non-finite. Both thresholds are dimensionless and,
    at 3, never fired on any of the 17 surviving N2-4.0 runs (F1b).

    - First trigger: SKIP. Parameters, momentum and probe are left as they were; only
      the step counter advances (so the learning-rate schedule keeps moving).
    - A second trigger within `rewind_within` steps: REWIND. Parameters and the probe
      iterate go back to the older of two alternating snapshots (1-2 x
      `snapshot_every` steps old), the momentum buffers are zeroed, and beta is capped
      at `beta_cap` for `hold_steps` steps. Walkers are not rewound; MCMC
      re-equilibrates them under the restored parameters.
    """
    core, guard = state.core, state.guard
    step_idx = core.step.astype(jnp.int32)

    holding = step_idx < guard.hold_until
    beta_in = jnp.where(holding, jnp.minimum(core.beta, cfg.beta_cap), core.beta)
    updates, new_core, diag = step_fn(
        centered_local_energies, params, positions, core._replace(beta=beta_in)
    )
    opt_metrics = get_update_norm_diagnostics(updates, constrain, norm_constraint)
    preclip = opt_metrics["update_sq_norm_preclip"]
    if constrain:
        applied = constrain_norm(updates, norm_constraint)
    else:
        applied = updates
    params_normal = optax.apply_updates(params, applied)

    # trailing median over the filled part of the ring buffer
    filled = jnp.arange(cfg.window) < guard.n_logged
    med = jnp.exp(jnp.nanmedian(jnp.where(filled, guard.log_step_buffer, jnp.nan)))
    enough = guard.n_logged >= cfg.min_fill
    finite = jnp.isfinite(preclip) & jnp.isfinite(diag["diag_bound_ratio"])
    trig_step = enough & (preclip > cfg.step_ratio * med)
    trig_bound = diag["diag_bound_ratio"] > cfg.bound_ratio
    trig = (step_idx >= cfg.start_step) & (~finite | trig_step | trig_bound)
    rewind = trig & (step_idx - guard.last_trigger <= cfg.rewind_within)
    skip = trig & ~rewind

    a_older = guard.snap_a_step <= guard.snap_b_step
    old_params = _tree_where(a_older, guard.snap_a_params, guard.snap_b_params)
    old_z_probe = _tree_where(a_older, guard.snap_a_z_probe, guard.snap_b_z_probe)

    params_out = _tree_where(
        rewind, old_params, _tree_where(skip, params, params_normal)
    )

    advanced_old = core._replace(step=core.step + 1)
    zeros = jax.tree_util.tree_map(jnp.zeros_like, core.phi)
    rewound = advanced_old._replace(phi=zeros, phi_probe=zeros, z_probe=old_z_probe)
    core_out = _tree_where(rewind, rewound, _tree_where(skip, advanced_old, new_core))

    # snapshots: on accepted steps every `snapshot_every`, overwrite the older slot
    take = (~trig) & (step_idx % cfg.snapshot_every == 0)
    write_a = take & a_older
    write_b = take & ~a_older
    slot = guard.n_logged % cfg.window
    new_guard = SafeguardState(
        log_step_buffer=jnp.where(
            trig,
            guard.log_step_buffer,
            guard.log_step_buffer.at[slot].set(jnp.log(preclip).astype(jnp.float32)),
        ),
        n_logged=guard.n_logged + (~trig).astype(jnp.int32),
        last_trigger=jnp.where(trig, step_idx, guard.last_trigger),
        hold_until=jnp.where(rewind, step_idx + cfg.hold_steps, guard.hold_until),
        n_triggers=guard.n_triggers + trig.astype(jnp.int32),
        n_rewinds=guard.n_rewinds + rewind.astype(jnp.int32),
        snap_a_params=_tree_where(write_a, params, guard.snap_a_params),
        snap_b_params=_tree_where(write_b, params, guard.snap_b_params),
        snap_a_z_probe=_tree_where(write_a, core.z_probe, guard.snap_a_z_probe),
        snap_b_z_probe=_tree_where(write_b, core.z_probe, guard.snap_b_z_probe),
        snap_a_step=jnp.where(write_a, step_idx, guard.snap_a_step),
        snap_b_step=jnp.where(write_b, step_idx, guard.snap_b_step),
    )
    opt_metrics.update(diag)
    opt_metrics.update(
        {
            "sg_trigger": trig.astype(jnp.float32),
            "sg_rewind": rewind.astype(jnp.float32),
            "sg_n_triggers": new_guard.n_triggers.astype(jnp.float32),
            "sg_n_rewinds": new_guard.n_rewinds.astype(jnp.float32),
            "sg_step_median": med,
            "sg_beta_used": beta_in,
        }
    )
    return params_out, SafeguardedState(core_out, new_guard), opt_metrics


def construct_same_sampled_spring_unified_update_param_fn(
    energy_and_statistics_fn: Callable,
    optimizer_apply: Callable,
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    apply_pmap: bool = True,
    record_param_l1_norm: bool = False,
) -> UpdateParamFn[P, D, S]:
    """Create the update_param_fn for same_sampled_spring_unified.

    Args:
        energy_and_statistics_fn: (params, positions) -> (energy, local_energies,
            stats), where stats has keys "variance", "energy_noclip", and
            "variance_noclip".
        optimizer_apply: (energy, local_energies, params, optimizer_state, data) ->
            (new_params, new_optimizer_state).
        get_position_fn: gets the walker positions from the MCMC data.
        update_data_fn: function which updates data for new params.
        apply_pmap: whether to pmap (True) or jit (False) the returned function.
        record_param_l1_norm: whether to record the L1 norm of the params in metrics.

    Returns:
        Callable: function which updates the parameters given the current data,
        params, and optimizer state. The signature of this function is
            (params, data, optimizer_state, key)
            -> (new_params, new_data, new_optimizer_state, metrics, key)
        The metrics include the adaptive momentum beta (logged as "mu", matching
        PRIME-SR's key), the convergence-rate estimate r_hat, the probe residual
        norm probe_res_norm, and the raw window ratio probe_r_ip, all read from
        the new optimizer state, plus the update-norm diagnostics
        update_sq_norm_preclip and norm_cap_applied. The function is pmapped if
        apply_pmap is True, and jitted if apply_pmap is False.
    """

    def update_param_fn(params, data, optimizer_state, key):
        position = get_position_fn(data)
        energy, local_energies, stats = energy_and_statistics_fn(params, position)
        params, optimizer_state, opt_metrics = optimizer_apply(
            energy, local_energies, params, optimizer_state, data
        )
        data = update_data_fn(data, params)

        metrics = {"energy": energy, "variance": stats["variance"]}
        metrics = update_metrics_with_noclip(
            stats["energy_noclip"], stats["variance_noclip"], metrics
        )
        # The adaptive momentum beta is logged as "mu" to match PRIME-SR's key.
        # probe_res_norm is the current step's probe residual (last buffer slot);
        # probe_r_ip is the raw pre-clip window ratio (> 1 == growing residual,
        # the precursor of the beta-locks-toward-1 failure mode).
        core = (
            optimizer_state.core
            if isinstance(optimizer_state, SafeguardedState)
            else optimizer_state
        )
        metrics.update(
            {
                "mu": core.beta,
                "r_hat": core.r_hat,
                "probe_res_norm": core.residual_buffer[-1],
                "probe_r_ip": core.r_ip,
            }
        )
        metrics.update(opt_metrics)
        if record_param_l1_norm:
            metrics.update({"param_l1_norm": tree_reduce_l1(params)})
        return params, data, optimizer_state, metrics, key

    return make_traced_fn_with_single_metrics(update_param_fn, apply_pmap)


def initialize_same_sampled_spring_unified(
    log_psi_apply: ModelApply[P],
    energy_and_statistics_fn: Callable,
    params: P,
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    learning_rate_schedule: LearningRateSchedule,
    optimizer_config: ConfigDict,
    key: PRNGKey,
    record_param_l1_norm: bool = False,
    apply_pmap: bool = True,
) -> Tuple[
    UpdateParamFn[P, D, SameSampledSPRINGUnifiedState],
    SameSampledSPRINGUnifiedState,
    PRNGKey,
]:
    """Get an update param function and initial state for same_sampled_spring_unified.

    Args:
        log_psi_apply: maps (params, positions) -> log|psi|, shape (nchains,).
        energy_and_statistics_fn: (params, positions) -> (energy, local_energies,
            stats), where stats has keys "variance", "energy_noclip", and
            "variance_noclip".
        params: initial model parameters.
        get_position_fn: gets the walker positions from the MCMC data.
        update_data_fn: function which updates data for new params.
        learning_rate_schedule: step -> base learning rate.
        optimizer_config: ConfigDict with keys learning_rate, mu, damping,
            constrain_norm, norm_constraint, lb_window, probe_lr, probe_damping,
            adaptive_eta, and adaptive_probe.
        key: PRNGKey used to draw the probe target `x_star`.
        record_param_l1_norm: whether to record the L1 norm of the params in metrics.
        apply_pmap: whether to pmap (True) or jit (False) the returned update fn and
            the initial state construction.

    Returns:
        Tuple (update_param_fn, optimizer_state, key), where `key` has been advanced
        past the subkey consumed to draw `x_star`.
    """
    p = int(optimizer_config.lb_window)
    mu = float(optimizer_config.mu)

    # probe_lr defaults to the base learning_rate value; a negative config value is
    # the "unset" sentinel (mirrors the PyTorch reference, where probe_lr falls back
    # to lr).
    probe_lr = (
        float(optimizer_config.learning_rate)
        if optimizer_config.probe_lr < 0
        else float(optimizer_config.probe_lr)
    )

    # Phase F options (2026-10). Read with .get so configs written before they existed
    # still load; both default to off, which leaves the update path unchanged.
    diagnostics = bool(optimizer_config.get("diagnostics", False))
    carried_cap = float(optimizer_config.get("carried_cap", 0.0))
    safeguard = bool(optimizer_config.get("safeguard", False))
    sg_cfg = SafeguardConfig(
        step_ratio=float(optimizer_config.get("safeguard_step_ratio", 3.0)),
        bound_ratio=float(optimizer_config.get("safeguard_bound_ratio", 3.0)),
        window=int(optimizer_config.get("safeguard_window", 500)),
        min_fill=int(optimizer_config.get("safeguard_min_fill", 100)),
        start_step=int(optimizer_config.get("safeguard_start_step", 1000)),
        rewind_within=int(optimizer_config.get("safeguard_rewind_within", 100)),
        snapshot_every=int(optimizer_config.get("safeguard_snapshot_every", 250)),
        beta_cap=float(optimizer_config.get("safeguard_beta_cap", 0.99)),
        hold_steps=int(optimizer_config.get("safeguard_hold_steps", 2000)),
    )

    step_fn = get_same_sampled_spring_unified_step(
        log_psi_apply,
        learning_rate_schedule,
        optimizer_config.damping,
        optimizer_config.probe_damping,
        p,
        probe_lr,
        bool(optimizer_config.adaptive_eta),
        bool(optimizer_config.adaptive_probe),
        return_diagnostics=diagnostics or safeguard,
        carried_cap=carried_cap,
    )

    def init_state(local_params: P, subkey: PRNGKey) -> SameSampledSPRINGUnifiedState:
        zeros = jax.tree_map(jnp.zeros_like, local_params)
        return SameSampledSPRINGUnifiedState(
            phi=zeros,
            z_probe=zeros,
            phi_probe=zeros,
            x_star=_draw_unit_norm_like(subkey, local_params),
            residual_buffer=jnp.zeros(2 * p),
            r_hat=jnp.array(1.0),
            beta=jnp.array(mu),
            checkpoint_idx=jnp.array(1, jnp.int32),
            step=jnp.array(0, jnp.int32),
            r_ip=jnp.array(1.0),
        )

    def optimizer_apply(energy, local_energies, params, optimizer_state, data):
        positions = get_position_fn(data)
        centered_local_energies = local_energies - energy
        if safeguard:
            # A plain core state (fresh, or reloaded from a checkpoint written without
            # the safeguard) is upgraded on the first traced call; jit retraces once.
            if isinstance(optimizer_state, SameSampledSPRINGUnifiedState):
                optimizer_state = SafeguardedState(
                    optimizer_state,
                    _init_safeguard_state(params, optimizer_state, sg_cfg.window),
                )
            return _safeguarded_apply(
                step_fn,
                sg_cfg,
                centered_local_energies,
                params,
                positions,
                optimizer_state,
                bool(optimizer_config.constrain_norm),
                optimizer_config.norm_constraint,
            )
        if diagnostics:
            updates, optimizer_state, diag = step_fn(
                centered_local_energies, params, positions, optimizer_state
            )
        else:
            updates, optimizer_state = step_fn(
                centered_local_energies, params, positions, optimizer_state
            )
        opt_metrics = get_update_norm_diagnostics(
            updates,
            optimizer_config.constrain_norm,
            optimizer_config.norm_constraint,
        )
        if diagnostics:
            opt_metrics.update(diag)
        if optimizer_config.constrain_norm:
            updates = constrain_norm(updates, optimizer_config.norm_constraint)
        params = optax.apply_updates(params, updates)
        return params, optimizer_state, opt_metrics

    update_param_fn = construct_same_sampled_spring_unified_update_param_fn(
        energy_and_statistics_fn,
        optimizer_apply,
        get_position_fn=get_position_fn,
        update_data_fn=update_data_fn,
        record_param_l1_norm=record_param_l1_norm,
        apply_pmap=apply_pmap,
    )

    key, subkey = utils.distribute.split_or_psplit_key(key, apply_pmap)
    if apply_pmap:
        optimizer_state = utils.distribute.pmap(init_state)(params, subkey)
    else:
        optimizer_state = init_state(params, subkey)

    return update_param_fn, optimizer_state, key
