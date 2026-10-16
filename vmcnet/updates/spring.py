"""SPRING implementation, see https://doi.org/10.1016/j.jcp.2024.113351."""

from typing import Callable, Dict
import jax
import jax.flatten_util
import jax.numpy as jnp
import neural_tangents as nt  # type: ignore
from ml_collections import ConfigDict
import chex
import optax

from vmcnet.utils.typing import Array, D, ModelApply, P, S, Tuple
from vmcnet.utils.pytree_helpers import (
    multiply_tree_by_scalar,
    tree_inner_product,
    tree_reduce_l1,
)
from vmcnet.utils.distribute import pmean_if_pmap
from vmcnet.utils.typing import UpdateDataFn, GetPositionFromData, LearningRateSchedule

from .update_param_fns import (
    UpdateParamFn,
    get_update_norm_diagnostics,
    make_traced_fn_with_single_metrics,
    update_metrics_with_noclip,
)
from .optax_utils import initialize_optax_optimizer


def construct_spring_update_param_fn(
    energy_and_statistics_fn,
    optimizer_apply: Callable[[P, P, S, D, Dict[str, Array]], Tuple[P, S, Dict]],
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    apply_pmap: bool = True,
    record_param_l1_norm: bool = False,
) -> UpdateParamFn[P, D, S]:
    """Create the `update_param_fn` based on the gradient of the total energy."""

    def update_param_fn(params, data, optimizer_state, key):
        position = get_position_fn(data)

        energy, local_energies, stats = energy_and_statistics_fn(params, position)

        params, optimizer_state, opt_metrics = optimizer_apply(
            energy,
            local_energies,
            params,
            optimizer_state,
            data,
        )
        data = update_data_fn(data, params)

        metrics = {"energy": energy, "variance": stats["variance"]}
        metrics = update_metrics_with_noclip(
            stats["energy_noclip"],
            stats["variance_noclip"],
            metrics,
        )
        metrics.update(opt_metrics)
        if record_param_l1_norm:
            metrics.update({"param_l1_norm": tree_reduce_l1(params)})
        return params, data, optimizer_state, metrics, key

    traced_fn = make_traced_fn_with_single_metrics(update_param_fn, apply_pmap)

    return traced_fn


def initialize_spring(
    log_psi_apply: ModelApply[P],
    energy_and_statistics_fn,
    params: P,
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    learning_rate_schedule: LearningRateSchedule,
    optimizer_config: ConfigDict,
    record_param_l1_norm: bool = False,
    apply_pmap: bool = True,
) -> Tuple[UpdateParamFn[P, D, optax.OptState], optax.OptState]:
    """Get an update param function and initial state for SPRING."""
    spring_step = get_spring_step(
        log_psi_apply,
        optimizer_config.damping,
        optimizer_config.mu,
    )
    # Optional momentum schedule. Absent keys keep the constant-mu behavior, so
    # configs written before this option existed still load.
    mu_schedule = get_mu_schedule(
        optimizer_config.mu,
        optimizer_config.get("mu_init", 0.0),
        int(optimizer_config.get("mu_warmup_steps", 0)),
        int(optimizer_config.get("mu_ramp_steps", 0)),
    )

    descent_optimizer = optax.sgd(
        learning_rate=learning_rate_schedule, momentum=0, nesterov=False
    )

    def prev_update(optimizer_state):
        return optimizer_state[0].trace

    def step_count(optimizer_state):
        # optax.sgd with a learning-rate SCHEDULE is chain(trace, scale_by_schedule),
        # so the step counter lives in the second state element (same positional
        # convention as prev_update reading [0].trace above).
        return optimizer_state[1].count

    def optimizer_apply(energy, local_energies, params, optimizer_state, data):
        positions = get_position_fn(data)

        centered_local_energies = local_energies - energy
        mu_t = (
            mu_schedule(step_count(optimizer_state))
            if mu_schedule is not None
            else jnp.asarray(optimizer_config.mu, dtype=jnp.float32)
        )
        grad = spring_step(
            centered_local_energies,
            params,
            prev_update(optimizer_state),
            positions,
            mu_override=mu_t if mu_schedule is not None else None,
        )

        updates, optimizer_state = descent_optimizer.update(
            grad, optimizer_state, params
        )

        opt_metrics = get_update_norm_diagnostics(
            updates,
            optimizer_config.constrain_norm,
            optimizer_config.norm_constraint,
        )
        # Log mu under the same key PRIME-SR and SS-SPRING use, so a fixed or
        # scheduled SPRING can be overlaid directly against their adaptive values.
        opt_metrics["mu"] = mu_t
        if optimizer_config.constrain_norm:
            updates = constrain_norm(
                updates,
                optimizer_config.norm_constraint,
            )

        params = optax.apply_updates(params, updates)
        return params, optimizer_state, opt_metrics

    update_param_fn = construct_spring_update_param_fn(
        energy_and_statistics_fn,
        optimizer_apply,
        get_position_fn=get_position_fn,
        update_data_fn=update_data_fn,
        record_param_l1_norm=record_param_l1_norm,
        apply_pmap=apply_pmap,
    )
    optimizer_state = initialize_optax_optimizer(
        descent_optimizer, params, apply_pmap=apply_pmap
    )

    return update_param_fn, optimizer_state


def get_mu_schedule(
    mu: chex.Scalar,
    mu_init: chex.Scalar = 0.0,
    warmup_steps: int = 0,
    ramp_steps: int = 0,
):
    """Build a step -> mu schedule, or None when mu is constant.

    Returning None for the constant case keeps the original code path exactly, so
    default-configured runs are bit-identical to before this option existed.

    The shape mirrors what `same_sampled_spring_unified` produces on its own: its
    adaptive beta is pinned at its initial value until the probe's residual buffer
    fills (2 * lb_window steps, i.e. 60 by default) and then jumps almost straight
    to its converged value. So the interesting knob is a DELAYED ONSET, not a
    gradual rise, and `ramp_steps=0` (a step function) is the faithful default.
    `ramp_steps > 0` linearly interpolates instead, to separate "delay" from
    "gradual increase" if that turns out to matter.

    Args:
        mu: the final momentum.
        mu_init: momentum held during warmup (0 = no momentum, matching SS-SPRING).
        warmup_steps: steps to hold `mu_init` before engaging.
        ramp_steps: length of the linear rise after warmup; 0 = instant step.

    Returns:
        A callable step -> mu, or None if the schedule is constant.
    """
    if warmup_steps <= 0 and ramp_steps <= 0:
        return None

    def mu_at(step: Array) -> Array:
        t = jnp.asarray(step, dtype=jnp.float32)
        if ramp_steps > 0:
            frac = jnp.clip((t - warmup_steps) / float(ramp_steps), 0.0, 1.0)
        else:
            frac = jnp.where(t >= warmup_steps, 1.0, 0.0)
        return mu_init + (mu - mu_init) * frac

    return mu_at


def get_spring_step(
    log_psi_apply: ModelApply[P],
    damping: chex.Scalar = 0.001,
    mu: chex.Scalar = 0.99,
):
    """Get the SPRING update function.

    The returned step accepts an optional `mu_override`, a traced scalar that
    replaces the fixed `mu` for that step. This is what lets a momentum schedule
    drive the update without duplicating the solve.
    """
    kernel_fn = nt.empirical_kernel_fn(log_psi_apply, vmap_axes=0, trace_axes=())

    def spring_step(
        centered_energies: P,
        params: P,
        prev_grad,
        positions: Array,
        mu_override=None,
    ) -> Tuple[Array, P]:
        nchains = positions.shape[0]
        mu_t = mu if mu_override is None else mu_override
        mu_prev = jax.tree_map(lambda x: mu_t * x, prev_grad)
        ones = jnp.ones((nchains, 1))

        # Calculate T = Ohat @ Ohat^T using neural-tangents
        # Some GPUs, particularly A100s and A5000s, can exhibit large numerical
        # errors in these calculations. As a result, we explicitly symmetrize T
        # and, rather than using a Cholesky solver to solve against T, we
        # calculate its eigendecomposition and explicitly fix any negative
        # eigenvalues. We then use the fixed and regularized igendecomposition
        # to solve against T. This appears to be more stable than Cholesky
        # in practice.
        T = kernel_fn(positions, positions, "ntk", params) / nchains
        T = T - jnp.mean(T, axis=0, keepdims=True)
        T = T - jnp.mean(T, axis=1, keepdims=True)
        T = T + ones @ ones.T / nchains
        T = (T + T.T) / 2
        Tvals, Tvecs = jnp.linalg.eigh(T)
        Tvals = jnp.maximum(Tvals, 0) + damping

        epsilon_bar = centered_energies / jnp.sqrt(nchains)
        O_prev = jax.jvp(
            log_psi_apply,
            (params, positions),
            (mu_prev, jnp.zeros_like(positions)),
        )[1] / jnp.sqrt(nchains)
        Ohat_prev = O_prev - jnp.mean(O_prev, axis=0, keepdims=True)
        epsilon_tilde = epsilon_bar - Ohat_prev

        zeta = Tvecs @ jnp.diag(1 / Tvals) @ Tvecs.T @ epsilon_tilde
        zeta_hat = zeta - jnp.mean(zeta)
        dtheta_residual = jax.vjp(log_psi_apply, params, positions)[1](zeta_hat)[0]

        return jax.tree_map(
            lambda dt, mup: dt / jnp.sqrt(nchains) + mup, dtheta_residual, mu_prev
        )

    return spring_step


def constrain_norm(
    grad: P,
    norm_constraint: chex.Numeric = 0.001,
) -> P:
    """Euclidean norm constraint."""
    sq_norm_scaled_grads = tree_inner_product(grad, grad)

    # Sync the norms here, see:
    # https://github.com/deepmind/deepmind-research/blob/30799687edb1abca4953aec507be87ebe63e432d/kfac_ferminet_alpha/optimizer.py#L585
    sq_norm_scaled_grads = pmean_if_pmap(sq_norm_scaled_grads)

    norm_scale_factor = jnp.sqrt(norm_constraint / sq_norm_scaled_grads)
    coefficient = jnp.minimum(norm_scale_factor, 1)
    constrained_grads = multiply_tree_by_scalar(grad, coefficient)

    return constrained_grads
