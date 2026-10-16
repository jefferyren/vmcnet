"""MinSR and MinSR with naive momentum (MinSR+M).

Both baselines from the SPRING paper (arXiv:2401.10190). MinSR is Eqs. (41)-(42):

    phi_k  = Obar^T (Obar Obar^T + lambda I + P)^{-1} epsbar_k
    dtheta = phi_k * min(eta_k, sqrt(C) / ||phi_k||)

and MinSR+M (Section 3.4, Eqs. 43-44) adds a naive convex-combination momentum:

    phi_k  = (1 - mu) Obar^T (Obar Obar^T + lambda I + P)^{-1} epsbar_k + mu phi_{k-1}

The contrast with SPRING is the point of the baseline: SPRING applies a regularized
projector to the carried-over phi_{k-1} before reusing it, whereas MinSR+M reuses it
untouched, and MinSR+M rescales the fresh solve by (1 - mu) to keep the combination
convex -- which is why mu here must stay strictly below 1, while SPRING's mu may reach
it.

Setting mu = 0 collapses the recursion to plain MinSR, so this single module supplies
both baseline columns. `tests/units/updates/test_minsr_momentum.py` pins the solve
against `spring.get_spring_step` at mu = 0, so the MinSR arm provably uses the same
linear solve as the SPRING arm rather than a lookalike reimplementation.

Implementation note: the recursion is left to optax rather than hand-rolled. optax's
trace is t_k = decay * t_{k-1} + g_k, so feeding it (1 - mu) * MinSR_k with decay = mu
reproduces Eq. (43) exactly, and the norm constraint is then applied to the eta-scaled
update exactly as in `spring.py` -- scaling by min(1, sqrt(C)/||eta phi||) is the same
thing as the paper's min(eta, sqrt(C)/||phi||).
"""

from typing import Dict

import jax
import jax.flatten_util
import jax.numpy as jnp
import neural_tangents as nt  # type: ignore
from ml_collections import ConfigDict
import chex
import optax

from vmcnet.utils.typing import Array, D, ModelApply, P, Tuple
from vmcnet.utils.pytree_helpers import multiply_tree_by_scalar
from vmcnet.utils.typing import UpdateDataFn, GetPositionFromData, LearningRateSchedule

from .update_param_fns import UpdateParamFn, get_update_norm_diagnostics
from .optax_utils import initialize_optax_optimizer
from .spring import constrain_norm, construct_spring_update_param_fn


def get_minsr_step(
    log_psi_apply: ModelApply[P],
    damping: chex.Scalar = 0.001,
):
    """Get the MinSR update direction function, Eq. (41).

    This is `spring.get_spring_step` specialized to mu = 0, with the momentum JVP
    dropped rather than evaluated against a zero tangent -- the tangent is a traced
    zero, so XLA cannot eliminate that JVP on its own, and MinSR runs would pay for it
    every step.

    Args:
        log_psi_apply: computes log|psi|, with signature (params, x) -> log|psi|(x).
        damping: Tikhonov regularization lambda applied to the Gram matrix.

    Returns:
        Callable: (centered_energies, params, positions) -> phi, the MinSR direction
        before the learning rate and the norm constraint are applied.
    """
    kernel_fn = nt.empirical_kernel_fn(log_psi_apply, vmap_axes=0, trace_axes=())

    def minsr_step(
        centered_energies: Array,
        params: P,
        positions: Array,
    ) -> P:
        nchains = positions.shape[0]
        ones = jnp.ones((nchains, 1))

        # T = Ohat @ Ohat^T, symmetrized and solved via its eigendecomposition with
        # negative eigenvalues clipped away. Same treatment as spring.py: on some GPUs
        # a Cholesky solve against this matrix fails outright in float32.
        T = kernel_fn(positions, positions, "ntk", params) / nchains
        T = T - jnp.mean(T, axis=0, keepdims=True)
        T = T - jnp.mean(T, axis=1, keepdims=True)
        T = T + ones @ ones.T / nchains
        T = (T + T.T) / 2
        Tvals, Tvecs = jnp.linalg.eigh(T)
        Tvals = jnp.maximum(Tvals, 0) + damping

        epsilon_bar = centered_energies / jnp.sqrt(nchains)
        zeta = Tvecs @ jnp.diag(1 / Tvals) @ Tvecs.T @ epsilon_bar
        zeta_hat = zeta - jnp.mean(zeta)
        dtheta = jax.vjp(log_psi_apply, params, positions)[1](zeta_hat)[0]

        return multiply_tree_by_scalar(dtheta, 1 / jnp.sqrt(nchains))

    return minsr_step


def initialize_minsr_momentum(
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
    """Get an update param function and initial state for MinSR / MinSR+M.

    `optimizer_config.mu` selects the baseline: 0.0 is MinSR, and the SPRING paper
    uses 0.9 for MinSR+M.
    """
    minsr_step = get_minsr_step(log_psi_apply, optimizer_config.damping)
    mu = float(optimizer_config.mu)

    # optax trace: t_k = mu * t_{k-1} + g_k. Feeding g_k = (1 - mu) * MinSR_k makes
    # t_k the paper's phi_k. At mu = 0 the trace decays instantly and t_k = MinSR_k.
    descent_optimizer = optax.sgd(
        learning_rate=learning_rate_schedule, momentum=mu, nesterov=False
    )

    def optimizer_apply(
        energy, local_energies, params, optimizer_state, data
    ) -> Tuple[P, optax.OptState, Dict[str, Array]]:
        positions = get_position_fn(data)
        centered_local_energies = local_energies - energy

        solve = minsr_step(centered_local_energies, params, positions)
        grad = multiply_tree_by_scalar(solve, 1.0 - mu)

        updates, optimizer_state = descent_optimizer.update(
            grad, optimizer_state, params
        )

        opt_metrics = get_update_norm_diagnostics(
            updates,
            optimizer_config.constrain_norm,
            optimizer_config.norm_constraint,
        )
        # Logged under the key SPRING, PRIME-SR and SS-SPRING all use, so a fixed mu
        # here overlays directly on their adaptive traces.
        opt_metrics["mu"] = jnp.asarray(mu, dtype=jnp.float32)

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
