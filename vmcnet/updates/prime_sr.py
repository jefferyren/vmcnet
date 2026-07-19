"""PRIME-SR: SPRING with principal-range-informed adaptive momentum.

Implementation of PRIME-SR (Algorithm 2 of "Momentum Stability and Adaptive
Control in Stochastic Reconfiguration", Wang & Liu, 2026, arXiv:2604.18357).
The update is SPRING (see ``vmcnet/updates/spring.py``) with the fixed momentum
``mu`` replaced each step by an adaptive ``mu_k`` computed from the
eigendecomposition of the sampled Gram matrix ``T_k = O_k^T O_k``:

* numerical rank ``r_k`` via the MATLAB-default tolerance ``N * eps * s2_max``,
* effective spectral dimension ``alpha_k = (sum s^2)^2 / sum s^4`` (Eq. 4.7),
* principal-subspace overlap ``beta_tilde_k = ||V_{k,a}^T V_{k-1,a}||_F``
  between the leading ``ceil(alpha)`` eigenvectors at successive steps
  (Eq. 4.4),
* ``mu_k`` via Eq. 4.5, increasing in both ``alpha_k`` and ``beta_tilde_k``.

See ``docs/superpowers/specs/2026-07-18-prime-sr-design.md``.
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
    PyTree,
    S,
    Tuple,
    UpdateDataFn,
)


class PRIMESRState(NamedTuple):
    """Optimizer state for PRIME-SR.

    Attributes:
        phi: previous update direction Delta theta_{k-1} (params-shaped pytree).
        V_prev_alpha: previous step's leading eigenvectors of T (nchains x
            nchains), with columns at index >= alpha_prev_ceil zeroed out.
        alpha_prev_ceil: ceil(alpha_{k-1}) (int scalar); 0 is the sentinel for
            "no valid cached subspace" (first step or rank-deficient history).
        step: step counter (int scalar, starts at 0), drives the lr schedule.
        mu: last adaptive momentum mu_k (diagnostic scalar).
        alpha: last effective spectral dimension alpha_k (diagnostic scalar).
        rank: last numerical rank r_k (diagnostic scalar, stored as float).
        beta_tilde: last subspace overlap beta_tilde_k (diagnostic scalar).
    """

    phi: PyTree
    V_prev_alpha: Array
    alpha_prev_ceil: Array
    step: Array
    mu: Array
    alpha: Array
    rank: Array
    beta_tilde: Array


def _spectral_indicators(s2: Array, nchains: int) -> Tuple[Array, Array, Array]:
    """Compute (rank, alpha, alpha_ceil) from descending eigenvalues of T.

    Args:
        s2: eigenvalues of T in descending order, clamped to be >= 0.
        nchains: number of samples N (T is N x N).

    Returns:
        Tuple (rank, alpha, alpha_ceil). `rank` (int scalar) is the numerical
        rank #{s2 > tol} with the MATLAB-default tolerance
        tol = N * eps_machine * s2[0]. `alpha` is the effective spectral
        dimension (paper Eq. 4.7) summed over the rank-truncated spectrum.
        `alpha_ceil` = clip(ceil(alpha), 1, rank), or 0 when rank == 0 (the
        "invalid" sentinel).
    """
    eps_m = jnp.finfo(s2.dtype).eps
    tol = nchains * eps_m * s2[0]
    mask = s2 > tol
    rank = jnp.sum(mask)
    sum_s2 = jnp.sum(jnp.where(mask, s2, 0.0))
    sum_s4 = jnp.sum(jnp.where(mask, s2 * s2, 0.0))
    alpha = (sum_s2 * sum_s2) / jnp.where(sum_s4 > 0.0, sum_s4, 1.0)
    alpha_ceil = jnp.clip(jnp.ceil(alpha).astype(jnp.int32), 1, jnp.maximum(rank, 1))
    alpha_ceil = jnp.where(rank > 0, alpha_ceil, 0)
    return rank, alpha, alpha_ceil


def _adaptive_mu(
    alpha: Array,
    rank: Array,
    alpha_ceil: Array,
    V_alpha: Array,
    V_prev_alpha: Array,
    alpha_prev_ceil: Array,
) -> Tuple[Array, Array]:
    """Compute the adaptive momentum mu_k and overlap beta_tilde_k.

    Implements paper Eqs. 4.4 and 4.5 on column-masked eigenvector matrices:
    the Frobenius norm of V_alpha^T V_prev_alpha over masked matrices equals
    the norm of the ceil(alpha_k) x ceil(alpha_{k-1}) overlap block. Both
    ratios are clipped to [0, 1] before fractional powers to guard float
    noise. When there is no valid cached subspace (alpha_prev_ceil == 0) or
    the current step is rank deficient (rank == 0), mu and beta_tilde are 0 —
    on the first step Delta theta_{-1} = 0, so mu is inert regardless.

    Args:
        alpha: effective spectral dimension alpha_k (float scalar).
        rank: numerical rank r_k (int scalar).
        alpha_ceil: ceil(alpha_k), clipped to [1, r_k] (int scalar).
        V_alpha: current leading eigenvectors, columns >= alpha_ceil zeroed.
        V_prev_alpha: previous leading eigenvectors, likewise column-masked.
        alpha_prev_ceil: previous ceil(alpha), 0 == no valid cache.

    Returns:
        Tuple (mu, beta_tilde), both float scalars in [0, 1] and
        [0, sqrt(min(alpha_ceil, alpha_prev_ceil))] respectively.
    """
    overlap = V_alpha.T @ V_prev_alpha
    beta_tilde = jnp.linalg.norm(overlap)
    min_alpha = jnp.maximum(jnp.minimum(alpha_ceil, alpha_prev_ceil), 1)
    ratio_b = jnp.clip(beta_tilde / jnp.sqrt(min_alpha.astype(alpha.dtype)), 0.0, 1.0)
    inner1 = 1.0 - jnp.sqrt(ratio_b)
    ratio_a = jnp.clip(alpha / jnp.maximum(rank, 1).astype(alpha.dtype), 0.0, 1.0)
    inner2 = 1.0 - ratio_a**0.25
    mu = jnp.clip(1.0 - inner1 * inner2, 0.0, 1.0)
    valid = (alpha_prev_ceil > 0) & (rank > 0)
    return jnp.where(valid, mu, 0.0), jnp.where(valid, beta_tilde, 0.0)


def get_prime_sr_step(
    log_psi_apply: ModelApply[P],
    learning_rate_schedule: LearningRateSchedule,
    damping: chex.Scalar = 0.001,
) -> Callable[[Array, P, Array, PRIMESRState], Tuple[P, PRIMESRState]]:
    """Get the PRIME-SR step kernel.

    Returns a pure function `step(centered_energies, params, positions, state)`
    returning `(updates, new_state)`, where `updates` is the parameter delta
    already scaled by `-lr(state.step)`. The SPRING solve reuses the same
    eigendecomposition that produces the momentum indicators.

    Args:
        log_psi_apply: maps (params, positions) -> log|psi|, shape (nchains,).
        learning_rate_schedule: step -> learning rate.
        damping: additive damping lambda in (T + lambda I).

    Returns:
        The step kernel described above.
    """
    kernel_fn = nt.empirical_kernel_fn(log_psi_apply, vmap_axes=0, trace_axes=())

    def prime_sr_step(
        centered_energies: Array,
        params: P,
        positions: Array,
        state: PRIMESRState,
    ) -> Tuple[P, PRIMESRState]:
        nchains = positions.shape[0]
        sqrt_n = jnp.sqrt(nchains)

        # T = Ohat Ohat^T as in spring.py, but WITHOUT the ones-term (spec D1):
        # the ones-term's spurious (eigenvalue 1, constant eigenvector) pair
        # would pollute rank/alpha/V_alpha/beta_tilde, and the solve is
        # unchanged because zeta is exactly mean-zero. Negative eigenvalues
        # are clamped as in spring.py.
        T = kernel_fn(positions, positions, "ntk", params) / nchains
        T = T - jnp.mean(T, axis=0, keepdims=True)
        T = T - jnp.mean(T, axis=1, keepdims=True)
        T = (T + T.T) / 2
        s2, V = jnp.linalg.eigh(T)
        s2 = jnp.maximum(jnp.flip(s2, 0), 0.0)
        V = jnp.flip(V, 1)

        rank, alpha, alpha_ceil = _spectral_indicators(s2, nchains)
        # column-mask the leading ceil(alpha) eigenvectors (dynamic slicing is
        # illegal under jit)
        V_alpha = V * (jnp.arange(nchains)[None, :] < alpha_ceil)
        mu, beta_tilde = _adaptive_mu(
            alpha, rank, alpha_ceil, V_alpha, state.V_prev_alpha, state.alpha_prev_ceil
        )

        # SPRING update with adaptive mu (same math as spring.py:140-181)
        mu_phi = multiply_tree_by_scalar(state.phi, mu)
        epsilon_bar = centered_energies / sqrt_n
        O_prev = (
            jax.jvp(
                log_psi_apply,
                (params, positions),
                (mu_phi, jnp.zeros_like(positions)),
            )[1]
            / sqrt_n
        )
        Ohat_prev = O_prev - jnp.mean(O_prev, axis=0, keepdims=True)
        epsilon_tilde = epsilon_bar - Ohat_prev

        zeta = V @ jnp.diag(1 / (s2 + damping)) @ V.T @ epsilon_tilde
        zeta_hat = zeta - jnp.mean(zeta)
        dtheta_residual = jax.vjp(log_psi_apply, params, positions)[1](zeta_hat)[0]
        phi_new = jax.tree_map(lambda dt, mp: dt / sqrt_n + mp, dtheta_residual, mu_phi)

        updates = multiply_tree_by_scalar(phi_new, -learning_rate_schedule(state.step))

        # keep the previous subspace cache on a (pathological) rank-0 step
        keep = rank > 0
        new_state = PRIMESRState(
            phi=phi_new,
            V_prev_alpha=jnp.where(keep, V_alpha, state.V_prev_alpha),
            alpha_prev_ceil=jnp.where(keep, alpha_ceil, state.alpha_prev_ceil),
            step=state.step + 1,
            mu=mu,
            alpha=alpha,
            rank=rank.astype(alpha.dtype),
            beta_tilde=beta_tilde,
        )
        return updates, new_state

    return prime_sr_step
