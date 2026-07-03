"""Same-Sampled SPRING with an interleaved same-sampled SPRING probe.

JAX/vmcnet port of the PyTorch reference ``same_sampled_spring_unified``. The main
update is ordinary SPRING (see ``vmcnet/updates/spring.py``) with a dynamic decay
factor ``beta``; a probe runs a second SPRING solve on the *same* sampled Jacobian
against a fixed synthetic target ``b = A @ x_star`` and its residual history drives an
adaptive-``beta`` schedule. See
``docs/superpowers/specs/2026-07-02-same-sampled-spring-unified-design.md``.
"""

from typing import Callable, NamedTuple, Tuple

import jax
import jax.flatten_util
import jax.numpy as jnp

from vmcnet.utils.pytree_helpers import multiply_tree_by_scalar
from vmcnet.utils.typing import Array, ModelApply, P, PRNGKey, PyTree


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
    """Build the shared SPRING linear operators at the current (params, positions).

    The main solve and the probe solve must use identical operators, so they are built
    once here and reused. `apply_A` centers its output and `apply_AT` centers its input
    (they are transposes of one another); the constant-mode term `(1/N) ones onesᵀ` and
    the negative-eigenvalue clip live in the returned eigendecomposition of `T`.

    Args:
        kernel_fn: neural-tangents empirical kernel, built once from `log_psi_apply`.
        log_psi_apply: maps (params, positions) -> log|psi| of shape (nchains,).
        params: current model parameters.
        positions: current walker positions, shape (nchains, ...).

    Returns:
        Tuple (apply_A, apply_AT, Tvals, Tvecs) where apply_A: P -> Array[nchains]
        (mean-zero), apply_AT: Array[nchains] -> P, and Tvals/Tvecs are the
        eigendecomposition of T = A Aᵀ + (1/N) ones onesᵀ.
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
    t = t - jnp.mean(t, axis=0, keepdims=True)
    t = t - jnp.mean(t, axis=1, keepdims=True)
    t = t + ones @ ones.T / nchains
    t = (t + t.T) / 2
    tvals, tvecs = jnp.linalg.eigh(t)
    return apply_A, apply_AT, tvals, tvecs
