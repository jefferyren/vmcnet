"""Same-Sampled SPRING with an interleaved same-sampled SPRING probe.

JAX/vmcnet port of the PyTorch reference ``same_sampled_spring_unified``. The main
update is ordinary SPRING (see ``vmcnet/updates/spring.py``) with a dynamic decay
factor ``beta``; a probe runs a second SPRING solve on the *same* sampled Jacobian
against a fixed synthetic target ``b = A @ x_star`` and its residual history drives an
adaptive-``beta`` schedule. See
``docs/superpowers/specs/2026-07-02-same-sampled-spring-unified-design.md``.
"""

from typing import NamedTuple

import jax
import jax.flatten_util
import jax.numpy as jnp

from vmcnet.utils.typing import Array, P, PRNGKey, PyTree


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
