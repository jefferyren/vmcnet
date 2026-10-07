"""The walker Gram kernel shared by the SR optimizers, optionally centred first.

Every SR optimizer here (SPRING, MinSR/MinSR+M, PRIME-SR, same_sampled_spring_unified)
forms T = Ohat Ohat^T by taking the neural-tangents NTK of log|psi|, i.e. the
UNCENTERED J J^T, and subtracting the row and column means afterwards. When the mean
gradient dominates the per-walker spread, that subtraction cancels most of the float32
digits. Phase F step F4d (docs/CAMPAIGN_LOG.md) measured it on a stretched-N2
checkpoint: the mean gradient was 15-20x the centred part, T had 263 negative
eigenvalues (min -0.0057, 5.6x the damping), and the resulting step was off by 4.6 of
its 6.0 against a float64 solve. Centring the rows BEFORE the contraction, still in
float32, matched float64 to ~1%.

`get_sr_kernel_fn(..., center_first=True)` does that cheaply. It computes the walker-
mean gradient m with one vjp, then takes the NTK of

    g(theta, x) = log|psi|(theta, x) - <stop_gradient(m), theta>,

whose per-walker gradient is J_i - m. The contraction then sees rows that are already
(nearly) mean-zero. In exact arithmetic the shift changes nothing after the callers'
own centring, which stays in place and removes whatever residual mean float32 leaves
in m.
"""

from typing import Callable

import jax
import jax.numpy as jnp
import neural_tangents as nt  # type: ignore

from vmcnet.utils.pytree_helpers import tree_inner_product
from vmcnet.utils.typing import Array, ModelApply, P


def get_sr_kernel_fn(
    log_psi_apply: ModelApply[P], center_first: bool = False
) -> Callable[[Array, Array, str, P], Array]:
    """Get the walker NTK, with the neural-tangents call signature.

    Args:
        log_psi_apply: maps (params, positions) -> log|psi|, shape (nchains,).
        center_first: if False, the plain neural-tangents empirical NTK (the
            original behavior). If True, the rows are shifted by the walker-mean
            gradient before the contraction (see the module docstring); the result
            differs from the plain NTK only by terms that the callers' row/column
            centring removes.

    Returns:
        kernel(x1, x2, get, params) -> (nchains, nchains) array, as returned by
        `nt.empirical_kernel_fn(log_psi_apply, vmap_axes=0, trace_axes=())`. With
        center_first the mean gradient is taken over `x1`, so `x1` should be the
        walker batch (every caller passes the same positions twice).
    """
    if not center_first:
        return nt.empirical_kernel_fn(log_psi_apply, vmap_axes=0, trace_axes=())

    def kernel(x1: Array, x2: Array, get: str, params: P) -> Array:
        out, vjp_fn = jax.vjp(lambda p: log_psi_apply(p, x1), params)
        mean_grad = vjp_fn(jnp.full_like(out, 1.0 / out.shape[0]))[0]
        mean_grad = jax.lax.stop_gradient(mean_grad)

        def shifted_apply(p: P, x: Array) -> Array:
            return log_psi_apply(p, x) - tree_inner_product(mean_grad, p)

        shifted_kernel_fn = nt.empirical_kernel_fn(
            shifted_apply, vmap_axes=0, trace_axes=()
        )
        return shifted_kernel_fn(x1, x2, get, params)

    return kernel
