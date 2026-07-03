"""Unit tests for the same_sampled_spring_unified optimizer."""

import jax
import jax.numpy as jnp
import numpy as np

# neural_tangents is imported after the vmcnet imports below: vmcnet.updates
# transitively imports kfac_jax (via vmcnet.physics), and in this environment
# importing neural_tangents first causes kfac_jax's tensorflow_probability
# dependency to misdetect the installed tensorflow version. Importing
# vmcnet.updates first avoids that import-order conflict.
from vmcnet.updates.same_sampled_spring_unified import (
    SameSampledSPRINGUnifiedState,
    _build_operators,
    _draw_unit_norm_like,
)
from vmcnet.utils.pytree_helpers import tree_inner_product

import neural_tangents as nt  # type: ignore


def _example_params():
    return {
        "w1": jnp.arange(6.0).reshape(2, 3),
        "b1": jnp.zeros(3),
        "w2": jnp.arange(3.0).reshape(3, 1),
        "b2": jnp.zeros(1),
    }


def test_draw_unit_norm_like_has_global_unit_l2_norm():
    """Test that _draw_unit_norm_like draws a pytree with global unit L2 norm."""
    params = _example_params()
    key = jax.random.PRNGKey(0)
    x_star = _draw_unit_norm_like(key, params)

    # same structure as params
    assert jax.tree_util.tree_structure(x_star) == jax.tree_util.tree_structure(params)

    flat = jax.flatten_util.ravel_pytree(x_star)[0]
    np.testing.assert_allclose(jnp.linalg.norm(flat), 1.0, rtol=1e-6)


def test_state_namedtuple_fields():
    """Test that SameSampledSPRINGUnifiedState has the expected field order."""
    fields = SameSampledSPRINGUnifiedState._fields
    assert fields == (
        "phi",
        "z_probe",
        "phi_probe",
        "x_star",
        "residual_buffer",
        "r_hat",
        "beta",
        "checkpoint_idx",
        "step",
    )


def _log_psi_apply(params, x):
    # x: (N, 2) -> (N,)
    h = jnp.tanh(x @ params["w1"] + params["b1"])  # (N, 3)
    return jnp.squeeze(h @ params["w2"] + params["b2"], axis=-1)  # (N,)


def _operator_setup(seed=0, nchains=7):
    key = jax.random.PRNGKey(seed)
    kp, kx = jax.random.split(key)
    params = {
        "w1": 0.5 * jax.random.normal(jax.random.fold_in(kp, 1), (2, 3)),
        "b1": 0.5 * jax.random.normal(jax.random.fold_in(kp, 2), (3,)),
        "w2": 0.5 * jax.random.normal(jax.random.fold_in(kp, 3), (3, 1)),
        "b2": 0.5 * jax.random.normal(jax.random.fold_in(kp, 4), (1,)),
    }
    positions = jax.random.normal(kx, (nchains, 2))
    kernel_fn = nt.empirical_kernel_fn(_log_psi_apply, vmap_axes=0, trace_axes=())
    return params, positions, kernel_fn


def test_apply_A_output_is_mean_zero():
    """Test that apply_A always returns a mean-zero vector."""
    params, positions, kernel_fn = _operator_setup()
    apply_A, _, _, _ = _build_operators(kernel_fn, _log_psi_apply, params, positions)
    v = jax.tree_map(
        lambda x: jax.random.normal(jax.random.PRNGKey(1), x.shape), params
    )
    out = apply_A(v)
    np.testing.assert_allclose(jnp.mean(out), 0.0, atol=1e-6)


def test_apply_A_and_apply_AT_are_adjoints():
    """Test that apply_A and apply_AT satisfy <A u, w> == <u, A^T w>."""
    params, positions, kernel_fn = _operator_setup()
    apply_A, apply_AT, _, _ = _build_operators(
        kernel_fn, _log_psi_apply, params, positions
    )
    u = jax.tree_map(
        lambda x: jax.random.normal(jax.random.PRNGKey(2), x.shape), params
    )
    w = jax.random.normal(jax.random.PRNGKey(3), (positions.shape[0],))

    lhs = jnp.sum(apply_A(u) * w)  # <A u, w>
    rhs = tree_inner_product(u, apply_AT(w))  # <u, A^T w>
    np.testing.assert_allclose(lhs, rhs, rtol=1e-5, atol=1e-6)


def test_T_matches_base_spring_construction():
    """Test that T built here equals the neural-tangents T built as in spring.py."""
    params, positions, kernel_fn = _operator_setup()
    _, _, Tvals, Tvecs = _build_operators(kernel_fn, _log_psi_apply, params, positions)
    T_reconstructed = (Tvecs * Tvals) @ Tvecs.T

    nchains = positions.shape[0]
    ones = jnp.ones((nchains, 1))
    T = kernel_fn(positions, positions, "ntk", params) / nchains
    T = T - jnp.mean(T, axis=0, keepdims=True)
    T = T - jnp.mean(T, axis=1, keepdims=True)
    T = T + ones @ ones.T / nchains
    T = (T + T.T) / 2
    np.testing.assert_allclose(T_reconstructed, T, rtol=1e-5, atol=1e-6)
