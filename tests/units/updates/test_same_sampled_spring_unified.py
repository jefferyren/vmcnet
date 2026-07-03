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
    _adaptive_beta_update,
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


def _numpy_adaptive_beta(buffer, res, r_hat, beta, ckpt, step, p):
    """Numpy reference for the adaptive-beta schedule used to cross-check the jax op."""
    buf = np.concatenate([np.asarray(buffer)[1:], [float(res)]])
    do_update = (step % p == 0) and (step >= 2 * p)
    if not do_update:
        return buf, float(r_hat), float(beta), int(ckpt)
    window_tp = buf[0:p]
    window_t = buf[p : 2 * p]
    r_ip = np.sum(window_t**2) / np.sum(window_tp**2)
    n_old = float(ckpt)
    n_new = n_old + 1.0
    alph = (n_old ** np.log(n_old)) / (n_new ** np.log(n_new))
    r_hat_new = alph * r_hat + (1.0 - alph) * min(1.0, r_ip)
    rho = max(0.0, 1.0 - r_hat_new ** (1.0 / p))
    beta_new = (1.0 - rho) / (1.0 + rho)
    return buf, r_hat_new, beta_new, ckpt + 1


def test_adaptive_beta_slides_buffer_but_no_update_before_2p():
    """Test that the buffer slides but beta/r_hat/checkpoint stay put before 2p."""
    p = 3
    buffer = jnp.arange(2 * p, dtype=jnp.float32)  # [0,1,2,3,4,5]
    out_buf, out_rhat, out_beta, out_ckpt = _adaptive_beta_update(
        buffer,
        jnp.array(9.0),
        jnp.array(1.0),
        jnp.array(0.9),
        jnp.array(1, jnp.int32),
        jnp.array(2, jnp.int32),
        p,
    )
    # step=2 (< 2p=6): buffer slides, everything else unchanged
    np.testing.assert_allclose(out_buf, np.array([1, 2, 3, 4, 5, 9]))
    np.testing.assert_allclose(out_beta, 0.9)
    np.testing.assert_allclose(out_rhat, 1.0)
    assert int(out_ckpt) == 1


def test_adaptive_beta_all_zero_buffer_no_nan():
    """All-zero buffer at a non-trigger step yields no NaN and no state change."""
    p = 3
    buffer = jnp.zeros(2 * p)
    out_buf, out_rhat, out_beta, out_ckpt = _adaptive_beta_update(
        buffer,
        jnp.array(0.0),
        jnp.array(1.0),
        jnp.array(0.9),
        jnp.array(1, jnp.int32),
        jnp.array(1, jnp.int32),
        p,
    )
    # step=1 (< 2p): no update; buffer slid a zero in; nothing is NaN.
    assert bool(jnp.all(jnp.isfinite(out_buf)))
    np.testing.assert_allclose(out_beta, 0.9)
    np.testing.assert_allclose(out_rhat, 1.0)
    assert int(out_ckpt) == 1


def test_adaptive_beta_updates_at_trigger_matches_numpy():
    """Test that _adaptive_beta_update matches the numpy reference at a trigger step."""
    p = 3
    rng = np.random.default_rng(0)
    buffer = jnp.asarray(rng.uniform(0.1, 1.0, size=2 * p), dtype=jnp.float32)
    res, r_hat, beta, ckpt, step = 0.4, 0.8, 0.9, 1, 2 * p  # step==6 triggers

    got = _adaptive_beta_update(
        buffer,
        jnp.array(res),
        jnp.array(r_hat),
        jnp.array(beta),
        jnp.array(ckpt, jnp.int32),
        jnp.array(step, jnp.int32),
        p,
    )
    exp = _numpy_adaptive_beta(buffer, res, r_hat, beta, ckpt, step, p)
    np.testing.assert_allclose(got[0], exp[0], rtol=1e-6)
    np.testing.assert_allclose(got[1], exp[1], rtol=1e-5)
    np.testing.assert_allclose(got[2], exp[2], rtol=1e-5)
    assert int(got[3]) == exp[3]


def test_adaptive_beta_is_jittable():
    """Test that _adaptive_beta_update runs under jax.jit without error."""
    p = 3
    f = jax.jit(lambda b, r, rh, be, c, s: _adaptive_beta_update(b, r, rh, be, c, s, p))
    buffer = jnp.ones(2 * p)
    out = f(
        buffer,
        jnp.array(1.0),
        jnp.array(1.0),
        jnp.array(0.9),
        jnp.array(1, jnp.int32),
        jnp.array(6, jnp.int32),
    )
    assert out[2].shape == ()  # beta scalar, no crash under jit
