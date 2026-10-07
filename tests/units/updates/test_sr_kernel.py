"""Tests for the centre-first SR Gram kernel (Phase F step F4d)."""

import jax
import jax.numpy as jnp
import numpy as np

from vmcnet.updates.sr_kernel import get_sr_kernel_fn
from vmcnet.updates.minsr_momentum import get_minsr_step
from vmcnet.updates.prime_sr import PRIMESRState, get_prime_sr_step
from vmcnet.updates.same_sampled_spring_unified import (
    SameSampledSPRINGUnifiedState,
    _draw_unit_norm_like,
    get_same_sampled_spring_unified_step,
)
from vmcnet.updates.spring import get_spring_step

import neural_tangents as nt  # type: ignore  # after vmcnet, see SS-SPRING tests


def _center(t):
    t = t - jnp.mean(t, axis=0, keepdims=True)
    return t - jnp.mean(t, axis=1, keepdims=True)


def _log_psi_apply(params, x):
    """Toy log|psi|: a small tanh MLP mapping (N, 2) -> (N,)."""
    h = jnp.tanh(x @ params["w1"] + params["b1"])
    return jnp.squeeze(h @ params["w2"] + params["b2"], axis=-1)


def _setup(seed=0, nchains=9):
    key = jax.random.PRNGKey(seed)
    kp, kx = jax.random.split(key)
    params = {
        "w1": 0.5 * jax.random.normal(jax.random.fold_in(kp, 1), (2, 3)),
        "b1": 0.5 * jax.random.normal(jax.random.fold_in(kp, 2), (3,)),
        "w2": 0.5 * jax.random.normal(jax.random.fold_in(kp, 3), (3, 1)),
        "b2": 0.5 * jax.random.normal(jax.random.fold_in(kp, 4), (1,)),
    }
    positions = jax.random.normal(kx, (nchains, 2))
    return params, positions


def test_off_is_the_plain_neural_tangents_kernel():
    """center_first=False returns exactly the original NTK."""
    params, positions = _setup()
    plain = nt.empirical_kernel_fn(_log_psi_apply, vmap_axes=0, trace_axes=())
    got = get_sr_kernel_fn(_log_psi_apply, center_first=False)
    np.testing.assert_array_equal(
        got(positions, positions, "ntk", params),
        plain(positions, positions, "ntk", params),
    )


def test_center_first_has_the_same_centred_kernel():
    """On a well-conditioned toy the centred kernels agree."""
    params, positions = _setup()
    plain = get_sr_kernel_fn(_log_psi_apply, center_first=False)
    first = get_sr_kernel_fn(_log_psi_apply, center_first=True)
    np.testing.assert_allclose(
        _center(first(positions, positions, "ntk", params)),
        _center(plain(positions, positions, "ntk", params)),
        rtol=1e-4,
        atol=1e-5,
    )


def _shifted_linear(params, x):
    # per-walker gradient wrt w is x + 100: the mean gradient dominates the spread
    return (x + 100.0) @ params["w"]


def test_center_first_removes_the_float32_cancellation():
    """Test the F4d failure mode: a dominant mean gradient.

    Centring after the product loses most digits; centring first does not.
    """
    n, d = 40, 300
    x = jax.random.normal(jax.random.PRNGKey(0), (n, d))
    params = {"w": jax.random.normal(jax.random.PRNGKey(1), (d,))}
    # exact reference from the float32-rounded gradient rows, centred in float64
    rows = np.asarray(x + 100.0, np.float64)
    rows = rows - rows.mean(axis=0)
    ref = rows @ rows.T

    def rel_err(center_first):
        k = get_sr_kernel_fn(_shifted_linear, center_first)(x, x, "ntk", params)
        t = np.asarray(_center(k), np.float64)
        return np.linalg.norm(t - ref) / np.linalg.norm(ref)

    err_after, err_first = rel_err(False), rel_err(True)
    assert err_first < 1e-5, err_first
    assert err_after > 100 * err_first, (err_after, err_first)


def _check_close(a_tree, b_tree):
    for a, b in zip(
        jax.tree_util.tree_leaves(a_tree), jax.tree_util.tree_leaves(b_tree)
    ):
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)


def _centered(nchains):
    c = jax.random.normal(jax.random.PRNGKey(4), (nchains,))
    return c - jnp.mean(c)


def test_spring_and_minsr_steps_unchanged_on_a_toy():
    """The flag is wired into SPRING and MinSR and changes nothing when exact.

    Damping 1e-2: at 1e-3 this toy's solve already amplifies float32 rounding to
    ~5e-4 relative, the very sensitivity F4d is about.
    """
    params, positions = _setup()
    centered = _centered(positions.shape[0])
    prev = jax.tree_util.tree_map(lambda p: 0.1 * jnp.ones_like(p), params)
    off = get_spring_step(_log_psi_apply, damping=1e-2, mu=0.9)
    on = get_spring_step(_log_psi_apply, damping=1e-2, mu=0.9, gram_center_first=True)
    _check_close(
        on(centered, params, prev, positions), off(centered, params, prev, positions)
    )
    off = get_minsr_step(_log_psi_apply, 1e-2)
    on = get_minsr_step(_log_psi_apply, 1e-2, gram_center_first=True)
    _check_close(on(centered, params, positions), off(centered, params, positions))


def test_prime_sr_step_unchanged_on_a_toy():
    """The flag is wired into PRIME-SR and changes nothing when exact."""
    params, positions = _setup()
    n = positions.shape[0]
    state = PRIMESRState(
        phi=jax.tree_util.tree_map(jnp.zeros_like, params),
        V_prev_alpha=jnp.zeros((n, n)),
        alpha_prev_ceil=jnp.array(0, jnp.int32),
        step=jnp.array(0, jnp.int32),
        mu=jnp.array(0.0),
        alpha=jnp.array(0.0),
        rank=jnp.array(0.0),
        beta_tilde=jnp.array(0.0),
    )
    off = get_prime_sr_step(_log_psi_apply, lambda t: 0.05, damping=1e-2)
    on = get_prime_sr_step(
        _log_psi_apply, lambda t: 0.05, damping=1e-2, gram_center_first=True
    )
    u_on, _ = on(_centered(n), params, positions, state)
    u_off, _ = off(_centered(n), params, positions, state)
    _check_close(u_on, u_off)


def test_ss_spring_step_and_stats_unchanged_on_a_toy():
    """Test the SS-SPRING wiring on a toy.

    The flag changes neither the step nor the mean-gradient statistic when exact.
    """
    params, positions = _setup()
    n = positions.shape[0]
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    phi = jax.tree_util.tree_map(lambda p: 0.1 * jnp.ones_like(p), params)
    state = SameSampledSPRINGUnifiedState(
        phi=phi,
        z_probe=zeros,
        phi_probe=zeros,
        x_star=_draw_unit_norm_like(jax.random.PRNGKey(99), params),
        residual_buffer=jnp.zeros(8),
        r_hat=jnp.array(1.0),
        beta=jnp.array(0.95),
        checkpoint_idx=jnp.array(1, jnp.int32),
        step=jnp.array(0, jnp.int32),
        r_ip=jnp.array(1.0),
    )
    kwargs = dict(
        damping=1e-3,
        probe_damping=1e-3,
        p=4,
        probe_lr=0.05,
        adaptive_eta=False,
        adaptive_probe=False,
        return_diagnostics=True,
    )
    off = get_same_sampled_spring_unified_step(_log_psi_apply, lambda t: 0.05, **kwargs)
    on = get_same_sampled_spring_unified_step(
        _log_psi_apply, lambda t: 0.05, gram_center_first=True, **kwargs
    )
    u_on, s_on, d_on = on(_centered(n), params, positions, state)
    u_off, s_off, d_off = off(_centered(n), params, positions, state)
    _check_close((u_on, s_on.phi), (u_off, s_off.phi))
    np.testing.assert_allclose(
        d_on["diag_mean_jac_sq_over_trace"],
        d_off["diag_mean_jac_sq_over_trace"],
        rtol=1e-5,
    )
    # the statistic is ||mean O||^2 / tr(centred kernel)
    rows = np.asarray(
        jax.vmap(
            lambda x: jax.flatten_util.ravel_pytree(
                jax.grad(lambda p: _log_psi_apply(p, x[None])[0])(params)
            )[0]
        )(positions),
        np.float64,
    )
    mean = rows.mean(axis=0)
    expected = (mean @ mean) / (((rows - mean) ** 2).sum() / n)
    np.testing.assert_allclose(
        d_off["diag_mean_jac_sq_over_trace"], expected, rtol=1e-4
    )


def test_default_config_has_the_flag_off_in_every_sr_block():
    """Every SR optimizer block carries gram_center_first, off by default."""
    from vmcnet.train.default_config import get_default_vmc_config

    opt = get_default_vmc_config()["optimizer"]
    for name in ("spring", "minsr_momentum", "same_sampled_spring_unified", "prime_sr"):
        assert opt[name]["gram_center_first"] is False, name
