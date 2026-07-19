"""Unit tests for the same_sampled_spring_unified optimizer."""

import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

# neural_tangents is imported after the vmcnet imports below: vmcnet.updates
# transitively imports kfac_jax (via vmcnet.physics), and in this environment
# importing neural_tangents first causes kfac_jax's tensorflow_probability
# dependency to misdetect the installed tensorflow version. Importing
# vmcnet.updates first avoids that import-order conflict.
from vmcnet.updates.same_sampled_spring_unified import (
    SameSampledSPRINGUnifiedState,
    _adaptive_beta_update,
    _adaptive_eta_main,
    _build_operators,
    _draw_unit_norm_like,
    get_same_sampled_spring_unified_step,
    initialize_same_sampled_spring_unified,
)
from vmcnet.updates.spring import get_spring_step
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


def test_adaptive_eta_main_off_returns_scheduled_lr():
    """With adaptive_eta off, eta_main is just the scheduled lr (base SPRING)."""
    lr = jnp.array(0.023)
    out = _adaptive_eta_main(lr, base_lr=0.05, beta=jnp.array(0.9), adaptive_eta=False)
    np.testing.assert_allclose(out, 0.023)


def test_adaptive_eta_main_constant_schedule_matches_pinn_form():
    """Constant schedule (decay == 1): eta_main == 1 - beta*(1 - eta0)."""
    base_lr, beta = 0.05, 0.9
    out = _adaptive_eta_main(
        jnp.array(base_lr), base_lr=base_lr, beta=jnp.array(beta), adaptive_eta=True
    )
    np.testing.assert_allclose(out, 1.0 - beta * (1.0 - base_lr), rtol=1e-6)


def test_adaptive_eta_main_applies_decay_as_outer_factor():
    """A decayed lr scales eta_main by the decay factor (outer multiplier)."""
    base_lr, beta = 0.05, 0.9
    # lr(t) = half of base_lr -> decay factor 0.5
    out = _adaptive_eta_main(
        jnp.array(0.5 * base_lr),
        base_lr=base_lr,
        beta=jnp.array(beta),
        adaptive_eta=True,
    )
    expected = 0.5 * (1.0 - beta * (1.0 - base_lr))
    np.testing.assert_allclose(out, expected, rtol=1e-6)


def test_adaptive_eta_main_anneals_to_zero_not_one_minus_beta():
    """As lr(t) -> 0 the step anneals to 0 (outer decay), NOT plateauing at 1-beta."""
    base_lr, beta = 0.05, 0.9
    tiny = _adaptive_eta_main(
        jnp.array(1e-8), base_lr=base_lr, beta=jnp.array(beta), adaptive_eta=True
    )
    # ~0, and far below the 1-beta=0.1 plateau the inner-decay form would give
    assert float(tiny) < 1e-6
    assert float(tiny) < 0.5 * (1.0 - beta)


def _zeros_like(params):
    """Return a params-shaped pytree of zeros."""
    return jax.tree_map(jnp.zeros_like, params)


def _make_state(params, p, beta, phi=None):
    """Build a SameSampledSPRINGUnifiedState with the given params/p/beta/phi."""
    zeros = _zeros_like(params)
    return SameSampledSPRINGUnifiedState(
        phi=zeros if phi is None else phi,
        z_probe=zeros,
        phi_probe=zeros,
        x_star=_draw_unit_norm_like(jax.random.PRNGKey(99), params),
        residual_buffer=jnp.zeros(2 * p),
        r_hat=jnp.array(1.0),
        beta=jnp.array(float(beta)),
        checkpoint_idx=jnp.array(1, jnp.int32),
        step=jnp.array(0, jnp.int32),
    )


def test_main_update_matches_base_spring():
    """Test that new_state.phi equals base SPRING's grad when beta==mu, no trigger."""
    # With adaptive_eta off, beta==mu, and no beta trigger, the main momentum update
    # (new_state.phi) must equal base SPRING's grad for arbitrary prior momentum.
    params, positions, _ = _operator_setup(nchains=9)
    mu, damping = 0.9, 1e-3
    prev_grad = jax.tree_map(
        lambda x: 0.1 * jax.random.normal(jax.random.PRNGKey(7), x.shape), params
    )
    centered_energies = jax.random.normal(jax.random.PRNGKey(8), (positions.shape[0],))

    base_step = get_spring_step(_log_psi_apply, damping=damping, mu=mu)
    base_grad = base_step(centered_energies, params, prev_grad, positions)

    lr_sched = lambda t: 0.05  # noqa: E731
    my_step = get_same_sampled_spring_unified_step(
        _log_psi_apply,
        lr_sched,
        damping=damping,
        probe_damping=damping,
        p=1000,
        probe_lr=0.05,
        adaptive_eta=False,
        adaptive_probe=False,
    )
    state = _make_state(params, p=1000, beta=mu, phi=prev_grad)
    _, new_state = my_step(centered_energies, params, positions, state)

    for a, b in zip(
        jax.tree_util.tree_leaves(new_state.phi),
        jax.tree_util.tree_leaves(base_grad),
    ):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


def test_step_advances_state_and_is_jittable():
    """Test that the step kernel is jittable and advances state as expected."""
    params, positions, _ = _operator_setup(nchains=8)
    p = 4
    lr_sched = lambda t: 0.05  # noqa: E731
    step_fn = get_same_sampled_spring_unified_step(
        _log_psi_apply,
        lr_sched,
        damping=1e-3,
        probe_damping=1e-3,
        p=p,
        probe_lr=0.05,
        adaptive_eta=True,
        adaptive_probe=True,
    )
    centered_energies = jax.random.normal(jax.random.PRNGKey(5), (positions.shape[0],))
    state = _make_state(params, p=p, beta=0.9)

    jstep = jax.jit(step_fn)
    updates, new_state = jstep(centered_energies, params, positions, state)

    # updates match params structure; step advanced by 1; buffer got a new entry.
    assert jax.tree_util.tree_structure(updates) == jax.tree_util.tree_structure(params)
    assert int(new_state.step) == 1
    assert new_state.residual_buffer.shape == (2 * p,)
    assert bool(jnp.all(jnp.isfinite(new_state.residual_buffer)))


def test_probe_residual_converges():
    """Probe residual decreases over steps: z_probe converges toward x_star."""
    params, positions, _ = _operator_setup(nchains=10)
    p = 1000  # large lb_window: adaptive-beta never triggers, beta stays == mu
    step_fn = get_same_sampled_spring_unified_step(
        _log_psi_apply,
        lambda t: 0.05,
        damping=1e-3,
        probe_damping=1e-3,
        p=p,
        probe_lr=0.5,
        adaptive_eta=False,
        adaptive_probe=False,
    )
    centered_energies = jax.random.normal(jax.random.PRNGKey(11), (positions.shape[0],))
    state = _make_state(params, p=p, beta=0.9)
    residuals = []
    for _ in range(15):
        _, state = step_fn(centered_energies, params, positions, state)
        residuals.append(float(state.residual_buffer[-1]))
    assert np.all(np.isfinite(residuals))
    assert residuals[-1] < residuals[0]
    assert residuals[-1] < 0.5 * residuals[0]  # substantial, genuine convergence


def _energy_and_statistics_fn(params, positions):
    # deterministic fake "local energies" so the test needs no MCMC/physics.
    local_energies = jnp.sum(_log_psi_apply(params, positions)) * 0.0 + jnp.arange(
        positions.shape[0], dtype=jnp.float32
    )
    energy = jnp.mean(local_energies)
    stats = {
        "variance": jnp.var(local_energies),
        "energy_noclip": energy,
        "variance_noclip": jnp.var(local_energies),
    }
    return energy, local_energies, stats


def _config(p=4):
    """Build an optimizer_config ConfigDict for the initializer test."""
    return ConfigDict(
        {
            "learning_rate": 0.05,
            "mu": 0.9,
            "damping": 1e-3,
            "constrain_norm": True,
            "norm_constraint": 1e-3,
            "lb_window": p,
            "probe_lr": -1.0,  # sentinel -> defaults to learning_rate
            "probe_damping": 1e-3,
            "adaptive_eta": False,
            "adaptive_probe": False,
        }
    )


def test_initialize_single_device_and_apply_reduces_state_step():
    """Test initialize_same_sampled_spring_unified builds state and applies once."""
    params, positions, _ = _operator_setup(nchains=8)
    data = positions  # get_position_fn is identity for this fake data

    update_param_fn, opt_state, key = initialize_same_sampled_spring_unified(
        _log_psi_apply,
        _energy_and_statistics_fn,
        params,
        get_position_fn=lambda d: d,
        update_data_fn=lambda d, p_: d,
        learning_rate_schedule=lambda t: 0.05,
        optimizer_config=_config(),
        key=jax.random.PRNGKey(0),
        record_param_l1_norm=False,
        apply_pmap=False,  # single device: jitted, not pmapped
    )

    # x_star has global unit norm; counters initialized.
    assert np.isclose(
        float(jnp.linalg.norm(jax.flatten_util.ravel_pytree(opt_state.x_star)[0])),
        1.0,
        atol=1e-5,
    )
    assert int(opt_state.step) == 0
    assert int(opt_state.checkpoint_idx) == 1

    new_params, new_data, new_state, metrics, new_key = update_param_fn(
        params, data, opt_state, key
    )
    assert int(new_state.step) == 1
    assert "energy" in metrics and "variance" in metrics
    # Adaptive momentum beta is logged as "mu" (PRIME-SR's key) plus r_hat.
    assert np.isclose(float(metrics["mu"]), float(new_state.beta))
    assert np.isclose(float(metrics["r_hat"]), float(new_state.r_hat))
    for leaf in jax.tree_util.tree_leaves(new_params):
        assert bool(jnp.all(jnp.isfinite(leaf)))


def test_default_config_has_block():
    """Test that the default VMC config has a same_sampled_spring_unified block."""
    from vmcnet.train.default_config import get_default_vmc_config

    cfg = get_default_vmc_config()
    block = cfg["optimizer"]["same_sampled_spring_unified"]
    for k in [
        "schedule_type",
        "learning_rate",
        "learning_decay_rate",
        "mu",
        "damping",
        "constrain_norm",
        "norm_constraint",
        "lb_window",
        "probe_lr",
        "probe_damping",
        "adaptive_eta",
        "adaptive_probe",
    ]:
        assert k in block, f"missing config key {k}"
    assert block["mu"] == 0.0
    assert block["probe_lr"] < 0  # sentinel


def test_parse_optimizer_config_dispatches_new_type():
    """Test that initialize_optimizer dispatches same_sampled_spring_unified."""
    # The dispatcher must recognize the new optimizer_type without raising ValueError.
    import inspect
    from vmcnet.updates import parse_optimizer_config as poc

    src = inspect.getsource(poc.initialize_optimizer)
    assert "same_sampled_spring_unified" in src
    assert hasattr(poc, "initialize_same_sampled_spring_unified")
