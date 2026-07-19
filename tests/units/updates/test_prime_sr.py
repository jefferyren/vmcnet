"""Unit tests for the PRIME-SR optimizer."""

import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

from vmcnet.updates.prime_sr import (
    PRIMESRState,
    _adaptive_mu,
    _spectral_indicators,
    get_prime_sr_step,
    initialize_prime_sr,
)
from vmcnet.updates.spring import get_spring_step


def test_state_namedtuple_fields():
    """Test that PRIMESRState has the expected field order."""
    fields = PRIMESRState._fields
    assert fields == (
        "phi",
        "V_prev_alpha",
        "alpha_prev_ceil",
        "step",
        "mu",
        "alpha",
        "rank",
        "beta_tilde",
    )


def test_spectral_indicators_match_numpy():
    """Test rank/alpha/ceil(alpha) against a direct numpy computation."""
    s2 = jnp.array([4.0, 3.0, 1.0, 1e-12, 0.0])
    nchains = 5
    rank, alpha, alpha_ceil = _spectral_indicators(s2, nchains)

    s2_np = np.asarray(s2)
    tol = nchains * np.finfo(s2_np.dtype).eps * s2_np[0]
    kept = s2_np[s2_np > tol]
    expected_alpha = kept.sum() ** 2 / (kept**2).sum()

    assert int(rank) == kept.size == 3
    np.testing.assert_allclose(float(alpha), expected_alpha, rtol=1e-6)
    assert int(alpha_ceil) == int(np.clip(np.ceil(expected_alpha), 1, kept.size))


def test_spectral_indicators_zero_spectrum():
    """A zero spectrum gives rank 0, alpha 0, and the alpha_ceil == 0 sentinel."""
    rank, alpha, alpha_ceil = _spectral_indicators(jnp.zeros(4), 4)
    assert int(rank) == 0
    assert int(alpha_ceil) == 0
    np.testing.assert_allclose(float(alpha), 0.0)


def test_adaptive_mu_matches_formula():
    """Test mu (Eq. 4.5) and beta_tilde (Eq. 4.4) against explicit numpy slicing."""
    n = 6
    key = jax.random.PRNGKey(0)
    q1, _ = jnp.linalg.qr(jax.random.normal(key, (n, n)))
    q2, _ = jnp.linalg.qr(jax.random.normal(jax.random.fold_in(key, 1), (n, n)))
    alpha = jnp.array(2.4)
    rank = jnp.array(5, jnp.int32)
    a_ceil = jnp.array(3, jnp.int32)
    a_prev = jnp.array(2, jnp.int32)
    col = jnp.arange(n)
    v_alpha = q1 * (col[None, :] < a_ceil)
    v_prev = q2 * (col[None, :] < a_prev)

    mu, beta_tilde = _adaptive_mu(alpha, rank, a_ceil, v_alpha, v_prev, a_prev)

    bt = np.linalg.norm(np.asarray(q1)[:, :3].T @ np.asarray(q2)[:, :2])
    ratio_b = min(1.0, bt / np.sqrt(min(3, 2)))
    ratio_a = min(1.0, 2.4 / 5.0)
    expected_mu = 1.0 - (1.0 - np.sqrt(ratio_b)) * (1.0 - ratio_a**0.25)
    np.testing.assert_allclose(float(beta_tilde), bt, rtol=1e-5)
    np.testing.assert_allclose(float(mu), expected_mu, rtol=1e-5)


def test_adaptive_mu_gated_to_zero_without_valid_cache():
    """Both mu and beta_tilde are 0 when alpha_prev_ceil == 0 or rank == 0."""
    n = 4
    v = jnp.eye(n)
    # no valid previous subspace
    mu, bt = _adaptive_mu(
        jnp.array(2.0),
        jnp.array(3, jnp.int32),
        jnp.array(2, jnp.int32),
        v,
        jnp.zeros((n, n)),
        jnp.array(0, jnp.int32),
    )
    np.testing.assert_allclose(float(mu), 0.0)
    np.testing.assert_allclose(float(bt), 0.0)
    # rank 0 this step
    mu, bt = _adaptive_mu(
        jnp.array(0.0),
        jnp.array(0, jnp.int32),
        jnp.array(0, jnp.int32),
        jnp.zeros((n, n)),
        v,
        jnp.array(2, jnp.int32),
    )
    np.testing.assert_allclose(float(mu), 0.0)
    np.testing.assert_allclose(float(bt), 0.0)


def _log_psi_apply(params, x):
    """Toy log|psi|: a small tanh MLP mapping (N, 2) -> (N,)."""
    h = jnp.tanh(x @ params["w1"] + params["b1"])  # (N, 3)
    return jnp.squeeze(h @ params["w2"] + params["b2"], axis=-1)  # (N,)


def _setup(seed=0, nchains=9):
    """Draw a small random params pytree and positions batch."""
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


def _fresh_state(params, nchains):
    """Build the step-0 PRIMESRState (zeros + alpha_prev_ceil sentinel 0)."""
    return PRIMESRState(
        phi=jax.tree_map(jnp.zeros_like, params),
        V_prev_alpha=jnp.zeros((nchains, nchains)),
        alpha_prev_ceil=jnp.array(0, jnp.int32),
        step=jnp.array(0, jnp.int32),
        mu=jnp.array(0.0),
        alpha=jnp.array(0.0),
        rank=jnp.array(0.0),
        beta_tilde=jnp.array(0.0),
    )


def test_first_step_matches_base_spring_mu_zero():
    """Step 0 has mu == 0 and phi equal to base SPRING's mu=0 update direction."""
    nchains = 9
    params, positions = _setup(nchains=nchains)
    damping = 1e-2
    centered = jax.random.normal(jax.random.PRNGKey(8), (nchains,))

    base_step = get_spring_step(_log_psi_apply, damping=damping, mu=0.0)
    base_grad = base_step(
        centered, params, jax.tree_map(jnp.zeros_like, params), positions
    )

    step_fn = get_prime_sr_step(_log_psi_apply, lambda t: 0.05, damping=damping)
    updates, new_state = step_fn(
        centered, params, positions, _fresh_state(params, nchains)
    )

    np.testing.assert_allclose(float(new_state.mu), 0.0)
    # D1: prime_sr omits the ones-term from the eigendecomposed matrix, so the
    # two solves agree only up to float32 eigendecomposition differences.
    for a, b in zip(
        jax.tree_util.tree_leaves(new_state.phi),
        jax.tree_util.tree_leaves(base_grad),
    ):
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)
    # updates are exactly -lr * phi
    for u, ph in zip(
        jax.tree_util.tree_leaves(updates),
        jax.tree_util.tree_leaves(new_state.phi),
    ):
        np.testing.assert_allclose(u, -0.05 * ph, rtol=1e-6, atol=1e-8)


def test_identical_T_drives_beta_tilde_to_upper_bound():
    """A second step on identical (params, positions) saturates beta_tilde and mu."""
    nchains = 9
    params, positions = _setup(nchains=nchains)
    centered = jax.random.normal(jax.random.PRNGKey(8), (nchains,))
    step_fn = get_prime_sr_step(_log_psi_apply, lambda t: 0.05, damping=1e-2)

    _, state1 = step_fn(centered, params, positions, _fresh_state(params, nchains))
    _, state2 = step_fn(centered, params, positions, state1)

    # same T both steps -> identical principal subspace -> overlap saturates
    m = min(int(state1.alpha_prev_ceil), int(state2.alpha_prev_ceil))
    np.testing.assert_allclose(float(state2.beta_tilde), np.sqrt(m), rtol=1e-4)
    np.testing.assert_allclose(float(state2.mu), 1.0, atol=1e-5)


def test_second_step_matches_base_spring_with_frozen_mu():
    """A perturbed-positions second step equals base SPRING run at mu = mu_2."""
    nchains = 9
    params, positions = _setup(nchains=nchains)
    damping = 1e-2
    centered = jax.random.normal(jax.random.PRNGKey(8), (nchains,))
    step_fn = get_prime_sr_step(_log_psi_apply, lambda t: 0.05, damping=damping)

    _, state1 = step_fn(centered, params, positions, _fresh_state(params, nchains))
    positions2 = positions + 0.1 * jax.random.normal(
        jax.random.PRNGKey(9), positions.shape
    )
    _, state2 = step_fn(centered, params, positions2, state1)

    mu2 = float(state2.mu)
    assert 0.0 < mu2 <= 1.0

    base_step = get_spring_step(_log_psi_apply, damping=damping, mu=mu2)
    base_grad = base_step(centered, params, state1.phi, positions2)
    for a, b in zip(
        jax.tree_util.tree_leaves(state2.phi),
        jax.tree_util.tree_leaves(base_grad),
    ):
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)


def test_step_is_jittable_and_advances_state():
    """The step kernel runs under jax.jit, advances counters, caches the subspace."""
    nchains = 8
    params, positions = _setup(nchains=nchains)
    centered = jax.random.normal(jax.random.PRNGKey(5), (nchains,))
    step_fn = jax.jit(get_prime_sr_step(_log_psi_apply, lambda t: 0.05, damping=1e-3))

    updates, state1 = step_fn(
        centered, params, positions, _fresh_state(params, nchains)
    )

    assert jax.tree_util.tree_structure(updates) == jax.tree_util.tree_structure(params)
    assert int(state1.step) == 1
    assert int(state1.alpha_prev_ceil) >= 1  # cache now valid
    assert state1.V_prev_alpha.shape == (nchains, nchains)
    assert 1 <= int(state1.rank) <= nchains
    assert float(state1.alpha) >= 1.0
    # masked columns are exactly zero
    n_keep = int(state1.alpha_prev_ceil)
    np.testing.assert_allclose(state1.V_prev_alpha[:, n_keep:], 0.0)
    for leaf in jax.tree_util.tree_leaves(state1.phi):
        assert bool(jnp.all(jnp.isfinite(leaf)))


def _energy_and_statistics_fn(params, positions):
    """Deterministic fake local energies so the test needs no MCMC/physics."""
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


def test_initialize_single_device_and_apply_advances_state():
    """initialize_prime_sr builds a fresh state and one update works end-to-end."""
    nchains = 8
    params, positions = _setup(nchains=nchains)
    data = positions  # get_position_fn is identity for this fake data

    update_param_fn, opt_state = initialize_prime_sr(
        _log_psi_apply,
        _energy_and_statistics_fn,
        params,
        data,
        get_position_fn=lambda d: d,
        update_data_fn=lambda d, p_: d,
        learning_rate_schedule=lambda t: 0.05,
        optimizer_config=ConfigDict(
            {
                "learning_rate": 0.05,
                "damping": 1e-3,
                "constrain_norm": True,
                "norm_constraint": 1e-3,
            }
        ),
        record_param_l1_norm=False,
        apply_pmap=False,  # single device: jitted, not pmapped
    )

    assert int(opt_state.step) == 0
    assert int(opt_state.alpha_prev_ceil) == 0  # sentinel: no cached subspace
    assert opt_state.V_prev_alpha.shape == (nchains, nchains)

    new_params, new_data, new_state, metrics, new_key = update_param_fn(
        params, data, opt_state, jax.random.PRNGKey(0)
    )
    assert int(new_state.step) == 1
    assert int(new_state.alpha_prev_ceil) >= 1
    for k in ["energy", "variance", "mu", "alpha", "rank", "beta_tilde"]:
        assert k in metrics, f"missing metric {k}"
    np.testing.assert_allclose(float(metrics["mu"]), 0.0)  # first step
    assert float(metrics["rank"]) >= 1.0
    for leaf in jax.tree_util.tree_leaves(new_params):
        assert bool(jnp.all(jnp.isfinite(leaf)))


def test_default_config_has_prime_sr_block():
    """The default VMC config has a prime_sr block with paper defaults and no mu."""
    from vmcnet.train.default_config import get_default_vmc_config

    cfg = get_default_vmc_config()
    block = cfg["optimizer"]["prime_sr"]
    for k in [
        "schedule_type",
        "learning_rate",
        "learning_decay_rate",
        "damping",
        "constrain_norm",
        "norm_constraint",
    ]:
        assert k in block, f"missing config key {k}"
    assert "mu" not in block  # tuning-free: momentum is adaptive
    assert block["learning_rate"] == 2e-2
    assert block["damping"] == 1e-3
    assert block["norm_constraint"] == 1e-3


def test_parse_optimizer_config_dispatches_prime_sr():
    """The optimizer dispatcher recognizes the prime_sr optimizer_type."""
    import inspect

    from vmcnet.updates import parse_optimizer_config as poc

    src = inspect.getsource(poc.initialize_optimizer)
    assert '"prime_sr"' in src
    assert hasattr(poc, "initialize_prime_sr")
