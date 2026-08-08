"""Unit tests for SPRING's optional momentum schedule.

The schedule exists to ablate the delayed-onset momentum that
same_sampled_spring_unified produces for free (its beta is pinned at its initial
value until the probe residual buffer fills, then jumps). The critical property is
that leaving it at its defaults changes nothing: constant-mu SPRING must stay
bit-identical to what it was before the option existed.
"""

import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

from vmcnet.updates.spring import (
    get_mu_schedule,
    get_spring_step,
    initialize_spring,
)


def test_schedule_is_none_when_disabled():
    """Defaults must return None so the original constant-mu path is untouched."""
    assert get_mu_schedule(0.99, 0.0, 0, 0) is None


def test_step_schedule_holds_then_jumps():
    """warmup with no ramp is a step function: mu_init, then mu."""
    sched = get_mu_schedule(0.99, mu_init=0.0, warmup_steps=60, ramp_steps=0)
    assert sched is not None
    for t in [0, 1, 30, 59]:
        np.testing.assert_allclose(float(sched(jnp.array(t))), 0.0)
    for t in [60, 61, 5000]:
        np.testing.assert_allclose(float(sched(jnp.array(t))), 0.99, rtol=1e-6)


def test_linear_ramp_interpolates_after_warmup():
    """With ramp_steps>0 mu rises linearly from mu_init to mu after the warmup."""
    sched = get_mu_schedule(0.8, mu_init=0.2, warmup_steps=10, ramp_steps=100)
    np.testing.assert_allclose(float(sched(jnp.array(10))), 0.2, atol=1e-6)
    np.testing.assert_allclose(float(sched(jnp.array(60))), 0.5, atol=1e-6)
    np.testing.assert_allclose(float(sched(jnp.array(110))), 0.8, atol=1e-6)
    np.testing.assert_allclose(float(sched(jnp.array(9999))), 0.8, atol=1e-6)


def test_schedule_is_jittable():
    """The schedule runs under jit (it is evaluated inside the traced update)."""
    sched = get_mu_schedule(0.99, 0.0, 60, 0)
    out = jax.jit(sched)(jnp.array(75, jnp.int32))
    np.testing.assert_allclose(float(out), 0.99, rtol=1e-6)


def _log_psi_apply(params, x):
    h = jnp.tanh(x @ params["w1"] + params["b1"])
    return jnp.squeeze(h @ params["w2"] + params["b2"], axis=-1)


def _setup(nchains=8, seed=0):
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


def test_mu_override_matches_a_step_built_with_that_mu():
    """spring_step(mu_override=m) equals a step constructed with mu=m."""
    params, positions = _setup()
    centered = jax.random.normal(jax.random.PRNGKey(3), (positions.shape[0],))
    prev = jax.tree_map(
        lambda x: 0.1 * jax.random.normal(jax.random.PRNGKey(4), x.shape), params
    )

    overridden = get_spring_step(_log_psi_apply, damping=1e-3, mu=0.99)(
        centered, params, prev, positions, mu_override=jnp.array(0.5)
    )
    direct = get_spring_step(_log_psi_apply, damping=1e-3, mu=0.5)(
        centered, params, prev, positions
    )
    for a, b in zip(jax.tree_util.tree_leaves(overridden),
                    jax.tree_util.tree_leaves(direct)):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-8)


def _energy_and_statistics_fn(params, positions):
    local_energies = jnp.sum(_log_psi_apply(params, positions)) * 0.0 + jnp.arange(
        positions.shape[0], dtype=jnp.float32
    )
    energy = jnp.mean(local_energies)
    stats = {"variance": jnp.var(local_energies), "energy_noclip": energy,
             "variance_noclip": jnp.var(local_energies)}
    return energy, local_energies, stats


def _config(**over):
    cfg = {"learning_rate": 0.05, "mu": 0.99, "damping": 1e-3,
           "constrain_norm": True, "norm_constraint": 1e-3,
           "mu_init": 0.0, "mu_warmup_steps": 0, "mu_ramp_steps": 0}
    cfg.update(over)
    return ConfigDict(cfg)


def _run(cfg, nsteps=5):
    params, positions = _setup()
    fn, state = initialize_spring(
        _log_psi_apply, _energy_and_statistics_fn, params,
        get_position_fn=lambda d: d, update_data_fn=lambda d, p_: d,
        learning_rate_schedule=lambda t: 0.05, optimizer_config=cfg,
        apply_pmap=False,
    )
    key, data, mus = jax.random.PRNGKey(0), positions, []
    for _ in range(nsteps):
        params, data, state, metrics, key = fn(params, data, state, key)
        mus.append(float(metrics["mu"]))
    return params, mus


def test_defaults_are_unchanged_by_the_new_option():
    """A default config must behave exactly like constant mu, and log mu==0.99."""
    params, mus = _run(_config())
    # float32 holds 0.99 as 0.99000001, so compare approximately.
    np.testing.assert_allclose(mus, [0.99] * 5, rtol=1e-6)
    for leaf in jax.tree_util.tree_leaves(params):
        assert bool(jnp.all(jnp.isfinite(leaf)))


def test_warmup_config_delays_momentum_and_logs_it():
    """With a warmup the logged mu is mu_init until the onset step, then mu."""
    _, mus = _run(_config(mu_warmup_steps=3), nsteps=5)
    np.testing.assert_allclose(mus, [0.0, 0.0, 0.0, 0.99, 0.99], rtol=1e-6)


def test_warmup_changes_the_trajectory():
    """Sanity: the warmup actually alters the parameters, not just the metric."""
    p_const, _ = _run(_config(), nsteps=5)
    p_warm, _ = _run(_config(mu_warmup_steps=3), nsteps=5)
    flat_c = jax.flatten_util.ravel_pytree(p_const)[0]
    flat_w = jax.flatten_util.ravel_pytree(p_warm)[0]
    assert not np.allclose(flat_c, flat_w), "warmup should change the trajectory"


def test_default_config_exposes_schedule_keys():
    """The spring block carries the schedule keys with no-op defaults."""
    from vmcnet.train.default_config import get_default_vmc_config

    block = get_default_vmc_config()["optimizer"]["spring"]
    assert block["mu_init"] == 0.0
    assert block["mu_warmup_steps"] == 0
    assert block["mu_ramp_steps"] == 0
