"""Unit tests for MinSR and MinSR with naive momentum (MinSR+M).

MinSR+M is the baseline the SPRING paper introduces in its Section 3.4 to show that
SPRING is not merely a momentum method (arXiv:2401.10190, Eqs. 43-44):

    phi_k  = (1 - mu) Obar^T (Obar Obar^T + lambda I + P)^{-1} epsbar_k + mu phi_{k-1}
    dtheta = phi_k * min(eta_k, sqrt(C) / ||phi_k||)

Unlike SPRING, no projector is applied to the carried-over phi_{k-1}, and the fresh
MinSR solve is scaled by (1 - mu) so the two form a convex combination -- which is why
mu must stay strictly below 1.

At mu = 0 the recursion collapses to plain MinSR (Eqs. 41-42), so this one optimizer
supplies both the MinSR and the MinSR+M column of the comparison table. The first test
below is what licenses that: it pins the solve against SPRING's own kernel at mu = 0,
so "our MinSR" is provably the same linear solve the SPRING arm uses.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ml_collections import ConfigDict

from vmcnet.updates.minsr_momentum import (
    get_minsr_step,
    initialize_minsr_momentum,
)
from vmcnet.updates.spring import get_spring_step


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


def _sq_norm(tree):
    return sum(float(jnp.sum(x**2)) for x in jax.tree_util.tree_leaves(tree))


def test_minsr_step_matches_spring_step_at_mu_zero():
    """The MinSR solve is exactly SPRING's update direction with mu = 0.

    This is the claim that lets the paper describe our MinSR column as SPRING's own
    solver with the momentum switched off, rather than as a separate implementation
    that merely resembles it.
    """
    nchains = 9
    params, positions = _setup(nchains=nchains)
    damping = 1e-2
    centered = jax.random.normal(jax.random.PRNGKey(8), (nchains,))

    minsr = get_minsr_step(_log_psi_apply, damping=damping)(centered, params, positions)
    spring_at_zero = get_spring_step(_log_psi_apply, damping=damping, mu=0.0)(
        centered, params, jax.tree_map(jnp.zeros_like, params), positions
    )

    for a, b in zip(
        jax.tree_util.tree_leaves(minsr),
        jax.tree_util.tree_leaves(spring_at_zero),
    ):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-8)


def test_minsr_step_is_jittable():
    """The step kernel runs under jax.jit; it is called inside a traced update."""
    nchains = 8
    params, positions = _setup(nchains=nchains)
    centered = jax.random.normal(jax.random.PRNGKey(5), (nchains,))

    step = jax.jit(get_minsr_step(_log_psi_apply, damping=1e-3))
    out = step(centered, params, positions)

    assert set(out) == set(params)
    for leaf in jax.tree_util.tree_leaves(out):
        assert np.all(np.isfinite(leaf))


def _config(mu, learning_rate, norm_constraint=1e-3, damping=1e-2):
    return ConfigDict(
        {
            "schedule_type": "inverse_time",
            "learning_rate": learning_rate,
            "learning_decay_rate": 1e-4,
            "mu": mu,
            "damping": damping,
            "constrain_norm": True,
            "norm_constraint": norm_constraint,
        }
    )


def _toy_problem(nchains=9, seed=0):
    """An energy_and_statistics_fn whose local energies do not depend on params.

    Holding the local energies and the walker positions fixed makes the parameter
    trajectory a deterministic function of the optimizer alone, which is what lets the
    reference loop below reproduce it exactly.
    """
    params, positions = _setup(seed=seed, nchains=nchains)
    local_energies = jax.random.normal(jax.random.PRNGKey(seed + 100), (nchains,))
    energy = jnp.mean(local_energies)

    def energy_and_statistics_fn(p, pos):
        del p, pos
        stats = {
            "variance": jnp.var(local_energies),
            "energy_noclip": energy,
            "variance_noclip": jnp.var(local_energies),
        }
        return energy, local_energies, stats

    centered = local_energies - energy
    return params, positions, centered, energy_and_statistics_fn


def _run(config, nsteps, nchains=9):
    """Drive the real update_param_fn for nsteps and return the params trajectory."""
    params, positions, _, energy_and_statistics_fn = _toy_problem(nchains=nchains)

    def schedule(t):
        return config.learning_rate / (1.0 + config.learning_decay_rate * t)

    update_param_fn, opt_state = initialize_minsr_momentum(
        _log_psi_apply,
        energy_and_statistics_fn,
        params,
        get_position_fn=lambda d: d,
        update_data_fn=lambda d, p: d,
        learning_rate_schedule=schedule,
        optimizer_config=config,
        apply_pmap=False,
    )

    key = jax.random.PRNGKey(0)
    data = positions
    trajectory, metrics_seen = [], []
    for _ in range(nsteps):
        params, data, opt_state, metrics, key = update_param_fn(
            params, data, opt_state, key
        )
        trajectory.append(params)
        metrics_seen.append(metrics)
    return trajectory, metrics_seen


def _reference(config, nsteps, nchains=9):
    """Eqs. (43)-(44) written out directly, as the trajectory to match."""
    params, positions, centered, _ = _toy_problem(nchains=nchains)
    minsr_step = get_minsr_step(_log_psi_apply, damping=config.damping)
    mu = config.mu

    phi = jax.tree_map(jnp.zeros_like, params)
    trajectory = []
    for k in range(nsteps):
        solve = minsr_step(centered, params, positions)
        phi = jax.tree_map(lambda s, p: (1.0 - mu) * s + mu * p, solve, phi)
        eta = config.learning_rate / (1.0 + config.learning_decay_rate * k)
        scale = min(eta, np.sqrt(config.norm_constraint / _sq_norm(phi)))
        params = jax.tree_map(lambda p, f: p - scale * f, params, phi)
        trajectory.append(params)
    return trajectory


@pytest.mark.parametrize("mu", [0.0, 0.9, 0.5])
def test_trajectory_matches_the_paper_convex_combination(mu):
    """Three steps of the optimizer reproduce Eqs. (43)-(44) exactly.

    mu = 0.0 is the MinSR column (the recursion has no memory); mu = 0.9 is the
    MinSR+M column at the value the SPRING paper uses.
    """
    config = _config(mu=mu, learning_rate=0.05)
    got, _ = _run(config, nsteps=3)
    want = _reference(config, nsteps=3)

    for step, (g, w) in enumerate(zip(got, want)):
        for a, b in zip(jax.tree_util.tree_leaves(g), jax.tree_util.tree_leaves(w)):
            np.testing.assert_allclose(
                a, b, rtol=1e-5, atol=1e-7, err_msg=f"step {step}"
            )


def test_momentum_carries_information_across_steps():
    """Momentum with mu > 0 must not silently behave like mu = 0.

    Guards the parametrized test above against an implementation that ignores the
    carried phi entirely, which would still satisfy the mu = 0 case.
    """
    memoryless, _ = _run(_config(mu=0.0, learning_rate=0.05), nsteps=3)
    with_momentum, _ = _run(_config(mu=0.9, learning_rate=0.05), nsteps=3)

    diff = max(
        float(jnp.max(jnp.abs(a - b)))
        for a, b in zip(
            jax.tree_util.tree_leaves(memoryless[-1]),
            jax.tree_util.tree_leaves(with_momentum[-1]),
        )
    )
    assert diff > 1e-6


def test_norm_constraint_caps_the_realized_step():
    """With a large eta the update is rescaled to the norm-constraint sphere.

    dtheta = phi * min(eta, sqrt(C)/||phi||), so once eta is the larger of the two the
    squared norm of the realized parameter change must sit exactly at C.
    """
    config = _config(mu=0.9, learning_rate=1e4, norm_constraint=1e-3)
    params, _, _, _ = _toy_problem()
    trajectory, metrics = _run(config, nsteps=2)

    step_taken = jax.tree_map(lambda new, old: new - old, trajectory[0], params)
    np.testing.assert_allclose(_sq_norm(step_taken), config.norm_constraint, rtol=1e-5)
    np.testing.assert_allclose(float(metrics[0]["norm_cap_applied"]), 1.0)


def test_metrics_report_mu_and_norm_diagnostics():
    """Metrics log mu under the shared key so all four optimizers share one axis."""
    _, metrics = _run(_config(mu=0.9, learning_rate=0.05), nsteps=1)
    m = metrics[0]

    np.testing.assert_allclose(float(m["mu"]), 0.9)
    assert "update_sq_norm_preclip" in m
    assert "norm_cap_applied" in m
    assert "energy" in m and "energy_noclip" in m


def test_unconstrained_norm_leaves_the_step_uncapped():
    """constrain_norm=False must skip the rescaling entirely."""
    config = _config(mu=0.9, learning_rate=0.05)
    config.constrain_norm = False
    params, _, _, _ = _toy_problem()
    trajectory, metrics = _run(config, nsteps=1)

    step_taken = jax.tree_map(lambda new, old: new - old, trajectory[0], params)
    assert _sq_norm(step_taken) < config.norm_constraint
    np.testing.assert_allclose(float(metrics[0]["norm_cap_applied"]), 0.0)


def test_default_config_has_minsr_momentum_block():
    """The config block carries the SPRING paper's own MinSR+M hyperparameters."""
    from vmcnet.train.default_config import get_default_vmc_config

    block = get_default_vmc_config()["optimizer"]["minsr_momentum"]
    for k in [
        "schedule_type",
        "learning_rate",
        "learning_decay_rate",
        "mu",
        "damping",
        "constrain_norm",
        "norm_constraint",
    ]:
        assert k in block, f"missing config key {k}"
    # Section 4 of the paper: lambda = C = 0.001, and mu = 0.9 for MinSR+M.
    assert block["mu"] == 0.9
    assert block["damping"] == 0.001
    assert block["norm_constraint"] == 0.001
    assert block["schedule_type"] == "inverse_time"
    assert block["learning_decay_rate"] == 1e-4


def test_parse_optimizer_config_dispatches_minsr_momentum():
    """The optimizer dispatcher recognizes the minsr_momentum optimizer_type."""
    import inspect

    from vmcnet.updates import parse_optimizer_config as poc

    src = inspect.getsource(poc.initialize_optimizer)
    assert '"minsr_momentum"' in src
    assert hasattr(poc, "initialize_minsr_momentum")
