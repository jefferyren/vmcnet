"""Integration smoke test: prime_sr lowers a fabricated quadratic energy."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ml_collections import ConfigDict

from vmcnet.updates.prime_sr import initialize_prime_sr


def _log_psi_apply(params, x):
    """Two-parameter toy log-amplitude with a rank-2 score covariance."""
    r2 = jnp.sum(x**2, axis=-1)
    r1 = jnp.sum(jnp.abs(x), axis=-1)
    return -0.5 * (params["a"] ** 2) * r2 + 0.5 * params["b"] * r1


def _make_energy_and_statistics_fn(target_a, target_b, k=0.2):
    """Fabricate local energies whose mean is (a-ta)^2 + (b-tb)^2.

    As in the same_sampled_spring_unified integration test, centering removes
    any per-chain fluctuation constant in the params, so each parameter gets a
    mean-zero fluctuation term k * (theta - target) * O_theta aligned with its
    own centered score (O_a = -a * fluct2, O_b = 0.5 * fluct1) — the literal
    shape of the VMC gradient theorem dE/dtheta = 2 Cov(E_L, O_theta), so the
    SR direction genuinely descends the surrogate for BOTH parameters. (A
    fluctuation with the wrong sign relative to the score sends that parameter
    uphill: SR follows Cov(E_L, O_theta), not the fluctuation itself.)
    """

    def energy_and_statistics_fn(params, positions):
        a, b = params["a"], params["b"]
        base = (a - target_a) ** 2 + (b - target_b) ** 2
        r2 = jnp.sum(positions**2, axis=-1)
        r1 = jnp.sum(jnp.abs(positions), axis=-1)
        score_a = -a * (r2 - jnp.mean(r2))  # centered d/da log|psi|
        score_b = 0.5 * (r1 - jnp.mean(r1))  # centered d/db log|psi|
        local_energies = (
            base + k * (a - target_a) * score_a + k * (b - target_b) * score_b
        )
        energy = jnp.mean(local_energies)
        stats = {
            "variance": jnp.var(local_energies),
            "energy_noclip": energy,
            "variance_noclip": jnp.var(local_energies),
        }
        return energy, local_energies, stats

    return energy_and_statistics_fn


@pytest.mark.slow
def test_energy_decreases_and_mu_adapts():
    """prime_sr lowers the fabricated energy and keeps mu in [0, 1]."""
    nchains = 16
    positions = jax.random.normal(jax.random.PRNGKey(0), (nchains, 2))
    params = {"a": jnp.array(2.5), "b": jnp.array(-0.5)}

    config = ConfigDict(
        {
            "learning_rate": 0.05,
            "damping": 1e-3,
            "constrain_norm": False,  # let the quadratic converge freely
            "norm_constraint": 1e-3,
        }
    )

    update_param_fn, opt_state = initialize_prime_sr(
        _log_psi_apply,
        _make_energy_and_statistics_fn(target_a=1.0, target_b=0.5),
        params,
        positions,
        get_position_fn=lambda d: d,
        # roll positions each step: same sample set, new chain indexing, so
        # successive eigenbases only partially overlap (as under real MCMC)
        # and mu stays interior rather than saturating at 1.
        update_data_fn=lambda d, p_: jnp.roll(d, 1, axis=0),
        learning_rate_schedule=lambda t: 0.05,
        optimizer_config=config,
        apply_pmap=False,
    )

    energies, mus = [], []
    p = params
    data = positions
    key = jax.random.PRNGKey(1)
    for _ in range(60):
        p, data, opt_state, metrics, key = update_param_fn(p, data, opt_state, key)
        energies.append(float(metrics["energy"]))
        mus.append(float(metrics["mu"]))

    assert np.all(np.isfinite(energies))
    assert energies[-1] < energies[0]
    assert energies[-1] < 0.5 * energies[0]  # substantial, genuine progress
    # first step has no cached subspace; afterwards mu is adaptive in [0, 1]
    assert mus[0] == 0.0
    assert all(0.0 <= m <= 1.0 for m in mus)
    assert max(mus[1:]) > 0.0
    assert int(opt_state.step) == 60
    assert float(opt_state.rank) >= 2.0  # the toy model must exercise rank > 1
