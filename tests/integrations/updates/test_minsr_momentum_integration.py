"""Integration smoke test: MinSR and MinSR+M lower a fabricated quadratic energy.

The unit tests pin the algebra (the solve equals SPRING's at mu = 0; the recursion is
the paper's convex combination). This one checks the thing algebra cannot: that both
baselines actually descend when driven end to end through the update_param_fn, at both
of the mu values the comparison table uses.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ml_collections import ConfigDict

from vmcnet.updates.minsr_momentum import initialize_minsr_momentum


def _log_psi_apply(params, x):
    """Two-parameter toy log-amplitude with a rank-2 score covariance."""
    r2 = jnp.sum(x**2, axis=-1)
    r1 = jnp.sum(jnp.abs(x), axis=-1)
    return -0.5 * (params["a"] ** 2) * r2 + 0.5 * params["b"] * r1


def _make_energy_and_statistics_fn(target_a, target_b, k=0.2):
    """Fabricate local energies whose mean is (a-ta)^2 + (b-tb)^2.

    Copied from the prime_sr integration test so the three optimizers are exercised
    against an identical surrogate. Each parameter gets a mean-zero fluctuation term
    aligned with its own centered score, which is the shape of the VMC gradient
    theorem dE/dtheta = 2 Cov(E_L, O_theta) -- so the SR direction genuinely descends
    for both parameters rather than only for the one the fluctuation happens to favor.
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


def _optimize(mu, nsteps=60, learning_rate=0.05):
    nchains = 16
    positions = jax.random.normal(jax.random.PRNGKey(0), (nchains, 2))
    params = {"a": jnp.array(2.5), "b": jnp.array(-0.5)}

    config = ConfigDict(
        {
            "learning_rate": learning_rate,
            "mu": mu,
            "damping": 1e-3,
            "constrain_norm": False,  # let the quadratic converge freely
            "norm_constraint": 1e-3,
        }
    )

    update_param_fn, opt_state = initialize_minsr_momentum(
        _log_psi_apply,
        _make_energy_and_statistics_fn(target_a=1.0, target_b=0.5),
        params,
        get_position_fn=lambda d: d,
        update_data_fn=lambda d, p_: jnp.roll(d, 1, axis=0),
        learning_rate_schedule=lambda t: learning_rate,
        optimizer_config=config,
        apply_pmap=False,
    )

    energies, mus = [], []
    p, data, key = params, positions, jax.random.PRNGKey(1)
    for _ in range(nsteps):
        p, data, opt_state, metrics, key = update_param_fn(p, data, opt_state, key)
        energies.append(float(metrics["energy"]))
        mus.append(float(metrics["mu"]))
    return energies, mus


@pytest.mark.slow
@pytest.mark.parametrize("mu", [0.0, 0.9])
def test_energy_decreases_for_both_baselines(mu):
    """Both the MinSR (mu=0) and MinSR+M (mu=0.9) arms make real progress."""
    energies, mus = _optimize(mu)

    assert np.all(np.isfinite(energies))
    assert energies[-1] < energies[0]
    assert energies[-1] < 0.5 * energies[0]  # substantial, genuine progress
    # mu is a fixed hyperparameter here, unlike prime_sr's adaptive one
    assert all(m == pytest.approx(mu) for m in mus)
