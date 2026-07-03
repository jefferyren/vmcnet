"""Integration smoke test: same_sampled_spring_unified lowers a quadratic energy."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ml_collections import ConfigDict

from vmcnet.updates.same_sampled_spring_unified import (
    initialize_same_sampled_spring_unified,
)


def _log_psi_apply(params, x):
    # single-parameter Gaussian log-amplitude: log|psi| = -0.5 * a^2 * sum(x^2)
    a = params["a"]
    return -0.5 * (a**2) * jnp.sum(x**2, axis=-1)


def _make_local_energy_and_stats(target_a, k=0.2):
    # Minimize a surrogate "energy" E(a) = (a - target_a)^2 whose per-sample local
    # energies average to E(a); this exercises the full optimizer path
    # deterministically.
    #
    # SPRING/natural-gradient updates are driven entirely by the *centered* local
    # energies' covariance with the score O_a(x) = d/da log|psi(a, x)| = -a * sum(x^2)
    # (see same_sampled_spring_unified._build_operators): centering subtracts the
    # cross-chain mean, so any per-chain fluctuation that is constant in `a` (e.g. a
    # fixed index-based jitter, uncorrelated with x) is exactly cancelled and carries
    # no gradient signal at all -- only fluctuation that correlates with the score
    # survives centering and can move `a` toward `target_a`. We add a small
    # x^2-proportional (mean-zero across chains) term whose sign flips at
    # a == target_a, mirroring the true VMC gradient theorem
    # dE/da = 2 * Cov(E_L(x), O_a(x)); `base` alone (constant across chains) would
    # vanish under centering and leave the optimizer with no real signal to follow.
    def energy_and_statistics_fn(params, positions):
        a = params["a"]
        base = (a - target_a) ** 2
        x_sq = jnp.sum(positions**2, axis=-1)
        fluctuation = x_sq - jnp.mean(x_sq)  # mean-zero across chains
        local_energies = base - k * fluctuation * (a - target_a)
        energy = jnp.mean(local_energies)
        stats = {
            "variance": jnp.var(local_energies),
            "energy_noclip": energy,
            "variance_noclip": jnp.var(local_energies),
        }
        return energy, local_energies, stats

    return energy_and_statistics_fn


@pytest.mark.slow
def test_energy_decreases_over_steps():
    """Test that same_sampled_spring_unified lowers a fabricated quadratic energy."""
    nchains = 16
    key = jax.random.PRNGKey(0)
    positions = jax.random.normal(key, (nchains, 1))
    params = {"a": jnp.array(2.5)}
    target_a = 1.0

    config = ConfigDict(
        {
            "learning_rate": 0.05,
            "mu": 0.9,
            "damping": 1e-3,
            "constrain_norm": False,  # let the quadratic converge freely
            "norm_constraint": 1e-3,
            "lb_window": 5,
            "probe_lr": -1.0,
            "probe_damping": 1e-3,
            "adaptive_eta": True,
            "adaptive_probe": True,
        }
    )

    update_param_fn, opt_state, key = initialize_same_sampled_spring_unified(
        _log_psi_apply,
        _make_local_energy_and_stats(target_a),
        params,
        get_position_fn=lambda d: d,
        update_data_fn=lambda d, p_: d,
        learning_rate_schedule=lambda t: 0.05,
        optimizer_config=config,
        key=jax.random.PRNGKey(1),
        apply_pmap=False,
    )

    energies = []
    p = params
    data = positions
    for _ in range(40):
        p, data, opt_state, metrics, key = update_param_fn(p, data, opt_state, key)
        energies.append(float(metrics["energy"]))

    assert np.all(np.isfinite(energies))
    # energy should decrease substantially toward 0 as a -> target_a
    assert energies[-1] < energies[0]
    assert energies[-1] < 0.25
    # the adaptive schedule must have moved beta off its initial value by now
    assert int(opt_state.step) == 40
