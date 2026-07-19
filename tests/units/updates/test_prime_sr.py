"""Unit tests for the PRIME-SR optimizer."""

import jax
import jax.numpy as jnp
import numpy as np

from vmcnet.updates.prime_sr import (
    PRIMESRState,
    _adaptive_mu,
    _spectral_indicators,
)


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
    """mu and beta_tilde are 0 when alpha_prev_ceil == 0 (sentinel) or rank == 0."""
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
