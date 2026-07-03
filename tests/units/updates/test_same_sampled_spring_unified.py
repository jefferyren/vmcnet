"""Unit tests for the same_sampled_spring_unified optimizer."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vmcnet.updates.same_sampled_spring_unified import (
    SameSampledSPRINGUnifiedState,
    _draw_unit_norm_like,
)


def _example_params():
    return {
        "w1": jnp.arange(6.0).reshape(2, 3),
        "b1": jnp.zeros(3),
        "w2": jnp.arange(3.0).reshape(3, 1),
        "b2": jnp.zeros(1),
    }


def test_draw_unit_norm_like_has_global_unit_l2_norm():
    params = _example_params()
    key = jax.random.PRNGKey(0)
    x_star = _draw_unit_norm_like(key, params)

    # same structure as params
    assert jax.tree_util.tree_structure(x_star) == jax.tree_util.tree_structure(params)

    flat = jax.flatten_util.ravel_pytree(x_star)[0]
    np.testing.assert_allclose(jnp.linalg.norm(flat), 1.0, rtol=1e-6)


def test_state_namedtuple_fields():
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
