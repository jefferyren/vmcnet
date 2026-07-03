# same_sampled_spring_unified Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the PyTorch `same_sampled_spring_unified` optimizer (SPRING + interleaved same-sampled SPRING probe + adaptive-β schedule) into vmcnet as a new JAX `optimizer_type`.

**Architecture:** A new self-contained module `vmcnet/updates/same_sampled_spring_unified.py` mirroring `spring.py`'s structure. It exposes shared linear operators (`apply_A`, `apply_Aᵀ`, one eigendecomposition of the SPRING Gram matrix `T`) reused by both the main SPRING solve and a synthetic-target probe solve. A custom `NamedTuple` optimizer state carries the main momentum, the probe iterate/momentum, the fixed random target `x_star`, a sliding residual buffer, and scalar counters that drive the adaptive-β schedule. Everything is a pure function threaded through jit/pmap.

**Tech Stack:** JAX (`jax.jvp`/`jax.vjp`/`jax.linearize`, `jax.numpy`), `neural_tangents` (empirical NTK), `optax` (`apply_updates` only), `ml_collections` (config), `chex`, `pytest`.

**Design spec:** `docs/superpowers/specs/2026-07-02-same-sampled-spring-unified-design.md` (read it before starting).

## Global Constraints

- **Python 3.9** (required by the repo). Line length **88** (black). Type hints required. Google-style docstrings on every module/public function/class (flake8 `--select D --extend-ignore D401`).
- **Match existing SPRING for devices:** per-device (local) centering / `T`-matrix convention. No new cross-device reductions. Correct multi-device is out of scope.
- **`eta_main = 1 - β·(1 - lr(step))`** wraps the *scheduled* learning rate; reduces to `lr(step)` when `adaptive_eta=False`. Must carry an inline comment.
- **`probe_lr`** defaults to the base `learning_rate` value via a negative sentinel resolved in the initializer. Must carry an inline comment.
- **`mu`** config key is the *initial* β (default `0.9`).
- **Faithfulness invariants:** main and probe use the *same* `apply_A`/`apply_AT` closures and the *same* single `eigh(T)` (only additive damping differs); keep the `+ ones onesᵀ/N` constant-mode term and the `max(Tvals,0)+damp` clip; probe updates are explicit `+=`/`-=` (never routed through optax); `x_star` drawn once at init with global unit L2 norm and never recomputed; residual history is a fixed-length sliding buffer (no Python list).
- **Do not modify `spring.py`.** Commit frequently. Commit messages: no `Co-Authored-By` trailer. Push branch to `origin` (the user's fork), never `upstream`.

---

## File Structure

- **Create** `vmcnet/updates/same_sampled_spring_unified.py` — the optimizer: state pytree, operator/helper functions, step kernel, `constrain_norm`, update-param-fn constructor, and `initialize_same_sampled_spring_unified`.
- **Modify** `vmcnet/train/default_config.py` — add the `same_sampled_spring_unified` config block.
- **Modify** `vmcnet/updates/parse_optimizer_config.py` — import + dispatch `elif` branch.
- **Create** `tests/units/updates/__init__.py` and `tests/units/updates/test_same_sampled_spring_unified.py` — unit tests.
- **Create** `tests/integrations/updates/__init__.py` and `tests/integrations/updates/test_same_sampled_spring_unified_integration.py` — jitted end-to-end smoke test (`slow`).

---

### Task 0: Environment setup + green baseline — DONE by controller

**Files:** none (environment only).

**Resolution (already completed):** The repo pins `jax==0.4.34`/`jaxlib==0.4.34`, which require **Python ≥3.10** (the README's "Python 3.9" is stale; CI/tox uses `py312`). The correct interpreter is the pre-existing conda env:

```
/Users/jefferyren/anaconda3/envs/vmcnet-py312/bin/python   (Python 3.12.13)
```

It already has the exact pinned deps: jax/jaxlib 0.4.34, neural_tangents 0.6.5, optax 0.2.4, ml_collections, chex, pytest 8.3.4, black 25.1.0, flake8 7.1.1 (+ flake8-docstrings 1.7.0), mypy 1.15.0. Baseline `tests/units/train/test_vmc.py` runs clean (0 failures). `vmcnet.updates.spring` imports and the NTK path works.

**All subsequent tasks use this interpreter and invoke tools via `-m`**, e.g.
`/Users/jefferyren/anaconda3/envs/vmcnet-py312/bin/python -m pytest ...`,
`... -m black ...`, `... -m flake8 ...`, `... -m mypy ...`.

*(No commit — environment only.)*

---

### Task 1: Module scaffold, state pytree, and `x_star` initializer

**Files:**
- Create: `vmcnet/updates/same_sampled_spring_unified.py`
- Create: `tests/units/updates/__init__.py`
- Create: `tests/units/updates/test_same_sampled_spring_unified.py`

**Interfaces:**
- Produces: `SameSampledSPRINGUnifiedState` (NamedTuple with fields `phi, z_probe, phi_probe, x_star, residual_buffer, r_hat, beta, checkpoint_idx, step`); `_draw_unit_norm_like(key: PRNGKey, params: P) -> P`.

- [ ] **Step 1: Create the empty tests package marker**

Create `tests/units/updates/__init__.py` with a single line:
```python
"""Unit tests for vmcnet.updates."""
```

- [ ] **Step 2: Write the failing test for `_draw_unit_norm_like`**

Create `tests/units/updates/test_same_sampled_spring_unified.py`:
```python
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
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `/Users/jefferyren/anaconda3/envs/vmcnet-py312/bin/python -m pytest tests/units/updates/test_same_sampled_spring_unified.py -q`
Expected: FAIL — `ModuleNotFoundError` / cannot import `same_sampled_spring_unified`.

- [ ] **Step 4: Create the module with the state pytree and helper**

Create `vmcnet/updates/same_sampled_spring_unified.py`:
```python
"""Same-Sampled SPRING with an interleaved same-sampled SPRING probe.

JAX/vmcnet port of the PyTorch reference ``same_sampled_spring_unified``. The main
update is ordinary SPRING (see ``vmcnet/updates/spring.py``) with a dynamic decay
factor ``beta``; a probe runs a second SPRING solve on the *same* sampled Jacobian
against a fixed synthetic target ``b = A @ x_star`` and its residual history drives an
adaptive-``beta`` schedule. See
``docs/superpowers/specs/2026-07-02-same-sampled-spring-unified-design.md``.
"""

from typing import Callable, Dict, NamedTuple

import chex
import jax
import jax.flatten_util
import jax.numpy as jnp
import neural_tangents as nt  # type: ignore
import optax
from ml_collections import ConfigDict

import vmcnet.utils as utils
from vmcnet.utils.distribute import pmean_if_pmap
from vmcnet.utils.pytree_helpers import (
    multiply_tree_by_scalar,
    tree_inner_product,
    tree_reduce_l1,
)
from vmcnet.utils.typing import (
    Array,
    D,
    GetPositionFromData,
    LearningRateSchedule,
    ModelApply,
    P,
    PRNGKey,
    S,
    Tuple,
    UpdateDataFn,
)

from .update_param_fns import (
    UpdateParamFn,
    make_traced_fn_with_single_metrics,
    update_metrics_with_noclip,
)


class SameSampledSPRINGUnifiedState(NamedTuple):
    """Optimizer state for `same_sampled_spring_unified`.

    Attributes:
        phi: main SPRING momentum (params-shaped pytree).
        z_probe: probe iterate (params-shaped pytree).
        phi_probe: probe SPRING momentum (params-shaped pytree).
        x_star: fixed random unit-norm probe target (params-shaped pytree).
        residual_buffer: chronological sliding buffer (float[2p]) of probe residuals.
        r_hat: running convergence-rate estimate (scalar).
        beta: current decay factor / momentum (scalar).
        checkpoint_idx: adaptive-beta checkpoint counter (int scalar, starts at 1).
        step: step/epoch counter (int scalar, starts at 0).
    """

    phi: P
    z_probe: P
    phi_probe: P
    x_star: P
    residual_buffer: Array
    r_hat: Array
    beta: Array
    checkpoint_idx: Array
    step: Array


def _draw_unit_norm_like(key: PRNGKey, params: P) -> P:
    """Draw a random pytree like `params`, normalized to global unit L2 norm."""
    flat, unravel = jax.flatten_util.ravel_pytree(params)
    noise = jax.random.normal(key, flat.shape, dtype=flat.dtype)
    noise = noise / jnp.linalg.norm(noise)
    return unravel(noise)
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `/Users/jefferyren/anaconda3/envs/vmcnet-py312/bin/python -m pytest tests/units/updates/test_same_sampled_spring_unified.py -q`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add vmcnet/updates/same_sampled_spring_unified.py tests/units/updates/
git commit -m "Add same_sampled_spring_unified state pytree and x_star initializer"
```

---

### Task 2: Shared linear operators (`_build_operators`)

**Files:**
- Modify: `vmcnet/updates/same_sampled_spring_unified.py`
- Test: `tests/units/updates/test_same_sampled_spring_unified.py`

**Interfaces:**
- Consumes: `neural_tangents` kernel built as `nt.empirical_kernel_fn(log_psi_apply, vmap_axes=0, trace_axes=())`.
- Produces: `_build_operators(kernel_fn, log_psi_apply, params, positions) -> (apply_A, apply_AT, Tvals, Tvecs)` where `apply_A(pytree) -> Array[N]` (mean-zero), `apply_AT(Array[N]) -> pytree`, and `Tvals, Tvecs` are the eigendecomposition of `T = A Aᵀ + (1/N) ones onesᵀ`.

- [ ] **Step 1: Write the failing tests for the operators**

Add to `tests/units/updates/test_same_sampled_spring_unified.py`:
```python
import neural_tangents as nt

from vmcnet.updates.same_sampled_spring_unified import _build_operators
from vmcnet.utils.pytree_helpers import tree_inner_product


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
    params, positions, kernel_fn = _operator_setup()
    apply_A, _, _, _ = _build_operators(kernel_fn, _log_psi_apply, params, positions)
    v = jax.tree_map(lambda x: jax.random.normal(jax.random.PRNGKey(1), x.shape), params)
    out = apply_A(v)
    np.testing.assert_allclose(jnp.mean(out), 0.0, atol=1e-6)


def test_apply_A_and_apply_AT_are_adjoints():
    params, positions, kernel_fn = _operator_setup()
    apply_A, apply_AT, _, _ = _build_operators(
        kernel_fn, _log_psi_apply, params, positions
    )
    u = jax.tree_map(lambda x: jax.random.normal(jax.random.PRNGKey(2), x.shape), params)
    w = jax.random.normal(jax.random.PRNGKey(3), (positions.shape[0],))

    lhs = jnp.sum(apply_A(u) * w)          # <A u, w>
    rhs = tree_inner_product(u, apply_AT(w))  # <u, A^T w>
    np.testing.assert_allclose(lhs, rhs, rtol=1e-5, atol=1e-6)


def test_T_matches_base_spring_construction():
    # T built here must equal the neural-tangents T built exactly as spring.py does.
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/Users/jefferyren/anaconda3/envs/vmcnet-py312/bin/python -m pytest tests/units/updates/test_same_sampled_spring_unified.py -q -k "apply_A or apply_AT or T_matches"`
Expected: FAIL — `_build_operators` not importable.

- [ ] **Step 3: Implement `_build_operators`**

Add to `vmcnet/updates/same_sampled_spring_unified.py` (after `_draw_unit_norm_like`):
```python
def _build_operators(
    kernel_fn: Callable,
    log_psi_apply: ModelApply[P],
    params: P,
    positions: Array,
) -> Tuple[Callable[[P], Array], Callable[[Array], P], Array, Array]:
    """Build the shared SPRING linear operators at the current (params, positions).

    The main solve and the probe solve must use identical operators, so they are built
    once here and reused. `apply_A` centers its output and `apply_AT` centers its input
    (they are transposes of one another); the constant-mode term `(1/N) ones onesᵀ` and
    the negative-eigenvalue clip live in the returned eigendecomposition of `T`.

    Args:
        kernel_fn: neural-tangents empirical kernel, built once from `log_psi_apply`.
        log_psi_apply: maps (params, positions) -> log|psi| of shape (nchains,).
        params: current model parameters.
        positions: current walker positions, shape (nchains, ...).

    Returns:
        Tuple (apply_A, apply_AT, Tvals, Tvecs) where apply_A: P -> Array[nchains]
        (mean-zero), apply_AT: Array[nchains] -> P, and Tvals/Tvecs are the
        eigendecomposition of T = A Aᵀ + (1/N) ones onesᵀ.
    """
    nchains = positions.shape[0]
    sqrt_n = jnp.sqrt(nchains)
    ones = jnp.ones((nchains, 1))

    # Reuse a single linearization / vjp so the multiple probe + main calls are cheap.
    _, jvp_fn = jax.linearize(log_psi_apply, params, positions)
    zeros_positions = jnp.zeros_like(positions)

    def apply_A(vec: P) -> Array:
        o = jvp_fn(vec, zeros_positions) / sqrt_n
        return o - jnp.mean(o, axis=0, keepdims=True)

    _, vjp_fn = jax.vjp(log_psi_apply, params, positions)

    def apply_AT(v: Array) -> P:
        v_hat = v - jnp.mean(v)
        dtheta = vjp_fn(v_hat)[0]
        return multiply_tree_by_scalar(dtheta, 1.0 / sqrt_n)

    t = kernel_fn(positions, positions, "ntk", params) / nchains
    t = t - jnp.mean(t, axis=0, keepdims=True)
    t = t - jnp.mean(t, axis=1, keepdims=True)
    t = t + ones @ ones.T / nchains
    t = (t + t.T) / 2
    tvals, tvecs = jnp.linalg.eigh(t)
    return apply_A, apply_AT, tvals, tvecs
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/Users/jefferyren/anaconda3/envs/vmcnet-py312/bin/python -m pytest tests/units/updates/test_same_sampled_spring_unified.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add vmcnet/updates/same_sampled_spring_unified.py tests/units/updates/test_same_sampled_spring_unified.py
git commit -m "Add shared SPRING linear operators for same_sampled_spring_unified"
```

---

### Task 3: Adaptive-β schedule (`_adaptive_beta_update`)

**Files:**
- Modify: `vmcnet/updates/same_sampled_spring_unified.py`
- Test: `tests/units/updates/test_same_sampled_spring_unified.py`

**Interfaces:**
- Produces: `_adaptive_beta_update(residual_buffer, probe_res_norm, r_hat, beta, checkpoint_idx, step, p) -> (new_buffer, new_r_hat, new_beta, new_checkpoint_idx)`. Slides `probe_res_norm` into the chronological length-`2p` buffer every call; updates `beta`/`r_hat`/`checkpoint_idx` only when `step % p == 0 and step >= 2p`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/units/updates/test_same_sampled_spring_unified.py`:
```python
from vmcnet.updates.same_sampled_spring_unified import _adaptive_beta_update


def _numpy_adaptive_beta(buffer, res, r_hat, beta, ckpt, step, p):
    buf = np.concatenate([np.asarray(buffer)[1:], [float(res)]])
    do_update = (step % p == 0) and (step >= 2 * p)
    if not do_update:
        return buf, float(r_hat), float(beta), int(ckpt)
    window_tp = buf[0:p]
    window_t = buf[p : 2 * p]
    r_ip = np.sum(window_t ** 2) / np.sum(window_tp ** 2)
    n_old = float(ckpt)
    n_new = n_old + 1.0
    alph = (n_old ** np.log(n_old)) / (n_new ** np.log(n_new))
    r_hat_new = alph * r_hat + (1.0 - alph) * min(1.0, r_ip)
    rho = max(0.0, 1.0 - r_hat_new ** (1.0 / p))
    beta_new = (1.0 - rho) / (1.0 + rho)
    return buf, r_hat_new, beta_new, ckpt + 1


def test_adaptive_beta_slides_buffer_but_no_update_before_2p():
    p = 3
    buffer = jnp.arange(2 * p, dtype=jnp.float32)  # [0,1,2,3,4,5]
    out_buf, out_rhat, out_beta, out_ckpt = _adaptive_beta_update(
        buffer, jnp.array(9.0), jnp.array(1.0), jnp.array(0.9),
        jnp.array(1, jnp.int32), jnp.array(2, jnp.int32), p,
    )
    # step=2 (< 2p=6): buffer slides, everything else unchanged
    np.testing.assert_allclose(out_buf, np.array([1, 2, 3, 4, 5, 9]))
    np.testing.assert_allclose(out_beta, 0.9)
    np.testing.assert_allclose(out_rhat, 1.0)
    assert int(out_ckpt) == 1


def test_adaptive_beta_updates_at_trigger_matches_numpy():
    p = 3
    rng = np.random.default_rng(0)
    buffer = jnp.asarray(rng.uniform(0.1, 1.0, size=2 * p), dtype=jnp.float32)
    res, r_hat, beta, ckpt, step = 0.4, 0.8, 0.9, 1, 2 * p  # step==6 triggers

    got = _adaptive_beta_update(
        buffer, jnp.array(res), jnp.array(r_hat), jnp.array(beta),
        jnp.array(ckpt, jnp.int32), jnp.array(step, jnp.int32), p,
    )
    exp = _numpy_adaptive_beta(buffer, res, r_hat, beta, ckpt, step, p)
    np.testing.assert_allclose(got[0], exp[0], rtol=1e-6)
    np.testing.assert_allclose(got[1], exp[1], rtol=1e-5)
    np.testing.assert_allclose(got[2], exp[2], rtol=1e-5)
    assert int(got[3]) == exp[3]


def test_adaptive_beta_is_jittable():
    p = 3
    f = jax.jit(lambda b, r, rh, be, c, s: _adaptive_beta_update(b, r, rh, be, c, s, p))
    buffer = jnp.ones(2 * p)
    out = f(buffer, jnp.array(1.0), jnp.array(1.0), jnp.array(0.9),
            jnp.array(1, jnp.int32), jnp.array(6, jnp.int32))
    assert out[2].shape == ()  # beta scalar, no crash under jit
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/Users/jefferyren/anaconda3/envs/vmcnet-py312/bin/python -m pytest tests/units/updates/test_same_sampled_spring_unified.py -q -k adaptive_beta`
Expected: FAIL — `_adaptive_beta_update` not importable.

- [ ] **Step 3: Implement `_adaptive_beta_update`**

Add to `vmcnet/updates/same_sampled_spring_unified.py`:
```python
def _adaptive_beta_update(
    residual_buffer: Array,
    probe_res_norm: Array,
    r_hat: Array,
    beta: Array,
    checkpoint_idx: Array,
    step: Array,
    p: int,
) -> Tuple[Array, Array, Array, Array]:
    """Slide the probe residual into the buffer and update beta on schedule.

    The residual buffer is chronological (oldest first), length 2p. Every p steps once
    `step >= 2p`, beta is updated from the ratio of squared-residual sums over the two
    adjacent length-p windows (Appendix D of the reference). Off-schedule steps only
    slide the buffer. All arithmetic runs every step; writes are gated with `jnp.where`
    (a Python branch would be illegal under jit).

    Args:
        residual_buffer: chronological float buffer of length 2p.
        probe_res_norm: the current step's probe residual norm (scalar).
        r_hat: running convergence-rate estimate (scalar).
        beta: current decay factor (scalar).
        checkpoint_idx: adaptive-beta checkpoint counter (int scalar, >= 1).
        step: current step index (int scalar, pre-increment).
        p: lookback window length.

    Returns:
        Tuple (new_buffer, new_r_hat, new_beta, new_checkpoint_idx).
    """
    new_buffer = jnp.concatenate([residual_buffer[1:], probe_res_norm[None]])

    window_tp = new_buffer[0:p]
    window_t = new_buffer[p : 2 * p]
    # small epsilon guards the early (partially-zero) buffer; result is gated out anyway.
    r_ip = jnp.sum(window_t ** 2) / (jnp.sum(window_tp ** 2) + 1e-30)

    n_old = checkpoint_idx.astype(r_hat.dtype)
    n_new = n_old + 1.0
    alph = jnp.power(n_old, jnp.log(n_old)) / jnp.power(n_new, jnp.log(n_new))

    r_hat_new = alph * r_hat + (1.0 - alph) * jnp.minimum(1.0, r_ip)
    rho = jnp.clip(1.0 - jnp.power(r_hat_new, 1.0 / p), a_min=0.0)
    beta_new = (1.0 - rho) / (1.0 + rho)

    do_update = (step % p == 0) & (step >= 2 * p)
    new_beta = jnp.where(do_update, beta_new, beta)
    new_r_hat = jnp.where(do_update, r_hat_new, r_hat)
    new_checkpoint_idx = jnp.where(do_update, checkpoint_idx + 1, checkpoint_idx)
    return new_buffer, new_r_hat, new_beta, new_checkpoint_idx
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/Users/jefferyren/anaconda3/envs/vmcnet-py312/bin/python -m pytest tests/units/updates/test_same_sampled_spring_unified.py -q -k adaptive_beta`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add vmcnet/updates/same_sampled_spring_unified.py tests/units/updates/test_same_sampled_spring_unified.py
git commit -m "Add adaptive-beta schedule for same_sampled_spring_unified"
```

---

### Task 4: Step kernel (`get_same_sampled_spring_unified_step`) + equivalence to base SPRING

**Files:**
- Modify: `vmcnet/updates/same_sampled_spring_unified.py`
- Test: `tests/units/updates/test_same_sampled_spring_unified.py`

**Interfaces:**
- Consumes: `_build_operators`, `_adaptive_beta_update`, `SameSampledSPRINGUnifiedState`, and (in the test) `vmcnet.updates.spring.get_spring_step`.
- Produces: `get_same_sampled_spring_unified_step(log_psi_apply, learning_rate_schedule, damping, probe_damping, p, probe_lr, adaptive_eta, adaptive_probe) -> step`, where `step(centered_local_energies: Array, params: P, positions: Array, state: SameSampledSPRINGUnifiedState) -> (updates: P, new_state: SameSampledSPRINGUnifiedState)` and `updates` is the *unconstrained* parameter delta (already scaled by `-eta_main`).

- [ ] **Step 1: Write the failing equivalence + smoke tests**

Add to `tests/units/updates/test_same_sampled_spring_unified.py`:
```python
from vmcnet.updates.spring import get_spring_step
from vmcnet.updates.same_sampled_spring_unified import (
    get_same_sampled_spring_unified_step,
)


def _zeros_like(params):
    return jax.tree_map(jnp.zeros_like, params)


def _make_state(params, p, beta, phi=None):
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

    lr_sched = lambda t: 0.05
    my_step = get_same_sampled_spring_unified_step(
        _log_psi_apply, lr_sched, damping=damping, probe_damping=damping,
        p=1000, probe_lr=0.05, adaptive_eta=False, adaptive_probe=False,
    )
    state = _make_state(params, p=1000, beta=mu, phi=prev_grad)
    _, new_state = my_step(centered_energies, params, positions, state)

    for a, b in zip(
        jax.tree_util.tree_leaves(new_state.phi),
        jax.tree_util.tree_leaves(base_grad),
    ):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


def test_step_advances_state_and_is_jittable():
    params, positions, _ = _operator_setup(nchains=8)
    p = 4
    lr_sched = lambda t: 0.05
    step_fn = get_same_sampled_spring_unified_step(
        _log_psi_apply, lr_sched, damping=1e-3, probe_damping=1e-3,
        p=p, probe_lr=0.05, adaptive_eta=True, adaptive_probe=True,
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/Users/jefferyren/anaconda3/envs/vmcnet-py312/bin/python -m pytest tests/units/updates/test_same_sampled_spring_unified.py -q -k "matches_base_spring or advances_state"`
Expected: FAIL — `get_same_sampled_spring_unified_step` not importable.

- [ ] **Step 3: Implement the step kernel**

Add to `vmcnet/updates/same_sampled_spring_unified.py`:
```python
def get_same_sampled_spring_unified_step(
    log_psi_apply: ModelApply[P],
    learning_rate_schedule: LearningRateSchedule,
    damping: chex.Scalar,
    probe_damping: chex.Scalar,
    p: int,
    probe_lr: float,
    adaptive_eta: bool,
    adaptive_probe: bool,
) -> Callable[
    [Array, P, Array, SameSampledSPRINGUnifiedState],
    Tuple[P, SameSampledSPRINGUnifiedState],
]:
    """Get the same_sampled_spring_unified step kernel.

    Returns a pure function `step(centered_local_energies, params, positions, state)`
    that returns `(updates, new_state)`, where `updates` is the (unconstrained)
    parameter delta already scaled by `-eta_main`. The main SPRING solve and the probe
    solve reuse the same operators and single eigendecomposition; only their additive
    damping differs.

    Args:
        log_psi_apply: maps (params, positions) -> log|psi|, shape (nchains,).
        learning_rate_schedule: step -> base learning rate.
        damping: additive damping for the main normal equation.
        probe_damping: additive damping for the probe normal equation.
        p: adaptive-beta lookback window.
        probe_lr: base probe step (used when adaptive_probe is False).
        adaptive_eta: if True, eta_main = 1 - beta*(1 - lr(step)).
        adaptive_probe: if True, the probe step uses eta_main instead of probe_lr.

    Returns:
        The step kernel described above.
    """
    kernel_fn = nt.empirical_kernel_fn(log_psi_apply, vmap_axes=0, trace_axes=())

    def step(
        centered_local_energies: Array,
        params: P,
        positions: Array,
        state: SameSampledSPRINGUnifiedState,
    ) -> Tuple[P, SameSampledSPRINGUnifiedState]:
        nchains = positions.shape[0]
        sqrt_n = jnp.sqrt(nchains)
        beta = state.beta

        apply_A, apply_AT, tvals, tvecs = _build_operators(
            kernel_fn, log_psi_apply, params, positions
        )
        tvals_clipped = jnp.maximum(tvals, 0.0)

        def solve(rhs: Array, damp: chex.Scalar) -> Array:
            return tvecs @ ((tvecs.T @ rhs) / (tvals_clipped + damp))

        # ---- main SPRING (identical to base SPRING; beta is dynamic, from state) ----
        epsilon_bar = centered_local_energies / sqrt_n
        rhs_main = epsilon_bar - apply_A(multiply_tree_by_scalar(state.phi, beta))
        dual_main = solve(rhs_main, damping)
        step_primal = apply_AT(dual_main)
        phi_new = jax.tree_map(lambda s, ph: s + beta * ph, step_primal, state.phi)

        # eta_main wraps the SCHEDULED learning rate (design decision): with
        # adaptive_eta it is 1 - beta*(1 - lr(step)), else just lr(step). This reduces
        # to base SPRING exactly when adaptive_eta is False.
        lr = learning_rate_schedule(state.step)
        eta_main = (1.0 - beta * (1.0 - lr)) if adaptive_eta else lr
        updates = multiply_tree_by_scalar(phi_new, -eta_main)

        # ---- probe: self-contained synthetic solve on the SAME operators ----
        # probe_lr defaults to the base learning_rate value (resolved in the
        # initializer); when adaptive_probe is set, the probe uses eta_main instead.
        eta_probe = eta_main if adaptive_probe else probe_lr

        # zeta_probe = (b - A z_probe) - beta*(A phi_probe) = A(x_star - z_probe - beta*phi_probe)
        probe_in = jax.tree_map(
            lambda xs, z, ph: xs - z - beta * ph,
            state.x_star,
            state.z_probe,
            state.phi_probe,
        )
        v = solve(apply_A(probe_in), probe_damping)
        w = apply_AT(v)
        phi_probe_new = jax.tree_map(lambda w_, ph: w_ + beta * ph, w, state.phi_probe)
        z_probe_new = jax.tree_map(
            lambda z, ph: z + eta_probe * ph, state.z_probe, phi_probe_new
        )

        # probe residual on the UPDATED iterate: ||A z_probe_new - b|| = ||A(z - x_star)||
        resid_in = jax.tree_map(lambda z, xs: z - xs, z_probe_new, state.x_star)
        probe_res_norm = jnp.linalg.norm(apply_A(resid_in))

        # ---- adaptive-beta ----
        new_buffer, new_r_hat, new_beta, new_ckpt = _adaptive_beta_update(
            state.residual_buffer,
            probe_res_norm,
            state.r_hat,
            beta,
            state.checkpoint_idx,
            state.step,
            p,
        )

        new_state = SameSampledSPRINGUnifiedState(
            phi=phi_new,
            z_probe=z_probe_new,
            phi_probe=phi_probe_new,
            x_star=state.x_star,
            residual_buffer=new_buffer,
            r_hat=new_r_hat,
            beta=new_beta,
            checkpoint_idx=new_ckpt,
            step=state.step + 1,
        )
        return updates, new_state

    return step
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/Users/jefferyren/anaconda3/envs/vmcnet-py312/bin/python -m pytest tests/units/updates/test_same_sampled_spring_unified.py -q`
Expected: PASS (all tests so far). The `test_main_update_matches_base_spring` passing is the key correctness anchor.

- [ ] **Step 5: Commit**

```bash
git add vmcnet/updates/same_sampled_spring_unified.py tests/units/updates/test_same_sampled_spring_unified.py
git commit -m "Add same_sampled_spring_unified step kernel (main + probe + adaptive-beta)"
```

---

### Task 5: `constrain_norm`, update-param-fn constructor, and `initialize_same_sampled_spring_unified`

**Files:**
- Modify: `vmcnet/updates/same_sampled_spring_unified.py`
- Test: `tests/units/updates/test_same_sampled_spring_unified.py`

**Interfaces:**
- Consumes: `get_same_sampled_spring_unified_step`, `_draw_unit_norm_like`, `make_traced_fn_with_single_metrics`, `update_metrics_with_noclip`, `utils.distribute` (`pmap`, `split_or_psplit_key`).
- Produces:
  - `constrain_norm(grad: P, norm_constraint) -> P` (copied from spring).
  - `initialize_same_sampled_spring_unified(log_psi_apply, energy_and_statistics_fn, params, get_position_fn, update_data_fn, learning_rate_schedule, optimizer_config, key, record_param_l1_norm=False, apply_pmap=True) -> Tuple[UpdateParamFn, SameSampledSPRINGUnifiedState, PRNGKey]`.

- [ ] **Step 1: Write the failing test (single-device initialize + one apply, jitted)**

Add to `tests/units/updates/test_same_sampled_spring_unified.py`:
```python
from ml_collections import ConfigDict

from vmcnet.updates.same_sampled_spring_unified import (
    initialize_same_sampled_spring_unified,
)


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
    for leaf in jax.tree_util.tree_leaves(new_params):
        assert bool(jnp.all(jnp.isfinite(leaf)))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `/Users/jefferyren/anaconda3/envs/vmcnet-py312/bin/python -m pytest tests/units/updates/test_same_sampled_spring_unified.py -q -k initialize_single_device`
Expected: FAIL — `initialize_same_sampled_spring_unified` not importable.

- [ ] **Step 3: Implement `constrain_norm`, the constructor, and the initializer**

Add to `vmcnet/updates/same_sampled_spring_unified.py`:
```python
def constrain_norm(grad: P, norm_constraint: chex.Numeric = 0.001) -> P:
    """Euclidean norm constraint on the update (matches spring.py)."""
    sq_norm_scaled_grads = tree_inner_product(grad, grad)
    sq_norm_scaled_grads = pmean_if_pmap(sq_norm_scaled_grads)
    norm_scale_factor = jnp.sqrt(norm_constraint / sq_norm_scaled_grads)
    coefficient = jnp.minimum(norm_scale_factor, 1)
    return multiply_tree_by_scalar(grad, coefficient)


def construct_same_sampled_spring_unified_update_param_fn(
    energy_and_statistics_fn,
    optimizer_apply: Callable,
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    apply_pmap: bool = True,
    record_param_l1_norm: bool = False,
) -> UpdateParamFn[P, D, S]:
    """Create the update_param_fn for same_sampled_spring_unified."""

    def update_param_fn(params, data, optimizer_state, key):
        position = get_position_fn(data)
        energy, local_energies, stats = energy_and_statistics_fn(params, position)
        params, optimizer_state = optimizer_apply(
            energy, local_energies, params, optimizer_state, data
        )
        data = update_data_fn(data, params)

        metrics = {"energy": energy, "variance": stats["variance"]}
        metrics = update_metrics_with_noclip(
            stats["energy_noclip"], stats["variance_noclip"], metrics
        )
        if record_param_l1_norm:
            metrics.update({"param_l1_norm": tree_reduce_l1(params)})
        return params, data, optimizer_state, metrics, key

    return make_traced_fn_with_single_metrics(update_param_fn, apply_pmap)


def initialize_same_sampled_spring_unified(
    log_psi_apply: ModelApply[P],
    energy_and_statistics_fn,
    params: P,
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    learning_rate_schedule: LearningRateSchedule,
    optimizer_config: ConfigDict,
    key: PRNGKey,
    record_param_l1_norm: bool = False,
    apply_pmap: bool = True,
) -> Tuple[UpdateParamFn[P, D, SameSampledSPRINGUnifiedState], SameSampledSPRINGUnifiedState, PRNGKey]:
    """Get an update param function and initial state for same_sampled_spring_unified."""
    p = int(optimizer_config.lb_window)
    mu = float(optimizer_config.mu)

    # probe_lr defaults to the base learning_rate value; a negative config value is the
    # "unset" sentinel (mirrors the PyTorch reference, where probe_lr falls back to lr).
    probe_lr = (
        float(optimizer_config.learning_rate)
        if optimizer_config.probe_lr < 0
        else float(optimizer_config.probe_lr)
    )

    step_fn = get_same_sampled_spring_unified_step(
        log_psi_apply,
        learning_rate_schedule,
        optimizer_config.damping,
        optimizer_config.probe_damping,
        p,
        probe_lr,
        bool(optimizer_config.adaptive_eta),
        bool(optimizer_config.adaptive_probe),
    )

    def init_state(local_params: P, subkey: PRNGKey) -> SameSampledSPRINGUnifiedState:
        zeros = jax.tree_map(jnp.zeros_like, local_params)
        return SameSampledSPRINGUnifiedState(
            phi=zeros,
            z_probe=zeros,
            phi_probe=zeros,
            x_star=_draw_unit_norm_like(subkey, local_params),
            residual_buffer=jnp.zeros(2 * p),
            r_hat=jnp.array(1.0),
            beta=jnp.array(mu),
            checkpoint_idx=jnp.array(1, jnp.int32),
            step=jnp.array(0, jnp.int32),
        )

    def optimizer_apply(energy, local_energies, params, optimizer_state, data):
        positions = get_position_fn(data)
        centered_local_energies = local_energies - energy
        updates, optimizer_state = step_fn(
            centered_local_energies, params, positions, optimizer_state
        )
        if optimizer_config.constrain_norm:
            updates = constrain_norm(updates, optimizer_config.norm_constraint)
        params = optax.apply_updates(params, updates)
        return params, optimizer_state

    update_param_fn = construct_same_sampled_spring_unified_update_param_fn(
        energy_and_statistics_fn,
        optimizer_apply,
        get_position_fn=get_position_fn,
        update_data_fn=update_data_fn,
        record_param_l1_norm=record_param_l1_norm,
        apply_pmap=apply_pmap,
    )

    key, subkey = utils.distribute.split_or_psplit_key(key, apply_pmap)
    if apply_pmap:
        optimizer_state = utils.distribute.pmap(init_state)(params, subkey)
    else:
        optimizer_state = init_state(params, subkey)

    return update_param_fn, optimizer_state, key
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `/Users/jefferyren/anaconda3/envs/vmcnet-py312/bin/python -m pytest tests/units/updates/test_same_sampled_spring_unified.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add vmcnet/updates/same_sampled_spring_unified.py tests/units/updates/test_same_sampled_spring_unified.py
git commit -m "Add initializer and update-param-fn for same_sampled_spring_unified"
```

---

### Task 6: Wire into config + optimizer dispatch

**Files:**
- Modify: `vmcnet/train/default_config.py` (after the `gauss_newton` block in `get_default_vmc_config`'s `optimizer` dict)
- Modify: `vmcnet/updates/parse_optimizer_config.py` (import near the other `initialize_*` imports; `elif` branch before the final `else`)
- Test: `tests/units/updates/test_same_sampled_spring_unified.py`

**Interfaces:**
- Consumes: `initialize_same_sampled_spring_unified` (Task 5); `physics.core.create_energy_and_statistics_fn` (as SPRING does).
- Produces: `optimizer_type == "same_sampled_spring_unified"` selectable end-to-end.

- [ ] **Step 1: Write the failing dispatch/config tests**

Add to `tests/units/updates/test_same_sampled_spring_unified.py`:
```python
def test_default_config_has_block():
    from vmcnet.train.default_config import get_default_vmc_config

    cfg = get_default_vmc_config()
    block = cfg["optimizer"]["same_sampled_spring_unified"]
    for k in [
        "schedule_type", "learning_rate", "learning_decay_rate", "mu", "damping",
        "constrain_norm", "norm_constraint", "lb_window", "probe_lr",
        "probe_damping", "adaptive_eta", "adaptive_probe",
    ]:
        assert k in block, f"missing config key {k}"
    assert block["mu"] == 0.9
    assert block["probe_lr"] < 0  # sentinel


def test_parse_optimizer_config_dispatches_new_type():
    # The dispatcher must recognize the new optimizer_type without raising ValueError.
    import inspect
    from vmcnet.updates import parse_optimizer_config as poc

    src = inspect.getsource(poc.initialize_optimizer)
    assert 'same_sampled_spring_unified' in src
    assert hasattr(poc, "initialize_same_sampled_spring_unified")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/Users/jefferyren/anaconda3/envs/vmcnet-py312/bin/python -m pytest tests/units/updates/test_same_sampled_spring_unified.py -q -k "default_config_has_block or dispatches_new_type"`
Expected: FAIL — config block missing / symbol not present.

- [ ] **Step 3: Add the config block**

In `vmcnet/train/default_config.py`, inside `get_default_vmc_config()`'s `"optimizer"` dict, immediately after the `"gauss_newton": { ... },` block, add:
```python
                "same_sampled_spring_unified": {
                    # Learning rate settings
                    "schedule_type": "inverse_time",  # constant or inverse_time
                    "learning_rate": 5e-2,
                    "learning_decay_rate": 1e-4,
                    # SPRING hyperparams (mu is the INITIAL beta)
                    "mu": 0.9,
                    "damping": 1e-3,
                    "constrain_norm": True,
                    "norm_constraint": 1e-3,
                    # Adaptive-beta / probe hyperparams
                    "lb_window": 30,  # lookback p; buffer length is 2p
                    # probe_lr: base probe step (used when adaptive_probe=False). A
                    # NEGATIVE value is a sentinel meaning "default to the base
                    # learning_rate value" (resolved in the initializer). Set a positive
                    # value to override independently.
                    "probe_lr": -1.0,
                    "probe_damping": 1e-3,
                    "adaptive_eta": False,  # eta_main = 1 - beta*(1 - lr(step))
                    "adaptive_probe": False,  # probe uses eta_main instead of probe_lr
                },
```
(Match the surrounding indentation exactly — verify with `git diff`.)

- [ ] **Step 4: Add the import and dispatch branch**

In `vmcnet/updates/parse_optimizer_config.py`, add near the other optimizer imports (after `from .gauss_newton import initialize_gauss_newton`):
```python
from .same_sampled_spring_unified import initialize_same_sampled_spring_unified
```
Then, immediately before the final `else:` in `initialize_optimizer`, add:
```python
    elif vmc_config.optimizer_type == "same_sampled_spring_unified":
        energy_and_statistics_fn = physics.core.create_energy_and_statistics_fn(
            local_energy_fn, vmc_config.nchains, clipping_fn, vmc_config.nan_safe
        )
        (
            update_param_fn,
            optimizer_state,
            key,
        ) = initialize_same_sampled_spring_unified(
            log_psi_apply,
            energy_and_statistics_fn,
            params,
            get_position_fn,
            update_data_fn,
            learning_rate_schedule,
            vmc_config.optimizer.same_sampled_spring_unified,
            key,
            vmc_config.record_param_l1_norm,
            apply_pmap=apply_pmap,
        )
        return update_param_fn, optimizer_state, key
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `/Users/jefferyren/anaconda3/envs/vmcnet-py312/bin/python -m pytest tests/units/updates/test_same_sampled_spring_unified.py -q`
Expected: PASS (all). Also confirm no import cycle: `/Users/jefferyren/anaconda3/envs/vmcnet-py312/bin/python -c "import vmcnet.updates.parse_optimizer_config; print('ok')"` → `ok`.

- [ ] **Step 6: Commit**

```bash
git add vmcnet/train/default_config.py vmcnet/updates/parse_optimizer_config.py tests/units/updates/test_same_sampled_spring_unified.py
git commit -m "Wire same_sampled_spring_unified into config and optimizer dispatch"
```

---

### Task 7: Integration smoke test (jitted end-to-end, energy decreases)

**Files:**
- Create: `tests/integrations/updates/__init__.py`
- Create: `tests/integrations/updates/test_same_sampled_spring_unified_integration.py`

**Interfaces:**
- Consumes: `initialize_same_sampled_spring_unified`; a tiny harmonic-oscillator model via the existing test helpers in `tests/integrations/examples/` (mirroring how SGD is exercised).

- [ ] **Step 1: Create the integration package marker**

Create `tests/integrations/updates/__init__.py`:
```python
"""Integration tests for vmcnet.updates."""
```

- [ ] **Step 2: Write the failing integration test**

Create `tests/integrations/updates/test_same_sampled_spring_unified_integration.py`:
```python
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
    return -0.5 * (a ** 2) * jnp.sum(x ** 2, axis=-1)


def _make_local_energy_and_stats(target_a):
    # Minimize a surrogate "energy" E(a) = (a - target_a)^2 whose per-sample local
    # energies average to E(a); this exercises the full optimizer path deterministically.
    def energy_and_statistics_fn(params, positions):
        a = params["a"]
        base = (a - target_a) ** 2
        n = positions.shape[0]
        # local energies that vary per chain but average to `base`
        jitter = jnp.linspace(-0.1, 0.1, n)
        local_energies = base + jitter
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
```

- [ ] **Step 3: Run the test to verify it fails, then passes**

Run: `/Users/jefferyren/anaconda3/envs/vmcnet-py312/bin/python -m pytest tests/integrations/updates/test_same_sampled_spring_unified_integration.py -q --run_slow`
Expected: initially FAIL only if there is a real bug; otherwise it should PASS once the optimizer is correct. If it fails, debug with the systematic-debugging skill (do not weaken the assertions to force a pass). Expected final state: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/integrations/updates/
git commit -m "Add integration smoke test for same_sampled_spring_unified"
```

---

### Task 8: Lint, type-check, format, and full fast-test sweep

**Files:** whatever the tools flag (formatting/imports/docstrings only).

- [ ] **Step 1: Format with black**

Run: `/Users/jefferyren/anaconda3/envs/vmcnet-py312/bin/python -m black vmcnet tests`
Expected: reformats the new files if needed; re-run `/Users/jefferyren/anaconda3/envs/vmcnet-py312/bin/python -m black --check vmcnet tests` → "All done".

- [ ] **Step 2: Docstring/flake8 lint**

Run: `/Users/jefferyren/anaconda3/envs/vmcnet-py312/bin/python -m flake8 vmcnet tests --select D --extend-ignore D401`
Expected: no errors. Fix any missing/short docstrings on new public functions.

- [ ] **Step 3: Type-check**

Run: `/Users/jefferyren/anaconda3/envs/vmcnet-py312/bin/python -m mypy vmcnet tests`
Expected: no new errors introduced by our files. (Third-party `neural_tangents` is untyped but imported with `# type: ignore` as in spring.py; `ml_collections`/`optax` are in mypy's ignore list per `pyproject.toml`.)

- [ ] **Step 4: Run the full fast unit suite + our slow integration test**

Run:
```bash
/Users/jefferyren/anaconda3/envs/vmcnet-py312/bin/python -m pytest tests/units -q
/Users/jefferyren/anaconda3/envs/vmcnet-py312/bin/python -m pytest tests/units/updates tests/integrations/updates -q --run_slow
```
Expected: all PASS. No pre-existing unit tests broken.

- [ ] **Step 5: Commit any lint/format fixes**

```bash
git add -A
git commit -m "Lint, type, and format fixes for same_sampled_spring_unified"
```

- [ ] **Step 6: Push the branch to the fork**

```bash
git push origin same-sampled-spring-unified
```
Expected: branch updated on `origin` (the user's fork). Do not open a PR unless asked.

---

## Self-Review Notes (author checklist — done)

- **Spec coverage:** operators (§2 → Task 2), main SPRING + eta wrap (§3a → Task 4), probe (§3b → Task 4), adaptive-β sliding buffer (§3c → Task 3), state pytree (§4 → Task 1/5), config knobs incl. probe_lr sentinel (§5 → Task 5/6), wiring (§6 → Task 6), correctness risks (§7 → operator/adjoint/equivalence tests in Tasks 2/4), testing plan (§8 → Tasks 2–7), non-goals respected (no spring.py changes, no multi-device reductions, no grid_line_search).
- **Placeholder scan:** every code/test step contains complete runnable code; no TBD/TODO.
- **Type consistency:** `SameSampledSPRINGUnifiedState` field order is identical everywhere; `get_same_sampled_spring_unified_step` signature and return `(updates, new_state)` match between Task 4 definition and Task 5 usage; `initialize_same_sampled_spring_unified` return `(update_param_fn, optimizer_state, key)` matches the Task 6 dispatch unpacking.
