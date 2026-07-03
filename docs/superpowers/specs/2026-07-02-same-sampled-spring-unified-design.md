# Design: `same_sampled_spring_unified` optimizer (JAX/vmcnet port)

**Date:** 2026-07-02
**Author:** Jeffery Ren (with Claude)
**Status:** Awaiting review

## 1. Purpose

Port the PyTorch reference optimizer `same_sampled_spring_unified` (written for the
`rla_pinns` PINN codebase) into `vmcnet` as a new `optimizer_type`, faithful to the
reference algorithm but expressed in vmcnet's JAX conventions. It is
**Same-Sampled SPRING with an interleaved same-sampled SPRING probe**:

- The **main** update is ordinary SPRING (identical to
  [`vmcnet/updates/spring.py`](../../../vmcnet/updates/spring.py)), except the
  momentum/decay factor `β` (vmcnet's `mu`) is no longer static — it is driven by an
  adaptive schedule.
- A **probe** runs a *second, independent* SPRING solve on the **same** sampled
  Jacobian `A` each step, against a fixed synthetic target `b = A · x_star`
  (`x_star` a fixed random unit-norm vector in parameter space). The probe keeps its
  own momentum `phi_probe` and iterate `z_probe`.
- The probe's residual norm history feeds an **adaptive-β schedule** (Appendix D of
  the reference): every `p` steps it estimates the linear-convergence rate `ρ` of the
  probe system and sets `β = (1−ρ)/(1+ρ)`.
- Optionally, an **adaptive step size** `η_main = 1 − β·(1 − η)` (and, if requested,
  an adaptive probe step) replaces the constant base step.

**Decisions locked with the user (2026-07-02):**

1. **Full faithful port** — probe + adaptive-β + `adaptive_eta` + `adaptive_probe`, all
   as config toggles. Drop the reference's `grid_line_search` (a PINN-repo feature not
   present in vmcnet SPRING); use vmcnet's existing learning-rate schedule +
   `constrain_norm` instead.
2. **`adaptive_eta` wraps the scheduled LR:** `η_main = 1 − β·(1 − lr(step))`, where
   `lr(step)` is vmcnet's `learning_rate_schedule` evaluated at the current step. This
   reduces exactly to base SPRING when `adaptive_eta=False`. **This choice must be
   documented with an inline comment in the code.**
3. **Match existing SPRING for devices:** use vmcnet SPRING's per-device (local)
   centering / `T`-matrix convention. The probe uses the *identical* local operator so
   the two solves stay consistent. Correct multi-device support is explicitly out of
   scope (the repo's pmapped SPRING path is itself untested/skipped).

## 2. Background: the two SPRING conventions

vmcnet's SPRING (`spring.py:132-183`) and the PyTorch reference build the same three
linear maps at the current `(params, positions)`. Let `N = nchains`,
`sqrtN = sqrt(N)`, `P = (1/N)·ones·onesᵀ` (the constant-mode/averaging projector), and
`log_psi_apply(params, positions) -> (N,)`. Everything reduces to three operators:

```python
# A : param-pytree -> centered sample-space vector (N,)      [= apply_joint_J]
def apply_A(vec):
    O = jax.jvp(log_psi_apply, (params, positions),
                (vec, jnp.zeros_like(positions)))[1] / sqrtN
    return O - jnp.mean(O, axis=0, keepdims=True)      # left-multiply by (I - P)

# Aᵀ : sample-space vector (N,) -> param-pytree             [= apply_joint_JT]
def apply_AT(v):
    v_hat = v - jnp.mean(v)                             # center the cotangent
    dtheta = jax.vjp(log_psi_apply, params, positions)[1](v_hat)[0]
    return jax.tree_map(lambda x: x / sqrtN, dtheta)

# T = A Aᵀ + P  (the Gram matrix the reference inverts as "JJᵀ")
T = kernel_fn(positions, positions, "ntk", params) / N  # neural_tangents NTK, (1/N) J Jᵀ
T = T - jnp.mean(T, axis=0, keepdims=True)              # (I - P) on the left
T = T - jnp.mean(T, axis=1, keepdims=True)              # (I - P) on the right -> A Aᵀ
T = T + ones @ ones.T / N                               # + P  (regularize constant mode)
T = (T + T.T) / 2
Tvals, Tvecs = jnp.linalg.eigh(T)

def solve(rhs, damp):                                    # (JJᵀ + damp·I)^{-1} rhs
    return Tvecs @ ((Tvecs.T @ rhs) / (jnp.maximum(Tvals, 0) + damp))
```

Correspondence to the PyTorch reference:

| PyTorch | Meaning | JAX expression |
|---|---|---|
| `apply_joint_J(x)` | `A @ x` (param → sample) | `apply_A(x)` |
| `apply_joint_JT(v)` | `Aᵀ @ v` (sample → param) | `apply_AT(v)` |
| `compute_joint_JJT()` | Gram matrix | `T` (**including** `+ ones onesᵀ/N`) |
| `cholesky_solve(r, L)` | `(JJᵀ + λI)^{-1} r` | `solve(r, λ)` (eigh, clip negatives, add λ) |

**Key faithfulness facts** (from the porting analysis):

- `A` centers its *output*, `Aᵀ` centers its *input*; they are transposes of each other.
  The probe must call the *same* `apply_A`/`apply_AT` — dropping either centering
  corrupts the constant mode and makes the probe solve a different system.
- The `+ ones onesᵀ/N` term lives only in `T` (the dual operator), **not** in `A`/`Aᵀ`.
  The probe inherits it automatically by reusing `T`.
- Damping is **clip-then-add** (`max(Tvals,0) + damp`), *not* Cholesky of `T + λI`.
  This is deliberate (numerical stability on GPUs — see the comment at `spring.py:151-157`).
- **Compute the NTK + eigendecomposition once per step** and reuse `solve` for both the
  main and probe solves. Only the additive damping differs
  (`damping` vs `probe_damping`). This is the single biggest efficiency win.
- **Sign convention:** vmcnet uses `epsilon_bar = +centered_local_energies/sqrtN` and
  applies `params -= η·phi` (via optax's subtraction); the PyTorch reference uses
  `-residuals/sqrtN` with `params += η·phi`. We follow vmcnet's sign convention
  throughout. The probe is a self-contained synthetic solve implemented with explicit
  `+=`/`-=` and is **not** routed through optax (which would flip its sign).

## 3. Algorithm (per step), in vmcnet conventions

At step start read `β = state.beta` (dynamic). Build `apply_A`, `apply_AT`, and
`eigh(T)` **once**; define `solve(rhs, damp)`.

### 3a. Main SPRING (identical to base SPRING, but `β` from state)

```python
epsilon_bar = centered_local_energies / sqrtN          # centered_local_energies = local_E - energy
rhs_main    = epsilon_bar - apply_A(beta * phi)        # = epsilon_tilde (spring.py:173)
dual_main   = solve(rhs_main, damping)
phi_new     = tree_add(apply_AT(dual_main), beta * phi)  # dtheta/sqrtN + mu*prev  (spring.py:179-181)

# --- adaptive_eta: eta_main = 1 - beta*(1 - lr(step)), wrapping the SCHEDULED lr ---
# adaptive_eta/adaptive_probe are STATIC config flags -> plain Python branches (not jnp.where).
lr       = learning_rate_schedule(state.step)
eta_main = (1.0 - beta * (1.0 - lr)) if adaptive_eta else lr

updates = multiply_tree_by_scalar(phi_new, -eta_main)  # params -= eta_main * phi_new
if constrain_norm:
    updates = constrain_norm(updates, norm_constraint) # unchanged from spring.py:186-201
params_new = optax.apply_updates(params, updates)
```

`constrain_norm` is copied verbatim from `spring.py` (Euclidean norm constraint with
`pmean_if_pmap` sync).

### 3b. Probe (self-contained synthetic solve on the same `A`)

Uses linearity of `A` to minimize JVP count and avoid subtractive cancellation:

```python
# zeta_probe = (b - A z_probe) - beta*(A phi_probe) = A(x_star - z_probe - beta*phi_probe)
zeta_probe = apply_A(tree(x_star - z_probe - beta * phi_probe))   # 1 JVP
v          = solve(zeta_probe, probe_damping)                     # reuses eigh; only damping differs
w          = apply_AT(v)                                          # 1 VJP
phi_probe_new = tree_add(w, beta * phi_probe)                     # beta*phi_probe + w

eta_probe  = eta_main if adaptive_probe else probe_lr             # numeric probe_lr otherwise
z_probe_new = tree_add(z_probe, eta_probe * phi_probe_new)

# residual of the UPDATED iterate: ||A z_probe_new - b|| = ||A(z_probe_new - x_star)||
probe_res_norm = jnp.linalg.norm(apply_A(tree(z_probe_new - x_star)))  # 1 JVP
```

`b = A·x_star` is **recomputed implicitly every step** (folded into the two `apply_A`
calls) because `A` changes with `(params, positions)`; it is never cached.

### 3c. Adaptive-β schedule (Appendix D), jit-safe

Store `probe_res_norm` into a fixed-length **sliding buffer** (always chronological,
oldest first), then — only every `p` steps, once `step ≥ 2p` — update `β`. All
arithmetic runs every step; the write-backs are gated with `jnp.where` (never a Python
`if`, which is illegal under jit). The sliding buffer is a fixed-shape `concatenate`
(drop the oldest, append the newest), so `window_tp`/`window_t` are plain static slices
— no ring index or dynamic `jnp.roll`:

```python
# residual_buffer stays chronological: [oldest ... newest]; length 2p, init zeros.
residual_buffer = jnp.concatenate([state.residual_buffer[1:], probe_res_norm[None]])
window_tp = residual_buffer[0:p]      # residuals[t-2p+1 : t-p+1]
window_t  = residual_buffer[p:2*p]    # residuals[t-p+1 : t+1]  (inclusive of current)
r_ip = jnp.sum(window_t**2) / jnp.sum(window_tp**2)

n_old = state.checkpoint_idx.astype(float)      # starts at 1  -> 1^ln(1)=1 is safe
n_new = n_old + 1.0
alph  = jnp.power(n_old, jnp.log(n_old)) / jnp.power(n_new, jnp.log(n_new))
r_hat_new = alph * state.r_hat + (1.0 - alph) * jnp.minimum(1.0, r_ip)
rho   = jnp.clip(1.0 - jnp.power(r_hat_new, 1.0 / p), a_min=0.0)
beta_from_sched = (1.0 - rho) / (1.0 + rho)

do_update = (state.step % p == 0) & (state.step >= 2 * p)
beta_next  = jnp.where(do_update, beta_from_sched, state.beta)
r_hat_next = jnp.where(do_update, r_hat_new,       state.r_hat)
ckpt_next  = jnp.where(do_update, state.checkpoint_idx + 1, state.checkpoint_idx)
step_next  = state.step + 1
```

`β` updates take effect on the **next** step (matching the reference, which mutates
`decay_factor` at the end of the step). Within a step `β` is constant. `state.step` here
is the pre-increment counter, matching the reference's `step_idx` (which is read before
`self.steps += 1`).

**Sliding-buffer window check:** at a trigger step, `state.step = t` is a multiple of
`p` and `≥ 2p`. After appending the current residual, the chronological length-`2p`
buffer holds residuals for steps `t-2p+1 … t` (all real, no init zeros remain once
`t ≥ 2p`). So `window_tp = buffer[0:p]` = steps `t-2p+1 … t-p` and
`window_t = buffer[p:2p]` = steps `t-p+1 … t` — exactly the reference's
`residuals[t-2p+1:t-p+1]` / `residuals[t-p+1:t+1]` slices.

## 4. Optimizer state

A single custom `NamedTuple` (a JAX pytree, like optax's own `TraceState`) threaded as
`optimizer_state`. We do **not** reuse the optax-trace momentum trick (base SPRING hides
its momentum in `optimizer_state[0].trace`) because the probe already needs several
extra pytrees; an explicit state is clearer and keeps the probe out of optax's sign
convention.

```python
class SameSampledSPRINGUnifiedState(NamedTuple):
    phi:             P          # main SPRING momentum          (params-shaped)
    z_probe:         P          # probe iterate                 (params-shaped)
    phi_probe:       P          # probe momentum                (params-shaped)
    x_star:          P          # fixed random unit-norm target (params-shaped)
    residual_buffer: Array      # float32[2p] sliding buffer of probe residual norms (chronological)
    r_hat:           Array      # scalar float
    beta:            Array      # scalar float, current decay factor (init = mu)
    checkpoint_idx:  Array      # scalar int32, starts at 1
    step:            Array      # scalar int32, starts at 0 (== epoch index)
```

**Initialization** (mirrors `initialize_optax_optimizer`'s pmap-if-needed pattern):

```python
def init_state(params, subkey):
    flat, unravel = jax.flatten_util.ravel_pytree(params)
    xf = jax.random.normal(subkey, flat.shape)
    x_star = unravel(xf / jnp.linalg.norm(xf))          # GLOBAL unit L2 norm (whole pytree)
    zeros = jax.tree_map(jnp.zeros_like, params)
    return SameSampledSPRINGUnifiedState(
        phi=zeros, z_probe=zeros, phi_probe=zeros, x_star=x_star,
        residual_buffer=jnp.zeros(2 * p),
        r_hat=jnp.array(1.0), beta=jnp.array(float(mu)),
        checkpoint_idx=jnp.array(1, jnp.int32), step=jnp.array(0, jnp.int32),
    )
# apply_pmap: state = utils.distribute.pmap(init_state)(params, subkey)   # params/subkey already device-batched
# else:       state = init_state(params, subkey)
```

`x_star` is drawn once from a split of the run key
(`utils.distribute.split_or_psplit_key`) and never recomputed. The whole state pytree
is auto-replicated / checkpointed by the existing
`distribute_vmc_state` / checkpoint machinery (it is arrays-in-a-NamedTuple, exactly
like optax states).

> Multi-device caveat (documented, out of scope): under `apply_pmap` with >1 device the
> pmapped init draws a *different* `x_star` per device (per-device keys), and centering
> is per-device — consistent with vmcnet SPRING being single-device in practice.

## 5. Config knobs

New block in `get_default_vmc_config()` → `optimizer` dict
([`default_config.py`](../../../vmcnet/train/default_config.py), after the
`gauss_newton` block). All keys must exist here because the config is locked before CLI
overrides.

```python
"same_sampled_spring_unified": {
    # Learning rate settings (read by _get_learning_rate_schedule)
    "schedule_type": "inverse_time",   # constant or inverse_time
    "learning_rate": 5e-2,
    "learning_decay_rate": 1e-4,
    # SPRING hyperparams
    "mu": 0.9,                 # INITIAL beta before the adaptive schedule kicks in
    "damping": 1e-3,
    "constrain_norm": True,
    "norm_constraint": 1e-3,
    # Adaptive-beta / probe hyperparams
    "lb_window": 30,           # lookback p; buffer length is 2p
    # probe_lr: base probe step used when adaptive_probe=False. A NEGATIVE value is a
    # sentinel meaning "default to the base learning_rate value" (resolved in the
    # initializer: probe_lr = learning_rate when probe_lr < 0). Set a positive value to
    # override independently.
    "probe_lr": -1.0,
    "probe_damping": 1e-3,     # probe normal-equation damping (defaults to `damping`)
    "adaptive_eta": False,     # eta_main = 1 - beta*(1 - lr(step))
    "adaptive_probe": False,   # probe uses eta_main instead of probe_lr
},
```

(`mu` default `0.9` matches the reference's initial β, not base SPRING's `0.99`.)

**`probe_lr` resolution (in `initialize_same_sampled_spring_unified`):**

```python
# probe_lr defaults to the base learning_rate value (mirrors the PyTorch reference,
# where probe_lr falls back to lr). A negative config value is the "unset" sentinel.
probe_lr = (
    optimizer_config.learning_rate
    if optimizer_config.probe_lr < 0
    else optimizer_config.probe_lr
)
```

This keeps `probe_lr` tracking `learning_rate` by default (even if `learning_rate` is
changed), stays overridable, and avoids `None`-under-locked-`ConfigDict` issues.

## 6. Wiring (files touched)

1. **New:** `vmcnet/updates/same_sampled_spring_unified.py` — the implementation:
   `SameSampledSPRINGUnifiedState`, `get_same_sampled_spring_unified_step`,
   `constrain_norm` (copied from spring), `construct_..._update_param_fn`,
   `initialize_same_sampled_spring_unified(...) -> (update_param_fn, optimizer_state, key)`.
2. **Edit:** `vmcnet/train/default_config.py` — add the config block (§5).
3. **Edit:** `vmcnet/updates/parse_optimizer_config.py` — add
   `from .same_sampled_spring_unified import initialize_same_sampled_spring_unified`
   and an `elif vmc_config.optimizer_type == "same_sampled_spring_unified":` branch
   (modeled on the `spring` branch) *before* the final `else`. It builds
   `energy_and_statistics_fn` the same way SPRING does and threads `key`.
4. **Edit (optional but tidy):** `vmcnet/updates/__init__.py` — no functional change
   needed (dispatch imports the module directly); skip unless we want the module
   importable as an attribute.

No allow-list/enum exists to update. No changes to `runners.py`, `mcmc`, or `physics`.

## 7. Correctness risks & how the design addresses them

| Risk | Mitigation |
|---|---|
| Centering mismatch main vs probe | Both solves route through the *same* `apply_A`/`apply_AT` closures. |
| Missing/extra `√N` | `A` has `/√N`, `Aᵀ` has `/√N`, `T` has `/N`; probe reuses these — no separate scaling. |
| Wrong damping semantics | Single `solve` with `max(Tvals,0)+damp`; probe only swaps the additive damping. |
| Constant mode `+P` dropped | `+ ones onesᵀ/N` kept in `T`; probe reuses `T`. |
| Probe sign flipped | Probe implemented with explicit `+=`; never routed through optax. |
| Growing residual list under jit | Fixed-length `2p` ring buffer + `jnp.where`-gated β update. |
| `checkpoint_idx = 0 → nⁿ NaN` | Initialize `checkpoint_idx = 1` (`1^ln1 = 1`). |
| `x_star` recomputed / per-leaf norm | Drawn once at init, **global** unit L2 norm via `ravel_pytree`, stored in state. |
| `b` cached | Never cached; folded into `apply_A` each step. |
| Dynamic β not threaded everywhere | `β` read from state and used in `apply_A(β·phi)`, probe, and `η_main`. |

## 8. Testing / verification plan

(Optimizers in this repo have only integration coverage; there are no unit tests for
`initialize_spring`/etc. We add focused unit tests plus a smoke test.)

1. **Operator sanity (unit, fast):** on a tiny model, assert
   (a) `apply_A(v)` is mean-zero to machine precision (centering),
   (b) `⟨apply_A(u), v⟩ ≈ ⟨u, apply_AT(v)⟩` (transpose + `√N` + centering consistency),
   (c) with `probe_damping == damping`, `solve` gives identical results for a shared RHS.
2. **Equivalence to base SPRING (unit, fast):** with `adaptive_eta=False` and
   `lb_window` large enough that the `step ≥ 2p` trigger never fires within the test
   budget, the **main** parameter update must match `initialize_spring`'s update (same
   `mu`, `damping`, seed) to tolerance. This anchors main-path correctness. (The passive
   probe runs but does not affect the main update until β changes.)
3. **Adaptive-β math (unit, fast):** feed a controlled residual buffer and check the
   `r_ip → r_hat → ρ → β` computation against a hand/numpy reference, including the
   first trigger (`checkpoint_idx = 1`) and the `min(1, r_ip)` / `clip(…, 0)` clamps.
4. **Jitted end-to-end smoke (integration, `slow`):** run several steps of the full
   `update_param_fn` under `jax.jit` on a tiny harmonic-oscillator-style model; assert
   no NaNs, state shapes are stable across steps, and energy decreases.
5. **Checkpoint round-trip (integration):** confirm the custom state pytree serializes
   and reloads via the existing reload path (state restored bit-for-bit).
6. **Lint/type/format:** `mypy vmcnet tests`,
   `flake8 vmcnet tests --select D --extend-ignore D401`,
   `black --check vmcnet tests` (per `lint.sh`).

## 9. Non-goals

- Correct multi-device (pmap) reductions — matches base SPRING (local) by decision.
- Porting `grid_line_search` — replaced by vmcnet's LR schedule + `constrain_norm`.
- Refactoring or changing base `spring.py`.
- Probe-every-k-steps or other cost optimizations beyond the shared eigendecomposition.
