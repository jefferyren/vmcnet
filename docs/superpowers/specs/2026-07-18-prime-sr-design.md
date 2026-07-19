# Design: `prime_sr` optimizer (PRIME-SR, JAX/vmcnet port)

**Date:** 2026-07-18
**Author:** Jeffery Ren (with Claude)
**Status:** Awaiting review

## 1. Purpose

Implement **PRIME-SR** (Principal Range Informed MomEntum SR) from *"Momentum
Stability and Adaptive Control in Stochastic Reconfiguration"* (Wang & Liu, 2026,
arXiv:2604.18357) as a new `optimizer_type` in vmcnet. PRIME-SR is a **tuning-free
momentum-adaptive variant of SPRING** (Algorithm 2 of the paper): the fixed momentum
`mu` of SPRING is replaced each step by an adaptive `mu_k` computed from spectral
indicators of the sampled Gram matrix `T_k = O_kᵀ O_k`.

The paper's electronic-structure experiments were run in VMCNet itself (Section 5),
so this is the algorithm's native setting; a PyTorch adaptation for PINNs
(`prime_sr.py`, `rla_pinns` repo) serves as a secondary implementation reference.

**Decisions locked with the user (2026-07-18):**

1. Implement from the paper PDF (primary) + PyTorch reference (secondary).
2. Cross-step subspace overlap is computed directly between successive sampled
   `T_k` eigenbases, exactly as the paper does (its own VMC experiments resample
   walkers via MCMC between iterations).
3. Diagnostics (`mu`, `alpha`, `rank`, `beta_tilde`) are surfaced through the
   per-epoch **metrics dict** (the JAX-idiomatic equivalent of the reference's
   `print_every`), not printed.
4. **Ones-term deviation approved:** eigendecompose the centered `T` *without*
   spring.py's `(1/N)·ones·onesᵀ` correction (see §3, Deviation D1).
5. Config defaults follow the paper's Section 5 (lr `0.02`, `inverse_time` decay
   `1e-4`, damping `1e-3`, norm constraint `1e-3`). No `mu` knob — tuning-free
   momentum is the point of the method.

## 2. Algorithm (paper → vmcnet mapping)

Per-step, with `N = nchains` (paper `Ns`), `O` the centered Jacobian of
`log|psi|` scaled by `1/sqrt(N)` (repo convention; the paper's `1/sqrt(Ns−1)`
differs only by a uniform spectrum rescale of `N/(N−1)`, which leaves `alpha`,
`beta_tilde`, and the rank tolerance invariant and negligibly rescales the
damping):

1. **Gram matrix:** `T = O Oᵀ` over samples (`N×N`), built exactly as
   `spring.py` does via `nt.empirical_kernel_fn` + double centering — but
   **without** the `ones·onesᵀ/N` term (D1). Symmetrize, `eigh`, flip to
   descending, clamp negative eigenvalues to 0 (GPU-robustness convention from
   `spring.py`).
2. **Numerical rank** (Algorithm 2, line 6): `r_k = #{s² > tol}` with
   `tol = N · eps_machine · s²_max` (MATLAB `rank` default, as implemented in the
   PyTorch reference).
3. **Effective spectral dimension** (Eq. 4.7): `alpha_k = (Σ s²)² / Σ s⁴` summed
   over the rank-truncated spectrum; `ceil_alpha_k = clip(ceil(alpha_k), 1, r_k)`.
4. **Subspace overlap** (Eq. 4.4): `beta_tilde_k = ‖V_{k,α}ᵀ V_{k−1,α}‖_F` where
   `V_{k,α}` = leading `ceil_alpha_k` eigenvectors of `T_k`.
5. **Adaptive momentum** (Eq. 4.5):

   ```
   m      = min(ceil_alpha_k, ceil_alpha_{k−1})
   mu_k   = 1 − (1 − sqrt(clip(beta_tilde_k / sqrt(m), 0, 1)))
              · (1 − (alpha_k / r_k)^(1/4))
   ```

   `mu_0 = 0` (first step; `Δθ_{−1} = 0` so it is inert regardless), and `mu_k = 0`
   whenever there is no valid cached subspace (sentinel `alpha_prev_ceil == 0`) or
   `r_k == 0`. On an `r_k == 0` step the previous cache is **kept**, matching the
   PyTorch reference.
6. **SPRING update** (Eqs. 2.12–2.13, identical math to `spring.py` with `mu →
   mu_k`): `epsilon_bar = centered_local_energies / sqrt(N)`;
   `zeta = epsilon_bar − mu_k·(O @ phi)`; solve
   `y = V diag(1/(s²_clamped + damping)) Vᵀ zeta`; re-center `y` (spring.py's
   post-solve centering, kept for numerical safety); `phi ← mu_k·phi + Oᵀ y`.
7. **Step + norm constraint** (Eq. 2.15): `updates = −lr(step)·phi`, then the
   repo's `constrain_norm` (cap `‖update‖² ≤ norm_constraint`). This composition
   is algebraically identical to the paper's
   `θ_{k+1} = θ_k + Δθ_k · min{η_k, sqrt(C)/‖Δθ_k‖}`.
8. **Cache** `V_{k,α}` (column-masked) and `ceil_alpha_k` for the next step.

### Deviation D1 — no ones-term in the eigendecomposed matrix

`spring.py` adds `ones·onesᵀ/N` to `T` before `eigh` (paper Eq. 2.14, a
conditioning trick that provably does not change the update in exact arithmetic).
That term injects a spurious eigenpair (eigenvalue 1, constant eigenvector) which
would pollute `r_k`, `alpha_k`, `V_α`, and `beta_tilde_k`. PRIME-SR therefore
eigendecomposes the centered `T` **without** it and reuses that single
decomposition for both the indicators and the solve. Equivalence for the solve:
`zeta` is exactly mean-zero and the orthogonal complement of `span(1)` is
invariant under `T`, so `(λI + T)⁻¹ zeta = (λI + T + ones·onesᵀ/N)⁻¹ zeta`;
robustness is retained via the negative-eigenvalue clamp + damping + post-solve
re-centering.

## 3. Architecture

New module `vmcnet/updates/prime_sr.py`, mirroring the
`same_sampled_spring_unified` pattern (custom NamedTuple state; the optax-trace
trick used by `spring.py` cannot hold the cached eigenbasis).

```python
class PRIMESRState(NamedTuple):
    phi: PyTree            # Δθ_{k−1}, params-shaped momentum
    V_prev_alpha: Array    # (N, N); columns ≥ alpha_prev_ceil zeroed
    alpha_prev_ceil: Array # int scalar; 0 == "no valid previous subspace"
    step: Array            # int scalar, drives the lr schedule
    mu: Array              # diagnostics (scalars), surfaced into metrics
    alpha: Array
    rank: Array
    beta_tilde: Array
```

**JIT constraints:** `r_k` and `ceil(alpha_k)` are data-dependent, so all
rank-truncated sums use boolean masks and `V_α` selection uses column masking
(`col_idx < ceil_alpha`) rather than slicing. `V_prev_alpha` is stored full-size
(`N×N`, same footprint as `T` itself; the paper's `O(N·⌈α⌉)` storage is not
expressible under jit). `beta_tilde` is computed on the masked matrices, which is
exactly the Frobenius norm of the `⌈α_k⌉×⌈α_{k−1}⌉` block.

**Module layout** (functions, mirroring sibling modules):

- `get_prime_sr_step(log_psi_apply, learning_rate_schedule, damping) -> step_fn`
  — builds the kernel fn once; `step_fn(centered_local_energies, params,
  positions, state) -> (updates, new_state)` with `updates` already scaled by
  `−lr`.
- `constrain_norm(grad, norm_constraint)` — same as siblings.
- `construct_prime_sr_update_param_fn(...)` — standard metrics plumbing, plus
  `metrics.update({"mu": ..., "alpha": ..., "rank": ..., "beta_tilde": ...})`
  read from the *new* optimizer state (cast to float for logging).
- `initialize_prime_sr(log_psi_apply, energy_and_statistics_fn, params,
  get_position_fn, update_data_fn, learning_rate_schedule, optimizer_config,
  record_param_l1_norm, apply_pmap) -> (update_param_fn, optimizer_state)` —
  init state with zeros / `alpha_prev_ceil=0` / `step=0`; state built under
  `pmap` when `apply_pmap` (per-device `T` over local chains, same convention as
  the repo's SPRING; no PRNG key needed, unlike `same_sampled_spring_unified`).

**Wiring:**

- `parse_optimizer_config.py`: `elif vmc_config.optimizer_type == "prime_sr"`
  branch using `create_energy_and_statistics_fn` (same as spring), returning
  `update_param_fn, optimizer_state, key` with the same
  `# type: ignore[return-value]` note as `same_sampled_spring_unified` (custom
  state not in the `OptimizerState` union).
- `default_config.py` optimizer block:

  ```python
  "prime_sr": {
      "schedule_type": "inverse_time",  # constant or inverse_time
      "learning_rate": 2e-2,
      "learning_decay_rate": 1e-4,
      # PRIME-SR hyperparams (no mu -- momentum is adaptive, Eq. 4.5)
      "damping": 1e-3,
      "constrain_norm": True,
      "norm_constraint": 1e-3,
  },
  ```

## 4. Error handling / edge cases

- **Step 0:** `alpha_prev_ceil == 0` sentinel forces `mu = 0`; the update is then
  plain SPRING-with-`mu=0` (pure MinSR step), matching Algorithm 2 line 1.
- **Rank-deficient step (`r_k == 0`):** `mu = 0`, indicators reported as 0, cache
  kept (all via `jnp.where`; no Python branches under jit).
- **`alpha/r` and `beta_tilde/sqrt(m)` ratios** are clipped to `[0, 1]` before
  fractional powers (guards float noise, matches reference).
- **NaN safety / clipping** are inherited from the repo's
  `create_energy_and_statistics_fn` path, as with spring.

## 5. Testing

Unit tests `tests/units/updates/test_prime_sr.py` (style of
`test_same_sampled_spring_unified.py`, including its neural-tangents import-order
workaround):

1. State NamedTuple field test.
2. Indicator math vs. direct NumPy on a small synthetic `T`: rank tolerance,
   `alpha` (Eq. 4.7), masked `beta_tilde` equals explicit sliced Frobenius norm,
   `mu` formula (Eq. 4.5) including both clips.
3. First step equals `spring.py`'s `get_spring_step` with `mu = 0` (same params,
   positions, energies) up to the D1 solve-path tolerance.
4. `mu` gating: step 0 gives `mu = 0`; a second step with a cached subspace gives
   `mu` in `(0, 1]`; identical consecutive `T` (same positions/params) drives
   `beta_tilde` to its upper bound `sqrt(m)`.
5. Momentum accumulation: `phi` after two steps equals the hand-unrolled
   recursion.

Integration test
`tests/integrations/updates/test_prime_sr_integration.py`: smoke test mirroring
`test_same_sampled_spring_unified_integration.py` (short VMC run on the existing
toy problem; asserts finite energy, finite/in-range `mu`, and state advancing).

## 6. Out of scope

- Multi-device (`pmap`) correctness beyond the repo's existing per-device SPRING
  convention (the pmapped SPRING path is itself untested in this repo).
- The left-subspace overlap `β(U)` variant (paper adopts the right-subspace form
  for exactly the cost reasons that apply here).
- Line search (PINN-reference feature; vmcnet uses schedules + norm constraint).
