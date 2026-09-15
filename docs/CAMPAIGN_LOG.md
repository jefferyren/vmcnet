# Campaign log — SS-SPRING vs SPRING vs PRIME-SR

**Handoff document.** Read this first in a new session; it is self-contained. Last
updated 2026-08-15, after E10 completed Phase C. 325 runs, ~1050 GPU-hours, all on Savio
GTX2080TIs.

**Campaign summary report for non-specialist readers (2026-08-17):** [SS-SPRING — full
campaign summary](https://wandb.ai/ren27-university-of-california-berkeley/vmcnet-phase-c/reports/SS-SPRING-%E2%80%94-full-campaign-summary--VmlldzoxNzc1MDYwNA) — covers all three phases with **no E-numbers or internal
terminology**: what was run, what was tested, what the result was, plus the ruled-out
list and the regret framing. 12 panels spanning all three wandb projects. Built by
`slurm/report_campaign_summary.py`, which edits in place via `from_url` (a bare
`wr.Report()` + `save()` would mint a duplicate). Re-verify after any change.

> **Current state (2026-08-15): Phase C is COMPLETE (E9, E10, E11 — 110 runs).**
>
> - **Claim B (SS-SPRING ≥ PRIME-SR) is the result.** Six systems, three learning rates,
>   **never lost**: 38σ on CO, 13σ on N2-eq, 12.5σ on carbon at eta=0.005, 3.9σ on N,
>   3.4σ on H4, ~7σ on carbon at 0.02; ties on O and carbon at eta=0.05. E10 won it on
>   PRIME-SR's *own* systems at its *own* settings. **This is the paper's headline.**
> - **Claim A (untuned == tuned SPRING) does NOT generalise.** Holds on carbon, H4, N, CO
>   and N2-eq (where SS-SPRING is ahead); **fails on oxygen** (0/5 seeds) and **carbon at
>   eta=0.05** (0/3). Scope it as a REGRET result: over all seven conditions SS-SPRING has
>   the lowest mean (0.036 mHa) and worst-case (0.178) regret of any arm and matches oracle
>   per-condition tuning to −0.009 mHa on average — that form holds; per-condition parity
>   does not.
> - **PRIME-SR breaks at small eta** (§4 item 8) — the clean failure this campaign was
>   built to find, now confirmed twice: 2.7x worse at eta=0.005 on carbon (E11) and
>   ~2.9 mHa behind on both N2 and CO at eta=0.002 (E10).
>
> **E10 is complete: 36/36 cells with eval energies.** Its eval phase was lost to an OOM
> and recovered from the 50k checkpoints; the two externally-killed cells were rerun in
> full. The ordering never changed through any of it.
>
> **PHASE D IS QUEUED, NOT RUN (added 2026-08-26).** All four paper experiments are
> being re-run at **100k epochs** -- the iteration count both the SPRING and PRIME-SR
> papers report at, and the answer to E10's non-convergence caveat below. Scripts are
> written and verified; nothing has been submitted. See §5 Phase D for what changed and
> §10 for the submit order. Phase C's numbers stand until Phase D replaces them.
>
> **Standing caveat on E10: nothing is converged at 50k** (every arm still improving
> 0.1–1.0 mHa per 5k epochs, absolute errors 9–20 mHa vs 0.2–0.8 on the atoms). It
> compares arms at a **fixed budget**, not at their asymptotes.

---

## 1. The goal

Two claims about **same-sampled SPRING (SS-SPRING)**, our adaptive-momentum variant:

- **Claim A** — untuned, it matches *optimally tuned* SPRING.
- **Claim B** — it matches or beats **PRIME-SR** (Wang & Liu, arXiv:2604.18357), the
  competing adaptive-momentum SR method.

**Status after Phase C (E9, E10, E11): the claims have separated.** Claim B holds
everywhere tested — **six systems, three learning rates, never once lost**, including on
PRIME-SR's own N2-eq and CO at PRIME-SR's own settings. **Claim A does not generalise**:
it holds on carbon, H4, N, CO and N2-eq, and fails on oxygen (0/5 seeds) and carbon at
eta=0.05 (0/3). See the table in §3 and the detail in §5.

A third result arrived unlooked-for and is arguably the most publishable: **PRIME-SR
fails badly at small learning rate** (§4 item 8) — the clean PRIME-SR breakage this
campaign was originally set up to find.

## 2. The three methods

All three are stochastic-reconfiguration optimizers for neural-network VMC that differ
*only* in how they set the momentum applied to the previous update:

| | momentum | sees the step size? | sees the energy residual? |
|---|---|---|---|
| **SPRING** | fixed `mu`, hand-tuned (paper default 0.99) | no | no |
| **PRIME-SR** | `mu_k` recomputed each step from the sampled Gram-matrix spectrum (effective dimension α, numerical rank, principal-subspace overlap) | **no** | **no** |
| **SS-SPRING** | `beta` from a *same-sampled probe*: a second SPRING solve on the same sampled operators against a fixed synthetic target, whose measured residual contraction sets the momentum | yes, via the probe | no |

Implementations: `vmcnet/updates/{spring,prime_sr,same_sampled_spring_unified}.py`.

## 3. Headline result (E7, as qualified by E9/E10/E11, then by Phase D/E)

> *** SUPERSEDED IN PART BY PHASE D/E (2026-09-13). CLAIM B NOW HAS A COUNTEREXAMPLE. ***
> Everything below is the 50k picture and its numbers still stand at 50k, but the
> protocol of record is now 100k (Phase D), and on **N2 at 4.0 Bohr — the seventh
> system, run for the first time in E14 — SS-SPRING diverged to NaNs on 3/3 seeds
> while SPRING and PRIME-SR both completed.** "Six systems, never lost" is no longer
> the sentence. Read §5 Phase D and Phase E before quoting any claim here, and state
> Claim B as "at least as good as PRIME-SR wherever it is stable, with one system where
> it is not stable at all."
>
> **Read §5 Phase C before quoting anything here.** E7's two-system tie is real but does
> **not** generalise: across eight conditions Claim B never loses, while Claim A fails on
> oxygen (0/5 seeds) and on carbon at eta=0.05 (0/3). The summary across everything
> measured:
>
> | condition | Claim A (ties best fixed mu) | Claim B (≥ PRIME-SR) |
> |---|---|---|
> | carbon, eta=0.02 (E7) | ✅ tie (−0.002 ± 0.013) | ✅ +7σ |
> | H4, eta=0.02 (E7) | ✅ tie (+0.017 ± 0.021) | ✅ +3.4σ |
> | N, eta=0.02 (E9) | ✅ tie (+0.024 ± 0.019) | ✅ +3.9σ, 5/5 seeds |
> | **O, eta=0.02 (E9)** | ❌ **+0.178 ± 0.066, 0/5 seeds** | ⚪ tie |
> | carbon, eta=0.005 (E11) | ✅ tie (+0.009 ± 0.028) | ✅ **+12.5σ** |
> | **carbon, eta=0.05 (E11)** | ❌ **+0.037 ± 0.010, 0/3 seeds** | ⚪ tie |
> | N2-eq, eta=0.002 (E10) † | ✅ **exceeded** (−0.257 ± 0.063, 3/3) | ✅ **+13.1σ** |
> | CO, eta=0.002 (E10) † | ✅ tie (−0.053 ± 0.095) | ✅ **+38.5σ** |
>
> † E10 is **not converged at 50k** — it compares arms at a fixed budget, not at their
> asymptotes — and its per-run MC error (0.14–0.33 mHa) is comparable to its Claim A
> margins though far below its Claim B gaps.
>
> **Claim B is the paper's defensible headline. Claim A must be scoped, not asserted.**

5 seeds, 50k epochs + 20k-epoch eval at 2000 walkers, eta=0.02, random init.

**Carbon** (mHa vs −37.8450 Ha, lower better):

| arm | mean | worst seed | spread |
|---|---|---|---|
| SPRING mu=0.99 | +0.147 ± 0.012 | +0.178 | 0.074 |
| SPRING mu=0.995 | +0.161 ± 0.004 | +0.176 | 0.023 |
| PRIME-SR | +0.217 ± 0.009 | +0.249 | 0.052 |
| **SS-SPRING** | **+0.148 ± 0.004** | **+0.157** | **0.019** |

**H4** (raw Ha, lower better): SS-SPRING **−2.03420**, SPRING(0.99) −2.03418,
PRIME-SR −2.03403, SPRING(0.8) −2.03379.

- **Claim A**: SS-SPRING ties the best fixed mu on both systems (carbon Δ = −0.002 ±
  0.013 mHa; H4 Δ = +0.017 ± 0.021). A *tie* — parity without tuning, not superiority.
- **Claim B**: beats PRIME-SR by **0.069 ± 0.009 mHa (~7σ)** on carbon, behind on
  every seed, and **0.172 ± 0.050 mHa (~3.4σ)** on H4.
- SS-SPRING is also the **most seed-stable** arm on both systems — which is precisely
  the property PRIME-SR advertises for itself.

## 4. Things we believed and then disproved

This is the most important section for a new session. Several plausible ideas died,
and re-proposing them wastes GPU-hours.

1. **Original hypothesis: PRIME-SR fails at small sample size** (as it did in PINNs).
   **Failed and inverted** (E3). At carbon N=100, PRIME-SR is the *best* arm and
   SS-SPRING the worst. Its mu_k does rise as N shrinks (0.953 → 0.970) but ~10× too
   little to matter; the predicted rise was 0.64 → 0.92.
2. **Both "adaptive" methods are nearly constant-momentum in this regime.** SS-SPRING's
   beta spans only 0.978–0.996 across every condition tested, PRIME-SR's 0.932–0.970.
   For SS-SPRING the cause is structural: `beta = (1−ρ)/(1+ρ)` with `ρ = 1 − r̂^(1/p)`,
   and the measured contraction sits at r̂ ≈ 0.91, which the 1/p root maps onto
   beta ≈ 0.994 for any sane `p`. To reach beta = 0.9 the probe would have to cut its
   squared residual 80% every 30 steps, sustained — it never does, because it chases a
   moving target and its residual plateaus at a noise floor.
3. **Momentum warm-up is the mechanism.** **Rejected** (E6). SS-SPRING's beta is
   exactly 0 for the first 60 steps (= 2·lb_window) then jumps to ≥0.95, which looked
   like an automatic warm-up. Removing it changes nothing (−2.03406 vs −2.03406).
4. **"It just finds a better constant."** **Rejected** (E8). SPRING pinned at
   SS-SPRING's measured converged beta (0.9944 carbon / 0.9924 H4) does *not*
   reproduce it — on H4 it repeats the same seed-0 failure that SPRING(0.99) has.
5. **E0's tuned-mu picks.** **Wrong** (E7). The single-seed 25k grid chose mu=0.8 for
   H4; at 5 seeds and 50k that is the *worst* arm, 0.410 ± 0.058 mHa behind SS-SPRING.
   Its carbon pick (0.995) is also slightly worse than the default 0.99. Short
   single-seed grids mis-rank momentum — this is now an argument *for* the method.
6. **E6's dramatic seed-0 anomaly.** An **undertraining artifact** — it vanishes by
   50k (E7). Do not read any 25k screen as a robustness result.
7. **eta is the step size.** It largely is not (E2) — **but only at 25k.** With the norm
   constraint on the realized step is `min(η‖φ‖, √C)`, and while the cap binds eta
   cancels out of the update entirely, so C (`norm_constraint`) behaves as the real
   step-size knob and energies look flat across a 40× eta range on carbon.
   **CORRECTED by E9/E11 (2026-08-11):** the cap *releases mid-run* on a 50k schedule —
   it binds on 37–97% of steps in the first half and is at **exactly 0** for the entire
   second half of every Phase C run. So over the back half of a long run eta **is** the
   operative step size, and E11 duly finds a 2.7× spread between arms at eta=0.005.
   Scope the original conclusion to short runs.

8. **PRIME-SR is robust.** **False for the learning rate** (E11) — this is the clean
   failure the campaign set out to find (§1). PRIME-SR is 2.7× worse than every other arm
   at eta=0.005, with a swing of 0.242 mHa across eta=0.005/0.02/0.05 versus 0.030–0.054
   for SPRING and SS-SPRING. Mechanism, and it is not subtle: its momentum is **blind to
   eta by construction** (median mu 0.9529 at eta=0.005 vs 0.9539 at 0.05 — 0.001 across
   a 10× change), and it settles far lower than the other arms. While the cap binds that
   is masked; in the uncapped small-eta regime the accumulated step scales as
   `eta/(1−mu)`, giving 0.106 for PRIME-SR against 1.25/1.00/0.50 for SS-SPRING and
   SPRING(0.995)/(0.99) — a ~10× smaller effective step, and it simply has not converged
   by 50k. The measured ordering follows that ratio. (The scaling is an interpretation
   consistent with the ordering, not a measurement.) The PRIME-SR paper advertises
   seed-robustness and reports **no eta study**.
   **Confirmed again by E10 (2026-08-13)** on PRIME-SR's own N2-eq and CO at its own
   eta = 0.002: it lands 2.85 and 2.91 mHa behind SS-SPRING (13.1σ, 38.5σ, every seed),
   and across the six arms there the correlation of log₁₀(eta/(1−mu)) with eval error is
   r = −0.89 on both systems.
   The arm ordering is the momentum ordering. Its momentum settles at ~0.953 — between
   the two values its own paper recommends — which is precisely the wrong place to be at
   a small learning rate.

9. **SS-SPRING is unconditionally stable.** **False** (E14, 2026-09-13). On N2 at a
   stretched 4.0 Bohr bond it diverges to NaNs on **3/3 seeds**, at epochs 18.6k/17.3k/
   23.6k of 100k, while SPRING(0.99) and PRIME-SR both complete. It is healthy through
   epoch 10k — tracking both other arms to within 0.03 Ha — then loses 4-9 Hartree
   between 10k and 15k. Same eta, same 14 electrons and the same preset family as N2-eq
   and CO, where it is the best arm on the board; the stretched geometry is the only
   variable. This was invisible for the whole campaign only because N2-4.0 had never
   been run. **It is not a 100k phenomenon** — the blow-up is at ~12k epochs, so a 50k
   Phase C run would have shown it too. Mechanism still open; see §5 Phase E for the
   exact diagnostic files to pull.

**Still unexplained:** SS-SPRING beats SPRING(0.99) on carbon by ~0.10 mHa at ~2σ, and
it is neither the warm-up (E6) nor the converged constant (E8). What remains is beta's
*trajectory* between those endpoints. The claims do not depend on resolving this;
treat it as future work rather than a blocker.

**Also unexplained, and this one *does* matter:** why the stretched-N2 divergence
(item 9). A referee will ask whether the same failure lurks on any system not yet
tried, and the campaign has no answer until the beta trace is examined.

## 5. Experiment-by-experiment

Every experiment has a wandb report with a description, takeaways, and native panels.
Reports link to their parents, and parents carry forward pointers to results that
superseded them.

### Phase A — screening (109 runs, seed 0, 25k epochs, carbon + H4, ~180 GPU-h)

| # | What | Result | Report |
|---|---|---|---|
| smoke | nchains=2000 OOM check, N2/H4 timing | no OOM on 11 GB | — |
| **E0** | SPRING mu grid {0…0.999}, carbon + H4 × eta {0.02, 0.002} | carbon plateau 0.95–0.999; H4 needs eta=0.02 (beats 0.002 by ~11 mHa). **mu picks later overturned** | [E0](https://wandb.ai/ren27-university-of-california-berkeley/vmcnet-phase-a/reports/Plots-E0-variance-vs-steps--VmlldzoxNzY4OTgyMA) |
| **E2** | eta sweep × 3 optimizers × norm constraint on/off | cap makes energy flat over 40× eta; constraint-off, 10/15 diverge within ~160 epochs; survival order SS-SPRING > PRIME-SR > SPRING | [E2](https://wandb.ai/ren27-university-of-california-berkeley/vmcnet-phase-a/reports/Plots-E2-variance-vs-steps--VmlldzoxNzY4OTgzNA) |
| **E3** | nchains {100…1000} × 3 optimizers × 2 systems | hypothesis **inverted**; but SS-SPRING won H4 at N=1000 — the thread everything after pulls on | [E3](https://wandb.ai/ren27-university-of-california-berkeley/vmcnet-phase-a/reports/Plots-E3-variance-vs-steps--VmlldzoxNzY4OTkyNw) |
| **E4** | H4 stress grid, eta × nchains, constraint off | SS-SPRING worst in the small-N/large-eta corner | [E4](https://wandb.ai/ren27-university-of-california-berkeley/vmcnet-phase-a/reports/Plots-E4-variance-vs-steps--VmlldzoxNzY4OTkyOQ) |
| **E5** | SS-SPRING knob screen (7 settings) | all within ~0.5 mHa — "untuned" claim holds; lb_window moves beta exactly as the formula predicts but not far enough to matter | [E5](https://wandb.ai/ren27-university-of-california-berkeley/vmcnet-phase-a/reports/Plots-E5-variance-vs-steps--VmlldzoxNzY4OTkzMA) |

### Phase B — mechanism and the main table (70 runs, ~215 GPU-h)

| # | What | Result | Report |
|---|---|---|---|
| **E6** | 2×2 warm-up × adaptivity, 2 systems × 3 seeds | warm-up **rejected**; both SS-SPRING arms still beat both SPRING arms; H4 margin traced to one seed → reframed as robustness | [E6](https://wandb.ai/ren27-university-of-california-berkeley/vmcnet-phase-b/reports/Plots-E6-momentum-warm-up-ablation--VmlldzoxNzY5NDE3Ng) |
| **E7** | 4 arms × 2 systems × **5 seeds**, 50k + eval | **both claims supported** (§3); E0's tuning overturned | [E7](https://wandb.ai/ren27-university-of-california-berkeley/vmcnet-phase-b/reports/Plots-E7-seeded-head-to-head-main-result--VmlldzoxNzcwMTkyNw) |
| **E8** | SPRING pinned at SS-SPRING's converged beta | matched constant **does not** reproduce SS-SPRING | [E8](https://wandb.ai/ren27-university-of-california-berkeley/vmcnet-phase-b/reports/Plots-E8-matched-constant-control--VmlldzoxNzcwMTkyOA) |

### Phase C — COMPLETE: E9, E10, E11 (110 runs, ~600 GPU-h)

Closed all three referee objections to E7. Project `vmcnet-phase-c`.
Also covered, at a high level and alongside Phases A and B, in the [campaign summary](https://wandb.ai/ren27-university-of-california-berkeley/vmcnet-phase-c/reports/SS-SPRING-%E2%80%94-full-campaign-summary--VmlldzoxNzc1MDYwNA).

| # | Script | Runs | Result | Report |
|---|---|---|---|---|
| **E9** | `e9_atoms_headtohead.sbatch` | 50/50 | Claim B holds (N +3.9σ, O tie); **Claim A fails on O**, 0/5 seeds | [E9](https://wandb.ai/ren27-university-of-california-berkeley/vmcnet-phase-c/reports/Plots-%E2%80%94-E9-N-and-O-atoms-%28generality%29--VmlldzoxNzcwOTQ1OA) |
| **E11** | `e11_eta_robustness_carbon.sbatch` | 24/24 | **PRIME-SR collapses at eta=0.005** (+12.5σ); Claim A fails at eta=0.05 | [E11](https://wandb.ai/ren27-university-of-california-berkeley/vmcnet-phase-c/reports/Plots-%E2%80%94-E11-learning-rate-robustness-%28carbon%29--VmlldzoxNzcwOTQ2NA) |
| **E10** | `e10_molecules_hometurf.sbatch` | 34/36 | **Claim B's best result**: +13σ N2, +38σ CO on PRIME-SR's home turf; SS-SPRING best arm on both | [E10](https://wandb.ai/ren27-university-of-california-berkeley/vmcnet-phase-c/reports/Plots-%E2%80%94-E10-N2-eq-and-CO-%28PRIME-SR-home-turf%29--VmlldzoxNzcyNTE3Mg) |

#### E10 — N2-eq and CO, PRIME-SR's home turf (eta = 0.002)

**EVAL energies, mHa above reference** (recovered from checkpoints — see below):

| arm | mu | eta/(1−mu) | N2-eq | CO |
|---|---|---|---|---|
| SPRING mu=0.9 | 0.90 | 0.020 | 20.018 ± 0.105 | 18.270 ± 0.132 |
| SPRING mu=0.95 | 0.95 | 0.040 | 14.900 ± 0.328 | 12.872 ± 0.248 |
| PRIME-SR | ~0.953 | 0.043 | 14.260 ± 0.123 | 12.063 ± 0.160 |
| SPRING mu=0.99 | 0.99 | 0.200 | 11.818 ± 0.095 | 9.607 ± 0.024 |
| SPRING mu=0.995 | 0.995 | 0.400 | 11.668 ± 0.048 | 9.210 ± 0.025 |
| **SS-SPRING** | ~0.995 | 0.400 | **11.410 ± 0.108** | **9.157 ± 0.118** |

- **Claim B, decisively — SS-SPRING is the best arm on both systems.** Ahead of PRIME-SR
  by **2.849 ± 0.217 mHa (13.1σ)** on N2 and **2.906 ± 0.075 (38.5σ)** on CO, 3/3 seeds
  on both. Strongest Claim B result in the campaign, on the ground most favourable to
  PRIME-SR.
- **Claim A met on CO, exceeded on N2.** CO: clean tie with the best fixed mu
  (−0.053 ± 0.095, 0.6σ). N2: SS-SPRING ahead of every fixed value **on every seed**
  (−0.257 ± 0.063 vs mu=0.995, 3/3). Temper the σ: 0.257 mHa is the order of the per-run
  MC error (0.14–0.33 mHa) and the paired s.e.m. of 0.063 is smaller than independent MC
  noise on 3 pairs would predict, so **the 3-of-3 sweep is the robust statement**.
- **The added arms were what made this interpretable.** Against the paper's own grid
  (mu ≤ 0.95), SS-SPRING would have "beaten tuned SPRING" by 3.5 mHa on N2 / 3.7 on CO —
  almost pure `eta/(1−mu)` artifact. See §4 item 8.
- **E11's mechanism reproduces**: r(log₁₀ eta/(1−mu), eval error) = **−0.89 on both**.
  The arm ordering is the momentum ordering.
- **Contradicts the PRIME-SR paper**: mu = 0.99 and 0.995 ran stably on both systems for
  every completed seed. Their reported instability at 0.99 here did not reproduce.

**⚠ Standing caveat: nothing is converged at 50k.** Every arm was still improving by
0.1–1.0 mHa per 5k training epochs at the end; absolute errors are 9–20 mHa against
0.2–0.8 on the atoms. The eval phase measures the wavefunction reached at 50k — it does
not fix under-convergence. **E10 compares arms at a fixed budget, not at their
asymptotes**; PRIME-SR is still falling ~0.3 mHa/5k, so its deficit would partially close
with a longer run. Per-run MC error is 0.14–0.33 mHa (eval ran at 1000 walkers, and these
are larger systems): far below the Claim B gaps, comparable to the Claim A margins.

**THE EVAL PHASE WAS LOST AND THEN RECOVERED — the operational lesson.** The first
submission finished 50k training on 34 of 36 runs and then *every one* died entering
eval with `RESOURCE_EXHAUSTED` (~7.94 GiB). The presets set `eval.nchains=2000` against
`vmc.nchains=1000`, so eval doubles the walker count; at 14 electrons × 16 determinants
that does not fit an 11 GB GTX2080TI. Carbon (6 electrons) did fit, which is why E7/E9/E11
never hit it. Recovery cost ~40 GPU-h instead of a ~300 GPU-h redo, via
`slurm/e10_eval_recovery.sbatch`:

- reload `checkpoints/50000.npz` — **not** `best_checkpoint.npz`, the reload default,
  which vmcnet picks on best error-adjusted *running average* energy: a best-of-training
  selection landing at a different epoch per arm, which would bias the comparison;
- `--config.vmc.nepochs=0` so `range(50000, 0)` is empty and control falls straight to
  eval (the training burn is skipped automatically when reloading);
- `--config.eval.nchains=1000`. Eval energy is an unbiased mean either way — walker count
  sets the error bar, not the expectation — so this stays comparable to the atoms.
- Keep `--job-name=e10-hometurf` so the `.out` matches the parser's glob; it prefers
  whichever file per index has eval epochs and merges the training tail back in.

`e10_molecules_hometurf.sbatch` now hard-codes `eval.nchains=1000`. **The presets
`preset_configs/{N2_eq,CO}.json` still say 2000** — fix them before any new script uses
those presets, or this recurs on anything N2-sized or larger.

**Recovering eval changed no conclusions**: arm means shifted by −0.65 to +0.41 mHa and
the ordering was identical on both systems to the training tails.

**The two externally-killed cells were rerun and are complete.**
`e10_N2_eq_spring_mu0.995_s2` and `e10_CO_spring_mu0.9_s0` died near epoch 1200 on the
first pass with no NaN and no traceback; both reran cleanly to 50k + eval, and the N2 one
turned the Claim A comparison there from a 2-seed hedge into a 3-of-3 sweep. Their
superseded wandb runs are retained as `*_killed1200` with `x_experiment=E10_superseded`
so they stay visible without polluting the panels.

**E9 — N and O atoms** (mHa above reference, 5 seeds, paired deltas):

| arm | N | O |
|---|---|---|
| SPRING mu=0.95 | 0.268 ± 0.007 | 0.764 ± 0.011 |
| SPRING mu=0.99 | 0.214 ± 0.013 | **0.604 ± 0.041** |
| SPRING mu=0.995 | **0.197 ± 0.012** | 0.630 ± 0.041 |
| PRIME-SR | 0.260 ± 0.010 | 0.795 ± 0.041 |
| SS-SPRING | 0.221 ± 0.011 | 0.783 ± 0.048 |

- N: Claim A holds (vs best fixed mu +0.024 ± 0.019, 1.3σ); Claim B holds (vs PRIME-SR
  -0.039 ± 0.010, 3.9σ, **5/5 seeds**).
- O: **Claim A fails** (vs SPRING(0.99) +0.178 ± 0.066, 2.7σ, **0/5 seeds won**); Claim B
  degrades to a tie (-0.012 ± 0.073). SS-SPRING is also the *least* seed-stable arm on O
  (spread 0.276) — the reverse of E7.
- The optimal mu differs by system: 0.995 on N, 0.99 on O. **Caveat: on N the grid does
  not bracket its optimum** (monotone in mu, best at the 0.995 edge), so N's tuned
  baseline is understated and Claim A's tie there would likely narrow against mu=0.999.
  Index-compatible to add: `case 5)` + `sbatch --array=50-59` (10 runs, ~45 GPU-h).

**E11 — carbon eta robustness** (mHa, 3 seeds; eta=0.02 column is E7, same protocol):

| arm | eta=0.005 | eta=0.02 | eta=0.05 | swing |
|---|---|---|---|---|
| SPRING mu=0.995 | **0.162 ± 0.001** | 0.161 | 0.191 ± 0.001 | 0.030 |
| SS-SPRING | 0.172 ± 0.027 | 0.148 | 0.192 ± 0.010 | 0.044 |
| SPRING mu=0.99 | 0.201 ± 0.012 | 0.147 | **0.155 ± 0.004** | 0.054 |
| PRIME-SR | **0.435 ± 0.025** | 0.217 | 0.193 ± 0.004 | **0.242** |

- **The ranking does NOT survive a change of eta.** PRIME-SR is 2.7x worse than every
  other arm at eta=0.005 and its swing is 4-8x larger than anyone else's.
- Claim A holds at 0.005 (+0.009 ± 0.028) and **fails at 0.05** (+0.037 ± 0.010, 3.8σ,
  0/3 seeds). Claim B: +12.5σ win at 0.005, tie at 0.05.

**Mean regret across all seven 50k+eval conditions** (C/N/O at eta=0.02, C at 0.005 and
0.05, N2/CO at 0.002), measured as mHa behind the best arm in each condition:

| arm | mean regret | worst-case regret |
|---|---|---|
| **SS-SPRING** | **0.036** | **0.178** |
| SPRING(0.995) | 0.055 | 0.257 |
| SPRING(0.99) | 0.131 | 0.450 |
| PRIME-SR | 0.913 | 2.906 |

**SS-SPRING has the lowest mean AND worst-case regret of any arm.** Against *per-condition
oracle tuning* it gives up only **−0.009 mHa on average** (i.e. marginally ahead), worst
case +0.178 (O), best −0.257 (N2). So the defensible form of Claim A is a **regret**
statement — best untuned default, parity with oracle tuning on average — not per-condition
parity, which fails on O and on carbon at eta=0.05.

> **CORRECTION (2026-08-15).** An earlier version of this log said "leaving SPRING at its
> published 0.99 beats SS-SPRING ~4x on mean regret". That was computed over only the four
> E9/E11 conditions, excluding E7 carbon and both E10 molecules — the two conditions
> SS-SPRING wins outright. Over all seven it is wrong and the ordering reverses. Use the
> table above.

**Next: E10, reframed.** It was gated on E9 confirming generality. Claim B generalised
and Claim A did not, so run E10 as a **Claim B** test on PRIME-SR's home turf — and do
not expect the tuned-SPRING arm to lose. E10 uses **eta = 0.002**, *below* E11's smallest
value and deep in the uncapped regime where PRIME-SR was worst by 2.7x. That is now the
sharpest prediction in the campaign.

**E10's arms were changed on 2026-08-11 in light of E11 — do not revert them.** The
original grid was mu ∈ {0.9, 0.95} only (the PRIME-SR paper's values), with 0.99
excluded because they report it unstable there. E11 makes that unusable: at eta=0.002
the cap essentially never binds, and in the uncapped regime the outcome tracks
`eta/(1−mu)` — 0.020 / 0.040 for the SPRING arms and 0.043 for PRIME-SR, against
**0.400** for SS-SPRING at its converged 0.995. SS-SPRING would win on a ~10× larger
effective step, for reasons unrelated to adaptivity, and any referee reading E10 next to
E11 would catch it. Arms 4 (mu=0.99, a fair tuned baseline) and 5 (mu=0.995, the
matched-constant control, cf. E8) were added; indices 0–23 keep their meaning. If the
high-mu arms really are unstable on N2/CO while SS-SPRING at beta≈0.995 is not, that is
itself a strong result — watch the `mu` trace and `check_for_nans`.

```bash
sbatch --exclude=n0135.savio3 --array=0-2,6-8,12-14,18-20,24-26,30-32 slurm/e10_molecules_hometurf.sbatch   # N2 first, 18 runs
```

Then CO with `--array=3-5,9-11,15-17,21-23,27-29,33-35`. **Re-time before committing to
the CO half**: the ~10–14 h/run figure extrapolates from a 200-epoch smoke test, so read
the per-epoch rate out of the first `.out` within ~30 min and multiply. All 36 runs is
~360–500 GPU-h.

**Infrastructure note — one bad GPU on n0135 cost 44 of the first 74 tasks.** All failures
were on `n0135.savio3`, dying in 1-57 s in the pre-flight `jax.devices()` check with
`CUDA_ERROR_NO_DEVICE`. n0135 *also* completed six normal 4-hour runs in the same window:
it has 4 GPUs and at least one is bad. The bad GPU freed its slot every few seconds and
was backfilled repeatedly, shredding tasks while healthy GPUs each held one run for 4+
hours. `sinfo` showed it `mixed-`, REASON `none` — **SLURM does not know it is broken**.
Resubmitting with `--exclude=n0135.savio3` recovered all 44 cleanly. Diagnose any repeat
with failures clustered on one NodeList at seconds-long Elapsed:

```bash
sacct -j <jobid> --format=JobID%22,NodeList%16,State%14,ExitCode,Elapsed | grep -v batch
```

### Phase D — COMPLETE (172 runs, 169 with eval): the same four experiments at 100k

**Result (2026-09-13).** 169/172 cells carry an eval energy. The three missing are the
appended `spring_mu0.999` arm on the molecules (`d10_N2_eq_spring_mu0.999_s1/s2`,
`d10_CO_spring_mu0.999_s1`), which NaN'd at ~15k epochs — `eta/(1-mu) = 2` at
eta=0.002, exactly the instability the D10 header flagged as possible. That arm is a
baseline-strengthening extra, not a table column; the other six arms are complete on
every system and seed.

**mHa above reference, 100k + eval.** Best fixed mu per system in *italics*.

| system | SPRING 0.95 | SPRING 0.99 | SPRING 0.995 | SPRING 0.999 | PRIME-SR | SS-SPRING |
|---|---|---|---|---|---|---|
| carbon (D7) | — | 0.138 ± 0.004 | *0.127 ± 0.006* | — | 0.183 ± 0.007 | 0.142 ± 0.013 |
| N (D9) | 0.230 | 0.170 ± 0.008 | *0.157 ± 0.001* | 0.204 ± 0.032 | 0.217 ± 0.009 | **0.157 ± 0.014** |
| O (D9) | *0.405 ± 0.135* | 0.453 ± 0.029 | 0.478 ± 0.030 | 0.572 | 0.577 ± 0.025 | 0.603 ± 0.045 |
| N2-eq (D10) | 14.050 | 11.120 ± 0.092 | *10.930 ± 0.056* | (1 seed) | 13.266 ± 0.105 | **10.576 ± 0.094** |
| CO (D10) | 11.667 | 8.838 ± 0.114 | *7.917 ± 0.743* | (2 seeds) | 11.219 ± 0.153 | 8.325 ± 0.210 |
| carbon eta=0.005 (D11) | — | 0.176 | *0.145 ± 0.008* | 0.159 | 0.378 ± 0.027 | 0.149 ± 0.004 |
| carbon eta=0.05 (D11) | — | *0.127 ± 0.009* | 0.142 | 0.146 | 0.138 ± 0.006 | 0.132 ± 0.007 |

H4 (D7, raw Ha): SS-SPRING **−2.03421**, SPRING(0.99) −2.03420, PRIME-SR −2.03410,
SPRING(0.8) −2.03393.

**What changed from 50k, and what did not.**

- **Claim A survives doubling the budget, with the same exception.** SS-SPRING ties or
  beats the best fixed mu on carbon, H4, N, N2-eq and both carbon etas, and still
  **loses on oxygen** (+0.198 against mu=0.95). Carbon at eta=0.05 is now a tie
  (+0.005 ± 0.011) where it lost at 50k, so that half of the 50k caveat was an
  undertraining artifact; oxygen was not.
- **CO flipped from a tie to a narrow loss** (SS-SPRING 8.325 vs mu=0.995's 7.917),
  but mu=0.995's s.e.m. there is 0.743 across 3 seeds — the largest on the board — so
  treat this as noise, not a result, until it has more seeds.
- **E10's non-convergence caveat is retired.** The molecular absolute errors fell from
  ~11.4/9.2 mHa to 10.6/8.3, and arm ordering is unchanged. 100k is now the protocol of
  record for the whole results section.
- **PRIME-SR's small-eta failure reproduces at 100k**: 0.378 mHa at eta=0.005 against
  0.145-0.176 for everything else, a 2.4x gap. §4 item 8's mechanism holds.
- **Claim B held on all seven Phase D conditions** (O is a statistical tie:
  0.603 ± 0.045 vs 0.577 ± 0.025). It is Phase E that breaks it.

### Phase E — COMPLETE (57 runs, 54 with eval): the paper's two missing baseline columns

**Why.** The comparison table in both papers carries MinSR and MinSR+M columns, and
neither baseline had ever been run in this campaign — the only mu=0 runs anywhere were
three Phase A E0 cells at 25k, one seed, at the wrong learning rate. N2 at 4.0 Bohr had
also never been run at all, despite the preset existing since Phase A.

- **E12** `e12_baselines_atoms_100k.sbatch` — MinSR and MinSR+M on carbon/N/O, 5 seeds,
  30 runs. 30/30.
- **E13** `e13_baselines_molecules_100k.sbatch` — the same two arms on N2-eq/N2-4.0/CO,
  3 seeds, 18 runs. 18/18.
- **E14** `e14_n2_stretched_100k.sbatch` — SPRING(0.99)/PRIME-SR/SS-SPRING on N2 at 4.0
  Bohr, 3 seeds, 9 runs. **6/9.**

Both baselines come from one new optimizer, `minsr_momentum`: mu=0 is MinSR (paper
Eqs. 41-42), mu=0.9 is MinSR+M (Eqs. 43-44). A unit test pins the mu=0 solve against
`spring.get_spring_step(mu=0.0)` to 1e-6, so the MinSR column is provably SPRING's own
linear solve with the momentum switched off rather than a lookalike reimplementation.
Learning rates are the SPRING paper's per-method tuned values (MinSR 0.1 / MinSR+M 0.2
on atoms, both 0.02 on molecules) — **not** SPRING's eta, which would manufacture a win
out of eta/(1-mu); see §4 item 8.

**mHa above reference (H4 excluded; N2-4.0 SPRING/PRIME-SR/SS-SPRING from E14).**

| system | MinSR | MinSR+M | SPRING 0.99 | PRIME-SR | SS-SPRING |
|---|---|---|---|---|---|
| carbon | 0.600 ± 0.029 | 0.254 ± 0.008 | 0.138 ± 0.004 | 0.183 ± 0.007 | 0.142 ± 0.013 |
| N | 0.741 ± 0.019 | 0.563 ± 0.172 | 0.170 ± 0.008 | 0.217 ± 0.009 | 0.157 ± 0.014 |
| O | 2.325 ± 0.050 | 3.002 ± 0.639 | 0.453 ± 0.029 | 0.577 ± 0.025 | 0.603 ± 0.045 |
| N2-eq | 21.086 ± 0.358 | 15.539 ± 0.062 | 11.120 ± 0.092 | 13.266 ± 0.105 | 10.576 ± 0.094 |
| **N2 4.0** | 35.479 ± 11.259 | 18.075 ± 0.085 | 14.990 ± 4.383 | 17.606 ± 0.714 | **DIVERGED 3/3** |
| CO | 19.829 ± 0.233 | 13.586 ± 0.283 | 8.838 ± 0.114 | 11.219 ± 0.153 | 8.325 ± 0.210 |

**Reproduction against the SPRING paper's Table 1** (their values as mHa above
benchmark: MinSR 0.5/0.7/2.1/15.5/22.7/16.3; SPRING 0.1/0.2/0.5/10.1/11.5/8.6):

- The **atoms reproduce closely** — our MinSR 0.600/0.741/2.325 against their
  0.5/0.7/2.1, our SPRING 0.138/0.170/0.453 against their 0.1/0.2/0.5.
- On **molecules our SPRING matches but our MinSR does not** (SPRING 11.1 vs 10.1 and
  8.8 vs 8.6; MinSR 21.1 vs 15.5 and 19.8 vs 16.3). That is the expected shape of our
  missing KFAC preliminary phase: it exists precisely to remove the chaotic early
  stage, and the arm with no momentum to carry it through suffers most. Say so in the
  table caption rather than leaving the gap unexplained.
- **The paper's ordering inverts on oxygen** — our MinSR+M (3.002) is worse than our
  MinSR (2.325). MinSR+M is also the least seed-stable arm on the atoms (s.e.m. 0.172
  on N, 0.639 on O, against MinSR's 0.019 and 0.050), again consistent with naive
  momentum being the arm most exposed to a random start.
- Transcription warning: the paper's N2-equilibrium MinSR+M cell reads **−108.5294**, a
  full Hartree above every other entry in that row and *below* the benchmark. Almost
  certainly a typo for −109.5294. Check the arXiv v2 before citing it.

#### E14 — SS-SPRING DIVERGES ON STRETCHED N2. Claim B's first counterexample.

All three `ssu_defaults` seeds ended `VMC terminated due to Nans! Aborting.`, at epochs
18572, 17342 and 23559. This is **not** a bad start and **not** infrastructure — the
noclip energy trace shows it tracking the other arms exactly and then coming apart:

| epoch | 5000 | 10000 | 15000 | 17000 |
|---|---|---|---|---|
| SPRING mu=0.99 s0 | −108.967 | −109.088 | −109.134 | −109.144 |
| PRIME-SR s0 | −108.993 | −109.011 | −109.086 | −109.140 |
| **SS-SPRING s0** | −109.015 | −109.010 | **−100.086** | −106.038 |
| **SS-SPRING s1** | −109.084 | −109.060 | **−105.657** | −107.376 |
| **SS-SPRING s2** | −108.835 | −109.011 | **−101.048** | −108.685 |

Healthy through 10k, loses 4-9 Hartree between 10k and 15k on every seed, partially
recovers, then dies. Same eta (0.002), same preset family and same 14 electrons as
N2-eq and CO, where it is the *best* arm — so this is specific to the stretched
geometry, not to molecules or to the learning rate.

**Do not resubmit `--array=6-8` expecting a number.** Same seeds and same random init
reproduce the same divergence; 3/3 seeds failing means more seeds very likely fail too.
The honest table cell is "diverged".

**The mechanism is not yet identified**, and this is the cleanest case the campaign has
ever had for settling the standing Phase A question — whether beta locks near 1 or
collapses toward 0 under stress (opposite fixes). The evidence needed is `mu.txt`,
`r_hat.txt` and `probe_r_ip.txt` from the three logdirs under
`phase_e/e14_n2_stretched_100k/`, epochs 9k-14k. Those are **not** in the `.out` files
and wandb drops nothing useful here either — they must be pulled from Savio. Zero
compute; highest information per GPU-hour of anything outstanding.

### Phase D — superseded header (the plan, as written before the runs)

**Why.** Both comparison papers report at 100k. E10 was visibly unconverged at 50k
(every arm still improving 0.1-1.0 mHa per 5k, no arm reaching chemical accuracy on
either molecule), so its result is a fixed-budget comparison and PRIME-SR's ~2.9 mHa
deficit would partially close given longer. 100k puts the whole results section on one
protocol that matches both papers and removes that caveat -- or forces it to be
restated honestly.

**What did NOT change, contrary to the review feedback that prompted this.** The
decaying learning rate and the norm constraint were already on. All 337 real runs in
Phases A-C used `inverse_time` with `learning_decay_rate=1e-4`, no exceptions; 303 of
337 used `constrain_norm=true, norm_constraint=1e-3`, and the only 34 that did not are
E2's constraint-off arm (15) and all of E4 (18), which were the stress tests that
established why it is on elsewhere -- 10 of 15 constraint-off runs diverged inside ~160
epochs. Phase D now passes all four settings explicitly on the command line so they
appear in the `.out` config dump instead of resting on defaults.

**What DID change:** `nepochs` 50000 -> 100000 (the presets already said 100000; the
Phase B/C scripts were overriding them *down*), and a **mu=0.999 arm** on D9/D10/D11.

**Why mu=0.999.** "Tuned SPRING" meant best-of-{0.95, 0.99, 0.995}, and on four of the
seven conditions the winner was the TOP of that grid with the trend still going: N
(0.268 -> 0.214 -> 0.197), N2 (14.900 -> 11.818 -> 11.668), CO (12.872 -> 9.607 ->
9.210) and carbon at eta=0.005 (0.201 -> 0.162). That baseline is censored, not tuned,
and claim A rests on it. A stronger fixed mu can only narrow or flip our claim-A ties --
which is the reason to run it, not to skip it. Claim B is untouched either way:
PRIME-SR settles near 0.953 regardless. O and carbon at eta=0.02/0.05 have interior
optima and need no new arm. Watch for instability: eta/(1-mu) jumps 5x at 0.999.

| # | script | array | runs | ancestor | note |
|---|---|---|---|---|---|
| **D7** | `d7_headtohead_seeds_100k.sbatch` | `0-39` | 40 | E7 | arms unchanged |
| **D9** | `d9_atoms_headtohead_100k.sbatch` | `0-59` | 60 | E9 | +mu=0.999 (arm 5) |
| **D11** | `d11_eta_robustness_carbon_100k.sbatch` | `0-29` | 30 | E11 | +mu=0.999 (arm 4) |
| **D10** | `d10_molecules_hometurf_100k.sbatch` | `0-41` | 42 | E10 | +mu=0.999 (arm 6) |

**Every new arm is APPENDED, never inserted**, so D-index *i* means what E-index *i*
meant and the two protocols compare cell by cell. Verified by simulating all 212 index
mappings against the sbatch arithmetic.

**The job names start `d`, not `e`, and this is load-bearing.**
`parse_eval_energies.py` globs on job name, so a job called `e9-atoms-100k` would match
`slurm-e9-atoms-*` and be silently averaged into the 50k results. Do not rename them.

**Parser support is in place.** `parse_eval_energies.py` now registers D7/D9/D10/D11 —
plus **E7, which was never registered**, because H4 has no literature reference and the
mHa-above-reference path did not apply to half its cells; `reference_for()` now returns
None there and both the report and the backfill fall back to raw Ha. Each experiment
carries its own wandb project, so `--experiment E10 D10` parses both and backfills each
to the right place. Phase D runs land in **`vmcnet-phase-d`**.

**Still not fixed:** `preset_configs/{N2_eq,CO}.json` say `eval.nchains=2000` against
`vmc.nchains=1000`. D10 overrides it, as E10 did, but the presets remain a trap for any
new script.

## 6. Protocol (hold these fixed)

- **Random init throughout**, no KFAC preliminary phase. Matches PRIME-SR's electronic
  protocol, and is *required* because reload restores the RNG key (see §7).
- float32, exactly 1 GPU per run (spectral indicators and the probe are per-device-batch
  quantities), eta=0.02 unless stated, `check_for_nans=True` on sweeps.
- **Always use `energy_noclip`.** The clipped energy drives the gradient but is biased,
  and the bias depends on walker count — it corrupts any cross-nchains comparison.
- **Score with mean ± s.e.m. across seeds, worst seed, and spread.** Not mean alone:
  E6 showed the effect can live entirely in seed sensitivity.
- Final energies come from the **eval phase**, not a training tail.
- Carbon reference −37.8450 Ha (Chakravorty 1993). **H4 has no literature benchmark** —
  the in-repo −2.0301 is our own FCI/cc-pVTZ estimate and VMC goes *below* it, so quote
  H4 as raw energies and compare arms to each other only.

## 7. Operational gotchas (each of these cost real time)

**vmcnet**
- `reload` restores `data`, `params` *and the RNG key*, so `--config.initial_seed` has
  no effect and `nchains` silently cannot change under a reloaded checkpoint. Hence
  random init everywhere.
- `H4.json` originally inherited `nepochs=200000`; all presets now set vmc/eval blocks
  explicitly.
- `check_for_nans` defaults to **False**, and `nan_safe` masks NaNs — a diverged run
  otherwise burns its whole allocation.

**wandb**
- The `ml_collections` ConfigDict is logged as **one YAML string**, so no sweep
  parameter is queryable. Backfill `x_*` config fields (skill script does this).
- The built-in summary holds **only the last step** — too noisy to rank runs. Backfill
  tail averages.
- **The eval phase restarts its step counter, so wandb DROPS it entirely.** Eval
  energies — the publishable numbers — must be parsed from the slurm `.out` files
  (eval lines lack the noclip parenthetical) and backfilled as `x_eval_energy`.
  Always pull `.out` files locally alongside syncing.
- `sync_wandb.sh`: run **unfiltered** on a login node. The name filter once matched
  only `wandb-metadata.json`, which is often absent, and silently synced 1 of 24 runs
  while looking successful. Fixed to fail open, but unfiltered is still safer.

**`wandb sync` beta path is broken on the Savio login nodes — use `--legacy`** (found
2026-09-13). A bare `wandb sync` prints "Using wandb beta sync" and then dies with
`ServicePollForTokenError: Failed to read port info after 30.0 seconds (wandb-core
PID=...)`. The helper process never comes up. `wandb sync --legacy "$d"` works every
time. Symptom to recognise: the run directory is fine and the run itself completed
normally — only the uploader fails. Also note `wandb` is **not on `PATH`** outside
`sync_wandb.sh`; prepend
`export PATH="/global/scratch/users/$USER/envs/vmcnet/bin:$PATH"` before any manual
`wandb` invocation or you get a bare `command not found` that looks like a broken env.

**A `.synced`-less directory tree makes a "sync Phase E" command re-upload the entire
campaign** (found 2026-09-13, cost several hours and wiped Phase C's backfill). The
older runs had no `.synced` markers, so a naive loop over `offline-run-*` walked from
July 30 forward, re-uploading ~500 runs that were already in wandb — and because a
re-sync reverts server-side state (below), it destroyed the backfilled `x_eval_*`
summary fields on Phases A-C *and* undid the `*_killed1200` renames on the two
superseded E10 cells, recreating a duplicate-name collision that then let the backfill
stamp a dead 1,200-epoch run with the good run's eval energy. Two defences: mark
everything already uploaded (`for d in .../offline-run-2026{0730,08}*; do touch
"$d/.synced"; done`), and **scope any manual sync loop to the date range you actually
mean**. Verify afterwards with a duplicate-name check, not just a run count.

**`wandb sync` REVERTS server-side edits — re-run the backfill after every sync.** Syncing
an offline run directory restores that run's *local* state, wiping anything changed
through the API: run name, notes, and **all backfilled `x_*` summary fields**. Observed
2026-08-15, when re-syncing two E10 runs undid both a rename and a field cleanup. Config
`x_*` survived; summary `x_*` did not. Consequences:

- After any `bash slurm/sync_wandb.sh`, re-run
  `python slurm/parse_eval_energies.py --experiment ... --backfill`. It is idempotent, so
  this is free insurance.
- Renaming a superseded run to free its cell name is not durable — a later sync brings the
  duplicate back. Either delete the stale offline run dir under
  `/global/scratch/users/$USER/wandb/wandb/` so it cannot be re-synced, or accept that the
  rename may need repeating.
- Verify writes with a **fresh** `wandb.Api()`. The client caches `api.runs()` within an
  instance, so a rename verified through the same object reads back stale and looks like
  it failed when it did not.

**Reports** — see the `wandb-experiment-report` skill (`~/.claude/skills/`), which
encodes these and ships a verifier. Two that bite hardest: `Runset` filters must use
`Config('k') == v` (the `config.k == v` form parses to an *empty* filter and silently
shows every run in the project), and `Report.from_url` never returns the real width, so
every load-edit-save clobbers `fluid` back to `readable` unless you re-set it.

Two more found while writing the E9/E11 reports:

- **Never put `/` in a report title.** The slug is built from the title, so a title like
  "29/50 runs" adds a path segment and `Report.from_url` dies with
  `ValueError: too many values to unpack (expected 5)` — the report becomes uneditable
  through the API. Write "29 of 50". (Both Phase C reports hit this and were renamed.)
- **To edit a report in place, set `report.id`** to the trailing base64 of its URL before
  `save()`. A bare `wr.Report(...)` + `save()` always mints a *new* report, so a
  re-run of a report script silently creates duplicates rather than updating.
- The verifier flags any runset matching 0 runs, because that is normally the empty-filter
  bug. When a cell is genuinely empty — E9 has no PRIME-SR runs on O — that FAIL is
  expected; confirm the same filter pattern returns runs elsewhere, keep the runset so
  the absence is visible, and say so in the report text. Do not delete the runset to
  make the verifier quiet.

## 8. Code and assets added during the campaign

- **Diagnostics** in all three optimizers: `update_sq_norm_preclip`,
  `norm_cap_applied`; SS-SPRING adds `probe_res_norm`, `probe_r_ip` (new `r_ip` state);
  PRIME-SR gains a `mu_cap` ablation knob (default 1.0 = no-op).
- **SPRING momentum schedule**: `mu_init`, `mu_warmup_steps`, `mu_ramp_steps` (defaults
  are a no-op — constant-mu SPRING is bit-identical). SPRING now logs `mu` too, so all
  three methods overlay on one momentum axis. Tests in
  `tests/units/updates/test_spring_mu_schedule.py`.
- **Presets**: `H4.json` (fixed), `N2_eq.json`, `N2_4.0.json`, `N.json`, `O.json`,
  `CO.json`, `reference_energies.json`.
- **Runbooks**: `docs/PHASE_A.md`, `docs/PHASE_B.md`, `docs/PHASE_C.md`, this log.
- **Paper draft**: `docs/results_section.md` — the results section, every number
  recomputed from the .out files / wandb at draft time rather than transcribed.
- **Skill**: `wandb-experiment-report` — builds these reports and verifies them.

## 9. How to keep this document current

Update it in the same pass as the wandb report for a finished experiment — while the
reasoning is still loaded. Deferring is how it rots. Per experiment, touch:

- **§3** if the headline result moved; **§5** always (one row: what ran, what it found,
  report link); **§10** so the document still ends with an executable next action.
- **§4** whenever a hypothesis dies — the highest-value section here, because it is
  what stops a future session re-proposing an idea that already cost GPU-hours. Record
  the number that killed it, not just the verdict.
- **§7** for any operational gotcha that cost real time. Those are invisible in the
  code and brutal to rediscover.

Two rules. Write contradicting results in the same plain voice as supporting ones — a
log that reads as advocacy is useless for planning. And when a later experiment
overturns an earlier conclusion, **correct the earlier entry in place**, don't only
append: a reader who stops halfway must not walk away believing something already
disproved. (E0's row and §4 item 5 are the worked example.)

Ideas rejected *without* running anything belong in §4 too, marked as reasoned rather
than measured — they are just as expensive to rediscover.

## 10. If you are picking this up cold

1. Read §3 (the qualified headline) and §4 (ideas already ruled out). The single most
   important fact is no longer the one this list used to open with. **Claim B held on
   six systems and three learning rates and then found a counterexample on the seventh:
   SS-SPRING diverges on N2 at 4.0 Bohr, 3/3 seeds (§4 item 9, §5 Phase E).** Claim A
   still does not generalise (oxygen). Lead with Claim B *scoped to stability*, and
   never write "never lost" again.
2. **All compute is done. Phases A-E: 617 runs, no outstanding cells worth re-running.**
   Phase D is 169/172 (the 3 misses are the appended mu=0.999 molecular arm, which
   diverged at eta/(1-mu)=2); Phase E is 54/57 (the 3 misses are the E14 SS-SPRING
   divergence, which is a result, not a gap). **Do not resubmit either set** — both are
   deterministic given the seed and will reproduce.
3. **The one outstanding task is zero-compute and high-value: diagnose the E14
   divergence.** Pull `mu.txt`, `r_hat.txt`, `probe_r_ip.txt` from the three logdirs
   under `phase_e/e14_n2_stretched_100k/` and read epochs 9k-14k. This settles the
   standing Phase A question (does beta lock near 1, or collapse toward 0?) on the
   cleanest instance the campaign has produced, and a referee will ask whether the same
   failure lurks elsewhere. Nothing in the paper is safe to write until it has an
   answer or an explicit "we do not know".
3. **The paper's spine, in the order the evidence supports:** (a) Claim B — SS-SPRING is
   at least as good as PRIME-SR on six systems and three learning rates **wherever it is
   stable**, beating it by ~2.5 mHa on PRIME-SR's *own* N2-eq and CO at their *own*
   settings — **and it is not stable on the seventh, N2 at 4.0 Bohr, where it diverges
   3/3 and PRIME-SR completes.** The stability exception must travel with the claim in
   the abstract, not be deferred to a limitations paragraph.
   (b) The mechanism — PRIME-SR's momentum is blind to eta, settles at ~0.953, and that
   costs it a ~10× smaller effective step wherever the norm cap is not binding (§4 item
   8; r = −0.89 to −0.92 between log₁₀(eta/(1−mu)) and final error). (c) Claim A, scoped
   honestly: parity without tuning on carbon, H4, N, CO and N2-eq; loses on O and on
   carbon at eta=0.05.
4. **Fix `preset_configs/{N2_eq,N2_4.0,CO}.json`** — all three still set
   `eval.nchains=2000` against `vmc.nchains=1000`, which OOM'd E10's entire eval phase.
   D10, E13 and E14 all override it, but any new script using those presets hits it
   again. (N2_4.0 was added to this list in Phase E — same 14 electrons as CO.)
5. ~~Optional: add mu = 0.999 to E9's grid.~~ **Done in Phase D** (appended as arm 5 to
   D9/D10/D11). On N it is *worse* than 0.995 (0.204 ± 0.032 vs 0.157 ± 0.001), so the
   grid-edge worry was unfounded — the optimum is interior after all, and N's tuned
   baseline was not understated. On the molecules at eta=0.002 it diverged on 3 of 6
   cells (`eta/(1-mu) = 2`).
6. **Do not** re-propose: the small-N hypothesis, the warm-up mechanism, the
   matched-constant explanation, or an eta sweep *as a step-size study on short runs*.
   All settled — see §4.
7. **Do not** overstate Claim A. It fails on oxygen (0/5 seeds) and carbon at eta=0.05
   (0/3). But do not swing too far the other way either: across all seven conditions
   SS-SPRING has the LOWEST mean (0.036 mHa) and worst-case (0.178) regret of any arm, and
   matches oracle per-condition tuning to −0.009 mHa on average. State Claim A as a
   regret result, not as per-condition parity.
8. **Quote D10, not E10, for the molecules.** The 50k non-convergence caveat is retired
   — D10 doubles the budget, absolute errors fall ~0.8 mHa, and arm ordering is
   unchanged. What still holds: the sub-0.5 mHa Claim A margins there remain the order
   of the per-run MC error, so quote the seed sweeps alongside any σ. Molecular eval
   uses **1000 inference walkers, not the papers' 2000** (memory limit at 14 electrons);
   the estimator is unbiased either way, the error bar is √2 wider, and the measured
   blocked MC error is ~0.19 mHa against ~0.035 on the atoms. Footnote it.
9. **Never compare arms at small eta without a high-momentum fixed baseline.** E10's
   original grid (mu ≤ 0.95) would have produced a 3.5 mHa "win" that was pure
   `eta/(1−mu)` artifact. §4 item 8 is the general statement of this trap.
10. **Always pull the `.out` files and check the eval phase actually ran**, and **re-run
    the backfill after every sync** (§7). wandb drops eval entirely and a sync reverts
    server-side edits, so both failures are invisible in the wandb UI.
