# Campaign log — SS-SPRING vs SPRING vs PRIME-SR

**Handoff document.** Read this first in a new session; it is self-contained. Last
updated 2026-08-11, after the partial E9/E11 submission. 209 runs, ~530 GPU-hours, all
on Savio GTX2080TIs.

> **Current state (2026-08-11): Phase C is half-submitted and mostly lost.** E9 and E11
> went in together; **44 of their 74 array tasks died at launch** on a GPU allocation
> fault (`CUDA_ERROR_NO_DEVICE`), not on anything physical. E11 has **1 of 24** runs and
> is effectively unrun; E9 has 29 of 50 with PRIME-SR absent on O and SS-SPRING down to
> one seed on N. **Next action is resubmitting both arrays** (§5, Phase C) — and
> checking node health first, or a blind resubmit may lose a similar fraction again.

---

## 1. The goal

Two claims about **same-sampled SPRING (SS-SPRING)**, our adaptive-momentum variant:

- **Claim A** — untuned, it matches *optimally tuned* SPRING.
- **Claim B** — it matches or beats **PRIME-SR** (Wang & Liu, arXiv:2604.18357), the
  competing adaptive-momentum SR method.

**Status: both supported on two systems (E7).** Phase C, written but not yet run,
hardens them against the obvious objections.

## 2. The three methods

All three are stochastic-reconfiguration optimizers for neural-network VMC that differ
*only* in how they set the momentum applied to the previous update:

| | momentum | sees the step size? | sees the energy residual? |
|---|---|---|---|
| **SPRING** | fixed `mu`, hand-tuned (paper default 0.99) | no | no |
| **PRIME-SR** | `mu_k` recomputed each step from the sampled Gram-matrix spectrum (effective dimension α, numerical rank, principal-subspace overlap) | **no** | **no** |
| **SS-SPRING** | `beta` from a *same-sampled probe*: a second SPRING solve on the same sampled operators against a fixed synthetic target, whose measured residual contraction sets the momentum | yes, via the probe | no |

Implementations: `vmcnet/updates/{spring,prime_sr,same_sampled_spring_unified}.py`.

## 3. Headline result (E7)

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
7. **eta is the step size.** It largely is not (E2). With the norm constraint on, the
   realized step is `min(η‖φ‖, √C)`; once the cap binds — which is nearly always — eta
   cancels out of the update entirely. **C (`norm_constraint`) is the real step-size
   knob.** Energies are flat across a 40× eta range on carbon.

**Still unexplained:** SS-SPRING beats SPRING(0.99) on carbon by ~0.10 mHa at ~2σ, and
it is neither the warm-up (E6) nor the converged constant (E8). What remains is beta's
*trajectory* between those endpoints. The claims do not depend on resolving this;
treat it as future work rather than a blocker.

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

### Phase C — E9/E11 submitted 2026-08-10, **44 of 74 runs lost at launch** (see `docs/PHASE_C.md`)

Closes the three referee objections to E7. **Order: E9 + E11 first, E10 only after E9.**
Project `vmcnet-phase-c`. Both submitted arrays hit the same GPU allocation fault, so
**neither experiment has yet answered its question.**

| # | Script | Ran | Cost | Status |
|---|---|---|---|---|
| **E9** | `slurm/e9_atoms_headtohead.sbatch` | **29/50** | ~130 GPU-h | inconclusive; leans *against* Claim A. [E9](https://wandb.ai/ren27-university-of-california-berkeley/vmcnet-phase-c/reports/Plots-%E2%80%94-E9-N-and-O-atoms-%28generality%29-%E2%80%94-INCOMPLETE%2C-29-of-50-runs--VmlldzoxNzcwOTQ1OA) |
| **E11** | `slurm/e11_eta_robustness_carbon.sbatch` | **1/24** | ~4 GPU-h | did not run. [E11](https://wandb.ai/ren27-university-of-california-berkeley/vmcnet-phase-c/reports/Plots-%E2%80%94-E11-learning-rate-robustness-%28carbon%29-%E2%80%94-DID-NOT-RUN%2C-1-of-24-runs--VmlldzoxNzcwOTQ2NA) |
| **E10** | `slurm/e10_molecules_hometurf.sbatch` | not submitted | ~250–330 GPU-h | **blocked** — gated on a complete E9 |

**The fault.** 21 E9 tasks and 23 E11 tasks were allocated nodes with no visible GPU and
died in the pre-flight `jax.devices()` check, before `vmc-molecule` ran:
`CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected`. No logdir is created on that
path, so the existing-logdir guard will not block a resubmit. Every one of the 30 runs
that *did* start finished cleanly — full 50k + 20k eval, no NaNs.

**Resubmit** (check node health first — a blind resubmit may lose a similar fraction):

```bash
sbatch --array=7,10,13,16-17,24,27,32-33,35-39,41-45,47-48 slurm/e9_atoms_headtohead.sbatch
sbatch --array=0-3,5-23 slurm/e11_eta_robustness_carbon.sbatch
```

**What the partial E9 shows** (eval energies, mHa above reference; unpaired — no seed is
present in all five arms):

- N: SPRING 0.995 **0.186 ± 0.008** (n=4) < 0.99 0.220 ± 0.020 (n=3) < SS-SPRING 0.245
  (n=**1**) < 0.95 0.268 ± 0.007 (n=5) ≈ PRIME-SR 0.270 ± 0.014 (n=3).
- O: SPRING 0.995 **0.623 ± 0.051** (n=4) < 0.99 0.637 ± 0.052 (n=3) < 0.95
  0.756 ± 0.010 (n=4) < SS-SPRING 0.770 ± 0.138 (n=**2**). PRIME-SR: **no runs**.
- On the only three same-seed comparisons against the best SPRING arm, SS-SPRING loses
  all three (N s0 0.245 vs 0.164; O s1 0.908 vs 0.567; O s4 0.632 vs 0.545). Against
  PRIME-SR exactly one paired point exists (N s0), where SS-SPRING is ahead 0.245 vs
  0.270. **Too thin to overturn E7 — and too thin to confirm it.**
- **The mu grid does not bracket its optimum**: the ordering is monotone in mu on both
  atoms, so the best value sits at the grid edge (0.995). Add **mu = 0.999** on the
  resubmit, or the "tuned baseline" is itself untuned. E0 found carbon flat to 0.999.
- **Momentum matters more on atoms than on carbon** — spread across the three fixed-mu
  arms is 0.082 mHa (N) and 0.133 mHa (O) vs a flat plateau on carbon. Good venue for
  the claim; worth finishing.

**New diagnostic — the norm cap releases mid-run.** `norm_cap_applied` binds on 37–97%
of steps in the first half of training and then sits at **exactly 0** for the entire
second half, on every arm and both atoms. Phase A's "capped nearly always" was measured
at 25k. Since eta only cancels *while* the cap binds, the back half of a 50k run is
genuinely eta-sensitive — this makes E11's premise **stronger** than its runbook assumed,
and is an argument for resubmitting it promptly.

**SS-SPRING's probe is at its ceiling.** `probe_r_ip` tail-averages 1.02–1.03 (>1 = the
probe residual is *growing*), pinning tail beta at 0.9943–0.9950 — just above the best
fixed mu here. Same near-constant behaviour recorded across the whole campaign (§4.2).

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

1. Read §3 and §4 — the result, and the ideas already ruled out.
2. **Resubmit the missing E9 and E11 array cells** — the two `sbatch --array=...` lines
   in §5, Phase C. That is the next action; 44 of 74 tasks were lost to a node fault,
   not to anything physical. Check node health first. Consider adding **mu = 0.999** to
   E9's grid: the current grid's optimum sits at its edge.
3. After they finish: `bash slurm/sync_wandb.sh` on a login node, pull the `.out` files,
   and update the two existing `vmcnet-phase-c` reports with the
   `wandb-experiment-report` skill, remembering to backfill eval energies from the logs.
   Both reports are already wired with per-arm runsets, so refreshed panels fill in
   automatically; the text and titles need rewriting once the arms are populated.
4. **Do not** re-propose: the small-N hypothesis, the warm-up mechanism, the
   matched-constant explanation, or an eta sweep as a step-size study. All settled —
   see §4.
5. **Do not** read E9's current numbers as overturning E7. SS-SPRING has one seed on N
   and two on O there; the apparent deficit is unpaired and within seed noise. It is a
   reason to finish the experiment, not a result.
