# Campaign log — SS-SPRING vs SPRING vs PRIME-SR

**Handoff document.** Read this first in a new session; it is self-contained. Last
updated 2026-08-11, after E9 and E11 completed. 253 runs, ~700 GPU-hours, all on Savio
GTX2080TIs.

> **Current state (2026-08-11): E9 and E11 are complete (74/74) and THE TWO CLAIMS HAVE
> COME APART.**
>
> - **Claim B (SS-SPRING ≥ PRIME-SR) is now the strong result.** Tested on four systems
>   and three learning rates, it has **never lost**: +12.5σ on carbon at eta=0.005, ~7σ
>   on carbon at 0.02, 3.9σ on N, 3.4σ on H4; ties on O and on carbon at eta=0.05.
> - **Claim A (untuned SS-SPRING matches tuned SPRING) does NOT generalise.** It holds on
>   carbon, H4, N and carbon at eta=0.005, but fails on **oxygen** (0.178 ± 0.066 mHa
>   behind SPRING(0.99), losing 5/5 seeds) and on **carbon at eta=0.05** (0.037 ± 0.010,
>   losing 3/3). Across the four Phase C conditions, leaving SPRING at its published
>   default of 0.99 beats SS-SPRING by ~4× in mean regret. Do not state Claim A as a
>   general property.
> - **New: PRIME-SR breaks at small eta** — 2.7× worse than every other arm at
>   eta=0.005, swing 0.242 mHa across eta vs 0.030–0.054 for everything else. This is the
>   clean PRIME-SR failure the campaign set out to find (§1), and it is mechanistically
>   explained (§4 item 8).
>
> **Next action: E10**, reframed as a Claim B test — see §5.

---

## 1. The goal

Two claims about **same-sampled SPRING (SS-SPRING)**, our adaptive-momentum variant:

- **Claim A** — untuned, it matches *optimally tuned* SPRING.
- **Claim B** — it matches or beats **PRIME-SR** (Wang & Liu, arXiv:2604.18357), the
  competing adaptive-momentum SR method.

**Status after Phase C (E9, E11): the claims have separated.** Claim B holds everywhere
tested — four systems, three learning rates, never once lost. **Claim A does not
generalise**: it holds on carbon, H4, N and carbon at eta=0.005, and fails on oxygen
(0/5 seeds) and carbon at eta=0.05 (0/3). See the table in §3 and the detail in §5.

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

## 3. Headline result (E7, as qualified by E9/E11)

> **Read §5 Phase C before quoting anything here.** E7's two-system tie is real but does
> **not** generalise: across six conditions Claim B never loses, while Claim A fails on
> oxygen (5/5 seeds) and on carbon at eta=0.05 (3/3). The summary across everything
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

### Phase C — E9 and E11 COMPLETE (74 runs, ~300 GPU-h); E10 outstanding

Closed two of the three referee objections to E7. Project `vmcnet-phase-c`.

| # | Script | Runs | Result | Report |
|---|---|---|---|---|
| **E9** | `e9_atoms_headtohead.sbatch` | 50/50 | Claim B holds (N +3.9σ, O tie); **Claim A fails on O**, 5/5 seeds | [E9](https://wandb.ai/ren27-university-of-california-berkeley/vmcnet-phase-c/reports/Plots-%E2%80%94-E9-N-and-O-atoms-%28generality%29--VmlldzoxNzcwOTQ1OA) |
| **E11** | `e11_eta_robustness_carbon.sbatch` | 24/24 | **PRIME-SR collapses at eta=0.005** (+12.5σ); Claim A fails at eta=0.05 | [E11](https://wandb.ai/ren27-university-of-california-berkeley/vmcnet-phase-c/reports/Plots-%E2%80%94-E11-learning-rate-robustness-%28carbon%29--VmlldzoxNzcwOTQ2NA) |
| **E10** | `e10_molecules_hometurf.sbatch` | not submitted | ~250-330 GPU-h | **next action** — reframe as a Claim B test, see below |

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

**Mean regret across the four Phase C conditions** (mHa behind the best arm in each):
SPRING(0.99) **0.014** < SPRING(0.995) 0.016 < SS-SPRING 0.063 < PRIME-SR 0.141. A
practitioner who never tuned and left SPRING at its published default would have beaten
SS-SPRING by ~4x. This is the central problem for Claim A.

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

1. Read §3 (the qualified headline) and §4 (ideas already ruled out). The single most
   important fact: **Claim B survived everything; Claim A did not generalise.**
2. **Next action is E10**, reframed as a Claim B test on PRIME-SR's home turf — see §5.
   Run N2 first. Keep `--exclude=n0135.savio3` until BRC drains that node. E10 runs at
   eta = 0.002, deep in the uncapped regime where E11 found PRIME-SR worst by 2.7×, so
   the prediction is that PRIME-SR does badly there. Do *not* expect tuned SPRING to lose.
3. Optional but honest: add **mu = 0.999** to E9's grid. On N the ordering is monotone in
   mu, so the optimum sits at the grid edge and N's tuned baseline is understated —
   Claim A's tie on N would likely narrow against 0.999. Index-compatible: append
   `case 5)` and submit `--array=50-59` (10 runs, ~45 GPU-h). It cuts against our own
   claim, which is the reason to run it.
4. After any run finishes: `bash slurm/sync_wandb.sh` **unfiltered** on a login node, pull
   the `.out` files, then `python slurm/parse_eval_energies.py --experiment E9 E11
   --backfill`. That tool globs across job ids, prints coverage, emits the exact resubmit
   command for anything missing, and flags duplicate or unsynced runs. Then update the
   reports with the `wandb-experiment-report` skill.
5. **Do not** re-propose: the small-N hypothesis, the warm-up mechanism, the
   matched-constant explanation, or an eta sweep *as a step-size study on short runs*.
   All settled — see §4. (An eta study on 50k runs was worth it; see §4 item 7.)
6. **Do not** overstate Claim A. "Untuned SS-SPRING matches tuned SPRING" is false as a
   general statement: it loses to SPRING(0.99) on oxygen on 5/5 seeds and on carbon at
   eta=0.05 on 3/3, and plain SPRING at its published default beats it ~4× on mean regret
   across the four Phase C conditions. Scope it or lead with Claim B.
