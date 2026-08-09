# Phase B runbook: nailing the two claims

Phase A (E0–E5) was exploratory screening; it killed the original hypothesis and
found a real but unexplained SS-SPRING effect. Phase B exists to turn that into
evidence for the two claims the project actually needs:

- **Claim A** — SS-SPRING, untuned, matches optimally *tuned* SPRING.
- **Claim B** — SS-SPRING matches or beats PRIME-SR.

Same environment as Phase A: random init, float32, one GPU per run, eta=0.02,
wandb project `vmcnet-phase-b` (offline; sync afterwards from a login node).

## Where the claims stand

| | evidence today | problem |
|---|---|---|
| A | E0 mu grids; E6 SS-SPRING beat SPRING(0.99) on carbon by ~0.11 mHa at ~2σ | every tuned-SPRING baseline is **single seed**, and on carbon E0's top six mu values were within each other's error bars, so "tuned" was never resolved |
| B | E2/E3 put PRIME-SR slightly behind on both systems | **single seed everywhere**, and E6 had no PRIME-SR arm at all |

Both gaps are the same gap: nothing has been replicated. That is what E7 fixes.

## What runs now

| # | Script | Runs | Cost | Purpose |
|---|--------|------|------|---------|
| 1 | `slurm/e7_headtohead_seeds.sbatch` | 40 × ~4h | ~160 GPU-h | the paper's main table: 4 arms × {carbon, H4} × 5 seeds, 50k epochs + eval |
| 2 | `slurm/e8_matched_constant.sbatch` | 6 × ~2h | ~12 GPU-h | mechanism control: SPRING held at SS-SPRING's converged beta |

They are independent — submit both:

```bash
sbatch slurm/e7_headtohead_seeds.sbatch
sbatch slurm/e8_matched_constant.sbatch
```

Then, from a **login node**: `bash slurm/sync_wandb.sh`

## How E7 must be scored

**Not on the mean alone.** E6 showed the H4 margin is carried by one seed out of
three: on the other two, every arm converged to within 0.04 mHa. A mean-only
comparison would report a null and miss the actual effect, which looks like reduced
sensitivity to an unlucky initialisation. Report, per arm:

- mean ± s.e.m. across the 5 seeds (the conventional number),
- **worst seed** (the robustness number — this is the one that separated arms in E6),
- the spread across seeds,
- final energy from the eval phase with autocorrelation-corrected error bars
  (`vmcnet/mcmc/statistics.py` returns these), not a training tail mean.

Claim A is supported if SS-SPRING's interval overlaps the better of the two SPRING
arms on both systems. Claim B is supported if SS-SPRING is at least as good as
PRIME-SR on both, and is *strengthened* — not weakened — if the two tie on the mean
but SS-SPRING has the better worst seed, since seed-robustness is PRIME-SR's own
headline claim.

## How E8 decides the mechanism story

E6 ruled out the momentum warm-up. E8 tests the remaining simple explanation: that
SS-SPRING simply finds a better constant. It converged to beta = 0.9944 on carbon
and 0.9924 on H4 — not 0.99 — and E0's carbon optimum sat at 0.995, right next door.

- **SPRING at the matched constant ties SS-SPRING** → the mechanism is "it finds the
  right constant with no tuning". That *is* Claim A, and it makes the method simple
  to explain.
- **SPRING at the matched constant is still worse** → beta's variation over training
  matters, and the adaptivity does something no constant can.

Both are good outcomes. The present ambiguity is the bad outcome, and it is the first
thing a reviewer will probe.

## Gated on E7/E8 — do not submit yet

- **Third system** (N or O atom, ~carbon cost) — "matches tuned SPRING" is much
  stronger on three systems than two. Needs its own mu grid for the tuned baseline.
- **PRIME-SR home turf** (N2-eq, CO) — their seed-robustness result is on those
  systems at mu ∈ {0.9, 0.95}. Claiming B without touching them invites the obvious
  objection. Expensive (~15–20 h/run), so only worth it once E7 shows a defensible
  ordering.
- **Norm-constraint sweep** — since the cap binds essentially always, C is the real
  step size and eta never was (E2). A robustness result in C is legitimate but does
  not serve A or B directly.

## Reference energies and reporting

`preset_configs/reference_energies.json`. Carbon −37.8450 Ha (Chakravorty 1993). H4
square 4 Bohr has no literature benchmark — the in-repo value is our own FCI/CBS
estimate, so quote H4 as raw energies and compare arms to each other, never to an
absolute error. Always use `energy_noclip`; the clipped energy is biased and its bias
depends on the walker count.

Reports go in `vmcnet-phase-b` via the `wandb-experiment-report` skill: description →
takeaways → native panels, fluid width, and a **Follows from** line naming the parent
experiment and quoting the number that motivated the run.
