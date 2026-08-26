# Results

## 1. Protocol

All three optimizers are stochastic-reconfiguration methods for neural-network VMC that
differ *only* in how the momentum applied to the previous update is chosen. SPRING uses a
fixed, hand-tuned μ. PRIME-SR recomputes μ_k each step from the sampled Gram-matrix
spectrum. SS-SPRING derives β from a same-sampled probe — a second SPRING solve on the
same sampled operators against a fixed synthetic target, whose measured residual
contraction sets the momentum. All other components (network, sampler, gradient
estimator, norm constraint) are identical.

Every run starts from **random initialization**, matching the PRIME-SR paper's electronic
protocol; there is no preliminary KFAC phase. Runs are float32 on a single GPU, since the
spectral indicators and the probe are per-device-batch quantities. Unless stated, training
is **50 000 epochs followed by a 20 000-epoch evaluation phase** at 1000–2000 walkers.

Three conventions matter for reading the numbers:

- **Energies are the mean over the full evaluation phase, never a training tail.** The
  training energy is clipped, and the clipping bias depends on the walker count, so we
  report the unclipped evaluation energy throughout.
- **Differences are paired by seed.** We write Δ = E(SS-SPRING) − E(comparator), so
  **negative Δ means SS-SPRING reached the lower energy**. The quoted uncertainty is the
  standard error of the per-seed differences, which cancels the seed-to-seed scatter that
  dominates the raw arm means. We report the **seed sweep** (e.g. 5/5) alongside σ,
  because with 3–5 seeds the σ alone is a thin estimate.
- **Arms are scored on mean, worst seed and spread**, not the mean alone.

Reference energies: C −37.8450, N −54.5892, O −75.0673 Ha (Chakravorty *et al.* 1993);
N₂ at 2.016 a₀ −109.5423 Ha (Filippi & Umrigar 1996); CO at 2.173 a₀ −113.3255 Ha (Pfau
*et al.* 2020). H₄ has no literature benchmark, so H₄ results are quoted as raw energies
and compared only between arms.

## 2. Summary across all benchmarked conditions

Seven conditions were run under the full 50k + evaluation protocol: five systems at the
tuned learning rate (C, H₄, N, O at η = 0.02; N₂ and CO at η = 0.002) and carbon at
η = 0.005 and 0.05. Table 1 gives the evaluation energy of each arm, in mHa above the
reference (lower is better).

**Table 1.** Evaluation energy, mHa above reference. Bold = best in that condition.

| condition | η | seeds | SS-SPRING | SPRING(0.995) | SPRING(0.99) | PRIME-SR |
|---|---|---|---|---|---|---|
| C | 0.02 | 5 | 0.148 | 0.161 | **0.147** | 0.217 |
| N | 0.02 | 5 | 0.221 | **0.197** | 0.214 | 0.260 |
| O | 0.02 | 5 | 0.783 | 0.630 | **0.604** | 0.795 |
| C | 0.005 | 3 | 0.172 | **0.162** | 0.201 | 0.435 |
| C | 0.05 | 3 | 0.192 | 0.191 | **0.155** | 0.193 |
| N₂ | 0.002 | 3 | **11.410** | 11.668 | 11.818 | 14.260 |
| CO | 0.002 | 3 | **9.157** | 9.210 | 9.607 | 12.063 |

Two summaries follow, and they answer different questions.

**Against per-condition oracle tuning.** If a practitioner could tune μ separately for
every system and learning rate and always pick the best fixed value, SS-SPRING — which is
never tuned — would give up **−0.009 mHa on average** across the seven conditions: a wash.
The spread is small and two-sided: SS-SPRING is ahead by 0.257 mHa in the best case (N₂),
behind by 0.178 mHa in the worst (O), and within 0.04 mHa in four of seven conditions.

**Against a single momentum used everywhere.** No fixed μ is optimal in all seven
conditions — the best value moves between 0.99 and 0.995 with both system and learning
rate. So the operationally relevant question is what a practitioner loses by committing to
one choice. Table 2 gives the regret of each arm relative to the best arm in each
condition.

**Table 2.** Regret relative to the best arm in each condition (mHa), over all seven
conditions.

| arm | mean regret | worst-case regret |
|---|---|---|
| **SS-SPRING** | **0.036** | **0.178** |
| SPRING(0.995) | 0.055 | 0.257 |
| SPRING(0.99) | 0.131 | 0.450 |
| PRIME-SR | 0.913 | 2.906 |

SS-SPRING has both the lowest mean regret and the lowest worst-case regret of any arm.
This is the central practical result: **an untuned SS-SPRING is a better default than any
untuned fixed momentum, and matches per-condition oracle tuning to within noise on
average.**

## 3. Head-to-head at the tuned learning rate

On carbon and H₄ at η = 0.02, five seeds each (Table 3), SS-SPRING is statistically
indistinguishable from the best fixed momentum and ahead of PRIME-SR on every seed.

**Table 3.** Paired differences Δ = E(SS-SPRING) − E(comparator), mHa. Negative favours
SS-SPRING.

| system | comparator | Δ | σ | seeds won |
|---|---|---|---|---|
| C | SPRING(0.99) | +0.002 | 0.011 | 4/5 |
| C | SPRING(0.995) | −0.012 | 0.003 | 4/5 |
| C | PRIME-SR | **−0.069** | 0.011 | **5/5** |
| H₄ | SPRING(0.99) | −0.017 | 0.007 | **5/5** |
| H₄ | PRIME-SR | **−0.172** | 0.038 | **5/5** |
| H₄ | SPRING(0.8), tuned by a short screen | **−0.410** | 0.047 | **5/5** |

SS-SPRING is also the most seed-stable arm on both systems — spread 0.019 mHa on carbon
against 0.023–0.074 for the others — which is precisely the property PRIME-SR advertises
for itself.

The last row is a methodological warning worth stating explicitly. A single-seed,
25 000-epoch grid search selected μ = 0.8 for H₄. At five seeds and 50 000 epochs that is
the **worst** arm tested, 0.410 ± 0.047 mHa behind SS-SPRING. Short screens mis-rank
momentum, which strengthens rather than weakens the case for removing the tuning step.

## 4. Generality across systems

Extending to the N and O atoms at η = 0.02, five seeds (Table 1, rows 2–3), separates the
two claims.

On **nitrogen**, SS-SPRING ties the best fixed momentum (Δ = +0.024 ± 0.019 vs
SPRING(0.995), 1.3σ; Δ = +0.006 ± 0.014 vs SPRING(0.99)) and beats PRIME-SR by
0.039 ± 0.010 mHa on 5/5 seeds.

On **oxygen**, SS-SPRING is **0.178 ± 0.066 mHa behind SPRING(0.99), losing on all five
seeds** (and 0.152 ± 0.049 behind SPRING(0.995), also 0/5). Losing every seed is what
makes this a real effect rather than a noisy mean, and it is the clearest evidence in this
work against a general parity claim. Against PRIME-SR, oxygen is a tie
(Δ = −0.012 ± 0.073). SS-SPRING is also the least seed-stable arm on oxygen (spread
0.276 mHa), reversing its behaviour on carbon and H₄.

One caveat on nitrogen: the fixed-μ ordering there is monotone in μ (0.268 → 0.214 →
0.197 for 0.95 → 0.99 → 0.995), so the optimum sits at the edge of the grid and the tuned
baseline is, if anything, understated. A μ = 0.999 arm would likely narrow SS-SPRING's tie
on that system.

## 5. Learning-rate robustness

All of the above was measured at a single learning rate. Repeating the carbon comparison
at η = 0.005 and η = 0.05 (three seeds) shows that **the ranking does not survive a change
of learning rate — and the arm that breaks is PRIME-SR** (Table 4).

**Table 4.** Carbon evaluation energy (mHa) across a 10× learning-rate span.

| arm | η = 0.005 | η = 0.02 | η = 0.05 | swing |
|---|---|---|---|---|
| SPRING(0.995) | 0.162 | 0.161 | 0.191 | 0.030 |
| SS-SPRING | 0.172 | 0.148 | 0.192 | 0.044 |
| SPRING(0.99) | 0.201 | 0.147 | 0.155 | 0.054 |
| **PRIME-SR** | **0.435** | 0.217 | 0.193 | **0.242** |

PRIME-SR's swing is 4–8× larger than any other arm's, and monotone: the smaller the
learning rate, the worse it does. At η = 0.005 it is 2.7× worse than every other arm, and
SS-SPRING beats it by 0.264 ± 0.021 mHa on 3/3 seeds. Seed-robustness is the property the
PRIME-SR paper advertises; robustness to the learning rate is not, and that paper reports
no learning-rate study.

For SS-SPRING the picture is mixed: it ties the best fixed momentum at η = 0.005
(Δ = +0.009 ± 0.028) but is 0.037 ± 0.010 mHa behind SPRING(0.99) at η = 0.05, losing 3/3
seeds — a second condition, alongside oxygen, where the parity claim fails.

## 6. PRIME-SR's own systems, at its own settings

The strongest test of the comparative claim is on the systems where PRIME-SR reports its
headline result — N₂ at equilibrium and CO — at the learning rate both papers use for
molecules, η = 0.002 (Table 1, rows 6–7).

**SS-SPRING is the best arm on both systems.** It beats PRIME-SR by **2.849 ± 0.217 mHa**
on N₂ and **2.906 ± 0.075 mHa** on CO, ahead on every seed on both. Against the best fixed
momentum it is ahead on N₂ (Δ = −0.257 ± 0.063, 3/3 seeds) and tied on CO
(Δ = −0.053 ± 0.095, 0.6σ).

Two qualifications. First, the momentum grid matters enormously here and the naive choice
would have been misleading: restricted to the μ ∈ {0.9, 0.95} grid the PRIME-SR paper uses
for these systems, SS-SPRING would appear to beat "tuned SPRING" by 3.5 mHa on N₂ and
3.7 on CO — almost entirely an artifact of comparing μ ≈ 0.995 against μ ≤ 0.95 at a
learning rate where momentum dominates (§7). The μ = 0.99 and 0.995 arms were added for
this reason, and they, not the paper's grid, are the honest baseline. Second, we found no
instability at μ = 0.99 or 0.995 on either system, for any seed, contrary to that paper's
report.

**These two systems are not converged at 50 000 epochs.** Every arm is still improving by
0.1–1.0 mHa per 5000 epochs at the end, and absolute errors are 9–20 mHa against 0.2–0.8
mHa on the atoms. N₂ and CO therefore compare arms **at a fixed budget, not at their
asymptotes**; PRIME-SR is still falling at ≈0.3 mHa per 5000 epochs, so its deficit would
partially close given a longer run. The comparison remains meaningful — equal budget,
equal protocol — but should not be read as an asymptotic accuracy claim.

## 7. Mechanism

Two diagnostics explain the entire pattern above.

**Neither adaptive method is meaningfully adaptive in this regime.** SS-SPRING's β
converges to 0.9943–0.9960 and PRIME-SR's μ_k to 0.9498–0.9539 across every system and
learning rate tested. PRIME-SR's momentum moves by 0.001 across a 10× change in learning
rate — it is blind to the step size by construction, computing μ_k from the Gram spectrum
alone. SS-SPRING's probe contraction ratio saturates (tail values 1.02–1.03, i.e. the
probe residual is growing), which pins β near its ceiling. Both behave, in practice, as
fixed-momentum SPRING that happens to select its own constant.

**What the learning rate changes is whether the norm constraint binds.** The realized
update is min(η‖φ‖, √C), so while the cap binds, η cancels out entirely. We find the cap
binds on 37–97% of steps in the first half of a 50 000-epoch run and then on **exactly 0%**
for the entire second half. It binds on only 1.6–4.3% of steps at η = 0.005 but 34–42% at
η = 0.05.

Together these predict the ordering. In the uncapped regime the accumulated update scales
as η/(1 − μ). At η = 0.002 that gives 0.400 for SS-SPRING and SPRING(0.995), 0.200 for
SPRING(0.99), **0.043 for PRIME-SR** and 0.020 for SPRING(0.9) — PRIME-SR trains at an
effective step roughly 10× smaller than the high-momentum arms and simply has not
converged. Across the six arms on N₂ and CO, the correlation between log₁₀(η/(1 − μ)) and
the final error is **r = −0.89 on both systems**; on carbon across learning rates the same
relation holds. The arm ordering is, to a good approximation, the momentum ordering.

This also delimits an earlier finding. A learning-rate sweep at 25 000 epochs found
energies nearly flat across a 40× span and concluded that the norm constraint C, not η, is
the operative step-size knob. That is true only while the cap binds; on a 50 000-epoch
schedule the cap releases mid-run and η governs the second half. PRIME-SR's failure at
small η is a direct consequence, and it is the clean failure mode this work set out to
find: its momentum rule cannot see the quantity that determines whether its choice is
appropriate.

## 8. Controls

Two mechanisms for SS-SPRING's behaviour were tested and rejected. **Both controls were
run at 25 000 epochs with the evaluation phase disabled, so their numbers are training
tails and are not directly comparable to Tables 1–4.** They are reported here as
ablations, and §9 notes the resulting limitation.

*Momentum warm-up.* SS-SPRING's β is exactly zero for the first 60 steps and then jumps
above 0.95, which resembles an automatic warm-up. Removing it changes nothing:
−2.03406 ± 0.00015 with warm-up versus −2.03404 ± 0.00016 without on H₄, and
0.117 ± 0.049 versus 0.144 ± 0.003 mHa on carbon. Grafting the same warm-up onto fixed-μ
SPRING recovers only part of the gap (H₄ −2.03364 → −2.03389 against SS-SPRING's
−2.03406). The warm-up is not the mechanism.

*A better constant.* SPRING pinned at SS-SPRING's own measured converged β — 0.9944 on
carbon, 0.9924 on H₄ — does not reproduce it: −2.03363 ± 0.00056 on H₄ against SS-SPRING's
−2.03406, and 0.191 ± 0.066 versus 0.118 mHa on carbon. On H₄ the matched constant repeats
the same unlucky-seed failure that SPRING(0.99) shows (seed 0: −2.03251 versus SS-SPRING's
−2.03375). What remains is β's trajectory between those endpoints rather than its
endpoint value.

Separately, a screen over SS-SPRING's own hyper-parameters found all settings within
0.15–0.32 mHa of one another, supporting the claim that its defaults require no tuning.

## 9. Scope and limitations

- **Parity with tuned SPRING is not universal.** SS-SPRING is behind the best fixed
  momentum on oxygen (0.178 ± 0.066 mHa, 0/5 seeds) and on carbon at η = 0.05
  (0.037 ± 0.010, 0/3). The defensible claim is the regret statement of §2 — best default
  across conditions, parity with oracle tuning on average — not per-condition parity.
- **N₂ and CO are not converged** at the budget used, so those two conditions compare
  fixed-budget performance rather than attainable accuracy.
- **Three seeds** on N₂, CO and the learning-rate sweep; five elsewhere. Sub-0.5 mHa
  differences on the molecules are the same order as the per-run Monte-Carlo error
  (0.14–0.33 mHa there, ≤0.05 mHa on the atoms), so we rely on seed sweeps rather than σ
  for those comparisons.
- **The warm-up and matched-constant controls (§8) were run at 25 000 epochs without an
  evaluation phase.** Given that a 25 000-epoch screen mis-ranked μ on H₄ (§3) and that a
  dramatic seed-0 anomaly at 25 000 epochs proved to be an undertraining artifact that
  vanished by 50 000, these two controls should be repeated under the main protocol before
  the mechanism argument is leaned on heavily.
- **H₄ has no literature benchmark**; its results are relative comparisons only.
- All results are single-GPU float32, one network architecture, and one sampler.
