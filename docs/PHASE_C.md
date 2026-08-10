# Phase C runbook: hardening the two claims

Phase B (E6–E8) established both claims on two systems. E7 is the main table:

- **Claim A** — SS-SPRING, untuned, ties the best fixed mu. Carbon +0.148 ± 0.004 vs
  SPRING(0.99) +0.147 ± 0.012 mHa; H4 −2.03420 vs −2.03418.
- **Claim B** — SS-SPRING beats PRIME-SR. Carbon by 0.069 ± 0.009 mHa (~7σ, behind on
  every seed); H4 by 0.172 ± 0.050 mHa.
- Bonus: E0's single-seed 25k grid **mis-picked** mu (its H4 choice, 0.8, is the worst
  arm at 50k), so SS-SPRING beats the mu you would realistically have *chosen*.

Phase C does not chase new effects. It closes the three objections a referee will
raise against those numbers, in descending order of how cheaply they can be closed.

| # | Script | Runs | Cost | Objection it closes |
|---|--------|------|------|--------------------|
| 1 | `slurm/e9_atoms_headtohead.sbatch` | 50 × ~4–5h | ~225 GPU-h | "only two systems" |
| 2 | `slurm/e11_eta_robustness_carbon.sbatch` | 24 × ~4h | ~96 GPU-h | "only one learning rate" |
| 3 | `slurm/e10_molecules_hometurf.sbatch` | 24 × ~10–14h | ~250–330 GPU-h | "you avoided PRIME-SR's systems" |

**Recommended order: E9 and E11 together, E10 only afterwards.** E9 and E11 are
independent and together cost about what E7 did. E10 costs more than everything else
in the campaign combined, so it should not start until E9 has confirmed the result
generalises — if it does not, E10's design changes.

```bash
sbatch slurm/e9_atoms_headtohead.sbatch
sbatch slurm/e11_eta_robustness_carbon.sbatch
```

Then, from a **login node**: `bash slurm/sync_wandb.sh`. New project:
`vmcnet-phase-c`.

## E9 — N and O atoms (generality)

The SPRING paper's own systems, so the numbers are directly comparable to published
values, at roughly carbon's cost. Five arms: SPRING at mu ∈ {0.95, 0.99, 0.995},
PRIME-SR, SS-SPRING; 5 seeds; eta=0.02; 50k + eval.

**The mu grid is run at full length rather than imported.** E7 showed a cheap
single-seed screen picks the wrong mu, so the tuned baseline here is the best of three
mu values run under the same conditions as the arms they are compared against. E7
found 0.99 best on both prior systems, with 0.95/0.995 bracketing it.

To trim to 3 seeds (30 runs, ~135 GPU-h), submit only seeds 0–2:
`--array=0-2,5-7,10-12,15-17,20-22,25-27,30-32,35-37,40-42,45-47`

## E11 — learning-rate robustness (carbon)

Every number behind the claims was taken at eta=0.02. This repeats E7's carbon
head-to-head at 0.005 and 0.05 (4 arms × 2 eta × 3 seeds).

**This is not a repeat of E2.** E2 found carbon energies nearly flat in eta because,
with the norm constraint on, the realized step is `min(η‖φ‖, √C)` and eta cancels once
the cap binds — a real finding, but it means E2 cannot answer this question. E11 runs
50k with an eval phase and asks whether the *ranking* holds, not whether the energy
moves. Read `norm_cap_applied` alongside the energies to know which regime each arm is
in. A ranking that flips with eta would undercut both claims.

## E10 — N2-eq and CO (PRIME-SR's home turf)

Claim B's remaining exposure. The PRIME-SR paper's headline electronic result is
seed-robustness on exactly these two systems, so testing there — **at their settings**
— is what makes the claim defensible rather than convenient.

Deliberately handicapped to their configuration: **eta = 0.002** (the tuned molecular
value in both papers, not the 0.02 used elsewhere here) and fixed mu ∈ {0.9, 0.95},
their reported grid. mu = 0.99 is excluded because they report it unstable on N2/CO
and we have no data of our own there. 3 seeds, not 5, purely for cost.

Watch the `mu` trace: SS-SPRING's beta converges near 0.99, so if it is stable on N2
where fixed mu = 0.99 reportedly is not, that is a result in its own right.

**Run N2 first** (12 runs, half the cost):
`sbatch --array=0-2,6-8,12-14,18-20 slurm/e10_molecules_hometurf.sbatch`
then CO with `--array=3-5,9-11,15-17,21-23`.

The ~10–14h/run estimate extrapolates from a single 200-epoch smoke test that put N2
at roughly 3× carbon per step. Re-time after the first run finishes rather than
trusting it.

## New presets

`preset_configs/{N,O,CO}.json`, mirroring `carbon.json`. Spin sectors are the atomic
ground states — N ⁴S (5,2), O ³P (5,3), CO closed-shell singlet (7,7) at 2.173 Bohr —
all charge-neutral and smoke-tested. Benchmarks already in
`preset_configs/reference_energies.json`: N −54.5892, O −75.0673, CO −113.3255,
N2-eq −109.5423 Ha.

## Scoring (unchanged from E7 — do not revert to mean-only)

Per arm: mean ± s.e.m. across seeds, **worst seed**, and the spread. Final energies
come from the eval phase, not a training tail. Note that vmcnet's eval phase restarts
its step counter, so **wandb drops it** — eval energies must be parsed from the slurm
logs (their lines lack the noclip parenthetical) and backfilled as `x_eval_energy`
before reporting.
