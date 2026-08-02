# Phase A runbook: SPRING vs same-sampled SPRING vs PRIME-SR screens

All runs start **from random init** (no KFAC prelim) — this matches the PRIME-SR
paper's electronic protocol, makes `--config.initial_seed` genuinely vary runs, and
makes `nchains` overrides safe (a reloaded checkpoint would pin the RNG key and the
walker count). Environment is pinned: float32, exactly 1 GPU per run (the spectral
indicators and the probe are per-device-batch quantities), wandb project
`vmcnet-phase-a` (offline; `wandb sync` from a login node afterwards).

## Submission order

| # | Script | Runs | Est. cost | Purpose |
|---|--------|------|-----------|---------|
| 1 | `slurm/phase_a_smoke.sbatch` | 5 × ~0.5h | ~3 GPU-h | nchains=2000 OOM check; N2-4.0 and H4 per-step timing |
| 2 | `slurm/e0_mu_grid_carbon.sbatch` | 10 × ~2h | ~20 GPU-h | tuned-SPRING benchmark, carbon stage 1 |
| 3 | `slurm/e0_mu_grid_h4.sbatch` | 20 × ~1.5h | ~30 GPU-h | tuned-SPRING benchmark + eta check, H4 |
| 4 | `slurm/e2_eta_sweep_carbon.sbatch` | 30 × ~2h | ~55 GPU-h | eta-robustness mechanism (constraint OFF arm) + practical arm (ON) |
| 5 | `slurm/e3_nchains_sweep.sbatch` | 24 × ≤2h | ~30 GPU-h | small-N mechanism (PINN-effect reproduction) |
| 6 | `slurm/e4_h4_stress_grid.sbatch` | 18 × ~1.5h | ~25 GPU-h | (eta × nchains) stability phase diagram, constraint OFF |
| 7 | `slurm/e5_ssu_knob_screen.sbatch` | 7 × ~2.5h | ~18 GPU-h | SS-SPRING knob robustness (protects the "untuned" claim) |

Cost estimates assume carbon 25k ≈ 1.5–2.5h from the measured 6.5–7h per 100k+20k run;
**recalibrate after the smoke tests** (the 200-epoch 1.16×/1.32× ratios in the old
sbatch comments are jit-warmup-dominated and contradict the full-run reports —
re-derive per-step cost from smoke `.out` timestamps).

Scripts 2–7 are independent; submit them all once the smoke passes. If the queue is
tight, priority is 4 > 5 > 2 > 3 > 6 > 7 (the mechanism kill-switches come first).

`sbatch --export=ALL,SEED=1 <script>` adds a second seed to scripts 2–7 (run names
include `_s{SEED}`, so no collisions; the smoke script ignores SEED). After E0
picks the best mu, add the tuned-SPRING arm by resubmitting ONLY the spring tasks
with `--export=ALL,SPRING_MU=<best>`:
`sbatch --array=0,1,6,7,12,13,18,19,24,25 --export=ALL,SPRING_MU=<best> slurm/e2_eta_sweep_carbon.sbatch`
and
`sbatch --array=0,1,6,7,12,13,18,19 --export=ALL,SPRING_MU=<best> slurm/e3_nchains_sweep.sbatch`
(resubmitting the full array is safe but the non-spring tasks exit on the logdir
guard as a wall of FAILED). If E0's best mu is exactly 0.99, the tuned arm IS the
default arm — no rerun needed.

## Divergence bookkeeping

Every sweep run sets `check_for_nans=True`: a NaN in metrics or params saves a
checkpoint and aborts the loop. Classify a run as **diverged** if it NaN-aborted
OR its mean training energy over the final 3000 epochs exceeds the all-methods
median at that setting by > 50 mHa (tune this threshold once data exists; state it
in the paper). NaN aborts are visible as truncated `energy.txt` files.

## What each experiment must show (kill-switch gates for Phase B/C)

- **E2 (eta sweep)** — from `mu.txt` per run: PRIME-SR's mu_k(t) curves must be
  statistically indistinguishable across eta in the constraint-OFF arm
  (pre-divergence), while SS-SPRING's beta(t) separates by eta with the predicted
  trend (1−β) ≈ 4η·m_eff. From `norm_cap_applied.txt`: report fraction-at-cap for
  every ON-arm run (while capped, spring/prime_sr trajectories are eta-invariant by
  construction — do not read the ON arm as a mechanism result). Watch
  `probe_r_ip.txt` > 1 sustained: that is the precursor of SS-SPRING's own
  beta→1 lock; if SS-SPRING also NaNs at eta 0.1–0.2 constraint-off, the claim is
  "extends the stable-eta window", not "unconditionally robust".
- **E3 (nchains sweep)** — PRIME-SR's mean mu_k must rise as N shrinks (pure-noise
  floor: mu = 1−(1−(alpha/rank)^¼)² ≈ 0.64 → 0.92 as N 1000 → 100 at alpha≈25;
  the existing local LiH run at N=100 already sits at mu_k ≈ 0.976). Energy gap:
  PRIME-SR should degrade faster than SPRING(0.99)/SS-SPRING as N shrinks.
- **E0 (mu grids)** — the SPRING mu-sensitivity band must span a resolvable energy
  margin (else "tuned SPRING" is not a meaningful benchmark and H4/carbon do not
  discriminate — reweight toward N2-4.0). Ranking criterion: mean training energy
  over the final 3000 epochs (PRIME-SR paper smoothing window), seed 0 (+1 for
  the top candidates).
- **E5 (knob screen)** — all deviations should land within noise of defaults;
  freeze the default knob set BEFORE any Phase B comparison run and never hand-set
  `probe_lr` (the −1 sentinel → base eta coupling is the mechanism).

**Proceed to Phase B/C (full 100k runs, 5 seeds, N2-eq home-turf + N2-4.0 money
runs) only if the E2 and E3 mechanism gates pass.**

## Reference energies

`preset_configs/reference_energies.json`. Carbon −37.8450 Ha (Chakravorty 1993),
N2-4.0 −109.2021 Ha (Le Roy MLR — spectroscopically derived; mHa-level offset vs
clamped-nuclei non-relativistic, fine for optimizer comparisons). H4 square 4 Bohr
has no literature benchmark; the in-repo value is our FCI/CBS estimate — treat
error-vs-reference plots for H4 as approximate and prefer relative-energy
comparisons between optimizers (all arms are variational on the identical system).

## Analysis inputs per run

Text metrics in each logdir: `energy.txt`, `variance.txt` (+noclip variants),
`update_sq_norm_preclip.txt`, `norm_cap_applied.txt`; prime_sr adds `mu.txt`,
`alpha.txt`, `rank.txt`, `beta_tilde.txt`; ssu adds `mu.txt` (=beta), `r_hat.txt`,
`probe_res_norm.txt`, `probe_r_ip.txt`. Wandb mirrors these (10k-point cap;
`slurm/backfill_wandb.py` replays full-resolution afterwards).
