#!/bin/bash
# F6 -- the beta* guard (F5) as a METHOD: SS-SPRING + beta_star_guard + gram_center_first,
# from scratch at the full 100k protocol, on every system SS-SPRING and PRIME-SR were
# benchmarked on. Plan and reasoning: docs/CAMPAIGN_LOG.md, Section 5 "Phase F"
# (F5 RESULT, F6).
#
# WHY. The F5 checkpoint replays survived 6/6 on stretched N2 (F4d replays 6/6 failed),
# but the guard binds on ~100% of rows on EVERY system and cuts the applied momentum
# from beta ~0.997 to ~0.5 (N2-4.0, N2-eq, CO) or ~0.2-0.3 (H4, N, O, carbon). That is a
# different method, so the 5k late-training windows cannot stand in for its accuracy:
# it has to be run from step 0 like every other cell in the results table.
#
# WHAT IS IDENTICAL TO THE ORIGINAL CELLS, AND WHAT IS NOT.
#   IDENTICAL: preset, seeds, eta, schedule (inverse_time, decay 1e-4), norm constraint
#     (C=1e-3), nepochs=100000, checkpoint_every, eval phase (nchains / nburn / nepochs),
#     random init (model init precedes the optimizer, so seed s gives the same initial
#     parameters as the original seed-s cells). Only default-off config keys were added
#     to the code since Phase D, and preset_configs/ is unchanged.
#   CHANGED (the method): beta_star_guard=True and gram_center_first=True. Both were on
#     in F5; the guard also turns on the probe sanitiser. The two are not separated here:
#     this run measures the F5 method, not the guard alone.
#   CHANGED (logging only): diagnostics=True. This is the exact code path the F5 replays
#     ran, it logs the guard's diag_beta_applied / diag_beta_star / diag_momentum_cos,
#     and it gives the full diag_* forensics if a stretched-N2 seed still fails. It costs
#     one extra jvp per step, so wall time is NOT comparable to the original cells.
#
# THE CELLS, one arm, 34 runs. Each pairs with the original SS-SPRING / PRIME-SR / SPRING
# cells of the same system and seed:
#    0-7   N2_4.0  s0-7  eta 0.002  eval 1000/10000/20000   vs E14 (s0-2) + E15 (s3-7)
#    8-10  N2_eq   s0-2  eta 0.002  eval 1000/10000/20000   vs D10
#   11-13  CO      s0-2  eta 0.002  eval 1000/10000/20000   vs D10
#   14-18  carbon  s0-4  eta 0.02   eval 2000/5000/20000    vs D7
#   19-23  H4      s0-4  eta 0.02   eval 2000/5000/20000    vs D7
#   24-28  N       s0-4  eta 0.02   eval 2000/5000/20000    vs D9
#   29-33  O       s0-4  eta 0.02   eval 2000/5000/20000    vs D9
# Stretched N2 comes first so its verdict lands first. Baseline there: old SS-SPRING
# diverged on 6/8 seeds (E14 3/3, E15 3/5).
#
# WHAT WOULD CONFIRM IT. Stretched N2: 0-1 of 8 seeds diverge, with eval energies in the
# range of the old survivors (12.3 mHa) and of SPRING / PRIME-SR at E14. Stable systems:
# eval energies within the seed spread of the original SS-SPRING cells. A systematic
# loss there (most likely early in training, where momentum matters most) is the price
# of the cut momentum, and has to be weighed against the N2 stability.
#
# COST. ~0.65 s/step on a 2080 Ti for the three molecules (~18-20 h + eval each), less
# for the atoms and H4 (D7: ~7-8 h). Roughly 450 GPU-h in all; walltime set for the
# molecules.
#
# Submit from the vmcnet repo root on a Savio login node:
#     sbatch slurm/f6_guard_scratch_100k.sbatch
#     sbatch --array=0-7 slurm/f6_guard_scratch_100k.sbatch     # stretched N2 only
# Stability verdict (reads .out files, no wandb needed):
#     python slurm/divergence_check.py --experiment E14 E15 F6
# Eval energies, then backfill x_* fields into wandb (after slurm/sync_wandb.sh, with
# `wandb sync --legacy` on Savio; see CAMPAIGN_LOG section 7):
#     python slurm/parse_eval_energies.py --experiment F6 D7 D9 D10 E14 E15
#     python slurm/parse_eval_energies.py --experiment F6 --backfill
#
#SBATCH --job-name=f6-guard-100k
#SBATCH --account=co_esmath
#SBATCH --partition=savio3_gpu
#SBATCH --qos=esmath_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --cpus-per-task=2
#SBATCH --time=36:00:00
#SBATCH --array=0-33
#SBATCH --output=slurm-%x-%A_%a.out
# A command-line --exclude REPLACES this rather than merging; list both to add a node.
#SBATCH --exclude=n0135.savio3

set -eo pipefail
export PATH="/global/scratch/users/$USER/envs/vmcnet/bin:$PATH"
set -u

export WANDB_MODE=offline
export WANDB_DIR="/global/scratch/users/$USER/wandb"
mkdir -p "$WANDB_DIR"

cd "$SLURM_SUBMIT_DIR"
LOGROOT="/global/scratch/users/$USER/vmcnet_logs/phase_f/f6_guard_scratch_100k"
mkdir -p "$LOGROOT"

SSU=same_sampled_spring_unified
I=$SLURM_ARRAY_TASK_ID
# index -> preset, seed, eta, eval settings, the original experiment it pairs with.
# Mirrored by _f6_cell in slurm/parse_eval_energies.py and F6 in divergence_check.py.
if   (( I < 8 ));  then PRESET=N2_4.0; SEED=$I;        PAIR="E14/E15"
elif (( I < 11 )); then PRESET=N2_eq;  SEED=$((I - 8));  PAIR=D10
elif (( I < 14 )); then PRESET=CO;     SEED=$((I - 11)); PAIR=D10
elif (( I < 19 )); then PRESET=carbon; SEED=$((I - 14)); PAIR=D7
elif (( I < 24 )); then PRESET=H4;     SEED=$((I - 19)); PAIR=D7
elif (( I < 29 )); then PRESET=N;      SEED=$((I - 24)); PAIR=D9
elif (( I < 34 )); then PRESET=O;      SEED=$((I - 29)); PAIR=D9
else echo "ERROR: array index $I out of range 0-33" >&2; exit 1
fi
case "$PRESET" in
  N2_4.0|N2_eq|CO) ETA=0.002; EVAL_NCHAINS=1000; EVAL_NBURN=10000 ;;
  *)               ETA=0.02;  EVAL_NCHAINS=2000; EVAL_NBURN=5000 ;;
esac

TAG=ssu_guard
RUN_NAME="f6_${PRESET}_${TAG}_s${SEED}"

export WANDB_TAGS="f6,phase-f,100k,from-scratch,follows-f5,beta-star-guard,center-first,arm-${TAG},sys-${PRESET}"
export WANDB_NOTES="F6 = the F5 beta* guard run as a method from scratch at the 100k \
protocol: SS-SPRING + beta_star_guard + gram_center_first, everything else identical \
to the original ${PAIR} cell for ${PRESET} seed ${SEED}. F5 replays survived 6/6 on \
stretched N2 (F4d 6/6 failed) but the guard bound on ~100% of rows everywhere, cutting \
momentum to ~0.2-0.6, so the method changes on every system and needs full runs."

if [[ -e "$LOGROOT/$RUN_NAME" ]]; then
  echo "ERROR: $LOGROOT/$RUN_NAME already exists; move or delete it first." >&2
  exit 1
fi

python -c "import jax; print('jax devices:', jax.devices())"
echo "F6 idx=$I preset=$PRESET seed=$SEED eta=$ETA pairs-with=$PAIR nepochs=100000  $(date -Is)"

vmc-molecule \
  --presets.path="preset_configs/${PRESET}.json" \
  --config.logdir="$LOGROOT" \
  --config.save_to_current_datetime_subfolder=False \
  --config.subfolder_name="$RUN_NAME" \
  --config.vmc.optimizer_type="$SSU" \
  --config.vmc.optimizer."$SSU".learning_rate="$ETA" \
  --config.vmc.optimizer."$SSU".mu=0.0 \
  --config.vmc.optimizer."$SSU".schedule_type=inverse_time \
  --config.vmc.optimizer."$SSU".learning_decay_rate=0.0001 \
  --config.vmc.optimizer."$SSU".constrain_norm=True \
  --config.vmc.optimizer."$SSU".norm_constraint=0.001 \
  --config.vmc.optimizer."$SSU".beta_star_guard=True \
  --config.vmc.optimizer."$SSU".gram_center_first=True \
  --config.vmc.optimizer."$SSU".diagnostics=True \
  --config.vmc.optimizer."$SSU".safeguard=False \
  --config.vmc.optimizer."$SSU".carried_cap=0.0 \
  --config.vmc.nepochs=100000 \
  --config.vmc.checkpoint_every=10000 \
  --config.vmc.check_for_nans=True \
  --config.eval.nchains="$EVAL_NCHAINS" \
  --config.eval.nburn="$EVAL_NBURN" \
  --config.eval.nepochs=20000 \
  --config.initial_seed="$SEED" \
  --config.wandb.mode=offline \
  --config.wandb.project=vmcnet-phase-f \
  --config.wandb.group="f6-guard-100k-${PRESET}" \
  --config.wandb.name="$RUN_NAME"

echo "F6 done $(date -Is)"
