#!/bin/bash
# E7 — H4 head-to-head, replicated over seeds.   FOLLOWS FROM: E0 + E3 (Phase A).
#
# WHY THIS RUNS. E3 produced the only clear win for SS-SPRING in Phase A: on H4 at
# N=1000 it reached E=-2.03385 with the lowest variance (2.1e-4), ahead of PRIME-SR
# (-2.03349), the best tuned fixed mu from E0 (mu=0.8, -2.03339) and mu=0.99
# (-2.03284). Two reasons that result cannot be claimed yet:
#   * it is a SINGLE seed, and the margin over tuned SPRING is only ~0.46 mHa;
#   * those 25k-epoch H4 runs were still drifting ~0.5 mHa over their last 6k
#     epochs, so they were not converged.
# E7 fixes both: 5 seeds and 50k epochs. It is the replication gate — no H4 claim
# ships until the ordering survives here.
#
# THE ARMS (all at eta=0.02, which E0 showed beats 0.002 on H4 by ~11 mHa):
#   0  SPRING mu=0.8    -- best fixed mu on H4 per the E0 grid ("tuned SPRING")
#   1  SPRING mu=0.99   -- the SPRING paper default, and E3's arm
#   2  PRIME-SR
#   3  SS-SPRING defaults
# An eval phase is enabled here (unlike the Phase A screens) so final energies come
# with proper autocorrelation-corrected error bars rather than a training tail mean.
#
# 4 arms x 5 seeds = 20 runs, ~4h each at 50k epochs (~80 GPU-h).
# Array index = arm*5 + seed.
#
# GATE: run slurm/e6_warmup_ablation.sbatch FIRST. If E6 shows the four arms tie,
# the H4 effect was noise and this replication is not worth 80 GPU-hours.
#
# Submit from the vmcnet repo root:  sbatch slurm/e7_h4_seeds.sbatch
# Then sync results to wandb from a LOGIN node:  bash slurm/sync_wandb.sh
#
#SBATCH --job-name=e7-h4-seeds
#SBATCH --account=co_esmath
#SBATCH --partition=savio3_gpu
#SBATCH --qos=esmath_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --cpus-per-task=2
#SBATCH --time=20:00:00
#SBATCH --array=0-19
#SBATCH --output=slurm-%x-%A_%a.out

set -eo pipefail
export PATH="/global/scratch/users/$USER/envs/vmcnet/bin:$PATH"
set -u

export WANDB_MODE=offline
export WANDB_DIR="/global/scratch/users/$USER/wandb"
mkdir -p "$WANDB_DIR"

cd "$SLURM_SUBMIT_DIR"
LOGROOT="/global/scratch/users/$USER/vmcnet_logs/phase_b/e7_h4_seeds"
mkdir -p "$LOGROOT"

SSU=same_sampled_spring_unified
ARM=$((SLURM_ARRAY_TASK_ID / 5))
SEED=$((SLURM_ARRAY_TASK_ID % 5))

case "$ARM" in
  0) TAG=spring_mu0.8;  OPT=spring
     ARM_FLAGS=(--config.vmc.optimizer.spring.mu=0.8) ;;
  1) TAG=spring_mu0.99; OPT=spring
     ARM_FLAGS=(--config.vmc.optimizer.spring.mu=0.99) ;;
  2) TAG=prime_sr;      OPT=prime_sr
     ARM_FLAGS=() ;;
  3) TAG=ssu_defaults;  OPT=$SSU
     ARM_FLAGS=(--config.vmc.optimizer."$SSU".mu=0.0) ;;
esac

RUN_NAME="e7_H4_${TAG}_s${SEED}"

export WANDB_TAGS="e7,phase-b,h4-head-to-head,replication,follows-e0,follows-e3,arm-${TAG}"
export WANDB_NOTES="E7 H4 head-to-head, 5 seeds at 50k epochs. Replication gate for \
the single-seed E3 result where SS-SPRING (E=-2.03385) beat PRIME-SR (-2.03349), \
tuned SPRING mu=0.8 from E0 (-2.03339) and mu=0.99 (-2.03284) at N=1000. E3's runs \
were still drifting ~0.5 mHa at 25k, hence 50k here plus an eval phase for \
autocorrelation-corrected error bars. Arm: ${TAG}."

if [[ -e "$LOGROOT/$RUN_NAME" ]]; then
  echo "ERROR: $LOGROOT/$RUN_NAME already exists; move or delete it first." >&2
  exit 1
fi

python -c "import jax; print('jax devices:', jax.devices())"
echo "E7 arm=$TAG seed=$SEED  $(date -Is)"

vmc-molecule \
  --presets.path=preset_configs/H4.json \
  --config.logdir="$LOGROOT" \
  --config.save_to_current_datetime_subfolder=False \
  --config.subfolder_name="$RUN_NAME" \
  --config.vmc.optimizer_type="$OPT" \
  --config.vmc.optimizer."$OPT".learning_rate=0.02 \
  "${ARM_FLAGS[@]}" \
  --config.vmc.nepochs=50000 \
  --config.vmc.checkpoint_every=10000 \
  --config.vmc.check_for_nans=True \
  --config.eval.nchains=2000 \
  --config.eval.nburn=5000 \
  --config.eval.nepochs=20000 \
  --config.initial_seed="$SEED" \
  --config.wandb.mode=offline \
  --config.wandb.project=vmcnet-phase-b \
  --config.wandb.group=e7-h4-seeds \
  --config.wandb.name="$RUN_NAME"

echo "E7 done $(date -Is)"
