#!/bin/bash
# E6 — momentum warm-up ablation.   FOLLOWS FROM: E0 + E3 (Phase A).
#
# WHY THIS RUNS. Phase A found that both adaptive methods collapse to a nearly
# constant momentum (SS-SPRING beta ~0.993, PRIME-SR mu_k ~0.95), so they were
# expected to behave like fixed-mu SPRING. On H4 at N=1000 SS-SPRING nevertheless
# BEAT every fixed mu -- E=-2.03385 with the lowest variance, vs -2.03339 for the
# best tuned fixed mu (0.8, from E0) and -2.03284 for mu=0.99 -- while sitting at
# beta=0.9928, i.e. essentially SPRING's own 0.99. A better constant cannot explain
# that, so the trajectory must be doing the work.
#
# The candidate mechanism: SS-SPRING's beta is pinned at its initial value (0) until
# the probe residual buffer fills at 2*lb_window = 60 steps, then jumps to ~0.95+
# and creeps to 0.99 by ~500. That is an automatic momentum WARM-UP through the
# chaotic random-init phase -- something fixed-mu SPRING never gets.
#
# THE TEST. Four arms crossing "has a warm-up" with "is adaptive":
#   0  SPRING mu=0.99, no warm-up          (control: what Phase A ran)
#   1  SPRING mu=0.99, warm-up 60 steps    (warm-up WITHOUT adaptivity)
#   2  SS-SPRING defaults (beta starts 0)  (warm-up WITH adaptivity)
#   3  SS-SPRING beta init 0.99            (adaptivity WITHOUT warm-up)
# If arm 1 reproduces arm 2's advantage, the win is the warm-up and can be had with
# a two-line schedule. If arm 3 loses to arm 2 but arm 1 does not match it either,
# the win is in beta's live variation. If arms 0-3 tie, the H4 result was noise.
#
# 4 arms x {H4, carbon} x 3 seeds = 24 runs, ~2h each (~45 GPU-h).
# Array index = arm*6 + system*3 + seed.
#
# Submit from the vmcnet repo root:  sbatch slurm/e6_warmup_ablation.sbatch
# Then sync results to wandb from a LOGIN node:  bash slurm/sync_wandb.sh
#
#SBATCH --job-name=e6-warmup
#SBATCH --account=co_esmath
#SBATCH --partition=savio3_gpu
#SBATCH --qos=esmath_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --array=0-23
#SBATCH --output=slurm-%x-%A_%a.out

set -eo pipefail
export PATH="/global/scratch/users/$USER/envs/vmcnet/bin:$PATH"
set -u

# Offline: compute nodes have no internet. slurm/sync_wandb.sh uploads afterwards.
export WANDB_MODE=offline
export WANDB_DIR="/global/scratch/users/$USER/wandb"
mkdir -p "$WANDB_DIR"

cd "$SLURM_SUBMIT_DIR"
LOGROOT="/global/scratch/users/$USER/vmcnet_logs/phase_b/e6_warmup_ablation"
mkdir -p "$LOGROOT"

SSU=same_sampled_spring_unified
ARM=$((SLURM_ARRAY_TASK_ID / 6))
SYS_I=$(((SLURM_ARRAY_TASK_ID % 6) / 3))
SEED=$((SLURM_ARRAY_TASK_ID % 3))
PRESETS=(H4 carbon)
PRESET=${PRESETS[$SYS_I]}

case "$ARM" in
  0) TAG=spring_nowarm;   OPT=spring
     ARM_FLAGS=(--config.vmc.optimizer.spring.mu=0.99) ;;
  1) TAG=spring_warm60;   OPT=spring
     ARM_FLAGS=(--config.vmc.optimizer.spring.mu=0.99
                --config.vmc.optimizer.spring.mu_init=0.0
                --config.vmc.optimizer.spring.mu_warmup_steps=60) ;;
  2) TAG=ssu_defaults;    OPT=$SSU
     ARM_FLAGS=(--config.vmc.optimizer."$SSU".mu=0.0) ;;
  3) TAG=ssu_nowarm;      OPT=$SSU
     ARM_FLAGS=(--config.vmc.optimizer."$SSU".mu=0.99) ;;
esac

RUN_NAME="e6_${PRESET}_${TAG}_s${SEED}"

# Lineage, so the run carries its own provenance into wandb.
export WANDB_TAGS="e6,phase-b,warmup-ablation,follows-e0,follows-e3,arm-${TAG}"
export WANDB_NOTES="E6 momentum warm-up ablation. Follows E0 (mu grid: H4 optimum \
mu=0.8-0.9, carbon 0.95-0.999) and E3 (SS-SPRING beat every fixed mu on H4 at \
N=1000 while sitting at beta=0.9928). Tests whether SS-SPRING's advantage is its \
delayed momentum onset (beta=0 until 2*lb_window=60 steps) rather than its \
adaptivity. Arm: ${TAG}."

if [[ -e "$LOGROOT/$RUN_NAME" ]]; then
  echo "ERROR: $LOGROOT/$RUN_NAME already exists; move or delete it first." >&2
  exit 1
fi

python -c "import jax; print('jax devices:', jax.devices())"
echo "E6 arm=$TAG preset=$PRESET seed=$SEED  $(date -Is)"

vmc-molecule \
  --presets.path="preset_configs/${PRESET}.json" \
  --config.logdir="$LOGROOT" \
  --config.save_to_current_datetime_subfolder=False \
  --config.subfolder_name="$RUN_NAME" \
  --config.vmc.optimizer_type="$OPT" \
  --config.vmc.optimizer."$OPT".learning_rate=0.02 \
  "${ARM_FLAGS[@]}" \
  --config.vmc.nepochs=25000 \
  --config.vmc.checkpoint_every=5000 \
  --config.vmc.check_for_nans=True \
  --config.eval.nepochs=0 \
  --config.eval.nburn=0 \
  --config.initial_seed="$SEED" \
  --config.wandb.mode=offline \
  --config.wandb.project=vmcnet-phase-b \
  --config.wandb.group="e6-warmup-${PRESET}" \
  --config.wandb.name="$RUN_NAME"

echo "E6 done $(date -Is)"
