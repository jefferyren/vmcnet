#!/bin/bash
# E7 — the definitive seeded head-to-head.   FOLLOWS FROM: E0, E3, E6.
#
# THIS IS THE PAPER'S MAIN TABLE. Everything else in the campaign feeds it.
#
# WHAT IT HAS TO ESTABLISH (the two claims the project exists to support):
#   A. SS-SPRING, untuned, matches optimally TUNED SPRING.
#   B. SS-SPRING matches or beats PRIME-SR.
# Neither is currently supportable. Every tuned-SPRING baseline (E0) and every
# PRIME-SR number (E2, E3) in the campaign is SINGLE SEED. E6 gave SS-SPRING three
# seeds but had no PRIME-SR arm at all. This run puts all four arms on the same
# footing on both systems at 5 seeds.
#
# WHY 5 SEEDS AND WHY THE STATS CHANGE. E6 showed the H4 margin is carried almost
# entirely by seed 0: control seeds were -2.03255 / -2.03418 / -2.03418 against
# SS-SPRING's -2.03375 / -2.03420 / -2.03422, i.e. on two of three seeds every arm
# agreed to within 0.04 mHa. So the effect is not a shift in the mean, it is
# reduced sensitivity to an unlucky initialisation -- and a mean-only comparison
# would report a null. Report WORST SEED and the spread across seeds alongside the
# mean; 5 seeds (PRNGKey 0-4) also matches the PRIME-SR paper's own protocol, which
# is the standard those authors will hold us to.
#
# WHY 50k EPOCHS. E3/E6's H4 runs were still drifting ~0.5 mHa over their last 6k
# epochs at 25k, so 25k cannot produce a final number. An eval phase (2000 walkers)
# runs afterwards for autocorrelation-corrected error bars.
#
# THE ARMS (eta=0.02 throughout; E0 showed 0.02 beats 0.002 on H4 by ~11 mHa):
#   0  SPRING mu=0.99          -- the SPRING paper default
#   1  SPRING mu=<tuned>       -- best of the E0 grid: 0.995 carbon, 0.8 H4
#   2  PRIME-SR                -- claim B's comparator, seeded for the first time
#   3  SS-SPRING defaults      -- untuned
# Arms 0 and 1 together ALSO give a properly seeded tuned-SPRING baseline, which the
# single-seed E0 grid could not: on carbon E0's top six mu values were within each
# other's error bars, so which one is "tuned" was never actually resolved.
#
# 4 arms x {carbon, H4} x 5 seeds = 40 runs, ~4h each (~160 GPU-h).
# Array index = arm*10 + system*5 + seed.
#
# Submit from the vmcnet repo root:  sbatch slurm/e7_headtohead_seeds.sbatch
# Then sync from a LOGIN node:       bash slurm/sync_wandb.sh
#
#SBATCH --job-name=e7-headtohead
#SBATCH --account=co_esmath
#SBATCH --partition=savio3_gpu
#SBATCH --qos=esmath_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --cpus-per-task=2
#SBATCH --time=20:00:00
#SBATCH --array=0-39
#SBATCH --output=slurm-%x-%A_%a.out

set -eo pipefail
export PATH="/global/scratch/users/$USER/envs/vmcnet/bin:$PATH"
set -u

export WANDB_MODE=offline
export WANDB_DIR="/global/scratch/users/$USER/wandb"
mkdir -p "$WANDB_DIR"

cd "$SLURM_SUBMIT_DIR"
LOGROOT="/global/scratch/users/$USER/vmcnet_logs/phase_b/e7_headtohead"
mkdir -p "$LOGROOT"

SSU=same_sampled_spring_unified
ARM=$((SLURM_ARRAY_TASK_ID / 10))
SYS_I=$(((SLURM_ARRAY_TASK_ID % 10) / 5))
SEED=$((SLURM_ARRAY_TASK_ID % 5))
PRESETS=(carbon H4)
PRESET=${PRESETS[$SYS_I]}

# Best mu from the E0 grid, per system. They differ, which is the whole reason an
# adaptive rule is worth having: no single constant is optimal on both.
TUNED_MUS=(0.995 0.8)
TUNED_MU=${TUNED_MUS[$SYS_I]}

case "$ARM" in
  0) TAG=spring_default;  OPT=spring
     ARM_FLAGS=(--config.vmc.optimizer.spring.mu=0.99) ;;
  1) TAG=spring_tuned;    OPT=spring
     ARM_FLAGS=(--config.vmc.optimizer.spring.mu="$TUNED_MU") ;;
  2) TAG=prime_sr;        OPT=prime_sr
     ARM_FLAGS=() ;;
  3) TAG=ssu_defaults;    OPT=$SSU
     ARM_FLAGS=(--config.vmc.optimizer."$SSU".mu=0.0) ;;
esac

RUN_NAME="e7_${PRESET}_${TAG}_s${SEED}"

export WANDB_TAGS="e7,phase-b,head-to-head,seeded,follows-e0,follows-e3,follows-e6,arm-${TAG},sys-${PRESET}"
export WANDB_NOTES="E7 definitive head-to-head, 5 seeds x 50k epochs + eval. Supplies \
the two claims the campaign needs: SS-SPRING (untuned) vs optimally tuned SPRING, and \
vs PRIME-SR -- both of which were single-seed everywhere before this (E0 tuned mu, \
E2/E3 PRIME-SR). E6 showed the H4 effect is seed-driven (two of three seeds identical \
across arms), so score worst-seed and spread, not just the mean. Arm: ${TAG} on \
${PRESET}, tuned mu for this system = ${TUNED_MU}."

if [[ -e "$LOGROOT/$RUN_NAME" ]]; then
  echo "ERROR: $LOGROOT/$RUN_NAME already exists; move or delete it first." >&2
  exit 1
fi

python -c "import jax; print('jax devices:', jax.devices())"
echo "E7 arm=$TAG preset=$PRESET seed=$SEED tuned_mu=$TUNED_MU  $(date -Is)"

vmc-molecule \
  --presets.path="preset_configs/${PRESET}.json" \
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
  --config.wandb.group="e7-headtohead-${PRESET}" \
  --config.wandb.name="$RUN_NAME"

echo "E7 done $(date -Is)"
