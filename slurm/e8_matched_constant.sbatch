#!/bin/bash
# E8 — the matched-constant control.   FOLLOWS FROM: E6 (and E0).
#
# WHY THIS RUNS, AND WHY IT IS CHEAP BUT IMPORTANT. E6 left one question open and a
# reviewer will ask it immediately: if SS-SPRING's beta converges to ~0.99, why does
# it beat SPRING at mu=0.99? E6 ruled out the warm-up (removing it changed nothing).
# The remaining candidates are (a) the small difference between beta's converged
# value and 0.99, or (b) beta's live variation over training.
#
# Measured in E6, SS-SPRING converges to beta = 0.9944 +/- 0.0002 on carbon and
# 0.9924 +/- 0.0007 on H4 -- NOT 0.99. And E0's carbon grid put its (noisy,
# single-seed) optimum at mu = 0.995, right next to 0.9944. So the simplest
# explanation is that SS-SPRING lands on the optimal constant by itself, and the
# apparent gap to "SPRING" is really a gap to a slightly mistuned SPRING.
#
# THE TEST. One arm: plain SPRING held at exactly SS-SPRING's measured converged
# beta for that system. Protocol is copied from E6 exactly -- 25k epochs, 3 seeds,
# eta=0.02, no eval -- so it drops straight into the E6 comparison with no
# re-running.
#
# HOW TO READ IT:
#   * matches SS-SPRING  -> the mechanism is "it finds the right constant, untuned".
#     That is the campaign's Claim A, cleanly demonstrated, and it makes the method
#     easy to explain rather than mysterious.
#   * still worse than SS-SPRING -> the time-variation of beta matters after all,
#     and the adaptivity is doing something a constant cannot.
# Either outcome is publishable; the current ambiguity is not.
#
# 1 arm x {carbon, H4} x 3 seeds = 6 runs, ~2h each (~12 GPU-h).
# Array index = system*3 + seed.
#
# Run this ALONGSIDE E7 -- it is small, independent, and answers a different question.
#
# Submit from the vmcnet repo root:  sbatch slurm/e8_matched_constant.sbatch
# Then sync from a LOGIN node:       bash slurm/sync_wandb.sh
#
#SBATCH --job-name=e8-matched-const
#SBATCH --account=co_esmath
#SBATCH --partition=savio3_gpu
#SBATCH --qos=esmath_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --array=0-5
#SBATCH --output=slurm-%x-%A_%a.out

set -eo pipefail
export PATH="/global/scratch/users/$USER/envs/vmcnet/bin:$PATH"
set -u

export WANDB_MODE=offline
export WANDB_DIR="/global/scratch/users/$USER/wandb"
mkdir -p "$WANDB_DIR"

cd "$SLURM_SUBMIT_DIR"
LOGROOT="/global/scratch/users/$USER/vmcnet_logs/phase_b/e8_matched_constant"
mkdir -p "$LOGROOT"

SYS_I=$((SLURM_ARRAY_TASK_ID / 3))
SEED=$((SLURM_ARRAY_TASK_ID % 3))
PRESETS=(carbon H4)
PRESET=${PRESETS[$SYS_I]}

# SS-SPRING's measured converged beta per system (E6, mean over 3 seeds).
MATCHED_MUS=(0.9944 0.9924)
MU=${MATCHED_MUS[$SYS_I]}

RUN_NAME="e8_${PRESET}_spring_matched${MU}_s${SEED}"

export WANDB_TAGS="e8,phase-b,matched-constant,mechanism,follows-e6,follows-e0,sys-${PRESET}"
export WANDB_NOTES="E8 matched-constant control. SPRING held at mu=${MU}, the beta \
SS-SPRING converged to on ${PRESET} in E6 (carbon 0.9944+/-0.0002, H4 \
0.9924+/-0.0007). Tests whether SS-SPRING's advantage over SPRING mu=0.99 is simply \
that it finds a better constant, or whether beta's time-variation matters. Protocol \
matches E6 exactly (25k, 3 seeds, eta=0.02, no eval) so it is directly comparable."

if [[ -e "$LOGROOT/$RUN_NAME" ]]; then
  echo "ERROR: $LOGROOT/$RUN_NAME already exists; move or delete it first." >&2
  exit 1
fi

python -c "import jax; print('jax devices:', jax.devices())"
echo "E8 preset=$PRESET matched mu=$MU seed=$SEED  $(date -Is)"

vmc-molecule \
  --presets.path="preset_configs/${PRESET}.json" \
  --config.logdir="$LOGROOT" \
  --config.save_to_current_datetime_subfolder=False \
  --config.subfolder_name="$RUN_NAME" \
  --config.vmc.optimizer_type=spring \
  --config.vmc.optimizer.spring.learning_rate=0.02 \
  --config.vmc.optimizer.spring.mu="$MU" \
  --config.vmc.nepochs=25000 \
  --config.vmc.checkpoint_every=5000 \
  --config.vmc.check_for_nans=True \
  --config.eval.nepochs=0 \
  --config.eval.nburn=0 \
  --config.initial_seed="$SEED" \
  --config.wandb.mode=offline \
  --config.wandb.project=vmcnet-phase-b \
  --config.wandb.group="e8-matched-constant" \
  --config.wandb.name="$RUN_NAME"

echo "E8 done $(date -Is)"
