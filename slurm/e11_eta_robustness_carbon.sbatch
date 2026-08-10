#!/bin/bash
# E11 — do the claims survive a different learning rate?   FOLLOWS FROM: E7.
#
# WHY THIS RUNS. Every number supporting the two campaign claims was taken at a single
# learning rate, eta=0.02. That is a one-line objection to the whole result, and it is
# cheap to close. This repeats E7's head-to-head on carbon at eta=0.005 and eta=0.05 --
# a 10x span bracketing the tuned value.
#
# WHAT MAKES THIS DIFFERENT FROM E2. E2 swept eta on carbon in Phase A and found the
# energies almost flat, because with the norm constraint ON the realized step is
# min(eta*||phi||, sqrt(C)) and eta cancels once the cap binds. That was a real finding
# but it means E2 could not test what we need here. Two things are different now:
# runs are 50k with an eval phase (E2 was 25k, training tail only), and the question is
# no longer "does eta matter" but "does the RANKING of the four methods hold". Even if
# the absolute energies move little, a ranking that flips with eta would undercut both
# claims. Read `norm_cap_applied` alongside the energies to know which regime each
# arm is in.
#
# THE GRID: 4 arms x eta {0.005, 0.05} x 3 seeds = 24 runs, carbon, 50k + eval.
#   0  SPRING mu=0.99    (E7's best fixed mu on carbon)
#   1  SPRING mu=0.995   (E0's pick; kept so the pair brackets the optimum)
#   2  PRIME-SR
#   3  SS-SPRING
# ~4h per run => ~96 GPU-h. Array index = arm*6 + eta_index*3 + seed.
#
# WHAT WOULD BREAK THE CLAIMS: SS-SPRING falling behind the best SPRING arm, or behind
# PRIME-SR, at either eta. What would strengthen them: the ordering from E7 (SS-SPRING
# ~ best SPRING > PRIME-SR) reproducing at both.
#
# Submit from the vmcnet repo root:  sbatch slurm/e11_eta_robustness_carbon.sbatch
# Then sync from a LOGIN node:       bash slurm/sync_wandb.sh
#
#SBATCH --job-name=e11-eta-robust
#SBATCH --account=co_esmath
#SBATCH --partition=savio3_gpu
#SBATCH --qos=esmath_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --cpus-per-task=2
#SBATCH --time=20:00:00
#SBATCH --array=0-23
#SBATCH --output=slurm-%x-%A_%a.out

set -eo pipefail
export PATH="/global/scratch/users/$USER/envs/vmcnet/bin:$PATH"
set -u

export WANDB_MODE=offline
export WANDB_DIR="/global/scratch/users/$USER/wandb"
mkdir -p "$WANDB_DIR"

cd "$SLURM_SUBMIT_DIR"
LOGROOT="/global/scratch/users/$USER/vmcnet_logs/phase_c/e11_eta_robustness"
mkdir -p "$LOGROOT"

SSU=same_sampled_spring_unified
ARM=$((SLURM_ARRAY_TASK_ID / 6))
ETA_I=$(((SLURM_ARRAY_TASK_ID % 6) / 3))
SEED=$((SLURM_ARRAY_TASK_ID % 3))
ETAS=(0.005 0.05)
ETA=${ETAS[$ETA_I]}

case "$ARM" in
  0) TAG=spring_mu0.99;  OPT=spring;   ARM_FLAGS=(--config.vmc.optimizer.spring.mu=0.99) ;;
  1) TAG=spring_mu0.995; OPT=spring;   ARM_FLAGS=(--config.vmc.optimizer.spring.mu=0.995) ;;
  2) TAG=prime_sr;       OPT=prime_sr; ARM_FLAGS=() ;;
  3) TAG=ssu_defaults;   OPT=$SSU;     ARM_FLAGS=(--config.vmc.optimizer."$SSU".mu=0.0) ;;
esac

RUN_NAME="e11_carbon_${TAG}_eta${ETA}_s${SEED}"

export WANDB_TAGS="e11,phase-c,eta-robustness,follows-e7,follows-e2,arm-${TAG},eta-${ETA}"
export WANDB_NOTES="E11 learning-rate robustness for E7's head-to-head result, which \
was taken entirely at eta=0.02. Tests whether the ranking (SS-SPRING ~ best fixed mu > \
PRIME-SR) holds at eta=0.005 and 0.05. Unlike E2 (25k, training tail, found energies \
flat because the norm cap makes the step eta-independent once it binds), this runs 50k \
with an eval phase and asks about the ORDERING, not the absolute energy. Arm: ${TAG} \
at eta=${ETA}."

if [[ -e "$LOGROOT/$RUN_NAME" ]]; then
  echo "ERROR: $LOGROOT/$RUN_NAME already exists; move or delete it first." >&2
  exit 1
fi

python -c "import jax; print('jax devices:', jax.devices())"
echo "E11 arm=$TAG eta=$ETA seed=$SEED  $(date -Is)"

vmc-molecule \
  --presets.path=preset_configs/carbon.json \
  --config.logdir="$LOGROOT" \
  --config.save_to_current_datetime_subfolder=False \
  --config.subfolder_name="$RUN_NAME" \
  --config.vmc.optimizer_type="$OPT" \
  --config.vmc.optimizer."$OPT".learning_rate="$ETA" \
  "${ARM_FLAGS[@]}" \
  --config.vmc.nepochs=50000 \
  --config.vmc.checkpoint_every=10000 \
  --config.vmc.check_for_nans=True \
  --config.eval.nchains=2000 \
  --config.eval.nburn=5000 \
  --config.eval.nepochs=20000 \
  --config.initial_seed="$SEED" \
  --config.wandb.mode=offline \
  --config.wandb.project=vmcnet-phase-c \
  --config.wandb.group="e11-eta-${ETA}" \
  --config.wandb.name="$RUN_NAME"

echo "E11 done $(date -Is)"
