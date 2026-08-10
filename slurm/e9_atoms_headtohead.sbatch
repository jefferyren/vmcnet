#!/bin/bash
# E9 — generality: the same head-to-head on the N and O atoms.   FOLLOWS FROM: E7.
#
# WHY THIS RUNS. E7 settled both campaign claims, but on only two systems: SS-SPRING
# ties the best fixed mu (carbon +0.148+/-0.004 vs SPRING0.99 +0.147+/-0.012 mHa; H4
# -2.03420 vs -2.03418) and beats PRIME-SR (carbon by 0.069+/-0.009 mHa, ~7 sigma; H4
# by 0.172+/-0.050). Two systems at one learning rate is the obvious objection, and
# N and O are the cheapest way to answer it: they are the SPRING paper's own systems,
# so the numbers are directly comparable to published values, and they cost about the
# same as carbon.
#
# WHY THE mu GRID IS BUILT IN. E7 also showed that E0's single-seed 25k grid MIS-PICKED
# the momentum -- its H4 choice (mu=0.8) turned out to be the worst of four arms at 5
# seeds and 50k. So the tuned baseline here is not imported from a cheap screen; three
# mu values are run at full length alongside the adaptive methods and the best of them
# IS the tuned baseline. E7 found 0.99 best on both systems, with 0.995 and 0.95
# bracketing it, which is what those three values are.
#
# THE ARMS (5), each x {N, O} x 5 seeds = 50 runs:
#   0  SPRING mu=0.95     |
#   1  SPRING mu=0.99     |  best of these three = the tuned baseline for claim A
#   2  SPRING mu=0.995    |
#   3  PRIME-SR              claim B comparator
#   4  SS-SPRING             untuned
#
# eta=0.02 (the SPRING paper's tuned value for atoms; carbon confirmed it in E0).
# 50k training + 20k eval at 2000 walkers, as in E7 -- 25k is not converged.
#
# ~4-5h per run => ~225 GPU-h. To trim, run 3 seeds instead of 5:
#     sbatch --array=0-49:1 ... then cancel seeds 3,4  (or edit SEEDS below)
# Array index = arm*10 + system*5 + seed.
#
# SCORE IT LIKE E7: mean +/- s.e.m. across seeds, WORST SEED, and spread. Claim A
# holds if SS-SPRING overlaps the best SPRING arm; claim B if it is at least as good
# as PRIME-SR. A tie on the mean with a better worst seed still counts for B.
#
# Submit from the vmcnet repo root:  sbatch slurm/e9_atoms_headtohead.sbatch
# Then sync from a LOGIN node:       bash slurm/sync_wandb.sh
#
#SBATCH --job-name=e9-atoms
#SBATCH --account=co_esmath
#SBATCH --partition=savio3_gpu
#SBATCH --qos=esmath_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --cpus-per-task=2
#SBATCH --time=20:00:00
#SBATCH --array=0-49
#SBATCH --output=slurm-%x-%A_%a.out

set -eo pipefail
export PATH="/global/scratch/users/$USER/envs/vmcnet/bin:$PATH"
set -u

export WANDB_MODE=offline
export WANDB_DIR="/global/scratch/users/$USER/wandb"
mkdir -p "$WANDB_DIR"

cd "$SLURM_SUBMIT_DIR"
LOGROOT="/global/scratch/users/$USER/vmcnet_logs/phase_c/e9_atoms"
mkdir -p "$LOGROOT"

SSU=same_sampled_spring_unified
ARM=$((SLURM_ARRAY_TASK_ID / 10))
SYS_I=$(((SLURM_ARRAY_TASK_ID % 10) / 5))
SEED=$((SLURM_ARRAY_TASK_ID % 5))
PRESETS=(N O)
PRESET=${PRESETS[$SYS_I]}

case "$ARM" in
  0) TAG=spring_mu0.95;  OPT=spring;   ARM_FLAGS=(--config.vmc.optimizer.spring.mu=0.95) ;;
  1) TAG=spring_mu0.99;  OPT=spring;   ARM_FLAGS=(--config.vmc.optimizer.spring.mu=0.99) ;;
  2) TAG=spring_mu0.995; OPT=spring;   ARM_FLAGS=(--config.vmc.optimizer.spring.mu=0.995) ;;
  3) TAG=prime_sr;       OPT=prime_sr; ARM_FLAGS=() ;;
  4) TAG=ssu_defaults;   OPT=$SSU;     ARM_FLAGS=(--config.vmc.optimizer."$SSU".mu=0.0) ;;
esac

RUN_NAME="e9_${PRESET}_${TAG}_s${SEED}"

export WANDB_TAGS="e9,phase-c,generality,head-to-head,follows-e7,follows-e0,arm-${TAG},sys-${PRESET}"
export WANDB_NOTES="E9 generality check on the N and O atoms, extending E7's two-system \
result (SS-SPRING tied the best fixed mu and beat PRIME-SR by 0.069+/-0.009 mHa on \
carbon, 0.172+/-0.050 on H4). The mu grid is run at full length here rather than \
imported, because E7 showed E0's single-seed 25k grid mis-picked mu (its H4 choice of \
0.8 was the worst arm at 50k). Arm: ${TAG} on ${PRESET}."

if [[ -e "$LOGROOT/$RUN_NAME" ]]; then
  echo "ERROR: $LOGROOT/$RUN_NAME already exists; move or delete it first." >&2
  exit 1
fi

python -c "import jax; print('jax devices:', jax.devices())"
echo "E9 arm=$TAG preset=$PRESET seed=$SEED  $(date -Is)"

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
  --config.wandb.project=vmcnet-phase-c \
  --config.wandb.group="e9-atoms-${PRESET}" \
  --config.wandb.name="$RUN_NAME"

echo "E9 done $(date -Is)"
