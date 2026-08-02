#!/bin/bash
# E3: sample-size sweep, 25k-epoch screens from RANDOM INIT, seed 0.
# Dims: nchains {100, 250, 500, 1000} x optimizer {spring(mu=SPRING_MU),
# prime_sr, same_sampled_spring_unified} x system {carbon, H4}.
# Array index = n_idx * 6 + opt_idx * 2 + sys_idx (24 tasks).
# eta=0.02, constraint ON (default), check_for_nans=True.
#
# This is the VMC reproduction of the PINN small-batch effect. Mechanism
# deliverable: prime_sr's mu.txt / rank.txt / alpha.txt vs nchains (prediction:
# mu_k rises toward 1 as N shrinks -- pure-noise floor gives
# mu = 1-(1-(alpha/rank)^{1/4})^2, e.g. 0.64 -> 0.92 as N goes 1000 -> 100 at
# alpha~25; the existing LiH run at N=100 already shows mu_k ~ 0.976).
# NOTE: from-scratch starts make the nchains override safe (a reloaded
# checkpoint would silently pin the walker count).
#
# Submit from the vmcnet repo root:  sbatch slurm/e3_nchains_sweep.sbatch
#
#SBATCH --job-name=e3-nchains
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

export WANDB_MODE=offline
export WANDB_DIR="/global/scratch/users/$USER/wandb"
mkdir -p "$WANDB_DIR"

cd "$SLURM_SUBMIT_DIR"
SEED="${SEED:-0}"
SPRING_MU="${SPRING_MU:-0.99}"
LOGROOT="/global/scratch/users/$USER/vmcnet_logs/phase_a/e3_nchains_sweep"
mkdir -p "$LOGROOT"

NCHAINS_LIST=(100 250 500 1000)
OPTS=(spring prime_sr same_sampled_spring_unified)
PRESETS=(carbon H4)

NCHAINS=${NCHAINS_LIST[$((SLURM_ARRAY_TASK_ID / 6))]}
OPT=${OPTS[$(((SLURM_ARRAY_TASK_ID % 6) / 2))]}
PRESET=${PRESETS[$((SLURM_ARRAY_TASK_ID % 2))]}

if [[ "$OPT" == "spring" ]]; then
  OPT_TAG="spring_mu${SPRING_MU}"
  EXTRA_FLAGS=(--config.vmc.optimizer.spring.mu="$SPRING_MU")
else
  OPT_TAG="$OPT"
  EXTRA_FLAGS=()
fi
RUN_NAME="e3_${PRESET}_${OPT_TAG}_n${NCHAINS}_s${SEED}"

if [[ -e "$LOGROOT/$RUN_NAME" ]]; then
  echo "ERROR: $LOGROOT/$RUN_NAME already exists; move or delete it first." >&2
  exit 1
fi

python -c "import jax; print('jax devices:', jax.devices())"

vmc-molecule \
  --presets.path="preset_configs/${PRESET}.json" \
  --config.logdir="$LOGROOT" \
  --config.save_to_current_datetime_subfolder=False \
  --config.subfolder_name="$RUN_NAME" \
  --config.vmc.optimizer_type="$OPT" \
  --config.vmc.optimizer."$OPT".learning_rate=0.02 \
  ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"} \
  --config.vmc.nchains="$NCHAINS" \
  --config.vmc.nepochs=25000 \
  --config.vmc.checkpoint_every=5000 \
  --config.vmc.check_for_nans=True \
  --config.eval.nepochs=0 \
  --config.eval.nburn=0 \
  --config.initial_seed="$SEED" \
  --config.wandb.mode=offline \
  --config.wandb.project=vmcnet-phase-a \
  --config.wandb.group="e3-nchains-${PRESET}" \
  --config.wandb.name="$RUN_NAME"
