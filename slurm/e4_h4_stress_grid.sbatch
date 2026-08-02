#!/bin/bash
# E4.1: H4 stability stress grid, constraint OFF, 25k epochs, RANDOM INIT.
# Dims: eta {0.02, 0.1, 0.3} x nchains {128, 1000} x optimizer {spring(0.99),
# prime_sr, same_sampled_spring_unified}. Array = eta_idx*6 + n_idx*3 + opt_idx
# (18 tasks). Constraint OFF is deliberate: with the cap on, the high-eta
# column is trivially all-stable and the phase diagram is uninformative.
# check_for_nans=True turns divergence into an early abort + checkpoint; the
# "diverged" label for the phase diagram = NaN abort OR final smoothed energy
# above threshold (see docs/PHASE_A.md).
# Deliverable: per-optimizer stability phase diagram over (eta, nchains).
#
# Submit from the vmcnet repo root:  sbatch slurm/e4_h4_stress_grid.sbatch
#
#SBATCH --job-name=e4-h4-stress
#SBATCH --account=co_esmath
#SBATCH --partition=savio3_gpu
#SBATCH --qos=esmath_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --cpus-per-task=2
#SBATCH --time=08:00:00
#SBATCH --array=0-17
#SBATCH --output=slurm-%x-%A_%a.out

set -eo pipefail
export PATH="/global/scratch/users/$USER/envs/vmcnet/bin:$PATH"
set -u

export WANDB_MODE=offline
export WANDB_DIR="/global/scratch/users/$USER/wandb"
mkdir -p "$WANDB_DIR"

cd "$SLURM_SUBMIT_DIR"
SEED="${SEED:-0}"
LOGROOT="/global/scratch/users/$USER/vmcnet_logs/phase_a/e4_h4_stress_grid"
mkdir -p "$LOGROOT"

ETAS=(0.02 0.1 0.3)
NCHAINS_LIST=(128 1000)
OPTS=(spring prime_sr same_sampled_spring_unified)

ETA=${ETAS[$((SLURM_ARRAY_TASK_ID / 6))]}
NCHAINS=${NCHAINS_LIST[$(((SLURM_ARRAY_TASK_ID % 6) / 3))]}
OPT=${OPTS[$((SLURM_ARRAY_TASK_ID % 3))]}
RUN_NAME="e4_h4_${OPT}_eta${ETA}_n${NCHAINS}_s${SEED}"

if [[ -e "$LOGROOT/$RUN_NAME" ]]; then
  echo "ERROR: $LOGROOT/$RUN_NAME already exists; move or delete it first." >&2
  exit 1
fi

python -c "import jax; print('jax devices:', jax.devices())"

vmc-molecule \
  --presets.path=preset_configs/H4.json \
  --config.logdir="$LOGROOT" \
  --config.save_to_current_datetime_subfolder=False \
  --config.subfolder_name="$RUN_NAME" \
  --config.vmc.optimizer_type="$OPT" \
  --config.vmc.optimizer."$OPT".learning_rate="$ETA" \
  --config.vmc.optimizer."$OPT".constrain_norm=False \
  --config.vmc.nchains="$NCHAINS" \
  --config.vmc.nepochs=25000 \
  --config.vmc.checkpoint_every=5000 \
  --config.vmc.check_for_nans=True \
  --config.eval.nepochs=0 \
  --config.eval.nburn=0 \
  --config.initial_seed="$SEED" \
  --config.wandb.mode=offline \
  --config.wandb.project=vmcnet-phase-a \
  --config.wandb.group=e4-h4-stress \
  --config.wandb.name="$RUN_NAME"
