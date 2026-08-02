#!/bin/bash
# E0 stage 1 (carbon): SPRING mu grid at fixed eta=0.02 (SPRING-paper atom
# value), 25k-epoch screening runs from RANDOM INIT, seed 0. The best-2 mu go
# on to full-length 100k runs (stage 2). mu grid follows the campaign plan:
# {0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999} -- mu=0 is a
# labeled MinSR floor, and the top of the grid is expected to be unstable
# (PRIME-SR paper App. E.1). check_for_nans=True aborts + checkpoints a NaN'd
# run instead of burning the full allocation.
# Ranking criterion: mean training energy over the final 3000 epochs (matches
# the PRIME-SR paper's smoothing window); eval is skipped in the screen.
#
# Submit from the vmcnet repo root:  sbatch slurm/e0_mu_grid_carbon.sbatch
# Second seed (optional): sbatch --export=ALL,SEED=1 slurm/e0_mu_grid_carbon.sbatch
#
#SBATCH --job-name=e0-mu-carbon
#SBATCH --account=co_esmath
#SBATCH --partition=savio3_gpu
#SBATCH --qos=esmath_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --array=0-9
#SBATCH --output=slurm-%x-%A_%a.out

set -eo pipefail
export PATH="/global/scratch/users/$USER/envs/vmcnet/bin:$PATH"
set -u

export WANDB_MODE=offline
export WANDB_DIR="/global/scratch/users/$USER/wandb"
mkdir -p "$WANDB_DIR"

cd "$SLURM_SUBMIT_DIR"
SEED="${SEED:-0}"
LOGROOT="/global/scratch/users/$USER/vmcnet_logs/phase_a/e0_mu_grid_carbon"
mkdir -p "$LOGROOT"

MUS=(0.0 0.5 0.8 0.9 0.95 0.98 0.99 0.995 0.998 0.999)
MU=${MUS[$SLURM_ARRAY_TASK_ID]}
RUN_NAME="e0_carbon_spring_mu${MU}_s${SEED}"

if [[ -e "$LOGROOT/$RUN_NAME" ]]; then
  echo "ERROR: $LOGROOT/$RUN_NAME already exists; move or delete it first." >&2
  exit 1
fi

python -c "import jax; print('jax devices:', jax.devices())"

vmc-molecule \
  --presets.path=preset_configs/carbon.json \
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
  --config.wandb.project=vmcnet-phase-a \
  --config.wandb.group=e0-mu-grid-carbon \
  --config.wandb.name="$RUN_NAME"
