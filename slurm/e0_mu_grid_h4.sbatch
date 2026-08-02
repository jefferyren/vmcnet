#!/bin/bash
# E0 stage 1 (H4 square, 4 Bohr): SPRING mu grid x eta {0.02, 0.002}, 25k-epoch
# screens from RANDOM INIT, seed 0. H4 is a molecule, so the SPRING-paper
# "atoms 0.02 / molecules 0.002" split leaves its eta genuinely unknown -- the
# two eta columns double as the H4 mini eta check from the campaign plan.
# Array index = mu_index * 2 + eta_index (mu in {0,...,0.999} as in the carbon
# grid; eta 0 -> 0.02, 1 -> 0.002).
#
# Submit from the vmcnet repo root:  sbatch slurm/e0_mu_grid_h4.sbatch
#
#SBATCH --job-name=e0-mu-h4
#SBATCH --account=co_esmath
#SBATCH --partition=savio3_gpu
#SBATCH --qos=esmath_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --cpus-per-task=2
#SBATCH --time=08:00:00
#SBATCH --array=0-19
#SBATCH --output=slurm-%x-%A_%a.out

set -eo pipefail
export PATH="/global/scratch/users/$USER/envs/vmcnet/bin:$PATH"
set -u

export WANDB_MODE=offline
export WANDB_DIR="/global/scratch/users/$USER/wandb"
mkdir -p "$WANDB_DIR"

cd "$SLURM_SUBMIT_DIR"
SEED="${SEED:-0}"
LOGROOT="/global/scratch/users/$USER/vmcnet_logs/phase_a/e0_mu_grid_h4"
mkdir -p "$LOGROOT"

MUS=(0.0 0.5 0.8 0.9 0.95 0.98 0.99 0.995 0.998 0.999)
ETAS=(0.02 0.002)
MU=${MUS[$((SLURM_ARRAY_TASK_ID / 2))]}
ETA=${ETAS[$((SLURM_ARRAY_TASK_ID % 2))]}
RUN_NAME="e0_h4_spring_mu${MU}_eta${ETA}_s${SEED}"

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
  --config.vmc.optimizer_type=spring \
  --config.vmc.optimizer.spring.learning_rate="$ETA" \
  --config.vmc.optimizer.spring.mu="$MU" \
  --config.vmc.nepochs=25000 \
  --config.vmc.checkpoint_every=5000 \
  --config.vmc.check_for_nans=True \
  --config.eval.nepochs=0 \
  --config.eval.nburn=0 \
  --config.initial_seed="$SEED" \
  --config.wandb.mode=offline \
  --config.wandb.project=vmcnet-phase-a \
  --config.wandb.group=e0-mu-grid-h4 \
  --config.wandb.name="$RUN_NAME"
