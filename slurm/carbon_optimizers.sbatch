#!/bin/bash
# Main Carbon-atom runs (SPRING paper tuned settings): spring, prime_sr, and
# same_sampled_spring_unified, each starting from the end of the preliminary
# KFAC optimization ($SCRATCH/vmcnet_logs/carbon_prelim/checkpoints/1000.npz)
# with a fresh
# optimizer state. 100k training iterations, then a 20k-iteration inference
# phase with 2000 walkers. lr=0.02 (inverse-time decay 1e-4), damping=1e-3,
# norm constraint=1e-3 for all three.
#
# Submit from the vmcnet repo root, after the prelim job finishes:
#   sbatch --dependency=afterok:<PRELIM_JOBID> slurm/carbon_optimizers.sbatch
#
#SBATCH --job-name=carbon-optim
#SBATCH --account=co_esmath
#SBATCH --partition=savio3_gpu
#SBATCH --qos=esmath_gpu3_normal     # co_esmath condo QoS on savio3_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:GTX2080TI:1       # GPU type must match the QoS (else the job hangs on QOSMinGRES)
#SBATCH --cpus-per-task=2            # Savio ratio: 2 cores per GTX2080TI, 8 per A40
# Measured on a GTX2080TI at 200 epochs: spring 1.00x, prime_sr ~1.16x,
# same_sampled_spring_unified ~1.32x -> 120k epochs lands at ~15-26h. 72h is
# requested purely as headroom; savio3_gpu has MaxTime=UNLIMITED and
# esmath_gpu3_normal has no MaxWall, so over-requesting costs nothing.
#SBATCH --time=72:00:00
#SBATCH --array=0-2
#SBATCH --output=slurm-%x-%A_%a.out
# NOTE: no --mem. Savio allocates memory proportionally to cores; an explicit
# request can conflict with partition policy.

# NOTE: `set -u` comes after conda activation; conda's shell functions reference
# unbound variables on some versions.
set -eo pipefail

# --- environment ---
# Deliberately NOT using `module load anaconda3` + `conda activate`. sbatch
# exports the submitting shell's environment, so CONDA_PREFIX arrives already
# set; `conda activate` then treats the env as active and skips its PATH edit,
# while `module load anaconda3` has put the module's own python first. The
# result is a silent fall-through to the wrong interpreter. The env is a
# self-contained prefix, so putting its bin on PATH is sufficient and needs no
# shell initialization.
export PATH="/global/scratch/users/$USER/envs/vmcnet/bin:$PATH"
set -u

export WANDB_MODE=disabled           # runners.py calls wandb.login(); keep it a no-op

cd "$SLURM_SUBMIT_DIR"

LOGROOT="/global/scratch/users/$USER/vmcnet_logs"
PRELIM_DIR="$LOGROOT/carbon_prelim"
CKPT="$PRELIM_DIR/checkpoints/1000.npz"
if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: expected prelim checkpoint not found at $CKPT" >&2
  exit 1
fi

OPTIMIZERS=(spring prime_sr same_sampled_spring_unified)
OPT=${OPTIMIZERS[$SLURM_ARRAY_TASK_ID]}
echo "Running optimizer: $OPT"

python -c "import jax; print('jax devices:', jax.devices())"

vmc-molecule \
  --reload.logdir="$PRELIM_DIR" \
  --reload.checkpoint_relative_file_path=checkpoints/1000.npz \
  --reload.new_optimizer_state=True \
  --reload.append=False \
  --config.vmc.optimizer_type="$OPT" \
  --config.vmc.optimizer."$OPT".learning_rate=0.02 \
  --config.vmc.nepochs=100000 \
  --config.vmc.checkpoint_every=5000 \
  --config.eval.nepochs=20000 \
  --config.subfolder_name=carbon_"$OPT"
