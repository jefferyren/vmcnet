#!/bin/bash
# Preliminary optimization for the Carbon atom (SPRING paper, Table B.5):
# 1000 iterations of KFAC at lr=0.05 from random init. All other settings
# (orthogonal kernel init, normal bias init, clip=5, 1000 walkers, 5000 burn-in,
# 10 MCMC steps/update) are vmcnet defaults + preset_configs/carbon.json.
# Writes params to $SCRATCH/vmcnet_logs/carbon_prelim/checkpoints/1000.npz,
# which carbon_optimizers.sbatch reloads for the main runs.
#
# Submit from the vmcnet repo root:  sbatch slurm/carbon_prelim.sbatch
#
#SBATCH --job-name=carbon-prelim
#SBATCH --account=co_esmath
#SBATCH --partition=savio3_gpu
#SBATCH --qos=esmath_gpu3_normal     # co_esmath condo QoS on savio3_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:GTX2080TI:1       # GPU type must match the QoS (else the job hangs on QOSMinGRES)
#SBATCH --cpus-per-task=2            # Savio ratio: 2 cores per GTX2080TI, 8 per A40
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
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

# wandb: offline, not disabled. `disabled` records nothing at all and leaves
# nothing to sync afterward; `offline` buffers the run under $WANDB_DIR for a
# later `wandb sync` from the login node. The env var keeps the unconditional
# wandb.login() in runners.py from reaching the network on a compute node, but
# runners.py passes config.wandb.mode explicitly to wandb.init(), so the
# --config.wandb.mode flag below is what actually sets the run's mode.
export WANDB_MODE=offline
export WANDB_DIR="/global/scratch/users/$USER/wandb"
mkdir -p "$WANDB_DIR"

cd "$SLURM_SUBMIT_DIR"

# Logs go to scratch, not the repo tree: 100k epochs of checkpoints will blow
# through the 10 GB $HOME quota.
LOGROOT="/global/scratch/users/$USER/vmcnet_logs"
mkdir -p "$LOGROOT"

# vmcnet's add_suffix_for_uniqueness (utils/io.py) NEVER overwrites an existing
# logdir -- it silently writes to carbon_prelim_1 instead. Since
# carbon_optimizers.sbatch reloads the hardcoded carbon_prelim path, a stale
# directory here would make all three optimizer runs restart from the wrong
# checkpoint while afterok still reports success. Fail loudly instead.
if [[ -e "$LOGROOT/carbon_prelim" ]]; then
  echo "ERROR: $LOGROOT/carbon_prelim already exists." >&2
  echo "Move or delete it, or the array job will reload a stale checkpoint." >&2
  exit 1
fi

python -c "import jax; print('jax devices:', jax.devices())"

vmc-molecule \
  --presets.path=preset_configs/carbon.json \
  --config.logdir="$LOGROOT" \
  --config.vmc.optimizer_type=kfac \
  --config.vmc.nepochs=1000 \
  --config.vmc.checkpoint_every=1000 \
  --config.eval.nepochs=0 \
  --config.save_to_current_datetime_subfolder=False \
  --config.subfolder_name=carbon_prelim \
  --config.wandb.mode=offline \
  --config.wandb.project=vmcnet-carbon \
  --config.wandb.group=carbon \
  --config.wandb.name=carbon_prelim
