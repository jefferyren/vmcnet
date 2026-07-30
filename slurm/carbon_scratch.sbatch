#!/bin/bash
# Carbon-atom runs from RANDOM INIT -- no KFAC preliminary phase. spring,
# prime_sr, and same_sampled_spring_unified each start from scratch, so this is
# the companion experiment to carbon_optimizers.sbatch (which reloads the
# prelim checkpoint). The README notes the prelim phase exists only to make
# optimizer comparisons fair; SPRING is effective from the start.
#
# 100k training iterations, then a 20k-iteration inference phase with 2000
# walkers. lr=0.02 (inverse-time decay 1e-4), damping=1e-3, norm
# constraint=1e-3 for all three. Walker counts, burn-in, and clipping come from
# preset_configs/carbon.json.
#
# Logs to $SCRATCH/vmcnet_logs/carbon_scratch_<optimizer>/ and to wandb project
# vmcnet-carbon, group "carbon-scratch" (the prelim-based runs use group
# "carbon", so the two experiments plot separately but live side by side).
#
# Submit from the vmcnet repo root:  sbatch slurm/carbon_scratch.sbatch
#
#SBATCH --job-name=carbon-scratch
#SBATCH --account=co_esmath
#SBATCH --partition=savio3_gpu
#SBATCH --qos=esmath_gpu3_normal     # co_esmath condo QoS on savio3_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:GTX2080TI:1       # GPU type must match the QoS (else the job hangs on QOSMinGRES)
#SBATCH --cpus-per-task=2            # Savio ratio: 2 cores per GTX2080TI, 8 per A40
# The prelim-based runs took 6.5-7h each for the same 100k+20k epochs; dropping
# the 1000-epoch prelim changes that by under a minute. 72h is pure headroom --
# savio3_gpu has MaxTime=UNLIMITED and esmath_gpu3_normal has no MaxWall, so
# over-requesting costs nothing.
#SBATCH --time=72:00:00
#SBATCH --array=0-2
#SBATCH --output=slurm-%x-%A_%a.out
# NOTE: no --mem. Savio allocates memory proportionally to cores; an explicit
# request can conflict with partition policy.

# NOTE: `set -u` comes after the PATH export for parity with the other scripts.
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

LOGROOT="/global/scratch/users/$USER/vmcnet_logs"
mkdir -p "$LOGROOT"

OPTIMIZERS=(spring prime_sr same_sampled_spring_unified)
OPT=${OPTIMIZERS[$SLURM_ARRAY_TASK_ID]}
RUN_NAME=carbon_scratch_"$OPT"
echo "Running optimizer: $OPT (from random init)"

# vmcnet's add_suffix_for_uniqueness (utils/io.py) NEVER overwrites an existing
# logdir -- it silently writes to <name>_1 instead. That would leave the real
# results under a name nothing downstream expects. Fail loudly instead.
if [[ -e "$LOGROOT/$RUN_NAME" ]]; then
  echo "ERROR: $LOGROOT/$RUN_NAME already exists." >&2
  echo "Move or delete it before resubmitting this array task." >&2
  exit 1
fi

python -c "import jax; print('jax devices:', jax.devices())"

# No --reload flags: this is the from-scratch arm. save_to_current_datetime_subfolder
# must be set explicitly -- it defaults to True, and the prelim-based runs only
# got False by inheriting it through the reloaded config.
vmc-molecule \
  --presets.path=preset_configs/carbon.json \
  --config.logdir="$LOGROOT" \
  --config.save_to_current_datetime_subfolder=False \
  --config.subfolder_name="$RUN_NAME" \
  --config.vmc.optimizer_type="$OPT" \
  --config.vmc.optimizer."$OPT".learning_rate=0.02 \
  --config.vmc.nepochs=100000 \
  --config.vmc.checkpoint_every=5000 \
  --config.eval.nepochs=20000 \
  --config.wandb.mode=offline \
  --config.wandb.project=vmcnet-carbon \
  --config.wandb.group=carbon-scratch \
  --config.wandb.name="$RUN_NAME"
