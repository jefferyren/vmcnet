#!/bin/bash
# E10 EVAL RECOVERY — re-run the eval phase alone, from the 50k checkpoints.
#
# WHY THIS EXISTS. Every one of the 34 E10 runs that finished training then died
# entering the eval phase:
#
#   INFO:root:Completed VMC! Evaluating
#   RESOURCE_EXHAUSTED: Out of memory while trying to allocate 8527085888 bytes (7.94 GiB)
#
# The presets set eval.nchains=2000 against vmc.nchains=1000, so eval doubles the walker
# count. On a 14-electron system with 16 determinants that does not fit an 11 GB
# GTX2080TI. Carbon (6 electrons) did fit, which is why this never showed up before.
# The consequence is that E10 has NO eval energies -- and eval energies are the
# publishable number, because wandb drops the eval phase entirely (its step counter
# restarts). The report currently quotes training tails and is marked provisional.
#
# The 50k training is DONE and checkpointed, so this does not retrain anything. It
# reloads the final checkpoint, runs zero training epochs, and goes straight to eval.
# ~1-2 h per run instead of the ~10 h a full redo would cost.
#
# THREE THINGS THIS GETS RIGHT, EACH OF WHICH WOULD SILENTLY CORRUPT THE RESULT:
#
#  1. It reloads `checkpoints/50000.npz`, NOT `best_checkpoint.npz`. The reload default
#     is the "best" file, which vmcnet selects on the best error-adjusted RUNNING AVERAGE
#     energy. That is a best-of-training pick, it lands at a different epoch for every
#     arm, and using it would bias the comparison in an uncontrolled direction.
#  2. It sets eval.nchains=1000. Training ran at 1000 on these systems without trouble;
#     anything above ~1500 is likely to OOM again on an 11 GB card.
#  3. It keeps --job-name=e10-hometurf so the output file matches
#     `slurm-e10-hometurf-*_<idx>.out`, which is the glob slurm/parse_eval_energies.py
#     walks. The parser prefers whichever file for an index actually has eval epochs and
#     merges the training tail back in from the original run, so E10 reconciles
#     automatically once these land.
#
# HOW THE RELOAD RESOLVES. `--reload.logdir` restores params/data/optimizer state and
# sets start_epoch=50000; `use_config_file` (default true) loads that run's config.json
# as the BASE, and command-line `--config.*` flags override it on top -- so the
# eval.nchains override does apply. vmc_loop then iterates `range(50000, 0)`, which is
# empty, and control falls through to eval. The training burn is skipped automatically
# (`skip_burn = reload_from_checkpoint and not reload.reburn`). Output goes to a NEW
# logdir (`..._1`, via add_suffix_for_uniqueness), so the original run is untouched.
#
# WHAT TO RUN. 34 of 36 cells need this -- all but indices 3 and 32, whose training was
# killed early and which need a real rerun via e10_molecules_hometurf.sbatch instead:
#     sbatch --exclude=n0135.savio3 --array=0-2,4-31,33-35 slurm/e10_eval_recovery.sbatch
# Re-time one run before launching all 34:
#     sbatch --exclude=n0135.savio3 --array=0 slurm/e10_eval_recovery.sbatch
#
# Then, from a LOGIN node:  bash slurm/sync_wandb.sh
# Then, locally after pulling the .out files:
#     python slurm/parse_eval_energies.py --experiment E10 --backfill
#
#SBATCH --job-name=e10-hometurf
#SBATCH --account=co_esmath
#SBATCH --partition=savio3_gpu
#SBATCH --qos=esmath_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --cpus-per-task=2
#SBATCH --time=08:00:00
#SBATCH --array=0-35
#SBATCH --output=slurm-%x-%A_%a.out

set -eo pipefail
export PATH="/global/scratch/users/$USER/envs/vmcnet/bin:$PATH"
set -u

export WANDB_MODE=offline
export WANDB_DIR="/global/scratch/users/$USER/wandb"
mkdir -p "$WANDB_DIR"

cd "$SLURM_SUBMIT_DIR"
LOGROOT="/global/scratch/users/$USER/vmcnet_logs/phase_c/e10_hometurf"

# Index arithmetic identical to e10_molecules_hometurf.sbatch -- same cell, same name.
ARM=$((SLURM_ARRAY_TASK_ID / 6))
SYS_I=$(((SLURM_ARRAY_TASK_ID % 6) / 3))
SEED=$((SLURM_ARRAY_TASK_ID % 3))
PRESETS=(N2_eq CO)
PRESET=${PRESETS[$SYS_I]}

case "$ARM" in
  0) TAG=spring_mu0.9   ;;
  1) TAG=spring_mu0.95  ;;
  2) TAG=prime_sr       ;;
  3) TAG=ssu_defaults   ;;
  4) TAG=spring_mu0.99  ;;
  5) TAG=spring_mu0.995 ;;
esac

RUN_NAME="e10_${PRESET}_${TAG}_s${SEED}"
SRC="$LOGROOT/$RUN_NAME"
CKPT="checkpoints/50000.npz"

export WANDB_TAGS="e10,phase-c,home-turf,eval-recovery,follows-e10,arm-${TAG},sys-${PRESET}"
export WANDB_NOTES="E10 eval-phase recovery for ${RUN_NAME}. The original run finished \
50k training and then OOMed entering eval (eval.nchains=2000 doubles the walkers; 14 \
electrons x 16 determinants does not fit 11 GB). This reloads ${CKPT} -- the final \
regular checkpoint, NOT best_checkpoint.npz, which is a best-of-training pick -- runs \
zero training epochs, and evaluates at nchains=1000."

if [[ ! -d "$SRC" ]]; then
  echo "ERROR: source run directory $SRC does not exist." >&2
  exit 1
fi
if [[ ! -f "$SRC/$CKPT" ]]; then
  echo "ERROR: $SRC/$CKPT not found. Available checkpoints:" >&2
  ls -1 "$SRC/checkpoints" 2>/dev/null >&2 || echo "  (no checkpoints directory)" >&2
  echo "Refusing to fall back to best_checkpoint.npz: it is selected on best running" >&2
  echo "energy, lands at a different epoch per arm, and would bias the comparison." >&2
  exit 1
fi

python -c "import jax; print('jax devices:', jax.devices())"
echo "E10-EVAL arm=$TAG preset=$PRESET seed=$SEED  from $SRC/$CKPT  $(date -Is)"

vmc-molecule \
  --reload.logdir="$SRC" \
  --reload.checkpoint_relative_file_path="$CKPT" \
  --reload.use_config_file=True \
  --reload.new_optimizer_state=False \
  --reload.reburn=False \
  --config.vmc.nepochs=0 \
  --config.eval.nchains=1000 \
  --config.eval.nburn=10000 \
  --config.eval.nepochs=20000 \
  --config.wandb.mode=offline \
  --config.wandb.project=vmcnet-phase-c \
  --config.wandb.group="e10-evalrecovery-${PRESET}" \
  --config.wandb.name="e10eval_${PRESET}_${TAG}_s${SEED}"

echo "E10-EVAL done $(date -Is)"
