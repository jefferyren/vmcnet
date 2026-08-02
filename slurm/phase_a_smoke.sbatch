#!/bin/bash
# Phase A smoke tests -- run these FIRST, before committing any sweep budget.
#   task 0-2: carbon at nchains=2000, 200 epochs, {spring, prime_sr, ssu}.
#             OOM check for the E3 2000-walker anchor on an 11GB GTX2080TI (the
#             NTK path can materialize N x P Jacobians; training above
#             nchains=1000 has never been run in this repo).
#   task 3:   N2 4.0 Bohr, spring, 200 epochs @1000 walkers. Timing anchor for
#             the Phase C budget (the 15-20h/run guess has no repo evidence).
#   task 4:   H4, spring, 200 epochs @1000 walkers. Timing anchor for the H4
#             screens.
# Per-step wall time = (t_last_epoch - t_first_epoch)/200 from the .out
# timestamps; date stamps are echoed around the run. nburn is cut to 1000 so
# the smoke stays short; eval is disabled.
#
# Submit from the vmcnet repo root:  sbatch slurm/phase_a_smoke.sbatch
#
#SBATCH --job-name=phase-a-smoke
#SBATCH --account=co_esmath
#SBATCH --partition=savio3_gpu
#SBATCH --qos=esmath_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH --array=0-4
#SBATCH --output=slurm-%x-%A_%a.out

set -eo pipefail
export PATH="/global/scratch/users/$USER/envs/vmcnet/bin:$PATH"
set -u

export WANDB_MODE=offline
export WANDB_DIR="/global/scratch/users/$USER/wandb"
mkdir -p "$WANDB_DIR"

cd "$SLURM_SUBMIT_DIR"
LOGROOT="/global/scratch/users/$USER/vmcnet_logs/phase_a/smoke"
mkdir -p "$LOGROOT"

case "$SLURM_ARRAY_TASK_ID" in
  0) PRESET=carbon;  OPT=spring;                       NCHAINS=2000 ;;
  1) PRESET=carbon;  OPT=prime_sr;                     NCHAINS=2000 ;;
  2) PRESET=carbon;  OPT=same_sampled_spring_unified;  NCHAINS=2000 ;;
  3) PRESET=N2_4.0;  OPT=spring;                       NCHAINS=1000 ;;
  4) PRESET=H4;      OPT=spring;                       NCHAINS=1000 ;;
esac
RUN_NAME="smoke_${PRESET}_${OPT}_n${NCHAINS}"

if [[ -e "$LOGROOT/$RUN_NAME" ]]; then
  echo "ERROR: $LOGROOT/$RUN_NAME already exists; move or delete it first." >&2
  exit 1
fi

python -c "import jax; print('jax devices:', jax.devices())"
echo "SMOKE START $(date -Is)"

vmc-molecule \
  --presets.path="preset_configs/${PRESET}.json" \
  --config.logdir="$LOGROOT" \
  --config.save_to_current_datetime_subfolder=False \
  --config.subfolder_name="$RUN_NAME" \
  --config.vmc.optimizer_type="$OPT" \
  --config.vmc.optimizer."$OPT".learning_rate=0.02 \
  --config.vmc.nchains="$NCHAINS" \
  --config.vmc.nburn=1000 \
  --config.vmc.nepochs=200 \
  --config.vmc.checkpoint_every=200 \
  --config.eval.nepochs=0 \
  --config.eval.nburn=0 \
  --config.initial_seed=0 \
  --config.wandb.mode=disabled

echo "SMOKE END $(date -Is)"
