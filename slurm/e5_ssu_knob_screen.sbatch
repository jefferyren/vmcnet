#!/bin/bash
# E5 (early slice): same_sampled_spring_unified knob-robustness screen on
# carbon, 25k epochs, RANDOM INIT, seed 0, eta=0.02. One knob deviates from
# defaults per task; task 0 is the all-defaults reference:
#   0: defaults (mu0=0, lb_window=30, probe_lr=-1->eta, probe_damping=1e-3,
#      adaptive_eta=False, adaptive_probe=False)
#   1: lb_window=10        2: lb_window=100      3: initial beta mu=0.99
#   4: probe_damping=1e-4  5: adaptive_eta=True  6: adaptive_probe=True
# Run this EARLY (Phase A): the "untuned" claim is question-begging if the
# defaults are only validated after the headline systems are chosen.
#
# Submit from the vmcnet repo root:  sbatch slurm/e5_ssu_knob_screen.sbatch
#
#SBATCH --job-name=e5-ssu-knobs
#SBATCH --account=co_esmath
#SBATCH --partition=savio3_gpu
#SBATCH --qos=esmath_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --array=0-6
#SBATCH --output=slurm-%x-%A_%a.out

set -eo pipefail
export PATH="/global/scratch/users/$USER/envs/vmcnet/bin:$PATH"
set -u

export WANDB_MODE=offline
export WANDB_DIR="/global/scratch/users/$USER/wandb"
mkdir -p "$WANDB_DIR"

cd "$SLURM_SUBMIT_DIR"
SEED="${SEED:-0}"
LOGROOT="/global/scratch/users/$USER/vmcnet_logs/phase_a/e5_ssu_knobs"
mkdir -p "$LOGROOT"

SSU=same_sampled_spring_unified
case "$SLURM_ARRAY_TASK_ID" in
  0) TAG=defaults;          EXTRA_FLAGS=() ;;
  1) TAG=lbwindow10;        EXTRA_FLAGS=(--config.vmc.optimizer.$SSU.lb_window=10) ;;
  2) TAG=lbwindow100;       EXTRA_FLAGS=(--config.vmc.optimizer.$SSU.lb_window=100) ;;
  3) TAG=mu0init0.99;       EXTRA_FLAGS=(--config.vmc.optimizer.$SSU.mu=0.99) ;;
  4) TAG=probedamp1e-4;     EXTRA_FLAGS=(--config.vmc.optimizer.$SSU.probe_damping=1e-4) ;;
  5) TAG=adaptiveeta;       EXTRA_FLAGS=(--config.vmc.optimizer.$SSU.adaptive_eta=True) ;;
  6) TAG=adaptiveprobe;     EXTRA_FLAGS=(--config.vmc.optimizer.$SSU.adaptive_probe=True) ;;
esac
RUN_NAME="e5_carbon_ssu_${TAG}_s${SEED}"

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
  --config.vmc.optimizer_type="$SSU" \
  --config.vmc.optimizer."$SSU".learning_rate=0.02 \
  ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"} \
  --config.vmc.nepochs=25000 \
  --config.vmc.checkpoint_every=5000 \
  --config.vmc.check_for_nans=True \
  --config.eval.nepochs=0 \
  --config.eval.nburn=0 \
  --config.initial_seed="$SEED" \
  --config.wandb.mode=offline \
  --config.wandb.project=vmcnet-phase-a \
  --config.wandb.group=e5-ssu-knobs \
  --config.wandb.name="$RUN_NAME"
