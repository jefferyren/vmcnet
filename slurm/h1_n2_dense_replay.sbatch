#!/bin/bash
# H1 step 1 -- replay two failing stretched-N2 SS-SPRING runs from their 10k checkpoints
# with a checkpoint every 50 epochs, until they blow up. The dense checkpoints are the
# frozen-parameter snapshots for slurm/h1_frozen_probe.sbatch (step 3).
#
# THE EXPERIMENT (H1). Hypothesis: in VMC the linear system A_k changes from step to
# step as theta moves, which slows the SS-SPRING probe; the adaptive controller reads a
# slow probe as "needs more momentum" and pushes beta up, and the high beta then
# diverges. Test: freeze theta at snapshots along a failing run and run Kaczmarz++ (the
# probe recursion, with MCMC supplying fresh rows each step) on the frozen system. If
# the probe converges there at the run's beta while the run itself diverges, the
# changing system is the problem; if it also stalls/grows, the system itself is.
#
# WHY THESE TWO STATES. Both have failed in every replay so far, with a long enough
# approach after the 10k reload to place snapshots at event -500 ... -20:
#   E14 s0: first event 12,780 (F3b no cap) / 11,150 (F4d)
#   E15 s7: first event 12,480 (F3b no cap) / 11,460 (F4d)
# A replay does not reproduce the original trajectory (float32 rounding), so the event
# epoch of THIS replay is what the snapshots are placed against; h1_frozen_probe.py
# finds it from the replay's own .txt metrics.
#
# CODE. Master's optimizer code is the original SS-SPRING (tag ss-spring-original), the
# same code the failing runs used. No Phase F keys exist on master, so the config is
# rebuilt from the preset + the original launch flags (copied from
# e14_n2_stretched_100k.sbatch / e15_n2_stretched_seedcheck.sbatch).
#
# COST. ~0.6 s/step on a 2080 Ti; the runs stop at their NaN (check_for_nans) or at
# 16000. ~1 GPU-h each. Disk: one checkpoint per 50 epochs (params + walkers + the
# SS-SPRING state, ~4 parameter copies) -- a few GB per run on scratch.
#
# Submit from the vmcnet repo root on a Savio login node:
#     sbatch slurm/h1_n2_dense_replay.sbatch
#
#SBATCH --job-name=h1-n2-dense-replay
#SBATCH --account=co_esmath
#SBATCH --partition=savio3_gpu
#SBATCH --qos=esmath_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --cpus-per-task=2
#SBATCH --time=06:00:00
#SBATCH --array=0-1
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --exclude=n0135.savio3

set -eo pipefail
export PATH="/global/scratch/users/$USER/envs/vmcnet/bin:$PATH"
set -u

export WANDB_MODE=offline
export WANDB_DIR="/global/scratch/users/$USER/wandb"
mkdir -p "$WANDB_DIR"

cd "$SLURM_SUBMIT_DIR"
PHASE_E="/global/scratch/users/$USER/vmcnet_logs/phase_e"
LOGROOT="/global/scratch/users/$USER/vmcnet_logs/phase_h/h1_dense_replay"
mkdir -p "$LOGROOT"

SSU=same_sampled_spring_unified
PRESET=N2_4.0
ETA=0.002

case "$SLURM_ARRAY_TASK_ID" in
  0) TAG=e14s0; SEED=0
     OLD="$PHASE_E/e14_n2_stretched_100k/e14_N2_4.0_ssu_defaults_s0" ;;
  1) TAG=e15s7; SEED=7
     OLD="$PHASE_E/e15_n2_seedcheck/e15_N2_4.0_ssu_defaults_s7" ;;
esac

RUN_NAME="h1_${PRESET}_${TAG}_dense"

if [[ ! -f "$OLD/checkpoints/10000.npz" ]]; then
  echo "ERROR: $OLD/checkpoints/10000.npz not found." >&2
  exit 1
fi
if [[ -e "$LOGROOT/$RUN_NAME" ]]; then
  echo "ERROR: $LOGROOT/$RUN_NAME already exists; move or delete it first." >&2
  exit 1
fi

export WANDB_TAGS="h1,phase-h,replay,dense-ckpt,stretched,${TAG},sys-${PRESET}"
export WANDB_NOTES="H1 step 1: replay of ${TAG} from its epoch-10000 checkpoint with a \
checkpoint every 50 epochs until it diverges, to supply frozen-parameter snapshots for \
the Kaczmarz++ probe test. Original SS-SPRING code. Original run: ${OLD}."

python -c "import jax; print('jax devices:', jax.devices())"
echo "H1 replay case=$TAG eta=$ETA seed=$SEED from $OLD  $(date -Is)"

vmc-molecule \
  --reload.logdir="$OLD" \
  --reload.use_config_file=False \
  --reload.use_checkpoint_file=True \
  --reload.checkpoint_relative_file_path=checkpoints/10000.npz \
  --reload.append=True \
  --presets.path="preset_configs/${PRESET}.json" \
  --config.logdir="$LOGROOT" \
  --config.save_to_current_datetime_subfolder=False \
  --config.subfolder_name="$RUN_NAME" \
  --config.vmc.optimizer_type="$SSU" \
  --config.vmc.optimizer."$SSU".learning_rate="$ETA" \
  --config.vmc.optimizer."$SSU".mu=0.0 \
  --config.vmc.optimizer."$SSU".schedule_type=inverse_time \
  --config.vmc.optimizer."$SSU".learning_decay_rate=0.0001 \
  --config.vmc.optimizer."$SSU".constrain_norm=True \
  --config.vmc.optimizer."$SSU".norm_constraint=0.001 \
  --config.vmc.nepochs=16000 \
  --config.vmc.checkpoint_every=50 \
  --config.vmc.check_for_nans=True \
  --config.eval.nchains=1000 \
  --config.eval.nburn=0 \
  --config.eval.nepochs=0 \
  --config.initial_seed="$SEED" \
  --config.wandb.mode=offline \
  --config.wandb.project=vmcnet-phase-h \
  --config.wandb.group="h1-n2-dense-replay" \
  --config.wandb.name="$RUN_NAME"

echo "H1 replay done $(date -Is)"
