#!/bin/bash
# F2/F3 -- replay two failing stretched-N2 SS-SPRING runs from their 10k checkpoints,
# with the Phase F diagnostics on, once WITHOUT and once WITH the safeguard.
# Plan and reasoning: docs/CAMPAIGN_LOG.md, Section 5 "Phase F".
#
# WHY THESE TWO RUNS.
#   E14 s1 (eta=0.002):  sharp first catastrophe at epoch 10,590, NaN at 17,341. Its
#                        10k checkpoint is 590 steps before the event.
#   E18 s2 (eta=0.0015): the slow failure -- no sharp event, above SPRING's exact bound
#                        for 1,673 steps from epoch 7,366, NaN at 15,691.
#
# ARMS (array index = 2*case + arm):
#   0  E14 s1, safeguard off     1  E14 s1, safeguard on
#   2  E18 s2, safeguard off     3  E18 s2, safeguard on
# All four log the diag_* metrics (diagnostics=True). Safeguard-on also logs sg_*.
#
# WHAT EACH ARM ANSWERS.
#   off: per-epoch anatomy of the onset -- does the equation residual / Gram lambda_max
#        / walker row-norm tail / carried-momentum size spike before the energy and
#        variance, or with them? Expect the original failure to recur; FIRST check it
#        reproduces the original trajectory (see READOUT): the diagnostics add ops and
#        can change float32 rounding, after which the trajectory may wander.
#   on:  does detect -> skip -> rewind + momentum reset turn the NaN into a recovery?
#        Success = reaches epoch 20,000 with energy back near its pre-event level.
#        Report the trigger and rewind counts either way.
#
# HOW THE RELOAD WORKS (and why it does not reuse config.json). --reload.use_config_file
# =False rebuilds the config from the preset + the ORIGINAL launch flags (copied from
# e14_n2_stretched_100k.sbatch / e18_n2_stretched_eta0015.sbatch), because a reloaded
# config.json is locked and cannot accept the new diagnostics/safeguard keys. Only the
# checkpoint is reloaded: params, walkers, optimizer state and RNG key at epoch 10000.
# reload.append=True copies the original .txt metrics up to epoch 10000 into the new
# logdir, so each replay logdir holds a full 0..20000 history (diag_*/sg_* from 10000).
#
# COST. ~0.6 s/step on a 2080 Ti => 10k steps ~2 h per arm; the safeguard-off arms will
# likely stop early at their NaN. Eval is skipped. ~6-8 GPU-h total at most.
#
# Submit from the vmcnet repo root on a Savio login node (code must include the Phase F
# changes to same_sampled_spring_unified.py and default_config.py):
#     sbatch slurm/f2_n2_replay.sbatch
# READOUT, afterwards, on the login node:
#     python slurm/f2_replay_summary.py
#
#SBATCH --job-name=f2-n2-replay
#SBATCH --account=co_esmath
#SBATCH --partition=savio3_gpu
#SBATCH --qos=esmath_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --cpus-per-task=2
#SBATCH --time=05:00:00
#SBATCH --array=0-3
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
LOGROOT="/global/scratch/users/$USER/vmcnet_logs/phase_f/f2_replay"
mkdir -p "$LOGROOT"

SSU=same_sampled_spring_unified
PRESET=N2_4.0
CASE=$((SLURM_ARRAY_TASK_ID / 2))
SG=$((SLURM_ARRAY_TASK_ID % 2))

case "$CASE" in
  0) TAG=e14s1; SEED=1; ETA=0.002
     OLD="$PHASE_E/e14_n2_stretched_100k/e14_N2_4.0_ssu_defaults_s1" ;;
  1) TAG=e18s2; SEED=2; ETA=0.0015
     OLD="$PHASE_E/e18_n2_stretched_eta0015/e18_N2_4.0_ssu_defaults_eta0.0015_s2" ;;
esac
case "$SG" in
  0) SGTAG=sgoff; SAFEGUARD=False ;;
  1) SGTAG=sgon;  SAFEGUARD=True ;;
esac

RUN_NAME="f2_${PRESET}_${TAG}_${SGTAG}"

if [[ ! -f "$OLD/checkpoints/10000.npz" ]]; then
  echo "ERROR: $OLD/checkpoints/10000.npz not found." >&2
  exit 1
fi
if [[ -e "$LOGROOT/$RUN_NAME" ]]; then
  echo "ERROR: $LOGROOT/$RUN_NAME already exists; move or delete it first." >&2
  exit 1
fi

export WANDB_TAGS="f2,phase-f,replay,stretched,${TAG},${SGTAG},sys-${PRESET}"
export WANDB_NOTES="Phase F F2/F3 replay of ${TAG} from its epoch-10000 checkpoint to \
20000 with diagnostics on and safeguard ${SGTAG}. Original run: ${OLD}."

python -c "import jax; print('jax devices:', jax.devices())"
echo "F2 case=$TAG safeguard=$SAFEGUARD eta=$ETA seed=$SEED from $OLD  $(date -Is)"

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
  --config.vmc.optimizer."$SSU".diagnostics=True \
  --config.vmc.optimizer."$SSU".safeguard="$SAFEGUARD" \
  --config.vmc.nepochs=20000 \
  --config.vmc.checkpoint_every=2000 \
  --config.vmc.check_for_nans=True \
  --config.eval.nchains=1000 \
  --config.eval.nburn=0 \
  --config.eval.nepochs=0 \
  --config.initial_seed="$SEED" \
  --config.wandb.mode=offline \
  --config.wandb.project=vmcnet-phase-f \
  --config.wandb.group="f2-n2-replay" \
  --config.wandb.name="$RUN_NAME"

echo "F2 done $(date -Is)"
