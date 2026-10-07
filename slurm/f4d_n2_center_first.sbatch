#!/bin/bash
# F4d -- does centring the Gram rows BEFORE the contraction stop the stretched-N2 failure?
# Plan and reasoning: docs/CAMPAIGN_LOG.md, Section 5 "Phase F" ([4b] RESULT).
#
# WHY. Every SR optimizer here forms the walker Gram from the UNCENTERED NTK and
# subtracts the means afterwards. On the E15 s6 60000 checkpoint the mean gradient is
# ~15-20x the centred part, so that subtraction cancels float32 digits: T has 263
# negative eigenvalues (min -0.0057, 5.6x the damping) and the step is off by 4.6 of its
# 6.0 against float64. Centring the rows first, still in float32, matched float64 to ~1%
# (slurm/f4_one_step_check.py, section 4b). gram_center_first=True does that
# (vmcnet/updates/sr_kernel.py).
#
# DESIGN. The same six failing states as F3b/F3c, ONE arm each: diagnostics on,
# gram_center_first=True, NO carried cap, NO safeguard -- the numerical fix alone.
#   0 E14 s0 (10k)  1 E14 s1 (10k)  2 E14 s2 (10k)  3 E15 s7 (10k)  4 E15 s6 (60k)
#   5 E18 s2 (10k, eta 0.0015)
# Baseline = the F3b no-cap arms already on disk: 5 of 6 failed (only E18 s2 survived).
# Replays are new trajectories, so read it as a rate over six.
#
# BEFORE SUBMITTING (cheap, recommended): rerun the one-step check; section [1]'s
# "F4d gram_center_first" row and 4b's "F4d kernel" row should match "rows f64":
#     python slurm/f4_one_step_check.py --dtype float32 --nperm 1
#
# WHAT SUCCESS LOOKS LIKE. Most of the six reach their end epoch at or below the
# pre-replay energy, with carried/eps staying near its healthy 2-4 and no NaN. If they
# still fail, the float32 error was real but not the driver -- compare onset anatomy
# with F3b's no-cap arms.
#
# COST. ~0.6-0.7 s/step on a 2080 Ti (one extra backward pass per step for the mean
# gradient), 10k steps per run => ~2 h each, 6 runs => ~12-14 GPU-h.
#
# Submit from the vmcnet repo root on a Savio login node (code must include F4d):
#     sbatch slurm/f4d_n2_center_first.sbatch
# READOUT, afterwards, on the login node:
#     python slurm/f2_replay_summary.py --carried \
#         --logroot /global/scratch/users/$USER/vmcnet_logs/phase_f/f4d_center_first
#
#SBATCH --job-name=f4d-n2-center-first
#SBATCH --account=co_esmath
#SBATCH --partition=savio3_gpu
#SBATCH --qos=esmath_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --cpus-per-task=2
#SBATCH --time=05:00:00
#SBATCH --array=0-5
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
LOGROOT="/global/scratch/users/$USER/vmcnet_logs/phase_f/f4d_center_first"
mkdir -p "$LOGROOT"

SSU=same_sampled_spring_unified
PRESET=N2_4.0
CASE=$SLURM_ARRAY_TASK_ID

case "$CASE" in
  0) TAG=e14s0; SEED=0; ETA=0.002;  FROM=10000
     OLD="$PHASE_E/e14_n2_stretched_100k/e14_N2_4.0_ssu_defaults_s0" ;;
  1) TAG=e14s1; SEED=1; ETA=0.002;  FROM=10000
     OLD="$PHASE_E/e14_n2_stretched_100k/e14_N2_4.0_ssu_defaults_s1" ;;
  2) TAG=e14s2; SEED=2; ETA=0.002;  FROM=10000
     OLD="$PHASE_E/e14_n2_stretched_100k/e14_N2_4.0_ssu_defaults_s2" ;;
  3) TAG=e15s7; SEED=7; ETA=0.002;  FROM=10000
     OLD="$PHASE_E/e15_n2_seedcheck/e15_N2_4.0_ssu_defaults_s7" ;;
  4) TAG=e15s6; SEED=6; ETA=0.002;  FROM=60000
     OLD="$PHASE_E/e15_n2_seedcheck/e15_N2_4.0_ssu_defaults_s6" ;;
  5) TAG=e18s2; SEED=2; ETA=0.0015; FROM=10000
     OLD="$PHASE_E/e18_n2_stretched_eta0015/e18_N2_4.0_ssu_defaults_eta0.0015_s2" ;;
esac
ARMTAG=cfirst
TO=$((FROM + 10000))
RUN_NAME="f4d_${PRESET}_${TAG}_${ARMTAG}"

if [[ ! -f "$OLD/checkpoints/${FROM}.npz" ]]; then
  echo "ERROR: $OLD/checkpoints/${FROM}.npz not found." >&2
  exit 1
fi
if [[ -e "$LOGROOT/$RUN_NAME" ]]; then
  echo "ERROR: $LOGROOT/$RUN_NAME already exists; move or delete it first." >&2
  exit 1
fi

export WANDB_TAGS="f4d,phase-f,replay,center-first,stretched,${TAG},${ARMTAG},sys-${PRESET}"
export WANDB_NOTES="Phase F F4d replay of ${TAG} from its epoch-${FROM} checkpoint to \
${TO}: gram_center_first=True, diagnostics on, no cap, no safeguard. \
Original run: ${OLD}."

python -c "import jax; print('jax devices:', jax.devices())"
echo "F4d case=$TAG arm=$ARMTAG eta=$ETA from $FROM to $TO  $(date -Is)"

vmc-molecule \
  --reload.logdir="$OLD" \
  --reload.use_config_file=False \
  --reload.use_checkpoint_file=True \
  --reload.checkpoint_relative_file_path="checkpoints/${FROM}.npz" \
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
  --config.vmc.optimizer."$SSU".safeguard=False \
  --config.vmc.optimizer."$SSU".carried_cap=0.0 \
  --config.vmc.optimizer."$SSU".gram_center_first=True \
  --config.vmc.nepochs="$TO" \
  --config.vmc.checkpoint_every=2000 \
  --config.vmc.check_for_nans=True \
  --config.eval.nchains=1000 \
  --config.eval.nburn=0 \
  --config.eval.nepochs=0 \
  --config.initial_seed="$SEED" \
  --config.wandb.mode=offline \
  --config.wandb.project=vmcnet-phase-f \
  --config.wandb.group="f4d-n2-center-first" \
  --config.wandb.name="$RUN_NAME"

echo "F4d done $(date -Is)"
