#!/bin/bash
# F3c -- does REWINDING when the carried cap keeps binding stop the stretched-N2 failure?
# Plan and reasoning: docs/CAMPAIGN_LOG.md, Section 5 "Phase F" (F3b RESULT).
#
# WHY. F3b (carried_cap=10 alone) did not prevent the catastrophe: 4 of 5 readable
# capped replays failed vs 5 of 6 uncapped. But the cap turned a ~50-row explosive
# approach into a 150-1,500-row slow drift, and the capped runs then died through the
# probe (probe residual overflow -> r_ip = inf/inf = NaN -> beta NaN). So the cap is a
# good ALARM with lead time, and the probe needs its own guard.
#
# WHAT THIS RUN ADDS (safeguard=True, new keys in same_sampled_spring_unified.py):
#   - safeguard_carried_window / _count / _k: once diag_carried_over_eps > K on
#     COUNT of the last WINDOW steps, rewind at once (no skip first): parameters and
#     probe iterate go back to a snapshot that predates the episode (no snapshots are
#     taken while the window holds a hit), momentum buffers are zeroed, and beta is
#     held <= 0.99 for 2000 steps.
#   - a non-finite probe residual also rewinds at once, and a non-finite probe ratio
#     can no longer reach r_hat/beta (it enters as 1, like any growing residual).
#   - the F3 triggers (pre-clip step > 3x trailing median, bound ratio > 3) stay on.
#
# DESIGN. The same six failing states as F3b (array index = 2*case + arm):
#   case 0  E14 s0  eta 0.002   from 10000      case 3  E15 s7  eta 0.002   from 10000
#   case 1  E14 s1  eta 0.002   from 10000      case 4  E15 s6  eta 0.002   from 60000
#   case 2  E14 s2  eta 0.002   from 10000      case 5  E18 s2  eta 0.0015  from 10000
#   arm 0 = safeguard + carried trigger, NO cap  (is the alarm + rewind enough alone?)
#   arm 1 = safeguard + carried trigger + carried_cap=10  (does trimming add anything?)
# Baselines are the F3b arms already on disk: no cap 5/6 fail, cap alone 4/5 fail.
# Replays are new trajectories (F2 RESULT), so read the result as a rate over six.
# Case 4's F3b cap arm was anomalous (63 rows); check its .out before trusting case 4.
#
# CALIBRATE WINDOW/COUNT FIRST (zero GPU), on the F3b logdirs:
#     python slurm/f2_replay_summary.py --sustained \
#         --logroot /global/scratch/users/$USER/vmcnet_logs/phase_f/f3b_carried_cap
# and on the F2 safeguard-on replays (10k healthy rows each, the false-alarm check):
#     python slurm/f2_replay_summary.py --sustained
# Pick the (W, M) with ~0 healthy firings and the longest lead, then pass it:
#     sbatch --export=ALL,SG_W=50,SG_M=5 slurm/f3c_n2_cap_rewind.sbatch
# Defaults (W=50, M=5) assume the healthy >K rows are isolated at ~0.1-0.5%.
#
# WHAT SUCCESS LOOKS LIKE. Most arms reach their end epoch with energy at or below the
# pre-replay level, with a few carried-triggered rewinds each (sg_trigger_carried) and
# no NaN. If arms still fail after rewinding, the parameters degrade faster than a
# rewind can undo, and the next step is per-walker gradient clipping (F3b RESULT).
#
# COST. ~0.6 s/step on a 2080 Ti, 10k steps per arm => ~2 h each, 12 arms => ~20-24
# GPU-h at most.
#
# Submit from the vmcnet repo root on a Savio login node (code must include F3c):
#     sbatch --export=ALL,SG_W=50,SG_M=5 slurm/f3c_n2_cap_rewind.sbatch
# READOUT, afterwards, on the login node:
#     python slurm/f2_replay_summary.py --carried \
#         --logroot /global/scratch/users/$USER/vmcnet_logs/phase_f/f3c_cap_rewind
#
#SBATCH --job-name=f3c-n2-cap-rewind
#SBATCH --account=co_esmath
#SBATCH --partition=savio3_gpu
#SBATCH --qos=esmath_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --cpus-per-task=2
#SBATCH --time=05:00:00
#SBATCH --array=0-11
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
LOGROOT="/global/scratch/users/$USER/vmcnet_logs/phase_f/f3c_cap_rewind"
mkdir -p "$LOGROOT"

SSU=same_sampled_spring_unified
PRESET=N2_4.0
CASE=$((SLURM_ARRAY_TASK_ID / 2))
ARM=$((SLURM_ARRAY_TASK_ID % 2))
CAP_K=10
SG_K=10
SG_W="${SG_W:-50}"
SG_M="${SG_M:-5}"

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
case "$ARM" in
  0) ARMTAG=rw; CAP=0.0 ;;
  1) ARMTAG=rw_cap${CAP_K}; CAP=$CAP_K ;;
esac
TO=$((FROM + 10000))
RUN_NAME="f3c_${PRESET}_${TAG}_${ARMTAG}"

if [[ ! -f "$OLD/checkpoints/${FROM}.npz" ]]; then
  echo "ERROR: $OLD/checkpoints/${FROM}.npz not found." >&2
  exit 1
fi
if [[ -e "$LOGROOT/$RUN_NAME" ]]; then
  echo "ERROR: $LOGROOT/$RUN_NAME already exists; move or delete it first." >&2
  exit 1
fi

export WANDB_TAGS="f3c,phase-f,replay,cap-rewind,stretched,${TAG},${ARMTAG},sys-${PRESET}"
export WANDB_NOTES="Phase F F3c replay of ${TAG} from its epoch-${FROM} checkpoint to \
${TO}: safeguard on with carried trigger (K=${SG_K}, ${SG_M} of ${SG_W} steps), \
carried_cap=${CAP}. Original run: ${OLD}."

python -c "import jax; print('jax devices:', jax.devices())"
echo "F3c case=$TAG arm=$ARMTAG cap=$CAP trigger=${SG_M}/${SG_W}>K=${SG_K} eta=$ETA" \
  "from $FROM to $TO  $(date -Is)"

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
  --config.vmc.optimizer."$SSU".safeguard=True \
  --config.vmc.optimizer."$SSU".safeguard_carried_k="$SG_K" \
  --config.vmc.optimizer."$SSU".safeguard_carried_window="$SG_W" \
  --config.vmc.optimizer."$SSU".safeguard_carried_count="$SG_M" \
  --config.vmc.optimizer."$SSU".carried_cap="$CAP" \
  --config.vmc.nepochs="$TO" \
  --config.vmc.checkpoint_every=2000 \
  --config.vmc.check_for_nans=True \
  --config.eval.nchains=1000 \
  --config.eval.nburn=0 \
  --config.eval.nepochs=0 \
  --config.initial_seed="$SEED" \
  --config.wandb.mode=offline \
  --config.wandb.project=vmcnet-phase-f \
  --config.wandb.group="f3c-n2-cap-rewind" \
  --config.wandb.name="$RUN_NAME"

echo "F3c done $(date -Is)"
