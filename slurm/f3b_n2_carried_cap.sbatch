#!/bin/bash
# F3b -- does capping the CARRIED momentum prevent the stretched-N2 catastrophe?
# Plan and reasoning: docs/CAMPAIGN_LOG.md, Section 5 "Phase F" (PRECURSOR RESULT).
#
# WHAT THE CAP DOES. On every step SS-SPRING already computes A(beta*phi), the change in
# log|psi| the carried momentum predicts on this step's (fresh) walkers. With
# carried_cap=K the momentum entering the step is shrunk so that
# ||A(beta*phi)|| <= K * ||eps||. It is free, dimensionless and acts BEFORE the
# catastrophe: in the F2 replays carried/eps sat at 2-4 in healthy stretches and rose
# from ~2 to 36 to 286 over the ~70 steps before E14 s1's catastrophe, and ran at
# 78-2e4 for thousands of steps in E18 s2's slow failure.
#
# WHY K=10 (from `f2_replay_summary.py --carried`). Healthy rows (two safeguard-on
# replays, 10k rows each): median 2.7-3.0, p99.9 8.6-11.3, max 23-33.5; K=10 is exceeded
# on 9-15 of 10,000 healthy rows (0.1%), where the soft cap only trims one step's
# momentum. Every failing run crossed 10 well before its catastrophe.
#
# DESIGN. Six failing states, each replayed from its last checkpoint before its first
# catastrophe, once without and once with the cap (array index = 2*case + arm):
#   case 0  E14 s0  eta 0.002   from 10000 (first event 13,980 in the original)
#   case 1  E14 s1  eta 0.002   from 10000 (10,590; 10,470 in the F2 off-replay)
#   case 2  E14 s2  eta 0.002   from 10000 (14,270)
#   case 3  E15 s7  eta 0.002   from 10000 (11,250)
#   case 4  E15 s6  eta 0.002   from 60000 (61,190)
#   case 5  E18 s2  eta 0.0015  from 10000 (slow failure; sick by ~10,150 in F2)
#   arm 0 = diagnostics only (baseline hazard from this state)
#   arm 1 = diagnostics + carried_cap=10
# Safeguard off in both arms, so the cap is tested alone. Each replay is a NEW trajectory
# from the checkpointed state (the extra diagnostic ops change float32 rounding; see the
# F2 RESULT), so read the result as a rate over six states: how many of the six baseline
# replays fail vs how many capped ones do.
#
# WHAT SUCCESS LOOKS LIKE. Capped arms reach their end epoch with energy at or below the
# pre-replay level, while most baseline arms fail; the cap active on a small fraction of
# rows except during the approach to what would have been a catastrophe. If capped arms
# still fail, look at whether carried stayed at the cap (the cap is binding but not
# enough) or the failure came through another channel (carried never reached K).
#
# HOW THE RELOAD WORKS: as in f2_n2_replay.sbatch -- config rebuilt from the preset +
# the original launch flags (reloaded config.json is locked), only the checkpoint
# (params, walkers, optimizer state, RNG key) is reloaded.
#
# COST. ~0.6 s/step on a 2080 Ti, 10k steps per arm => ~2 h each, 12 arms => ~20-24
# GPU-h at most (failing baseline arms stop early).
#
# Submit from the vmcnet repo root on a Savio login node (code must include the Phase F
# changes):
#     sbatch slurm/f3b_n2_carried_cap.sbatch
# READOUT, afterwards, on the login node:
#     python slurm/f2_replay_summary.py --carried \
#         --logroot /global/scratch/users/$USER/vmcnet_logs/phase_f/f3b_carried_cap
#
#SBATCH --job-name=f3b-n2-carried-cap
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
LOGROOT="/global/scratch/users/$USER/vmcnet_logs/phase_f/f3b_carried_cap"
mkdir -p "$LOGROOT"

SSU=same_sampled_spring_unified
PRESET=N2_4.0
CASE=$((SLURM_ARRAY_TASK_ID / 2))
ARM=$((SLURM_ARRAY_TASK_ID % 2))
CAP_K=10

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
  0) ARMTAG=nocap; CAP=0.0 ;;
  1) ARMTAG=cap${CAP_K}; CAP=$CAP_K ;;
esac
TO=$((FROM + 10000))
RUN_NAME="f3b_${PRESET}_${TAG}_${ARMTAG}"

if [[ ! -f "$OLD/checkpoints/${FROM}.npz" ]]; then
  echo "ERROR: $OLD/checkpoints/${FROM}.npz not found." >&2
  exit 1
fi
if [[ -e "$LOGROOT/$RUN_NAME" ]]; then
  echo "ERROR: $LOGROOT/$RUN_NAME already exists; move or delete it first." >&2
  exit 1
fi

export WANDB_TAGS="f3b,phase-f,replay,carried-cap,stretched,${TAG},${ARMTAG},sys-${PRESET}"
export WANDB_NOTES="Phase F F3b replay of ${TAG} from its epoch-${FROM} checkpoint to \
${TO}: diagnostics on, safeguard off, carried_cap=${CAP}. Original run: ${OLD}."

python -c "import jax; print('jax devices:', jax.devices())"
echo "F3b case=$TAG arm=$ARMTAG cap=$CAP eta=$ETA from $FROM to $TO  $(date -Is)"

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
  --config.wandb.group="f3b-n2-carried-cap" \
  --config.wandb.name="$RUN_NAME"

echo "F3b done $(date -Is)"
