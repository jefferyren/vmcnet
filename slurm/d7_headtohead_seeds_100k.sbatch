#!/bin/bash
# D7 — E7 re-run at 100k epochs.   ANCESTOR: e7_headtohead_seeds.sbatch (50k).
#
# WHY PHASE D EXISTS. Phase B/C ran 50k training epochs. The SPRING paper and the
# PRIME-SR paper both report at 100k, and E10 showed N2/CO are demonstrably not
# converged at 50k (every arm still improving 0.1-1.0 mHa per 5k, absolute errors
# 9-20 mHa). Phase D re-runs the four experiments that produce the paper's tables at
# 100k so the whole results section sits on one protocol that matches both papers.
#
# WHAT CHANGED FROM E7, AND WHAT DID NOT.
#   CHANGED:  nepochs 50000 -> 100000 (which is what preset_configs/*.json already say;
#             E7 was overriding them DOWN). Walltime 20h -> 24h. New logroot, new wandb
#             project, new job name.
#   EXPLICIT: constrain_norm / norm_constraint / schedule_type / learning_decay_rate are
#             now passed on the command line. They were already at these values in every
#             Phase A-C run via the defaults -- this only makes them auditable in the
#             .out config dump instead of implicit.
#   UNCHANGED: arms, array indexing, seeds, eta, walker counts, eval phase, random init.
#             D7 index i means exactly what E7 index i meant, so the two are directly
#             comparable cell by cell.
#
# WHY THE JOB NAME IS d7- AND NOT e7-. slurm/parse_eval_energies.py globs on job name
# (`slurm-e7-...-*_{idx}.out`). Anything starting `e7-` would be swept into the 50k
# results and silently averaged with them. Do not rename these back.
#
# THE ARMS (4), each x {carbon, H4} x 5 seeds = 40 runs:
#   0  SPRING mu=0.99      the published default
#   1  SPRING mu=tuned     E0's per-system grid pick (0.995 carbon / 0.8 H4)
#   2  PRIME-SR            claim B comparator
#   3  SS-SPRING           untuned
#
# KEEP ARM 1 AT mu=0.8 ON H4 even though E7 showed it is the worst arm at 50k. That
# result -- a single-seed 25k grid mis-ranking momentum -- is cited in the paper, and
# dropping the arm removes the evidence for it. If it stops being the worst arm at 100k
# that is itself worth knowing, and the claim becomes a fixed-budget one.
#
# Array index = arm*10 + system*5 + seed.
# ~7-8 h per run => ~300 GPU-h.
#
# Submit from the vmcnet repo root:  sbatch slurm/d7_headtohead_seeds_100k.sbatch
# Then sync from a LOGIN node:       bash slurm/sync_wandb.sh
# Then, locally:                     python slurm/parse_eval_energies.py --experiment D7 --backfill
#   (that registry entry does not exist yet -- add it before the runs land.)
#
#SBATCH --job-name=d7-headtohead
#SBATCH --account=co_esmath
#SBATCH --partition=savio3_gpu
#SBATCH --qos=esmath_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --array=0-39
#SBATCH --output=slurm-%x-%A_%a.out

set -eo pipefail
export PATH="/global/scratch/users/$USER/envs/vmcnet/bin:$PATH"
set -u

export WANDB_MODE=offline
export WANDB_DIR="/global/scratch/users/$USER/wandb"
mkdir -p "$WANDB_DIR"

cd "$SLURM_SUBMIT_DIR"
LOGROOT="/global/scratch/users/$USER/vmcnet_logs/phase_d/d7_headtohead_100k"
mkdir -p "$LOGROOT"

SSU=same_sampled_spring_unified
ARM=$((SLURM_ARRAY_TASK_ID / 10))
SYS_I=$(((SLURM_ARRAY_TASK_ID % 10) / 5))
SEED=$((SLURM_ARRAY_TASK_ID % 5))
PRESETS=(carbon H4)
PRESET=${PRESETS[$SYS_I]}

# E0's per-system grid picks, carried over from E7 unchanged so the cells still match.
TUNED_MUS=(0.995 0.8)
TUNED_MU=${TUNED_MUS[$SYS_I]}

case "$ARM" in
  0) TAG=spring_default;  OPT=spring
     ARM_FLAGS=(--config.vmc.optimizer.spring.mu=0.99) ;;
  1) TAG=spring_tuned;    OPT=spring
     ARM_FLAGS=(--config.vmc.optimizer.spring.mu="$TUNED_MU") ;;
  2) TAG=prime_sr;        OPT=prime_sr
     ARM_FLAGS=() ;;
  3) TAG=ssu_defaults;    OPT=$SSU
     ARM_FLAGS=(--config.vmc.optimizer."$SSU".mu=0.0) ;;
esac

RUN_NAME="d7_${PRESET}_${TAG}_s${SEED}"

export WANDB_TAGS="d7,phase-d,100k,head-to-head,seeded,rerun-of-e7,arm-${TAG},sys-${PRESET}"
export WANDB_NOTES="D7 = E7 at 100k epochs, the iteration count both the SPRING and \
PRIME-SR papers report at. Same arms, same indexing, same seeds, same eta=0.02 as E7 \
-- only nepochs changed, so D7 cell i is directly comparable to E7 cell i. E7 at 50k \
gave SS-SPRING +0.148+/-0.004 mHa on carbon against SPRING(0.99) +0.147+/-0.012 and \
PRIME-SR +0.217+/-0.009. Score mean +/- s.e.m., worst seed and spread, not the mean \
alone. Arm: ${TAG} on ${PRESET}, tuned mu for this system = ${TUNED_MU}."

if [[ -e "$LOGROOT/$RUN_NAME" ]]; then
  echo "ERROR: $LOGROOT/$RUN_NAME already exists; move or delete it first." >&2
  exit 1
fi

python -c "import jax; print('jax devices:', jax.devices())"
echo "D7 arm=$TAG preset=$PRESET seed=$SEED tuned_mu=$TUNED_MU nepochs=100000  $(date -Is)"

vmc-molecule \
  --presets.path="preset_configs/${PRESET}.json" \
  --config.logdir="$LOGROOT" \
  --config.save_to_current_datetime_subfolder=False \
  --config.subfolder_name="$RUN_NAME" \
  --config.vmc.optimizer_type="$OPT" \
  --config.vmc.optimizer."$OPT".learning_rate=0.02 \
  --config.vmc.optimizer."$OPT".schedule_type=inverse_time \
  --config.vmc.optimizer."$OPT".learning_decay_rate=0.0001 \
  --config.vmc.optimizer."$OPT".constrain_norm=True \
  --config.vmc.optimizer."$OPT".norm_constraint=0.001 \
  "${ARM_FLAGS[@]}" \
  --config.vmc.nepochs=100000 \
  --config.vmc.checkpoint_every=10000 \
  --config.vmc.check_for_nans=True \
  --config.eval.nchains=2000 \
  --config.eval.nburn=5000 \
  --config.eval.nepochs=20000 \
  --config.initial_seed="$SEED" \
  --config.wandb.mode=offline \
  --config.wandb.project=vmcnet-phase-d \
  --config.wandb.group="d7-headtohead-${PRESET}" \
  --config.wandb.name="$RUN_NAME"

echo "D7 done $(date -Is)"
