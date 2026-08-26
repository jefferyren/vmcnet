#!/bin/bash
# D9 — E9 re-run at 100k epochs.   ANCESTOR: e9_atoms_headtohead.sbatch (50k).
#
# WHY PHASE D EXISTS. See the header of d7_headtohead_seeds_100k.sbatch. In short: both
# comparison papers report at 100k, and 50k left E10 visibly unconverged, so the whole
# results section moves onto one protocol.
#
# WHAT CHANGED FROM E9, AND WHAT DID NOT.
#   CHANGED:  nepochs 50000 -> 100000. Walltime 20h -> 24h. New logroot, wandb project,
#             job name.
#   EXPLICIT: constrain_norm / norm_constraint / schedule_type / learning_decay_rate now
#             passed on the command line. Same values every Phase A-C run already used.
#   UNCHANGED: arms, indexing, seeds, eta=0.02, 1000 walkers, eval phase, random init.
#             D9 index i == E9 index i, cell for cell.
#
# WHY d9- AND NOT e9-. parse_eval_energies.py globs `slurm-e9-atoms-*_{idx}.out`. A job
# named e9-atoms-100k would match that glob and be merged into the 50k results without
# any error. Do not rename.
#
# THE ARMS (6), each x {N, O} x 5 seeds = 60 runs:
#   0  SPRING mu=0.95     |
#   1  SPRING mu=0.99     |  best of these FOUR = the tuned baseline for claim A
#   2  SPRING mu=0.995    |
#   5  SPRING mu=0.999    |  (added in D9; see below)
#   3  PRIME-SR              claim B comparator
#   4  SS-SPRING             untuned
#
# WHY mu=0.999 WAS ADDED, AND WHY IT CAN ONLY HURT US. At 50k the fixed-mu ordering on
# N was MONOTONE in mu (0.268 -> 0.214 -> 0.197 mHa for 0.95 -> 0.99 -> 0.995), so the
# optimum sat at the EDGE of E9's grid. "Tuned SPRING" there was not a tuned optimum, it
# was a censored one, and that flatters claim A on N. The same censoring holds on N2 and
# CO (D10) and on carbon at eta=0.005 (D11) -- four of the seven conditions in the paper.
# A stronger fixed-mu baseline can only narrow or flip our claim-A ties; that is exactly
# the reason to run it rather than to leave it out. It does not touch claim B: PRIME-SR
# settles near mu=0.953 regardless.
# Two outcomes are both informative. At eta=0.02, eta/(1-mu) = 20 at mu=0.999 against
# 4.0 at 0.995, so this arm may simply destabilize -- the PRIME-SR paper claims exactly
# that at mu=0.99 and E10 already contradicted it once. Watch the mu trace and
# check_for_nans, and do not silently drop a diverged seed.
# Arm 5 is APPENDED so indices 0-49 keep the meaning they had in E9 and in earlier D9
# submissions. Do not renumber.
#
# Array index = arm*10 + system*5 + seed.
# ~7-8 h per run => ~450 GPU-h.
#
# Submit from the vmcnet repo root:  sbatch slurm/d9_atoms_headtohead_100k.sbatch
# Then sync from a LOGIN node:       bash slurm/sync_wandb.sh
# Then, locally:                     python slurm/parse_eval_energies.py --experiment D9 --backfill
#   (that registry entry does not exist yet -- add it before the runs land.)
#
#SBATCH --job-name=d9-atoms
#SBATCH --account=co_esmath
#SBATCH --partition=savio3_gpu
#SBATCH --qos=esmath_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --array=0-59
#SBATCH --output=slurm-%x-%A_%a.out

set -eo pipefail
export PATH="/global/scratch/users/$USER/envs/vmcnet/bin:$PATH"
set -u

export WANDB_MODE=offline
export WANDB_DIR="/global/scratch/users/$USER/wandb"
mkdir -p "$WANDB_DIR"

cd "$SLURM_SUBMIT_DIR"
LOGROOT="/global/scratch/users/$USER/vmcnet_logs/phase_d/d9_atoms_100k"
mkdir -p "$LOGROOT"

SSU=same_sampled_spring_unified
ARM=$((SLURM_ARRAY_TASK_ID / 10))
SYS_I=$(((SLURM_ARRAY_TASK_ID % 10) / 5))
SEED=$((SLURM_ARRAY_TASK_ID % 5))
PRESETS=(N O)
PRESET=${PRESETS[$SYS_I]}

case "$ARM" in
  0) TAG=spring_mu0.95;  OPT=spring;   ARM_FLAGS=(--config.vmc.optimizer.spring.mu=0.95) ;;
  1) TAG=spring_mu0.99;  OPT=spring;   ARM_FLAGS=(--config.vmc.optimizer.spring.mu=0.99) ;;
  2) TAG=spring_mu0.995; OPT=spring;   ARM_FLAGS=(--config.vmc.optimizer.spring.mu=0.995) ;;
  3) TAG=prime_sr;       OPT=prime_sr; ARM_FLAGS=() ;;
  4) TAG=ssu_defaults;   OPT=$SSU;     ARM_FLAGS=(--config.vmc.optimizer."$SSU".mu=0.0) ;;
  # Appended, not inserted -- indices 0-49 must keep their E9 meaning. See the header.
  5) TAG=spring_mu0.999; OPT=spring;   ARM_FLAGS=(--config.vmc.optimizer.spring.mu=0.999) ;;
esac

RUN_NAME="d9_${PRESET}_${TAG}_s${SEED}"

export WANDB_TAGS="d9,phase-d,100k,generality,head-to-head,rerun-of-e9,arm-${TAG},sys-${PRESET}"
export WANDB_NOTES="D9 = E9 at 100k epochs. Same arms, indexing, seeds and eta=0.02 as \
E9, so D9 cell i is directly comparable to E9 cell i. At 50k, E9 gave claim A holding \
on N (+0.024+/-0.019 vs the best fixed mu) and FAILING on O (+0.178+/-0.066, 0/5 seeds), \
with claim B holding on N (3.9 sigma, 5/5) and a tie on O. Oxygen is the condition to \
watch: it is the clearest evidence against a general parity claim, and 100k is the \
protocol on which that has to be restated. D9 also ADDS a mu=0.999 arm, absent from E9, \
because E9's fixed-mu optimum on N sat at the edge of its grid and so understated the \
tuned baseline. Arm: ${TAG} on ${PRESET}."

if [[ -e "$LOGROOT/$RUN_NAME" ]]; then
  echo "ERROR: $LOGROOT/$RUN_NAME already exists; move or delete it first." >&2
  exit 1
fi

python -c "import jax; print('jax devices:', jax.devices())"
echo "D9 arm=$TAG preset=$PRESET seed=$SEED nepochs=100000  $(date -Is)"

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
  --config.wandb.group="d9-atoms-${PRESET}" \
  --config.wandb.name="$RUN_NAME"

echo "D9 done $(date -Is)"
