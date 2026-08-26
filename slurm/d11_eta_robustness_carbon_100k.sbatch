#!/bin/bash
# D11 — E11 re-run at 100k epochs.   ANCESTOR: e11_eta_robustness_carbon.sbatch (50k).
#
# WHY PHASE D EXISTS. See d7_headtohead_seeds_100k.sbatch.
#
# WHY THIS ONE MATTERS MOST. E11 is not an ablation, it is the MECHANISM behind the
# head-to-head result. PRIME-SR's momentum is blind to eta by construction (median mu
# 0.9529 at eta=0.005 vs 0.9539 at 0.05 -- 0.001 across a 10x change) and settles near
# 0.953. Once the norm cap stops binding, the accumulated update scales as eta/(1-mu),
# which puts PRIME-SR roughly 10x below the high-momentum arms; at eta=0.005 it came in
# 2.7x worse than every other arm. That relation (r = -0.89 between log10(eta/(1-mu))
# and final error) is what explains E10, and it has to be measured on the same protocol
# as E10 or the two cannot be quoted together.
#
# THE 100k CAVEAT TO WATCH. The whole eta story turns on WHEN the norm cap releases.
# At 50k it bound on 37-97% of steps in the first half and exactly 0% in the second.
# At 100k the uncapped stretch is proportionally longer, so if anything the eta effect
# should STRENGTHEN -- but that is a prediction, not a result. Read norm_cap_applied
# out of the logs before repeating the 50k mechanism sentence at 100k.
#
# WHAT CHANGED FROM E11, AND WHAT DID NOT.
#   CHANGED:  nepochs 50000 -> 100000. Walltime 20h -> 24h. New logroot, wandb project,
#             job name.
#   EXPLICIT: constrain_norm / norm_constraint / schedule_type / learning_decay_rate now
#             passed on the command line, at the values Phase A-C already used.
#   UNCHANGED: arms, indexing, seeds, the two etas, 1000 walkers, eval, random init.
#             D11 index i == E11 index i.
#
# WHY d11- AND NOT e11-. parse_eval_energies.py globs `slurm-e11-eta-robust-*_{idx}.out`
# and would silently merge a job named e11-*-100k into the 50k results.
#
# THE ARMS (5), each x eta {0.005, 0.05} x 3 seeds = 30 runs:
#   0  SPRING mu=0.99
#   1  SPRING mu=0.995
#   2  PRIME-SR
#   3  SS-SPRING
#   4  SPRING mu=0.999   (added in D11; see below)
#
# WHY mu=0.999 WAS ADDED. At eta=0.005 the E11 fixed-mu ordering was 0.201 (mu=0.99) ->
# 0.162 (0.995), i.e. the optimum sat at the EDGE of the grid, so "tuned SPRING" at that
# eta was censored rather than tuned -- and eta=0.005 is where PRIME-SR collapses, so it
# is the condition the claim-B headline leans on hardest. A stronger fixed-mu baseline
# can only narrow our claim-A tie there; that is the reason to run it, not to skip it.
# At eta=0.05 the E11 optimum was interior (0.155 at 0.99 vs 0.191 at 0.995), so the arm
# is a control there rather than a correction -- and it is the eta where eta/(1-mu) = 50
# at mu=0.999, which is where instability is most likely. Watch check_for_nans and the mu
# trace; do not silently drop a diverged seed.
# Arm 4 is APPENDED so indices 0-23 keep their E11 meaning. Do not renumber.
# eta=0.02 for carbon is NOT re-run here -- it comes from D7, exactly as E11's 0.02
# column came from E7. Build the three-eta table from D7 + D11 together.
#
# Array index = arm*6 + eta_index*3 + seed.
# ~7-8 h per run => ~225 GPU-h.
#
# Submit from the vmcnet repo root:  sbatch slurm/d11_eta_robustness_carbon_100k.sbatch
# Then sync from a LOGIN node:       bash slurm/sync_wandb.sh
# Then, locally:                     python slurm/parse_eval_energies.py --experiment D11 --backfill
#   (that registry entry does not exist yet -- add it before the runs land.)
#
#SBATCH --job-name=d11-eta-robust
#SBATCH --account=co_esmath
#SBATCH --partition=savio3_gpu
#SBATCH --qos=esmath_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --array=0-29
#SBATCH --output=slurm-%x-%A_%a.out

set -eo pipefail
export PATH="/global/scratch/users/$USER/envs/vmcnet/bin:$PATH"
set -u

export WANDB_MODE=offline
export WANDB_DIR="/global/scratch/users/$USER/wandb"
mkdir -p "$WANDB_DIR"

cd "$SLURM_SUBMIT_DIR"
LOGROOT="/global/scratch/users/$USER/vmcnet_logs/phase_d/d11_eta_100k"
mkdir -p "$LOGROOT"

SSU=same_sampled_spring_unified
ARM=$((SLURM_ARRAY_TASK_ID / 6))
ETA_I=$(((SLURM_ARRAY_TASK_ID % 6) / 3))
SEED=$((SLURM_ARRAY_TASK_ID % 3))
ETAS=(0.005 0.05)
ETA=${ETAS[$ETA_I]}

case "$ARM" in
  0) TAG=spring_mu0.99;  OPT=spring;   ARM_FLAGS=(--config.vmc.optimizer.spring.mu=0.99) ;;
  1) TAG=spring_mu0.995; OPT=spring;   ARM_FLAGS=(--config.vmc.optimizer.spring.mu=0.995) ;;
  2) TAG=prime_sr;       OPT=prime_sr; ARM_FLAGS=() ;;
  3) TAG=ssu_defaults;   OPT=$SSU;     ARM_FLAGS=(--config.vmc.optimizer."$SSU".mu=0.0) ;;
  # Appended, not inserted -- indices 0-23 must keep their E11 meaning. See the header.
  4) TAG=spring_mu0.999; OPT=spring;   ARM_FLAGS=(--config.vmc.optimizer.spring.mu=0.999) ;;
esac

RUN_NAME="d11_carbon_${TAG}_eta${ETA}_s${SEED}"

export WANDB_TAGS="d11,phase-d,100k,eta-robustness,mechanism,rerun-of-e11,arm-${TAG},eta-${ETA}"
export WANDB_NOTES="D11 = E11 at 100k epochs. Same arms, indexing, seeds and etas as \
E11, so D11 cell i is directly comparable to E11 cell i. At 50k, E11 found PRIME-SR 2.7x \
worse than every other arm at eta=0.005 (0.435+/-0.025 mHa vs 0.162-0.201) with a swing \
of 0.242 mHa across the 10x eta span against 0.030-0.054 for the others, and claim A \
failing at eta=0.05 (+0.037+/-0.010, 0/3 seeds). This is the mechanism experiment for \
the whole paper: check norm_cap_applied and the mu trace, not just the energies. \
D11 also ADDS a mu=0.999 arm, absent from E11, because the fixed-mu optimum at \
eta=0.005 sat at the edge of E11's grid and so understated the tuned baseline at \
exactly the eta the claim-B headline leans on. Arm: ${TAG} at eta=${ETA}."

if [[ -e "$LOGROOT/$RUN_NAME" ]]; then
  echo "ERROR: $LOGROOT/$RUN_NAME already exists; move or delete it first." >&2
  exit 1
fi

python -c "import jax; print('jax devices:', jax.devices())"
echo "D11 arm=$TAG eta=$ETA seed=$SEED nepochs=100000  $(date -Is)"

vmc-molecule \
  --presets.path=preset_configs/carbon.json \
  --config.logdir="$LOGROOT" \
  --config.save_to_current_datetime_subfolder=False \
  --config.subfolder_name="$RUN_NAME" \
  --config.vmc.optimizer_type="$OPT" \
  --config.vmc.optimizer."$OPT".learning_rate="$ETA" \
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
  --config.wandb.group="d11-eta-${ETA}" \
  --config.wandb.name="$RUN_NAME"

echo "D11 done $(date -Is)"
