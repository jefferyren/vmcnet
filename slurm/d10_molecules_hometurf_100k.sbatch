#!/bin/bash
# D10 — E10 re-run at 100k epochs.   ANCESTOR: e10_molecules_hometurf.sbatch (50k).
#
# WHY PHASE D EXISTS, AND WHY THIS EXPERIMENT IS THE REASON. E10 is the only condition
# where the 50k budget was demonstrably insufficient: every arm was still improving
# 0.1-1.0 mHa per 5k training epochs at the end, absolute errors were 9-20 mHa against
# 0.2-0.8 mHa on the atoms, and no arm on either system ever reached chemical accuracy.
# E10 therefore compares arms at a FIXED BUDGET, not at their asymptotes, and PRIME-SR
# was still falling ~0.3 mHa/5k, so its 2.85-2.91 mHa deficit would partially close with
# a longer run. Both comparison papers report at 100k. This run is what lets the N2/CO
# result be quoted without that caveat -- or forces the caveat to be restated honestly.
#
# WHAT CHANGED FROM E10, AND WHAT DID NOT.
#   CHANGED:  nepochs 50000 -> 100000. New logroot, wandb project, job name.
#   EXPLICIT: constrain_norm / norm_constraint / schedule_type / learning_decay_rate now
#             passed on the command line, at the values Phase A-C already used.
#   UNCHANGED: arms, indexing, seeds, eta=0.002, 1000 training walkers, the eval phase,
#             random init, and the eval.nchains=1000 override. D10 index i == E10 index i.
#
# WHY d10- AND NOT e10-. parse_eval_energies.py globs `slurm-e10-hometurf-*_{idx}.out`
# and would silently merge a job named e10-*-100k into the 50k results.
#
# *** eval.nchains=1000 IS LOAD-BEARING. DO NOT REMOVE. ***
# preset_configs/{N2_eq,CO}.json still say eval.nchains=2000 against vmc.nchains=1000.
# At 14 electrons x 16 determinants that does not fit an 11 GB GTX2080TI, and it killed
# the eval phase of all 34 completed E10 runs with RESOURCE_EXHAUSTED (~7.94 GiB) after
# the training had finished. Eval energy is an unbiased mean either way -- the walker
# count sets the error bar, not the expectation -- so 1000 stays comparable to the atoms.
# Fix the presets themselves at some point; until then every script must override.
#
# IF EVAL DIES ANYWAY, DO NOT RETRAIN. Adapt slurm/e10_eval_recovery.sbatch: reload
# `checkpoints/100000.npz` (NOT best_checkpoint.npz, which is a best-of-training pick
# landing at a different epoch per arm) with --config.vmc.nepochs=0, so range(100000, 0)
# is empty and control falls straight through to eval. ~1-2 h per run instead of ~14.
#
# THE ARMS (7) x {N2_eq, CO} x 3 seeds = 42 runs:
#   0  SPRING mu=0.9      |  the PRIME-SR paper's own grid
#   1  SPRING mu=0.95     |
#   2  PRIME-SR              claim B comparator, at its own settings
#   3  SS-SPRING             untuned
#   4  SPRING mu=0.99        fair tuned baseline  -- see below
#   5  SPRING mu=0.995       matched-constant control (cf. E8)
#   6  SPRING mu=0.999       uncensors the grid (added in D10; see below)
#
# WHY mu=0.999 WAS ADDED. E10's fixed-mu ordering was monotone right up to the top of
# its grid on BOTH systems -- N2: 14.900 -> 11.818 -> 11.668 mHa and CO: 12.872 -> 9.607
# -> 9.210 for mu = 0.95 -> 0.99 -> 0.995 -- so the "tuned SPRING" baseline here was
# censored at the grid edge, not tuned. Claim A on these two systems (a tie on CO, a
# 3/3 win on N2) rests on that baseline, so a stronger fixed mu can only narrow or flip
# it. That is the reason to run the arm. Claim B is untouched either way: PRIME-SR
# settles near 0.953 regardless of what the SPRING arms do.
# Expect the gains to be flattening -- on N2, 0.95 -> 0.99 bought 3.08 mHa but 0.99 ->
# 0.995 bought only 0.15 -- so a small further gain, a wash, or a divergence are all
# plausible. eta/(1-mu) = 2.0 at mu=0.999 here against 0.400 at 0.995. Watch the mu
# trace and check_for_nans, and do not silently drop a diverged seed.
# Arm 6 is APPENDED so indices 0-35 keep their E10 meaning. Do not renumber.
#
# *** DO NOT CUT ARMS 4 AND 5. *** At eta=0.002 the norm cap essentially never binds,
# and in the uncapped regime the outcome tracks eta/(1-mu): 0.020 and 0.040 for arms 0
# and 1, 0.043 for PRIME-SR, against 0.400 for SS-SPRING at its converged ~0.995.
# Restricted to the paper's mu <= 0.95 grid, SS-SPRING "beats tuned SPRING" by 3.5 mHa
# on N2 and 3.7 on CO -- almost pure effective-step artifact, and any referee reading
# this next to D11 will catch it. E10 also found NO instability at mu=0.99 or 0.995 on
# either system for any seed, contrary to the PRIME-SR paper's report; re-check that at
# 100k via the mu trace and check_for_nans rather than assuming it carries over.
#
# Array index = arm*6 + system*3 + seed.
# ~13-14 h per run => ~570 GPU-h. RE-TIME BEFORE COMMITTING TO THE SECOND HALF: read
# the per-epoch rate out of the first .out within ~30 min and multiply. Stage it as
#     sbatch --exclude=n0135.savio3 --array=0-2,6-8,12-14,18-20,24-26,30-32,36-38 slurm/d10_molecules_hometurf_100k.sbatch
# for N2 first (21 runs), then CO with --array=3-5,9-11,15-17,21-23,27-29,33-35,39-41.
#
# INFRASTRUCTURE: one bad GPU on n0135.savio3 shredded 44 of E10's first 74 tasks --
# CUDA_ERROR_NO_DEVICE in the pre-flight jax.devices() check, dying in 1-57 s, while
# SLURM reported the node healthy. Keep --exclude=n0135.savio3 until proven fixed, and
# diagnose any repeat with failures clustered on one NodeList at seconds-long Elapsed:
#     sacct -j <jobid> --format=JobID%22,NodeList%16,State%14,ExitCode,Elapsed | grep -v batch
#
# Then sync from a LOGIN node:  bash slurm/sync_wandb.sh
# Then, locally:                python slurm/parse_eval_energies.py --experiment D10 --backfill
#   (that registry entry does not exist yet -- add it before the runs land.)
#
#SBATCH --job-name=d10-hometurf
#SBATCH --account=co_esmath
#SBATCH --partition=savio3_gpu
#SBATCH --qos=esmath_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --cpus-per-task=2
#SBATCH --time=36:00:00
#SBATCH --array=0-41
#SBATCH --output=slurm-%x-%A_%a.out

set -eo pipefail
export PATH="/global/scratch/users/$USER/envs/vmcnet/bin:$PATH"
set -u

export WANDB_MODE=offline
export WANDB_DIR="/global/scratch/users/$USER/wandb"
mkdir -p "$WANDB_DIR"

cd "$SLURM_SUBMIT_DIR"
LOGROOT="/global/scratch/users/$USER/vmcnet_logs/phase_d/d10_hometurf_100k"
mkdir -p "$LOGROOT"

SSU=same_sampled_spring_unified
ARM=$((SLURM_ARRAY_TASK_ID / 6))
SYS_I=$(((SLURM_ARRAY_TASK_ID % 6) / 3))
SEED=$((SLURM_ARRAY_TASK_ID % 3))
PRESETS=(N2_eq CO)
PRESET=${PRESETS[$SYS_I]}

# The molecular learning rate from both papers. NOT the 0.02 used elsewhere here.
ETA=0.002

case "$ARM" in
  0) TAG=spring_mu0.9;   OPT=spring;   ARM_FLAGS=(--config.vmc.optimizer.spring.mu=0.9) ;;
  1) TAG=spring_mu0.95;  OPT=spring;   ARM_FLAGS=(--config.vmc.optimizer.spring.mu=0.95) ;;
  2) TAG=prime_sr;       OPT=prime_sr; ARM_FLAGS=() ;;
  3) TAG=ssu_defaults;   OPT=$SSU;     ARM_FLAGS=(--config.vmc.optimizer."$SSU".mu=0.0) ;;
  4) TAG=spring_mu0.99;  OPT=spring;   ARM_FLAGS=(--config.vmc.optimizer.spring.mu=0.99) ;;
  5) TAG=spring_mu0.995; OPT=spring;   ARM_FLAGS=(--config.vmc.optimizer.spring.mu=0.995) ;;
  # Appended, not inserted -- indices 0-35 must keep their E10 meaning. See the header.
  6) TAG=spring_mu0.999; OPT=spring;   ARM_FLAGS=(--config.vmc.optimizer.spring.mu=0.999) ;;
esac

RUN_NAME="d10_${PRESET}_${TAG}_s${SEED}"

export WANDB_TAGS="d10,phase-d,100k,home-turf,head-to-head,rerun-of-e10,arm-${TAG},sys-${PRESET}"
export WANDB_NOTES="D10 = E10 at 100k epochs, on PRIME-SR's own systems (N2 at 2.016 \
Bohr, CO at 2.173) at its own eta=0.002. Same arms, indexing and seeds as E10, so D10 \
cell i is directly comparable to E10 cell i. E10's whole standing caveat was \
non-convergence at 50k: every arm still improving 0.1-1.0 mHa per 5k, no arm reaching \
chemical accuracy on either system, PRIME-SR still falling ~0.3 mHa/5k against its \
2.85 (N2) / 2.91 (CO) mHa deficit to SS-SPRING. This run is what removes that caveat, \
or restates it. Arms 4 and 5 (mu=0.99, 0.995) are the fair tuned baseline and the \
matched-constant control -- without them the comparison is an eta/(1-mu) artifact. \
D10 also ADDS a mu=0.999 arm, absent from E10, because the fixed-mu optimum was at the \
edge of E10's grid on both systems and so understated the tuned baseline. \
Arm: ${TAG} on ${PRESET}."

if [[ -e "$LOGROOT/$RUN_NAME" ]]; then
  echo "ERROR: $LOGROOT/$RUN_NAME already exists; move or delete it first." >&2
  exit 1
fi

python -c "import jax; print('jax devices:', jax.devices())"
echo "D10 arm=$TAG preset=$PRESET seed=$SEED eta=$ETA nepochs=100000  $(date -Is)"

vmc-molecule \
  --presets.path="preset_configs/${PRESET}.json" \
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
  --config.eval.nchains=1000 \
  --config.eval.nburn=10000 \
  --config.eval.nepochs=20000 \
  --config.initial_seed="$SEED" \
  --config.wandb.mode=offline \
  --config.wandb.project=vmcnet-phase-d \
  --config.wandb.group="d10-hometurf-${PRESET}" \
  --config.wandb.name="$RUN_NAME"

echo "D10 done $(date -Is)"
