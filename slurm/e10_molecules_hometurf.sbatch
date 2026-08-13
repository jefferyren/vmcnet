#!/bin/bash
# E10 — PRIME-SR's home turf: N2 at equilibrium and CO.   FOLLOWS FROM: E7.
#
# RUN THIS LAST, AND ONLY AFTER E9. It is by far the most expensive script in the
# campaign and it exists to answer one specific objection rather than to discover
# anything.
#
# WHY IT EXISTS. E7 showed SS-SPRING beats PRIME-SR on carbon (0.069+/-0.009 mHa,
# ~7 sigma) and H4 (0.172+/-0.050). But the PRIME-SR paper's own headline electronic
# result is seed-robustness on **N2 at equilibrium and CO** at mu in {0.9, 0.95}. Until
# claim B is tested there, the obvious response is "you avoided the systems where we
# demonstrated our advantage." This closes that, on their systems, at their settings.
#
# THEIR SETTINGS, DELIBERATELY. eta = 0.002, which is the tuned molecular value in BOTH
# papers -- not the 0.02 used everywhere else in this campaign. The fixed-mu arms are
# 0.9 and 0.95, the values PRIME-SR reported for these systems, rather than the 0.99
# that won on carbon and H4. Handicapping ourselves to their configuration is the
# point; a win here is worth far more than a win on settings we chose.
#
# THE ARMS (6) x {N2_eq, CO} x 3 seeds = 36 runs:
#   0  SPRING mu=0.9      (PRIME-SR paper's grid)
#   1  SPRING mu=0.95     (PRIME-SR paper's grid)
#   2  PRIME-SR
#   3  SS-SPRING
#   4  SPRING mu=0.99     (added post-E11 -- see below)
#   5  SPRING mu=0.995    (added post-E11: matched constant to SS-SPRING's own beta)
# 3 seeds not 5, purely for cost. Array index = arm*6 + system*3 + seed.
#
# *** WHY ARMS 4 AND 5 WERE ADDED (2026-08-11, after E11). READ THIS BEFORE CUTTING THEM. ***
# The original design used only mu in {0.9, 0.95} because that is the PRIME-SR paper's
# grid, and excluded 0.99 because they report it unstable on N2/CO. E11 made that
# indefensible. At eta=0.002 the norm cap essentially never binds (it bound on only
# 1.6-4.3% of steps at eta=0.005 on carbon), and in the uncapped regime E11 showed the
# outcome tracks the accumulated step eta/(1-mu):
#     SPRING(0.9) 0.020 | SPRING(0.95) 0.040 | PRIME-SR(~0.953) 0.043 | SS-SPRING(~0.995) 0.400
# So against the original four arms SS-SPRING would enjoy a ~10x larger effective step
# and would win for reasons that have nothing to do with adaptivity -- an unpublishable
# result, and one a referee reading E11 alongside E10 would catch immediately.
# Arm 4 restores a fair tuned baseline; arm 5 is the matched-constant control that
# separates "the adaptive rule works" from "0.995 is a good constant here" (cf. E8).
# If mu=0.99/0.995 really are unstable on N2/CO while SS-SPRING at beta~0.995 is not,
# THAT is a strong result in its own right -- watch the mu trace and check_for_nans.
#
# *** eval.nchains WAS LOWERED 2000 -> 1000 ON 2026-08-13. DO NOT PUT IT BACK. ***
# The first E10 submission finished 50k training on 34 of 36 runs and then EVERY ONE of
# them died entering eval with RESOURCE_EXHAUSTED (~7.94 GiB). The presets set
# eval.nchains=2000 against vmc.nchains=1000, so eval doubles the walker count; at 14
# electrons and 16 determinants that does not fit an 11 GB GTX2080TI. Carbon (6
# electrons) did fit, which is why E7/E9/E11 never hit it. Eval energy is an unbiased
# mean either way -- walker count sets the error bar, not the expectation -- so 1000 is
# comparable to the 2000 used on the atoms, just slightly noisier.
# Recovering the first submission's lost eval phase: slurm/e10_eval_recovery.sbatch.
#
# COST WARNING. N2 and CO are 14-electron systems; the only measurement we have is a
# 200-epoch smoke test, which put N2 at roughly 3x carbon per step. At 50k + 20k eval
# that extrapolates to ~10-14h per run, so ~360-500 GPU-h for all 36 -- more than every
# other experiment in the campaign combined. RE-TIME AFTER THE FIRST RUN rather than
# trusting the extrapolation: read the per-epoch rate out of the .out file within the
# first ~30 min and multiply, before committing to the CO half.
# TO RUN N2 ONLY FIRST (18 runs, half the cost), submit with:
#     sbatch --array=0-2,6-8,12-14,18-20,24-26,30-32 slurm/e10_molecules_hometurf.sbatch
# and add CO afterwards with --array=3-5,9-11,15-17,21-23,27-29,33-35.
#
# Submit from the vmcnet repo root (see the N2-only form above)
# Then sync from a LOGIN node:  bash slurm/sync_wandb.sh
#
#SBATCH --job-name=e10-hometurf
#SBATCH --account=co_esmath
#SBATCH --partition=savio3_gpu
#SBATCH --qos=esmath_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --cpus-per-task=2
#SBATCH --time=36:00:00
#SBATCH --array=0-35
#SBATCH --output=slurm-%x-%A_%a.out

set -eo pipefail
export PATH="/global/scratch/users/$USER/envs/vmcnet/bin:$PATH"
set -u

export WANDB_MODE=offline
export WANDB_DIR="/global/scratch/users/$USER/wandb"
mkdir -p "$WANDB_DIR"

cd "$SLURM_SUBMIT_DIR"
LOGROOT="/global/scratch/users/$USER/vmcnet_logs/phase_c/e10_hometurf"
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
  # Added 2026-08-11 after E11. Without a high-mu fixed arm this experiment cannot be
  # interpreted: at eta=0.002 the norm cap essentially never binds, and E11 showed that
  # in the uncapped regime the outcome tracks eta/(1-mu). SS-SPRING settles at ~0.995,
  # so against a grid capped at 0.95 it enjoys a ~10x larger effective step and would
  # win for reasons that have nothing to do with adaptivity.
  4) TAG=spring_mu0.99;  OPT=spring;   ARM_FLAGS=(--config.vmc.optimizer.spring.mu=0.99) ;;
  # Matched-constant control: SS-SPRING's own converged beta as a fixed value. This is
  # the arm that separates "the adaptive rule works" from "0.995 is simply a good
  # constant here" -- cf. E8, which ran the same control on carbon and H4.
  5) TAG=spring_mu0.995; OPT=spring;   ARM_FLAGS=(--config.vmc.optimizer.spring.mu=0.995) ;;
esac

RUN_NAME="e10_${PRESET}_${TAG}_s${SEED}"

export WANDB_TAGS="e10,phase-c,home-turf,head-to-head,follows-e7,follows-e9,arm-${TAG},sys-${PRESET}"
export WANDB_NOTES="E10 tests campaign claim B on PRIME-SR's own systems (N2 at 2.016 \
Bohr, CO at 2.173) at their own settings: eta=0.002 and fixed mu in {0.9, 0.95}, the \
values their paper used, with mu=0.99 excluded because they report it unstable here. \
E7 already showed SS-SPRING beats PRIME-SR on carbon (0.069+/-0.009 mHa) and H4 \
(0.172+/-0.050); this removes the 'you avoided our systems' objection. Arm: ${TAG} on \
${PRESET}."

if [[ -e "$LOGROOT/$RUN_NAME" ]]; then
  echo "ERROR: $LOGROOT/$RUN_NAME already exists; move or delete it first." >&2
  exit 1
fi

python -c "import jax; print('jax devices:', jax.devices())"
echo "E10 arm=$TAG preset=$PRESET seed=$SEED eta=$ETA  $(date -Is)"

vmc-molecule \
  --presets.path="preset_configs/${PRESET}.json" \
  --config.logdir="$LOGROOT" \
  --config.save_to_current_datetime_subfolder=False \
  --config.subfolder_name="$RUN_NAME" \
  --config.vmc.optimizer_type="$OPT" \
  --config.vmc.optimizer."$OPT".learning_rate="$ETA" \
  "${ARM_FLAGS[@]}" \
  --config.vmc.nepochs=50000 \
  --config.vmc.checkpoint_every=10000 \
  --config.vmc.check_for_nans=True \
  --config.eval.nchains=1000 \
  --config.eval.nburn=10000 \
  --config.eval.nepochs=20000 \
  --config.initial_seed="$SEED" \
  --config.wandb.mode=offline \
  --config.wandb.project=vmcnet-phase-c \
  --config.wandb.group="e10-hometurf-${PRESET}" \
  --config.wandb.name="$RUN_NAME"

echo "E10 done $(date -Is)"
