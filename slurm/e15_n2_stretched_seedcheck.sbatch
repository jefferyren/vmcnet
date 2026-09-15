#!/bin/bash
# E15 -- is the E14 SS-SPRING divergence on stretched N2 consistent, or a three-seed fluke?
#
# WHAT E14 FOUND. All three SS-SPRING seeds on N2 at 4.0 Bohr terminated on NaNs, at
# epochs 18572 (s0), 17342 (s1) and 23559 (s2) of 100000, while SPRING(0.99) and
# PRIME-SR completed on all three seeds. The energy trace tracks the other two arms to
# within 0.03 Ha through epoch 10000, then loses 4-9 Hartree between 10k and 15k on
# every seed, partially recovers, and dies. 3/3 is suggestive; it is not enough to write
# "SS-SPRING is unstable on stretched N2" in a paper.
#
# WHAT THIS RUNS. The identical E14 SS-SPRING cell at five FRESH seeds, 3-7, at the
# FULL 100k protocol. Nothing changes from E14 except the seed -- same preset, same
# eta=0.002, same damping, norm constraint, walker count, epoch count, eval phase and
# random init -- so an E15 seed is exchangeable with an E14 seed cell for cell, and the
# two sets pool into one 8-seed sample with no protocol caveat attached.
#
# WHY THE FULL 100k RATHER THAN A SHORT STABILITY PROBE. A 40k run would answer the
# divergence question for ~40% of the cost, since the three observed failures are all
# under 23.6k. Two things make full length worth the extra ~50 GPU-h. A survivor at 40k
# only supports "did not diverge by 40k", which is a weaker sentence than the one the
# paper needs; and a survivor at 100k additionally produces an eval energy directly
# comparable to E14's SPRING (14.990 mHa) and PRIME-SR (17.606 mHa), which would fill
# the one empty cell in the results table. A short run buys the verdict but not the
# number.
#
# The eval phase is therefore ON, with eval.nchains forced to 1000 -- see the preset
# trap below. A diverged seed produces no eval phase at all, which is the expected
# outcome for most of them and is not an error.
#
# check_for_nans=True is load-bearing: without it nan_safe masks the NaNs and the run
# burns its full allocation producing garbage instead of stopping at the failure epoch.
#
# NO CONTROL ARM, on purpose. E14 already has SPRING(0.99) and PRIME-SR completing 3/3
# on this exact system at 100k; repeating them at new seeds costs ~80 GPU-h to confirm
# something already measured. If E15 comes back mixed (say 2 of 5 diverge), THEN a
# control becomes worth running, because the question changes from "does SS-SPRING fail
# here" to "how much of this is the system rather than the optimizer".
#
# HOW TO READ THE RESULT:
#   5/5 diverge  -> deterministic property of the method on this geometry. Report it as
#                   such; 8/8 across E14+E15 is not a fluke under any reading.
#   0/5 diverge  -> E14's three seeds were unlucky, or something about that submission
#                   differed. Check the E14 configs against these before concluding.
#   mixed        -> seed-dependent instability, which is a different and more awkward
#                   claim than either of the above. Report the fraction, not a mean.
#
# Array index = seed - 3, so --array=0-4 is seeds 3,4,5,6,7.
# ~16 h per surviving run (D10's measured rate on these 14-electron systems) plus ~1 h
# of eval => ~85 GPU-h. Seeds that diverge cost far less, ~3 h each; E14's three died
# at 2.7-3.5 h. Budget for the pessimistic case where none of them diverge.
#
# *** PRESET TRAP. *** preset_configs/N2_4.0.json sets eval.nchains=2000, which OOMs at
# 14 electrons on an 11GB 2080TI -- it destroyed E10's entire eval phase. Overridden to
# 1000 below, as D10, E13 and E14 all do. Do not remove it.
#
# Submit from the vmcnet repo root:
#     sbatch --exclude=n0135.savio3 slurm/e15_n2_stretched_seedcheck.sbatch
# Then, locally, for the stability verdict (reads .out directly, no wandb sync needed):
#     python slurm/divergence_check.py --experiment E14 E15
# and for the eval energy of any seed that survived:
#     python slurm/parse_eval_energies.py --experiment E15 --backfill
#
#SBATCH --job-name=e15-n2-seedcheck
#SBATCH --account=co_esmath
#SBATCH --partition=savio3_gpu
#SBATCH --qos=esmath_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --cpus-per-task=2
#SBATCH --time=36:00:00
#SBATCH --array=0-4
#SBATCH --output=slurm-%x-%A_%a.out

set -eo pipefail
export PATH="/global/scratch/users/$USER/envs/vmcnet/bin:$PATH"
set -u

export WANDB_MODE=offline
export WANDB_DIR="/global/scratch/users/$USER/wandb"
mkdir -p "$WANDB_DIR"

cd "$SLURM_SUBMIT_DIR"
LOGROOT="/global/scratch/users/$USER/vmcnet_logs/phase_e/e15_n2_seedcheck"
mkdir -p "$LOGROOT"

SSU=same_sampled_spring_unified
SEED=$((SLURM_ARRAY_TASK_ID + 3))   # seeds 0-2 are already spent in E14
PRESET=N2_4.0
ETA=0.002
NEPOCHS=100000

RUN_NAME="e15_${PRESET}_ssu_defaults_s${SEED}"

export WANDB_TAGS="e15,phase-e,seedcheck,follows-e14,stability,arm-ssu_defaults,sys-N2_4.0"
export WANDB_NOTES="E15 tests whether E14's SS-SPRING divergence on N2 at 4.0 Bohr is \
consistent or a three-seed fluke. E14: NaNs on 3/3 seeds at epochs 18572/17342/23559 of \
100000, after tracking SPRING and PRIME-SR to within 0.03 Ha through epoch 10000. This \
is the same cell at fresh seeds 3-7, identical to E14 in every setting including the \
100k epoch count and the eval phase, so an E15 seed pools with an E14 seed cell for \
cell. Full length rather than a short stability probe because a survivor at 100k also \
yields an eval energy comparable to E14's SPRING (14.990 mHa) and PRIME-SR (17.606 \
mHa), filling the one empty cell in the results table; a 40k run would buy the verdict \
but not the number. A diverged seed produces no eval phase, which is expected. \
Seed ${SEED}."

if [[ -e "$LOGROOT/$RUN_NAME" ]]; then
  echo "ERROR: $LOGROOT/$RUN_NAME already exists; move or delete it first." >&2
  exit 1
fi

python -c "import jax; print('jax devices:', jax.devices())"
echo "E15 arm=ssu_defaults preset=$PRESET seed=$SEED eta=$ETA nepochs=$NEPOCHS  $(date -Is)"

vmc-molecule \
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
  --config.vmc.nepochs="$NEPOCHS" \
  --config.vmc.checkpoint_every=10000 \
  --config.vmc.check_for_nans=True \
  --config.eval.nchains=1000 \
  --config.eval.nburn=10000 \
  --config.eval.nepochs=20000 \
  --config.initial_seed="$SEED" \
  --config.wandb.mode=offline \
  --config.wandb.project=vmcnet-phase-e \
  --config.wandb.group="e15-n2-seedcheck" \
  --config.wandb.name="$RUN_NAME"

echo "E15 done $(date -Is)"
