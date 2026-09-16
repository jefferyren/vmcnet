#!/bin/bash
# E16 -- SPRING at mu=0.995 on stretched N2, the one knob turned off E14's SPRING arm.
#
# WHY THIS EXISTS. E14 ran N2 at 4.0 Bohr with SPRING at the paper's own mu=0.99 and
# got 14.990 mHa, against PRIME-SR's 17.606 mHa, while SS-SPRING diverged on 3/3 seeds
# (E15 is the follow-up on that). E14's header reserved the tuned-SPRING extension --
# "mu=0.9/0.95/0.995/0.999" -- for exactly this question: whether the SPRING column on
# this row is sitting at its own optimum or merely at the transferred default. This runs
# the single most interesting point of that grid, mu=0.995, the next step up in memory.
#
# Stretched N2 is the cell where a longer momentum tail is most likely to matter in
# either direction: the extra static correlation makes the local-energy signal noisier,
# which argues for more averaging, and also makes the landscape less forgiving of a
# stale direction, which argues for less. E14 measured only one point on that tradeoff.
#
# IDENTICAL TO E14 ARM 0 IN EVERY OTHER SETTING -- same preset, eta=0.002, inverse-time
# schedule and decay rate, norm constraint, epoch count, checkpoint interval, NaN check,
# eval phase and seeds 0-2. Only mu changes, 0.99 -> 0.995. That makes each E16 seed
# comparable cell for cell against the matching E14 spring_mu0.99 seed, and against
# E14's PRIME-SR and E13's MinSR cells for the same system, with no protocol caveat.
#
# SEEDS 0-2 ON PURPOSE, not fresh ones. This is a paired comparison against E14's SPRING
# arm, so sharing the random init removes one source of variance from the difference.
# (Contrast E15, which needed fresh seeds because its question was about seed-to-seed
# variability rather than about a parameter change.)
#
# *** PRESET TRAP -- eval.nchains. *** preset_configs/N2_4.0.json sets eval.nchains=2000
# against vmc.nchains=1000. At 14 electrons that OOMs on an 11GB 2080TI -- it destroyed
# E10's entire eval phase. Overridden to 1000 below, as D10, E13, E14 and E15 all do.
# Do not remove it.
#
# check_for_nans=True is load-bearing: without it nan_safe masks a divergence and the
# run burns its whole allocation producing garbage instead of stopping at the failure
# epoch. E14's SS-SPRING arm is why this matters on this system.
#
# CAVEAT ON THE REFERENCE ENERGY, repeated from E14. The -109.2021 Ha benchmark is
# Le Roy's MLR4(6,8) spectroscopic potential (via the SPRING paper), not a
# clamped-nuclei electronic structure calculation, so it carries mHa-level
# adiabatic/relativistic content. Fine for ranking arms against each other; state the
# source when quoting an absolute error.
#
# HOW TO READ THE RESULT. The difference against E14's 14.990 mHa is the whole point,
# and with three seeds it needs to clear the seed spread to mean anything -- report the
# paired per-seed differences, not just the two arm means.
#
# Array index = seed. ~16 h per run (D10's measured rate on these 14-electron systems)
# plus ~1 h of eval => ~50 GPU-h.
#
# Submit from the vmcnet repo root:
#     sbatch --exclude=n0135.savio3 slurm/e16_n2_stretched_spring_mu0995.sbatch
# Then sync from a LOGIN node:  bash slurm/sync_wandb.sh
# Then, locally:                python slurm/parse_eval_energies.py --experiment E16 --backfill
# (that last one needs an E16 entry in parse_eval_energies.py's EXPERIMENTS registry;
#  it is not there yet.)
#
#SBATCH --job-name=e16-n2-mu0995
#SBATCH --account=co_esmath
#SBATCH --partition=savio3_gpu
#SBATCH --qos=esmath_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --cpus-per-task=2
#SBATCH --time=36:00:00
#SBATCH --array=0-2
#SBATCH --output=slurm-%x-%A_%a.out

set -eo pipefail
export PATH="/global/scratch/users/$USER/envs/vmcnet/bin:$PATH"
set -u

export WANDB_MODE=offline
export WANDB_DIR="/global/scratch/users/$USER/wandb"
mkdir -p "$WANDB_DIR"

cd "$SLURM_SUBMIT_DIR"
LOGROOT="/global/scratch/users/$USER/vmcnet_logs/phase_e/e16_n2_stretched_mu0995"
mkdir -p "$LOGROOT"

SEED=$SLURM_ARRAY_TASK_ID     # 0-2, the same three inits E14's SPRING arm used
PRESET=N2_4.0
OPT=spring
MU=0.995

# The molecular learning rate from both papers, transferred from N2 at equilibrium.
# Unchanged from E14 -- mu is the only thing this script varies.
ETA=0.002
NEPOCHS=100000

TAG=spring_mu${MU}
RUN_NAME="e16_${PRESET}_${TAG}_s${SEED}"

export WANDB_TAGS="e16,phase-e,100k,stretched,mu-sweep,follows-e14,arm-${TAG},sys-${PRESET}"
export WANDB_NOTES="E16 turns SPRING's momentum up one step, mu=0.99 -> 0.995, on N2 at \
a stretched 4.0 Bohr bond, and changes nothing else from E14's SPRING arm: same preset, \
eta=0.002, inverse-time schedule, norm constraint, 100k epochs, eval phase and seeds \
0-2, so the comparison is paired seed by seed. E14 put SPRING(0.99) at 14.990 mHa and \
PRIME-SR at 17.606 mHa on this system, with SS-SPRING diverging on 3/3 seeds; what E14 \
did not establish is whether the SPRING column sits at its own optimum or just at the \
learning setting transferred from equilibrium N2. Stretched N2 is where a longer \
momentum tail could plausibly help (noisier local-energy signal) or hurt (less \
forgiving of a stale direction). Reference -109.2021 Ha is a spectroscopic MLR \
potential, not a clamped-nuclei calculation -- prefer arm-vs-arm comparisons. \
Arm: ${TAG} on ${PRESET}, seed ${SEED}."

if [[ -e "$LOGROOT/$RUN_NAME" ]]; then
  echo "ERROR: $LOGROOT/$RUN_NAME already exists; move or delete it first." >&2
  exit 1
fi

python -c "import jax; print('jax devices:', jax.devices())"
echo "E16 arm=$TAG preset=$PRESET seed=$SEED eta=$ETA mu=$MU nepochs=$NEPOCHS  $(date -Is)"

vmc-molecule \
  --presets.path="preset_configs/${PRESET}.json" \
  --config.logdir="$LOGROOT" \
  --config.save_to_current_datetime_subfolder=False \
  --config.subfolder_name="$RUN_NAME" \
  --config.vmc.optimizer_type="$OPT" \
  --config.vmc.optimizer."$OPT".learning_rate="$ETA" \
  --config.vmc.optimizer."$OPT".mu="$MU" \
  --config.vmc.optimizer."$OPT".schedule_type=inverse_time \
  --config.vmc.optimizer."$OPT".learning_decay_rate=0.0001 \
  --config.vmc.optimizer."$OPT".constrain_norm=True \
  --config.vmc.optimizer."$OPT".norm_constraint=0.001 \
  --config.vmc.nepochs="$NEPOCHS" \
  --config.vmc.checkpoint_every=10000 \
  --config.vmc.check_for_nans=True \
  --config.eval.nchains=1000 \
  --config.eval.nburn=10000 \
  --config.eval.nepochs=20000 \
  --config.initial_seed="$SEED" \
  --config.wandb.mode=offline \
  --config.wandb.project=vmcnet-phase-e \
  --config.wandb.group="e16-n2-stretched-mu0995" \
  --config.wandb.name="$RUN_NAME"

echo "E16 done $(date -Is)"
