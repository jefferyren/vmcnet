#!/bin/bash
# E14 -- N2 at 4.0 Bohr for the three campaign arms, at 100k epochs.
#
# Companion to e12_baselines_atoms_100k.sbatch; read that header first for why Phase E
# exists and why the "e1x" prefixes are glob-safe.
#
# WHY THIS EXISTS. N2 at a stretched 4.0 Bohr bond is the one system in the SPRING
# paper's Table 1 that this campaign has never touched -- a wandb sweep across all six
# projects returns zero runs matching it. It is also the most interesting of the six for
# our claims: stretching N2 raises the static-correlation content, so it is the cell
# where an optimizer that merely takes bigger steps can most plausibly come apart. The
# preset has existed since Phase A and has never been used.
#
# It is deliberately NOT in the PRIME-SR paper, so this is a system where their method
# has no tuned settings of its own to fall back on.
#
# THE ARMS (3) x 3 seeds = 9 runs:
#   0  SPRING mu=0.99   the paper's own SPRING setting
#   1  PRIME-SR         claim B comparator
#   2  SS-SPRING        untuned
#
# The MinSR and MinSR+M cells for this same system come from E13, at their own tuned
# eta of 0.02. Between the two scripts N2_4.0 gets the full five-method row.
#
# WHY ONLY mu=0.99 AND NOT THE FULL FIXED-mu GRID. This script fills a table row, it
# does not re-litigate claim A: the row's SPRING column is the paper's SPRING, which is
# mu=0.99. If you later want a tuned-SPRING baseline on this system to match D10's
# treatment of N2_eq and CO, append arms as `case 3)` onward (mu=0.9/0.95/0.995/0.999)
# and submit --array=9-20 -- indices 0-8 keep their meaning. Do not renumber.
#
# *** PRESET TRAP -- eval.nchains. *** preset_configs/N2_4.0.json sets eval.nchains=2000
# against vmc.nchains=1000. Same 14 electrons as CO, so the same OOM that destroyed
# E10's eval phase applies. Overridden to 1000 below. Do not remove.
#
# CAVEAT ON THE REFERENCE ENERGY. The -109.2021 benchmark is Le Roy's MLR4(6,8)
# spectroscopic potential (via the SPRING paper), not a clamped-nuclei electronic
# structure calculation, so it carries mHa-level adiabatic/relativistic content. Fine
# for ranking optimizers against each other; state the source when quoting an absolute
# error.
#
# Array index = arm*3 + seed.
# ~16 h per run (measured on D10 for the same 14-electron systems) => ~145 GPU-h.
#
# Submit from the vmcnet repo root:
#     sbatch --exclude=n0135.savio3 slurm/e14_n2_stretched_100k.sbatch
# Then sync from a LOGIN node:  bash slurm/sync_wandb.sh
# Then, locally:                python slurm/parse_eval_energies.py --experiment E14 --backfill
#
#SBATCH --job-name=e14-n2-stretched
#SBATCH --account=co_esmath
#SBATCH --partition=savio3_gpu
#SBATCH --qos=esmath_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --cpus-per-task=2
#SBATCH --time=36:00:00
#SBATCH --array=0-8
#SBATCH --output=slurm-%x-%A_%a.out

set -eo pipefail
export PATH="/global/scratch/users/$USER/envs/vmcnet/bin:$PATH"
set -u

export WANDB_MODE=offline
export WANDB_DIR="/global/scratch/users/$USER/wandb"
mkdir -p "$WANDB_DIR"

cd "$SLURM_SUBMIT_DIR"
LOGROOT="/global/scratch/users/$USER/vmcnet_logs/phase_e/e14_n2_stretched_100k"
mkdir -p "$LOGROOT"

SSU=same_sampled_spring_unified
ARM=$((SLURM_ARRAY_TASK_ID / 3))
SEED=$((SLURM_ARRAY_TASK_ID % 3))
PRESET=N2_4.0

# The molecular learning rate from both papers, transferred from N2 at equilibrium.
ETA=0.002

case "$ARM" in
  0) TAG=spring_mu0.99; OPT=spring;   ARM_FLAGS=(--config.vmc.optimizer.spring.mu=0.99) ;;
  1) TAG=prime_sr;      OPT=prime_sr; ARM_FLAGS=() ;;
  2) TAG=ssu_defaults;  OPT=$SSU;     ARM_FLAGS=(--config.vmc.optimizer."$SSU".mu=0.0) ;;
esac

RUN_NAME="e14_${PRESET}_${TAG}_s${SEED}"

export WANDB_TAGS="e14,phase-e,100k,stretched,table1,arm-${TAG},sys-${PRESET}"
export WANDB_NOTES="E14 fills the one system in the SPRING paper's Table 1 that this \
campaign has never run: N2 stretched to a 4.0 Bohr bond. Same 100k protocol, seeds and \
eta=0.002 as D10, so these cells are directly comparable to D10's N2_eq and CO. The \
system is absent from the PRIME-SR paper, so PRIME-SR has no tuned settings of its own \
here. Stretched N2 carries more static correlation than the equilibrium geometry, which \
makes it the cell most likely to separate the arms on something other than effective \
step size. The MinSR and MinSR+M cells of this row come from E13. Reference \
-109.2021 Ha is a spectroscopic MLR potential, not a clamped-nuclei calculation -- \
prefer arm-vs-arm comparisons. Arm: ${TAG} on ${PRESET}."

if [[ -e "$LOGROOT/$RUN_NAME" ]]; then
  echo "ERROR: $LOGROOT/$RUN_NAME already exists; move or delete it first." >&2
  exit 1
fi

python -c "import jax; print('jax devices:', jax.devices())"
echo "E14 arm=$TAG preset=$PRESET seed=$SEED eta=$ETA nepochs=100000  $(date -Is)"

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
  --config.wandb.project=vmcnet-phase-e \
  --config.wandb.group="e14-n2-stretched" \
  --config.wandb.name="$RUN_NAME"

echo "E14 done $(date -Is)"
