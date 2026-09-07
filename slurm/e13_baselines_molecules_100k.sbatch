#!/bin/bash
# E13 -- MinSR and MinSR+M on the three molecules, at 100k epochs.
#
# Companion to e12_baselines_atoms_100k.sbatch; read that header first for why Phase E
# exists and why the "e12"/"e13" prefixes are glob-safe.
#
# THE ARMS (2), each x {N2_eq, N2_4.0, CO} x 3 seeds = 18 runs:
#   0  MinSR     minsr_momentum, mu=0.0, eta=0.02
#   1  MinSR+M   minsr_momentum, mu=0.9, eta=0.02
#
# WHY BOTH ARMS SHARE ETA HERE BUT NOT ON THE ATOMS. The SPRING paper re-tunes at the
# molecular scale on N2 at equilibrium and transfers to N2 4.0 Bohr and CO (Section 4.2,
# Figure 3): KFAC 0.05, MinSR 0.02, MinSR+M 0.02, SPRING 0.002. MinSR and MinSR+M
# happen to land on the same value there; on the atoms they do not (0.1 vs 0.2). The
# paper flags this itself -- MinSR+M's optimum moves by 10x between the two system
# classes, SPRING's by 10x, MinSR's by only 5x.
#
# NOTE THE 10x GAP TO THE SPRING ARM. D10 runs SPRING/PRIME-SR/SS-SPRING at eta=0.002
# on these systems; these baselines run at 0.02. That is not an inconsistency to fix --
# it is the paper's per-method tuning, and equalizing eta across methods is precisely
# what campaign log s4 item 8 says produces artifacts. Keep the per-method values and
# say so in the table caption.
#
# *** PRESET TRAP -- eval.nchains. *** preset_configs/{N2_eq,N2_4.0,CO}.json all still
# set eval.nchains=2000 against vmc.nchains=1000. At 14 electrons x 16 determinants that
# OOMs an 11GB GTX2080TI (RESOURCE_EXHAUSTED ~7.94GiB) and silently destroys the eval
# phase -- it cost E10 all 34 of its eval runs, and wandb shows nothing because it drops
# the eval phase anyway. This script overrides to 1000, as D10 does. Do not remove it,
# and do not trust the preset in any new script.
#
# Array index = arm*9 + system*3 + seed.
# ~16 h per run (measured on D10) => ~290 GPU-h. RE-TIME after the first few land:
# read the per-epoch rate out of a .out within ~30 min and multiply.
#
# Stage N2_eq first (both arms, 6 runs) before committing the rest:
#     sbatch --exclude=n0135.savio3 --array=0-2,9-11 slurm/e13_baselines_molecules_100k.sbatch
# then the remainder:
#     sbatch --exclude=n0135.savio3 --array=3-8,12-17 slurm/e13_baselines_molecules_100k.sbatch
#
# Then sync from a LOGIN node:  bash slurm/sync_wandb.sh
# Then, locally:                python slurm/parse_eval_energies.py --experiment E13 --backfill
#
#SBATCH --job-name=e13-baselines-mol
#SBATCH --account=co_esmath
#SBATCH --partition=savio3_gpu
#SBATCH --qos=esmath_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --cpus-per-task=2
#SBATCH --time=36:00:00
#SBATCH --array=0-17
#SBATCH --output=slurm-%x-%A_%a.out

set -eo pipefail
export PATH="/global/scratch/users/$USER/envs/vmcnet/bin:$PATH"
set -u

export WANDB_MODE=offline
export WANDB_DIR="/global/scratch/users/$USER/wandb"
mkdir -p "$WANDB_DIR"

cd "$SLURM_SUBMIT_DIR"
LOGROOT="/global/scratch/users/$USER/vmcnet_logs/phase_e/e13_baselines_molecules_100k"
mkdir -p "$LOGROOT"

OPT=minsr_momentum
ARM=$((SLURM_ARRAY_TASK_ID / 9))
SYS_I=$(((SLURM_ARRAY_TASK_ID % 9) / 3))
SEED=$((SLURM_ARRAY_TASK_ID % 3))
PRESETS=(N2_eq N2_4.0 CO)
PRESET=${PRESETS[$SYS_I]}

# Molecular tuned learning rate from the SPRING paper, Section 4.2. Same for both arms.
ETA=0.02

case "$ARM" in
  0) TAG=minsr;    MU=0.0 ;;
  1) TAG=minsr_m;  MU=0.9 ;;
esac

RUN_NAME="e13_${PRESET}_${TAG}_s${SEED}"

export WANDB_TAGS="e13,phase-e,100k,table1-baselines,arm-${TAG},sys-${PRESET}"
export WANDB_NOTES="E13 supplies the MinSR and MinSR+M columns of the comparison \
table on the three molecules, at the 100k protocol Phase D uses. N2_eq and CO sit \
beside D10; N2 at 4.0 Bohr has never been run in this campaign at all and its other \
three arms come from E14. Learning rate is the paper's molecular tuned value for both \
baselines (0.02), which is 10x the 0.002 the SPRING/PRIME-SR/SS-SPRING arms use -- \
that gap is the paper's own per-method tuning, not an error. eval.nchains is forced to \
1000: the presets say 2000 and that OOMs at 14 electrons on an 11GB 2080TI, which is \
what destroyed E10's entire eval phase. Arm: ${TAG} (mu=${MU}, eta=${ETA}) on \
${PRESET}."

if [[ -e "$LOGROOT/$RUN_NAME" ]]; then
  echo "ERROR: $LOGROOT/$RUN_NAME already exists; move or delete it first." >&2
  exit 1
fi

python -c "import jax; print('jax devices:', jax.devices())"
echo "E13 arm=$TAG preset=$PRESET seed=$SEED mu=$MU eta=$ETA nepochs=100000  $(date -Is)"

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
  --config.vmc.optimizer."$OPT".damping=0.001 \
  --config.vmc.optimizer."$OPT".constrain_norm=True \
  --config.vmc.optimizer."$OPT".norm_constraint=0.001 \
  --config.vmc.nepochs=100000 \
  --config.vmc.checkpoint_every=10000 \
  --config.vmc.check_for_nans=True \
  --config.eval.nchains=1000 \
  --config.eval.nburn=10000 \
  --config.eval.nepochs=20000 \
  --config.initial_seed="$SEED" \
  --config.wandb.mode=offline \
  --config.wandb.project=vmcnet-phase-e \
  --config.wandb.group="e13-baselines-${PRESET}" \
  --config.wandb.name="$RUN_NAME"

echo "E13 done $(date -Is)"
