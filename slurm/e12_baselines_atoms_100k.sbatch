#!/bin/bash
# E12 -- MinSR and MinSR+M on the three atoms, at 100k epochs.
#
# WHY PHASE E EXISTS. Phases A-D compared SPRING, PRIME-SR and SS-SPRING against each
# other. The paper's Table 1 also carries a MinSR and a MinSR+M column, and NEITHER
# baseline has ever been run in this campaign: `grep -ri minsr vmcnet/` returned nothing
# before the minsr_momentum optimizer was added, and the only mu=0 runs anywhere are
# three Phase A E0 cells at 25k, one seed, at the wrong learning rate. E12-E14 fill in
# the two missing columns and the one missing system (N2 at 4.0 Bohr).
#
# ON THE "e" PREFIX. In Phases A-C "e" happened to mean 50k and "d" meant the 100k
# reruns, but the load-bearing rule in the campaign log is narrower: never reuse a
# prefix that an existing parse_eval_energies.py glob already matches. The registered
# globs are slurm-e7-headtohead-*, slurm-e9-atoms-*, slurm-e10-hometurf-*,
# slurm-e11-eta-robust-* and their d7/d9/d10/d11 counterparts. `e12-` collides with
# none of them. *** EVERY PHASE E RUN IS 100k, same protocol as Phase D. ***
#
# THE ARMS (2), each x {carbon, N, O} x 5 seeds = 30 runs:
#   0  MinSR     minsr_momentum, mu=0.0,  eta=0.1   paper Eqs. (41)-(42)
#   1  MinSR+M   minsr_momentum, mu=0.9,  eta=0.2   paper Eqs. (43)-(44), Section 3.4
#
# WHY THE ETAS DIFFER FROM EVERY OTHER SCRIPT HERE. The SPRING paper tunes eta PER
# METHOD on the carbon atom and then transfers it to N and O (Section 4.1, Figure 1):
# KFAC 0.02, MinSR 0.1, MinSR+M 0.2, SPRING 0.02. Running these baselines at SPRING's
# 0.02 would hand us a win by mis-tuning the comparators -- the exact eta/(1-mu)
# artifact that campaign log s4 item 8 warns about, only pointed the other way. Use the
# paper's own tuned values.
#
# *** SMOKE-TEST BEFORE SUBMITTING THE ARRAY. *** eta=0.1 and 0.2 are 5x and 10x
# anything this campaign has run. D11 established that the norm cap stops binding
# halfway through a 100k schedule, so a learning rate that looks safe early can still
# blow up late; the paper reports both as stable with C=0.001, but verify rather than
# assume. Run one carbon MinSR seed first and watch for NaNs:
#     sbatch --exclude=n0135.savio3 --array=0 slurm/e12_baselines_atoms_100k.sbatch
#
# Array index = arm*15 + system*5 + seed.
# ~7-8 h per run (measured on D9) => ~220 GPU-h.
#
# Submit from the vmcnet repo root:
#     sbatch --exclude=n0135.savio3 slurm/e12_baselines_atoms_100k.sbatch
# Then sync from a LOGIN node:   bash slurm/sync_wandb.sh
# Then, locally:                 python slurm/parse_eval_energies.py --experiment E12 --backfill
#
#SBATCH --job-name=e12-baselines-atoms
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
LOGROOT="/global/scratch/users/$USER/vmcnet_logs/phase_e/e12_baselines_atoms_100k"
mkdir -p "$LOGROOT"

OPT=minsr_momentum
ARM=$((SLURM_ARRAY_TASK_ID / 15))
SYS_I=$(((SLURM_ARRAY_TASK_ID % 15) / 5))
SEED=$((SLURM_ARRAY_TASK_ID % 5))
PRESETS=(carbon N O)
PRESET=${PRESETS[$SYS_I]}

# Per-method tuned learning rates from the SPRING paper, Section 4.1.
case "$ARM" in
  0) TAG=minsr;    MU=0.0; ETA=0.1 ;;
  1) TAG=minsr_m;  MU=0.9; ETA=0.2 ;;
esac

RUN_NAME="e12_${PRESET}_${TAG}_s${SEED}"

export WANDB_TAGS="e12,phase-e,100k,table1-baselines,arm-${TAG},sys-${PRESET}"
export WANDB_NOTES="E12 supplies the MinSR and MinSR+M columns of the comparison \
table on the three atoms, at the 100k protocol Phase D uses, so these cells drop \
straight in beside D7 (carbon) and D9 (N, O). Both arms come from the same \
minsr_momentum optimizer: mu=0 is MinSR, mu=0.9 is MinSR+M, and a unit test pins the \
mu=0 solve against SPRING's own kernel so the MinSR column is provably the same linear \
solve as the SPRING arm rather than a lookalike. Learning rates are the paper's own \
per-method tuned values (MinSR 0.1, MinSR+M 0.2 on atoms), NOT SPRING's 0.02 -- \
running the comparators at SPRING's eta would manufacture a win out of eta/(1-mu). \
Arm: ${TAG} (mu=${MU}, eta=${ETA}) on ${PRESET}."

if [[ -e "$LOGROOT/$RUN_NAME" ]]; then
  echo "ERROR: $LOGROOT/$RUN_NAME already exists; move or delete it first." >&2
  exit 1
fi

python -c "import jax; print('jax devices:', jax.devices())"
echo "E12 arm=$TAG preset=$PRESET seed=$SEED mu=$MU eta=$ETA nepochs=100000  $(date -Is)"

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
  --config.eval.nchains=2000 \
  --config.eval.nburn=5000 \
  --config.eval.nepochs=20000 \
  --config.initial_seed="$SEED" \
  --config.wandb.mode=offline \
  --config.wandb.project=vmcnet-phase-e \
  --config.wandb.group="e12-baselines-${PRESET}" \
  --config.wandb.name="$RUN_NAME"

echo "E12 done $(date -Is)"
