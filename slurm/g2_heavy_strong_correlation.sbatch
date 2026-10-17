#!/bin/bash
# G2 -- heavy-atom strong correlation away from N2, ORIGINAL SS-SPRING only, 100k + eval.
#
# WHY THIS EXISTS. Read g1_n2_bond_scan.sbatch first: Phase G asks whether SS-SPRING's
# divergence on N2 at 4.0 Bohr (6/8 seeds, E14+E15) is a common failure on strongly
# correlated systems. G2 keeps heavy nuclei and molecular eta but changes the chemistry:
#   - C2 at equilibrium (2.348 Bohr, 12 electrons): strongly multireference WITHOUT a
#     stretched bond. Separates "multireference" from "stretched geometry".
#   - CO stretched to 4.0 Bohr (14 electrons): the isoelectronic twin of N2_4.0, a
#     stretched triple bond with the same electron count. CO at equilibrium is a clean
#     SS-SPRING result (D10, 0/3 diverged, 8.3 mHa). Separates "breaking a triple
#     bond" from "N2 specifically".
#
# THE CELLS: {C2, CO_4.0} x seeds 0-4 = 10 runs, SS-SPRING only. Append other arms as
# indices 10+ if wanted later. Do not renumber.
#
# THE OPTIMIZER IS THE ORIGINAL SS-SPRING (the ssu_defaults arm of E14 exactly) and the
# code must be the tag ss-spring-original; see g1_n2_bond_scan.sbatch for the settings
# and why the guard below exists.
#
# *** PRESET TRAP -- eval.nchains. *** Both presets set eval.nchains=2000, the OOM that
# destroyed E10's eval at 14 electrons; C2 at 12 is close enough to share it.
# Overridden to 1000 below. Do not remove.
#
# STATISTICS as in G1: 5 seeds, 0/5 bounds the per-seed failure rate below ~45%;
# survival needs the full 100k.
#
# References: neither is pinned. CO_4.0 has no literature value; compare C2 against a
# literature value once chosen. parse_eval_energies.py reports raw Ha meanwhile.
#
# Array index = system_index*5 + seed.
# ~14-16 h per run (E14/D10 at 14 electrons) => ~150 GPU-h.
#
# Before submitting, on Savio:  git fetch --tags && git checkout ss-spring-original
# then from the vmcnet repo root:
#     sbatch slurm/g2_heavy_strong_correlation.sbatch
# Then sync from a LOGIN node:  bash slurm/sync_wandb.sh
# Then, locally:
#     python slurm/divergence_check.py --experiment G2
#     python slurm/parse_eval_energies.py --experiment G2 --backfill
#
#SBATCH --job-name=g2-heavy-strongcorr
#SBATCH --account=co_esmath
#SBATCH --partition=savio3_gpu
#SBATCH --qos=esmath_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --cpus-per-task=2
#SBATCH --time=36:00:00
#SBATCH --array=0-9
#SBATCH --output=slurm-%x-%A_%a.out
# n0135's bad GPU shredded 44 of E10's first 74 tasks (CUDA_ERROR_NO_DEVICE). A
# command-line --exclude REPLACES this rather than merging; list both to add a node.
#SBATCH --exclude=n0135.savio3

set -eo pipefail
export PATH="/global/scratch/users/$USER/envs/vmcnet/bin:$PATH"
set -u

export WANDB_MODE=offline
export WANDB_DIR="/global/scratch/users/$USER/wandb"
mkdir -p "$WANDB_DIR"

cd "$SLURM_SUBMIT_DIR"

# ---- code guard: run only the original SS-SPRING ----
if ! git rev-parse -q --verify refs/tags/ss-spring-original >/dev/null; then
  echo "ERROR: tag ss-spring-original not found; run 'git fetch --tags' first." >&2
  exit 1
fi
# Python sources only: a tracked Finder .DS_Store under vmcnet/ must not block a run.
if ! git diff --quiet ss-spring-original -- 'vmcnet/*.py'; then
  echo "ERROR: vmcnet/ differs from ss-spring-original; Phase G must run the" \
       "original SS-SPRING. 'git diff ss-spring-original -- vmcnet' shows what." >&2
  exit 1
fi
VMCNET_FILE=$(python -c "import vmcnet; print(vmcnet.__file__)")
if [[ "$VMCNET_FILE" != "$PWD/"* ]]; then
  echo "ERROR: python imports vmcnet from $VMCNET_FILE, not this checkout." >&2
  exit 1
fi

LOGROOT="/global/scratch/users/$USER/vmcnet_logs/phase_g/g2_heavy_strong_correlation"
mkdir -p "$LOGROOT"

SSU=same_sampled_spring_unified
PRESETS=(C2 CO_4.0)
PRESET=${PRESETS[$((SLURM_ARRAY_TASK_ID / 5))]}
SEED=$((SLURM_ARRAY_TASK_ID % 5))
TAG=ssu_defaults
ETA=0.002

RUN_NAME="g2_${PRESET}_${TAG}_s${SEED}"

export WANDB_TAGS="g2,phase-g,100k,strong-correlation,heavy-atom,arm-${TAG},sys-${PRESET}"
export WANDB_NOTES="G2: heavy-atom strong correlation away from N2 with the ORIGINAL \
SS-SPRING (tag ss-spring-original = E14's code). C2 at equilibrium is multireference \
without stretching; CO at 4.0 Bohr is the isoelectronic twin of N2_4.0, where SS-SPRING \
diverged 6/8. Same eta=0.002, norm cap 1e-3, 100k + eval as E14. Score the divergence \
fraction per system. System ${PRESET}."

if [[ -e "$LOGROOT/$RUN_NAME" ]]; then
  echo "ERROR: $LOGROOT/$RUN_NAME already exists; move or delete it first." >&2
  exit 1
fi

python -c "import jax; print('jax devices:', jax.devices())"
echo "G2 code=$(git describe --tags --always --dirty) preset=$PRESET seed=$SEED" \
     "eta=$ETA nepochs=100000  $(date -Is)"

vmc-molecule \
  --presets.path="preset_configs/${PRESET}.json" \
  --config.logdir="$LOGROOT" \
  --config.save_to_current_datetime_subfolder=False \
  --config.subfolder_name="$RUN_NAME" \
  --config.vmc.optimizer_type="$SSU" \
  --config.vmc.optimizer."$SSU".learning_rate="$ETA" \
  --config.vmc.optimizer."$SSU".schedule_type=inverse_time \
  --config.vmc.optimizer."$SSU".learning_decay_rate=0.0001 \
  --config.vmc.optimizer."$SSU".constrain_norm=True \
  --config.vmc.optimizer."$SSU".norm_constraint=0.001 \
  --config.vmc.optimizer."$SSU".mu=0.0 \
  --config.vmc.nepochs=100000 \
  --config.vmc.checkpoint_every=10000 \
  --config.vmc.check_for_nans=True \
  --config.eval.nchains=1000 \
  --config.eval.nburn=10000 \
  --config.eval.nepochs=20000 \
  --config.initial_seed="$SEED" \
  --config.wandb.mode=offline \
  --config.wandb.project=vmcnet-phase-g \
  --config.wandb.group="g2-heavy-strongcorr" \
  --config.wandb.name="$RUN_NAME"

echo "G2 done $(date -Is)"
