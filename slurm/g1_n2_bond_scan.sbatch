#!/bin/bash
# G1 -- N2 bond-length scan, ORIGINAL SS-SPRING only, 100k epochs + eval.
#
# WHY THIS EXISTS. SS-SPRING diverged on 6/8 seeds on N2 at 4.0 Bohr (E14+E15) and was
# the best arm on N2 at equilibrium (D10). It also passed the H4 square at 4.0 Bohr
# (D7), which is strongly correlated too, so "strong correlation breaks SS-SPRING" is
# already 1 pass / 1 fail. Phase G asks whether the N2-4.0 failure is a common failure
# mode on strongly correlated systems or a one-off. G1 is the closest test: the same
# molecule, 14 electrons, eta, network and protocol, with ONLY the bond length moved.
# The static correlation grows with R, so this is a dose-response. Read it for:
#   - a threshold (fails above some R),
#   - a smooth rise in the divergence fraction with R,
#   - or nothing below 4.0 (the failure is specific to that geometry).
#
# THE CELLS: R in {2.4, 2.8, 3.2, 3.6} Bohr x seeds 0-4 = 20 runs, SS-SPRING only.
# The two ends already exist: N2_eq (2.016, D10, 3 seeds, 0/3 diverged) and N2_4.0
# (E14+E15, 8 seeds, 6/8 diverged). No SPRING/PRIME-SR arms by design; append them as
# indices 20+ if they are wanted later. Do not renumber.
#
# THE OPTIMIZER IS THE ORIGINAL SS-SPRING, the ssu_defaults arm of E14 exactly:
# same_sampled_spring_unified, mu=0.0 at init, eta=0.002, inverse_time decay 1e-4,
# constrain_norm=True with norm_constraint=1e-3. The code must be the tag
# ss-spring-original (vmcnet/ identical to d9605b5, the commit E14 ran). The guard
# below refuses to start on any other vmcnet/ tree, because even inactive code paths
# changed the float32 trajectory in Phase F.
#
# *** PRESET TRAP -- eval.nchains. *** The N2 presets set eval.nchains=2000; at 14
# electrons that is the OOM that destroyed E10's eval phase. Overridden to 1000 below,
# as in E14. Do not remove.
#
# STATISTICS. 5 seeds per R. If a geometry diverged as often as N2-4.0 (6/8), 0/5
# would happen 0.1% of the time; 0/5 only bounds the per-seed failure rate below ~45%
# (95%). Report divergence fractions per R, never a mean over survivors alone.
# Survival needs the full 100k: E15 seeds died as late as 56k and 68k.
#
# Array index = R_index*5 + seed.
# ~16 h per run (E14, same 14-electron system) => ~320 GPU-h.
#
# Before submitting, on Savio:  git fetch --tags && git checkout ss-spring-original
# (or any commit whose vmcnet/ matches the tag), then from the vmcnet repo root:
#     sbatch slurm/g1_n2_bond_scan.sbatch
# Then sync from a LOGIN node:  bash slurm/sync_wandb.sh
# Then, locally:
#     python slurm/divergence_check.py --experiment G1 E14 E15
#     python slurm/parse_eval_energies.py --experiment G1 --backfill
#
#SBATCH --job-name=g1-n2-scan
#SBATCH --account=co_esmath
#SBATCH --partition=savio3_gpu
#SBATCH --qos=esmath_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --cpus-per-task=2
#SBATCH --time=36:00:00
#SBATCH --array=0-19
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

LOGROOT="/global/scratch/users/$USER/vmcnet_logs/phase_g/g1_n2_bond_scan"
mkdir -p "$LOGROOT"

SSU=same_sampled_spring_unified
BONDS=(2.4 2.8 3.2 3.6)
R=${BONDS[$((SLURM_ARRAY_TASK_ID / 5))]}
SEED=$((SLURM_ARRAY_TASK_ID % 5))
PRESET="N2_${R}"
TAG=ssu_defaults
ETA=0.002

RUN_NAME="g1_${PRESET}_${TAG}_s${SEED}"

export WANDB_TAGS="g1,phase-g,100k,strong-correlation,n2-scan,arm-${TAG},sys-${PRESET}"
export WANDB_NOTES="G1: N2 bond-length scan with the ORIGINAL SS-SPRING (tag \
ss-spring-original = E14's code), to ask whether the N2-4.0 divergence (6/8 seeds) \
switches on with bond length. Same 14 electrons, eta=0.002, norm cap 1e-3, 100k + eval \
as E14; only the geometry changes. Ends of the scan: N2_eq 0/3 diverged (D10), N2_4.0 \
6/8 (E14+E15). Score the divergence fraction per bond length; healthy-until epoch from \
divergence_check.py. Bond length ${R} Bohr."

if [[ -e "$LOGROOT/$RUN_NAME" ]]; then
  echo "ERROR: $LOGROOT/$RUN_NAME already exists; move or delete it first." >&2
  exit 1
fi

python -c "import jax; print('jax devices:', jax.devices())"
echo "G1 code=$(git describe --tags --always --dirty) preset=$PRESET seed=$SEED" \
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
  --config.wandb.group="g1-n2-scan" \
  --config.wandb.name="$RUN_NAME"

echo "G1 done $(date -Is)"
