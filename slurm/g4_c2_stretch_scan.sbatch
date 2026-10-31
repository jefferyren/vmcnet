#!/bin/bash
# G4 -- C2 stretch scan, ORIGINAL SS-SPRING + SPRING(0.99), 100k epochs + eval.
#
# WHY THIS EXISTS. Phase G found SS-SPRING diverging on stretched N2 (2.8 / 3.2 / 3.6 /
# 4.0 Bohr: 0/5, 1/5, 5/5, 6/8) and once on CO at 4.0 Bohr (1/5), and on nothing else.
# But C2 was only run at equilibrium and CO at one stretched point, so "N2 only" and
# "any stretched dimer" are not yet separated. G4 and G5 stretch C2 and CO to the SAME
# multiples of their own equilibrium bond length as the N2 scan points (1.39x, 1.59x,
# 1.79x, 1.98x of the presets' equilibria: N2 2.016, C2 2.348, CO 2.173 Bohr), so
# divergence rate can be read against stretch ratio for all three molecules:
#   - C2 and CO stay clean while N2 fails  -> N2-specific
#   - all three fail as they stretch       -> stretched dimers in general
#   - C2 fails but CO does not             -> bond type matters (homonuclear like N2)
#
# C2 is 12 electrons and multireference already at equilibrium (G2: 0/5 diverged at
# 2.348 Bohr). As the bond stretches, low-lying states approach the ground state, so
# the (6,6) singlet sector may not stay X1Sigma_g+ -- energies compare the two arms,
# not states. Stability, the main readout, is unaffected.
#
# THE CELLS (20 + 12 = 32 runs):
#   0-19   SS-SPRING      C2 at 3.26, 3.73, 4.19, 4.66 Bohr x seeds 0-4
#   20-31  SPRING(0.99)   same points x seeds 0-2
# Equilibrium is G2's C2 cell. Do not renumber.
#
# THE OPTIMIZERS ARE THE ORIGINAL CODE, E14's ssu_defaults and spring_mu0.99 arms
# exactly (eta=0.002, inverse_time decay 1e-4, norm cap 1e-3), and the code must match
# the tag ss-spring-original; see g1_n2_bond_scan.sbatch for the settings and why the
# guard below exists.
#
# *** PRESET TRAP -- eval.nchains. *** The presets set eval.nchains=2000, the OOM that
# destroyed E10's eval at 14 electrons. Overridden to 1000 below. Do not remove.
#
# STATISTICS as in G1: 5 SS-SPRING seeds per point (0/5 bounds the per-seed failure
# rate below ~45%); survival needs the full 100k. No reference energies: compare
# SS-SPRING against SPRING(0.99) at the same point.
#
# Array index: SS-SPRING point_index*5 + seed; SPRING 20 + point_index*3 + seed.
# ~13-16 h per run => ~450-510 GPU-h.
#
# Before submitting, on Savio:  git pull && git fetch --tags  (master; its vmcnet/*.py
# matches the tag -- do NOT check out the tag itself, it predates these scripts),
# then from the vmcnet repo root:
#     sbatch slurm/g4_c2_stretch_scan.sbatch
# Then sync (Phase G's scoped `wandb sync --legacy` loop, campaign log s8), and locally:
#     python slurm/divergence_check.py --experiment G4 G2
#     python slurm/parse_eval_energies.py --experiment G4 --backfill
#     python slurm/g_backfill_divergence.py
#
#SBATCH --job-name=g4-c2-scan
#SBATCH --account=co_esmath
#SBATCH --partition=savio3_gpu
#SBATCH --qos=esmath_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --cpus-per-task=2
#SBATCH --time=36:00:00
#SBATCH --array=0-31
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

# ---- code guard: run only the original SS-SPRING / SPRING code ----
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

LOGROOT="/global/scratch/users/$USER/vmcnet_logs/phase_g/g4_c2_stretch_scan"
mkdir -p "$LOGROOT"

SSU=same_sampled_spring_unified
PRESETS=(C2_3.26 C2_3.73 C2_4.19 C2_4.66)
NSSU=$((${#PRESETS[@]} * 5))
if (( SLURM_ARRAY_TASK_ID < NSSU )); then
  SYS_I=$((SLURM_ARRAY_TASK_ID / 5)); SEED=$((SLURM_ARRAY_TASK_ID % 5))
  TAG=ssu_defaults; OPT=$SSU; ARM_FLAGS=(--config.vmc.optimizer."$SSU".mu=0.0)
else
  J=$((SLURM_ARRAY_TASK_ID - NSSU)); SYS_I=$((J / 3)); SEED=$((J % 3))
  TAG=spring_mu0.99; OPT=spring; ARM_FLAGS=(--config.vmc.optimizer.spring.mu=0.99)
fi
PRESET=${PRESETS[$SYS_I]}
ETA=0.002

RUN_NAME="g4_${PRESET}_${TAG}_s${SEED}"

export WANDB_TAGS="g4,phase-g,100k,stretch-scan,c2,arm-${TAG},sys-${PRESET}"
export WANDB_NOTES="G4: C2 stretched to the same multiples of its equilibrium bond as \
the N2 scan points (1.39-1.98x), with the ORIGINAL code (tag ss-spring-original). Asks \
whether SS-SPRING's stretched-N2 divergence is N2-specific or common to stretched \
dimers. SPRING(0.99) is the comparison and stability control. eta=0.002, norm cap 1e-3, \
100k + eval as E14. Arm: ${TAG} on ${PRESET}."

if [[ -e "$LOGROOT/$RUN_NAME" ]]; then
  echo "ERROR: $LOGROOT/$RUN_NAME already exists; move or delete it first." >&2
  exit 1
fi

python -c "import jax; print('jax devices:', jax.devices())"
echo "G4 code=$(git describe --tags --always --dirty) arm=$TAG preset=$PRESET" \
     "seed=$SEED eta=$ETA nepochs=100000  $(date -Is)"

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
  --config.wandb.project=vmcnet-phase-g \
  --config.wandb.group="g4-c2-scan" \
  --config.wandb.name="$RUN_NAME"

echo "G4 done $(date -Is)"
