#!/bin/bash
# G1 -- N2 bond-length scan, ORIGINAL SS-SPRING + SPRING(0.99), 100k epochs + eval.
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
# THE CELLS (20 + 12 = 32 runs):
#   0-19   SS-SPRING      R in {2.4, 2.8, 3.2, 3.6} Bohr x seeds 0-4
#   20-31  SPRING(0.99)   same R x seeds 0-2, appended so 0-19 keep their meaning
# The two ends already exist: N2_eq (2.016, D10, 0/3 diverged) and N2_4.0 (E14+E15,
# SS-SPRING 6/8 diverged, SPRING(0.99) 0/3). Do not renumber.
#
# WHY THE SPRING ARM. None of the four bond lengths has a pinned reference energy, so
# SS-SPRING's eval energy needs something to be compared against. SPRING(0.99) is the
# paper's own SPRING setting and the arm that stayed stable on N2_4.0 (14.99 mHa,
# 3/3), so it doubles as the stability control: if SPRING(0.99) also diverges at some
# R, the failure is not SS-SPRING's. 3 seeds, as E14 gave it.
#
# THE OPTIMIZERS ARE THE ORIGINAL CODE, E14's arms exactly, at eta=0.002 with
# inverse_time decay 1e-4 and constrain_norm=True, norm_constraint=1e-3:
#   ssu_defaults   same_sampled_spring_unified, mu=0.0 at init
#   spring_mu0.99  spring, mu=0.99
# The code must match the tag ss-spring-original (vmcnet/ identical to d9605b5, the
# commit E14 ran). The guard below refuses to start on any other vmcnet/*.py, because
# even inactive code paths changed the float32 trajectory in Phase F.
#
# *** PRESET TRAP -- eval.nchains. *** The N2 presets set eval.nchains=2000; at 14
# electrons that is the OOM that destroyed E10's eval phase. Overridden to 1000 below,
# as in E14. Do not remove.
#
# STATISTICS. 5 SS-SPRING seeds per R. If a geometry diverged as often as N2-4.0 (6/8),
# 0/5 would happen 0.1% of the time; 0/5 only bounds the per-seed failure rate below
# ~45% (95%). Report divergence fractions per R, never a mean over survivors alone.
# Survival needs the full 100k: E15 seeds died as late as 56k and 68k. Compare eval
# energies SS-SPRING vs SPRING(0.99) at the same R, seed for seed on seeds 0-2.
#
# Array index: SS-SPRING R_index*5 + seed; SPRING 20 + R_index*3 + seed.
# ~16 h per run (E14, same 14-electron system) => ~510 GPU-h.
#
# Before submitting, on Savio:  git pull && git fetch --tags  (master; its vmcnet/*.py
# matches the tag -- do NOT check out the tag itself, it predates these scripts),
# then from the vmcnet repo root:
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

LOGROOT="/global/scratch/users/$USER/vmcnet_logs/phase_g/g1_n2_bond_scan"
mkdir -p "$LOGROOT"

SSU=same_sampled_spring_unified
PRESETS=(N2_2.4 N2_2.8 N2_3.2 N2_3.6)
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

RUN_NAME="g1_${PRESET}_${TAG}_s${SEED}"

export WANDB_TAGS="g1,phase-g,100k,strong-correlation,n2-scan,arm-${TAG},sys-${PRESET}"
export WANDB_NOTES="G1: N2 bond-length scan with the ORIGINAL code (tag \
ss-spring-original = E14's), to ask whether SS-SPRING's N2-4.0 divergence (6/8 seeds) \
switches on with bond length. SPRING(0.99) is the comparison arm, since the scan points \
have no reference energies, and the stability control (3/3 stable on N2-4.0). Same 14 \
electrons, eta=0.002, norm cap 1e-3, 100k + eval as E14; only the geometry changes. \
Score the divergence fraction per bond length and eval energy against SPRING at the same \
R. Arm: ${TAG} on ${PRESET}."

if [[ -e "$LOGROOT/$RUN_NAME" ]]; then
  echo "ERROR: $LOGROOT/$RUN_NAME already exists; move or delete it first." >&2
  exit 1
fi

python -c "import jax; print('jax devices:', jax.devices())"
echo "G1 code=$(git describe --tags --always --dirty) arm=$TAG preset=$PRESET" \
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
  --config.wandb.group="g1-n2-scan" \
  --config.wandb.name="$RUN_NAME"

echo "G1 done $(date -Is)"
