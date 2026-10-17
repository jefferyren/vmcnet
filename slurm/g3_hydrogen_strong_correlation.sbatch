#!/bin/bash
# G3 -- hydrogen-only strong correlation, ORIGINAL SS-SPRING only, 100k + eval.
#
# WHY THIS EXISTS. Read g1_n2_bond_scan.sbatch first: Phase G asks whether SS-SPRING's
# divergence on N2 at 4.0 Bohr (6/8 seeds, E14+E15) is a common failure on strongly
# correlated systems. SS-SPRING already passed the H4 square at 4.0 Bohr (D7, 0/5
# diverged, -2.03421 Ha against the -2.0342 FCI/CBS of arXiv:2112.03491). G3 removes the
# heavy nuclei and adds correlated pairs:
#   - H10 chain at 1.8 Bohr (mild) and 3.6 Bohr (strongly stretched): many strongly
#     correlated H-H pairs, no core electrons (Motta et al., PRX 7, 031059 (2017)).
#     If H10_3.6 passes, the N2 failure points at heavy nuclei / system size or at
#     the eta=0.002 regime rather than at static correlation itself.
#   - H4 square at 1.0 Bohr: the other geometry in arXiv:2112.03491 (FCI/CBS -1.4390),
#     completing the pair with D7's 4.0 Bohr square. Cheap.
#
# THE CELLS: {H10_1.8, H10_3.6, H4_1.0} x seeds 0-4 = 15 runs, SS-SPRING only. Append
# other arms as indices 15+ if wanted later. Do not renumber.
#
# LEARNING RATE. eta=0.02, the hydrogen/atom family value H4 has always used (D7). This
# keeps "untuned per family" but leaves eta tied to atom type: if G3 passes while G1/G2
# fail, the follow-up is H10_3.6 at eta=0.002 to separate the two.
#
# THE OPTIMIZER IS THE ORIGINAL SS-SPRING (the ssu_defaults arm of E14/D7 exactly) and the
# code must be the tag ss-spring-original; see g1_n2_bond_scan.sbatch for the settings
# and why the guard below exists.
#
# eval.nchains: 1000 for H10 (10 electrons, close to the 14-electron OOM at 2000; not
# measured), 2000 for H4 as in D7.
#
# STATISTICS as in G1: 5 seeds, 0/5 bounds the per-seed failure rate below ~45%;
# survival needs the full 100k.
#
# Array index = system_index*5 + seed.
# Runtime not measured for H10; ~16 h assumed (H4 is much cheaper) => ~180 GPU-h at most.
#
# Before submitting, on Savio:  git fetch --tags && git checkout ss-spring-original
# then from the vmcnet repo root:
#     sbatch slurm/g3_hydrogen_strong_correlation.sbatch
# Then sync from a LOGIN node:  bash slurm/sync_wandb.sh
# Then, locally:
#     python slurm/divergence_check.py --experiment G3
#     python slurm/parse_eval_energies.py --experiment G3 --backfill
#
#SBATCH --job-name=g3-hydrogen-strongcorr
#SBATCH --account=co_esmath
#SBATCH --partition=savio3_gpu
#SBATCH --qos=esmath_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --cpus-per-task=2
#SBATCH --time=36:00:00
#SBATCH --array=0-14
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

LOGROOT="/global/scratch/users/$USER/vmcnet_logs/phase_g/g3_hydrogen_strong_correlation"
mkdir -p "$LOGROOT"

SSU=same_sampled_spring_unified
PRESETS=(H10_1.8 H10_3.6 H4_1.0)
EVAL_NCHAINS=(1000 1000 2000)
SYS_I=$((SLURM_ARRAY_TASK_ID / 5))
PRESET=${PRESETS[$SYS_I]}
SEED=$((SLURM_ARRAY_TASK_ID % 5))
TAG=ssu_defaults
ETA=0.02

RUN_NAME="g3_${PRESET}_${TAG}_s${SEED}"

export WANDB_TAGS="g3,phase-g,100k,strong-correlation,hydrogen,arm-${TAG},sys-${PRESET}"
export WANDB_NOTES="G3: hydrogen-only strong correlation with the ORIGINAL SS-SPRING \
(tag ss-spring-original = E14's code). H10 chains at 1.8 and 3.6 Bohr add many \
correlated pairs without heavy nuclei; H4 at 1.0 Bohr completes arXiv:2112.03491's pair \
with D7's 4.0 Bohr square, which SS-SPRING passed. eta=0.02 as for H4, norm cap 1e-3, \
100k + eval. Score the divergence fraction per system. System ${PRESET}."

if [[ -e "$LOGROOT/$RUN_NAME" ]]; then
  echo "ERROR: $LOGROOT/$RUN_NAME already exists; move or delete it first." >&2
  exit 1
fi

python -c "import jax; print('jax devices:', jax.devices())"
echo "G3 code=$(git describe --tags --always --dirty) preset=$PRESET seed=$SEED" \
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
  --config.eval.nchains="${EVAL_NCHAINS[$SYS_I]}" \
  --config.eval.nburn=5000 \
  --config.eval.nepochs=20000 \
  --config.initial_seed="$SEED" \
  --config.wandb.mode=offline \
  --config.wandb.project=vmcnet-phase-g \
  --config.wandb.group="g3-hydrogen-strongcorr" \
  --config.wandb.name="$RUN_NAME"

echo "G3 done $(date -Is)"
