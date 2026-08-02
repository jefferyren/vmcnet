#!/bin/bash
# E2 (carbon): eta-robustness sweep, 25k-epoch screens from RANDOM INIT.
# Dims: eta {0.005, 0.02, 0.05, 0.1, 0.2} x optimizer {spring(mu=SPRING_MU),
# prime_sr, same_sampled_spring_unified} x norm constraint {on, off}.
# Array index = eta_idx * 6 + opt_idx * 2 + constraint_idx (30 tasks).
#
# The MECHANISM claims live in the constraint-OFF arm: while the norm cap
# binds, eta cancels exactly for spring/prime_sr (the direction never contains
# eta), so the ON arm largely measures the cap -- report it as the "practical
# settings" arm and read fraction-of-steps-at-cap from norm_cap_applied.txt.
# Key deliverables per run: energy.txt (robustness curve), mu.txt
# (prime_sr eta-invariance / ssu beta(eta) response), probe_r_ip.txt (ssu
# beta-lock-at-1 precursor), update_sq_norm_preclip.txt + norm_cap_applied.txt.
# check_for_nans=True aborts NaN'd runs (expected at large eta, constraint off).
#
# Submit from the vmcnet repo root:  sbatch slurm/e2_eta_sweep_carbon.sbatch
# Tuned-mu SPRING arm later (after E0):
#   sbatch --export=ALL,SPRING_MU=<best_mu> slurm/e2_eta_sweep_carbon.sbatch
# (spring runs are then named ..._mu<best_mu>_... and won't collide.)
#
#SBATCH --job-name=e2-eta-carbon
#SBATCH --account=co_esmath
#SBATCH --partition=savio3_gpu
#SBATCH --qos=esmath_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --array=0-29
#SBATCH --output=slurm-%x-%A_%a.out

set -eo pipefail
export PATH="/global/scratch/users/$USER/envs/vmcnet/bin:$PATH"
set -u

export WANDB_MODE=offline
export WANDB_DIR="/global/scratch/users/$USER/wandb"
mkdir -p "$WANDB_DIR"

cd "$SLURM_SUBMIT_DIR"
SEED="${SEED:-0}"
SPRING_MU="${SPRING_MU:-0.99}"
LOGROOT="/global/scratch/users/$USER/vmcnet_logs/phase_a/e2_eta_sweep_carbon"
mkdir -p "$LOGROOT"

ETAS=(0.005 0.02 0.05 0.1 0.2)
OPTS=(spring prime_sr same_sampled_spring_unified)
CONSTRAINTS=(True False)
CON_TAGS=(conON conOFF)

ETA=${ETAS[$((SLURM_ARRAY_TASK_ID / 6))]}
OPT=${OPTS[$(((SLURM_ARRAY_TASK_ID % 6) / 2))]}
CON=${CONSTRAINTS[$((SLURM_ARRAY_TASK_ID % 2))]}
CON_TAG=${CON_TAGS[$((SLURM_ARRAY_TASK_ID % 2))]}

if [[ "$OPT" == "spring" ]]; then
  OPT_TAG="spring_mu${SPRING_MU}"
  EXTRA_FLAGS=(--config.vmc.optimizer.spring.mu="$SPRING_MU")
else
  OPT_TAG="$OPT"
  EXTRA_FLAGS=()
fi
RUN_NAME="e2_carbon_${OPT_TAG}_eta${ETA}_${CON_TAG}_s${SEED}"

if [[ -e "$LOGROOT/$RUN_NAME" ]]; then
  echo "ERROR: $LOGROOT/$RUN_NAME already exists; move or delete it first." >&2
  exit 1
fi

python -c "import jax; print('jax devices:', jax.devices())"

vmc-molecule \
  --presets.path=preset_configs/carbon.json \
  --config.logdir="$LOGROOT" \
  --config.save_to_current_datetime_subfolder=False \
  --config.subfolder_name="$RUN_NAME" \
  --config.vmc.optimizer_type="$OPT" \
  --config.vmc.optimizer."$OPT".learning_rate="$ETA" \
  --config.vmc.optimizer."$OPT".constrain_norm="$CON" \
  ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"} \
  --config.vmc.nepochs=25000 \
  --config.vmc.checkpoint_every=5000 \
  --config.vmc.check_for_nans=True \
  --config.eval.nepochs=0 \
  --config.eval.nburn=0 \
  --config.initial_seed="$SEED" \
  --config.wandb.mode=offline \
  --config.wandb.project=vmcnet-phase-a \
  --config.wandb.group="e2-eta-carbon-${CON_TAG}" \
  --config.wandb.name="$RUN_NAME"
