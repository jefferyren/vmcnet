#!/bin/bash
# F5 -- does the beta* guard (F4a, a noise-aware momentum) stop the stretched-N2
# failure, and does it leave the stable systems alone?
# Plan and reasoning: docs/CAMPAIGN_LOG.md, Section 5 "Phase F" (F4d RESULT, route).
#
# WHY. F4d fixed the float32 Gram but 6/6 replays still failed: the momentum keeps
# growing on fresh walkers (carried/eps 15 -> 300 over ~50 rows) while the step obeys
# SPRING's exact bound and beta sits frozen at ~0.997. beta_star_guard=True makes the
# main solve use momentum min(beta, 2 beta*), where beta* = <eps, A phi>/||A phi||^2
# is the momentum that best fits this step's target on its (fresh) walkers. Past
# 2 beta* the carried term raises the residual the solve must fit, so the old direction
# is doing harm on the new walkers. No tunable constant; the probe and the beta
# controller are untouched (vmcnet/updates/same_sampled_spring_unified.py).
#
# DESIGN. One arm everywhere: gram_center_first=True (F4d), beta_star_guard=True,
# diagnostics on, no carried cap, no safeguard.
#   0-5  the six failing N2-4.0 states, 10k steps each (as F3b/F4d):
#        0 E14 s0 (10k)  1 E14 s1 (10k)  2 E14 s2 (10k)  3 E15 s7 (10k)
#        4 E15 s6 (60k)  5 E18 s2 (10k, eta 0.0015)
#        Baseline = the F4d replays on disk: 6 of 6 failed. 0-1 failures here is a
#        clear effect (baseline ~11/12 across F3b no-cap + F4d).
#   6-11 the stable systems, seed 0, from their 50k checkpoint for 5k steps:
#        6 carbon (D7)  7 H4 (D7)  8 N (D9)  9 O (D9)  10 N2_eq (D10)  11 CO (D10)
#        The question is how often the guard binds there (diag_carried_scale < 1),
#        and whether the energy stays at the original run's level over the same
#        epochs. A guard that binds often on these changes the method everywhere.
#
# WHAT SUCCESS LOOKS LIKE. N2 arms reach their end epoch at or below the pre-replay
# energy, the guard binding mostly in the rows where carried/eps used to climb.
# Stable arms: guard rarely binding, energy unchanged. If N2 arms still fail, check
# whether they fail fast (as F4d) or by slow drift (as the F3b capped arms) -- the
# latter means the parameters degrade even with the momentum held down.
#
# COST. ~0.6-0.7 s/step on a 2080 Ti for N2 (10k steps => ~2 h each); the stable arms
# are 5k steps (<= 1 h each). ~16 GPU-h in all.
#
# Submit from the vmcnet repo root on a Savio login node (code must include the guard):
#     sbatch slurm/f5_beta_star_guard.sbatch
# READOUT, afterwards, on the login node:
#     python slurm/f2_replay_summary.py --carried --guard --pattern "f5_*" \
#         --logroot /global/scratch/users/$USER/vmcnet_logs/phase_f/f5_beta_star_guard
#
#SBATCH --job-name=f5-beta-star-guard
#SBATCH --account=co_esmath
#SBATCH --partition=savio3_gpu
#SBATCH --qos=esmath_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --cpus-per-task=2
#SBATCH --time=05:00:00
#SBATCH --array=0-11
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --exclude=n0135.savio3

set -eo pipefail
export PATH="/global/scratch/users/$USER/envs/vmcnet/bin:$PATH"
set -u

export WANDB_MODE=offline
export WANDB_DIR="/global/scratch/users/$USER/wandb"
mkdir -p "$WANDB_DIR"

cd "$SLURM_SUBMIT_DIR"

PHASE_D="/global/scratch/users/$USER/vmcnet_logs/phase_d"
PHASE_E="/global/scratch/users/$USER/vmcnet_logs/phase_e"
LOGROOT="/global/scratch/users/$USER/vmcnet_logs/phase_f/f5_beta_star_guard"
mkdir -p "$LOGROOT"

SSU=same_sampled_spring_unified
CASE=$SLURM_ARRAY_TASK_ID
NSTEPS=10000

case "$CASE" in
  0) TAG=e14s0; PRESET=N2_4.0; SEED=0; ETA=0.002;  FROM=10000
     OLD="$PHASE_E/e14_n2_stretched_100k/e14_N2_4.0_ssu_defaults_s0" ;;
  1) TAG=e14s1; PRESET=N2_4.0; SEED=1; ETA=0.002;  FROM=10000
     OLD="$PHASE_E/e14_n2_stretched_100k/e14_N2_4.0_ssu_defaults_s1" ;;
  2) TAG=e14s2; PRESET=N2_4.0; SEED=2; ETA=0.002;  FROM=10000
     OLD="$PHASE_E/e14_n2_stretched_100k/e14_N2_4.0_ssu_defaults_s2" ;;
  3) TAG=e15s7; PRESET=N2_4.0; SEED=7; ETA=0.002;  FROM=10000
     OLD="$PHASE_E/e15_n2_seedcheck/e15_N2_4.0_ssu_defaults_s7" ;;
  4) TAG=e15s6; PRESET=N2_4.0; SEED=6; ETA=0.002;  FROM=60000
     OLD="$PHASE_E/e15_n2_seedcheck/e15_N2_4.0_ssu_defaults_s6" ;;
  5) TAG=e18s2; PRESET=N2_4.0; SEED=2; ETA=0.0015; FROM=10000
     OLD="$PHASE_E/e18_n2_stretched_eta0015/e18_N2_4.0_ssu_defaults_eta0.0015_s2" ;;
  6) TAG=d7s0;  PRESET=carbon; SEED=0; ETA=0.02;   FROM=50000; NSTEPS=5000
     OLD="$PHASE_D/d7_headtohead_100k/d7_carbon_ssu_defaults_s0" ;;
  7) TAG=d7s0;  PRESET=H4;     SEED=0; ETA=0.02;   FROM=50000; NSTEPS=5000
     OLD="$PHASE_D/d7_headtohead_100k/d7_H4_ssu_defaults_s0" ;;
  8) TAG=d9s0;  PRESET=N;      SEED=0; ETA=0.02;   FROM=50000; NSTEPS=5000
     OLD="$PHASE_D/d9_atoms_100k/d9_N_ssu_defaults_s0" ;;
  9) TAG=d9s0;  PRESET=O;      SEED=0; ETA=0.02;   FROM=50000; NSTEPS=5000
     OLD="$PHASE_D/d9_atoms_100k/d9_O_ssu_defaults_s0" ;;
  10) TAG=d10s0; PRESET=N2_eq; SEED=0; ETA=0.002;  FROM=50000; NSTEPS=5000
     OLD="$PHASE_D/d10_hometurf_100k/d10_N2_eq_ssu_defaults_s0" ;;
  11) TAG=d10s0; PRESET=CO;    SEED=0; ETA=0.002;  FROM=50000; NSTEPS=5000
     OLD="$PHASE_D/d10_hometurf_100k/d10_CO_ssu_defaults_s0" ;;
esac
ARMTAG=guard
TO=$((FROM + NSTEPS))
RUN_NAME="f5_${PRESET}_${TAG}_${ARMTAG}"

if [[ ! -f "$OLD/checkpoints/${FROM}.npz" ]]; then
  echo "ERROR: $OLD/checkpoints/${FROM}.npz not found." >&2
  exit 1
fi
if [[ -e "$LOGROOT/$RUN_NAME" ]]; then
  echo "ERROR: $LOGROOT/$RUN_NAME already exists; move or delete it first." >&2
  exit 1
fi
# The original's learning rate must match the one hard-coded above: the reloaded
# config.json is not used (it is locked), so the config is rebuilt from preset + flags.
OLD_ETA=$(python -c "import json; c = json.load(open('$OLD/config.json')); \
print(c['vmc']['optimizer']['$SSU']['learning_rate'])")
if ! python -c "import sys; sys.exit(abs($OLD_ETA - $ETA) > 1e-12)"; then
  echo "ERROR: $OLD used learning_rate=$OLD_ETA, this script says $ETA." >&2
  exit 1
fi

export WANDB_TAGS="f5,phase-f,replay,beta-star-guard,center-first,${TAG},${ARMTAG},sys-${PRESET}"
export WANDB_NOTES="Phase F F5 replay of ${PRESET} ${TAG} from its epoch-${FROM} \
checkpoint to ${TO}: beta_star_guard=True, gram_center_first=True, diagnostics on, no \
cap, no safeguard. Original run: ${OLD}."

python -c "import jax; print('jax devices:', jax.devices())"
echo "F5 case=$CASE $PRESET $TAG arm=$ARMTAG eta=$ETA from $FROM to $TO  $(date -Is)"

vmc-molecule \
  --reload.logdir="$OLD" \
  --reload.use_config_file=False \
  --reload.use_checkpoint_file=True \
  --reload.checkpoint_relative_file_path="checkpoints/${FROM}.npz" \
  --reload.append=True \
  --presets.path="preset_configs/${PRESET}.json" \
  --config.logdir="$LOGROOT" \
  --config.save_to_current_datetime_subfolder=False \
  --config.subfolder_name="$RUN_NAME" \
  --config.vmc.optimizer_type="$SSU" \
  --config.vmc.optimizer."$SSU".learning_rate="$ETA" \
  --config.vmc.optimizer."$SSU".mu=0.0 \
  --config.vmc.optimizer."$SSU".schedule_type=inverse_time \
  --config.vmc.optimizer."$SSU".learning_decay_rate=0.0001 \
  --config.vmc.optimizer."$SSU".constrain_norm=True \
  --config.vmc.optimizer."$SSU".norm_constraint=0.001 \
  --config.vmc.optimizer."$SSU".diagnostics=True \
  --config.vmc.optimizer."$SSU".safeguard=False \
  --config.vmc.optimizer."$SSU".carried_cap=0.0 \
  --config.vmc.optimizer."$SSU".gram_center_first=True \
  --config.vmc.optimizer."$SSU".beta_star_guard=True \
  --config.vmc.nepochs="$TO" \
  --config.vmc.checkpoint_every=2000 \
  --config.vmc.check_for_nans=True \
  --config.eval.nchains=1000 \
  --config.eval.nburn=0 \
  --config.eval.nepochs=0 \
  --config.initial_seed="$SEED" \
  --config.wandb.mode=offline \
  --config.wandb.project=vmcnet-phase-f \
  --config.wandb.group="f5-beta-star-guard" \
  --config.wandb.name="$RUN_NAME"

echo "F5 done $(date -Is)"
