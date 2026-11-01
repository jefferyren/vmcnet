#!/bin/bash
# H1 step 3 -- freeze theta at snapshots of the two dense replays (h1_n2_dense_replay.
# sbatch) and run Kaczmarz++ (the SS-SPRING probe recursion, MCMC supplying fresh rows)
# on each frozen system. Plan and reading: the docstring of slurm/h1_frozen_probe.py.
#
# SNAPSHOTS (array index = 5*run + snapshot): the earliest replay checkpoint
# ("healthy") and the checkpoints at or before event-500, -200, -50, -20, where the
# event is each replay's own first catastrophe. Check the picks before submitting:
#     python slurm/h1_frozen_probe.py --logdir <replay logdir> --list
# (a snapshot that collapses onto an earlier one is skipped, and its task exits 0).
#
# ARMS per snapshot, 2 targets x* each: beta fixed at 0.99, 0.995 and the run's own
# beta at that checkpoint, plus a FRESH adaptive controller. 3000 frozen steps.
#
# COST. One Gram + eigh per step shared by the 8 probes, no local energies: roughly a
# training step, ~0.5 s => ~25 min per task, 10 tasks ~4 GPU-h.
#
# Submit AFTER both replays have diverged, from the vmcnet repo root:
#     sbatch slurm/h1_frozen_probe.sbatch
# READOUT, afterwards, on the login node:
#     python slurm/h1_frozen_probe.py --summary \
#         /global/scratch/users/$USER/vmcnet_logs/phase_h/h1_frozen
#
#SBATCH --job-name=h1-frozen-probe
#SBATCH --account=co_esmath
#SBATCH --partition=savio3_gpu
#SBATCH --qos=esmath_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --array=0-9
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --exclude=n0135.savio3

set -eo pipefail
export PATH="/global/scratch/users/$USER/envs/vmcnet/bin:$PATH"
set -u

cd "$SLURM_SUBMIT_DIR"
REPLAY="/global/scratch/users/$USER/vmcnet_logs/phase_h/h1_dense_replay"
OUT="/global/scratch/users/$USER/vmcnet_logs/phase_h/h1_frozen"
mkdir -p "$OUT"

RUN=$((SLURM_ARRAY_TASK_ID / 5))
SNAP=$((SLURM_ARRAY_TASK_ID % 5))
case "$RUN" in
  0) TAG=e14s0 ;;
  1) TAG=e15s7 ;;
esac
LOGDIR="$REPLAY/h1_N2_4.0_${TAG}_dense"

if [[ ! -f "$LOGDIR/config.json" ]]; then
  echo "ERROR: $LOGDIR not found; run h1_n2_dense_replay.sbatch first." >&2
  exit 1
fi

python -c "import jax; print('jax devices:', jax.devices())"
echo "H1 frozen probe run=$TAG snapshot=$SNAP  $(date -Is)"

python slurm/h1_frozen_probe.py \
  --logdir "$LOGDIR" \
  --snapshot-index "$SNAP" \
  --offsets 500,200,50,20 \
  --betas 0.99,0.995,run \
  --nxstar 2 \
  --nsteps 3000 \
  --out "$OUT"

echo "H1 frozen probe done $(date -Is)"
