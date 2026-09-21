#!/bin/bash
# E18 -- is there ANY learning rate where SS-SPRING is both stable AND competitive on
# stretched N2? This fills in eta=0.0015, between E14/E15's unstable-but-accurate 0.002
# and E17's stable-but-hopeless 0.001.
#
# WHERE THE FRONTIER STANDS (N2 at 4.0 Bohr, SS-SPRING, 100k + eval, seeds as noted):
#   eta      beta      eta/(1-beta)   diverged      eval [mHa above ref]
#   0.002    0.9965    0.57           6/8           12.272 +/- 0.807  (survivors only)
#   0.001    0.9975    0.46           0/3           29.186 +/- 8.206
#   0.0005   0.9980    0.24           0/3           87.705 +/- 18.442
# For reference at eta=0.002: SPRING(0.99) 0/3 diverged, 14.990 mHa; SPRING(0.995) 2/3
# diverged, 18.049 (n=1); PRIME-SR 0/3, 17.606.
#
# WHY 0.0015 SPECIFICALLY, and not another bisection point. Two reasons, and the second
# is the one that makes this worth 50 GPU-h.
#
# (1) It is the only untested point that could still be BOTH. The accuracy cost of
# lowering eta is brutal and superlinear -- halving it from 0.002 to 0.001 cost ~17 mHa,
# halving again cost ~59 more. Anything at or below 0.001 is already known to be far
# worse than every baseline, so 0.0015 is the last candidate.
#
# (2) *** IT SITS ALMOST EXACTLY ON THE NORM-CAP THRESHOLD, WHICH IS THE REAL TEST. ***
# The pre-clip squared update norm scales as eta^2, so relative to E14 this run sees
# (0.0015/0.002)^2 = 0.5625 of it: E14's ~2.0e-3 median over epochs 5-20k becomes
# ~1.1e-3, straddling the norm_constraint of 1e-3. E14 rode the cap on ~100% of steps
# for 10k epochs and diverged 6/8; E17 left the cap by epoch ~2000 (cap fraction
# 0.70-0.78 over 0-500, then 0.00 for the rest) and diverged 0/6. That correlation --
# sustained cap engagement predicts divergence, and the accumulated step does NOT (E17
# at 0.46 survived 3/3 while E16's SPRING(0.995) at 0.40 diverged 2/3) -- is currently
# the leading mechanism, and it is untested at the boundary. This run puts the optimizer
# right on it.
#
# *** DO NOT PREDICT THE ACCUMULATED STEP FROM ETA ALONE. THAT IS WHAT E17 GOT WRONG. ***
# E17 was designed expecting eta/(1-beta) to halve with eta, on the strength of D11/E11
# showing beta flat across a 10x eta range on carbon. It did not: beta ROSE (0.9965 ->
# 0.9975 -> 0.9980) and the accumulated step only went 0.57 -> 0.46 -> 0.24. The carbon
# sweep was measured in the CAP-BOUND regime, where the trajectory is eta-invariant by
# construction and the probe therefore sees identical parameters at every eta; when the
# cap does not bind, beta responds to eta. Expect beta ~0.997 here and an accumulated
# step near 0.50, but READ THE LOGGED mu rather than assuming it.
#
# HONEST PRIOR ON THE ENERGY, stated before the run. Interpolating 12.272 (eta=0.002,
# survivors) and 29.186 (eta=0.001) puts 0.0015 around 18-20 mHa -- which would be WORSE
# than SPRING(0.99)'s 14.990 on this system. So "stable and competitive" requires the
# accuracy curve to be markedly flatter near 0.002 than a straight line through those
# two points. That is possible, because the 12.272 figure is itself survivorship-biased
# upward in quality (2 of 8 seeds) and the eta=0.001 runs were still descending at 100k,
# but it is not the way to bet. A stable run at ~19 mHa is a NEGATIVE result for the
# "tune eta" program and should be reported as one.
#
# SEEDS 0-2, REUSED FROM E14 AND E17 ON PURPOSE. Every cell here pairs with an E14 and
# an E17 cell at the same init, so the eta axis is read within-seed.
#
# NO CONTROL ARM. E14, E15, E16 and E17 already cover eta=0.002 for three optimizers and
# eta=0.001/0.0005 for this one, all at 100k on these seeds.
#
# HOW TO READ THE RESULT:
#   0/3 diverge AND eval < ~15 mHa -> the program works: there is an eta that is both
#                   stable and competitive, and the fix is a shared learning rate. This
#                   is the outcome that rescues the "operating point, not method"
#                   framing. Check the cap trace regardless -- if it also never bound,
#                   that is consistent with the cap mechanism rather than evidence
#                   against it.
#   0/3 diverge AND eval ~18-25 mHa -> stable but NOT competitive, which is the prior
#                   above. The eta program is then closed: every stable eta is worse
#                   than SPRING(0.99), and the next move is the norm_constraint
#                   experiment at eta=0.002, not another point on this axis.
#   ANY diverge  -> the transition sits between 0.001 and 0.0015, which is a tight and
#                   useful bracket. Cross-check the cap fraction against E14's: if the
#                   diverging seeds rode the cap and E17's survivors did not, the
#                   trust-region mechanism is confirmed on a third condition.
#   ALWAYS       -> read `mu` and `norm_cap_applied` before the energy. Those two decide
#                   which mechanism this run supports; the eval number alone does not.
#
# check_for_nans=True is load-bearing: without it nan_safe masks the NaNs and a dying run
# burns its full allocation producing garbage instead of stopping at the failure epoch.
# A diverged seed produces no eval phase, which is an expected outcome and not an error.
#
# Array index IS the seed, 0-2. ~16 h per surviving run (D10's measured rate on these
# 14-electron systems) plus ~1 h of eval => ~50 GPU-h if none diverge.
#
# *** PRESET TRAP. *** preset_configs/N2_4.0.json sets eval.nchains=2000, which OOMs at
# 14 electrons on an 11GB 2080TI -- it destroyed E10's entire eval phase. Overridden to
# 1000 below, as D10, E13, E14, E15, E16 and E17 all do. Do not remove it.
#
# Submit from the vmcnet repo root (the node exclusion is baked into the directives
# below, so the bare command is safe):
#     sbatch slurm/e18_n2_stretched_eta0015.sbatch
# Then, locally, for the stability verdict (reads .out directly, no wandb sync needed):
#     python slurm/divergence_check.py --experiment E14 E17 E18
# and for the eval energies:
#     python slurm/parse_eval_energies.py --experiment E18 --backfill
#
#SBATCH --job-name=e18-n2-eta0015
#SBATCH --account=co_esmath
#SBATCH --partition=savio3_gpu
#SBATCH --qos=esmath_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --cpus-per-task=2
#SBATCH --time=36:00:00
#SBATCH --array=0-2
#SBATCH --output=slurm-%x-%A_%a.out
# BAKED IN rather than left to the submit-command comment above. One bad GPU on
# n0135.savio3 shredded 44 of E10's first 74 tasks, dying in 1-57 s in the pre-flight
# jax.devices() check with CUDA_ERROR_NO_DEVICE while SLURM reported the node healthy.
# Keep this until the node is proven fixed. NOTE that a command-line --exclude REPLACES
# this rather than merging, so to exclude another node as well, list both.
#SBATCH --exclude=n0135.savio3

set -eo pipefail
export PATH="/global/scratch/users/$USER/envs/vmcnet/bin:$PATH"
set -u

export WANDB_MODE=offline
export WANDB_DIR="/global/scratch/users/$USER/wandb"
mkdir -p "$WANDB_DIR"

cd "$SLURM_SUBMIT_DIR"
LOGROOT="/global/scratch/users/$USER/vmcnet_logs/phase_e/e18_n2_stretched_eta0015"
mkdir -p "$LOGROOT"

SSU=same_sampled_spring_unified
PRESET=N2_4.0
NEPOCHS=100000

# Index IS the seed. Must match _e18_cell in slurm/parse_eval_energies.py and the E18
# entry in slurm/divergence_check.py; change one and you change all three.
SEED=$SLURM_ARRAY_TASK_ID          # E14's and E17's own seeds, so the eta axis is paired
ETA=0.0015

RUN_NAME="e18_${PRESET}_ssu_defaults_eta${ETA}_s${SEED}"

export WANDB_TAGS="e18,phase-e,100k,stretched,eta-sweep,follows-e17,follows-e14,stability,arm-ssu_defaults,eta-${ETA},sys-${PRESET}"
export WANDB_NOTES="E18 fills in eta=0.0015 between E14/E15 (eta=0.002: 6/8 diverged, \
12.272 mHa on the two survivors) and E17 (eta=0.001: 0/3 diverged but 29.186 mHa; \
eta=0.0005: 0/3, 87.705 mHa), asking whether any learning rate is both stable and \
competitive on this system. It is the last candidate -- everything at or below 0.001 is \
already far worse than SPRING(0.99)'s 14.990 here. It also sits on the norm-cap \
threshold: pre-clip step norm scales as eta^2, so E14's ~2.0e-3 median over epochs \
5-20k becomes ~1.1e-3 against a norm_constraint of 1e-3. E14 rode the cap ~100% of \
steps for 10k epochs and diverged; E17 left it by epoch ~2000 and did not. Sustained \
cap engagement, not the accumulated step, is the leading mechanism, and this run puts \
the optimizer on the boundary. Do NOT assume eta/(1-beta) scales with eta -- that is \
what E17 got wrong; beta rose 0.9965 -> 0.9975 -> 0.9980 as eta fell, so read the \
logged mu. Honest prior: interpolation puts the energy near 18-20 mHa, i.e. stable but \
WORSE than SPRING(0.99), which would close the eta program. Seeds 0-2, paired with E14 \
and E17 cell for cell. A diverged seed produces no eval phase, which is expected. \
eta=${ETA}, seed ${SEED}."

if [[ -e "$LOGROOT/$RUN_NAME" ]]; then
  echo "ERROR: $LOGROOT/$RUN_NAME already exists; move or delete it first." >&2
  exit 1
fi

python -c "import jax; print('jax devices:', jax.devices())"
echo "E18 arm=ssu_defaults preset=$PRESET seed=$SEED eta=$ETA nepochs=$NEPOCHS  $(date -Is)"

vmc-molecule \
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
  --config.vmc.nepochs="$NEPOCHS" \
  --config.vmc.checkpoint_every=10000 \
  --config.vmc.check_for_nans=True \
  --config.eval.nchains=1000 \
  --config.eval.nburn=10000 \
  --config.eval.nepochs=20000 \
  --config.initial_seed="$SEED" \
  --config.wandb.mode=offline \
  --config.wandb.project=vmcnet-phase-e \
  --config.wandb.group="e18-n2-eta-${ETA}" \
  --config.wandb.name="$RUN_NAME"

echo "E18 done $(date -Is)"
