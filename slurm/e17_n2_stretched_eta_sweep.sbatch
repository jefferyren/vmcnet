#!/bin/bash
# E17 -- is E14's SS-SPRING divergence on stretched N2 a property of the METHOD, or of
# the (eta, beta) OPERATING POINT it happens to sit at?
#
# WHAT E14 FOUND. All three SS-SPRING seeds on N2 at 4.0 Bohr terminated on NaNs (epochs
# 18572/17342/23559 of 100000) while SPRING(0.99) and PRIME-SR completed 3/3 at the same
# eta=0.002. The diagnosis from the E14 run history: SS-SPRING's adaptive beta settles at
# 0.9965 on this cell, so its ACCUMULATED step eta/(1-beta) is 0.57 -- against 0.20 for
# SPRING(0.99) and 0.042 for PRIME-SR, both of which survive here. Its pre-clip step norm
# never decays (2.28e-3 -> 1.84e-3 -> 2.10e-3 over epochs 2k-10k, versus a 6x decay for
# the same optimizer on N2-eq), so the norm cap rescales essentially every step to ~10.4k,
# and all three seeds die 1-3k epochs after that cap first releases.
#
# THE HYPOTHESIS THIS TESTS. beta is not the free parameter -- eta is. If the instability
# is the accumulated step rather than the method, then halving eta to 0.001 moves this
# cell to 0.28 and quartering it to 0.0005 moves it to 0.14, i.e. BELOW SPRING(0.99)'s
# known-survivable 0.20. Nothing else changes from E14's ssu_defaults cell.
#
# THE CHECK THAT MADE THIS WORTH RUNNING, rather than a guess. The design only works if
# beta does NOT compensate when eta moves. It does not: on carbon at eta 0.005/0.02/0.05
# (D11+E11, medians past epoch 5k) beta is 0.9954/0.9942/0.9939 -- flat to 0.0016 across
# a 10x range, and drifting the WRONG way for compensation. So eta/(1-beta) tracks eta
# nearly proportionally (1.05-1.25 -> 3.3-3.5 -> 7.9-8.8) and halving eta really does
# halve the accumulated step. Had beta risen to 0.998 as eta fell, this experiment would
# be a no-op and must not be run.
#
# WHY ETA AND NOT A BETA CAP. Capping beta at 0.99 would very likely also work -- SPRING
# at exactly that momentum completes 3/3 here -- but it tunes the one knob Claim A exists
# to leave untuned, and SS-SPRING's margins on N2-eq and CO come precisely from its high
# momentum. eta is already a declared, shared setting, and D11/E11 make it the cheap knob
# for THIS method specifically: SS-SPRING's eval error swings only 0.030-0.054 mHa across
# 10x eta, against PRIME-SR's 0.242. Buying stability with eta costs SS-SPRING very
# little accuracy; buying it with beta costs the claim.
#
# *** THIS IS NOT "E14 WITH GENTLER STEPS". READ THIS BEFORE INTERPRETING THE RESULT. ***
# In same_sampled_spring_unified, `phi` evolves independently of eta -- `step_primal`
# depends only on (params, positions, phi, beta), and eta enters ONLY at the final
# `updates = -eta_main * phi_new`. So while constrain_norm binds, the applied update is
# -sqrt(C) * phi/||phi||, EXACTLY eta-independent (this is E2's "the cap masks eta", made
# precise). The pre-clip squared norm scales as eta^2, so at eta=0.001 it is quartered
# (2.1e-3 -> 5.3e-4, already under the 1e-3 cap) and at eta=0.0005 it is cut 16x. The cap
# that bound 100% of E14's steps out to ~10.4k will therefore release inside the first
# ~1k epochs here, and at 0.0005 may never bind after the first few hundred. These runs
# test the FREE dynamics -- which is the point -- but they are a different regime, not a
# milder E14, and the norm_cap_applied trace will look nothing like E14's.
#
# PROBE KNOBS ARE LEFT AT DEFAULTS, INCLUDING THE probe_lr = -1 SENTINEL, which resolves
# to this run's learning_rate. So the probe step halves with eta too. That is deliberate
# -- it keeps the cell identical to E14's in every knob except eta -- but it is also the
# one channel through which beta could still compensate, which is why the logged `mu` is
# the first thing to read regardless of how the runs end.
#
# SEEDS 0-2, REUSED FROM E14 ON PURPOSE, not fresh ones. The question is paired: what
# does this init do at a smaller eta. Fresh seeds would answer a weaker question and
# could not be read against E14 cell for cell.
#
# NO CONTROL ARM. E14 already has this exact cell at eta=0.002 on these three seeds, plus
# SPRING(0.99) and PRIME-SR completing 3/3. Repeating any of them costs GPU-h to confirm
# something already measured.
#
# HOW TO READ THE RESULT:
#   both etas survive 3/3 -> the OPERATING POINT was wrong, not the method. Report the
#                   eval energies against E14's SPRING (14.990 mHa) and PRIME-SR (17.606
#                   mHa) and state the fix as a shared eta, not a per-arm mu. Note that
#                   this is still per-system tuning unless eta is halved everywhere,
#                   which means re-running the other six systems.
#   0.0005 lives, 0.001 dies -> the stability threshold on this geometry sits between
#                   accumulated step 0.14 and 0.28. That brackets it usefully and is
#                   consistent with the step-size reading; SPRING(0.99) at 0.20 surviving
#                   puts the boundary in a narrow window.
#   both die     -> the accumulated step is NOT the mechanism, and the remaining
#                   candidate is the trust-region interaction: constrain_norm scales the
#                   applied update but NOT `phi`, while the next rhs subtracts
#                   A(beta*phi) as though the full step had been taken. Go there next,
#                   with a norm_constraint sweep rather than another eta.
#   ANY outcome  -> check the logged `mu` first. This whole design assumes beta stays near
#                   0.9965 at these etas. If it rises toward 0.998, the probe IS
#                   compensating, eta/(1-beta) never moved, and both the prediction and
#                   the D11/E11 reasoning above need revisiting before anything else in
#                   the result is interpreted.
#
# check_for_nans=True is load-bearing: without it nan_safe masks the NaNs and a dying run
# burns its full allocation producing garbage instead of stopping at the failure epoch.
# A diverged seed produces no eval phase at all, which is an expected outcome here and is
# not a parse error.
#
# Array index -> (eta, seed): idx/3 selects eta from (0.001 0.0005), idx%3 is the seed.
# So --array=0-5 is {0.001} x {0,1,2} then {0.0005} x {0,1,2}.
# ~16 h per surviving run (D10's measured rate on these 14-electron systems) plus ~1 h of
# eval => ~100 GPU-h if none of them diverge. Budget for that pessimistic case; seeds
# that die cost far less, ~3 h each, as E14's three did.
#
# *** PRESET TRAP. *** preset_configs/N2_4.0.json sets eval.nchains=2000, which OOMs at
# 14 electrons on an 11GB 2080TI -- it destroyed E10's entire eval phase. Overridden to
# 1000 below, as D10, E13, E14, E15 and E16 all do. Do not remove it.
#
# Submit from the vmcnet repo root:
#     sbatch --exclude=n0135.savio3 slurm/e17_n2_stretched_eta_sweep.sbatch
# Then, locally, for the stability verdict (reads .out directly, no wandb sync needed):
#     python slurm/divergence_check.py --experiment E14 E17
# and for the eval energy of any seed that survived:
#     python slurm/parse_eval_energies.py --experiment E17 --backfill
#
#SBATCH --job-name=e17-n2-eta
#SBATCH --account=co_esmath
#SBATCH --partition=savio3_gpu
#SBATCH --qos=esmath_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --cpus-per-task=2
#SBATCH --time=36:00:00
#SBATCH --array=0-5
#SBATCH --output=slurm-%x-%A_%a.out
# BAKED IN, unlike every other script in this directory, which only carries the exclude
# in its submit-command comment. One bad GPU on n0135.savio3 shredded 44 of E10's first
# 74 tasks -- dying in 1-57 s in the pre-flight jax.devices() check with
# CUDA_ERROR_NO_DEVICE, while SLURM reported the node healthy and it completed six normal
# 4-hour runs in the same window. This array is ~100 GPU-h and the exclusion is one line;
# do not rely on remembering the flag. NOTE that a command-line --exclude REPLACES this
# rather than merging, so if you ever need to exclude another node, list both.
#SBATCH --exclude=n0135.savio3

set -eo pipefail
export PATH="/global/scratch/users/$USER/envs/vmcnet/bin:$PATH"
set -u

export WANDB_MODE=offline
export WANDB_DIR="/global/scratch/users/$USER/wandb"
mkdir -p "$WANDB_DIR"

cd "$SLURM_SUBMIT_DIR"
LOGROOT="/global/scratch/users/$USER/vmcnet_logs/phase_e/e17_n2_stretched_eta"
mkdir -p "$LOGROOT"

SSU=same_sampled_spring_unified
PRESET=N2_4.0
NEPOCHS=100000

# Index arithmetic MUST match _e17_cell in slurm/parse_eval_energies.py and the E17 entry
# in slurm/divergence_check.py. Change one and you change all three.
ETA_I=$((SLURM_ARRAY_TASK_ID / 3))
SEED=$((SLURM_ARRAY_TASK_ID % 3))   # E14's own seeds, so the comparison is paired
ETAS=(0.001 0.0005)
ETA=${ETAS[$ETA_I]}

RUN_NAME="e17_${PRESET}_ssu_defaults_eta${ETA}_s${SEED}"

export WANDB_TAGS="e17,phase-e,100k,stretched,eta-sweep,follows-e14,stability,arm-ssu_defaults,eta-${ETA},sys-${PRESET}"
export WANDB_NOTES="E17 asks whether E14's SS-SPRING divergence on N2 at 4.0 Bohr is a \
property of the method or of the operating point. E14: NaNs on 3/3 seeds at eta=0.002 \
while SPRING(0.99) and PRIME-SR completed. SS-SPRING's beta settles at 0.9965 there, so \
its accumulated step eta/(1-beta) is 0.57 against 0.20 for SPRING(0.99) and 0.042 for \
PRIME-SR. This halves and quarters eta to 0.001 and 0.0005, putting the accumulated step \
at 0.28 and 0.14, and changes nothing else -- same preset, damping, norm constraint, \
probe knobs, walker count, 100k epochs, eval phase and seeds 0-2, so each cell pairs with \
its E14 twin. Justified by D11/E11, where beta is flat to 0.0016 across a 10x eta range, \
so eta/(1-beta) tracks eta rather than being compensated away. NOTE that while the norm \
cap binds the applied update is exactly eta-independent, and the pre-clip norm scales as \
eta^2, so the cap releases far earlier here than E14's ~10.4k -- these runs probe the \
free dynamics and are not a milder E14. Read the logged mu first: if it rises toward \
0.998 the probe is compensating and the premise fails. A diverged seed produces no eval \
phase, which is expected. eta=${ETA}, seed ${SEED}."

if [[ -e "$LOGROOT/$RUN_NAME" ]]; then
  echo "ERROR: $LOGROOT/$RUN_NAME already exists; move or delete it first." >&2
  exit 1
fi

python -c "import jax; print('jax devices:', jax.devices())"
echo "E17 arm=ssu_defaults preset=$PRESET seed=$SEED eta=$ETA nepochs=$NEPOCHS  $(date -Is)"

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
  --config.wandb.group="e17-n2-eta-${ETA}" \
  --config.wandb.name="$RUN_NAME"

echo "E17 done $(date -Is)"
