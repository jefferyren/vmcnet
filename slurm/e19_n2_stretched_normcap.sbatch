#!/bin/bash
# E19 -- THE DISCRIMINATING RUN. Does SS-SPRING diverge on stretched N2 because the norm
# cap's interaction with the momentum buffer is broken, or simply because its raw step is
# too big? Every measurement so far fits BOTH, and this is the experiment that separates
# them. It is also the only outstanding run that could yield a FIX rather than a
# diagnosis, because it keeps eta=0.002 and therefore keeps the accuracy.
#
# WHAT E14-E18 ESTABLISHED.
# (1) The accumulated step eta/(1-beta) does NOT predict divergence. Over all 12
#     SS-SPRING runs on this system: diverged 0.361-0.528, survived 0.198-0.525 -- almost
#     complete overlap. The two largest in the whole set are E18 seed 2 (0.528, DIVERGED)
#     and E18 seed 0 (0.525, SURVIVED). That hypothesis is dead; do not revive it.
# (2) What separates perfectly is the FRACTION OF THE FIRST 12k EPOCHS WITH THE NORM CAP
#     BINDING: diverged 68/80/86/88%, survived 0/0/1/3/3/4/35/37%. No overlap. The
#     decisive comparison is within E18, where all three seeds share eta AND beta (s0
#     0.99714, s2 0.99716) and differ only by init: 35% capped survived, 80% diverged.
# (3) The eta axis is closed. eta=0.002 6/8 diverged at 12.272 mHa; 0.0015 1/3 at 17.216;
#     0.001 0/3 at 29.186; 0.0005 0/3 at 87.705. Every STABLE eta is worse than
#     SPRING(0.99)'s 14.990 on this system, so there is no fix on that axis.
#
# *** WHY (2) IS NOT YET A MECHANISM. *** Cap fraction is BY CONSTRUCTION "fraction of
# steps whose pre-clip norm exceeds C", so it is collinear with the raw step norm
# (diverged 9.5e-4 to 1.7e-3 over epochs 5-12k; survived 2.5e-5 to 5.0e-4). Two readings
# fit the data identically and make opposite predictions here:
#   (a) MISMATCH. constrain_norm scales the applied update but NOT `phi`, while the next
#       rhs subtracts A(beta*phi) as though the full step had been taken. Sustained
#       capping therefore accumulates an inconsistency between the SPRING recursion's
#       assumed trajectory and the realized one. Predicts: remove the sustained capping
#       and the run is STABLE, at unchanged accuracy.
#   (b) BIG STEP. A large raw step causes both the capping and the divergence; the cap is
#       a symptom and was in fact holding the run together. Predicts: remove it and the
#       run diverges AS BADLY OR WORSE, and probably sooner.
# E19 breaks the collinearity by holding eta at 0.002 -- so the raw step stays exactly as
# large as E14's, pre-clip median ~2e-3 -- while removing the sustained capping.
#
# TWO ARMS, because "loosen" and "remove" are different claims.
#   arm 0, norm_constraint=1e-2: a brake that is PRESENT but inactive in normal running.
#       E14's pre-clip norm peaks ~6.7e-3 at epoch 200 and medians ~2e-3 thereafter, all
#       below 1e-2, so this should never bind -- except on a genuine blow-up, which it
#       still catches.
#   arm 1, constrain_norm=False: NO brake at all.
# If arm 0 survives and arm 1 dies, the cap is doing real work as a rare emergency brake
# and the E14 pathology is SUSTAINED binding specifically -- a more interesting and more
# publishable statement than either (a) or (b) alone.
#
# *** THE EARLY-PHASE CONFOUND, AND WHY ARM 0 EXISTS. *** E2 found 10 of 15
# constraint-off runs dying inside ~160 epochs at eta >= 0.005 -- random-init chaos, NOT
# the asymptotic mechanism this experiment is about. At eta=0.002 the init-phase pre-clip
# (~6.7e-3) sits under arm 0's 1e-2 but has nothing above it in arm 1. *** A death inside
# the first ~500 epochs in arm 1 is the E2 phenomenon and MUST NOT be read as confirming
# (b). *** Only a death in the 10k-20k window, where E14's seeds went, speaks to this
# question. Arm 0 is the control that tells the two apart.
#
# *** METRIC GOTCHA -- norm_cap_applied IS USELESS IN BOTH ARMS FOR CROSS-COMPARISON. ***
# In update_param_fns.get_update_norm_diagnostics it is hard-wired to 0.0 whenever
# constrain_norm is False, so arm 1 logs a flat zero that means "no cap configured", not
# "step was small". And arm 0 reports binding against 1e-2, not against E14's 1e-3, so
# its trace is not comparable to E14's either. IN BOTH ARMS the quantity to compute is
# the fraction of steps with update_sq_norm_preclip > 1e-3 -- the COUNTERFACTUAL cap
# fraction, i.e. how often the old cap WOULD have bound. That is the number to put next
# to E14's 68-88% and E17/E18's 0-37%.
#
# SEEDS 0-2, REUSED FROM E14 ON PURPOSE. Those three inits diverged 3/3 under the old
# cap, so each cell here is a paired counterfactual on an init known to fail.
#
# HOW TO READ THE RESULT:
#   both arms 0/3 diverge  -> reading (a). The mismatch is the mechanism, and this is a
#                   FIX: stability at eta=0.002, so compare the eval energies against
#                   E14's SS-SPRING survivors (12.272 mHa), SPRING(0.99) (14.990) and
#                   PRIME-SR (17.606). If it holds at ~12 mHa with 3/3 stability it is
#                   the best arm on this system and the paper's story changes entirely.
#   both arms diverge in the 10k-20k window -> reading (b). The raw step is the cause and
#                   the trust region was load-bearing. The fix must then live in beta or
#                   the probe, not in the constraint, and the honest table cell for this
#                   system stays "diverged".
#   arm 0 survives, arm 1 dies LATE (10k-20k) -> the cap is a genuine emergency brake and
#                   the pathology is sustained binding. Best outcome for understanding;
#                   report it as such.
#   arm 1 dies inside ~500 epochs -> that cell is the E2 init-chaos artifact, NOT a result
#                   about this mechanism. Read arm 0 alone and say so explicitly.
#   ALWAYS       -> compute the counterfactual cap fraction (see the gotcha above) and
#                   check `mu`. If beta moves off ~0.9955 when the cap stops binding, the
#                   probe is responding to the constraint change and the arms are not the
#                   clean contrast they look like.
#
# check_for_nans=True is load-bearing: without it nan_safe masks the NaNs and a dying run
# burns its full allocation producing garbage instead of stopping at the failure epoch.
# A diverged seed produces no eval phase, which is expected here and is not an error.
#
# Array index -> (arm, seed): idx/3 selects the arm, idx%3 is the seed. --array=0-5 is
# {C=1e-2} x {0,1,2} then {no cap} x {0,1,2}.
# ~16 h per surviving run plus ~1 h of eval => ~100 GPU-h if none diverge.
#
# IF THE ANSWER IS AMBIGUOUS, the follow-up is a dose-response arm at norm_constraint
# =3e-3, which should land a counterfactual cap fraction between E14's ~88% and E19's
# ~0% and test the 37-68% threshold directly. Do not add it pre-emptively; the primary
# question here is binary.
#
# *** PRESET TRAP. *** preset_configs/N2_4.0.json sets eval.nchains=2000, which OOMs at
# 14 electrons on an 11GB 2080TI -- it destroyed E10's entire eval phase. Overridden to
# 1000 below, as D10 and E13-E18 all do. Do not remove it.
#
# Submit from the vmcnet repo root (node exclusion is baked into the directives):
#     sbatch slurm/e19_n2_stretched_normcap.sbatch
# Then, locally:
#     python slurm/divergence_check.py --experiment E14 E19
#     python slurm/parse_eval_energies.py --experiment E19 --backfill
#
#SBATCH --job-name=e19-n2-normcap
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
LOGROOT="/global/scratch/users/$USER/vmcnet_logs/phase_e/e19_n2_stretched_normcap"
mkdir -p "$LOGROOT"

SSU=same_sampled_spring_unified
PRESET=N2_4.0
NEPOCHS=100000
ETA=0.002                      # UNCHANGED from E14 -- the raw step must stay large

# Index arithmetic MUST match _e19_cell in slurm/parse_eval_energies.py and the E19 entry
# in slurm/divergence_check.py. Change one and you change all three.
ARM=$((SLURM_ARRAY_TASK_ID / 3))
SEED=$((SLURM_ARRAY_TASK_ID % 3))   # E14's own seeds, which diverged 3/3 under C=1e-3

case "$ARM" in
  0) TAG=ssu_cap0.01 ; CONSTRAIN=True  ; NORMC=0.01 ;;
  1) TAG=ssu_nocap   ; CONSTRAIN=False ; NORMC=0.001 ;;   # NORMC unused when off
  *) echo "ERROR: bad ARM $ARM" >&2; exit 1 ;;
esac

RUN_NAME="e19_${PRESET}_${TAG}_s${SEED}"

export WANDB_TAGS="e19,phase-e,100k,stretched,normcap,follows-e18,follows-e14,mechanism,arm-${TAG},eta-${ETA},sys-${PRESET}"
export WANDB_NOTES="E19 separates the two mechanisms that fit every E14-E18 measurement \
equally well. Cap fraction over the first 12k epochs separates diverged (68-88%) from \
survived (0-37%) perfectly, including within E18 where eta and beta are matched and only \
the init differs -- but cap fraction is collinear with the raw step norm by construction, \
so it cannot distinguish (a) the constrain_norm/phi MISMATCH, where the cap scales the \
applied update but not the momentum buffer while the next rhs subtracts A(beta*phi) as if \
the full step were taken, from (b) a LARGE RAW STEP causing both the capping and the \
divergence. This holds eta at 0.002, so the raw step stays exactly as large as E14's, and \
removes the sustained capping: arm ssu_cap0.01 raises norm_constraint to 1e-2 (a brake \
present but inactive, since E14's pre-clip peaks ~6.7e-3), arm ssu_nocap disables it. (a) \
predicts stable at unchanged accuracy, which would be a FIX worth 12.272 mHa against \
SPRING(0.99)'s 14.990; (b) predicts divergence as bad or worse. Seeds 0-2 are E14's own, \
which diverged 3/3, so each cell is a paired counterfactual. CAUTION: a death inside the \
first ~500 epochs in the nocap arm is E2's random-init chaos, not this mechanism. NOTE \
that norm_cap_applied is hard-wired to 0 when constrain_norm is False and measures \
against 1e-2 in the other arm, so the comparable quantity is the fraction of steps with \
update_sq_norm_preclip > 1e-3. Arm ${TAG}, seed ${SEED}."

if [[ -e "$LOGROOT/$RUN_NAME" ]]; then
  echo "ERROR: $LOGROOT/$RUN_NAME already exists; move or delete it first." >&2
  exit 1
fi

python -c "import jax; print('jax devices:', jax.devices())"
echo "E19 arm=$TAG preset=$PRESET seed=$SEED eta=$ETA constrain_norm=$CONSTRAIN norm_constraint=$NORMC nepochs=$NEPOCHS  $(date -Is)"

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
  --config.vmc.optimizer."$SSU".constrain_norm="$CONSTRAIN" \
  --config.vmc.optimizer."$SSU".norm_constraint="$NORMC" \
  --config.vmc.nepochs="$NEPOCHS" \
  --config.vmc.checkpoint_every=10000 \
  --config.vmc.check_for_nans=True \
  --config.eval.nchains=1000 \
  --config.eval.nburn=10000 \
  --config.eval.nepochs=20000 \
  --config.initial_seed="$SEED" \
  --config.wandb.mode=offline \
  --config.wandb.project=vmcnet-phase-e \
  --config.wandb.group="e19-n2-normcap-${TAG}" \
  --config.wandb.name="$RUN_NAME"

echo "E19 done $(date -Is)"
