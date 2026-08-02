#!/bin/bash
# TIER-2 campaign: feedback under uncertain MCB efficacy
# (PREREGISTRATION.md Amendment 3, frozen 2026-07-31), diya.
#
# Design: each episode draws an UNOBSERVED efficacy eta ~ U[0.6, 1.4];
# applied forcing = eta x command. Static/open-loop cannot compensate
# (expected static error ~20 mK >> 1.5 mK detection floor); feedback arms
# (NN, PI) can infer eta from realized cooling in the anomaly features.
# Primary hypotheses (Holm family of 2): NN-feedback beats static (H2) and
# beats the trained open-loop schedule (H3).
#
# Cost estimate: ICs ~20 min; manipulation check + widening probe ~1 h;
# 6 trainings (2 arms x 3 seeds) x ~4 h (+~5 min baseline regen each);
# final eval (8 arms x 20 ICs x k=4 = 720 rollouts) ~3.5 h.
# TOTAL ~28-30 h. The manipulation check gates the expensive steps.
CONTAINER=aeon-vllm
ts() { date +%H:%M:%S; }
step() { echo ""; echo "[$(ts)] ========== $1 =========="; }

echo "[$(ts)] stopping $CONTAINER"
docker stop $CONTAINER >/dev/null 2>&1
trap 'echo "[$(ts)] restarting '"$CONTAINER"'"; docker start '"$CONTAINER"' >/dev/null 2>&1' EXIT INT TERM

cd ~/workspace/jax-gcm || exit 1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
export PYTHONUNBUFFERED=1
PY=~/mcb-env/bin/python

E=mcb_experiments_gpu
BASE=$E/equilibrated/base_carry.pkl
S1PAT=$E/stage1_v2/stage1_optimized_pattern.pkl
TRAIN_ICS=$E/ics_t2_train      # train seeds 4000-4009 + validation (heldout split) 4010-4015
EVAL_ICS=$E/ics_t2_eval        # confirmatory n=20 (seed0 5000)
DAYS=60
CI=15
CAP=0.09
ERANGE="0.6 1.4"
EPOCHS=40
NONE=/tmp/nonexistent_ckpt.pkl

step "0/6 Smoke: full Tier-2 eval path (tiny; on the SPENT ics_confirm set)"
# Uses Tier-1's already-spent confirmatory ICs so the fresh 4000/5000 sets
# stay untouched until the real steps. Exercises eta-randomization + PI arm
# + efficacy threading on the real model end to end.
$PY run_confirmatory_eval.py --ic-dir $E/ics_confirm --split heldout \
  --arm static=static=$S1PAT --arm pi=pi=$S1PAT \
  --efficacy-mode randomized --efficacy-range $ERANGE --efficacy-seed 999 \
  --members 2 --days 4 --control-interval 2 --tail-days 2 \
  --max-perturbation $CAP --output /tmp/t2_smoke.pkl \
  || { echo "SMOKE FAILED — aborting"; exit 1; }

step "1/6 Fresh Tier-2 IC sets"
if [ ! -f $TRAIN_ICS/manifest.json ]; then
  # 10 train + 6 "heldout" = the VALIDATION split (selection/early-stop only)
  $PY run_generate_ics_independent.py --base-carry $BASE \
    --num-train 10 --num-heldout 6 --decorr-days 30 --horizon $DAYS \
    --seed0 4000 --output-dir $TRAIN_ICS || { echo "IC GEN FAILED"; exit 1; }
else echo "  $TRAIN_ICS exists — reusing"; fi
if [ ! -f $EVAL_ICS/manifest.json ]; then
  $PY run_generate_ics_independent.py --base-carry $BASE \
    --num-train 0 --num-heldout 20 --decorr-days 30 --horizon $DAYS \
    --seed0 5000 --output-dir $EVAL_ICS || { echo "IC GEN FAILED"; exit 1; }
else echo "  $EVAL_ICS exists — reusing"; fi

step "2/6 MANIPULATION CHECK: static under eta-randomization (validation ICs)"
$PY run_confirmatory_eval.py --ic-dir $TRAIN_ICS --split heldout \
  --arm static=static=$S1PAT \
  --efficacy-mode randomized --efficacy-range $ERANGE --efficacy-seed 921 \
  --efficacy-antithetic \
  --members 2 --days $DAYS --control-interval $CI --tail-days 10 \
  --max-perturbation $CAP --output $E/t2_manipulation_check.pkl \
  || { echo "MANIPULATION CHECK FAILED TO RUN"; exit 1; }
# Gate: Amendment 3 requires mean |err| > 3x the Tier-1 static error
# (3 x 0.0054 = 0.0162 K). Antithetic draws give deterministic expected
# ~0.020-0.025 K, so the false-stop risk is small (<2%).
$PY - <<'PYEOF' || { echo "MANIPULATION CHECK NOT PASSED — stopping for redesign"; exit 1; }
import pickle, numpy as np
r = pickle.load(open("mcb_experiments_gpu/t2_manipulation_check.pkl", "rb"))
err = np.abs(r["per_ic"]["static"]["dsst_10d"] + 0.1)
print(f"manipulation check: static mean|err| under eta-rand = {err.mean():.4f} K "
      f"(threshold 0.0162 = 3x Tier-1 per Amendment 3)")
assert err.mean() > 0.0162
PYEOF

# Widened-deployment efficiency probe (2026-07-31 review): a uniform ocean
# field at the static pattern's mean forcing (0.0265) and at the boosted
# level a low-eta episode needs (0.0442), at eta=1 on the validation ICs.
# Measures how much cooling the cells OUTSIDE the optimized pattern deliver
# before the training spend relies on them. Reported, ungated.
$PY run_confirmatory_eval.py --ic-dir $TRAIN_ICS --split heldout \
  --arm static=static=$S1PAT \
  --arm uniform_match=uniform=0.0265 \
  --arm uniform_boost=uniform=0.0442 \
  --members 2 --days $DAYS --control-interval $CI --tail-days 10 \
  --max-perturbation $CAP --output $E/t2_widening_probe.pkl \
  || echo "WIDENING PROBE FAILED (continuing; reported, ungated)"

step "3/6 Train NN-feedback (fc13, warm-started, eta-randomized) x 3 seeds"
for SEED in 42 43 44; do
  OUT=$E/t2_feedback_s$SEED
  [ -f $OUT/stage5_trained_policy.pkl ] && { echo "  $OUT exists — skip"; continue; }
  $PY run_stage5_training.py --ic-dir $TRAIN_ICS --days $DAYS \
    --control-interval $CI --epochs $EPOCHS --max-perturbation $CAP \
    --warm-start-stage1 $S1PAT --loss-mode tail_dsst --select-on-heldout \
    --efficacy-range $ERANGE --efficacy-seed $((910 + SEED)) --seed $SEED \
    --efficacy-resample-per-epoch --regen-baselines \
    --output-dir $OUT || { echo "FEEDBACK TRAIN s$SEED FAILED"; exit 1; }
done

step "4/6 Train open-loop (time-only, random init, eta-randomized) x 3 seeds"
for SEED in 42 43 44; do
  OUT=$E/t2_openloop_s$SEED
  [ -f $OUT/stage5_trained_policy.pkl ] && { echo "  $OUT exists — skip"; continue; }
  $PY run_stage5_training.py --ic-dir $TRAIN_ICS --days $DAYS \
    --control-interval $CI --epochs $EPOCHS --max-perturbation $CAP \
    --feature-mode time-only --loss-mode tail_dsst --select-on-heldout \
    --init-checkpoint $NONE --allow-random-init \
    --efficacy-range $ERANGE --efficacy-seed $((910 + SEED)) --seed $SEED \
    --efficacy-resample-per-epoch --regen-baselines \
    --output-dir $OUT || { echo "OPENLOOP TRAIN s$SEED FAILED"; exit 1; }
done

step "5/6 Confirmatory Tier-2 eval (8 arms, 20 fresh ICs, k=4, eta seed 920)"
$PY run_confirmatory_eval.py --ic-dir $EVAL_ICS --split heldout \
  --arm static=static=$S1PAT \
  --arm pi=pi=$S1PAT \
  --arm feedback_s42=fc13=$E/t2_feedback_s42/stage5_trained_policy.pkl \
  --arm feedback_s43=fc13=$E/t2_feedback_s43/stage5_trained_policy.pkl \
  --arm feedback_s44=fc13=$E/t2_feedback_s44/stage5_trained_policy.pkl \
  --arm openloop_s42=time-only=$E/t2_openloop_s42/stage5_trained_policy.pkl \
  --arm openloop_s43=time-only=$E/t2_openloop_s43/stage5_trained_policy.pkl \
  --arm openloop_s44=time-only=$E/t2_openloop_s44/stage5_trained_policy.pkl \
  --comparator static --members 4 --member-perturb-amp 0.001 \
  --efficacy-mode randomized --efficacy-range $ERANGE --efficacy-seed 920 \
  --efficacy-antithetic \
  --days $DAYS --control-interval $CI --tail-days 10 \
  --max-perturbation $CAP --target-cooling -0.1 \
  --output $E/tier2_eval.pkl || { echo "TIER2 EVAL FAILED"; exit 1; }

step "6/6 Pre-committed primary analysis (analyze_tier2.py)"
$PY analyze_tier2.py $E/tier2_eval.pkl || echo "ANALYSIS FAILED (data safe in tier2_eval.pkl)"
echo ""
echo "[$(ts)] ========== TIER-2 CAMPAIGN COMPLETE =========="
echo "Artifacts: $E/{t2_manipulation_check.pkl, t2_feedback_s*, t2_openloop_s*, tier2_eval.pkl}"
