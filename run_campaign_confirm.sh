#!/bin/bash
# CONFIRMATORY campaign (PREREGISTRATION.md Amendment 2, frozen 2026-07-30), diya.
#
# Tier 1 of the meta-audit plan (MCB_META_AUDIT.md Part 4): fix the
# measurement channel, then re-judge the existing v3 arms on data and metrics
# that can actually support a verdict:
#   * FRESH confirmatory ICs (seed0=3000; the 1000-1019 set is retired —
#     13 gate tests + ~123 selection looks exhausted it)
#   * registered metric: final-10-day time-mean dSST (not day-60 snapshot)
#   * k=8 micro-ensembles per (arm, IC) with per-member IN-PROCESS baselines
#     -> per-run chaos noise (~0.014-0.017 K) shrinks by sqrt(8); detection
#     floor drops from ~12 mK to ~4-5 mK at n=20
#   * formal paired t / Wilcoxon + TOST equivalence bounds
#   * cross-process noise floor (what sigma_compile=0 could not see)
#
# Expected outcome: either a significant primary comparison, or a TIGHT
# (<~5 mK) equivalence bound on the feedback benefit — both defensible.
# Runtime estimate: ~15 min (ICs) + ~1 h (noise floor) + ~3 h (eval) ~= 4-5 h.
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
ICS=$E/ics_confirm
S1PAT=$E/stage1_v2/stage1_optimized_pattern.pkl
DAYS=60
CI=15
CAP=0.09

step "0/3 Smoke test (tiny: 2 ICs x 2 members x 4 days)"
$PY run_confirmatory_eval.py --ic-dir $E/ics_independent --split heldout \
  --arm static=static=$S1PAT \
  --days 4 --control-interval 2 --tail-days 2 --members 2 \
  --max-perturbation $CAP --output /tmp/confirm_smoke.pkl \
  || { echo "SMOKE FAILED — aborting"; exit 1; }

step "1/3 Fresh confirmatory ICs (seed0 3000, 20 held-out, NEVER used before)"
if [ ! -f $ICS/manifest.json ]; then
  $PY run_generate_ics_independent.py \
    --base-carry $E/equilibrated/base_carry.pkl \
    --num-train 0 --num-heldout 20 --decorr-days 30 --horizon $DAYS \
    --seed0 3000 --output-dir $ICS || { echo "IC GEN FAILED"; exit 1; }
else
  echo "  $ICS exists — reusing"
fi

step "2/3 Cross-process noise floor (the real per-run noise, 6 workers x 10 ICs)"
$PY run_noise_floor.py --ic-dir $ICS --stage1 $S1PAT \
  --days $DAYS --control-interval $CI --max-perturbation $CAP \
  --cross-process --reps 6 --num-ics 10 --regen-baseline \
  --output $E/noise_floor_crossproc.pkl || echo "NOISE FLOOR FAILED (continuing)"

step "3/3 Confirmatory micro-ensemble eval (3 arms, k=8, registered metric)"
$PY run_confirmatory_eval.py --ic-dir $ICS --split heldout \
  --arm static=static=$S1PAT \
  --arm retrain=fc13=$E/retrain_v3/stage5_trained_policy.pkl \
  --arm openloop=time-only=$E/ablation_openloop_v3/stage5_trained_policy.pkl \
  --comparator static --members 8 --member-perturb-amp 0.001 \
  --days $DAYS --control-interval $CI --tail-days 10 \
  --max-perturbation $CAP --target-cooling -0.1 \
  --output $E/confirmatory_eval.pkl || { echo "CONFIRM EVAL FAILED"; exit 1; }

echo ""
echo "[$(ts)] ========== CONFIRMATORY CAMPAIGN COMPLETE =========="
echo "Artifacts: $E/ics_confirm, $E/noise_floor_crossproc.pkl, $E/confirmatory_eval.pkl"
