#!/bin/bash
# Step 0 (macro starting states) + Experiment 1 (gradient fidelity) for the
# long-horizon gradient study (PREREGISTRATION.md "Amendment 9", frozen
# 2026-09-29, and its revisions 0.1-0.4). Run on the GPU machine only after the
# Amendment-9 commit is pushed and posted to OSF.
#
# Question (Experiment 1): does cutting gradient flow through the chaotic
# atmosphere every W days, while keeping it through the slab ocean, give
# gradients that match the ensemble-mean ("climate") sensitivity out to
# 60-120 days, where ordinary backpropagation should not?
#
# Steps (each gated; re-running skips anything already on disk):
#   0/5 smoke: plumbing check of the three drivers on a cold start
#   1/5 Q-flux base climate (revisions 0.2-0.3): settle 10 years from the committed
#       Q-flux file and apply the registered gate (~0.3 GPU-h); abort on FAIL
#   2/5 macro starting states: 16 macro states x registered branches (~1.4 GPU-h)
#   3/5 Experiment 1 truth + estimators on ics_macro/exp1 (~12.2 GPU-h,
#       including ~2.4 for the exploratory damped estimators, revision 0.1,
#       and ~1.5 for the forward-mode map Jacobians, revision 0.4)
#   4/5 pre-committed analysis -> exp1_gradient_fidelity_analysis.json
#
# Cost: ~14 GPU-h in total. vLLM is stopped with a restart trap, as in every
# earlier campaign.
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
QFLUX=mcb_experiments/qflux/qflux_monthly_t30_v2.nc  # corrected Q-flux (revision 0.3, PASS)
QDIR=$E/equilibrated_qflux
BASE=$QDIR/base_carry.pkl
ICS=$E/ics_macro
OUT=$E/exp1_gradient_fidelity
SMOKE=/tmp/step0_smoke

[ -f $QFLUX ] || { echo "MISSING $QFLUX (committed with revision 0.3) — git pull"; exit 1; }

step "0/5 Smoke: plumbing of the three drivers (cold start, 2-4 days)"
rm -rf $SMOKE && mkdir -p $SMOKE
cat > $SMOKE/plan.json <<'EOF'
{"exp1": [{"macro": [0, 2], "branches": [0], "split": "heldout", "horizon": 3}],
 "exp2_eval": [{"macro": [1, 3], "branches": [0], "split": "heldout", "horizon": 3}]}
EOF
$PY run_generate_macro_ics.py --base-carry cold --num-macro 4 \
  --spacing-days 2 --decorr-days 1 --plan-json $SMOKE/plan.json \
  --output-root $SMOKE/ics || { echo "SMOKE (macro ICs) FAILED"; exit 1; }
$PY run_gradient_fidelity.py --ic-dir $SMOKE/ics/exp1 --fd-members 2 \
  --grad-members 1 --horizons 2 3 --tail-days 1 --windows 1 0 \
  --damped-efold-days 3 --band-centers 20 -20 --map-block-days 1 \
  --output $SMOKE/exp1 \
  || { echo "SMOKE (Experiment 1) FAILED"; exit 1; }
$PY analyze_gradient_fidelity.py $SMOKE/exp1 --n-boot 50 >/dev/null \
  || { echo "SMOKE (analysis) FAILED"; exit 1; }
# 4 days cannot pass the gate; exit 0 or 3 (gate verdict) both mean plumbing works.
$PY run_qflux_base_climate.py settle --qflux $QFLUX --days 4 --eval-days 2 \
  --chunk-days 2 --output-dir $SMOKE/qflux --base-carry-out $SMOKE/qflux/base.pkl
rc=$?; [ $rc -eq 0 ] || [ $rc -eq 3 ] || { echo "SMOKE (Q-flux settle) FAILED"; exit 1; }

step "1/5 Q-flux base climate: settle 10 years + registered gate (revision 0.2)"
if [ ! -f $BASE ]; then
  $PY run_qflux_base_climate.py settle --qflux $QFLUX --days 3650 \
    --eval-days 1460 --output-dir $QDIR --base-carry-out $BASE
  rc=$?
  if [ $rc -ne 0 ]; then
    [ -f $BASE ] && mv $BASE $BASE.gate_failed
    echo "Q-FLUX BASE CLIMATE FAILED (exit $rc; 3 = registered gate) — see"
    echo "$QDIR/settle_summary.json. Step 0 NOT run (revision 0.2 fallback:"
    echo "discuss before any new attempt)."
    exit 1
  fi
else echo "  $BASE exists — reusing"; fi

step "2/5 Macro starting states (registered plan, 16 x 730 d, Q-flux climate)"
if [ ! -f $ICS/exp3_eval/manifest.json ]; then
  $PY run_generate_macro_ics.py --base-carry $BASE --num-macro 16 \
    --spacing-days 730 --decorr-days 30 --perturb-amp 0.05 --seed0 12000 \
    --require-qflux --output-root $ICS || { echo "MACRO ICs FAILED"; exit 1; }
else echo "  $ICS exists — reusing"; fi

step "3/5 Experiment 1: truth (8 ICs x 4 members) + estimators (W 1/7/14/full; damped tau 3/7, exploratory) + maps (rev 0.4)"
if [ ! -f $OUT.json ] || ! grep -q '"finished_utc"' $OUT.json; then
  $PY run_gradient_fidelity.py --ic-dir $ICS/exp1 \
    --fd-members 4 --grad-members 1 --member-seed0 91000 \
    --a0 0.03 --delta 0.03 --horizons 15 30 60 120 --tail-days 10 \
    --windows 1 7 14 0 --damped-efold-days 3 7 --align episode \
    --map-block-days 5 \
    --output $OUT \
    || { echo "EXPERIMENT 1 FAILED (partial arrays kept in $OUT.npz)"; exit 1; }
else echo "  $OUT complete — reusing"; fi

step "4/5 Pre-committed analysis"
$PY analyze_gradient_fidelity.py $OUT || { echo "ANALYSIS FAILED"; exit 1; }

echo ""
echo "[$(ts)] ========== STEP 0 + EXPERIMENT 1 COMPLETE =========="
echo "Artifacts: $ICS/, $OUT.npz, ${OUT}_maps.npz, $OUT.json, ${OUT}_analysis.json"
