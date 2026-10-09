#!/bin/bash
# Experiment 3c pilot: 3b's pilot states, planners that see the ocean but
# not the weather (MCB_PROJECT_REPORT.md Part 28; revision 3 is frozen only
# after this).
#
# 16 states (exp3b_train branch 1), each with its 3b hidden strength, 3
# members, whole episodes in 3b's test world, the old land model (3b's).
#   A. The ocean-only oracle planner (told the true strength): its forecast
#      misses set the learner's noise level by 3b's rule.
#   B. The ocean-only learner and the ocean-only naive planner.
#   C. Summary against 3b's pilot runs of the arms that never used the
#      weather and of 3b's exact-sensing planners (analyze_exp3c_pilot.py).
# Re-running skips every run whose summary exists.
ts() { date +%H:%M:%S; }

cd ~/workspace/jax-gcm || exit 1
unset JCM_LAND_CLIMATOLOGY                  # 3b's (old) land model
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.12
export PYTHONUNBUFFERED=1
PY=~/mcb-env/bin/python
E=mcb_experiments_gpu
ICS=$E/ics_macro3b/exp3b_train
REFS=$E/test_world_refs_ramp6_3b/exp3b_train_b1
OUT=$E/exp3c/pilot
TABLE=$E/exp3b/hidden_strength.json
RAMP=0.03296703296703297
MU=0.0006325
LAM=0.006325
MEMBERS=3
PARALLEL=${PARALLEL:-3}
mkdir -p $OUT/logs

job() {
  arm=$1; pos=$2; idx=$3
  out=$OUT/$arm/ic${idx}.json
  [ -f $out ] && return 0
  mkdir -p $OUT/$arm
  strength=$($PY -c "import json; print(' '.join(str(x) for x in json.load(open('$TABLE'))['states']['$idx']['strength']))")
  common="--ic-dir $ICS --ic-position $pos --references $REFS/ic${idx}_references.npz
    --segments 13 --segment-days 14 --members $MEMBERS --score-start-day 98
    --warming-step-wm2 0.0 --warming-ramp-wm2-per-day $RAMP --mu $MU --lam $LAM
    --efficacy $strength --save-fields --output $out"
  planner="--preset short14 --optimizer gauss_newton --iterations 1 --copies 1
    --representation zonal --sensing ocean"
  case $arm in
    plan_oracle) cmd="run_test_world.py plan $common $planner" ;;
    plan_naive)  cmd="run_test_world.py plan $common $planner --planner-efficacy 1" ;;
    plan_learn)  cmd="run_test_world.py plan $common $planner --learn-strength bands --learn-noise-k $NOISE_K" ;;
  esac
  $PY $cmd > $OUT/logs/${arm}_ic${idx}.log 2>&1 \
    && echo "[$(ts)] $arm ic$idx finished" || echo "[$(ts)] $arm ic$idx FAILED"
}
export -f job ts
export PY ICS REFS OUT TABLE RAMP MU LAM MEMBERS

# Branch-1 states: odd manifest positions, macro 16 + 2i, index 100 m + 1.
states() { for i in $(seq 0 15); do
  printf "%d %04d\n" $((2 * i + 1)) $((100 * (16 + 2 * i) + 1)); done; }

echo "[$(ts)] A: ocean-only oracle planner on 16 pilot states"
states | sed "s/^/plan_oracle /" | xargs -P $PARALLEL -L 1 bash -c 'job $0 $1 $2'
JAX_PLATFORMS=cpu $PY analyze_exp3b_pilot.py noise --runs-dir $OUT --references-dir $REFS \
  || { echo "NOISE RULE FAILED"; exit 1; }
export NOISE_K=$($PY -c "import json; print(json.load(open('$OUT/noise.json'))['noise_k'])")
echo "[$(ts)] learner noise level $NOISE_K K"

echo "[$(ts)] B: the ocean-only learner and naive planner"
{ states | sed "s/^/plan_learn /"; states | sed "s/^/plan_naive /"; } \
  | xargs -P $PARALLEL -L 1 bash -c 'job $0 $1 $2'

echo "[$(ts)] C: summary against 3b's pilot"
JAX_PLATFORMS=cpu $PY analyze_exp3c_pilot.py --pilot-dir $OUT \
  --pilot3b-dir $E/exp3b/pilot --references-dir $REFS \
  > $OUT/logs/summary.log 2>&1 || { echo "SUMMARY FAILED"; tail $OUT/logs/summary.log; exit 1; }
cat $OUT/logs/summary.log
echo "[$(ts)] ========== EXP3C PILOT COMPLETE =========="
