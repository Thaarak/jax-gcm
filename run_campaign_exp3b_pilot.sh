#!/bin/bash
# Experiment 3b pilot: training states only (MCB_PROJECT_REPORT.md Part 24).
#
# 16 states (exp3b_train branch 1, one per training macro state), each with
# its hidden strength (jcm.mcb.hidden_strength), 3 members, whole episodes
# (13 segments of 14 days, scored on days 98-182), the 3a test world.
#   A. The oracle planner (told the true strength): its forecast misses set
#      the learner's noise level (analyze_exp3b_pilot.py noise).
#   B. Every other arm and the registered variants: the naive planner, the
#      learner (per band; 4x noise; one shared factor), PI (42- and 84-day
#      loops), the adaptive law, the fixed design, no brightening.
#   C. The fixed rules: defaults, members per evaluation run, power
#      (analyze_exp3b_pilot.py decide).
# Re-running skips every run whose summary exists.
ts() { date +%H:%M:%S; }

cd ~/workspace/jax-gcm || exit 1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.12
export PYTHONUNBUFFERED=1
PY=~/mcb-env/bin/python
E=mcb_experiments_gpu
ICS=$E/ics_macro3b/exp3b_train
REFS=$E/test_world_refs_ramp6_3b/exp3b_train_b1
DESIGN_REFS=$E/test_world_refs_ramp6_3b/exp3b_train_b0
TRAIN=$E/exp3b/train
OUT=$E/exp3b/pilot
TABLE=$E/exp3b/hidden_strength.json
RAMP=0.03296703296703297
MU=0.0006325
LAM=0.006325
MEMBERS=3
PARALLEL=${PARALLEL:-4}
mkdir -p $OUT/logs
[ -f $TRAIN/ladder.json ] || { echo "no fixed design yet"; exit 1; }
[ -f $TABLE ] || { echo "no hidden-strength table"; exit 1; }
export LINEAR=$($PY -c "import json; print(' '.join(str(x) for x in json.load(open('$TRAIN/ladder.json'))['pooled']['linear_response']['amplitudes']))")
echo "[$(ts)] fixed design: [$LINEAR]"

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
    --representation zonal"
  sens="--sensitivity-from $TRAIN/responses/ic*_responses.npz
    --sensitivity-references-dir $DESIGN_REFS"
  case $arm in
    uncontrolled)  cmd="run_test_world.py episode $common --uniform 0.0" ;;
    fixed)         cmd="run_test_world.py episode $common --amplitudes $LINEAR" ;;
    pi)            cmd="run_controllers.py feedback --controller pi $common $sens --feedforward --closed-loop-days 42" ;;
    pi_slow)       cmd="run_controllers.py feedback --controller pi $common $sens --feedforward --closed-loop-days 84" ;;
    adaptive)      cmd="run_controllers.py feedback --controller adaptive $common $sens --pattern $LINEAR" ;;
    plan_oracle)   cmd="run_test_world.py plan $common $planner" ;;
    plan_naive)    cmd="run_test_world.py plan $common $planner --planner-efficacy 1" ;;
    plan_learn)    cmd="run_test_world.py plan $common $planner --learn-strength bands --learn-noise-k $NOISE_K" ;;
    plan_learn_cautious) cmd="run_test_world.py plan $common $planner --learn-strength bands --learn-noise-k $NOISE_K4" ;;
    plan_learn_global)   cmd="run_test_world.py plan $common $planner --learn-strength global --learn-noise-k $NOISE_K" ;;
  esac
  $PY $cmd > $OUT/logs/${arm}_ic${idx}.log 2>&1 \
    && echo "[$(ts)] $arm ic$idx finished" || echo "[$(ts)] $arm ic$idx FAILED"
}
export -f job ts
export PY ICS REFS DESIGN_REFS TRAIN OUT TABLE RAMP MU LAM MEMBERS

# Branch-1 states: odd manifest positions, macro 16 + 2i, index 100 m + 1.
states() { for i in $(seq 0 15); do
  printf "%d %04d\n" $((2 * i + 1)) $((100 * (16 + 2 * i) + 1)); done; }

echo "[$(ts)] A: oracle planner on 16 pilot states"
states | sed "s/^/plan_oracle /" | xargs -P $PARALLEL -L 1 bash -c 'job $0 $1 $2'
JAX_PLATFORMS=cpu $PY analyze_exp3b_pilot.py noise --runs-dir $OUT --references-dir $REFS \
  || { echo "NOISE RULE FAILED"; exit 1; }
export NOISE_K=$($PY -c "import json; print(json.load(open('$OUT/noise.json'))['noise_k'])")
export NOISE_K4=$($PY -c "print(4 * $NOISE_K)")
echo "[$(ts)] learner noise level $NOISE_K K (cautious: $NOISE_K4)"

echo "[$(ts)] B: every other arm"
{
  for arm in plan_learn plan_naive plan_learn_global plan_learn_cautious \
             pi pi_slow adaptive fixed uncontrolled; do
    states | sed "s/^/$arm /"
  done
} | xargs -P $PARALLEL -L 1 bash -c 'job $0 $1 $2'

echo "[$(ts)] C: the pilot's fixed rules"
JAX_PLATFORMS=cpu $PY analyze_exp3b_pilot.py decide --runs-dir $OUT --references-dir $REFS \
  > $OUT/logs/decide.log 2>&1 || { echo "DECIDE FAILED"; tail $OUT/logs/decide.log; exit 1; }
cat $OUT/logs/decide.log
echo "[$(ts)] ========== EXP3B PILOT COMPLETE =========="
