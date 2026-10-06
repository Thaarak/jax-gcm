#!/bin/bash
# Experiment 3a on the 24 evaluation states (Amendment 9 revision 1;
# MCB_PROJECT_REPORT.md Part 18 step 28). Run ONLY after revision 1 is frozen
# and posted: it builds the evaluation references (--allow-eval-roles).
#
# Ten arms, every one with the same 3 weather members (the references' first
# 3 seeds), 13 segments of 14 days, scored on days 98-182:
#   uncontrolled; plan120 / plan60 / plan14 (Gauss-Newton, 1 step, 1 copy,
#   zonal objective); the four fixed rungs of the ladder designed on the
#   training states (exp3a/train/ladder.json); the GLENS-style PI controller
#   (feedforward, 42-day closed loop) and the adaptive law (rung-4 pattern),
#   both with sensitivities from the training response runs.
# Re-running skips every run whose summary exists. Longest jobs go first.
#
# SMOKE=1 runs every arm's exact command on ONE TRAINING state (ic0003), for
# two segments and one member, into /tmp/exp3a_smoke: no evaluation data.
ts() { date +%H:%M:%S; }

cd ~/workspace/jax-gcm || exit 1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.12
export PYTHONUNBUFFERED=1
PY=~/mcb-env/bin/python
E=mcb_experiments_gpu
ICS=$E/ics_macro/exp3_eval
REFS=$E/test_world_refs_ramp6/exp3_eval
TRAIN=$E/exp3a/train
TRAIN_REFS=$E/test_world_refs_ramp6/exp3_train
OUT=$E/exp3a/eval
RAMP=0.03296703296703297
MU=0.0006325
LAM=0.006325
MEMBERS=3
SEGMENTS=13
SCORE_START=98
PARALLEL=${PARALLEL:-4}
if [ "${SMOKE:-0}" = 1 ]; then
  ICS=$E/ics_macro/exp3_train; REFS=$TRAIN_REFS; OUT=/tmp/exp3a_smoke
  MEMBERS=1; SEGMENTS=2; SCORE_START=0; rm -rf $OUT
else
  : "${REVISION1_COMMIT:?set REVISION1_COMMIT to the commit that froze revision 1}"
fi
mkdir -p $OUT/logs
[ -f $TRAIN/ladder.json ] || { echo "no ladder designs"; exit 1; }
if [ "${SMOKE:-0}" != 1 ]; then
echo "[$(ts)] revision 1 frozen at $REVISION1_COMMIT" | tee $OUT/REVISION1_COMMIT

echo "[$(ts)] evaluation references: 24 states x 5 members x 300 d"
$PY - <<EOF || $PY run_test_world.py references --ic-dir $ICS --max-ics 24 \
    --days 300 --members 5 --member-block-days 5 --member-seed0 93000 \
    --warming-step-wm2 0.0 --warming-ramp-wm2-per-day $RAMP \
    --allow-eval-roles --output-dir $REFS > $OUT/logs/references.log 2>&1 \
  || { echo "REFERENCES FAILED"; exit 1; }
import json, sys
m = json.load(open("$REFS/references_manifest.json"))
sys.exit(0 if len(m.get("ics", [])) >= 24 and "finished_utc" in m else 1)
EOF
echo "[$(ts)] references ready"
fi

# The frozen fixed designs.
rung() { $PY -c "import json; print(' '.join(str(x) for x in json.load(open('$TRAIN/ladder.json'))['pooled']['$1']['amplitudes']))"; }
export UNIFORM_CANCEL=$(rung uniform_cancel) UNIFORM_EFFORT=$(rung uniform_effort)
export PLANNER_AVERAGE=$(rung planner_average) LINEAR_RESPONSE=$(rung linear_response)
echo "[$(ts)] rungs: cancel [$UNIFORM_CANCEL] effort [$UNIFORM_EFFORT]" \
  "average [$PLANNER_AVERAGE] linear [$LINEAR_RESPONSE]"

job() {
  arm=$1; pos=$2; idx=$3
  out=$OUT/$arm/ic${idx}.json
  [ -f $out ] && return 0
  mkdir -p $OUT/$arm
  common="--ic-dir $ICS --ic-position $pos --references $REFS/ic${idx}_references.npz
    --segments $SEGMENTS --segment-days 14 --members $MEMBERS --score-start-day $SCORE_START
    --warming-step-wm2 0.0 --warming-ramp-wm2-per-day $RAMP --mu $MU --lam $LAM
    --save-fields --output $out"
  planner="--optimizer gauss_newton --iterations 1 --copies 1 --representation zonal"
  sens="--sensitivity-from $TRAIN/responses/ic*_responses.npz
    --sensitivity-references-dir $TRAIN_REFS"
  case $arm in
    uncontrolled)    cmd="run_test_world.py episode $common --uniform 0.0" ;;
    uniform_cancel)  cmd="run_test_world.py episode $common --amplitudes $UNIFORM_CANCEL" ;;
    uniform_effort)  cmd="run_test_world.py episode $common --amplitudes $UNIFORM_EFFORT" ;;
    planner_average) cmd="run_test_world.py episode $common --amplitudes $PLANNER_AVERAGE" ;;
    linear_response) cmd="run_test_world.py episode $common --amplitudes $LINEAR_RESPONSE" ;;
    plan120) cmd="run_test_world.py plan $common $planner --preset snipped60 --lookahead-days 120" ;;
    plan60)  cmd="run_test_world.py plan $common $planner --preset snipped60 --lookahead-days 60" ;;
    plan14)  cmd="run_test_world.py plan $common $planner --preset short14" ;;
    pi)       cmd="run_controllers.py feedback --controller pi $common $sens --feedforward --closed-loop-days 42" ;;
    adaptive) cmd="run_controllers.py feedback --controller adaptive $common $sens --pattern $LINEAR_RESPONSE" ;;
  esac
  $PY $cmd > $OUT/logs/${arm}_ic${idx}.log 2>&1 \
    && echo "[$(ts)] $arm ic$idx finished" || echo "[$(ts)] $arm ic$idx FAILED"
}
export -f job ts
export PY ICS REFS TRAIN TRAIN_REFS OUT RAMP MU LAM MEMBERS SEGMENTS SCORE_START

# (position, index): macro states 1, 3, ..., 15 x branches 2, 3, 4.
states() {
  if [ "${SMOKE:-0}" = 1 ]; then echo "1 0003"; return; fi
  for p in $(seq 0 23); do
    printf "%d %04d\n" $p $(( (2 * (p / 3) + 1) * 100 + 2 + p % 3 )); done; }
{
  for arm in plan120 plan60 plan14 pi adaptive uncontrolled uniform_cancel \
             uniform_effort planner_average linear_response; do
    states | sed "s/^/$arm /"
  done
} | xargs -P $PARALLEL -L 1 bash -c 'job $0 $1 $2'

if [ "${SMOKE:-0}" = 1 ]; then
  echo "[$(ts)] smoke done:"; ls $OUT/*/ic0003.json; grep -l Traceback $OUT/logs/*.log
  exit 0
fi
echo "[$(ts)] analysis"
$PY analyze_experiment3a.py --runs-dir $OUT --references-dir $REFS \
  --output $E/exp3a/exp3a_analysis.json > $OUT/logs/analysis.log 2>&1 \
  || echo "ANALYSIS FAILED (see logs/analysis.log)"
tail -20 $OUT/logs/analysis.log
echo "[$(ts)] ========== EXP3A EVALUATION COMPLETE =========="
