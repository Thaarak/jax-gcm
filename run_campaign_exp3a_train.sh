#!/bin/bash
# Experiment 3a, training side (MCB_PROJECT_REPORT.md Part 18 steps 18-19, 28).
#
# Everything the opponents need, from TRAINING states only, before revision 1
# is frozen, so their designs can be written into it:
#   1. ramp-6 references for the 16 exp3_train training states (branches 2, 3);
#   2. one-band response runs on those 16 states (rungs 1 and 4 of the ladder,
#      and the classical controllers' sensitivities);
#   3. the 120-day planner (the pilot's settings, Part 22) on the 8 branch-3
#      states, one member each (rungs 2 and 3: its effort and average pattern);
#   4. the ladder designs on the zonal profile over the scoring window.
# Re-running skips whatever already exists.
ts() { date +%H:%M:%S; }

cd ~/workspace/jax-gcm || exit 1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.12
export PYTHONUNBUFFERED=1
PY=~/mcb-env/bin/python
E=mcb_experiments_gpu
ICS=$E/ics_macro/exp3_train
REFS=$E/test_world_refs_ramp6/exp3_train
OUT=$E/exp3a/train
RAMP=0.03296703296703297        # 6 W m-2 at day 182 (pilot rule R5)
MU=0.0006325                    # pilot rule R7
LAM=0.006325
PARALLEL=${PARALLEL:-4}
mkdir -p $OUT/logs $OUT/responses $OUT/plan120

echo "[$(ts)] references: 16 training states x 5 members x 300 d"
$PY - <<EOF || $PY run_test_world.py references --ic-dir $ICS --max-ics 16 \
    --days 300 --members 5 --member-block-days 5 --member-seed0 93000 \
    --warming-step-wm2 0.0 --warming-ramp-wm2-per-day $RAMP \
    --output-dir $REFS > $OUT/logs/references.log 2>&1 \
  || { echo "REFERENCES FAILED"; exit 1; }
import json, sys
m = json.load(open("$REFS/references_manifest.json"))
sys.exit(0 if len(m.get("ics", [])) >= 16 and "finished_utc" in m else 1)
EOF
echo "[$(ts)] references ready"

job() {
  kind=$1; pos=$2; idx=$3
  case $kind in
    responses)
      out=$OUT/responses/ic${idx}_responses.npz
      [ -f $out ] && return 0
      $PY run_controllers.py responses --ic-dir $ICS --ic-position $pos \
        --references $REFS/ic${idx}_references.npz --days 182 --delta 0.1 \
        --warming-step-wm2 0.0 --warming-ramp-wm2-per-day $RAMP \
        --output-dir $OUT/responses > $OUT/logs/responses_ic${idx}.log 2>&1 ;;
    plan120)
      out=$OUT/plan120/ic${idx}.json
      [ -f $out ] && return 0
      $PY run_test_world.py plan --ic-dir $ICS --ic-position $pos \
        --references $REFS/ic${idx}_references.npz --segments 13 \
        --segment-days 14 --members 1 --score-start-day 98 \
        --warming-step-wm2 0.0 --warming-ramp-wm2-per-day $RAMP \
        --preset snipped60 --lookahead-days 120 --optimizer gauss_newton \
        --iterations 1 --copies 1 --representation zonal --mu $MU --lam $LAM \
        --save-fields --output $out > $OUT/logs/plan120_ic${idx}.log 2>&1 ;;
  esac
  [ $? -eq 0 ] && echo "[$(ts)] $kind ic$idx finished" \
    || echo "[$(ts)] $kind ic$idx FAILED"
}
export -f job ts
export PY ICS REFS OUT RAMP MU LAM

# Longest first: the planner runs, then the response runs.
{
  for p in 1 3 5 7 9 11 13 15; do
    printf "plan120 %d %04d\n" $p $(( (p / 2) * 200 + 3 ))
  done
  for p in $(seq 0 15); do
    printf "responses %d %04d\n" $p $(( (p / 2) * 200 + 2 + p % 2 ))
  done
} | xargs -P $PARALLEL -L 1 bash -c 'job $0 $1 $2'

echo "[$(ts)] ladder designs (zonal, days 98-182)"
$PY run_controllers.py ladder --responses $OUT/responses/ic*_responses.npz \
  --references-dir $REFS --window 98 182 \
  --planner-summaries $OUT/plan120/ic*.json --mu $MU \
  --representation zonal --output $OUT/ladder.json \
  > $OUT/logs/ladder.log 2>&1 || { echo "LADDER FAILED"; exit 1; }
cat $OUT/logs/ladder.log | tail -6
echo "[$(ts)] ========== EXP3A TRAINING COMPLETE =========="
