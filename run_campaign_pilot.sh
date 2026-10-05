#!/bin/bash
# The planner pilot (MCB_PROJECT_REPORT.md Part 18 step 23), training states only.
#
# One process per training state (exp3_train, branch 2 of each of the eight
# macro states), three at a time on the GPU. Each spins up the uncontrolled
# ramp-warmed run to day 112, re-plans once in every candidate setting, and
# holds each result for 120 days in 8 shared weather samples
# (run_planner_pilot.py ic). The decision rules are in run_planner_pilot.py;
# `analyze` applies them afterwards.
#
# Forward and forward-mode runs need a few GB each, so this runs ALONGSIDE
# the vLLM container. Re-running skips any state whose output exists.
# Cost: about 35 GPU-minutes per state; about 1.5-2 hours wall with 3 at once.
ts() { date +%H:%M:%S; }

cd ~/workspace/jax-gcm || exit 1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.12
export PYTHONUNBUFFERED=1
PY=~/mcb-env/bin/python
E=mcb_experiments_gpu
ICS=$E/ics_macro/exp3_train
REFS=$E/test_world_refs/exp3_train
OUT=$E/pilot_step23
PARALLEL=${PARALLEL:-3}
mkdir -p $OUT/logs

echo "[$(ts)] smoke: one state, tiny look-aheads"
$PY run_planner_pilot.py ic --ic-dir $ICS --ic-position 0 \
  --references $REFS/ic0002_references.npz --output-dir /tmp/pilot_smoke \
  --start-day 2 --lookaheads 2 3 --truth-days 3 --members 2 --copies 2 \
  --single-copies 2 > $OUT/logs/smoke.log 2>&1 || { echo "SMOKE FAILED"; exit 1; }

run_one() {
  pos=$1; idx=$2
  if [ -f $OUT/ic${idx}_pilot.json ]; then echo "[$(ts)] ic$idx done - skipping"; return 0; fi
  echo "[$(ts)] ic$idx start"
  $PY run_planner_pilot.py ic --ic-dir $ICS --ic-position $pos \
    --references $REFS/ic${idx}_references.npz --output-dir $OUT \
    > $OUT/logs/ic${idx}.log 2>&1 \
    && echo "[$(ts)] ic$idx finished" || echo "[$(ts)] ic$idx FAILED"
}
export -f run_one ts
export PY ICS REFS OUT

# (position, index): branch 2 of macro states 0, 2, ..., 14.
printf "0 0002\n2 0202\n4 0402\n6 0602\n8 0802\n10 1002\n12 1202\n14 1402\n" \
  | xargs -P $PARALLEL -L 1 bash -c 'run_one $0 $1'

echo "[$(ts)] ========== PILOT STATES COMPLETE =========="
ls $OUT/*_pilot.json
