#!/bin/bash
# The planner pilot, part 2 (MCB_PROJECT_REPORT.md Part 18 step 23): closed loop.
#
# Part 1 (run_campaign_pilot.sh) re-planned once, from day 112 of an
# UNCONTROLLED warmed run, which is the worst case for how much brightening a
# re-plan asks for. Here the planner runs a whole 182-day episode from day 0
# with the settings part 1 chose, so the brightening it needs while tracking
# the ramp from the start can be measured (rule R5's headroom), along with
# the error it leaves and the cost per episode.
#
# Training states only: ic0002 and ic0202 (two macro states). One member each
# (a sanity check, not a scored result), next to the uncontrolled warmed
# references built here for the pilot ramp.
#
# Settings come from the environment (defaults = part 1's expected choices):
#   COPIES (1), ITERATIONS (1), LOOKAHEAD (120), MU, LAM (rule R7's
#   translated penalties). Part 1's decisions: pilot_decisions.json.
# The planner here uses the MAP objective: planner.py has no zonal option
# yet (part 1 chose the zonal one), which matters little for headroom.
ts() { date +%H:%M:%S; }

cd ~/workspace/jax-gcm || exit 1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.12
export PYTHONUNBUFFERED=1
PY=~/mcb-env/bin/python
E=mcb_experiments_gpu
ICS=$E/ics_macro/exp3_train
REFS=$E/test_world_refs_ramp6/exp3_train
OUT=$E/pilot_step23/loop
COPIES=${COPIES:-1}
ITERATIONS=${ITERATIONS:-1}
LOOKAHEAD=${LOOKAHEAD:-120}
MU=${MU:-0.0006325}
LAM=${LAM:-0.006325}
RAMP=0.03296703296703297        # 6 W m-2 at day 182
# References must reach the last re-plan (day 168) plus its look-ahead.
DAYS=$(( LOOKAHEAD > 60 ? 300 : 240 ))
mkdir -p $OUT

echo "[$(ts)] references with the pilot ramp (3 states: positions 0-2)"
if ! grep -q '"finished_utc"' $REFS/references_manifest.json 2>/dev/null; then
  $PY run_test_world.py references --ic-dir $ICS --max-ics 3 --days $DAYS \
    --members 5 --member-block-days 5 --member-seed0 93000 \
    --warming-step-wm2 0.0 --warming-ramp-wm2-per-day $RAMP \
    --output-dir $REFS > $OUT/references.log 2>&1 \
    || { echo "REFERENCES FAILED"; exit 1; }
fi

run_one() {
  pos=$1; idx=$2
  $PY run_test_world.py plan --ic-dir $ICS --ic-position $pos \
    --references $REFS/ic${idx}_references.npz --segments 13 \
    --segment-days 14 --members 1 --score-start-day 98 \
    --warming-step-wm2 0.0 --warming-ramp-wm2-per-day $RAMP \
    --preset snipped60 --optimizer gauss_newton --iterations $ITERATIONS \
    --copies $COPIES --lookahead-days $LOOKAHEAD --mu $MU --lam $LAM \
    --save-fields \
    --output $OUT/ic${idx}_plan.json > $OUT/ic${idx}_plan.log 2>&1 \
    && echo "[$(ts)] ic$idx plan finished" || echo "[$(ts)] ic$idx plan FAILED"
}
export -f run_one ts
export PY ICS REFS OUT COPIES ITERATIONS LOOKAHEAD MU LAM RAMP DAYS
printf "0 0002\n2 0202\n" | xargs -P 2 -L 1 bash -c 'run_one $0 $1'
echo "[$(ts)] ========== PILOT LOOP COMPLETE =========="
