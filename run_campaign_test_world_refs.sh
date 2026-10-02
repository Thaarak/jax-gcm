#!/bin/bash
# Test-world references for the TRAINING states (MCB_PROJECT_REPORT.md
# Part 18 steps 10-11), for the pilot of step 23.
#
# For every IC of exp2_train (8) and exp3_train (24), five member runs with
# no brightening for 240 days: once without warming (the normal climate,
# the target) and once with a steady 4 W m-2 probe warming (the uncontrolled
# warmed run). The ocean responds close to linearly, so the pilot can scale
# the probe to the warming it finally chooses.
#
# Evaluation roles are NOT built here: run_test_world.py refuses them until
# revision 1 of Amendment 9 is frozen.
#
# Forward runs only (a few GB), so by default this runs ALONGSIDE the vLLM
# container. If memory is short, run it with STOP_VLLM=1 to pause the
# container (a trap restarts it).
#
# Steps (re-running skips any role whose manifest is finished):
#   0/2 smoke: one IC, 2 members x 4 days, into /tmp
#   1/2 exp2_train   (8 ICs)
#   2/2 exp3_train  (24 ICs)
# Cost: 32 ICs x 2 x 5 x 240 d, about 1.4 GPU-h at 0.065 s per model day.
CONTAINER=aeon-vllm
ts() { date +%H:%M:%S; }
step() { echo ""; echo "[$(ts)] ========== $1 =========="; }

if [ "${STOP_VLLM:-0}" = "1" ]; then
  echo "[$(ts)] stopping $CONTAINER"
  docker stop $CONTAINER >/dev/null 2>&1
  trap 'echo "[$(ts)] restarting '"$CONTAINER"'"; docker start '"$CONTAINER"' >/dev/null 2>&1' EXIT INT TERM
fi

cd ~/workspace/jax-gcm || exit 1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
export PYTHONUNBUFFERED=1
PY=~/mcb-env/bin/python
E=mcb_experiments_gpu
ICS=$E/ics_macro
OUT=$E/test_world_refs
SMOKE=/tmp/test_world_refs_smoke
REFS="--days 240 --members 5 --member-block-days 5 --member-seed0 93000 \
  --warming-step-wm2 4.0 --warming-ramp-wm2-per-day 0.0"

step "0/2 Smoke: one IC, 2 members x 4 days"
rm -rf $SMOKE
$PY run_test_world.py references --ic-dir $ICS/exp2_train --max-ics 1 \
  --days 4 --members 2 --member-block-days 2 --output-dir $SMOKE \
  || { echo "SMOKE FAILED"; exit 1; }

for role in exp2_train exp3_train; do
  n=$([ $role = exp2_train ] && echo 1 || echo 2)
  step "$n/2 References: $role"
  if grep -q '"finished_utc"' $OUT/$role/references_manifest.json 2>/dev/null; then
    echo "  $OUT/$role complete - reusing"
    continue
  fi
  $PY run_test_world.py references --ic-dir $ICS/$role $REFS \
    --output-dir $OUT/$role || { echo "REFERENCES FAILED ($role)"; exit 1; }
done

echo ""
echo "[$(ts)] ========== TEST-WORLD REFERENCES COMPLETE =========="
echo "Artifacts: $OUT/exp2_train/, $OUT/exp3_train/"
