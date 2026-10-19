#!/bin/bash
# Experiment 3e's pilot (Amendment 9 revision 4, run BEFORE its freeze, on training
# states only; MCB_PROJECT_REPORT.md Part 29).
#
# The oracle planner (told the true strength) with analysis sensing: before every
# forecast it corrects its own last forecast toward the true weather and keeps a
# share f of that forecast's error (--analysis-error; f = 0 is 3d's exact planner,
# f = 1 is 3c's ocean-only one). One run per level f = 0.1, 0.3 and 1.0 on each of
# 3d's 16 branch-1 training states (3 members, 13 segments of 14 days), in the
# fixed-land model, with 3d's planner settings, warming and hidden strengths.
#
# Rules, fixed before it ran (analyze_exp3e_pilot.py):
#   * each level's learner noise level is 3b's rule, unchanged
#     (analyze_exp3b_pilot.py noise, written to <level>/noise.json);
#   * each level's detection scale kappa, and the descriptive drift, scores and
#     noise curve, go to summary.json.
# A failed run is re-run once. No evaluation state is touched.
#
# SMOKE=1 runs every level on ONE training state (macro 100, branch 1) for two
# segments with one member into /tmp/exp3e_pilot_smoke, then both rules.
ts() { date +%H:%M:%S; }

cd ~/workspace/jax-gcm || exit 1
export JCM_LAND_CLIMATOLOGY=monthly
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.12
export PYTHONUNBUFFERED=1
PY=~/mcb-env/bin/python
E=mcb_experiments_gpu
ICS=$E/ics_macro3d/exp3d_train
REFS=$E/test_world_refs_ramp6_3d/exp3d_train_b1
TABLE=$E/exp3d/hidden_strength.json
OUT=$E/exp3e/pilot
RAMP=0.03296703296703297
MU=0.0006325
LAM=0.006325
SEGMENTS=13
MEMBERS=3
SCORE_START=98
LEVELS="a10:0.1 a30:0.3 a100:1.0"
PARALLEL=${PARALLEL:-5}
$PY -c "from jcm.mcb.land_climatology import land_climatology_mode as m; assert m() == 'monthly', m()" \
  || { echo "NOT THE FIXED LAND MODEL"; exit 1; }
if [ "${SMOKE:-0}" = 1 ]; then
  OUT=/tmp/exp3e_pilot_smoke; SEGMENTS=2; MEMBERS=1; SCORE_START=0; rm -rf $OUT
fi
mkdir -p $OUT/logs
echo "[$(ts)] pilot: oracle with analysis sensing at $LEVELS; $MEMBERS members," \
  "$SEGMENTS segments; land model monthly; $PARALLEL at a time"

job() {
  level=$1; f=$2; pos=$3; idx=$4
  out=$OUT/$level/plan_oracle/ic${idx}.json
  [ -f $out ] && return 0
  mkdir -p $OUT/$level/plan_oracle
  strength=$($PY -c "import json; print(' '.join(str(x) for x in json.load(open('$TABLE'))['states']['$idx']['strength']))")
  $PY run_test_world.py plan --ic-dir $ICS --ic-position $pos \
    --references $REFS/ic${idx}_references.npz \
    --segments $SEGMENTS --segment-days 14 --members $MEMBERS \
    --score-start-day $SCORE_START --warming-step-wm2 0.0 \
    --warming-ramp-wm2-per-day $RAMP --mu $MU --lam $LAM --efficacy $strength \
    --save-fields --output $out \
    --preset short14 --optimizer gauss_newton --iterations 1 --copies 1 \
    --representation zonal --sensing analysis --analysis-error $f \
    > $OUT/logs/${level}_ic${idx}.log 2>&1 \
    && echo "[$(ts)] $level ic$idx finished" || echo "[$(ts)] $level ic$idx FAILED"
}
export -f job ts
export PY ICS REFS TABLE OUT RAMP MU LAM SEGMENTS MEMBERS SCORE_START

# (level, f, position, index): branch-1 training states sit at odd positions,
# index 100 m + 1 for macro states m = 100, 102, ..., 130. Level-outer, so the
# first level finishes first.
pilot_jobs() {
  for lf in $LEVELS; do
    level=${lf%%:*}; f=${lf##*:}
    if [ "${SMOKE:-0}" = 1 ]; then echo "$level $f 1 10001"; continue; fi
    for i in $(seq 0 15); do
      echo "$level $f $((2 * i + 1)) $((100 * (100 + 2 * i) + 1))"
    done
  done
}
for pass in 1 2; do       # the second pass re-runs a failed run once
  pilot_jobs | xargs -P $PARALLEL -L 1 bash -c 'job $0 $1 $2 $3'
done
want=$([ "${SMOKE:-0}" = 1 ] && echo 1 || echo 16)
for lf in $LEVELS; do
  level=${lf%%:*}
  n=$(ls $OUT/$level/plan_oracle/ic*.json 2>/dev/null | wc -l | tr -d ' ')
  echo "[$(ts)] $level: $n of $want runs"
  JAX_PLATFORMS=cpu $PY analyze_exp3b_pilot.py noise --runs-dir $OUT/$level \
    --references-dir $REFS || echo "NOISE RULE FAILED ($level)"
done
JAX_PLATFORMS=cpu $PY analyze_exp3e_pilot.py --pilot-dir $OUT \
  || echo "PILOT SUMMARY FAILED"
echo "[$(ts)] ========== EXP3E PILOT COMPLETE =========="
