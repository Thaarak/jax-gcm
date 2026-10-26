#!/bin/bash
# Experiment 3f on 3d's 48 evaluation states (Amendment 9 revision 5). Run ONLY
# after revision 5 is frozen and posted.
#
# First the normal-only target extensions to day 280 for the 48 evaluation
# states (10 members, 3d's member seeds), then the registered arms
# (exp3f/registered_design.json, after the pilot's budget rule) on every
# state: 6 members, 13 segments of 14 days, scored on days 98-182 against
# 3d's 10-member references. 3d's runs of uncontrolled, pi, adaptive, fixed,
# plan_learn and plan_naive are reused by the registered analysis.
# A failed run is re-run once (each run's attempts are counted); the second
# pass runs only while the GPU is visible.
#
# SMOKE=1: the arms' exact commands on ONE training state (macro 100,
# branch 1, which has the pilot's target extension) for two segments with one
# member into /tmp/exp3f_smoke, then the analysis in its smoke mode (3d's
# smoke runs in /tmp/exp3d_smoke are the reused arms, on branch 0;
# run_campaign_exp3d_eval.sh SMOKE=1 makes them). Smoke scores are nan: two
# segments end before the scored days.
cd ~/workspace/jax-gcm || exit 1
source exp3f_common.sh
$PY -c "from jcm.mcb.land_climatology import land_climatology_mode as m; assert m() == 'monthly', m()" \
  || { echo "NOT THE FIXED LAND MODEL"; exit 1; }
gpu_visible || { echo "NO VISIBLE GPU"; exit 1; }
DESIGN=${DESIGN:-$E/exp3f/registered_design.json}
[ -f $DESIGN ] || { echo "no registered design ($DESIGN)"; exit 1; }
ARMS=$($PY -c "import json; print(' '.join(json.load(open('$DESIGN'))['arms']))")
ICS=$E/ics_macro3d/exp3d_eval
REFS_ROOT=$E/test_world_refs_ramp6_3d
REF_SET=exp3d_eval
EXT_ROOT=$E/exp3f/target_ext
OUT=$E/exp3f/eval
RUNS_3D=$E/exp3d/eval
SEGMENTS=13; MEMBERS=6; SCORE_START=98
PARALLEL=${PARALLEL:-5}
states() {
  if [ "${SMOKE:-0}" = 1 ]; then echo "1 10001 1"; return; fi
  for p in $(seq 0 47); do
    m=$((101 + 2 * (p / 3))); b=$((p % 3)); echo "$p $((100 * m + b)) $b"; done; }
if [ "${SMOKE:-0}" = 1 ]; then
  ICS=$E/ics_macro3d/exp3d_train; REF_SET=exp3d_train; OUT=/tmp/exp3f_smoke
  RUNS_3D=/tmp/exp3d_smoke; SEGMENTS=2; MEMBERS=1; SCORE_START=0; rm -rf $OUT
else
  : "${REVISION5_COMMIT:?set REVISION5_COMMIT to the commit that froze revision 5}"
  mkdir -p $OUT; echo "[$(ts)] revision 5 frozen at $REVISION5_COMMIT" | tee -a $OUT/REVISION5_COMMIT
fi
export ICS REFS_ROOT REF_SET EXT_ROOT OUT SEGMENTS MEMBERS SCORE_START
mkdir -p $OUT/logs
if [ "${SMOKE:-0}" != 1 ]; then
  echo "[$(ts)] target extensions: 48 evaluation states x 10 members x $EXT_DAYS d (normal only)"
  for b in 0 1 2; do
    build_extension $ICS $b 10 $EXT_ROOT/${REF_SET}_b$b --allow-eval-roles \
      > $OUT/logs/target_ext_b$b.log 2>&1 &
  done
  wait
  for b in 0 1 2; do
    grep -q finished_utc $EXT_ROOT/${REF_SET}_b$b/references_manifest.json 2>/dev/null \
      || { echo "TARGET EXTENSION b$b FAILED"; exit 1; }
  done
fi
echo "[$(ts)] runs, in this order: $ARMS; $MEMBERS members, $PARALLEL at a time"
run_all() {
  for arm in $ARMS; do states | sed "s/^/$arm /"; done \
    | xargs -P $PARALLEL -L 1 bash -c 'job $0 $1 $2 $3'
}
run_all
gpu_visible || { echo "[$(ts)] NO VISIBLE GPU after the first pass:" \
  "start this script again by hand once it is back (the single re-run)"; exit 1; }
run_all
REFS_DIRS=$([ "${SMOKE:-0}" = 1 ] \
  && echo "$REFS_ROOT/exp3d_train_b0 $REFS_ROOT/exp3d_train_b1" \
  || echo "$REFS_ROOT/exp3d_eval_b0 $REFS_ROOT/exp3d_eval_b1 $REFS_ROOT/exp3d_eval_b2")
if [ "${SMOKE:-0}" = 1 ]; then
  JAX_PLATFORMS=cpu $PY analyze_experiment3f.py --runs-dir $OUT --runs-3d-dir $RUNS_3D \
    --references-dirs $REFS_DIRS --design $DESIGN --arms $ARMS \
    --output $OUT/analysis.json --smoke || echo "SMOKE ANALYSIS FAILED"
  echo "[$(ts)] ========== EXP3F SMOKE COMPLETE =========="; exit 0
fi
for arm in $ARMS; do
  echo "[$(ts)] $arm: $(ls $OUT/$arm/ic*[0-9].json 2>/dev/null | wc -l | tr -d ' ') of 48 runs"
done
echo "[$(ts)] analysis"
JAX_PLATFORMS=cpu $PY analyze_experiment3f.py --runs-dir $OUT --runs-3d-dir $RUNS_3D \
  --references-dirs $REFS_DIRS --design $DESIGN --arms $ARMS \
  --output $E/exp3f/exp3f_analysis.json > $OUT/logs/analysis.log 2>&1 \
  || echo "ANALYSIS FAILED (see logs/analysis.log)"
tail -30 $OUT/logs/analysis.log
echo "[$(ts)] ========== EXP3F EVALUATION COMPLETE =========="
