#!/bin/bash
# Experiment 3e on 3d's 48 evaluation states (Amendment 9 revision 4;
# MCB_PROJECT_REPORT.md Part 29). Run ONLY after revision 4 is frozen and
# posted.
#
# Before each two-week forecast the planners correct their own last forecast
# toward the true weather and keep a share f of its error (--sensing analysis
# --analysis-error f). Four arms run here, in this order: plan_learn_a10 and
# plan_naive_a10 (f = 0.1, the primary level), plan_learn_a100 (f = 1) and
# plan_learn_a30 (f = 0.3). Everything else is 3d's: the fixed land model, its
# 48 states (16 macro states x 3 branches), hidden strengths, 10-member
# evaluation references (built by 3d), 6 members, 13 segments of 14 days,
# scored on days 98-182. 3d's runs of pi, adaptive, fixed, uncontrolled,
# plan_learn and plan_naive are reused by the registered analysis. Registered
# settings come from exp3e/registered_design.json.
#
# A failed run is re-run once: each run's attempts are counted, and the second
# pass runs only while the GPU is visible. Otherwise the script stops, and the
# re-run is made by starting it again by hand (it skips finished runs).
#
# SMOKE=1 runs the four arms' exact commands on ONE TRAINING state (macro 100,
# branch 0) for two segments with one member into /tmp/exp3e_smoke, next to
# 3d's smoke runs (/tmp/exp3d_smoke, made by run_campaign_exp3d_eval.sh's
# smoke mode if missing), then the registered analysis in its smoke mode: no
# evaluation data.
ts() { date +%H:%M:%S; }

cd ~/workspace/jax-gcm || exit 1
export JCM_LAND_CLIMATOLOGY=monthly
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.12
export PYTHONUNBUFFERED=1
PY=~/mcb-env/bin/python
E=mcb_experiments_gpu
ICS=$E/ics_macro3d/exp3d_eval
REFS_ROOT=$E/test_world_refs_ramp6_3d
DESIGN=${DESIGN:-$E/exp3e/registered_design.json}
TABLE=$E/exp3d/hidden_strength.json
OUT=$E/exp3e/eval
RUNS_3D=$E/exp3d/eval
EXP3C=$E/exp3c/exp3c_analysis.json
RAMP=0.03296703296703297
MU=0.0006325
LAM=0.006325
SEGMENTS=13
SCORE_START=98
ARMS="plan_learn_a10 plan_naive_a10 plan_learn_a100 plan_learn_a30"
PARALLEL=${PARALLEL:-5}
$PY -c "from jcm.mcb.land_climatology import land_climatology_mode as m; assert m() == 'monthly', m()" \
  || { echo "NOT THE FIXED LAND MODEL"; exit 1; }
gpu_visible() { $PY -c "import jax; assert jax.devices()[0].platform == 'gpu'" > /dev/null 2>&1; }
gpu_visible || { echo "NO VISIBLE GPU"; exit 1; }
[ -f $DESIGN ] || { echo "no registered design ($DESIGN)"; exit 1; }
get() { $PY -c "import json; v = json.load(open('$DESIGN'))$1; print(' '.join(map(str, v)) if isinstance(v, list) else v)"; }
MEMBERS=$(get "['members']")
export LEARN_MODE=$(get "['learner_mode']")
for level in a10 a30 a100; do
  export "F_$level=$(get "['levels']['$level']['analysis_error']")"
  export "NOISE_$level=$(get "['levels']['$level']['learner_noise_k']")"
done
REF_SUFFIX=eval
if [ "${SMOKE:-0}" = 1 ]; then
  ICS=$E/ics_macro3d/exp3d_train; REF_SUFFIX=train; OUT=/tmp/exp3e_smoke
  RUNS_3D=/tmp/exp3d_smoke; MEMBERS=1; SEGMENTS=2; SCORE_START=0; rm -rf $OUT
  if ! ls $RUNS_3D/{uncontrolled,pi,adaptive,fixed,plan_learn,plan_naive}/ic10000.json > /dev/null 2>&1; then
    echo "[$(ts)] 3d's smoke runs first (the reused arms)"
    env -u DESIGN SMOKE=1 bash run_campaign_exp3d_eval.sh > /tmp/exp3d_smoke.log 2>&1
    ls $RUNS_3D/{uncontrolled,pi,adaptive,fixed,plan_learn,plan_naive}/ic10000.json > /dev/null \
      || { echo "3D SMOKE FAILED (see /tmp/exp3d_smoke.log)"; exit 1; }
  fi
else
  : "${REVISION4_COMMIT:?set REVISION4_COMMIT to the commit that froze revision 4}"
fi
mkdir -p $OUT/logs
echo "[$(ts)] design: $MEMBERS members, analysis error $F_a10 / $F_a30 / $F_a100," \
  "learner $LEARN_MODE noise $NOISE_a10 / $NOISE_a30 / $NOISE_a100 K," \
  "land model monthly, $PARALLEL at a time"
[ "${SMOKE:-0}" = 1 ] || echo "[$(ts)] revision 4 frozen at $REVISION4_COMMIT" | tee -a $OUT/REVISION4_COMMIT

job() {
  arm=$1; pos=$2; idx=$3; branch=$4
  out=$OUT/$arm/ic${idx}.json
  [ -f $out ] && return 0
  mkdir -p $OUT/$arm
  tries=$OUT/logs/${arm}_ic${idx}.tries
  n=$(cat $tries 2>/dev/null || echo 0)
  if [ "$n" -ge 2 ]; then echo "[$(ts)] $arm ic$idx failed twice: not run again"; return 0; fi
  echo $((n + 1)) > $tries
  log=$OUT/logs/${arm}_ic${idx}$([ "$n" -ge 1 ] && echo .retry).log
  level=${arm##*_}
  f_var=F_$level; noise_var=NOISE_$level
  refs=$REFS_ROOT/exp3d_${REF_SUFFIX}_b$branch
  strength=$($PY -c "import json; print(' '.join(str(x) for x in json.load(open('$TABLE'))['states']['$idx']['strength']))")
  common="--ic-dir $ICS --ic-position $pos --references $refs/ic${idx}_references.npz
    --segments $SEGMENTS --segment-days 14 --members $MEMBERS
    --score-start-day $SCORE_START --warming-step-wm2 0.0
    --warming-ramp-wm2-per-day $RAMP --mu $MU --lam $LAM --efficacy $strength
    --save-fields --output $out"
  planner="--preset short14 --optimizer gauss_newton --iterations 1 --copies 1
    --representation zonal --sensing analysis --analysis-error ${!f_var}"
  case $arm in
    plan_naive_*) cmd="run_test_world.py plan $common $planner --planner-efficacy 1" ;;
    plan_learn_*) cmd="run_test_world.py plan $common $planner --learn-strength $LEARN_MODE --learn-noise-k ${!noise_var}" ;;
  esac
  $PY $cmd > $log 2>&1 \
    && echo "[$(ts)] $arm ic$idx finished" || echo "[$(ts)] $arm ic$idx FAILED"
}
export -f job ts
export PY ICS REFS_ROOT REF_SUFFIX TABLE OUT RAMP MU LAM MEMBERS SEGMENTS SCORE_START

# (position, index, branch): macro states 101, 103, ..., 131 x branches 0-2.
states() {
  if [ "${SMOKE:-0}" = 1 ]; then echo "0 10000 0"; return; fi
  for p in $(seq 0 47); do
    m=$((101 + 2 * (p / 3))); b=$((p % 3))
    echo "$p $((100 * m + b)) $b"; done; }
run_all() {
  for arm in $ARMS; do states | sed "s/^/$arm /"; done \
    | xargs -P $PARALLEL -L 1 bash -c 'job $0 $1 $2 $3'
}
echo "[$(ts)] runs, in this order: $ARMS"
run_all
gpu_visible || { echo "[$(ts)] NO VISIBLE GPU after the first pass:" \
  "start this script again by hand once it is back (the single re-run)"; exit 1; }
run_all        # the second pass re-runs a failed run once

if [ "${SMOKE:-0}" = 1 ]; then
  echo "[$(ts)] smoke runs:"; ls $OUT/*/ic10000.json $OUT/*/ic10000.innovations.npz
  grep -l Traceback $OUT/logs/*.log
  JAX_PLATFORMS=cpu $PY analyze_experiment3e.py --runs-dir $OUT \
    --runs-3d-dir $RUNS_3D --references-dirs $REFS_ROOT/exp3d_train_b0 \
    --design $DESIGN --exp3c-analysis $EXP3C --output $OUT/analysis.json \
    --smoke || echo "SMOKE ANALYSIS FAILED"
  echo "[$(ts)] ========== EXP3E SMOKE COMPLETE =========="
  exit 0
fi
for arm in $ARMS; do
  echo "[$(ts)] $arm: $(ls $OUT/$arm/ic*[0-9].json 2>/dev/null | wc -l | tr -d ' ') of 48 runs"
done
echo "[$(ts)] analysis"
JAX_PLATFORMS=cpu $PY analyze_experiment3e.py --runs-dir $OUT --runs-3d-dir $RUNS_3D \
  --references-dirs $REFS_ROOT/exp3d_eval_b0 $REFS_ROOT/exp3d_eval_b1 \
  $REFS_ROOT/exp3d_eval_b2 --design $DESIGN --exp3c-analysis $EXP3C \
  --output $E/exp3e/exp3e_analysis.json \
  > $OUT/logs/analysis.log 2>&1 || echo "ANALYSIS FAILED (see logs/analysis.log)"
tail -30 $OUT/logs/analysis.log
echo "[$(ts)] ========== EXP3E EVALUATION COMPLETE =========="
