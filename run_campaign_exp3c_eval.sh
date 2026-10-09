#!/bin/bash
# Experiment 3c on 3b's 48 evaluation states (Amendment 9 revision 3, Part A;
# MCB_PROJECT_REPORT.md Part 28). Run ONLY after revision 3 is frozen and
# posted.
#
# The planners see the ocean but not the weather (--sensing ocean): on day 0
# their weather is another branch's of the same macro state, after that their
# own forecast's. Everything else is 3b's: the states, hidden strengths,
# 10-member references, members (6) and segments. Two arms run here,
# plan_learn and plan_naive; 3b's runs of the arms that never used the weather
# (pi, adaptive, fixed, uncontrolled) and of its exact-sensing planners are
# reused from exp3b/eval by the registered analysis. Registered settings come
# from exp3c/registered_design.json.
#
# SMOKE=1 runs both arms' exact commands on ONE TRAINING state (macro 16,
# branch 0) for two segments with one member into /tmp/exp3c_smoke, then the
# registered analysis in its smoke mode against 3b's smoke runs.
ts() { date +%H:%M:%S; }

cd ~/workspace/jax-gcm || exit 1
unset JCM_LAND_CLIMATOLOGY                  # 3b's (old) land model
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.12
export PYTHONUNBUFFERED=1
PY=~/mcb-env/bin/python
E=mcb_experiments_gpu
ICS=$E/ics_macro3b/exp3b_eval
REFS_ROOT=$E/test_world_refs_ramp6_3b
DESIGN=${DESIGN:-$E/exp3c/registered_design.json}
TABLE=$E/exp3b/hidden_strength.json
OUT=$E/exp3c/eval
RUNS_3B=$E/exp3b/eval
RAMP=0.03296703296703297
MU=0.0006325
LAM=0.006325
SEGMENTS=13
SCORE_START=98
PARALLEL=${PARALLEL:-3}
[ -f $DESIGN ] || { echo "no registered design ($DESIGN)"; exit 1; }
get() { $PY -c "import json; v = json.load(open('$DESIGN'))['$1']; print(' '.join(map(str, v)) if isinstance(v, list) else v)"; }
MEMBERS=$(get members)
export NOISE_K=$(get learner_noise_k) LEARN_MODE=$(get learner_mode)
REF_SUFFIX=eval
if [ "${SMOKE:-0}" = 1 ]; then
  ICS=$E/ics_macro3b/exp3b_train; REF_SUFFIX=train; OUT=/tmp/exp3c_smoke
  RUNS_3B=/tmp/exp3b_smoke; MEMBERS=1; SEGMENTS=2; SCORE_START=0; rm -rf $OUT
else
  : "${REVISION3_COMMIT:?set REVISION3_COMMIT to the commit that froze revision 3}"
fi
mkdir -p $OUT/logs
echo "[$(ts)] design: $MEMBERS members, ocean-only sensing, learner $LEARN_MODE noise $NOISE_K K"
[ "${SMOKE:-0}" = 1 ] || echo "[$(ts)] revision 3 frozen at $REVISION3_COMMIT" | tee $OUT/REVISION3_COMMIT

job() {
  arm=$1; pos=$2; idx=$3; branch=$4
  out=$OUT/$arm/ic${idx}.json
  [ -f $out ] && return 0
  mkdir -p $OUT/$arm
  refs=$REFS_ROOT/exp3b_${REF_SUFFIX}_b$branch
  strength=$($PY -c "import json; print(' '.join(str(x) for x in json.load(open('$TABLE'))['states']['$idx']['strength']))")
  common="--ic-dir $ICS --ic-position $pos --references $refs/ic${idx}_references.npz
    --segments $SEGMENTS --segment-days 14 --members $MEMBERS
    --score-start-day $SCORE_START --warming-step-wm2 0.0
    --warming-ramp-wm2-per-day $RAMP --mu $MU --lam $LAM --efficacy $strength
    --save-fields --output $out"
  planner="--preset short14 --optimizer gauss_newton --iterations 1 --copies 1
    --representation zonal --sensing ocean"
  case $arm in
    plan_naive) cmd="run_test_world.py plan $common $planner --planner-efficacy 1" ;;
    plan_learn) cmd="run_test_world.py plan $common $planner --learn-strength $LEARN_MODE --learn-noise-k $NOISE_K" ;;
  esac
  $PY $cmd > $OUT/logs/${arm}_ic${idx}.log 2>&1 \
    && echo "[$(ts)] $arm ic$idx finished" || echo "[$(ts)] $arm ic$idx FAILED"
}
export -f job ts
export PY ICS REFS_ROOT REF_SUFFIX TABLE OUT RAMP MU LAM MEMBERS SEGMENTS SCORE_START

# (position, index, branch): macro states 17, 19, ..., 47 x branches 0-2.
states() {
  if [ "${SMOKE:-0}" = 1 ]; then echo "0 1600 0"; return; fi
  for p in $(seq 0 47); do
    m=$((17 + 2 * (p / 3))); b=$((p % 3))
    printf "%d %04d %d\n" $p $((100 * m + b)) $b; done; }
{
  for arm in plan_learn plan_naive; do states | sed "s/^/$arm /"; done
} | xargs -P $PARALLEL -L 1 bash -c 'job $0 $1 $2 $3'

if [ "${SMOKE:-0}" = 1 ]; then
  echo "[$(ts)] smoke runs:"; ls $OUT/*/ic1600.json
  grep -l Traceback $OUT/logs/*.log
  JAX_PLATFORMS=cpu $PY analyze_experiment3c.py --runs-dir $OUT \
    --runs-3b-dir $RUNS_3B --references-dirs $REFS_ROOT/exp3b_train_b0 \
    --output $OUT/analysis.json --smoke || echo "SMOKE ANALYSIS FAILED"
  exit 0
fi
echo "[$(ts)] analysis"
JAX_PLATFORMS=cpu $PY analyze_experiment3c.py --runs-dir $OUT --runs-3b-dir $RUNS_3B \
  --references-dirs $REFS_ROOT/exp3b_eval_b0 $REFS_ROOT/exp3b_eval_b1 \
  $REFS_ROOT/exp3b_eval_b2 --output $E/exp3c/exp3c_analysis.json \
  > $OUT/logs/analysis.log 2>&1 || echo "ANALYSIS FAILED (see logs/analysis.log)"
tail -20 $OUT/logs/analysis.log
echo "[$(ts)] ========== EXP3C EVALUATION COMPLETE =========="
