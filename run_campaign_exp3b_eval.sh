#!/bin/bash
# Experiment 3b on the 48 evaluation states (Amendment 9 revision 2;
# MCB_PROJECT_REPORT.md Part 24). Run ONLY after revision 2 is frozen and
# posted: it builds the evaluation references (--allow-eval-roles).
#
# 16 fresh macro states (17, 19, ..., 47) x 3 weather branches, one at each
# overall strength factor (0.5, 1, 2), each with its hidden per-band strength
# (exp3b/hidden_strength.json). Seven arms, every one with the same members
# (the references' first seeds), 13 segments of 14 days, scored on days
# 98-182. Every registered setting comes from exp3b/registered_design.json,
# frozen with revision 2 (members, the learner's noise level and mode, the PI
# loop, the fixed design).
#
# SMOKE=1 runs every arm's exact command on ONE TRAINING state (macro 16,
# branch 0) for two segments with one member into /tmp/exp3b_smoke, then the
# registered analysis in its smoke mode: no evaluation data.
ts() { date +%H:%M:%S; }

cd ~/workspace/jax-gcm || exit 1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.12
export PYTHONUNBUFFERED=1
PY=~/mcb-env/bin/python
E=mcb_experiments_gpu
ICS=$E/ics_macro3b/exp3b_eval
REFS_ROOT=$E/test_world_refs_ramp6_3b
TRAIN=$E/exp3b/train
DESIGN=$E/exp3b/registered_design.json
TABLE=$E/exp3b/hidden_strength.json
OUT=$E/exp3b/eval
RAMP=0.03296703296703297
MU=0.0006325
LAM=0.006325
SEGMENTS=13
SCORE_START=98
PARALLEL=${PARALLEL:-4}
[ -f $DESIGN ] || { echo "no registered design ($DESIGN)"; exit 1; }
get() { $PY -c "import json; v = json.load(open('$DESIGN'))['$1']; print(' '.join(map(str, v)) if isinstance(v, list) else v)"; }
MEMBERS=$(get members)
REF_MEMBERS=$(get reference_members)
export NOISE_K=$(get learner_noise_k) LEARN_MODE=$(get learner_mode)
export PI_DAYS=$(get pi_closed_loop_days) FIXED=$(get fixed_amplitudes)
REF_SUFFIX=eval
if [ "${SMOKE:-0}" = 1 ]; then
  ICS=$E/ics_macro3b/exp3b_train; REF_SUFFIX=train; OUT=/tmp/exp3b_smoke
  MEMBERS=1; SEGMENTS=2; SCORE_START=0; rm -rf $OUT
else
  : "${REVISION2_COMMIT:?set REVISION2_COMMIT to the commit that froze revision 2}"
fi
mkdir -p $OUT/logs
echo "[$(ts)] design: $MEMBERS members, learner $LEARN_MODE noise $NOISE_K K," \
  "PI $PI_DAYS d, fixed [$FIXED]"

if [ "${SMOKE:-0}" != 1 ]; then
echo "[$(ts)] revision 2 frozen at $REVISION2_COMMIT" | tee $OUT/REVISION2_COMMIT
refs_done() {
  $PY - "$1" "$2" <<'EOF'
import json, sys
try:
    m = json.load(open(f"{sys.argv[1]}/references_manifest.json"))
except OSError:
    sys.exit(1)
sys.exit(0 if len(m.get("ics", [])) >= int(sys.argv[2]) and "finished_utc" in m else 1)
EOF
}
echo "[$(ts)] evaluation references: 48 states x $REF_MEMBERS members x 200 d"
for b in 0 1 2; do
  if ! refs_done $REFS_ROOT/exp3b_eval_b$b 16; then
    $PY run_test_world.py references --ic-dir $ICS --branches $b \
      --days 200 --members $REF_MEMBERS --member-block-days 5 \
      --member-seed0 93000 --warming-step-wm2 0.0 \
      --warming-ramp-wm2-per-day $RAMP --allow-eval-roles \
      --output-dir $REFS_ROOT/exp3b_eval_b$b \
      > $OUT/logs/references_b$b.log 2>&1 &
  fi
done
wait
for b in 0 1 2; do
  refs_done $REFS_ROOT/exp3b_eval_b$b 16 || { echo "REFERENCES b$b FAILED"; exit 1; }
done
echo "[$(ts)] references ready"
fi

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
    --representation zonal"
  sens="--sensitivity-from $TRAIN/responses/ic*_responses.npz
    --sensitivity-references-dir $REFS_ROOT/exp3b_train_b0"
  case $arm in
    uncontrolled) cmd="run_test_world.py episode $common --uniform 0.0" ;;
    fixed)        cmd="run_test_world.py episode $common --amplitudes $FIXED" ;;
    pi)           cmd="run_controllers.py feedback --controller pi $common $sens --feedforward --closed-loop-days $PI_DAYS" ;;
    adaptive)     cmd="run_controllers.py feedback --controller adaptive $common $sens --pattern $FIXED" ;;
    plan_oracle)  cmd="run_test_world.py plan $common $planner" ;;
    plan_naive)   cmd="run_test_world.py plan $common $planner --planner-efficacy 1" ;;
    plan_learn)   cmd="run_test_world.py plan $common $planner --learn-strength $LEARN_MODE --learn-noise-k $NOISE_K" ;;
  esac
  $PY $cmd > $OUT/logs/${arm}_ic${idx}.log 2>&1 \
    && echo "[$(ts)] $arm ic$idx finished" || echo "[$(ts)] $arm ic$idx FAILED"
}
export -f job ts
export PY ICS REFS_ROOT REF_SUFFIX TRAIN TABLE OUT RAMP MU LAM MEMBERS SEGMENTS SCORE_START

# (position, index, branch): macro states 17, 19, ..., 47 x branches 0-2.
states() {
  if [ "${SMOKE:-0}" = 1 ]; then echo "0 1600 0"; return; fi
  for p in $(seq 0 47); do
    m=$((17 + 2 * (p / 3))); b=$((p % 3))
    printf "%d %04d %d\n" $p $((100 * m + b)) $b; done; }
{
  for arm in plan_learn plan_naive plan_oracle pi adaptive fixed uncontrolled; do
    states | sed "s/^/$arm /"
  done
} | xargs -P $PARALLEL -L 1 bash -c 'job $0 $1 $2 $3'

if [ "${SMOKE:-0}" = 1 ]; then
  echo "[$(ts)] smoke runs:"; ls $OUT/*/ic1600.json
  grep -l Traceback $OUT/logs/*.log
  JAX_PLATFORMS=cpu $PY analyze_experiment3b.py --runs-dir $OUT \
    --references-dirs $REFS_ROOT/exp3b_train_b0 --output $OUT/analysis.json \
    --smoke || echo "SMOKE ANALYSIS FAILED"
  exit 0
fi
echo "[$(ts)] analysis"
JAX_PLATFORMS=cpu $PY analyze_experiment3b.py --runs-dir $OUT \
  --references-dirs $REFS_ROOT/exp3b_eval_b0 $REFS_ROOT/exp3b_eval_b1 \
  $REFS_ROOT/exp3b_eval_b2 --output $E/exp3b/exp3b_analysis.json \
  > $OUT/logs/analysis.log 2>&1 || echo "ANALYSIS FAILED (see logs/analysis.log)"
tail -20 $OUT/logs/analysis.log
echo "[$(ts)] ========== EXP3B EVALUATION COMPLETE =========="
