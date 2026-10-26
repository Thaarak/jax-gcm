# Shared by run_campaign_exp3f_pilot.sh and run_campaign_exp3f_eval.sh
# (Experiment 3f, Amendment 9 revision 5; sourced, not run).
#
# Every arm is 3d's learning planner (Gauss-Newton, 1 step, 1 copy, zonal,
# exact sensing, per-band learner at 3d's registered noise level), learning
# from its two-week forecast of the segment it applies. Only the look-ahead
# it plans over differs:
#   learn14        two weeks (Dubey et al.'s receding horizon; = 3d's plan_learn)
#   learn112_mpc   16 weeks, level + trend per band over 8 blocks, 30-day snip
#   learn56_mpc     8 weeks, the same over 4 blocks
#   learn112_hold  16 weeks, one setting held (Experiment 3a's objective), 30-day snip
# The long arms' target beyond the scoring references' 200 days comes from a
# normal-climate-only extension (run_test_world.py plan --target-extension).
export JCM_LAND_CLIMATOLOGY=monthly
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${MEM_FRACTION:-0.12}
export PYTHONUNBUFFERED=1
PY=~/mcb-env/bin/python
E=mcb_experiments_gpu
TABLE=$E/exp3d/hidden_strength.json
RAMP=0.03296703296703297
MU=0.0006325
LAM=0.006325
NOISE_K=0.00069104
LEARN_MODE=bands
EXT_DAYS=280            # last re-plan (day 168) + the 112-day look-ahead
ts() { date +%H:%M:%S; }

arm_flags() {   # arm -> its planner flags (the target extension is added by job)
  base="--optimizer gauss_newton --iterations 1 --copies 1 --representation zonal
    --learn-strength $LEARN_MODE --learn-noise-k $NOISE_K --preset short14"
  case $1 in
    learn14)       echo "$base" ;;
    learn112_mpc)  echo "$base --lookahead-days 112 --window-days 30 --schedule linear" ;;
    learn56_mpc)   echo "$base --lookahead-days 56 --window-days 30 --schedule linear" ;;
    learn112_hold) echo "$base --lookahead-days 112 --window-days 30 --schedule hold" ;;
    *) echo "UNKNOWN ARM $1" >&2; return 1 ;;
  esac
}

# build_extension <ic-dir> <branch> <members> <out-dir> [--allow-eval-roles]
build_extension() {
  [ -f $4/references_manifest.json ] && grep -q finished_utc $4/references_manifest.json && return 0
  $PY run_test_world.py references --ic-dir $1 --branches $2 --days $EXT_DAYS \
    --members $3 --member-block-days 0 --member-seed0 93000 --no-warmed \
    --warming-step-wm2 0.0 --warming-ramp-wm2-per-day $RAMP $5 \
    --output-dir $4
}

# job <arm> <position> <index> <branch>; needs ICS REFS_ROOT REF_SET EXT_ROOT
# OUT SEGMENTS MEMBERS SCORE_START.
job() {
  arm=$1; pos=$2; idx=$3; branch=$4
  out=$OUT/$arm/ic${idx}.json
  [ -f $out ] && return 0
  mkdir -p $OUT/$arm $OUT/logs
  tries=$OUT/logs/${arm}_ic${idx}.tries
  n=$(cat $tries 2>/dev/null || echo 0)
  if [ "$n" -ge 2 ]; then echo "[$(ts)] $arm ic$idx failed twice: not run again"; return 0; fi
  echo $((n + 1)) > $tries
  log=$OUT/logs/${arm}_ic${idx}$([ "$n" -ge 1 ] && echo .retry).log
  refs=$REFS_ROOT/${REF_SET}_b$branch/ic${idx}_references.npz
  ext=""
  [ $arm != learn14 ] && ext="--target-extension $EXT_ROOT/${REF_SET}_b$branch/ic${idx}_references.npz"
  strength=$($PY -c "import json; print(' '.join(str(x) for x in json.load(open('$TABLE'))['states']['$idx']['strength']))")
  $PY run_test_world.py plan --ic-dir $ICS --ic-position $pos --references $refs \
    --segments $SEGMENTS --segment-days 14 --members $MEMBERS \
    --score-start-day $SCORE_START --warming-step-wm2 0.0 \
    --warming-ramp-wm2-per-day $RAMP --mu $MU --lam $LAM --efficacy $strength \
    --save-fields --output $out $(arm_flags $arm) $ext > $log 2>&1 \
    && echo "[$(ts)] $arm ic$idx finished" || echo "[$(ts)] $arm ic$idx FAILED"
}
gpu_visible() { $PY -c "import jax; assert jax.devices()[0].platform == 'gpu'" > /dev/null 2>&1; }
export -f job arm_flags ts
export PY TABLE RAMP MU LAM NOISE_K LEARN_MODE
