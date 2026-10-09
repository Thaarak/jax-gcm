#!/bin/bash
# Experiment 3d, preparation (MCB_PROJECT_REPORT.md Part 28; Amendment 9
# revision 3 is frozen only after this and the pilots). Everything runs with
# the fixed land model (JCM_LAND_CLIMATOLOGY=monthly).
#
#   1. Base climate by the procedure of revisions 0.2-0.3
#      (run_qflux_base_climate.py): diagnose a monthly Q-flux, settle 10
#      years and apply the gate (G1 drift < 0.02 K per 60 d, G2 |mean SST
#      - observed| < 0.5 K). If the gate fails: one Newton correction and a
#      second settle; if that fails too, stop.
#   2. 32 macro states (100-131) continued from the settled carry every 365
#      days, with their weather branches: exp3d_train (even states, branches
#      0-1) and exp3d_eval (odd states, branches 0-2). Generating states
#      uses no outcome; evaluation references wait for revision 3.
#   3. The new states' hidden strengths (3b's distribution and seed rule).
#   4. Ramp-6 references (5 members x 200 d) for the training states, the
#      one-band response runs on the 16 branch-0 states, and the fixed
#      design, exactly as 3b's training side.
#   5. The oracle pilot on the 16 branch-1 training states (3 members): the
#      learner's noise level by 3b's rule (analyze_exp3b_pilot.py noise).
# Re-running skips whatever already exists.
ts() { date +%H:%M:%S; }

cd ~/workspace/jax-gcm || exit 1
export JCM_LAND_CLIMATOLOGY=monthly
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.12
export PYTHONUNBUFFERED=1
PY=~/mcb-env/bin/python
E=mcb_experiments_gpu
Q=$E/exp3d/qflux
ROOT=$E/ics_macro3d
REFS=$E/test_world_refs_ramp6_3d
TRAIN=$E/exp3d/train
PILOT=$E/exp3d/pilot
TABLE=$E/exp3d/hidden_strength.json
RAMP=0.03296703296703297        # 6 W m-2 at day 182 (as 3a and 3b)
MU=0.0006325
LAM=0.006325
PARALLEL=${PARALLEL:-4}
mkdir -p $Q $TRAIN/logs $TRAIN/responses $PILOT/logs
$PY -c "from jcm.mcb.land_climatology import land_climatology_mode as m; assert m() == 'monthly', m()" \
  || { echo "NOT THE FIXED LAND MODEL"; exit 1; }

# 1. Base climate.
if [ ! -f $Q/qflux_monthly_t30.nc ]; then
  echo "[$(ts)] base climate: diagnosing the Q-flux (365 + 1460 days)"
  $PY run_qflux_base_climate.py diagnose --output-dir $Q \
    > $Q/diagnose.log 2>&1 || { echo "DIAGNOSE FAILED"; exit 1; }
  grep -E "Q-flux|ocean-mean annual" $Q/diagnose.log | tail -2
fi
gate_passed() {  # gate_passed <settle dir>
  $PY -c "import json, sys; sys.exit(0 if json.load(open('$1/settle_summary.json'))['gate']['pass'] else 1)" 2>/dev/null
}
settle() {  # settle <qflux> <dir>
  if [ ! -f $2/settle_summary.json ]; then
    echo "[$(ts)] base climate: settling 3650 days with $1"
    $PY run_qflux_base_climate.py settle --qflux $1 --output-dir $2 \
      --base-carry-out $2/base_carry.pkl > $2.log 2>&1
  fi
  grep -E "^GATE" $2.log
}
settle $Q/qflux_monthly_t30.nc $Q/settle1
if gate_passed $Q/settle1; then
  BASE=$Q/settle1/base_carry.pkl
else
  if [ ! -f $Q/qflux_monthly_t30_v2.nc ]; then
    echo "[$(ts)] base climate: gate failed; one Newton correction (revision 0.3)"
    $PY run_qflux_base_climate.py correct --qflux $Q/qflux_monthly_t30.nc \
      --settle-dir $Q/settle1 --output $Q/qflux_monthly_t30_v2.nc \
      > $Q/correct.log 2>&1 || { echo "CORRECTION FAILED"; exit 1; }
  fi
  settle $Q/qflux_monthly_t30_v2.nc $Q/settle2
  gate_passed $Q/settle2 \
    || { echo "BASE CLIMATE FAILED ITS GATE TWICE: stopping"; exit 1; }
  BASE=$Q/settle2/base_carry.pkl
fi
echo "[$(ts)] base climate passed: $BASE"

# 2. States.
if [ ! -f $ROOT/exp3d_eval/manifest.json ]; then
  echo "[$(ts)] generating macro states 100-131 and their branches"
  $PY run_generate_macro_ics.py --plan exp3d --macro-offset 99 \
    --num-macro 33 --spacing-days 365 --require-qflux --base-carry $BASE \
    --output-root $ROOT > $TRAIN/logs/generate.log 2>&1 \
    || { echo "GENERATION FAILED"; exit 1; }
  grep -E "trend|split balance" $TRAIN/logs/generate.log | tail -3
fi
echo "[$(ts)] states ready"

# 3. Hidden strengths.
if [ ! -f $TABLE ]; then
  $PY -m jcm.mcb.hidden_strength --manifests $ROOT/exp3d_train/manifest.json \
    $ROOT/exp3d_eval/manifest.json --output $TABLE \
    || { echo "STRENGTH TABLE FAILED"; exit 1; }
fi
echo "[$(ts)] hidden strengths: $(sha256sum $TABLE | cut -c1-16)"

# 4. Training side.
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
for b in 0 1; do
  if ! refs_done $REFS/exp3d_train_b$b 16; then
    echo "[$(ts)] references: branch $b, 16 states x 5 members x 200 d"
    $PY run_test_world.py references --ic-dir $ROOT/exp3d_train --branches $b \
      --days 200 --members 5 --member-block-days 5 --member-seed0 93000 \
      --warming-step-wm2 0.0 --warming-ramp-wm2-per-day $RAMP \
      --output-dir $REFS/exp3d_train_b$b > $TRAIN/logs/references_b$b.log 2>&1 &
  fi
done
wait
for b in 0 1; do
  refs_done $REFS/exp3d_train_b$b 16 || { echo "REFERENCES b$b FAILED"; exit 1; }
done
echo "[$(ts)] training references ready"

responses_job() {
  pos=$1; idx=$2
  out=$TRAIN/responses/ic${idx}_responses.npz
  [ -f $out ] && return 0
  $PY run_controllers.py responses --ic-dir $ROOT/exp3d_train --ic-position $pos \
    --references $REFS/exp3d_train_b0/ic${idx}_references.npz --days 182 \
    --delta 0.1 --warming-step-wm2 0.0 --warming-ramp-wm2-per-day $RAMP \
    --output-dir $TRAIN/responses > $TRAIN/logs/responses_ic${idx}.log 2>&1 \
    && echo "[$(ts)] responses ic$idx finished" \
    || echo "[$(ts)] responses ic$idx FAILED"
}
export -f responses_job ts
export PY ROOT REFS TRAIN RAMP JCM_LAND_CLIMATOLOGY

# Branch-0 training states sit at even manifest positions: macro 100 + 2i,
# index 100 * (100 + 2i).
for i in $(seq 0 15); do
  echo "$((2 * i)) $((100 * (100 + 2 * i)))"
done | xargs -P $PARALLEL -L 1 bash -c 'responses_job $0 $1'

if [ ! -f $TRAIN/ladder.json ]; then
  echo "[$(ts)] fixed design (zonal, days 98-182)"
  $PY run_controllers.py ladder --responses $TRAIN/responses/ic*_responses.npz \
    --references-dir $REFS/exp3d_train_b0 --window 98 182 --mu $MU \
    --representation zonal --output $TRAIN/ladder.json \
    > $TRAIN/logs/ladder.log 2>&1 || { echo "LADDER FAILED"; exit 1; }
  tail -4 $TRAIN/logs/ladder.log
fi

# 5. The oracle pilot (branch-1 training states: odd positions, index
# 100 m + 1) and the learner's noise level by 3b's rule.
oracle_job() {
  pos=$1; idx=$2
  out=$PILOT/plan_oracle/ic${idx}.json
  [ -f $out ] && return 0
  mkdir -p $PILOT/plan_oracle
  strength=$($PY -c "import json; print(' '.join(str(x) for x in json.load(open('$TABLE'))['states']['$idx']['strength']))")
  $PY run_test_world.py plan --ic-dir $ROOT/exp3d_train --ic-position $pos \
    --references $REFS/exp3d_train_b1/ic${idx}_references.npz \
    --segments 13 --segment-days 14 --members 3 --score-start-day 98 \
    --warming-step-wm2 0.0 --warming-ramp-wm2-per-day $RAMP --mu $MU \
    --lam $LAM --efficacy $strength --save-fields --output $out \
    --preset short14 --optimizer gauss_newton --iterations 1 --copies 1 \
    --representation zonal > $PILOT/logs/plan_oracle_ic${idx}.log 2>&1 \
    && echo "[$(ts)] plan_oracle ic$idx finished" \
    || echo "[$(ts)] plan_oracle ic$idx FAILED"
}
export -f oracle_job
export PILOT TABLE MU LAM
echo "[$(ts)] oracle pilot: 16 states x 3 members"
for i in $(seq 0 15); do
  echo "$((2 * i + 1)) $((100 * (100 + 2 * i) + 1))"
done | xargs -P $PARALLEL -L 1 bash -c 'oracle_job $0 $1'
JAX_PLATFORMS=cpu $PY analyze_exp3b_pilot.py noise --runs-dir $PILOT \
  --references-dir $REFS/exp3d_train_b1 || { echo "NOISE RULE FAILED"; exit 1; }
echo "[$(ts)] learner noise level: $($PY -c "import json; print(json.load(open('$PILOT/noise.json'))['noise_k'])") K"
echo "[$(ts)] ========== EXP3D PREPARATION COMPLETE =========="
