#!/bin/bash
# Experiment 3b, training side (MCB_PROJECT_REPORT.md Part 24; Amendment 9
# revision 2 is frozen only after this and the pilot).
#
#   1. 32 fresh macro states (16-47), continued from macro state 15 every 365
#      days, with their weather branches: exp3b_train (even states, branches
#      0-1) and exp3b_eval (odd states, branches 0-2). Generating states uses
#      no outcome; evaluation references wait for revision 2.
#   2. Ramp-6 references (5 members x 200 d) for the training states:
#      branch 0 (the designs) and branch 1 (the pilot), side by side into
#      separate folders.
#   3. One-band response runs on the 16 branch-0 states (the PI and adaptive
#      sensitivities and the fixed design), at nominal strength.
#   4. The fixed design (ladder rung 4, zonal objective, days 98-182).
# Re-running skips whatever already exists.
ts() { date +%H:%M:%S; }

cd ~/workspace/jax-gcm || exit 1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.12
export PYTHONUNBUFFERED=1
PY=~/mcb-env/bin/python
E=mcb_experiments_gpu
ROOT=$E/ics_macro3b
ICS=$ROOT/exp3b_train
REFS=$E/test_world_refs_ramp6_3b
OUT=$E/exp3b/train
RAMP=0.03296703296703297        # 6 W m-2 at day 182 (as Experiment 3a)
MU=0.0006325
PARALLEL=${PARALLEL:-4}
mkdir -p $OUT/logs $OUT/responses

if [ ! -f $ROOT/exp3b_eval/manifest.json ]; then
  echo "[$(ts)] generating macro states 16-47 and their branches"
  $PY run_generate_macro_ics.py --plan exp3b --macro-offset 15 \
    --num-macro 33 --spacing-days 365 --require-qflux \
    --base-carry $E/ics_macro/macro_bases/macro_15_carry.pkl \
    --output-root $ROOT > $OUT/logs/generate.log 2>&1 \
    || { echo "GENERATION FAILED"; exit 1; }
  grep -E "ocean-mean SST|trend|split balance" $OUT/logs/generate.log | tail -40
fi
echo "[$(ts)] states ready"

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
  if ! refs_done $REFS/exp3b_train_b$b 16; then
    echo "[$(ts)] references: branch $b, 16 states x 5 members x 200 d"
    $PY run_test_world.py references --ic-dir $ICS --branches $b \
      --days 200 --members 5 --member-block-days 5 --member-seed0 93000 \
      --warming-step-wm2 0.0 --warming-ramp-wm2-per-day $RAMP \
      --output-dir $REFS/exp3b_train_b$b > $OUT/logs/references_b$b.log 2>&1 &
  fi
done
wait
for b in 0 1; do
  refs_done $REFS/exp3b_train_b$b 16 || { echo "REFERENCES b$b FAILED"; exit 1; }
done
echo "[$(ts)] references ready"

job() {
  pos=$1; idx=$2
  out=$OUT/responses/ic${idx}_responses.npz
  [ -f $out ] && return 0
  $PY run_controllers.py responses --ic-dir $ICS --ic-position $pos \
    --references $REFS/exp3b_train_b0/ic${idx}_references.npz --days 182 \
    --delta 0.1 --warming-step-wm2 0.0 --warming-ramp-wm2-per-day $RAMP \
    --output-dir $OUT/responses > $OUT/logs/responses_ic${idx}.log 2>&1 \
    && echo "[$(ts)] responses ic$idx finished" \
    || echo "[$(ts)] responses ic$idx FAILED"
}
export -f job ts
export PY ICS REFS OUT RAMP

# Branch-0 states sit at even manifest positions: macro 16 + 2i, index
# 100 * (16 + 2i).
for i in $(seq 0 15); do
  printf "%d %04d\n" $((2 * i)) $((100 * (16 + 2 * i)))
done | xargs -P $PARALLEL -L 1 bash -c 'job $0 $1'

echo "[$(ts)] fixed design (zonal, days 98-182)"
$PY run_controllers.py ladder --responses $OUT/responses/ic*_responses.npz \
  --references-dir $REFS/exp3b_train_b0 --window 98 182 --mu $MU \
  --representation zonal --output $OUT/ladder.json \
  > $OUT/logs/ladder.log 2>&1 || { echo "LADDER FAILED"; exit 1; }
tail -4 $OUT/logs/ladder.log
echo "[$(ts)] ========== EXP3B TRAINING SIDE COMPLETE =========="
