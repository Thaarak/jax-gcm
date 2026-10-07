#!/bin/bash
# Experiment 2 and the snip test (Amendment 9 revision 1.1), on the GX10.
#
# Run from an export of the FROZEN commit, never from the shared working copy
# that other experiments deploy into:
#   git archive <frozen commit> | ssh gx10 'mkdir -p ~/workspace/jax-gcm-exp2 &&
#     tar -x -C ~/workspace/jax-gcm-exp2'
# with ~/workspace/jax-gcm-exp2/mcb_experiments_gpu/ics_macro linked to the
# shared, read-only starting states. Outputs go to mcb_experiments_gpu/exp2/
# and mcb_experiments_gpu/snip_test/ inside the export.
#
# Order (each phase resumes: finished outputs are skipped):
#   0. the snip test (exp1 states; Experiment 1's truth is in the export);
#   1. training: references, one-band responses (b5, b13), timing, and the
#      gradient designs (b5 snipped, b5 full BPTT, b13 snipped);
#   2. the design stage: designs.json, whose SHA-256 is printed and logged
#      before any evaluation reference exists;
#   3. evaluation references (the first use of exp2_eval);
#   4. 11 arms x 16 states, PARALLEL at a time;
#   5. the registered analyses, on the CPU.
# Memory: every JAX process is capped (XLA_PYTHON_CLIENT_MEM_FRACTION), and
# the analyses run with JAX_PLATFORMS=cpu. An uncapped run filled the GX10's
# shared memory and rebooted it on 2026-10-07.
set -u
ts() { date +%H:%M:%S; }
cd "${EXP2_DIR:-$HOME/workspace/jax-gcm-exp2}" || exit 1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${MEM_FRACTION:-0.10}
export PYTHONUNBUFFERED=1
PY=${PY:-$HOME/mcb-env/bin/python}
PARALLEL=${PARALLEL:-2}
E=mcb_experiments_gpu
TRAIN=$E/ics_macro/exp2_train
EVAL=$E/ics_macro/exp2_eval
OUT=$E/exp2
mkdir -p $OUT/logs $OUT/train/responses $OUT/train/gradient $E/snip_test

step() {  # step <done-file> <log-name> <command...>
  local done=$1 log=$2; shift 2
  if [ -e "$done" ]; then echo "[$(ts)] $log: done, skipping"; return 0; fi
  echo "[$(ts)] $log: start"
  if "$@" > $OUT/logs/$log.log 2>&1; then echo "[$(ts)] $log: finished"
  else echo "[$(ts)] $log: FAILED (see $OUT/logs/$log.log)"; exit 1; fi
}

# 0. The snip test (about 1 GPU-hour).
step $E/snip_test/snip_test.npz snip_test \
  $PY run_snip_test.py --ic-dir $E/ics_macro/exp1 --output $E/snip_test/snip_test

# 1. Training phase (exp2_train only).
step $OUT/train/refs/references_manifest.json references_train \
  $PY run_experiment2.py references --ic-dir $TRAIN --output-dir $OUT/train/refs
for layout in b5 b13; do
  step $OUT/train/responses/ic1401_responses_$layout.npz responses_$layout \
    $PY run_experiment2.py responses --ic-dir $TRAIN --layout $layout \
    --output-dir $OUT/train/responses
done
step $OUT/train/timing.json timing \
  $PY run_experiment2.py timing --ic-dir $TRAIN --output $OUT/train/timing.json
for spec in b5:snipped b5:bptt b13:snipped; do
  layout=${spec%%:*}; mode=${spec##*:}
  step $OUT/train/gradient/gradient_${layout}_${mode}.json gradient_${layout}_$mode \
    $PY run_experiment2.py gradient --ic-dir $TRAIN \
    --references-dir $OUT/train/refs --layout $layout --mode $mode \
    --output $OUT/train/gradient/gradient_${layout}_${mode}.json
done

# 2. Freeze the designs (no model).
step $OUT/designs.json design \
  env JAX_PLATFORMS=cpu $PY run_experiment2.py design \
  --references-dir $OUT/train/refs --responses-dir $OUT/train/responses \
  --timing $OUT/train/timing.json --gradient-dir $OUT/train/gradient \
  --output $OUT/designs.json
echo "[$(ts)] designs.json SHA-256: $(sha256sum $OUT/designs.json | cut -d' ' -f1)" \
  | tee -a $OUT/logs/designs_sha256.txt

# 3. Evaluation references (first use of exp2_eval).
step $OUT/eval/refs/references_manifest.json references_eval \
  $PY run_experiment2.py references --ic-dir $EVAL --output-dir $OUT/eval/refs \
  --allow-eval-roles

# 4. Every arm on every evaluation state.
arms="uncontrolled uniform brute5 brute5_eq grad5 bptt5 sunlight5 brute13 brute13_eq grad13 sunlight13"
jobs=()
for pos in $(seq 0 15); do for arm in $arms; do jobs+=("$pos $arm"); done; done
run_one() {
  local pos=$1 arm=$2
  local idx; idx=$($PY -c "import json;print(json.load(open('$EVAL/manifest.json'))['ics'][$pos]['index'])")
  local done; done=$(printf "%s/eval/runs/%s/ic%04d.json" $OUT $arm $idx)
  if [ -e "$done" ]; then return 0; fi
  $PY run_experiment2.py episode --ic-dir $EVAL --ic-position $pos \
    --references-dir $OUT/eval/refs --designs $OUT/designs.json --arm $arm \
    --output-dir $OUT/eval/runs --allow-eval-roles \
    > $OUT/logs/episode_${arm}_$pos.log 2>&1 \
    || { echo "[$(ts)] $arm $pos FAILED, retrying once"; \
         $PY run_experiment2.py episode --ic-dir $EVAL --ic-position $pos \
           --references-dir $OUT/eval/refs --designs $OUT/designs.json \
           --arm $arm --output-dir $OUT/eval/runs --allow-eval-roles \
           >> $OUT/logs/episode_${arm}_$pos.log 2>&1 \
           || echo "[$(ts)] $arm $pos FAILED twice"; }
}
export -f run_one ts
export PY EVAL OUT
echo "[$(ts)] evaluation: ${#jobs[@]} episodes, $PARALLEL at a time"
printf "%s\n" "${jobs[@]}" | xargs -P $PARALLEL -L 1 bash -c 'run_one $0 $1'

# 5. The registered analyses (CPU).
JAX_PLATFORMS=cpu $PY analyze_snip_test.py --snip-prefix $E/snip_test/snip_test \
  --output $E/snip_test/snip_test_analysis.json
JAX_PLATFORMS=cpu $PY analyze_experiment2.py --runs-dir $OUT/eval/runs \
  --references-dir $OUT/eval/refs --designs $OUT/designs.json \
  --output $OUT/exp2_analysis.json
echo "[$(ts)] ========== EXPERIMENT 2 AND THE SNIP TEST COMPLETE =========="
