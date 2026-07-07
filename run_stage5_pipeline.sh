#!/usr/bin/env bash
# Stage 5 full pipeline runner (diya GB10). Launched inside tmux.
# Stops vLLM before JAX, ALWAYS restarts it at the end (trap), even on failure.
set -uo pipefail

cd ~/workspace/jax-gcm
PY=~/mcb-env/bin/python
S5=mcb_experiments/stage5

# Always restore the vLLM container on exit (success, failure, or abort).
restore_vllm() {
  echo "[pipeline] restoring vLLM container aeon-ultimate-xs ..."
  docker start aeon-ultimate-xs || echo "[pipeline] WARN: docker start failed"
}
trap restore_vllm EXIT

echo "[pipeline] stopping vLLM container to free GPU memory ..."
docker stop aeon-ultimate-xs || echo "[pipeline] WARN: docker stop failed (continuing)"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
export PYTHONUNBUFFERED=1

echo "[pipeline] ===== IC GENERATION ====="
$PY run_stage5_generate_ics.py --realistic-terrain \
  --output-dir $S5/ics 2>&1 | tee $S5/ics_gen.log
if [ ! -f $S5/ics/manifest.json ]; then
  echo "[pipeline] FATAL: IC generation did not produce manifest.json; aborting."
  touch $S5/DONE
  exit 1
fi

echo "[pipeline] ===== TRAINING (150 epochs) ====="
$PY run_stage5_training.py --epochs 150 \
  --init-checkpoint mcb_experiments/stage4/stage4_trained_policy.pkl \
  --ic-dir $S5/ics --output-dir $S5 \
  --amazon 0.05 --sahel 0.05 --tropics 0.05 2>&1 | tee $S5/train.log
if [ ! -f $S5/stage5_trained_policy.pkl ]; then
  echo "[pipeline] FATAL: training did not produce stage5_trained_policy.pkl; aborting."
  touch $S5/DONE
  exit 1
fi

echo "[pipeline] ===== EVALUATION ====="
$PY run_stage5_eval.py --ic-dir $S5/ics \
  --checkpoint $S5/stage5_trained_policy.pkl \
  --stage4-checkpoint mcb_experiments/stage4/stage4_trained_policy.pkl \
  --stage1 mcb_experiments/stage1/stage1_optimized_pattern.pkl \
  2>&1 | tee $S5/eval.log

echo "[pipeline] ===== DONE ====="
touch $S5/DONE
