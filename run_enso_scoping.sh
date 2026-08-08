#!/bin/bash
# ENSO scoping run (exploratory calibration — see PREREGISTRATION.md
# pre-amendment note, 2026-08-08), diya.
#
# Measures GMST-per-Nino3.4 sensitivity, pacemaker fidelity, El Nino /
# La Nina asymmetry, and the 365-day chaos noise floor. No gates.
# Cost: (1 control + 2 arms) x 4 members x 365 d ~= 12 x ~100 s GPU
# + compile — well under an hour.
CONTAINER=aeon-vllm
ts() { date +%H:%M:%S; }

echo "[$(ts)] stopping $CONTAINER"
docker stop $CONTAINER >/dev/null 2>&1
trap 'echo "[$(ts)] restarting '"$CONTAINER"'"; docker start '"$CONTAINER"' >/dev/null 2>&1' EXIT INT TERM

cd ~/workspace/jax-gcm || exit 1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
export PYTHONUNBUFFERED=1
PY=~/mcb-env/bin/python

E=mcb_experiments_gpu

echo "[$(ts)] ENSO scoping: 365 d, 4 members, A = +/-2 K"
$PY run_enso_scoping.py \
  --base-carry $E/equilibrated/base_carry.pkl \
  --days 365 --members 4 \
  --amplitude 2.0 --tau-days 5 --ramp-days 30 \
  --tail-days 90 \
  --output $E/enso_scoping.pkl \
  || { echo "[$(ts)] SCOPING FAILED"; exit 1; }

echo "[$(ts)] DONE -> $E/enso_scoping.pkl"
