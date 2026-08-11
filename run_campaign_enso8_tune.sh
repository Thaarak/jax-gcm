#!/bin/bash
# Amendment 7 revision 2, STEP 1 of 2: fresh ICs + feedback-gain tuning sweep.
# (PREREGISTRATION.md "Amendment 7, revision 2", frozen 2026-08-10.)
#
# Why this step exists: the anticipation ablation must not be run against a
# DETUNED comparator. Deleting the feedforward term while freezing the
# feedback gain at the value chosen when feedforward carried the disturbance
# understates a purely reactive controller by ~7 mK/K (surrogate: +12.5 at
# fb=1 vs +1.8 at fb=6). So both laws get their gain tuned here, on TRAIN
# ICs only, by the registered rule: minimise RMS_A, ties to the SMALLER gain.
#
# Cost: ICs ~12 min (already done on a rerun); sweep (9 arms x 6 train ICs
# x k=2, baselines shared) ~35 min. Idempotent — rerun to resume.
#
# v2: the first sweep was INVALID — KIND@FF was read as a raw
# enso_effect_per_K, so "@1" meant a feedforward gain of 1.0 K/K, 18x the
# measured 0.05473. FF/FB are now WEIGHTS (0 = ablate, 1 = designed
# strength) and a regression test pins pi-enso@1@1 == plain pi-enso.
CONTAINER=aeon-vllm
ts() { date +%H:%M:%S; }
step() { echo ""; echo "[$(ts)] ========== $1 =========="; }

echo "[$(ts)] stopping $CONTAINER"
docker stop $CONTAINER >/dev/null 2>&1
trap 'echo "[$(ts)] restarting '"$CONTAINER"'"; docker start '"$CONTAINER"' >/dev/null 2>&1' EXIT INT TERM

cd ~/workspace/jax-gcm || exit 1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
export PYTHONUNBUFFERED=1
PY=~/mcb-env/bin/python

E=mcb_experiments_gpu
BASE=$E/equilibrated/base_carry.pkl
PAT=$E/stage1_pattern_enso7.pkl          # metric-space rescaled pattern
ICS=$E/ics_enso8
DAYS=180
CI=15
TAIL=60
CAP=0.09
FF=0.05473                                # measured on the registered metric
AMAX=1.8273                               # frozen from enso7_plant.json

for f in "$BASE" "$PAT"; do
  [ -f "$f" ] || { echo "MISSING PREREQUISITE: $f"; exit 1; }
done

step "1/2 Fresh ICs (seed0 9000; 6 train + 24 held-out, horizon $DAYS)"
if [ -f $ICS/manifest.json ]; then
  echo "  $ICS exists — skip"
else
  $PY run_generate_ics_independent.py --base-carry $BASE \
    --num-train 6 --num-heldout 24 --decorr-days 30 --horizon $DAYS \
    --seed0 9000 --output-dir $ICS || { echo "IC GEN FAILED"; exit 1; }
fi

step "2/2 Feedback-gain sweep on TRAIN ICs (ff=0 reactive, ff=1 anticipating)"
if [ -f $E/enso8_tune_v2.pkl ]; then
  echo "  $E/enso8_tune_v2.pkl exists — skip"
else
  $PY run_confirmatory_eval.py --ic-dir $ICS --split train \
    --arm b1=pi-enso@0@1=$PAT \
    --arm b15=pi-enso@0@1.5=$PAT \
    --arm b2=pi-enso@0@2=$PAT \
    --arm b3=pi-enso@0@3=$PAT \
    --arm b4=pi-enso@0@4=$PAT \
    --arm b6=pi-enso@0@6=$PAT \
    --arm p1=pi-enso@1@1=$PAT \
    --arm p2=pi-enso@1@2=$PAT \
    --arm p3=pi-enso@1@3=$PAT \
    --comparator p1 \
    --members 2 --days $DAYS --control-interval $CI --tail-days $TAIL \
    --max-perturbation $CAP --enso-effect-per-k $FF \
    --enso-mode randomized --enso-amp-design deterministic \
    --enso-amp-range -$AMAX $AMAX \
    --output $E/enso8_tune_v2.pkl || { echo "SWEEP FAILED"; exit 1; }
fi

echo ""
echo "[$(ts)] TUNING SWEEP DONE -> $E/enso8_tune_v2.pkl"
echo "Next: pull it and apply the registered selection rule before the eval."
