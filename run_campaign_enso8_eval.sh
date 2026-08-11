#!/bin/bash
# Amendment 7 revision 3, STEP 2 of 2: the anticipation ablation, held-out.
# (PREREGISTRATION.md "Amendment 7, revision 3", frozen 2026-08-11.)
#
# H10: does ANTICIPATING an observed disturbance (feedforward on the Nino3.4
# index) beat REACTING to the error it causes (outcome feedback alone)?
#
# The comparator problem, and why this design. Revision 2 tuned the reactive
# controller's gain on train ICs, but the sweep put the optimum AT THE GRID
# EDGE (fb=6, largest tested, curve still falling) with heavily overlapping
# bootstrap CIs. Declaring fb=6 "the best reactive controller" would risk the
# very strawman the tuning step exists to prevent. So instead the eval carries
# a LADDER of reactive gains and the H10 comparator is defined as WHICHEVER
# SCORES BEST ON THE HELD-OUT DATA. Selecting the comparator on the test set
# inflates it — that is the CONSERVATIVE direction, biasing against H10 — and
# it removes any "you detuned the opponent" objection entirely.
#
# Cost: 6 arms x 24 held-out ICs x k=4 at 180 d, baselines shared:
# ~38 min per arm + ~35 min baselines = ~3.1 GPU-h. Idempotent.
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
PAT=$E/stage1_pattern_enso7.pkl
CKPT=$E/enso7_imitation_s72.pkl
ICS=$E/ics_enso8
DAYS=180
CI=15
TAIL=60
CAP=0.09
FF=0.05473
AMAX=1.8273

for f in "$PAT" "$CKPT" "$ICS/manifest.json"; do
  [ -f "$f" ] || { echo "MISSING PREREQUISITE: $f"; exit 1; }
done

echo ""
echo "[$(ts)] ===== FINAL EVAL: 24 held-out ICs, k=4, deterministic A ====="
echo "  reactive ladder b6/b12 (comparator = best on held-out, conservative)"
echo "  anticipating p1 (= the Amendment-7 controller) and p2 (train-selected)"

if [ -f $E/enso8_eval.pkl ] && $PY -c "
import pickle; r=pickle.load(open('mcb_experiments_gpu/enso8_eval.pkl','rb'))
raise SystemExit(0 if 'gates' in r else 1)" 2>/dev/null; then
  echo "  enso8_eval.pkl already complete — skip"
else
  $PY run_confirmatory_eval.py --ic-dir $ICS --split heldout \
    --arm static=static=$PAT \
    --arm b6=pi-enso@0@6=$PAT \
    --arm b12=pi-enso@0@12=$PAT \
    --arm p1=pi-enso@1@1=$PAT \
    --arm p2=pi-enso@1@2=$PAT \
    --arm imitation_s72=fc14=$CKPT \
    --comparator static \
    --members 4 --days $DAYS --control-interval $CI --tail-days $TAIL \
    --max-perturbation $CAP --enso-effect-per-k $FF \
    --enso-mode randomized --enso-amp-design deterministic \
    --enso-amp-range -$AMAX $AMAX \
    --output $E/enso8_eval.pkl || { echo "EVAL FAILED"; exit 1; }
fi

echo ""
echo "[$(ts)] ANALYSIS (pre-committed analyze_enso8.py)"
$PY analyze_enso8.py $E/enso8_eval.pkl || { echo "ANALYSIS FAILED"; exit 1; }

echo ""
echo "[$(ts)] DONE -> $E/enso8_eval.pkl + _analysis.pkl"
