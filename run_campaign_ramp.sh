#!/bin/bash
# RAMP campaign — does closed-loop rejection survive a disturbance that never
# stops growing? (PREREGISTRATION.md "Amendment 8", frozen 2026-08-12.)
#
# WHY THIS IS NOT THE CO2 EXPERIMENT. The transient-CO2 channel was enabled,
# verified live, and then REJECTED on a structural ground: `increase_co2` lives
# in the Parameters closure on SpeedyPhysics, so it is present in BOTH the arm
# rollout and its paired no-MCB baseline (run_confirmatory_eval.py:563 hands the
# same step_fn to compute_baseline_trajectory). The registered metric is a
# paired difference and the controller's inputs are paired anomalies, so a CO2
# trend cancels to EXACTLY zero in the score and in the controller's only input.
# The experiment would have been guaranteed to measure nothing — the Tier-1
# "null baked in" failure. (Secondarily, band 1 is already near-opaque at the
# reference absorptivity, so the forcing saturates and cannot be amplified to
# compensate for a short horizon.)
#
# The scientific question survives, and sharpens. The existing ENSO disturbance
# is NOT an oscillation: EnsoConfig.period_days defaults to 0, so A(t) ramps
# over 30 days and then HOLDS for 150. The 85-89% rejection result is already
# persistent-STEP rejection. The genuine increment is step (type-0) -> a
# MONOTONICALLY GROWING RAMP (type-1), which classical control says is the
# case where proportional action leaves a steady-state error. The existing
# pacemaker delivers it with one flag: --enso-ramp-days = the full horizon.
# It is applied to ARM rollouts only, never the baseline, so unlike CO2 the
# disturbance is visible in both the score and the controller's input.
#
# Cost: scoping 0.3 + ICs 0.4 + calibration 0.5 + eval 5.5 = ~6.7 GPU-h.
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
PAT=$E/stage1_pattern_enso7.pkl
CKPT=$E/enso7_imitation_s72.pkl
ICS=$E/ics_ramp
PLANT=$E/ramp_plant.json
DAYS=180
RAMP=180            # the one change: ramp across the WHOLE episode, never holds
CI=15
TAIL=60
CAP=0.09
PROBE=3.5           # ramp halves the tail-window disturbance, so probe larger

for f in "$BASE" "$PAT" "$CKPT"; do
  [ -f "$f" ] || { echo "MISSING PREREQUISITE: $f"; exit 1; }
done

step "1/4 Ramp scoping: disturbance shape + 180-day chaos floor at k=6"
if [ -f $E/ramp_scoping.pkl ]; then echo "  exists — skip"; else
  $PY run_enso_scoping.py --base-carry $BASE \
    --days $DAYS --ramp-days $RAMP --period-days 0 --tail-days $TAIL \
    --amplitude $PROBE --members 6 --skip-lanina \
    --output $E/ramp_scoping.pkl || { echo "SCOPING FAILED"; exit 1; }
fi

step "2/4 Fresh ICs (seed0 10000; 8 train + 32 held-out)"
if [ -f $ICS/manifest.json ]; then echo "  exists — skip"; else
  $PY run_generate_ics_independent.py --base-carry $BASE \
    --num-train 8 --num-heldout 32 --decorr-days 30 --horizon $DAYS \
    --seed0 10000 --output-dir $ICS || { echo "IC GEN FAILED"; exit 1; }
fi

step "3/4 Plant calibration UNDER THE RAMP (train ICs only)"
if [ -f $PLANT ]; then echo "  exists — skip"; else
  $PY run_calibrate_enso_plant.py --ic-dir $ICS --split train --num-ics 6 \
    --stage1 $PAT --days $DAYS --tail-days $TAIL --members 4 \
    --enso-ramp-days $RAMP --probe-amp $PROBE \
    --output $PLANT || { echo "CALIBRATION FAILED"; exit 1; }
fi

# Two derived constants. The feedforward term reads the INSTANTANEOUS index,
# but under a full-episode ramp the tail-window index is f*A with
# f = mean(t/H over the tail) = 0.8361. Feeding the raw slope would under-dose
# feedforward by 16% and MANUFACTURE an H10 null — i.e. confirm the previous
# headline by detuning the arm under test. Correct by 1/f, and carry an FF
# ladder (p1, p15) so the conclusion cannot hinge on this constant.
FF=$($PY -c "import json;print(f\"{json.load(open('$PLANT'))['enso_effect_per_K']/0.8361:.5f}\")")
AMAX=$($PY -c "import json;print(f\"{json.load(open('$PLANT'))['amplitude_floor_K']:.3f}\")")
echo "  measured s_ramp -> --enso-effect-per-k $FF (tail-corrected); A_max $AMAX K"

$PY - << PYEOF || { echo "FREEZE GATE FAILED — reconsider before spending"; exit 1; }
import json
p = json.load(open("$PLANT"))
a = p["amplitude_floor_K"]
ok = 2.5 <= a <= 5.0
print(f"  freeze gate: actuator floor A_max = {a:.2f} K -> "
      f"{'PASS' if ok else 'FAIL'} (must be 2.5-5.0 K; above 5 K the "
      f"disturbance is no longer ENSO-shaped and the framing must change)")
raise SystemExit(0 if ok else 1)
PYEOF

step "4/4 EVAL: 32 held-out ICs, k=4, GROWING ramp, deterministic amplitudes"
if [ -f $E/ramp_eval.pkl ] && $PY -c "
import pickle; r=pickle.load(open('mcb_experiments_gpu/ramp_eval.pkl','rb'))
raise SystemExit(0 if 'gates' in r else 1)" 2>/dev/null; then
  echo "  ramp_eval.pkl complete — skip"
else
  $PY run_confirmatory_eval.py --ic-dir $ICS --split heldout \
    --arm static=static=$PAT \
    --arm b6=pi-enso@0@6=$PAT \
    --arm b12=pi-enso@0@12=$PAT \
    --arm p1=pi-enso@1@1=$PAT \
    --arm p15=pi-enso@1.5@1=$PAT \
    --arm imitation_s72=fc14=$CKPT \
    --comparator static \
    --members 4 --days $DAYS --control-interval $CI --tail-days $TAIL \
    --max-perturbation $CAP --enso-effect-per-k $FF \
    --enso-mode randomized --enso-amp-design deterministic \
    --enso-ramp-days $RAMP --enso-amp-range -$AMAX $AMAX \
    --output $E/ramp_eval.pkl || { echo "EVAL FAILED"; exit 1; }
fi

step "ANALYSIS (pre-committed analyze_enso8.py)"
$PY analyze_enso8.py $E/ramp_eval.pkl || { echo "ANALYSIS FAILED"; exit 1; }

echo ""
echo "[$(ts)] CAMPAIGN COMPLETE -> $E/ramp_eval.pkl + _analysis.pkl"
