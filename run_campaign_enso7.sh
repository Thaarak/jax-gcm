#!/bin/bash
# CORRECTED ENSO campaign (PREREGISTRATION.md Amendment 7, frozen 2026-08-09),
# diya.
#
# Fixes the four Amendment-6 design errors the audit found
# (MCB_META_AUDIT.md Addendum 5):
#   1. ZERO-MEAN disturbance (El Nino AND La Nina) so no constant can absorb
#      it — an ENSO-blind retuned gain tied the feedback arms last time.
#   2. Plant constants measured on the REGISTERED metric (ocean dSST), not on
#      atmospheric GMST, and from several ICs rather than one.
#   3. Control-law reference matched to the tail-averaged scoring window, so a
#      perfect controller can actually score the target.
#   4. PRIMARY endpoint = disturbance SENSITIVITY (slope of error on hidden
#      amplitude), which no constant rescaling can move, with the LOO-tuned
#      blind-gain arm as a registered control every claim must beat.
#
# Cost: ICs ~10 min; plant calibration ~35 min; gates ~30 min; distillation
# 3 x ~6 min; final eval (20 ICs x k=4 x [baseline + 5 arms]) ~2.2 h.
# TOTAL ~4 h. Steps are idempotent — rerun to resume.
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
S1PAT=$E/stage1_v2/stage1_optimized_pattern.pkl
S1PAT7=$E/stage1_pattern_enso7.pkl
ICS=$E/ics_enso7
PLANT=$E/enso7_plant.json
ANCHORS=$E/enso7_feature_anchors.json
DAYS=180
CI=15
TAIL=60
CAP=0.09

step "1/8 Fresh ICs for Amendment 7 (seed0 8000; 6 train + 20 held-out)"
if [ -f $ICS/manifest.json ]; then echo "  $ICS exists — skip"; else
  $PY run_generate_ics_independent.py --base-carry $BASE \
    --num-train 6 --num-heldout 20 --decorr-days 30 --horizon $DAYS \
    --seed0 8000 --output-dir $ICS || { echo "IC GEN FAILED"; exit 1; }
fi

step "2/8 Plant calibration IN METRIC SPACE (train ICs only)"
if [ -f $PLANT ]; then echo "  $PLANT exists — skip"; else
  $PY run_calibrate_enso_plant.py --ic-dir $ICS --split train --num-ics 6 \
    --stage1 $S1PAT --days $DAYS --tail-days $TAIL --members 4 \
    --probe-amp 1.4 --output $PLANT || { echo "CALIBRATION FAILED"; exit 1; }
fi
# Amplitude range is sized to the measured actuator floor: at |A| = AMP the
# cold side is exactly compensable by spraying nothing.
AMP=$($PY -c "import json;print(f\"{min(json.load(open('$PLANT'))['amplitude_floor_K'],2.0):.3f}\")")
FF=$($PY -c "import json;print(f\"{json.load(open('$PLANT'))['enso_effect_per_K']:.5f}\")")
RESCALE=$($PY -c "import json;print(f\"{json.load(open('$PLANT'))['rescale']:.5f}\")")
echo "  measured: enso_effect_per_K=$FF  rescale=$RESCALE  amplitude range +/-$AMP"

step "3/8 Rescale the pattern with the METRIC-SPACE authority (x $RESCALE)"
if [ -f $S1PAT7 ]; then echo "  $S1PAT7 exists — skip"; else
  $PY - "$S1PAT" "$S1PAT7" "$RESCALE" << 'PYEOF'
import pickle, sys
import numpy as np
src, dst, scale = sys.argv[1], sys.argv[2], float(sys.argv[3])
with open(src, "rb") as f:
    s1 = pickle.load(f)
out = {"best_pattern": np.asarray(s1["best_pattern"]) * scale,
       "rescale_from": src, "rescale_factor": scale}
with open(dst, "wb") as f:
    pickle.dump(out, f)
print(f"  {dst}: mean {out['best_pattern'].mean():.5f} "
      f"max {out['best_pattern'].max():.5f}")
PYEOF
  [ $? -ne 0 ] && { echo "RESCALE FAILED"; exit 1; }
fi

step "4/8 G-cal: static (rescaled) with ENSO OFF must land in [-0.12,-0.08]"
if [ -f $E/enso7_gcal.pkl ]; then echo "  exists — skip run"; else
  $PY run_confirmatory_eval.py --ic-dir $ICS --split train \
    --arm static=static=$S1PAT7 \
    --members 4 --days $DAYS --control-interval $CI --tail-days $TAIL \
    --max-perturbation $CAP --enso-mode off \
    --output $E/enso7_gcal.pkl || { echo "G-CAL RUN FAILED"; exit 1; }
fi
$PY - << 'PYEOF' || { echo "G-CAL FAILED — recalibrate before spending"; exit 1; }
import pickle
r = pickle.load(open("mcb_experiments_gpu/enso7_gcal.pkl", "rb"))
m = float(r["per_ic"]["static"]["dsst_10d"].mean())
ok = -0.12 <= m <= -0.08
print(f"  G-cal: static no-ENSO mean dSST {m:+.4f} K -> "
      f"{'PASS' if ok else 'FAIL'}")
raise SystemExit(0 if ok else 1)
PYEOF

step "5/8 Feature-envelope probe (train ICs)"
if [ -f $ANCHORS ]; then echo "  $ANCHORS exists — skip"; else
  $PY run_feature_envelope.py --ic-dir $ICS --split train --num-ics 3 \
    --days $DAYS --output $ANCHORS || { echo "ENVELOPE FAILED"; exit 1; }
fi

step "6/8 Distill the corrected pienso law -> fc14 (3 seeds)"
for SEED in 72 73 74; do
  OUT=$E/enso7_imitation_s$SEED.pkl
  [ -f $OUT ] && { echo "  $OUT exists — skip"; continue; }
  $PY run_pi_imitation.py --stage1 $S1PAT7 --law pi-enso \
    --num-intervals $((DAYS / CI)) --f0-range -0.40 0.30 \
    --nino-range -$AMP $AMP \
    --enso-effect-per-k $FF --reference-days $DAYS --reference-tail $TAIL \
    --anchors-json $ANCHORS --seed $SEED \
    --output $OUT || { echo "DISTILL s$SEED FAILED"; exit 1; }
done

step "7/8 G-arm: pienso itself must land in band on train ICs (amp seed 951)"
# Amendment-6 lesson: gate the ARM UNDER TEST, not only the comparator.
if [ -f $E/enso7_garm.pkl ]; then echo "  exists — skip run"; else
  $PY run_confirmatory_eval.py --ic-dir $ICS --split train \
    --arm pienso=pi-enso=$S1PAT7 \
    --arm imitation_s72=fc14=$E/enso7_imitation_s72.pkl \
    --comparator pienso \
    --members 4 --days $DAYS --control-interval $CI --tail-days $TAIL \
    --max-perturbation $CAP --enso-effect-per-k $FF \
    --enso-mode randomized --enso-amp-range -$AMP $AMP --enso-seed 951 \
    --enso-antithetic \
    --output $E/enso7_garm.pkl || { echo "G-ARM RUN FAILED"; exit 1; }
fi
$PY - << 'PYEOF' || { echo "G-ARM FAILED — the law cannot reach target; fix before eval"; exit 1; }
import pickle
import numpy as np
r = pickle.load(open("mcb_experiments_gpu/enso7_garm.pkl", "rb"))
m = float(r["per_ic"]["pienso"]["dsst_10d"].mean())
i = float(r["per_ic"]["imitation_s72"]["dsst_10d"].mean())
ok = -0.12 <= m <= -0.08
print(f"  G-arm: pienso mean dSST {m:+.4f} K (imitation {i:+.4f}) -> "
      f"{'PASS' if ok else 'FAIL'}")
raise SystemExit(0 if ok else 1)
PYEOF

step "8/8 FINAL EVAL: 20 held-out ICs, k=4, hidden A ~ U[-$AMP,+$AMP] (seed 950)"
if [ -f $E/enso7_eval.pkl ] && $PY -c "
import pickle; r=pickle.load(open('mcb_experiments_gpu/enso7_eval.pkl','rb'))
raise SystemExit(0 if 'gates' in r else 1)" 2>/dev/null; then
  echo "  enso7_eval.pkl complete — skip"
else
  $PY run_confirmatory_eval.py --ic-dir $ICS --split heldout \
    --arm static=static=$S1PAT7 \
    --arm pienso=pi-enso=$S1PAT7 \
    --arm imitation_s72=fc14=$E/enso7_imitation_s72.pkl \
    --arm imitation_s73=fc14=$E/enso7_imitation_s73.pkl \
    --arm imitation_s74=fc14=$E/enso7_imitation_s74.pkl \
    --comparator static \
    --members 4 --days $DAYS --control-interval $CI --tail-days $TAIL \
    --max-perturbation $CAP --enso-effect-per-k $FF \
    --enso-mode randomized --enso-amp-range -$AMP $AMP --enso-seed 950 \
    --enso-antithetic \
    --output $E/enso7_eval.pkl || { echo "FINAL EVAL FAILED"; exit 1; }
fi

step "ANALYSIS (pre-committed analyze_enso7.py — slope is PRIMARY)"
$PY analyze_enso7.py $E/enso7_eval.pkl || { echo "ANALYSIS FAILED"; exit 1; }

echo ""
echo "[$(ts)] CAMPAIGN COMPLETE -> $E/enso7_eval.pkl + _analysis.pkl"
