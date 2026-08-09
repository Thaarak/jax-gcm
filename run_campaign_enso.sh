#!/bin/bash
# ENSO feedback campaign (PREREGISTRATION.md Amendment 6, frozen 2026-08-08),
# diya.
#
# Can a feedback controller cancel the GMST effect of imposed El Nino
# variability while delivering the -0.1 K MCB target? Hidden per-IC
# amplitude A ~ U[0.5, 2.0]; 180-day episodes; registered tail-60 metric
# vs the no-ENSO baseline. Arms: rescaled static, mean-feedforward
# schedule, classical pienso law, imitation-fc14 x3 seeds. No BPTT.
#
# Cost: IC gen ~10 min; gates ~40 min; distillation 3 x ~6 min; final eval
# (20 ICs x k=4 x [baseline + 6 arms] = 560 rollouts at 180 d) ~2.6 h.
# TOTAL ~3.5 h. Gates stop the spend early on failure. Steps are
# idempotent — rerun the script to resume after an interruption.
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
S1PAT180=$E/stage1_pattern_enso180.pkl
ICS=$E/ics_enso
ANCHORS=$E/enso_feature_anchors.json
DAYS=180
CI=15
TAIL=60
CAP=0.09
RESCALE=0.3668   # 0.10 / 0.2726 (authority scoping, frozen in Amendment 6)

step "1/8 Fresh ENSO ICs (seed0 7000; 6 train for gates, 20 held-out)"
if [ -f $ICS/manifest.json ]; then echo "  $ICS exists — skip"; else
  $PY run_generate_ics_independent.py --base-carry $BASE \
    --num-train 6 --num-heldout 20 --decorr-days 30 --horizon $DAYS \
    --seed0 7000 --output-dir $ICS || { echo "IC GEN FAILED"; exit 1; }
fi

step "2/8 Rescale Tier-1 pattern for the 180-d horizon (x $RESCALE)"
if [ -f $S1PAT180 ]; then echo "  $S1PAT180 exists — skip"; else
  $PY - "$S1PAT" "$S1PAT180" "$RESCALE" << 'PYEOF'
import pickle, sys
import numpy as np
src, dst, scale = sys.argv[1], sys.argv[2], float(sys.argv[3])
with open(src, "rb") as f:
    stage1 = pickle.load(f)
out = {"best_pattern": np.asarray(stage1["best_pattern"]) * scale,
       "rescale_from": src, "rescale_factor": scale}
with open(dst, "wb") as f:
    pickle.dump(out, f)
print(f"  {dst}: mean {out['best_pattern'].mean():.5f} "
      f"max {out['best_pattern'].max():.5f}")
PYEOF
  [ $? -ne 0 ] && { echo "RESCALE FAILED"; exit 1; }
fi

step "3/8 Feature-envelope probe (train ICs; Tier-2b lesson: measure)"
if [ -f $ANCHORS ]; then echo "  $ANCHORS exists — skip"; else
  $PY run_feature_envelope.py --ic-dir $ICS --split train --num-ics 3 \
    --days $DAYS --output $ANCHORS || { echo "ENVELOPE FAILED"; exit 1; }
fi

step "4/8 G-cal: static without ENSO must land in [-0.12, -0.08]"
if [ -f $E/enso_gcal.pkl ]; then echo "  gate output exists — skip run"; else
  $PY run_confirmatory_eval.py --ic-dir $ICS --split train \
    --arm static=static=$S1PAT180 \
    --members 2 --days $DAYS --control-interval $CI --tail-days $TAIL \
    --max-perturbation $CAP --enso-mode off \
    --output $E/enso_gcal.pkl || { echo "G-CAL RUN FAILED"; exit 1; }
fi
$PY - << 'PYEOF' || { echo "G-CAL GATE FAILED — fix rescale before spending"; exit 1; }
import pickle
r = pickle.load(open("mcb_experiments_gpu/enso_gcal.pkl", "rb"))
m = float(r["per_ic"]["static"]["dsst_10d"].mean())
ok = -0.12 <= m <= -0.08
print(f"  G-cal: static no-ENSO mean dSST {m:+.4f} K -> "
      f"{'PASS' if ok else 'FAIL'}")
raise SystemExit(0 if ok else 1)
PYEOF

step "5/8 G-manip: static under fixed A=2.0 must miss by > 0.05 K"
if [ -f $E/enso_gmanip.pkl ]; then echo "  gate output exists — skip run"; else
  $PY run_confirmatory_eval.py --ic-dir $ICS --split train \
    --arm static=static=$S1PAT180 \
    --members 2 --days $DAYS --control-interval $CI --tail-days $TAIL \
    --max-perturbation $CAP \
    --enso-mode fixed --enso-amp-range 2.0 2.0 \
    --output $E/enso_gmanip.pkl || { echo "G-MANIP RUN FAILED"; exit 1; }
fi
$PY - << 'PYEOF' || { echo "G-MANIP GATE FAILED — disturbance too weak"; exit 1; }
import pickle
import numpy as np
r = pickle.load(open("mcb_experiments_gpu/enso_gmanip.pkl", "rb"))
miss = float(np.mean(np.abs(r["per_ic"]["static"]["dsst_10d"] - (-0.1))))
ok = miss > 0.05
print(f"  G-manip: static under A=2 mean |miss| {miss:.4f} K -> "
      f"{'PASS' if ok else 'FAIL'} (bound 0.05)")
raise SystemExit(0 if ok else 1)
PYEOF

step "6/8 Distill pienso -> fc14 (3 seeds; Amendment-4 fidelity gate inside)"
for SEED in 62 63 64; do
  OUT=$E/enso_imitation_s$SEED.pkl
  [ -f $OUT ] && { echo "  $OUT exists — skip"; continue; }
  $PY run_pi_imitation.py --stage1 $S1PAT180 --law pi-enso \
    --num-intervals $((DAYS / CI)) --f0-range -0.40 0.30 \
    --anchors-json $ANCHORS --seed $SEED \
    --output $OUT || { echo "DISTILL s$SEED FAILED (fidelity gate)"; exit 1; }
done

step "7/8 G-imit: imitation on train ICs (amp seed 941) <= 1.5x pienso"
if [ -f $E/enso_gimit.pkl ]; then echo "  gate output exists — skip run"; else
  $PY run_confirmatory_eval.py --ic-dir $ICS --split train \
    --arm pienso=pi-enso=$S1PAT180 \
    --arm imitation_s62=fc14=$E/enso_imitation_s62.pkl \
    --comparator pienso \
    --members 2 --days $DAYS --control-interval $CI --tail-days $TAIL \
    --max-perturbation $CAP \
    --enso-mode randomized --enso-amp-range 0.5 2.0 --enso-seed 941 \
    --enso-antithetic \
    --output $E/enso_gimit.pkl || { echo "G-IMIT RUN FAILED"; exit 1; }
fi
$PY - << 'PYEOF' || { echo "G-IMIT GATE FAILED — distillation did not transfer"; exit 1; }
import pickle
import numpy as np
r = pickle.load(open("mcb_experiments_gpu/enso_gimit.pkl", "rb"))
pi = float(np.mean(np.abs(r["per_ic"]["pienso"]["dsst_10d"] - (-0.1))))
im = float(np.mean(np.abs(r["per_ic"]["imitation_s62"]["dsst_10d"] - (-0.1))))
ok = im <= 1.5 * pi
print(f"  G-imit: pienso {pi * 1000:.1f} mK, imitation {im * 1000:.1f} mK "
      f"-> {'PASS' if ok else 'FAIL'} (bound 1.5x)")
raise SystemExit(0 if ok else 1)
PYEOF

step "8/8 FINAL EVAL: 20 held-out ICs, k=4, hidden A ~ U[0.5, 2] (seed 940)"
if [ -f $E/enso_eval.pkl ] && $PY -c "
import pickle; r = pickle.load(open('mcb_experiments_gpu/enso_eval.pkl','rb'))
raise SystemExit(0 if 'gates' in r else 1)" 2>/dev/null; then
  echo "  enso_eval.pkl complete — skip"
else
  $PY run_confirmatory_eval.py --ic-dir $ICS --split heldout \
    --arm static=static=$S1PAT180 \
    --arm ffmean=ff-mean=$S1PAT180 \
    --arm pienso=pi-enso=$S1PAT180 \
    --arm imitation_s62=fc14=$E/enso_imitation_s62.pkl \
    --arm imitation_s63=fc14=$E/enso_imitation_s63.pkl \
    --arm imitation_s64=fc14=$E/enso_imitation_s64.pkl \
    --comparator static \
    --members 4 --days $DAYS --control-interval $CI --tail-days $TAIL \
    --max-perturbation $CAP \
    --enso-mode randomized --enso-amp-range 0.5 2.0 --enso-seed 940 \
    --enso-antithetic \
    --output $E/enso_eval.pkl || { echo "FINAL EVAL FAILED"; exit 1; }
fi

step "ANALYSIS (pre-committed analyze_enso.py)"
$PY analyze_enso.py $E/enso_eval.pkl || { echo "ANALYSIS FAILED"; exit 1; }

echo ""
echo "[$(ts)] CAMPAIGN COMPLETE -> $E/enso_eval.pkl + _analysis.pkl"
