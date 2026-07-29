#!/bin/bash
# P2 pre-registered campaign v2 (PREREGISTRATION.md), diya, vLLM stopped w/ trap.
# Changes from v1: forcing cap lowered 0.15 -> 0.09 (fixes G2 overcooling), the
# two ablations forced to random-init via a nonexistent --init-checkpoint (v1
# picked up a leftover default checkpoint and warm-started), and the broken x64
# noise-floor step removed (float32/64 lax.scan carry mismatch — separate fix).
CONTAINER=aeon-ultimate-xs
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
ICS=$E/ics_independent          # reuse the v1 independent ICs (physics unchanged)
S1=$E/stage1_v2
RT=$E/retrain_v2
OL=$E/ablation_openloop_v2
NW=$E/ablation_nowarmstart_v2
DAYS=60
CI=15
CAP=0.09                        # forcing cap (was 0.15; targets -0.1 K, fixes G2)
NONE=/tmp/nonexistent_ckpt.pkl  # forces --allow-random-init for the ablations
PAT=$S1/stage1_optimized_pattern.pkl

step "1/7 Stage-1 pattern (corrected physics, cap=$CAP)"
$PY optimize_mcb_pattern.py --base-carry $BASE --days $DAYS --iters 150 \
  --max-amplitude $CAP --output-dir $S1 || { echo "STAGE1 FAILED"; exit 1; }

step "2/7 Noise floor float32 (cap=$CAP; confirm gate power at new cap)"
$PY run_noise_floor.py --ic-dir $ICS --days $DAYS --control-interval $CI \
  --reps 8 --regen-baseline --num-ics 10 --max-perturbation $CAP --stage1 $PAT \
  --output $E/noise_floor_v2_f32.pkl || echo "NOISEFLOOR FAILED"

step "3/7 Retrain (warm-started from cap=$CAP Stage-1)"
$PY run_stage5_training.py --ic-dir $ICS --days $DAYS --control-interval $CI \
  --epochs 80 --max-perturbation $CAP --warm-start-stage1 $PAT \
  --output-dir $RT || { echo "RETRAIN FAILED"; exit 1; }

step "4/7 Open-loop ablation (time-only, RANDOM init) — feedback vs schedule test"
$PY run_stage5_training.py --ic-dir $ICS --days $DAYS --control-interval $CI \
  --epochs 80 --max-perturbation $CAP --feature-mode time-only \
  --init-checkpoint $NONE --allow-random-init \
  --output-dir $OL || echo "OPENLOOP FAILED"

step "5/7 No-warm-start ablation (RANDOM init) — NN over static?"
$PY run_stage5_training.py --ic-dir $ICS --days $DAYS --control-interval $CI \
  --epochs 80 --max-perturbation $CAP \
  --init-checkpoint $NONE --allow-random-init \
  --output-dir $NW || echo "NOWARMSTART FAILED"

step "6/7 Eval — pre-registered gates (retrain vs cap=$CAP Stage-1 static)"
$PY run_stage5_eval.py --ic-dir $ICS --days $DAYS --control-interval $CI \
  --max-perturbation $CAP --checkpoint $RT/stage5_trained_policy.pkl --stage1 $PAT \
  --output $E/eval_v2.pkl || echo "EVAL FAILED"

step "7/7 Ablation comparison (open-loop & no-warm-start vs retrain, held-out dSST)"
$PY - <<'PYEOF' || echo "ABLATION-COMPARE FAILED"
import pickle, json, numpy as np, jax, jax.numpy as jnp, jax_datetime as jdt
from jcm.mcb import CoupledControllerConfig, CoupledFeatureConfig, MCBPolicyMLP, load_carry
from jcm.mcb.coupled_controller import evaluate_coupled_policy
from jcm.mcb.coupled_train import ocean_mask_from_coupler
from jcm.mcb.gates import improvement_gate
from jcm.mcb.train import load_checkpoint
from run_coupled_training import setup_coupled_model, coupler_workflow
from run_stage5_training import START_DATE, load_ics
E="mcb_experiments_gpu"; DAYS=60; CI=15; CAP=0.09; TGT=-0.1
cpl,coords,terrain,atm=setup_coupled_model(jdt.to_datetime(START_DATE),jdt.to_timedelta(1,"day"),realistic_terrain=True)
wf=coupler_workflow(cpl); tmpl=cpl.initialize(); om=ocean_mask_from_coupler(cpl)
man,tr,ho=load_ics(f"{E}/ics_independent",DAYS,tmpl); held=[(e,c,b) for e,c,b in ho]
def cfg(fc): return CoupledControllerConfig(control_interval_steps=CI,total_steps=DAYS,target_cooling=TGT,feature_config=fc,max_perturbation=CAP,use_checkpointing=True)
def evalpol(ckpt,fc):
    p,_=load_checkpoint(ckpt); pol=MCBPolicyMLP(output_shape=coords.horizontal.nodal_shape,hidden_dims=(256,256),max_perturbation=CAP)
    return [float(evaluate_coupled_policy(coupler=cpl,workflow=wf,policy_fn=pol.apply,policy_params=p,initial_carry=c,baseline_trajectory=b,coords=coords,ocean_mask=om,config=cfg(fc))["metrics"]["final_sst_change"]) for e,c,b in held]
fc13=CoupledFeatureConfig(include_absolute_sst=True); fc1=CoupledFeatureConfig(include_sst=False,include_sst_regions=False,include_heat_flux=False,include_atm_temperature=False,include_precipitation=False,include_time=True,include_absolute_sst=False)
rt=evalpol(f"{E}/retrain_v2/stage5_trained_policy.pkl",fc13)
ol=evalpol(f"{E}/ablation_openloop_v2/stage5_trained_policy.pkl",fc1)
nw=evalpol(f"{E}/ablation_nowarmstart_v2/stage5_trained_policy.pkl",fc13)
err=lambda x:[abs(v-TGT) for v in x]
g_ol=improvement_gate(err(rt),err(ol)); g_nw=improvement_gate(err(rt),err(nw))
print(f"held-out dSST mean: retrain {np.mean(rt):+.4f}  open-loop {np.mean(ol):+.4f}  no-warmstart {np.mean(nw):+.4f}")
print(f"FEEDBACK gate (retrain vs open-loop): {g_ol['verdict']} (improvement {g_ol['improvement']:+.4f} +/- {g_ol['se']:.4f})")
print(f"NN-vs-random gate (retrain vs no-warmstart): {g_nw['verdict']} (improvement {g_nw['improvement']:+.4f} +/- {g_nw['se']:.4f})")
pickle.dump({"retrain":rt,"openloop":ol,"nowarmstart":nw,"feedback_gate":g_ol,"nn_gate":g_nw}, open(f"{E}/ablation_compare_v2.pkl","wb"))
PYEOF

echo ""
echo "[$(ts)] ========== CAMPAIGN v2 COMPLETE =========="
