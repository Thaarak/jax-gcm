#!/bin/bash
# P2 pre-registered campaign v3 (PREREGISTRATION.md), diya.
#
# v3 tests whether the v2 "feedback doesn't beat static" negative was a
# TRAINING ARTIFACT (adversarial review: the summed objective is <0.2%
# gate-relevant and rewards overcooling; model selection was on train loss;
# per-IC gradients conflict so the ensemble mean cancels). Fixes:
#   * --loss-mode terminal_dsst : train on exactly what the gate scores
#     (final-step global-mean dSST error), not the transient/uniformity sum.
#   * --select-on-heldout        : gate model selection + early-stop on the
#     mean held-out loss (evaluated every epoch), not train loss (P0.3).
#   * gradient-coherence probe    : dispositive artifact-vs-real measurement.
# Physics unchanged from v2 -> reuse the v2 independent ICs and cap 0.09.
CONTAINER=aeon-vllm             # vLLM container (renamed from aeon-ultimate-xs)
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
ICS=$E/ics_independent           # reuse v2 independent ICs (physics unchanged)
S1=$E/stage1_v2                  # cap-0.09 static pattern = warm-start + comparator
PAT=$S1/stage1_optimized_pattern.pkl
RT=$E/retrain_v3
OL=$E/ablation_openloop_v3
NW=$E/ablation_nowarmstart_v3
DAYS=60
CI=15
CAP=0.09
EPOCHS=60
NONE=/tmp/nonexistent_ckpt.pkl   # forces --allow-random-init for ablations

step "1/6 Gradient-coherence + headroom probe (artifact vs real)"
$PY run_gradient_probe.py --ic-dir $ICS --warm-start-stage1 $PAT \
  --days $DAYS --control-interval $CI --max-perturbation $CAP \
  --output $E/gradient_probe_v3.pkl || echo "PROBE FAILED"

step "2/6 Retrain (gate-aligned terminal_dsst, held-out-gated, warm-started)"
$PY run_stage5_training.py --ic-dir $ICS --days $DAYS --control-interval $CI \
  --epochs $EPOCHS --max-perturbation $CAP --warm-start-stage1 $PAT \
  --loss-mode terminal_dsst --select-on-heldout \
  --output-dir $RT || { echo "RETRAIN FAILED"; exit 1; }

step "3/6 Open-loop ablation (time-only, terminal_dsst, held-out-gated, RANDOM init)"
$PY run_stage5_training.py --ic-dir $ICS --days $DAYS --control-interval $CI \
  --epochs $EPOCHS --max-perturbation $CAP --feature-mode time-only \
  --loss-mode terminal_dsst --select-on-heldout \
  --init-checkpoint $NONE --allow-random-init \
  --output-dir $OL || echo "OPENLOOP FAILED"

step "4/6 No-warm-start ablation (terminal_dsst, held-out-gated, RANDOM init)"
$PY run_stage5_training.py --ic-dir $ICS --days $DAYS --control-interval $CI \
  --epochs $EPOCHS --max-perturbation $CAP \
  --loss-mode terminal_dsst --select-on-heldout \
  --init-checkpoint $NONE --allow-random-init \
  --output-dir $NW || echo "NOWARMSTART FAILED"

step "5/6 Eval — pre-registered gates (retrain vs cap-0.09 static)"
$PY run_stage5_eval.py --ic-dir $ICS --days $DAYS --control-interval $CI \
  --max-perturbation $CAP --checkpoint $RT/stage5_trained_policy.pkl \
  --stage1 $PAT --output $E/eval_v3.pkl || echo "EVAL FAILED"

step "6/6 Ablation comparison (open-loop & no-warm-start vs retrain, held-out dSST)"
$PY - <<'PYEOF' || echo "ABLATION-COMPARE FAILED"
import pickle, numpy as np, jax_datetime as jdt
from jcm.mcb import CoupledControllerConfig, CoupledFeatureConfig, MCBPolicyMLP
from jcm.mcb.coupled_controller import evaluate_coupled_policy
from jcm.mcb.coupled_train import ocean_mask_from_coupler
from jcm.mcb.gates import improvement_gate
from jcm.mcb.train import load_checkpoint
from run_coupled_training import setup_coupled_model, coupler_workflow
from run_stage5_training import START_DATE, load_ics
E="mcb_experiments_gpu"; DAYS=60; CI=15; CAP=0.09; TGT=-0.1
cpl,coords,terrain,atm=setup_coupled_model(jdt.to_datetime(START_DATE),jdt.to_timedelta(1,"day"),realistic_terrain=True)
wf=coupler_workflow(cpl); tmpl=cpl.initialize(); om=ocean_mask_from_coupler(cpl)
_,tr,ho=load_ics(f"{E}/ics_independent",DAYS,tmpl); held=[(e,c,b) for e,c,b in ho]
def cfg(fc): return CoupledControllerConfig(control_interval_steps=CI,total_steps=DAYS,target_cooling=TGT,feature_config=fc,max_perturbation=CAP,use_checkpointing=True)
def evalpol(ckpt,fc):
    p,_=load_checkpoint(ckpt); pol=MCBPolicyMLP(output_shape=coords.horizontal.nodal_shape,hidden_dims=(256,256),max_perturbation=CAP)
    return [float(evaluate_coupled_policy(coupler=cpl,workflow=wf,policy_fn=pol.apply,policy_params=p,initial_carry=c,baseline_trajectory=b,coords=coords,ocean_mask=om,config=cfg(fc))["metrics"]["final_sst_change"]) for e,c,b in held]
fc13=CoupledFeatureConfig(include_absolute_sst=True)
fc1=CoupledFeatureConfig(include_sst=False,include_sst_regions=False,include_heat_flux=False,include_atm_temperature=False,include_precipitation=False,include_time=True,include_absolute_sst=False)
rt=evalpol(f"{E}/retrain_v3/stage5_trained_policy.pkl",fc13)
ol=evalpol(f"{E}/ablation_openloop_v3/stage5_trained_policy.pkl",fc1)
nw=evalpol(f"{E}/ablation_nowarmstart_v3/stage5_trained_policy.pkl",fc13)
err=lambda x:[abs(v-TGT) for v in x]
g_ol=improvement_gate(err(rt),err(ol)); g_nw=improvement_gate(err(rt),err(nw))
print(f"held-out dSST mean: retrain {np.mean(rt):+.4f}  open-loop {np.mean(ol):+.4f}  no-warmstart {np.mean(nw):+.4f}")
print(f"FEEDBACK gate (retrain vs open-loop): {g_ol['verdict']} (improvement {g_ol['improvement']:+.4f} +/- {g_ol['se']:.4f})")
print(f"NN-vs-random gate (retrain vs no-warmstart): {g_nw['verdict']} (improvement {g_nw['improvement']:+.4f} +/- {g_nw['se']:.4f})")
pickle.dump({"retrain":rt,"openloop":ol,"nowarmstart":nw,"feedback_gate":g_ol,"nn_gate":g_nw}, open(f"{E}/ablation_compare_v3.pkl","wb"))
PYEOF

echo ""
echo "[$(ts)] ========== CAMPAIGN v3 COMPLETE =========="
