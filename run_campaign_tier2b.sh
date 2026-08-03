#!/bin/bash
# TIER-2b campaign: PI-imitation initialization
# (PREREGISTRATION.md Amendment 4, frozen 2026-08-02), diya.
#
# Tests the verified Tier-2 diagnosis: the NN failure was TRAINING (chaotic
# BPTT gradients), not information or authority. Distill the PI law into the
# fc13 MLP (noiseless supervised regression, minutes), gate on imitation
# quality, fine-tune with low-LR BPTT, and ask the pre-registered question:
# can a LEARNED policy exceed the hand-designed law (H4: finetuned vs PI),
# by widening deployment on low-eta episodes where the cap blocks PI?
#
# Cost: distill 3 x ~3 min; imitation gate ~20 min; fine-tune 3 x ~3 h
# (30 epochs, lr 1e-3); eval (6 arms x 20 fresh ICs x k=4 = 560 rollouts)
# ~2.5 h. TOTAL ~12 h. Gates stop the spend early on failure.
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
TRAIN_ICS=$E/ics_t2_train        # reuse: train 4000-4009, validation 4010-4015
EVAL_ICS=$E/ics_t2b_eval         # FRESH confirmatory n=20 (seed0 6000)
DAYS=60
CI=15
CAP=0.09
ERANGE="0.6 1.4"

step "0/6 Fresh Tier-2b confirmatory ICs (seed0 6000; the 5000s are spent)"
if [ ! -f $EVAL_ICS/manifest.json ]; then
  $PY run_generate_ics_independent.py --base-carry $BASE \
    --num-train 0 --num-heldout 20 --decorr-days 30 --horizon $DAYS \
    --seed0 6000 --output-dir $EVAL_ICS || { echo "IC GEN FAILED"; exit 1; }
else echo "  $EVAL_ICS exists — reusing"; fi

step "1/6 Distill PI law into fc13 MLP x 3 seeds (fidelity-gated)"
for SEED in 52 53 54; do
  OUT=$E/t2b_imitation_s$SEED.pkl
  [ -f $OUT ] && { echo "  $OUT exists — skip"; continue; }
  $PY run_pi_imitation.py --stage1 $S1PAT --seed $SEED \
    --max-perturbation $CAP --output $OUT \
    || { echo "DISTILL s$SEED FAILED (fidelity gate)"; exit 1; }
done

step "2/6 IMITATION GATE: imitation-only vs pi vs static on validation ICs"
$PY run_confirmatory_eval.py --ic-dir $TRAIN_ICS --split heldout \
  --arm static=static=$S1PAT --arm pi=pi=$S1PAT \
  --arm imitation_s52=fc13=$E/t2b_imitation_s52.pkl \
  --efficacy-mode randomized --efficacy-range $ERANGE --efficacy-seed 921 \
  --efficacy-antithetic \
  --members 2 --days $DAYS --control-interval $CI --tail-days 10 \
  --max-perturbation $CAP --output $E/t2b_imitation_gate.pkl \
  || { echo "IMITATION GATE EVAL FAILED"; exit 1; }
# Amendment 4 gate: imitation-only mean |err| < 15 mK (PI ~10-12, static ~20).
$PY - <<'PYEOF' || { echo "IMITATION GATE NOT PASSED — distillation does not transfer; stopping"; exit 1; }
import pickle, numpy as np
r = pickle.load(open("mcb_experiments_gpu/t2b_imitation_gate.pkl", "rb"))
e = {n: np.abs(np.asarray(c["dsst_10d"]) + 0.1).mean()
     for n, c in r["per_ic"].items()}
print("imitation gate mean|err| (mK): " +
      ", ".join(f"{n} {1000*v:.1f}" for n, v in e.items()) +
      "  [gate: imitation < 15 mK]")
assert e["imitation_s52"] < 0.015
PYEOF

step "3/6 Fine-tune from imitation x 3 seeds (lr 1e-3, tail_dsst, eta-randomized)"
for SEED in 52 53 54; do
  OUT=$E/t2b_finetuned_s$SEED
  [ -f $OUT/stage5_trained_policy.pkl ] && { echo "  $OUT exists — skip"; continue; }
  $PY run_stage5_training.py --ic-dir $TRAIN_ICS --days $DAYS \
    --control-interval $CI --epochs 30 --learning-rate 0.001 \
    --max-perturbation $CAP \
    --init-checkpoint $E/t2b_imitation_s$SEED.pkl \
    --loss-mode tail_dsst --select-on-heldout \
    --efficacy-range $ERANGE --efficacy-seed $((910 + SEED)) --seed $SEED \
    --efficacy-resample-per-epoch --regen-baselines \
    --output-dir $OUT || { echo "FINETUNE s$SEED FAILED"; exit 1; }
done

step "4/6 Confirmatory Tier-2b eval (6 arms, 20 FRESH ICs, k=4, eta seed 930)"
$PY run_confirmatory_eval.py --ic-dir $EVAL_ICS --split heldout \
  --arm static=static=$S1PAT \
  --arm pi=pi=$S1PAT \
  --arm imitation_s52=fc13=$E/t2b_imitation_s52.pkl \
  --arm finetuned_s52=fc13=$E/t2b_finetuned_s52/stage5_trained_policy.pkl \
  --arm finetuned_s53=fc13=$E/t2b_finetuned_s53/stage5_trained_policy.pkl \
  --arm finetuned_s54=fc13=$E/t2b_finetuned_s54/stage5_trained_policy.pkl \
  --comparator pi --members 4 --member-perturb-amp 0.001 \
  --efficacy-mode randomized --efficacy-range $ERANGE --efficacy-seed 930 \
  --efficacy-antithetic \
  --days $DAYS --control-interval $CI --tail-days 10 \
  --max-perturbation $CAP --target-cooling -0.1 \
  --output $E/tier2b_eval.pkl || { echo "TIER2B EVAL FAILED"; exit 1; }

step "5/6 Pre-committed primary analysis (analyze_tier2b.py)"
$PY analyze_tier2b.py $E/tier2b_eval.pkl || echo "ANALYSIS FAILED (data safe in tier2b_eval.pkl)"

step "6/6 done"
echo ""
echo "[$(ts)] ========== TIER-2B CAMPAIGN COMPLETE =========="
echo "Artifacts: $E/{t2b_imitation_s5*.pkl, t2b_imitation_gate.pkl, t2b_finetuned_s5*, tier2b_eval.pkl}"
