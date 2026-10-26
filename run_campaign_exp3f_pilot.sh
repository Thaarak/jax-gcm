#!/bin/bash
# Experiment 3f's pilot (Amendment 9 revision 5, run BEFORE its freeze, on
# training states only). Every 3f arm (exp3f_common.sh) on four of 3d's
# branch-1 training states (macro 100, 108, 116, 124), 3 members, 13 segments
# of 14 days, against 3d's training references, with normal-only target
# extensions to day 280 built first for all 16 branch-1 training states.
#
# What it decides, by rules fixed before it ran (analyze_exp3f_pilot.py):
#   * feasibility: every run finishes with finite scores and a target
#     extension within 0.05 K of the references where they overlap;
#   * the BUDGET RULE: arms are dropped from the end of learn14,
#     learn112_mpc, learn56_mpc, learn112_hold until the projected
#     evaluation fits in 60 hours (learn14 and learn112_mpc always stay).
# Its scores are recorded for feasibility only; no setting is chosen from them.
# A failed run is re-run once. No evaluation state is touched.
#
# SMOKE=1: one training state (macro 100, branch 1), 2 segments, 1 member,
# into /tmp/exp3f_pilot_smoke.
cd ~/workspace/jax-gcm || exit 1
source exp3f_common.sh
$PY -c "from jcm.mcb.land_climatology import land_climatology_mode as m; assert m() == 'monthly', m()" \
  || { echo "NOT THE FIXED LAND MODEL"; exit 1; }
gpu_visible || { echo "NO VISIBLE GPU"; exit 1; }
ICS=$E/ics_macro3d/exp3d_train
REFS_ROOT=$E/test_world_refs_ramp6_3d
REF_SET=exp3d_train
EXT_ROOT=$E/exp3f/target_ext
OUT=$E/exp3f/pilot
SEGMENTS=13; MEMBERS=3; SCORE_START=98
PARALLEL=${PARALLEL:-4}
ARMS="learn14 learn112_mpc learn56_mpc learn112_hold"
STATES="1 10001
9 10801
17 11601
25 12401"
if [ "${SMOKE:-0}" = 1 ]; then
  OUT=/tmp/exp3f_pilot_smoke; SEGMENTS=2; MEMBERS=1; SCORE_START=0
  STATES="1 10001"; rm -rf $OUT
fi
export ICS REFS_ROOT REF_SET EXT_ROOT OUT SEGMENTS MEMBERS SCORE_START
mkdir -p $OUT/logs
echo "[$(ts)] target extensions: 16 branch-1 training states x 5 members x $EXT_DAYS d (normal only)"
build_extension $ICS 1 5 $EXT_ROOT/${REF_SET}_b1 > $OUT/logs/target_ext_b1.log 2>&1 \
  || { echo "TARGET EXTENSION FAILED (see logs/target_ext_b1.log)"; exit 1; }
echo "[$(ts)] pilot: $ARMS; $MEMBERS members, $SEGMENTS segments, $PARALLEL at a time"
t0=$(date +%s)
for pass in 1 2; do       # the second pass re-runs a failed run once
  for arm in $ARMS; do echo "$STATES" | sed "s/^/$arm /"; done \
    | xargs -P $PARALLEL -L 1 bash -c 'job $0 $1 $2 1'
done
wall=$(( $(date +%s) - t0 ))
n_states=$(echo "$STATES" | wc -l | tr -d ' ')
JAX_PLATFORMS=cpu $PY analyze_exp3f_pilot.py --pilot-dir $OUT --wall-seconds $wall \
  --parallel $PARALLEL --states $n_states --members $MEMBERS \
  $([ "${SMOKE:-0}" = 1 ] || echo "--design-out $E/exp3f/registered_design.json") \
  || echo "PILOT SUMMARY FAILED"
echo "[$(ts)] ========== EXP3F PILOT COMPLETE =========="
