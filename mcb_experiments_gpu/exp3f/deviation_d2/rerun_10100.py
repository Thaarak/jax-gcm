"""Deviation D2 evidence: can 3d's evaluation references be reproduced now?

State 10100 (branch 0) was run again on the GX10 on 2026-10-26 (22:13-22:17 PDT,
both at once, while four learn14 runs shared the GPU), with 2 of 3d's 10 members
and 3d's member blocks (5 days), into /tmp/exp3f_extdiag (kept afterwards in
~/workspace/exp3f_extdiag):
  T1: 3d's own 200-day program; T2: the extensions' 280-day program.
Members run one at a time (jcm.mcb.test_world.reference_ensemble), so a member's
trajectory does not depend on --members, --member-block-days or --no-warmed.
Compares each member's 5-day block means with 3d's stored ones and T1 with T2.
Outcome-blind: references only. Run in the repository on the GX10:
python rerun_10100.py > rerun_10100.json
"""
import json
import os
import sys

import numpy as np

E = "mcb_experiments_gpu"
D = os.path.expanduser("~/workspace/exp3f_extdiag")
stored = np.load(f"{E}/test_world_refs_ramp6_3d/exp3d_eval_b0/ic10100_references.npz")["normal_members_sst"]
t1 = np.load(f"{D}/T1_200d/ic10100_references.npz")["normal_members_sst"]
t2 = np.load(f"{D}/T2_280d/ic10100_references.npz")["normal_members_sst"][:, :40]
blocks = [0, 1, 2, 3, 5, 10, 20, 39]


def curve(a, b):
    d = np.abs(a - b).reshape(a.shape[0], -1).max(1)
    return {f"block{k}_days{5 * k}-{5 * k + 4}": float(d[k]) for k in blocks}


out = {"members": {}, "runs": {}}
for m in range(2):
    out["members"][str(m)] = {
        "note": "member 0 is the initial state itself (no perturbation)" if m == 0
        else "member 1: seed 1072701, as in 3d",
        "stored_vs_T1_200d": curve(stored[m], t1[m]),
        "stored_vs_T2_280d": curve(stored[m], t2[m]),
        "T1_vs_T2": curve(t1[m], t2[m])}
for name in ("T1_200d", "T2_280d"):
    man = json.load(open(f"{D}/{name}/references_manifest.json"))
    out["runs"][name] = {k: man.get(k) for k in ("command", "started_utc", "finished_utc",
                                                 "land_climatology")}
    out["runs"][name]["member_seeds"] = man["ics"][0]["member_seeds"]
json.dump(out, sys.stdout, indent=1)
for m, c in out["members"].items():
    print(f"member {m}: stored vs T1, first block {c['stored_vs_T1_200d']['block0_days0-4']:.3g} K, "
          f"last {c['stored_vs_T1_200d']['block39_days195-199']:.3g} K", file=sys.stderr)
