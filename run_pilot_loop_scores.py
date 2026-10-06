"""Score the step-23 closed loop (Part 22.3) against the ramp-6 references.

Run on the GX10 (the references are not in git): writes
mcb_experiments_gpu/pilot_step23/loop/loop_scores.json.
"""
import json
import numpy as np
from run_planner_pilot import ocean_weights, zonal_project, objective
D = "mcb_experiments_gpu/"
g = np.load(D + "test_world_refs_ramp6/exp3_train/grid.npz")
lat = g["latitudes_rad"]
ocean = g["ocean_mask"]
w = ocean_weights(lat, ocean)
out = {}
for i in ("0002", "0202"):
    r = np.load(D + f"test_world_refs_ramp6/exp3_train/ic{i}_references.npz")
    c = np.load(D + f"pilot_step23/loop/ic{i}_plan.fields.npz")
    sl = slice(98, 182)
    tgt = r["normal_sst"][sl].mean(0)
    ctl = c["sst"][sl].mean(0)
    unc = r["warmed_sst"][sl].mean(0)
    res = {}
    for name, f in (("controlled_1member", ctl), ("uncontrolled_5mean", unc)):
        e = (f - tgt) * ocean
        res[name] = {"bias_K": float(np.sum(w * e)),
                     "J_map": float(objective(e, w)),
                     "J_zonal": float(objective(zonal_project(e, ocean), w))}
    # single-member noise of a 60-day mean (first 60 d of the normal members):
    m = r["normal_members_sst"].mean(1)                 # (5, ix, il)
    dev = (m - m.mean(0)) * ocean
    k = 5 / 4                                           # unbiased
    res["noise_1member_vs_mean_60d"] = {
        "J_map": float(k * np.mean([objective(d_, w) for d_ in dev])),
        "J_zonal": float(k * np.mean([objective(zonal_project(d_, ocean), w)
                                       for d_ in dev]))}
    # trajectory of the ocean-mean error, by 14-day block
    ts_c = [float(np.sum(w * ((c["sst"][a:a+14] - r["normal_sst"][a:a+14]).mean(0)) * ocean))
            for a in range(0, 182, 14)]
    ts_u = [float(np.sum(w * ((r["warmed_sst"][a:a+14] - r["normal_sst"][a:a+14]).mean(0)) * ocean))
            for a in range(0, 182, 14)]
    res["bias_by_fortnight_controlled"] = ts_c
    res["bias_by_fortnight_uncontrolled"] = ts_u
    out[i] = res
json.dump(out, open(D + "pilot_step23/loop/loop_scores.json", "w"), indent=1)
print(json.dumps(out, indent=1))
