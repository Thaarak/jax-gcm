"""Deviation D2 evidence: Experiment 3f's 48 evaluation target extensions vs 3d's references.

Outcome-blind: reads only the normal-climate extensions (exp3f/target_ext) and
3d's evaluation references; no planner run. For every state it reports
  - overlap_k: the planner's own statistic (mean |difference| of the two files'
    14-day mean SST over their last 14 shared days; F4's limit was 0.05 K);
  - noise_k: the same statistic expected for two independent 10-member
    ensembles of the state, from 3d's own members (members 0-4 against 5-9 over
    days 185-199, the 5-day blocks the references store, scaled by
    sqrt(0.2 / 0.4) from 5-against-5 to 10-against-10);
  - signed_k: the signed grid-mean difference (extension - references) over the
    last 14 shared days (a climate offset would show here);
  - max_shared_k and first_day_differing: the largest difference of the daily
    ensemble means over all shared days, and the first day it is non-zero;
  - seeds_equal: whether the member seeds are 3d's;
  - warmed_k, warmed_signed_k: for scale, the same two statistics between 3d's
    warmed and normal references of the state (another climate).
Run in the repository on the GX10: python check_extensions.py > extensions_check.json
"""
import json
import sys

import numpy as np

E = "mcb_experiments_gpu"
states, manifests = [], {}
for b in range(3):
    ext_dir = f"{E}/exp3f/target_ext/exp3d_eval_b{b}"
    ref_dir = f"{E}/test_world_refs_ramp6_3d/exp3d_eval_b{b}"
    me = json.load(open(f"{ext_dir}/references_manifest.json"))
    mr = json.load(open(f"{ref_dir}/references_manifest.json"))
    manifests[f"b{b}"] = {k: {"command": m["command"], "started_utc": m["started_utc"],
                              "finished_utc": m.get("finished_utc"),
                              "land_climatology": m.get("land_climatology")}
                          for k, m in (("extension", me), ("references", mr))}
    seeds = {e["entry"]["index"]: e["member_seeds"] for e in mr["ics"]}
    for e in me["ics"]:
        i = e["entry"]["index"]
        r = np.load(f"{ref_dir}/ic{i}_references.npz")
        own = r["normal_sst"]
        longer = np.load(f"{ext_dir}/ic{i}_references.npz")["normal_sst"]
        n = own.shape[0]
        d = longer[n - 14:n].mean(0) - own[n - 14:n].mean(0)
        mem = r["normal_members_sst"][:, 37:40].mean(1)
        noise = float(np.mean(np.abs(mem[:5].mean(0) - mem[5:].mean(0)))) * np.sqrt(0.2 / 0.4)
        w = r["warmed_sst"][n - 14:n].mean(0) - own[n - 14:n].mean(0)
        daily = np.abs(longer[:n] - own).reshape(n, -1).max(1)
        nonzero = np.nonzero(daily)[0]
        states.append({"index": i, "branch": b,
                       "overlap_k": float(np.mean(np.abs(d))), "noise_k": noise,
                       "signed_k": float(np.mean(d)),
                       "warmed_k": float(np.mean(np.abs(w))),
                       "warmed_signed_k": float(np.mean(w)),
                       "max_shared_k": float(daily.max()),
                       "first_day_differing": int(nonzero[0]) if nonzero.size else None,
                       "seeds_equal": e["member_seeds"] == seeds.get(i),
                       "extension_days": int(longer.shape[0]),
                       "finite": bool(np.isfinite(longer).all())})
col = lambda k: np.array([s[k] for s in states])  # noqa: E731
ov, nz, sg = col("overlap_k"), col("noise_k"), col("signed_k")
q = lambda a: {"min": float(a.min()), "median": float(np.median(a)), "max": float(a.max())}  # noqa: E731
summary = {"states": len(states), "overlap_k": q(ov), "noise_k": q(nz),
           "ratio": q(ov / nz),
           "signed_k": {"mean": float(sg.mean()), "sd": float(sg.std()),
                        "min": float(sg.min()), "max": float(sg.max()),
                        "positive": int((sg > 0).sum())},
           "warmed_k": q(col("warmed_k")), "warmed_signed_k": q(col("warmed_signed_k")),
           "max_shared_k": float(col("max_shared_k").max()),
           "first_day_differing": q(col("first_day_differing")),
           "all_seeds_equal": bool(col("seeds_equal").all()),
           "all_finite": bool(col("finite").all()),
           "states_over_0.05_k": int((ov > 0.05).sum()),
           "states_over_0.20_k": int((ov > 0.20).sum()),
           "states_signed_over_0.05_k": int((np.abs(sg) > 0.05).sum())}
json.dump({"summary": summary, "manifests": manifests, "states": states}, sys.stdout, indent=1)
print(json.dumps(summary, indent=1), file=sys.stderr)
