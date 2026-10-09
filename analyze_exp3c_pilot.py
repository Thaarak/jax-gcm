#!/usr/bin/env python
"""The Experiment 3c pilot (MCB_PROJECT_REPORT.md Part 28).

Experiment 3c repeats 3b with planners that see the ocean but not the
weather (``run_test_world.py plan --sensing ocean``). The pilot runs on 3b's
pilot states (``exp3b_train`` branch 1, 3 members) and reuses 3b's pilot
runs of every arm that never used the weather (PI, the adaptive law, the
fixed design, no brightening) and of 3b's exact-sensing planners.

Its rules were written before it ran:

* *The learner's noise level* is 3b's rule applied to the ocean-only oracle
  (``analyze_exp3b_pilot.py noise``), run by the campaign script.
* *Members per evaluation run* are fixed at 6, as in 3b, so that 3b's
  evaluation runs of the arms that never used the weather can be reused
  unchanged. The pilot reports the minimum detectable effect of H1 at 6
  members (3b's power method) for the record; it chooses nothing.

This script summarizes the pilot: arm means, the descriptive pilot effects
and that power. Writes ``summary.json`` next to the 3c pilot runs.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from analyze_exp3b_pilot import (
    TAU_MACRO,
    fit_noise,
    minimum_detectable_effect,
    profile_j,
)
from analyze_experiment3a import compare
from analyze_experiment3b import load_runs
from jcm.mcb.scores import area_weights

ARMS_3C = ("plan_oracle", "plan_learn", "plan_naive")       # ocean-only
ARMS_3B = ("uncontrolled", "fixed", "pi", "adaptive", "plan_naive",
           "plan_oracle", "plan_learn")                     # reused
MEMBERS = 6
WINDOW = (98, 182)
# (name, arm A, arm B); "_rs" marks 3c's ocean-only planners.
EFFECTS = (("H1 learn_rs vs pi", "plan_learn_rs", "pi"),
           ("H2 learn_rs vs naive_rs", "plan_learn_rs", "plan_naive_rs"),
           ("H3 learn_rs vs learn (exact)", "plan_learn_rs", "plan_learn"),
           ("naive_rs vs pi", "plan_naive_rs", "pi"),
           ("oracle_rs vs naive_rs", "plan_oracle_rs", "plan_naive_rs"),
           ("learn_rs vs oracle_rs", "plan_learn_rs", "plan_oracle_rs"))


def difference_variance_two(dir_a, arm_a, dir_b, arm_b, refs_dir):
    """Variance over states of ``J_A - J_B`` with 1..M matched members.

    As ``analyze_exp3b_pilot.difference_variance``, with the two arms'
    runs in different folders.
    """
    with np.load(Path(refs_dir) / "grid.npz") as g:
        lats, ocean = g["latitudes_rad"], np.asarray(g["ocean_mask"], float)
    lat_w = np.asarray(area_weights(lats, ocean), float).sum(axis=0)
    counts = np.maximum(ocean.sum(axis=0), 1.0)
    per_m = {}
    for fa in sorted((Path(dir_a) / arm_a).glob("ic*.fields.npz")):
        idx = fa.name.split(".")[0][2:]
        fb = Path(dir_b) / arm_b / f"ic{idx}.fields.npz"
        if not fb.exists():
            continue
        with np.load(Path(refs_dir) / f"ic{idx}_references.npz") as r:
            s, e = WINDOW
            target = ((np.asarray(r["normal_sst"][s:e], float).mean(0)
                       * ocean).sum(axis=0) / counts)
        with np.load(fa) as za, np.load(fb) as zb:
            pa, pb = za["member_zonal_sst"], zb["member_zonal_sst"]
        n = min(len(pa), len(pb))
        for m in range(1, n + 1):
            for subset in itertools.combinations(range(n), m):
                d = (profile_j(pa, target, lat_w, subset)
                     - profile_j(pb, target, lat_w, subset))
                per_m.setdefault(m, {}).setdefault(subset, []).append(d)
    return {m: float(np.mean([np.var(v, ddof=1) for v in subsets.values()]))
            for m, subsets in per_m.items()}


def merge_arms(runs_3c, runs_3b):
    """One table of arms: 3c's planners get the suffix ``_rs``."""
    merged = {f"{arm}_rs": v for arm, v in runs_3c.items() if v}
    merged.update({arm: v for arm, v in runs_3b.items() if v})
    return merged


def summarize(pilot_dir, pilot3b_dir, refs_dir):
    """Arm means, descriptive pilot effects, learning curve and power."""
    per_arm = merge_arms(load_runs(pilot_dir, [refs_dir], arms=ARMS_3C),
                         load_runs(pilot3b_dir, [refs_dir], arms=ARMS_3B))
    states = sorted(set.intersection(*(set(v) for v in per_arm.values())))
    macros = [i // 100 for i in states]

    def col(arm, key="J_zonal"):
        return np.array([per_arm[arm][i][key] for i in states])

    means = {arm: {k: float(np.mean(col(arm, k))) for k in
                   ("J_zonal", "J_bias_term", "J_pattern_term", "bias_K",
                    "cap_share")} for arm in per_arm}
    effects = {name: compare(col(a), col(b), macros, n_boot=2000)
               for name, a, b in EFFECTS if a in per_arm and b in per_arm}
    var_by_m = difference_variance_two(pilot_dir, "plan_learn", pilot3b_dir,
                                       "pi", refs_dir)
    fit = fit_noise(var_by_m)
    var_6 = fit["A"] + fit["B"] / MEMBERS
    mde = minimum_detectable_effect(var_6, tau=TAU_MACRO)
    j_pi = float(np.mean(col("pi")))
    curves = {}
    for arm in ("plan_learn_rs", "plan_learn"):
        c = [per_arm[arm][i].get("learning_curve") for i in states]
        c = [x for x in c if x]
        curves[arm] = np.mean(c, axis=0).tolist() if c else None
    return {"states": states, "arm_means": means, "pilot_effects": effects,
            "noise_by_members": var_by_m, "noise_fit": fit,
            "power_at_6_members": {"var_state": var_6, "mde": mde,
                                   "mde_fraction_of_J_pi": mde / j_pi},
            "learning_curves": curves}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--pilot-dir", required=True,
                   help="Experiment 3c's pilot runs.")
    p.add_argument("--pilot3b-dir", required=True,
                   help="Experiment 3b's pilot runs (reused arms).")
    p.add_argument("--references-dir", required=True)
    args = p.parse_args(argv)
    result = summarize(args.pilot_dir, args.pilot3b_dir, args.references_dir)
    for arm, m in sorted(result["arm_means"].items(),
                         key=lambda kv: kv[1]["J_zonal"]):
        print(f"  {arm:>16}  J_zonal {m['J_zonal']:.5f}  bias "
              f"{m['bias_K']:+.4f} K  cap {m['cap_share']:.2f}")
    for name, e in result["pilot_effects"].items():
        print(f"  {name}: {e['rel']:+.1%} (t p {e['t_p']:.3g})")
    pw = result["power_at_6_members"]
    print(f"  H1 at 6 members: MDE {pw['mde']:.5f} = "
          f"{pw['mde_fraction_of_J_pi']:.0%} of J_pi")
    curve = result["learning_curves"]["plan_learn_rs"]
    if curve:
        print("  learning curve (ocean-only): "
              + " ".join(f"{x:.2f}" for x in curve))
    out = Path(args.pilot_dir) / "summary.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
