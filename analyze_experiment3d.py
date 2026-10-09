#!/usr/bin/env python
"""Registered analysis of Experiment 3d (Amendment 9, revision 3, Part B).

MCB_PROJECT_REPORT.md Part 28. Frozen with revision 3, before any
evaluation reference, run or score of Experiment 3d existed. Changing a
registered constant afterwards is a deviation that must be logged.

Experiment 3d repeats Experiment 3b's tests in the model with the land
fixed (``JCM_LAND_CLIMATOLOGY=monthly``): a new base climate, 32 new macro
states (100-131), their own training side and hidden strengths, 10-member
references and 6 members. The arms are 3b's except the oracle:
``plan_learn``, ``plan_naive``, ``pi``, ``adaptive``, ``fixed``,
``uncontrolled``.

Endpoint, tests and verdicts are 3b's (``analyze_experiment3b.analyze``):

* **H1 (primary, alone at alpha = 0.05):** ``plan_learn`` vs ``pi``.
* **Secondary family (Holm over two):** H2 ``plan_learn`` vs
  ``plan_naive``; H3 ``pi`` vs ``fixed``.
* **Descriptive:** ``plan_naive`` vs ``pi``; ``adaptive`` vs ``pi``;
  ``plan_learn`` vs ``adaptive``; the learning curve; everything by overall
  factor; and the replication: for H1-H3, 3b's relative difference, 3d's,
  whether they point the same way, and whether 3b's estimate lies inside
  3d's 95% interval.

Before any statistics the script checks that every run was made with the
fixed land model.

Example (GX10):
    python analyze_experiment3d.py --runs-dir mcb_experiments_gpu/exp3d/eval \
        --references-dirs mcb_experiments_gpu/test_world_refs_ramp6_3d/exp3d_eval_b* \
        --exp3b-analysis mcb_experiments_gpu/exp3b/exp3b_analysis.json \
        --output mcb_experiments_gpu/exp3d/exp3d_analysis.json
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from analyze_experiment3a import (
    ALPHA_TEST,
    BOOT_SEED,
    EQUIVALENCE_BOUND,
    N_BOOT,
    REL_MIN,
)
from analyze_experiment3b import (
    PRIMARY,
    SCORE_WINDOW,
    SECONDARY_FAMILY,
    analyze,
    load_runs,
)

# --- Registered constants (revision 3, Part B) -------------------------------
ARMS = ("uncontrolled", "plan_learn", "plan_naive", "pi", "adaptive", "fixed")
DESCRIPTIVE = (("plan_naive", "pi"), ("adaptive", "pi"),
               ("plan_learn", "adaptive"))
LAND_MODE = "monthly"
# -------------------------------------------------------------------------------


def check_land(runs_dir):
    """Refuse any run that was not made with the fixed land model."""
    for arm in ARMS:
        for js in sorted((Path(runs_dir) / arm).glob("ic*.json")):
            got = json.loads(js.read_text()).get("land_climatology")
            if got != LAND_MODE:
                raise SystemExit(f"{js}: land model {got!r}, expected "
                                 f"{LAND_MODE!r}")


def replication(hypotheses, exp3b_hypotheses):
    """Compare each hypothesis with Experiment 3b's (descriptive)."""
    out = {}
    for name, h in hypotheses.items():
        old = exp3b_hypotheses.get(name)
        if old is None:
            continue
        lo, hi = h["ci95_rel"]
        out[name] = {"rel_3b": old["rel"], "rel_3d": h["rel"],
                     "ci95_3d": [lo, hi],
                     "same_direction": (old["rel"] < 0) == (h["rel"] < 0),
                     "3b_inside_3d_ci95": lo <= old["rel"] <= hi,
                     "verdict_3b": old["verdict"], "verdict_3d": h["verdict"]}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs-dir", required=True)
    p.add_argument("--references-dirs", nargs="+", required=True)
    p.add_argument("--exp3b-analysis", required=True,
                   help="Experiment 3b's registered output, for the "
                        "replication comparison.")
    p.add_argument("--output", required=True)
    p.add_argument("--smoke", action="store_true",
                   help="Load and print whatever exists, without statistics.")
    args = p.parse_args(argv)
    check_land(args.runs_dir)
    per_arm = load_runs(args.runs_dir, args.references_dirs, arms=ARMS)
    missing = [a for a in ARMS if not per_arm.get(a)]
    if missing:
        raise SystemExit(f"arms with no runs: {missing}")
    if args.smoke:
        for arm, rows in per_arm.items():
            for idx, row in rows.items():
                print(f"  {arm:>13} ic{idx:04d}: J_zonal {row['J_zonal']:.5f}"
                      f" bias {row['bias_K']:+.4f} K overall {row['overall']}"
                      f" members {row['members']}")
        print("smoke analysis: every arm loaded")
        return
    result = analyze(per_arm, primary=PRIMARY, secondary=SECONDARY_FAMILY,
                     descriptive_pairs=DESCRIPTIVE)
    exp3b = json.loads(Path(args.exp3b_analysis).read_text())
    result.update(replication=replication(result["hypotheses"],
                                          exp3b["hypotheses"]),
                  per_state={a: {str(i): v for i, v in rows.items()}
                             for a, rows in per_arm.items()},
                  registered={"window": SCORE_WINDOW, "primary": PRIMARY,
                              "secondary_family": SECONDARY_FAMILY,
                              "descriptive": DESCRIPTIVE,
                              "land_climatology": LAND_MODE,
                              "alpha": ALPHA_TEST, "rel_min": REL_MIN,
                              "equivalence_bound": EQUIVALENCE_BOUND,
                              "n_boot": N_BOOT, "seed": BOOT_SEED},
                  created_utc=datetime.now(timezone.utc).isoformat())
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"{result['n_states']} states, {result['n_macro']} macro states")
    for arm, m in sorted(result["arm_means"].items(),
                         key=lambda kv: kv[1]["J_zonal"]):
        print(f"  {arm:>13}  J_zonal {m['J_zonal']:.5f} (bias term "
              f"{m['J_bias_term']:.5f})  bias {m['bias_K']:+.4f} K  cap "
              f"{m['cap_share']:.2f}")
    for name, h in result["hypotheses"].items():
        r = result["replication"].get(name, {})
        print(f"  {name} {h['arm']} vs {h['comparator']}: rel {h['rel']:+.1%} "
              f"(90% CI {h['ci90_rel'][0]:+.1%}..{h['ci90_rel'][1]:+.1%}), "
              f"p {h['p_used']:.3g} -> {h['verdict']} (3b: "
              f"{r.get('rel_3b', float('nan')):+.1%})")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
