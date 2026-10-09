#!/usr/bin/env python
"""Registered analysis of Experiment 3c (Amendment 9, revision 3, Part A).

MCB_PROJECT_REPORT.md Part 28. Frozen with revision 3, before any
evaluation run of Experiment 3c existed. Changing a registered constant
afterwards is a deviation that must be logged.

Experiment 3c repeats Experiment 3b on the same 48 evaluation states, hidden
strengths, 10-member references and 6 members, with planners that see the
ocean but not the weather (``run_test_world.py plan --sensing ocean``):

* 3c's runs (``--runs-dir``): ``plan_learn`` and ``plan_naive``, reported as
  ``plan_learn_rs`` and ``plan_naive_rs``;
* 3b's runs (``--runs-3b-dir``), reused because what they know is
  unchanged: ``pi``, ``adaptive``, ``fixed`` and ``uncontrolled`` never used
  the weather, and 3b's exact-sensing planners ``plan_learn``,
  ``plan_naive`` and ``plan_oracle`` are the comparison.

Endpoint, tests and verdicts are 3b's (``analyze_experiment3b.analyze``):

* **H1 (primary, alone at alpha = 0.05):** ``plan_learn_rs`` vs ``pi``.
* **Secondary family (Holm over two):** H2 ``plan_learn_rs`` vs
  ``plan_naive_rs`` (does learning still help?); H3 ``plan_learn_rs`` vs
  ``plan_learn`` (what does not seeing the weather cost?).
* **Descriptive:** ``plan_naive_rs`` vs ``pi``; ``plan_learn_rs`` vs
  ``adaptive``; ``plan_naive_rs`` vs ``plan_naive``; ``plan_learn_rs`` and
  ``plan_naive_rs`` vs ``plan_oracle``; both learners' learning curves;
  everything by overall factor.

Before any statistics the script checks that every 3c run used ocean-only
sensing, every 3b planner exact sensing, and that every arm has the same
number of members.

Example (GX10):
    python analyze_experiment3c.py --runs-dir mcb_experiments_gpu/exp3c/eval \
        --runs-3b-dir mcb_experiments_gpu/exp3b/eval \
        --references-dirs mcb_experiments_gpu/test_world_refs_ramp6_3b/exp3b_eval_b* \
        --output mcb_experiments_gpu/exp3c/exp3c_analysis.json
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
from analyze_experiment3b import SCORE_WINDOW, analyze, load_runs

# --- Registered constants (revision 3, Part A) -------------------------------
ARMS_3C = ("plan_learn", "plan_naive")
ARMS_3B = ("uncontrolled", "pi", "adaptive", "fixed", "plan_learn",
           "plan_naive", "plan_oracle")
SUFFIX = "_rs"
PRIMARY = ("H1", "plan_learn_rs", "pi",
           "does the learning planner still beat the classical controller "
           "when it cannot see the weather?")
SECONDARY_FAMILY = (
    ("H2", "plan_learn_rs", "plan_naive_rs",
     "does learning the strength still help when the weather is unknown?"),
    ("H3", "plan_learn_rs", "plan_learn",
     "what does not seeing the weather cost the learning planner?"),
)
DESCRIPTIVE = (("plan_naive_rs", "pi"), ("plan_learn_rs", "adaptive"),
               ("plan_naive_rs", "plan_naive"),
               ("plan_learn_rs", "plan_oracle"),
               ("plan_naive_rs", "plan_oracle"))
LEARNERS = ("plan_learn_rs", "plan_learn")
# -------------------------------------------------------------------------------


def check_runs(runs_dir, runs_3b_dir):
    """Refuse runs made with the wrong sensing, or unequal member counts."""
    members = set()
    for folder, arms, sensing in ((runs_dir, ARMS_3C, "ocean"),
                                  (runs_3b_dir, ARMS_3B, "exact")):
        for arm in arms:
            for js in sorted((Path(folder) / arm).glob("ic*.json")):
                summary = json.loads(js.read_text())
                members.add(summary.get("members"))
                if "planner" in summary:
                    got = summary["planner"].get("sensing", "exact")
                    if got != sensing:
                        raise SystemExit(f"{js}: sensing {got!r}, expected "
                                         f"{sensing!r}")
    if len(members) > 1:
        raise SystemExit(f"arms differ in members: {sorted(members)}")


def load(runs_dir, runs_3b_dir, references_dirs):
    """Return ``{arm: {index: metrics}}`` with 3c's planners suffixed ``_rs``."""
    new = load_runs(runs_dir, references_dirs, arms=ARMS_3C,
                    uncontrolled_dir=runs_3b_dir)
    per_arm = {f"{arm}{SUFFIX}": rows for arm, rows in new.items()}
    per_arm.update(load_runs(runs_3b_dir, references_dirs, arms=ARMS_3B))
    return per_arm


def analyze_3c(per_arm, n_boot=N_BOOT):
    """Revision 3's tests for Experiment 3c."""
    return analyze(per_arm, n_boot, primary=PRIMARY,
                   secondary=SECONDARY_FAMILY, descriptive_pairs=DESCRIPTIVE,
                   learner_arms=LEARNERS)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs-dir", required=True, help="Experiment 3c's runs.")
    p.add_argument("--runs-3b-dir", required=True,
                   help="Experiment 3b's runs (the reused arms).")
    p.add_argument("--references-dirs", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--smoke", action="store_true",
                   help="Load and print whatever exists, without statistics.")
    args = p.parse_args(argv)
    check_runs(args.runs_dir, args.runs_3b_dir)
    per_arm = load(args.runs_dir, args.runs_3b_dir, args.references_dirs)
    missing = [a for a, rows in per_arm.items() if not rows]
    if missing:
        raise SystemExit(f"arms with no runs: {missing}")
    if args.smoke:
        for arm, rows in per_arm.items():
            for idx, row in rows.items():
                print(f"  {arm:>14} ic{idx:04d}: J_zonal {row['J_zonal']:.5f}"
                      f" bias {row['bias_K']:+.4f} K members "
                      f"{row['members']}")
        print("smoke analysis: every arm loaded")
        return
    result = analyze_3c(per_arm)
    result.update(per_state={a: {str(i): v for i, v in rows.items()}
                             for a, rows in per_arm.items()},
                  registered={"window": SCORE_WINDOW, "primary": PRIMARY,
                              "secondary_family": SECONDARY_FAMILY,
                              "descriptive": DESCRIPTIVE,
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
        print(f"  {arm:>14}  J_zonal {m['J_zonal']:.5f} (bias term "
              f"{m['J_bias_term']:.5f})  bias {m['bias_K']:+.4f} K  cap "
              f"{m['cap_share']:.2f}")
    for name, h in result["hypotheses"].items():
        print(f"  {name} {h['arm']} vs {h['comparator']}: rel {h['rel']:+.1%} "
              f"(90% CI {h['ci90_rel'][0]:+.1%}..{h['ci90_rel'][1]:+.1%}), "
              f"p {h['p_used']:.3g} -> {h['verdict']}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
