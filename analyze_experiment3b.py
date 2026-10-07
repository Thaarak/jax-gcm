#!/usr/bin/env python
"""Registered analysis of Experiment 3b (Amendment 9, revision 2).

MCB_PROJECT_REPORT.md Part 24. Frozen with revision 2, before any
evaluation reference or run of Experiment 3b existed. Changing a registered
constant afterwards is a deviation that must be logged.

Every state hides a spraying strength per band from the controllers
(``jcm.mcb.hidden_strength``): an overall factor of 0.5, 1 or 2 (each
evaluation ocean state has one weather branch at each), times regional
factors. Arms (one folder each under ``--runs-dir``, run summaries
``ic<index>.json`` with ``ic<index>.fields.npz``):

* ``plan_learn``: the 14-day Gauss-Newton planner that learns the strength
  from its forecast misses (``jcm.mcb.planner.LearningPlanner``);
* ``plan_naive``: the same planner assuming nominal strength;
* ``plan_oracle``: the same planner told the true strength (the ceiling);
* ``pi``: the GLENS-style PI controller (T0-T2, feedforward);
* ``adaptive``: the Tier 2 law, learning one overall strength;
* ``fixed``: the best fixed pattern, designed at nominal strength;
* ``uncontrolled``: no brightening (for the gains).

Endpoint (PRIMARY): ``J_zonal`` over days 98-182, exactly as in Experiment
3a (``analyze_experiment3a.state_metrics``): ``alpha <e>^2 + beta Var(e)`` of
the ocean latitude profile of the member-mean SST error against the
normal-climate target. Its two terms are also reported separately.

Tests. A comparison uses ``d = J_A - J_B`` per state, ``rel = mean(d) /
mean(J_B)``, macro states as the unit (branches averaged first: every macro
mean averages one branch per overall factor), the paired t test (two-sided)
and the exact Wilcoxon test on the macro means, and a hierarchical bootstrap
interval (``analyze_experiment3a``'s functions, unchanged).

* **H1 (primary, alone at alpha = 0.05):** ``plan_learn`` vs ``pi``.
* **Secondary family (Holm over two):** H2 ``plan_learn`` vs ``plan_naive``
  (does learning help?), H3 ``pi`` vs ``fixed`` (is feedback essential under
  this uncertainty?).
* **Verdicts:** Experiment 3a's grid (better / worse / equivalent /
  negligible / inconclusive; 5% and +-10%).
* **Descriptive:** every arm against the oracle and against no brightening;
  ``plan_naive`` vs ``pi``; ``adaptive`` vs ``pi``; every arm by overall
  factor; the learning curves (``|log(estimate / truth)|`` by segment).

``--smoke`` loads whatever runs exist and prints them without statistics,
so the campaign's smoke test exercises this script end to end.

Example (GX10):
    python analyze_experiment3b.py --runs-dir mcb_experiments_gpu/exp3b/eval \
        --references-dirs mcb_experiments_gpu/test_world_refs_ramp6_3b/exp3b_eval_b* \
        --output mcb_experiments_gpu/exp3b/exp3b_analysis.json
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from analyze_experiment3a import (
    ALPHA_TEST,
    BOOT_SEED,
    EQUIVALENCE_BOUND,
    N_BOOT,
    REL_MIN,
    compare,
    gains,
    holm,
    state_metrics,
    verdict,
)
from jcm.mcb.scores import PATTERN_ALPHA, PATTERN_BETA, area_weights, zonal_projection

# --- Registered constants (revision 2) ---------------------------------------
SCORE_WINDOW = (98, 182)
UNCONTROLLED_ARM = "uncontrolled"
ORACLE_ARM = "plan_oracle"
ARMS = ("uncontrolled", "plan_learn", "plan_naive", "plan_oracle", "pi",
        "adaptive", "fixed")
PRIMARY = ("H1", "plan_learn", "pi",
           "does a planner that learns the strength beat the classical "
           "feedback controller?")
SECONDARY_FAMILY = (
    ("H2", "plan_learn", "plan_naive",
     "does learning the strength help the planner?"),
    ("H3", "pi", "fixed",
     "is feedback essential when the strength is unknown?"),
)
DESCRIPTIVE = (("plan_naive", "pi"), ("adaptive", "pi"),
               ("plan_learn", "adaptive"), ("plan_learn", "plan_oracle"),
               ("plan_naive", "plan_oracle"), ("pi", "plan_oracle"))
OVERALL_FACTORS = (0.5, 1.0, 2.0)
# -------------------------------------------------------------------------------


def overall_factor(strength) -> float:
    """Return the registered overall factor of a per-band strength.

    The regional factors have a geometric mean of exactly 1, so the
    strength's geometric mean is the overall factor.
    """
    g = float(np.exp(np.mean(np.log(np.asarray(strength, np.float64)))))
    return min(OVERALL_FACTORS, key=lambda f: abs(np.log(g / f)))


def j_terms(fields, target_sst, ocean, w_ocean, window=SCORE_WINDOW):
    """Return the two terms of ``J_zonal``: the bias term and the pattern term."""
    s, e = window
    err = (np.asarray(fields["sst"][s:e], np.float64).mean(0)
           - np.asarray(target_sst[s:e], np.float64).mean(0)) * ocean
    z = zonal_projection(err, ocean)
    mean = float(np.sum(w_ocean * z))
    return {"J_bias_term": PATTERN_ALPHA * mean ** 2,
            "J_pattern_term": PATTERN_BETA * float(np.sum(w_ocean
                                                          * (z - mean) ** 2))}


def learning_curve(summary):
    """Return ``|log(estimate / truth)|`` per segment, averaged over bands and members."""
    truth = np.asarray(summary["true_efficacy"], np.float64)
    logs = summary.get("planner_logs") or []
    per_member = []
    for log in logs:
        beliefs = np.asarray([r["efficacy_belief"] for r in log], np.float64)
        per_member.append(np.abs(np.log(beliefs / truth)).mean(axis=1))
    return np.mean(per_member, axis=0).tolist() if per_member else None


def find_references(idx, references_dirs):
    """Return the path of ``ic<idx>_references.npz`` in any of the folders."""
    for d in references_dirs:
        path = Path(d) / f"ic{idx:04d}_references.npz"
        if path.exists():
            return path
    raise SystemExit(f"no references for state {idx} in {references_dirs}")


def load_runs(runs_dir, references_dirs, arms=ARMS):
    """Return ``{arm: {index: metrics}}`` for every arm and state with runs."""
    with np.load(Path(references_dirs[0]) / "grid.npz") as g:
        lats = g["latitudes_rad"]
        ocean = np.asarray(g["ocean_mask"], np.float64)
        land = np.asarray(g["land_mask"], np.float64)
    masks = {"ocean": np.asarray(area_weights(lats, ocean), np.float64),
             "land": np.asarray(area_weights(lats, land), np.float64),
             "global": np.asarray(area_weights(lats, np.ones_like(ocean)),
                                  np.float64)}
    runs = Path(runs_dir)
    table, uncontrolled = {}, {}
    for arm in arms:
        table[arm] = {}
        for js in sorted((runs / arm).glob("ic*.json")):
            idx = int(js.stem[2:])
            fields_path = js.with_suffix(".fields.npz")
            if not fields_path.exists():
                continue
            summary = json.loads(js.read_text())
            refs = np.load(find_references(idx, references_dirs))
            target = {k: refs[f"normal_{k}"] for k in
                      ("sst", "land_temperature", "precipitation")}
            with np.load(fields_path) as f:
                fields = {k: f[k] for k in ("sst", "land_temperature",
                                            "precipitation")}
            row = state_metrics(fields, target["sst"], ocean,
                                masks["ocean"], summary, SCORE_WINDOW)
            row.update(j_terms(fields, target["sst"], ocean, masks["ocean"]))
            row["members"] = summary.get("members")
            row["seconds"] = summary.get("seconds")
            row["overall"] = overall_factor(summary["true_efficacy"])
            if summary.get("learner"):
                row["learning_curve"] = learning_curve(summary)
            if arm == UNCONTROLLED_ARM:
                uncontrolled[idx] = fields
            table[arm][idx] = (row, fields, target)
    out = {}
    for arm, rows in table.items():
        out[arm] = {}
        for idx, (row, fields, target) in rows.items():
            if idx in uncontrolled:
                row["G"] = gains(fields, uncontrolled[idx], target, masks,
                                 SCORE_WINDOW)
            out[arm][idx] = row
    return out


def analyze(per_arm, n_boot=N_BOOT):
    """Apply the registered tests to ``{arm: {index: metrics}}``."""
    complete = set.intersection(*(set(v) for v in per_arm.values() if v))
    states = sorted(complete)
    macros = [i // 100 for i in states]

    def col(arm, key="J_zonal", subset=None):
        keep = states if subset is None else subset
        return [per_arm[arm][i][key] for i in keep]

    name, a, b, question = PRIMARY
    r = compare(col(a), col(b), macros, n_boot)
    hyps = {name: {"arm": a, "comparator": b, "question": question, **r,
                   "p_used": r["t_p"], "verdict": verdict(r, r["t_p"]),
                   "wilcoxon_agrees": (r["wilcoxon_p"] < ALPHA_TEST)
                   == (r["t_p"] < ALPHA_TEST)}}
    family = [compare(col(a), col(b), macros, n_boot)
              for _, a, b, _ in SECONDARY_FAMILY]
    adj = holm([x["t_p"] for x in family])
    for (name, a, b, question), x, p in zip(SECONDARY_FAMILY, family, adj):
        hyps[name] = {"arm": a, "comparator": b, "question": question, **x,
                      "p_used": p, "t_p_holm": p, "verdict": verdict(x, p),
                      "wilcoxon_agrees": (x["wilcoxon_p"] < ALPHA_TEST)
                      == (x["t_p"] < ALPHA_TEST)}
    descriptive = {}
    for a, b in DESCRIPTIVE:
        x = compare(col(a), col(b), macros, n_boot)
        descriptive[f"{a}_vs_{b}"] = {**x, "verdict_unadjusted":
                                      verdict(x, x["t_p"])}
    vs_uncontrolled = {arm: compare(col(arm), col(UNCONTROLLED_ARM), macros,
                                    n_boot)
                       for arm in per_arm if arm != UNCONTROLLED_ARM}
    keys = ("J_zonal", "J_bias_term", "J_pattern_term", "J_map", "bias_K",
            "effort", "cap_share", "max_band_share")
    means = {arm: {k: float(np.mean(col(arm, k))) for k in keys}
             for arm in per_arm}
    by_factor = {}
    for f in OVERALL_FACTORS:
        subset = [i for i in states if per_arm[UNCONTROLLED_ARM][i]["overall"]
                  == f]
        by_factor[str(f)] = {
            "n_states": len(subset),
            **{arm: {k: float(np.mean(col(arm, k, subset))) for k in
                     ("J_zonal", "J_bias_term", "J_pattern_term", "bias_K",
                      "cap_share")}
               for arm in per_arm}} if subset else {"n_states": 0}
    curves = [per_arm["plan_learn"][i].get("learning_curve") for i in states]
    curves = [c for c in curves if c]
    learning = ({"mean_abs_log_error_by_segment":
                 np.mean(curves, axis=0).tolist()} if curves else None)
    for arm in per_arm:
        if all("G" in per_arm[arm][i] for i in states):
            for g in per_arm[arm][states[0]]["G"]:
                vals = [per_arm[arm][i]["G"][g] for i in states]
                means[arm][f"G_{g}"] = float(np.median(vals))
    return {"states": states, "n_states": len(states),
            "n_macro": len(set(macros)), "hypotheses": hyps,
            "descriptive": descriptive, "vs_uncontrolled": vs_uncontrolled,
            "arm_means": means, "by_overall_factor": by_factor,
            "learning": learning}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs-dir", required=True)
    p.add_argument("--references-dirs", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--smoke", action="store_true",
                   help="Load and print whatever exists, without statistics.")
    args = p.parse_args(argv)
    per_arm = load_runs(args.runs_dir, args.references_dirs)
    missing = [a for a in ARMS if not per_arm.get(a)]
    if missing:
        raise SystemExit(f"arms with no runs: {missing}")
    if args.smoke:
        for arm, rows in per_arm.items():
            for idx, row in rows.items():
                print(f"  {arm:>13} ic{idx:04d}: J_zonal {row['J_zonal']:.5f} "
                      f"bias {row['bias_K']:+.4f} K overall {row['overall']}"
                      f" members {row['members']}")
        print("smoke analysis: every arm loaded")
        return
    result = analyze(per_arm)
    result.update(per_state={a: {str(i): v for i, v in rows.items()}
                             for a, rows in per_arm.items()},
                  registered={"window": SCORE_WINDOW, "primary": PRIMARY,
                              "secondary_family": SECONDARY_FAMILY,
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
        print(f"  {name} {h['arm']} vs {h['comparator']}: rel {h['rel']:+.1%} "
              f"(90% CI {h['ci90_rel'][0]:+.1%}..{h['ci90_rel'][1]:+.1%}), "
              f"p {h['p_used']:.3g} -> {h['verdict']}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
