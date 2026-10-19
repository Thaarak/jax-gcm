#!/usr/bin/env python
"""Registered analysis of Experiment 3e (Amendment 9, revision 4).

MCB_PROJECT_REPORT.md Part 29. Frozen with revision 4, before any
evaluation run of Experiment 3e existed. Changing a registered constant
afterwards is a deviation that must be logged.

Experiment 3e asks how well the planner must know the weather. Before each
two-week forecast it corrects its own last forecast toward the true weather
and keeps a share f of that forecast's error (``run_test_world.py plan
--sensing analysis --analysis-error f``): f = 0 is Experiment 3d's planner,
which forecasts from the exact state, and f = 1 sees only the ocean, as in
Experiment 3c. Everything else is 3d's: the fixed land model, its 48
evaluation states, hidden strengths, 10-member references and 6 members.

* 3e's runs (``--runs-dir``): the learning planner at f = 0.1 (the primary
  level), 0.3 and 1, and the planner that assumes nominal strength at
  f = 0.1;
* 3d's runs (``--runs-3d-dir``), reused because what they know is
  unchanged: ``pi``, ``adaptive``, ``fixed`` and ``uncontrolled`` never used
  the weather, and 3d's ``plan_learn`` and ``plan_naive`` are the f = 0 end.

Tests and verdicts are 3b's (``analyze_experiment3b``), on ``J_zonal`` except
for H4:

* **H1 (primary, alone at alpha = 0.05):** ``plan_learn_a10`` vs ``pi``.
* **Secondary family (Holm over three):** H2 ``plan_learn_a10`` vs
  ``plan_naive_a10``; H3 ``plan_learn_a10`` vs 3d's ``plan_learn``; H4
  ``plan_learn_a10`` vs ``plan_learn_a100`` on the learner's strength error
  on day 98, the first scored day (``|log(estimate / truth)|`` averaged over
  bands and members: the learning curve on that day).
* **Descriptive:** the dose-response (each learner, f = 0, 0.1, 0.3 and 1,
  against ``pi``, with its strength error on day 98 and its learning curve);
  ``plan_naive_a10`` vs ``pi`` and vs 3d's ``plan_naive``; ``plan_learn_a10``
  vs ``adaptive``; f = 0.3 and 1 against f = 0.1; f = 1 in the fixed-land
  world against Experiment 3c's logged result; for each new arm, the size of
  its forecast errors and revision 4's detection statistic; everything by
  overall factor.

Before any statistics the script checks every run's land model, sensing,
analysis error and learner noise level against the registered design, and
that every arm has the same number of members.

Example (GX10):
    python analyze_experiment3e.py --runs-dir mcb_experiments_gpu/exp3e/eval \
        --runs-3d-dir mcb_experiments_gpu/exp3d/eval \
        --references-dirs mcb_experiments_gpu/test_world_refs_ramp6_3d/exp3d_eval_b* \
        --design mcb_experiments_gpu/exp3e/registered_design.json \
        --exp3c-analysis mcb_experiments_gpu/exp3c/exp3c_analysis.json \
        --output mcb_experiments_gpu/exp3e/exp3e_analysis.json
"""

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from analyze_exp3e_pilot import detection_terms, detection_z
from analyze_experiment3a import (
    ALPHA_TEST,
    BOOT_SEED,
    EQUIVALENCE_BOUND,
    N_BOOT,
    REL_MIN,
    compare,
    holm,
    verdict,
)
from analyze_experiment3b import SCORE_WINDOW, analyze, load_runs

# --- Registered constants (revision 4) ---------------------------------------
LAND_MODE = "monthly"
ARMS_3E = ("plan_learn_a10", "plan_naive_a10", "plan_learn_a30",
           "plan_learn_a100")
ARMS_3D = ("uncontrolled", "pi", "adaptive", "fixed", "plan_learn",
           "plan_naive")
# Each planner arm's analysis error f, and the design level of 3e's arms.
ANALYSIS_ERROR = {"plan_learn": 0.0, "plan_naive": 0.0,
                  "plan_learn_a10": 0.1, "plan_naive_a10": 0.1,
                  "plan_learn_a30": 0.3, "plan_learn_a100": 1.0}
LEVEL = {"plan_learn_a10": "a10", "plan_naive_a10": "a10",
         "plan_learn_a30": "a30", "plan_learn_a100": "a100"}
SEGMENT_DAYS = 14
LEARNING_DAY = 98                          # the first scored day
PRIMARY = ("H1", "plan_learn_a10", "pi",
           "does the learning planner beat the classical controller when it "
           "knows the weather only through a realistic analysis?")
SECONDARY_FAMILY = (
    ("H2", "plan_learn_a10", "plan_naive_a10", "J_zonal",
     "does learning the strength help when the weather is known only "
     "through a realistic analysis?"),
    ("H3", "plan_learn_a10", "plan_learn", "J_zonal",
     "what does a realistic analysis cost, against knowing the weather "
     "exactly?"),
    ("H4", "plan_learn_a10", "plan_learn_a100", "strength_error_day98",
     "does a realistic analysis let the planner learn the strength?"),
)
DESCRIPTIVE = (("plan_naive_a10", "pi"), ("plan_learn_a10", "adaptive"),
               ("plan_naive_a10", "plan_naive"),
               ("plan_learn_a30", "plan_learn_a10"),
               ("plan_learn_a100", "plan_learn_a10"))
DOSE = (("plan_learn", 0.0), ("plan_learn_a10", 0.1),
        ("plan_learn_a30", 0.3), ("plan_learn_a100", 1.0))
LEARNERS = ("plan_learn_a10", "plan_learn", "plan_learn_a30",
            "plan_learn_a100")
DETECTION_Z = 1.6448536269514722           # one-sided 5%: norm.ppf(0.95)
DETECTION_DAYS = (28, 56, 98, 168)
# -------------------------------------------------------------------------------


def load_design(path):
    """Return the registered design, after checking its levels against these."""
    design = json.loads(Path(path).read_text())
    for arm, level in LEVEL.items():
        got = design["levels"][level]["analysis_error"]
        if got != ANALYSIS_ERROR[arm]:
            raise SystemExit(f"design level {level}: analysis error {got}, "
                             f"expected {ANALYSIS_ERROR[arm]}")
    return design


def expected(arm, design):
    """Return a planner arm's sensing, analysis error and noise level (None: naive)."""
    f = ANALYSIS_ERROR[arm]
    noise = None
    if arm.startswith("plan_learn"):
        noise = (design["exact_learner_noise_k"] if f == 0.0
                 else design["levels"][LEVEL[arm]]["learner_noise_k"])
    return ("exact" if f == 0.0 else "analysis"), f, noise


def check_runs(runs_dir, runs_3d_dir, design):
    """Refuse runs made with the wrong land, sensing, analysis error or noise level, or unequal members."""
    members = set()
    for folder, arms in ((runs_dir, ARMS_3E), (runs_3d_dir, ARMS_3D)):
        for arm in arms:
            for js in sorted((Path(folder) / arm).glob("ic*.json")):
                s = json.loads(js.read_text())
                members.add(s.get("members"))
                if s.get("land_climatology") != LAND_MODE:
                    raise SystemExit(f"{js}: land model "
                                     f"{s.get('land_climatology')!r}, "
                                     f"expected {LAND_MODE!r}")
                if arm not in ANALYSIS_ERROR:
                    continue
                sensing, f, noise = expected(arm, design)
                cfg = s.get("planner") or {}
                if cfg.get("sensing", "exact") != sensing:
                    raise SystemExit(f"{js}: sensing {cfg.get('sensing')!r}"
                                     f", expected {sensing!r}")
                if sensing == "analysis" and cfg.get("analysis_error") != f:
                    raise SystemExit(f"{js}: analysis error "
                                     f"{cfg.get('analysis_error')}, "
                                     f"expected {f}")
                learner = s.get("learner")
                if noise is None:
                    belief = s.get("planner_efficacy") or []
                    if learner or not belief or any(b != 1.0 for b in belief):
                        raise SystemExit(f"{js}: not the planner that "
                                         "assumes nominal strength")
                    continue
                got = ((learner or {}).get("config") or {}).get("noise_k")
                if got is None or not math.isclose(got, noise, rel_tol=1e-9):
                    raise SystemExit(f"{js}: learner noise level {got}, "
                                     f"expected {noise}")
    if len(members) > 1:
        raise SystemExit(f"arms differ in members: {sorted(members)}")


def strength_error(summary, day=LEARNING_DAY):
    """Return ``|log(estimate / truth)|`` on ``day``, averaged over bands and members.

    The estimate is the one the planner planned with on that day; ``day``
    None takes each member's last re-plan.
    """
    truth = np.log(np.asarray(summary["true_efficacy"], np.float64))
    per_member = []
    for log in summary["planner_logs"]:
        records = [r for r in log if day is None or r.get("day") == day]
        if not records or (day is not None and len(records) != 1):
            raise SystemExit(f"no single re-plan on day {day}")
        belief = np.asarray(records[-1]["efficacy_belief"], np.float64)
        per_member.append(np.mean(np.abs(np.log(belief) - truth)))
    return float(np.mean(per_member))


def load(runs_dir, runs_3d_dir, references_dirs, learning_day=LEARNING_DAY):
    """Return ``{arm: {index: metrics}}`` for 3e's and 3d's arms.

    3e's arms get their gains from 3d's no-brightening runs, and every
    learner gets its strength error on ``learning_day``.
    """
    per_arm = load_runs(runs_dir, references_dirs, arms=ARMS_3E,
                        uncontrolled_dir=runs_3d_dir)
    per_arm.update(load_runs(runs_3d_dir, references_dirs, arms=ARMS_3D))
    for arm in LEARNERS:
        folder = Path(runs_dir if arm in ARMS_3E else runs_3d_dir) / arm
        for idx, row in per_arm.get(arm, {}).items():
            summary = json.loads((folder / f"ic{idx:04d}.json").read_text())
            row["strength_error_day98"] = strength_error(summary,
                                                         learning_day)
    return per_arm


def secondary_family(per_arm, states, n_boot=N_BOOT):
    """H2-H4 with Holm over the three, each on its own endpoint."""
    macros = [i // 100 for i in states]
    results = []
    for name, a, b, key, question in SECONDARY_FAMILY:
        x = compare([per_arm[a][i][key] for i in states],
                    [per_arm[b][i][key] for i in states], macros, n_boot)
        results.append((name, a, b, key, question, x))
    adj = holm([r[-1]["t_p"] for r in results])
    return {name: {"arm": a, "comparator": b, "endpoint": key,
                   "question": question, **x, "p_used": p, "t_p_holm": p,
                   "verdict": verdict(x, p),
                   "wilcoxon_agrees": (x["wilcoxon_p"] < ALPHA_TEST)
                   == (x["t_p"] < ALPHA_TEST)}
            for (name, a, b, key, question, x), p in zip(results, adj)}


def dose_response(per_arm, states, n_boot=N_BOOT):
    """Each learner against ``pi``, from f = 0 to f = 1 (descriptive)."""
    macros = [i // 100 for i in states]
    out = []
    for arm, f in DOSE:
        x = compare([per_arm[arm][i]["J_zonal"] for i in states],
                    [per_arm["pi"][i]["J_zonal"] for i in states], macros,
                    n_boot)
        out.append({"analysis_error": f, "arm": arm, "J_zonal": x["mean_A"],
                    "rel_vs_pi": x["rel"], "ci90_rel": x["ci90_rel"],
                    "ci95_rel": x["ci95_rel"], "t_p": x["t_p"],
                    "strength_error_day98": float(np.mean(
                        [per_arm[arm][i]["strength_error_day98"]
                         for i in states]))})
    return out


def replication_3c(dose, exp3c):
    """Compare f = 1 in the fixed-land world with Experiment 3c's result (descriptive)."""
    h1 = exp3c["hypotheses"]["H1"]
    curve = ((exp3c.get("learning") or {}).get("by_arm") or {}).get(
        "plan_learn_rs")
    a100 = next(d for d in dose if d["arm"] == "plan_learn_a100")
    lo, hi = a100["ci95_rel"]
    return {"rel_3c": h1["rel"], "rel_3e": a100["rel_vs_pi"],
            "ci95_3e": [lo, hi],
            "same_direction": (h1["rel"] < 0) == (a100["rel_vs_pi"] < 0),
            "3c_inside_3e_ci95": lo <= h1["rel"] <= hi,
            "strength_error_day98_3c": (curve[LEARNING_DAY // SEGMENT_DAYS]
                                        if curve else None),
            "strength_error_day98_3e": a100["strength_error_day98"]}


def forecast_error_ms(path):
    """Return each forecast's weighted mean square error, ``(members, re-plans)``.

    The miss less its part from the strength error, ``G (truth - belief)``:
    what the forecast got wrong about the weather (and the linearization).
    """
    with np.load(path) as f:
        r = (f["observed"].astype(np.float64)
             - f["predicted"].astype(np.float64)
             - np.einsum("mnik,mnk->mni", f["sensitivity"].astype(np.float64),
                         f["true_efficacy"].astype(np.float64)[None, None]
                         - f["belief"].astype(np.float64)))
        w = f["profile_weights"].astype(np.float64)
    return (w * r * r).sum(-1)


def detection(runs_dir, design, states, overall, days=DETECTION_DAYS):
    """Revision 4's detection statistic and the forecast errors' size, per new arm.

    Brightening works in every state, so the share of (state, member) pairs
    whose statistic exceeds the one-sided 5% level at a day is how often one
    Earth would tell, from that day's look alone (descriptive).
    """
    out = {}
    for arm in ARMS_3E:
        level = design["levels"][LEVEL[arm]]
        z_at = {d: [] for d in days}
        by_factor = {d: {} for d in days}
        ms, files = [], 0
        for idx in states:
            path = Path(runs_dir) / arm / f"ic{idx:04d}.innovations.npz"
            if not path.exists():
                continue
            files += 1
            u, v, _ = detection_terms(path)
            z = detection_z(u, v, level["detection_kappa"])
            with np.load(path) as f:
                day = f["day"]
            for d in days:
                j = np.flatnonzero(day == d)
                if j.size:
                    z_at[d].extend(z[:, j[0]].tolist())
                    by_factor[d].setdefault(str(overall[idx]), []).extend(
                        z[:, j[0]].tolist())
            ms.append(forecast_error_ms(path).ravel())

        def share(zs):
            return float(np.mean(np.asarray(zs) >= DETECTION_Z)) if zs else None

        out[arm] = {
            "analysis_error": level["analysis_error"], "files": files,
            "kappa": level["detection_kappa"],
            "share_detected": {str(d): share(z_at[d]) for d in days},
            "median_z": {str(d): (float(np.median(z_at[d])) if z_at[d]
                                  else None) for d in days},
            "share_detected_by_factor": {
                str(d): {k: share(v) for k, v in sorted(by_factor[d].items())}
                for d in days},
            "forecast_error_noise_k": (float(np.sqrt(np.median(
                np.concatenate(ms)))) if ms else None),
            "registered_noise_k": level["learner_noise_k"]}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs-dir", required=True, help="Experiment 3e's runs.")
    p.add_argument("--runs-3d-dir", required=True,
                   help="Experiment 3d's runs (the reused arms).")
    p.add_argument("--references-dirs", nargs="+", required=True)
    p.add_argument("--design", required=True,
                   help="exp3e/registered_design.json")
    p.add_argument("--exp3c-analysis", required=True,
                   help="Experiment 3c's registered output, for the "
                        "comparison at f = 1.")
    p.add_argument("--output", required=True)
    p.add_argument("--smoke", action="store_true",
                   help="Load and print whatever exists, without statistics.")
    args = p.parse_args(argv)
    design = load_design(args.design)
    check_runs(args.runs_dir, args.runs_3d_dir, design)
    per_arm = load(args.runs_dir, args.runs_3d_dir, args.references_dirs,
                   learning_day=None if args.smoke else LEARNING_DAY)
    missing = [a for a in (*ARMS_3E, *ARMS_3D) if not per_arm.get(a)]
    if missing:
        raise SystemExit(f"arms with no runs: {missing}")
    if args.smoke:
        for arm, rows in per_arm.items():
            for idx, row in rows.items():
                extra = (f" strength error {row['strength_error_day98']:.3f}"
                         if "strength_error_day98" in row else "")
                print(f"  {arm:>15} ic{idx:04d}: J_zonal {row['J_zonal']:.5f}"
                      f" bias {row['bias_K']:+.4f} K members "
                      f"{row['members']}{extra}")
        states = sorted(set.intersection(*(set(r) for r in per_arm.values())))
        overall = {i: per_arm["uncontrolled"][i]["overall"] for i in states}
        for arm, d in detection(args.runs_dir, design, states, overall,
                                days=(SEGMENT_DAYS,)).items():
            print(f"  {arm:>15}: {d['files']} miss files, forecast error "
                  f"{d['forecast_error_noise_k']} K, median Z on day "
                  f"{SEGMENT_DAYS} {d['median_z'][str(SEGMENT_DAYS)]}")
        print("smoke analysis: every arm loaded")
        return
    result = analyze(per_arm, primary=PRIMARY, secondary=(),
                     descriptive_pairs=DESCRIPTIVE, learner_arms=LEARNERS)
    states = result["states"]
    result["hypotheses"].update(secondary_family(per_arm, states))
    dose = dose_response(per_arm, states)
    exp3c = json.loads(Path(args.exp3c_analysis).read_text())
    overall = {i: per_arm["uncontrolled"][i]["overall"] for i in states}
    result.update(
        dose_response=dose, replication_3c=replication_3c(dose, exp3c),
        detection=detection(args.runs_dir, design, states, overall),
        per_state={a: {str(i): v for i, v in rows.items()}
                   for a, rows in per_arm.items()},
        registered={"window": SCORE_WINDOW, "primary": PRIMARY,
                    "secondary_family": SECONDARY_FAMILY,
                    "descriptive": DESCRIPTIVE, "dose": DOSE,
                    "learning_day": LEARNING_DAY,
                    "detection_z": DETECTION_Z,
                    "detection_days": DETECTION_DAYS,
                    "land_climatology": LAND_MODE,
                    "design_sha256": hashlib.sha256(
                        Path(args.design).read_bytes()).hexdigest(),
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
        print(f"  {arm:>15}  J_zonal {m['J_zonal']:.5f} (bias term "
              f"{m['J_bias_term']:.5f})  bias {m['bias_K']:+.4f} K  cap "
              f"{m['cap_share']:.2f}")
    for name, h in result["hypotheses"].items():
        print(f"  {name} {h['arm']} vs {h['comparator']} "
              f"({h.get('endpoint', 'J_zonal')}): rel {h['rel']:+.1%} "
              f"(90% CI {h['ci90_rel'][0]:+.1%}..{h['ci90_rel'][1]:+.1%}), "
              f"p {h['p_used']:.3g} -> {h['verdict']}")
    for d in dose:
        print(f"  f = {d['analysis_error']:.1f} {d['arm']:>15}: vs pi "
              f"{d['rel_vs_pi']:+.1%} (95% CI {d['ci95_rel'][0]:+.1%}.."
              f"{d['ci95_rel'][1]:+.1%}), strength error on day "
              f"{LEARNING_DAY} {d['strength_error_day98']:.3f}")
    for arm, d in result["detection"].items():
        print(f"  {arm:>15}: forecast error {d['forecast_error_noise_k']:.5f} K"
              f" (registered {d['registered_noise_k']:.5f}); detected by "
              + ", ".join(f"day {k} {v:.0%}" for k, v in
                          d["share_detected"].items() if v is not None))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
