#!/usr/bin/env python
"""Registered analysis of Experiment 3f (Amendment 9, revision 5).

Frozen with revision 5, before any evaluation run of Experiment 3f existed.
Changing a registered constant afterwards is a deviation that must be logged.

Experiment 3f asks whether looking months ahead helps once the brightening's
strength is hidden and must be learned. Every planner learns exactly as 3d's
``plan_learn`` does: from its two-week forecast of the segment it is about to
apply. Only the look-ahead it plans over changes:

* ``learn14``: 3d's learning planner (two weeks, Dubey et al.'s receding
  horizon), run again alongside the new arms;
* ``learn112_mpc``: 16 weeks, as model predictive control: a level and a
  trend per band over eight two-week blocks, every block scored, only the
  first applied, with the 30-day snipped gradient;
* ``learn56_mpc``: the same over 8 weeks (four blocks);
* ``learn112_hold``: 16 weeks with one setting held for the whole look-ahead
  and scored on its mean, as Experiment 3a's long planners did (also with
  the 30-day snip).

Everything else is 3d's: the fixed land model, its 48 evaluation states
(16 macro states x 3 weather branches), hidden strengths, 10-member
references (scoring, days 98-182) and 6 members. Beyond day 200 the long
planners' target comes from a normal-climate-only extension of those
references; the warming they forecast continues the ramp. 3d's ``pi``,
``adaptive``, ``fixed``, ``uncontrolled``, ``plan_learn`` and ``plan_naive``
runs are reused.

Tests and verdicts are 3b's (``analyze_experiment3b``) on ``J_zonal``:

* **H1 (primary, alone at alpha = 0.05):** ``learn112_mpc`` vs ``learn14``.
* **Secondary family (Holm over two):** H2 ``learn56_mpc`` vs ``learn14``;
  H3 ``learn112_mpc`` vs ``learn112_hold``.
* **Descriptive:** every arm against ``pi``; ``learn14`` against 3d's
  ``plan_learn`` (the same planner run again: a drift check);
  ``learn112_mpc`` vs ``learn56_mpc``; bias, effort and cap share; each
  learner's strength error on day 98; compute per re-plan and per run.

An inconclusive H1 is reported with its 90% interval as the bound on how
much a 16-week look-ahead can help or hurt (with 16 macro states the
smallest effect this design can resolve is about 20%).

Before any statistics the script checks every run's land model, sensing,
look-ahead, snip, schedule and learner noise level against the registered
design, that the long planners' target extension matches the scoring
references where they overlap, and that every arm has the same members.
"""

import argparse
import hashlib
import json
import math
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
    holm,
    verdict,
)
from analyze_experiment3b import SCORE_WINDOW, analyze, load_runs
from analyze_experiment3e import strength_error

# --- Registered constants (revision 5) ---------------------------------------
LAND_MODE = "monthly"
ARMS_3F = ("learn14", "learn112_mpc", "learn56_mpc", "learn112_hold")
ARMS_3D = ("uncontrolled", "pi", "adaptive", "fixed", "plan_learn",
           "plan_naive")
# What each new arm's planner must be: (look-ahead, snip window, schedule).
# A window of None is no snip (full backpropagation over two weeks).
PLANNERS = {"learn14": (14, None, "none"),
            "learn112_mpc": (112, 30, "linear"),
            "learn56_mpc": (56, 30, "linear"),
            "learn112_hold": (112, 30, "hold")}
LONG = ("learn112_mpc", "learn56_mpc", "learn112_hold")
LEARNING_DAY = 98                          # the first scored day
MAX_OVERLAP_K = 0.05      # target extension vs references, last 14 shared days
PRIMARY = ("H1", "learn112_mpc", "learn14",
           "when the strength is hidden and learned, does planning 16 weeks "
           "ahead beat planning two weeks ahead?")
SECONDARY_FAMILY = (
    ("H2", "learn56_mpc", "learn14",
     "does planning 8 weeks ahead beat planning two weeks ahead?"),
    ("H3", "learn112_mpc", "learn112_hold",
     "over 16 weeks, does planning a schedule beat holding one setting?"),
)
DESCRIPTIVE = (("learn14", "pi"), ("learn112_mpc", "pi"),
               ("learn56_mpc", "pi"), ("learn112_hold", "pi"),
               ("learn14", "plan_learn"), ("learn112_mpc", "learn56_mpc"),
               ("learn112_hold", "learn14"))
LEARNERS = ("learn14", "learn112_mpc", "learn56_mpc", "learn112_hold",
            "plan_learn")
# -------------------------------------------------------------------------------


def no_snip(window):
    """True when a run's window means no truncation (a huge sentinel)."""
    return window is None or window >= 10_000


def check_runs(runs_dir, runs_3d_dir, design, arms_3f=ARMS_3F):
    """Refuse runs that are not the registered planners, land model or members."""
    members, overlaps = set(), []
    noise = design["learner_noise_k"]
    for folder, arms in ((runs_dir, arms_3f), (runs_3d_dir, ARMS_3D)):
        for arm in arms:
            for js in sorted((Path(folder) / arm).glob("ic*.json")):
                s = json.loads(js.read_text())
                members.add(s.get("members"))
                if s.get("land_climatology") != LAND_MODE:
                    raise SystemExit(f"{js}: land model "
                                     f"{s.get('land_climatology')!r}")
                if arm not in PLANNERS:
                    continue
                look, window, schedule = PLANNERS[arm]
                cfg = s.get("planner") or {}
                got = (cfg.get("lookahead_days"), cfg.get("window_days"),
                       cfg.get("schedule", "none"), cfg.get("sensing"))
                ok = (got[0] == look and got[2] == schedule
                      and got[3] == "exact"
                      and (no_snip(got[1]) if window is None
                           else got[1] == window))
                if not ok:
                    raise SystemExit(f"{js}: planner {got}, expected "
                                     f"{(look, window, schedule, 'exact')}")
                k = ((s.get("learner") or {}).get("config") or {}).get(
                    "noise_k")
                if k is None or not math.isclose(k, noise, rel_tol=1e-9):
                    raise SystemExit(f"{js}: learner noise {k}, expected "
                                     f"{noise}")
                ext = s.get("target_extension")
                if arm in LONG:
                    if not ext:
                        raise SystemExit(f"{js}: no target extension")
                    overlaps.append(ext["overlap_mean_abs_k"])
                    if ext["overlap_mean_abs_k"] > MAX_OVERLAP_K:
                        raise SystemExit(f"{js}: target extension differs "
                                         "from the references by "
                                         f"{ext['overlap_mean_abs_k']:.3f} K")
    if len(members) > 1:
        raise SystemExit(f"arms differ in members: {sorted(members)}")
    return {"members": sorted(members),
            "target_overlap_mean_abs_k_max": max(overlaps, default=None)}


def compute_cost(runs_dir, arm):
    """Median seconds per re-plan and per run (all members) for one arm."""
    replans, runs = [], []
    for js in sorted((Path(runs_dir) / arm).glob("ic*.json")):
        s = json.loads(js.read_text())
        runs.append(s["seconds"])
        replans += [r["seconds"] for log in s["planner_logs"] for r in log
                    if r.get("segment", 0) > 0]      # first re-plan compiles
    return {"replan_s_median": float(np.median(replans)),
            "run_s_median": float(np.median(runs)), "n_runs": len(runs)}


def load(runs_dir, runs_3d_dir, references_dirs, learning_day=LEARNING_DAY,
         arms_3f=ARMS_3F):
    """Return ``{arm: {index: metrics}}``, with each learner's day-98 strength error."""
    per_arm = load_runs(runs_dir, references_dirs, arms=arms_3f,
                        uncontrolled_dir=runs_3d_dir)
    per_arm.update(load_runs(runs_3d_dir, references_dirs, arms=ARMS_3D))
    for arm in LEARNERS:
        folder = Path(runs_dir if arm in arms_3f else runs_3d_dir) / arm
        for idx, row in per_arm.get(arm, {}).items():
            summary = json.loads((folder / f"ic{idx:04d}.json").read_text())
            row["strength_error_day98"] = strength_error(summary,
                                                         learning_day)
    return per_arm


def secondary_family(per_arm, states, family=SECONDARY_FAMILY,
                     n_boot=N_BOOT):
    """H2 and H3 with Holm over those of them that were run."""
    macros = [i // 100 for i in states]
    results = [(name, a, b, q,
                compare([per_arm[a][i]["J_zonal"] for i in states],
                        [per_arm[b][i]["J_zonal"] for i in states], macros,
                        n_boot))
               for name, a, b, q in family]
    adj = holm([r[-1]["t_p"] for r in results])
    return {name: {"arm": a, "comparator": b, "question": q, **x,
                   "p_used": p, "t_p_holm": p, "verdict": verdict(x, p),
                   "wilcoxon_agrees": (x["wilcoxon_p"] < ALPHA_TEST)
                   == (x["t_p"] < ALPHA_TEST)}
            for (name, a, b, q, x), p in zip(results, adj)}


def h1_bound(h1):
    """The registered reading of H1: the verdict, or the bound it implies."""
    lo, hi = h1["ci90_rel"]
    if h1["verdict"] == "inconclusive":
        return (f"inconclusive: a 16-week look-ahead changes J_zonal by "
                f"between {lo:+.0%} and {hi:+.0%} (90% interval)")
    return f"{h1['verdict']}: {h1['rel']:+.0%} (90% CI {lo:+.0%}..{hi:+.0%})"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs-dir", required=True, help="Experiment 3f's runs.")
    p.add_argument("--runs-3d-dir", required=True,
                   help="Experiment 3d's runs (the reused arms).")
    p.add_argument("--references-dirs", nargs="+", required=True)
    p.add_argument("--design", required=True,
                   help="exp3f/registered_design.json")
    p.add_argument("--output", required=True)
    p.add_argument("--arms", nargs="+", default=None,
                   help="Only these 3f arms (the registered budget rule may "
                        "drop learn112_hold; then H3 is not tested).")
    p.add_argument("--smoke", action="store_true",
                   help="Load and print whatever exists, without statistics.")
    args = p.parse_args(argv)
    arms_3f = ARMS_3F
    if args.arms:
        unknown = set(args.arms) - set(ARMS_3F)
        if unknown or "learn14" not in args.arms \
                or "learn112_mpc" not in args.arms:
            raise SystemExit(f"--arms must keep learn14 and learn112_mpc; "
                             f"unknown {sorted(unknown)}")
        arms_3f = tuple(a for a in ARMS_3F if a in args.arms)
    design = json.loads(Path(args.design).read_text())
    checks = check_runs(args.runs_dir, args.runs_3d_dir, design, arms_3f)
    per_arm = load(args.runs_dir, args.runs_3d_dir, args.references_dirs,
                   learning_day=None if args.smoke else LEARNING_DAY,
                   arms_3f=arms_3f)
    missing = [a for a in (*arms_3f, *ARMS_3D) if not per_arm.get(a)]
    if missing:
        raise SystemExit(f"arms with no runs: {missing}")
    cost = {a: compute_cost(args.runs_dir, a) for a in arms_3f}
    if args.smoke:
        for arm, rows in per_arm.items():
            for idx, row in rows.items():
                print(f"  {arm:>14} ic{idx:04d}: J_zonal {row['J_zonal']:.5f}"
                      f" bias {row['bias_K']:+.4f} K members "
                      f"{row['members']}")
        for arm, c in cost.items():
            print(f"  {arm:>14}: {c['replan_s_median']:.1f} s per re-plan, "
                  f"{c['run_s_median']:.0f} s per run")
        print(f"checks: {checks}")
        print("smoke analysis: every arm loaded")
        return
    learners = tuple(a for a in LEARNERS if a in per_arm)
    pairs = tuple(d for d in DESCRIPTIVE
                  if d[0] in per_arm and d[1] in per_arm)
    result = analyze(per_arm, primary=PRIMARY, secondary=(),
                     descriptive_pairs=pairs, learner_arms=learners)
    states = result["states"]
    family = tuple(h for h in SECONDARY_FAMILY
                   if h[1] in per_arm and h[2] in per_arm)
    if family:
        result["hypotheses"].update(secondary_family(per_arm, states, family))
    base = cost["learn14"]["replan_s_median"]
    result.update(
        h1_reading=h1_bound(result["hypotheses"]["H1"]),
        compute={a: {**c, "replan_vs_learn14": c["replan_s_median"] / base}
                 for a, c in cost.items()},
        checks=checks,
        per_state={a: {str(i): v for i, v in rows.items()}
                   for a, rows in per_arm.items()},
        registered={"window": SCORE_WINDOW, "primary": PRIMARY,
                    "secondary_family": SECONDARY_FAMILY,
                    "descriptive": DESCRIPTIVE, "planners": PLANNERS,
                    "arms": arms_3f, "learning_day": LEARNING_DAY,
                    "land_climatology": LAND_MODE,
                    "max_overlap_k": MAX_OVERLAP_K,
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
        print(f"  {arm:>14}  J_zonal {m['J_zonal']:.5f}  bias "
              f"{m['bias_K']:+.4f} K  effort {m['effort']:.4f}  cap "
              f"{m['cap_share']:.2f}")
    for name, h in result["hypotheses"].items():
        print(f"  {name} {h['arm']} vs {h['comparator']}: rel {h['rel']:+.1%}"
              f" (90% CI {h['ci90_rel'][0]:+.1%}..{h['ci90_rel'][1]:+.1%}),"
              f" p {h['p_used']:.3g} -> {h['verdict']}")
    print(f"  H1 reading: {result['h1_reading']}")
    for arm, c in result["compute"].items():
        print(f"  {arm:>14}: {c['replan_s_median']:.1f} s per re-plan "
              f"({c['replan_vs_learn14']:.1f}x learn14)")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
