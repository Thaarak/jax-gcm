#!/usr/bin/env python
"""Experiment 3f's pilot summary and registered budget rule (Amendment 9 revision 5).

Run on 3d's training states only, BEFORE the freeze. The pilot runs every 3f
arm (``learn14``, ``learn112_mpc``, ``learn56_mpc``, ``learn112_hold``) on a
few training states with 3 members, and this script:

* checks that every run finished, with finite scores, and that the long
  planners' target extension matches the references where they overlap;
* measures each arm's cost: seconds per re-plan and per run;
* applies the BUDGET RULE, fixed before the pilot ran: project the
  evaluation's wall-clock time (48 states, 6 members, the pilot's measured
  speed-up from running ``PARALLEL`` runs at once) and drop arms from the end
  of the priority list ``learn14, learn112_mpc, learn56_mpc, learn112_hold``
  until it is at most ``BUDGET_H`` hours. ``learn14`` and ``learn112_mpc``
  (H1) are never dropped.
* writes ``registered_design.json``: the arms, members, learner settings and
  the projection.

The pilot's scores (J_zonal, bias, cap share) are written to the summary as
a feasibility record only; no setting is chosen from them.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

PRIORITY = ("learn14", "learn112_mpc", "learn56_mpc", "learn112_hold")
KEEP = ("learn14", "learn112_mpc")
BUDGET_H = 60.0
EVAL_STATES = 48
EVAL_MEMBERS = 6
LEARNER_NOISE_K = 0.00069104      # 3d's registered level (exact sensing)
LEARNER_MODE = "bands"
MAX_OVERLAP_K = 0.05


def arm_record(folder: Path):
    runs = sorted(folder.glob("ic*.json"))
    rec = {"runs": len(runs), "J_zonal": [], "bias_K": [], "cap_share": [],
           "run_s": [], "replan_s": [], "members": set(), "overlap_k": []}
    for js in runs:
        s = json.loads(js.read_text())
        rec["members"].add(s["members"])
        ctrl = s["objective"]["controlled"]
        # Feasibility only: the zonal score's two terms from the summary.
        rec["J_zonal"].append(ctrl["bias_sq"] + ctrl["zonal_pattern"])
        rec["bias_K"].append(ctrl["bias_K"])
        sched = np.asarray(s["schedules"], np.float64)
        cap = (s.get("planner") or {}).get("cap", 0.15)
        rec["cap_share"].append(float(np.mean(sched >= cap - 1e-6)))
        rec["run_s"].append(s["seconds"])
        rec["replan_s"] += [r["seconds"] for log in s["planner_logs"]
                            for r in log if r.get("segment", 0) > 0]
        ext = s.get("target_extension")
        if ext:
            rec["overlap_k"].append(ext["overlap_mean_abs_k"])
    return rec


def finite(values):
    return all(v is not None and math.isfinite(v) for v in values)


def summarize(pilot_dir: Path, wall_s: float, parallel: int, n_states: int,
              members: int):
    arms, problems = {}, []
    for arm in PRIORITY:
        rec = arm_record(pilot_dir / arm)
        if rec["runs"] != n_states:
            problems.append(f"{arm}: {rec['runs']} of {n_states} runs")
        if not finite([v for v in rec["J_zonal"]]):
            problems.append(f"{arm}: non-finite scores")
        if any(o > MAX_OVERLAP_K for o in rec["overlap_k"]):
            problems.append(f"{arm}: target extension off by "
                            f"{max(rec['overlap_k']):.3f} K")
        arms[arm] = {
            "runs": rec["runs"], "members": sorted(rec["members"]),
            "run_s_median": float(np.median(rec["run_s"])) if rec["run_s"]
            else None,
            "replan_s_median": float(np.median(rec["replan_s"]))
            if rec["replan_s"] else None,
            "feasibility_only": {
                "J_zonal_mean": float(np.mean(rec["J_zonal"]))
                if finite(rec["J_zonal"]) and rec["J_zonal"] else None,
                "bias_K_mean": float(np.mean(rec["bias_K"]))
                if finite(rec["bias_K"]) and rec["bias_K"] else None,
                "cap_share_mean": float(np.mean(rec["cap_share"]))
                if finite(rec["cap_share"]) and rec["cap_share"] else None},
            "target_overlap_k_max": max(rec["overlap_k"], default=None)}
    serial = sum(sum(arm_record(pilot_dir / a)["run_s"]) for a in PRIORITY)
    speedup = max(1.0, min(float(parallel), serial / wall_s)) \
        if wall_s > 0 else 1.0
    # Evaluation cost per arm: the pilot's median run, scaled to 6 members
    # (the planner runs once per member), 48 states, at the measured speed-up.
    scale = EVAL_MEMBERS / members
    projected = {a: arms[a]["run_s_median"] * scale * EVAL_STATES / speedup
                 / 3600.0 for a in PRIORITY if arms[a]["run_s_median"]}
    chosen = list(PRIORITY)
    while sum(projected.get(a, 0.0) for a in chosen) > BUDGET_H \
            and chosen[-1] not in KEEP:
        chosen.pop()
    return {"arms": arms, "problems": problems, "pilot_wall_s": wall_s,
            "pilot_serial_s": serial, "speedup": speedup,
            "projected_eval_h": projected,
            "projected_total_h": sum(projected.get(a, 0.0) for a in chosen),
            "budget_h": BUDGET_H, "eval_arms": chosen,
            "dropped": [a for a in PRIORITY if a not in chosen]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--pilot-dir", required=True)
    p.add_argument("--wall-seconds", type=float, required=True,
                   help="The pilot's wall-clock time (all arms).")
    p.add_argument("--parallel", type=int, required=True)
    p.add_argument("--states", type=int, required=True)
    p.add_argument("--members", type=int, required=True)
    p.add_argument("--design-out", default=None,
                   help="Write registered_design.json here.")
    args = p.parse_args(argv)
    pilot = Path(args.pilot_dir)
    summary = summarize(pilot, args.wall_seconds, args.parallel, args.states,
                        args.members)
    (pilot / "summary.json").write_text(json.dumps(summary, indent=2))
    for arm, a in summary["arms"].items():
        print(f"  {arm:>14}: {a['runs']} runs, {a['replan_s_median']} s per "
              f"re-plan, {a['run_s_median']} s per run; projected "
              f"{summary['projected_eval_h'].get(arm, float('nan')):.1f} h")
    print(f"speed-up {summary['speedup']:.2f}; evaluation arms "
          f"{summary['eval_arms']} ({summary['projected_total_h']:.1f} h); "
          f"dropped {summary['dropped']}")
    if summary["problems"]:
        print("PROBLEMS: " + "; ".join(summary["problems"]))
        raise SystemExit(1)
    if args.design_out:
        design = {"experiment": "3f",
                  "revision": "Amendment 9 revision 5",
                  "land_climatology": "monthly",
                  "members": EVAL_MEMBERS, "reference_members": 10,
                  "learner_mode": LEARNER_MODE,
                  "learner_noise_k": LEARNER_NOISE_K,
                  "learner_prior_sd": 1.0,
                  "arms": summary["eval_arms"],
                  "dropped_by_budget": summary["dropped"],
                  "target_extension_days": 280,
                  "pilot": {k: summary[k] for k in
                            ("speedup", "projected_eval_h",
                             "projected_total_h", "budget_h")}}
        Path(args.design_out).write_text(json.dumps(design, indent=2))
        print(f"-> {args.design_out}")


if __name__ == "__main__":
    main()
