#!/usr/bin/env python
"""Registered analysis of Experiment 2 (Amendment 9, revision 1.1, Part B).

Frozen with revision 1.1, before any Experiment 2 data existed. Changing a
constant below after the evaluation runs is a deviation that must be logged.

Inputs:
- ``--runs-dir``: one folder per arm holding ``ic<index>.json`` and
  ``ic<index>.fields.npz`` (``run_experiment2.py episode``).
- ``--references-dir``: the evaluation references (``grid.npz``,
  ``ic<index>_references.npz``).
- ``--designs``: the frozen ``designs.json``, for the costs.

**Metrics.** Revision 1's own frozen functions, imported from
``analyze_experiment3a.py``, so the two experiments are scored identically.
Per state and arm, ``e`` is the 5-member-mean SST over days 98-182 minus
the normal-climate target over the same days.
- ``J_zonal`` (PRIMARY): ``<e>^2 + 0.5 Var(e)`` of the error's ocean
  zonal-mean profile.
- Also: ``J_map``, the ocean-mean bias, the effort, the cap share, and the
  gains G against the matched uncontrolled arm (descriptive).

**Tests.** Revision 1's, imported too:
- paired differences on the 8 macro-state means (branches averaged);
- a two-sided paired t test with the exact Wilcoxon test alongside;
- a hierarchical bootstrap of the relative effect (10000 replicates, seed
  2026);
- the verdict grid (better, worse, equivalent, negligible, inconclusive)
  with the same 5% and 10% bounds.

The four primary hypotheses form one Holm family. Secondary comparisons are
unadjusted.

**Costs (descriptive).** The model time each design needed, from
``designs.json``:
- gradient: 4 Jacobian runs per training state;
- brute force: K + 1 forward runs per state per weather sample.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from analyze_experiment3a import compare, gains, holm, state_metrics, verdict
from jcm.mcb.scores import area_weights

ARMS = ("uncontrolled", "uniform", "brute5", "brute5_eq", "grad5", "bptt5",
        "sunlight5", "brute13", "brute13_eq", "grad13", "sunlight13")
UNCONTROLLED_ARM = "uncontrolled"
HYPOTHESES = (
    ("H1", "grad13", "brute13_eq",
     "with 13 knobs and the same model time, does the gradient design beat "
     "brute force?"),
    ("H2", "grad13", "sunlight13",
     "does the model's gradient know more than the sunlight formula?"),
    ("H3", "grad5", "bptt5",
     "does snipping the atmosphere make gradient design work at these "
     "horizons?"),
    ("H4", "grad13", "grad5", "do 13 knobs beat 5 for the gradient design?"),
)
SECONDARY = (
    ("grad13", "brute13"),
    ("grad5", "brute5"),
    ("grad5", "brute5_eq"),
    ("brute13", "brute5"),
    ("brute13", "brute13_eq"),
    ("brute5", "brute5_eq"),
    ("grad13", "uniform"),
    ("brute5", "uniform"),
    ("grad5", "sunlight5"),
    ("sunlight13", "uniform"),
    ("sunlight5", "uniform"),
)


def load_runs(runs_dir, references_dir):
    """Return ``{arm: {index: metrics}}`` for every complete run."""
    with np.load(Path(references_dir) / "grid.npz") as g:
        lats = g["latitudes_rad"]
        ocean = np.asarray(g["ocean_mask"], np.float64)
        land = np.asarray(g["land_mask"], np.float64)
    masks = {"ocean": np.asarray(area_weights(lats, ocean), np.float64),
             "land": np.asarray(area_weights(lats, land), np.float64),
             "global": np.asarray(area_weights(lats, np.ones_like(ocean)),
                                  np.float64)}
    table, uncontrolled = {}, {}
    for arm in ARMS:
        table[arm] = {}
        for js in sorted((Path(runs_dir) / arm).glob("ic*.json")):
            fields_path = js.with_suffix(".fields.npz")
            if not fields_path.exists():
                continue
            summary = json.loads(js.read_text())
            if summary.get("smoke"):
                raise SystemExit(f"{js} is a smoke run")
            idx = int(js.stem[2:])
            refs = np.load(Path(references_dir)
                           / f"ic{idx:04d}_references.npz")
            target = {k: refs[f"normal_{k}"] for k in
                      ("sst", "land_temperature", "precipitation")}
            with np.load(fields_path) as f:
                fields = {k: f[k] for k in ("sst", "land_temperature",
                                            "precipitation")}
            row = state_metrics(fields, target["sst"], ocean, masks["ocean"],
                                summary)
            row["members"] = summary.get("members")
            row["seconds"] = summary.get("seconds")
            if arm == UNCONTROLLED_ARM:
                uncontrolled[idx] = fields
            table[arm][idx] = (row, fields, target)
    out = {}
    for arm, rows in table.items():
        out[arm] = {}
        for idx, (row, fields, target) in rows.items():
            if idx in uncontrolled:
                row["G"] = gains(fields, uncontrolled[idx], target, masks)
            out[arm][idx] = row
    return out


def analyze(per_arm, n_boot=10000):
    """Apply the registered tests to ``{arm: {index: metrics}}``."""
    present = [arm for arm in ARMS if per_arm.get(arm)]
    states = sorted(set.intersection(*(set(per_arm[a]) for a in present)))
    macros = [i // 100 for i in states]

    def col(arm, key="J_zonal"):
        return [per_arm[arm][i][key] for i in states]

    # A hypothesis whose arm was dropped (failed twice in training) is "not
    # computed"; Holm then runs over the hypotheses that can be computed.
    computable = [h for h in HYPOTHESES if h[1] in present and h[2] in present]
    primary = [compare(col(a), col(b), macros, n_boot)
               for _, a, b, _ in computable]
    adjusted = holm([r["t_p"] for r in primary]) if primary else []
    hypotheses = {name: {"arm": a, "comparator": b, "question": question,
                         "verdict": "not computed"}
                  for name, a, b, question in HYPOTHESES}
    for (name, a, b, question), r, p in zip(computable, primary, adjusted):
        hypotheses[name] = {"arm": a, "comparator": b, "question": question,
                            **r, "t_p_holm": p, "verdict": verdict(r, p),
                            "wilcoxon_agrees": (r["wilcoxon_p"] < 0.05)
                            == (r["t_p"] < 0.05)}
    secondary = {}
    for a, b in SECONDARY:
        if a not in present or b not in present:
            secondary[f"{a}_vs_{b}"] = {"verdict_unadjusted": "not computed"}
            continue
        r = compare(col(a), col(b), macros, n_boot)
        secondary[f"{a}_vs_{b}"] = {**r, "verdict_unadjusted":
                                    verdict(r, r["t_p"])}
    vs_uncontrolled = {arm: compare(col(arm), col(UNCONTROLLED_ARM), macros,
                                    n_boot)
                       for arm in present if arm != UNCONTROLLED_ARM}
    keys = ("J_zonal", "J_map", "bias_K", "effort", "cap_share",
            "max_band_share")
    means = {arm: {k: float(np.mean(col(arm, k))) for k in keys}
             for arm in present}
    for arm in present:
        if all("G" in per_arm[arm][i] for i in states):
            for g in per_arm[arm][states[0]]["G"]:
                means[arm][f"G_{g}"] = float(np.median(
                    [per_arm[arm][i]["G"][g] for i in states]))
    return {"states": states, "n_states": len(states),
            "missing_arms": [a for a in ARMS if a not in present],
            "hypotheses": hypotheses, "secondary": secondary,
            "vs_uncontrolled": vs_uncontrolled, "arm_means": means}


def costs(designs):
    """Return the descriptive cost comparison from ``designs.json``."""
    out = {}
    for layout, c in designs["costs"].items():
        grad = c["gradient_cost_seconds"]
        per_member = c["brute_cost_seconds_per_member"]
        out[layout] = {"gradient_s": grad,
                       "brute_5_members_s": 5 * per_member,
                       "brute_equal_cost_members": c["equal_cost_members"],
                       "brute_5_over_gradient": 5 * per_member / grad}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs-dir", required=True)
    p.add_argument("--references-dir", required=True)
    p.add_argument("--designs", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--n-boot", type=int, default=10000)
    args = p.parse_args(argv)
    designs = json.loads(Path(args.designs).read_text())
    if designs.get("smoke"):
        raise SystemExit("designs.json is from a smoke run")
    per_arm = load_runs(args.runs_dir, args.references_dir)
    result = analyze(per_arm, args.n_boot)
    result["costs"] = costs(designs)
    result["created_utc"] = datetime.now(timezone.utc).isoformat()
    result["inputs"] = vars(args)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    for name, h in result["hypotheses"].items():
        if h["verdict"] == "not computed":
            print(f"{name} {h['arm']} vs {h['comparator']}: not computed")
            continue
        print(f"{name} {h['arm']} vs {h['comparator']}: rel {h['rel']:+.1%}, "
              f"p {h['t_p']:.3g} (Holm {h['t_p_holm']:.3g}) -> {h['verdict']}")
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
