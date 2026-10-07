#!/usr/bin/env python
"""Registered analysis of Experiment 3a (Amendment 9, revision 1).

MCB_PROJECT_REPORT.md Part 18 step 28. Frozen with revision 1, before any
evaluation-state data existed. The constants below are the registered
choices; changing any of them after the evaluation runs is a deviation that
must be logged.

Inputs: one folder per arm under ``--runs-dir``, each holding the run
summaries ``ic<index>.json`` with their member-mean daily fields
``ic<index>.fields.npz`` (``run_test_world.py`` / ``run_controllers.py``
with ``--save-fields``), and the evaluation references (``--references-dir``:
``grid.npz`` and ``ic<index>_references.npz``).

Per state and arm, over the scoring window, with ``e`` the time-mean SST
error against the normal-climate target (5-member mean):

* ``J_zonal`` (PRIMARY): ``alpha <e>^2 + beta Var(e)`` of the error's ocean
  zonal-mean profile (``jcm.mcb.scores.zonal_projection``), no penalties.
* ``J_map``, ``bias_K``: the same on the grid-point map, and the ocean mean.
* ``G``: Dubey et al.'s gain against the uncontrolled arm, which runs the
  same member seeds (so both sides carry the same weather noise), for SST
  (ocean), land temperature (land) and rainfall (land, global).
  Descriptive: the pilot found their signal below the noise (rule R8).
* ``effort`` (from the summary) and ``cap_share``: the mean setting over the
  window as a share of the cap.

Tests (registered). Every arm runs on the same 24 states (8 macro states x 3
branches). A comparison of arm A with comparator B uses the paired
difference ``d = J_A - J_B`` and the relative effect ``rel = mean(d) /
mean(J_B)``:

* *Unit of replication:* the macro state. Branches of one macro state share
  its ocean, so they are averaged first; the paired t test (two-sided) and
  the exact Wilcoxon signed-rank test run on the 8 macro-state means.
* *Interval:* a hierarchical bootstrap of ``rel`` (macro states, then
  branches within each; 10000 replicates, seed 2026), giving 90% and 95%
  percentile intervals.
* *Family:* the four primary hypotheses, Holm-corrected at alpha = 0.05.
* *Verdicts:* ``better`` if the Holm-adjusted t p < 0.05 and rel <= -5%;
  ``worse`` if p < 0.05 and rel >= +5%; ``equivalent`` if the 90% interval
  lies inside [-10%, +10%] (two one-sided tests at 5%); ``negligible`` if
  p < 0.05 but |rel| < 5%; otherwise ``inconclusive``. A Wilcoxon result
  that disagrees with the t test is reported next to it, not resolved.

Example (GX10):
    python analyze_experiment3a.py \
        --runs-dir mcb_experiments_gpu/exp3a/eval \
        --references-dir mcb_experiments_gpu/test_world_refs_ramp6/exp3_eval \
        --output mcb_experiments_gpu/exp3a/exp3a_analysis.json
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

from jcm.mcb.scores import (
    PATTERN_ALPHA,
    PATTERN_BETA,
    area_weights,
    pattern_objective,
    zonal_projection,
)
from jcm.mcb.test_world import BRIGHTENING_CAP

# --- Registered constants (revision 1) ---------------------------------------
SCORE_WINDOW = (98, 182)
SEGMENT_DAYS = 14
PRIMARY_ARM = "plan120"
UNCONTROLLED_ARM = "uncontrolled"
ARMS = ("uncontrolled", "plan120", "plan60", "plan14", "uniform_cancel",
        "uniform_effort", "planner_average", "linear_response", "pi",
        "adaptive")
# (name, comparator, question): PRIMARY_ARM against each, Holm family of 4.
HYPOTHESES = (
    ("H1", "plan14", "does looking 120 days ahead beat looking 14 days ahead?"),
    ("H2", "linear_response",
     "does the planner beat the best fixed pattern (classical design)?"),
    ("H3", "pi", "does the planner beat the classical feedback controller?"),
    ("H4", "planner_average",
     "does changing the pattern over time beat the planner's own average "
     "pattern held fixed?"),
)
# Secondary comparisons (A, B): reported with the same statistics, no Holm.
SECONDARY = (
    ("plan60", "plan120"), ("plan60", "plan14"),
    ("linear_response", "uniform_cancel"), ("linear_response", "pi"),
    ("adaptive", "pi"), ("planner_average", "uniform_effort"),
)
ALPHA_TEST = 0.05
REL_MIN = 0.05
EQUIVALENCE_BOUND = 0.10
N_BOOT = 10000
BOOT_SEED = 2026
# -------------------------------------------------------------------------------


def state_metrics(fields, target_sst, ocean, w_ocean, summary,
                  window=SCORE_WINDOW):
    """Return one run's metrics (no gains; those need the uncontrolled arm).

    ``ocean`` is the 0/1 ocean mask and ``w_ocean`` its area weights.
    """
    s, e = window
    err = (np.asarray(fields["sst"][s:e], np.float64).mean(0)
           - np.asarray(target_sst[s:e], np.float64).mean(0)) * ocean
    if "schedules" in summary:
        sched = np.asarray(summary["schedules"], np.float64)  # (M, n_seg, K)
    else:
        # Fixed-pattern episodes store one constant pattern, held all episode.
        sched = np.asarray(summary["amplitudes"], np.float64)[None, None, :]
        sched = np.repeat(sched, -(-e // SEGMENT_DAYS), axis=1)
    daily = np.repeat(sched, SEGMENT_DAYS, axis=1)[:, s:e]
    return {"J_zonal": float(pattern_objective(zonal_projection(err, ocean),
                                               w_ocean, PATTERN_ALPHA,
                                               PATTERN_BETA)),
            "J_map": float(pattern_objective(err, w_ocean, PATTERN_ALPHA,
                                             PATTERN_BETA)),
            "bias_K": float(np.sum(w_ocean * err)),
            "effort": float(summary["effort"]),
            "cap_share": float(daily.mean() / BRIGHTENING_CAP),
            "max_band_share": float(daily.mean(axis=(0, 1)).max()
                                    / BRIGHTENING_CAP)}



def gains(fields, uncontrolled, target, masks, window=SCORE_WINDOW):
    """Return G against the matched uncontrolled arm (time-mean fields)."""
    s, e = window
    out = {}
    for var, domains in (("sst", ("ocean",)),
                         ("land_temperature", ("land",)),
                         ("precipitation", ("land", "global"))):
        x = np.asarray(fields[var][s:e], np.float64).mean(0)
        u = np.asarray(uncontrolled[var][s:e], np.float64).mean(0)
        t = np.asarray(target[var][s:e], np.float64).mean(0)
        for d in domains:
            w = masks[d]
            den = float(np.sum(w * (x - t) ** 2))
            out[f"{var}@{d}"] = (float(np.sum(w * (u - t) ** 2)) / den
                                 if den > 0 else float("inf"))
    return out


def macro_means(values, macros):
    """Average ``values`` over the branches of each macro state."""
    keys = sorted(set(macros))
    return np.array([np.mean([v for v, m in zip(values, macros) if m == k])
                     for k in keys])


def hierarchical_rel(a, b, macros, n_boot=N_BOOT, seed=BOOT_SEED):
    """Bootstrap ``mean(a - b) / mean(b)`` over macro states, then branches."""
    rng = np.random.default_rng(seed)
    groups = {}
    for i, m in enumerate(macros):
        groups.setdefault(m, []).append(i)
    keys = sorted(groups)
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    out = np.empty(n_boot)
    for r in range(n_boot):
        idx = []
        for k in rng.choice(keys, size=len(keys), replace=True):
            members = groups[k]
            idx.extend(rng.choice(members, size=len(members), replace=True))
        idx = np.asarray(idx)
        out[r] = np.mean(a[idx] - b[idx]) / np.mean(b[idx])
    return out


def compare(a, b, macros, n_boot=N_BOOT):
    """Paired comparison of arm values ``a`` against comparator ``b``."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    da, db = macro_means(a, macros), macro_means(b, macros)
    d = da - db
    t = stats.ttest_rel(da, db)
    try:
        wil = float(stats.wilcoxon(d, method="exact").pvalue)
    except ValueError:                     # all differences zero
        wil = 1.0
    boot = hierarchical_rel(a, b, macros, n_boot)
    return {"n_states": int(a.size), "n_macro": int(d.size),
            "mean_A": float(a.mean()), "mean_B": float(b.mean()),
            "diff": float(np.mean(a - b)),
            "rel": float(np.mean(a - b) / np.mean(b)),
            "se_macro": float(d.std(ddof=1) / np.sqrt(d.size)),
            "t_p": float(t.pvalue), "wilcoxon_p": wil,
            "ci90_rel": np.percentile(boot, [5, 95]).tolist(),
            "ci95_rel": np.percentile(boot, [2.5, 97.5]).tolist()}


def holm(pvalues):
    """Return Holm-adjusted p-values in the input order."""
    p = np.asarray(pvalues, np.float64)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (p.size - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj.tolist()


def verdict(result, p_adj):
    """Apply the registered outcome grid to one comparison."""
    rel = result["rel"]
    lo, hi = result["ci90_rel"]
    if p_adj < ALPHA_TEST and rel <= -REL_MIN:
        return "better"
    if p_adj < ALPHA_TEST and rel >= REL_MIN:
        return "worse"
    if -EQUIVALENCE_BOUND < lo and hi < EQUIVALENCE_BOUND:
        return "equivalent"
    if p_adj < ALPHA_TEST:
        return "negligible"
    return "inconclusive"


def load_runs(runs_dir, references_dir):
    """Return ``{arm: {index: metrics}}`` for every complete arm and state."""
    with np.load(Path(references_dir) / "grid.npz") as g:
        lats = g["latitudes_rad"]
        ocean = np.asarray(g["ocean_mask"], np.float64)
        land = np.asarray(g["land_mask"], np.float64)
    masks = {"ocean": np.asarray(area_weights(lats, ocean), np.float64),
             "land": np.asarray(area_weights(lats, land), np.float64),
             "global": np.asarray(area_weights(lats, np.ones_like(ocean)),
                                  np.float64)}
    runs = Path(runs_dir)
    uncontrolled = {}
    table = {}
    for arm in ARMS:
        table[arm] = {}
        for js in sorted((runs / arm).glob("ic*.json")):
            idx = int(js.stem[2:])
            fields_path = js.with_suffix(".fields.npz")
            if not fields_path.exists():
                continue
            summary = json.loads(js.read_text())
            refs = np.load(Path(references_dir)
                           / f"ic{idx:04d}_references.npz")
            target = {k: refs[f"normal_{k}"] for k in
                      ("sst", "land_temperature", "precipitation")}
            with np.load(fields_path) as f:
                fields = {k: f[k] for k in ("sst", "land_temperature",
                                            "precipitation")}
            row = state_metrics(fields, target["sst"], ocean,
                                masks["ocean"], summary)
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


def analyze(per_arm, n_boot=N_BOOT):
    """Apply the registered tests to ``{arm: {index: metrics}}``."""
    complete = set.intersection(*(set(v) for v in per_arm.values() if v))
    states = sorted(complete)
    macros = [i // 100 for i in states]

    def col(arm, key="J_zonal"):
        return [per_arm[arm][i][key] for i in states]

    primary = [compare(col(PRIMARY_ARM), col(b), macros, n_boot)
               for _, b, _ in HYPOTHESES]
    adj = holm([r["t_p"] for r in primary])
    hyps = {}
    for (name, b, question), r, p in zip(HYPOTHESES, primary, adj):
        hyps[name] = {"comparator": b, "question": question, **r,
                      "t_p_holm": p, "verdict": verdict(r, p),
                      "wilcoxon_agrees": (r["wilcoxon_p"] < ALPHA_TEST)
                      == (r["t_p"] < ALPHA_TEST)}
    secondary = {}
    for a, b in SECONDARY:
        r = compare(col(a), col(b), macros, n_boot)
        secondary[f"{a}_vs_{b}"] = {**r, "verdict_unadjusted":
                                    verdict(r, r["t_p"])}
    vs_uncontrolled = {arm: compare(col(arm), col(UNCONTROLLED_ARM), macros,
                                    n_boot)
                       for arm in per_arm if arm != UNCONTROLLED_ARM}
    keys = ("J_zonal", "J_map", "bias_K", "effort", "cap_share",
            "max_band_share")
    means = {arm: {k: float(np.mean(col(arm, k))) for k in keys}
             for arm in per_arm}
    for arm in per_arm:
        if all("G" in per_arm[arm][i] for i in states):
            for g in per_arm[arm][states[0]]["G"]:
                vals = [per_arm[arm][i]["G"][g] for i in states]
                means[arm][f"G_{g}"] = float(np.median(vals))
    return {"states": states, "n_states": len(states),
            "hypotheses": hyps, "secondary": secondary,
            "vs_uncontrolled": vs_uncontrolled, "arm_means": means}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs-dir", required=True)
    p.add_argument("--references-dir", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    per_arm = load_runs(args.runs_dir, args.references_dir)
    missing = [a for a in ARMS if not per_arm.get(a)]
    if missing:
        raise SystemExit(f"arms with no runs: {missing}")
    result = analyze(per_arm)
    result.update(per_state={a: {str(i): v for i, v in rows.items()}
                             for a, rows in per_arm.items()},
                  registered={"window": SCORE_WINDOW, "primary": PRIMARY_ARM,
                              "alpha": ALPHA_TEST, "rel_min": REL_MIN,
                              "equivalence_bound": EQUIVALENCE_BOUND,
                              "n_boot": N_BOOT, "seed": BOOT_SEED},
                  created_utc=datetime.now(timezone.utc).isoformat())
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"{result['n_states']} states")
    for arm, m in sorted(result["arm_means"].items(),
                         key=lambda kv: kv[1]["J_zonal"]):
        print(f"  {arm:>16}  J_zonal {m['J_zonal']:.5f}  J_map "
              f"{m['J_map']:.4f}  bias {m['bias_K']:+.3f} K  cap "
              f"{m['cap_share']:.2f}")
    for name, h in result["hypotheses"].items():
        print(f"  {name} vs {h['comparator']:>16}: rel {h['rel']:+.1%} "
              f"(90% CI {h['ci90_rel'][0]:+.1%}..{h['ci90_rel'][1]:+.1%}), "
              f"Holm p {h['t_p_holm']:.3g} -> {h['verdict']}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
