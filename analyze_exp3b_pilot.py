#!/usr/bin/env python
"""The Experiment 3b pilot's fixed rules (MCB_PROJECT_REPORT.md Part 24).

The pilot runs on training states only (``exp3b_train`` branch 1, one state
per training macro state, each with its hidden strength). Its rules were
written before it ran:

``noise``
    The learner's noise level. From the oracle planner's runs (the true
    strength known), every re-plan after the first logs the weighted mean
    square of its forecast miss on the latitude profile (``innovation_ms``).
    ``noise_k = sqrt(median)`` over all members and segments. It is a
    measurement, not tuned on performance.

``decide``
    * Defaults are replaced only by a clearly better variant, with the same
      rule for our method and the opponent: a variant replaces its default
      if its mean ``J_zonal`` is lower by more than 2 standard errors of the
      paired difference over the pilot states. Pairs: ``pi_slow`` (84-day
      loop) vs ``pi`` (42); ``plan_learn_cautious`` (4x noise) vs
      ``plan_learn``; ``plan_learn_global`` (one shared factor) vs
      ``plan_learn``.
    * Members per evaluation run: the smallest of 6, 8 and 10 for which the
      minimum detectable effect of H1 (80% power, two-sided 5%, 16 macro
      states x 3 branches, macro means) is at most 20% of the pilot's mean
      ``J_zonal`` of the classical controller. Otherwise 10. Noise at M
      members comes from each member's saved latitude profile: the
      variance over states of the paired difference, computed with 1, 2 and
      3 members, is fitted by ``A + B / M`` (A >= 0) and extrapolated. A
      between-macro spread of the effect of 0.0005 (the largest seen in
      Experiment 3a, rounded up) is added for safety.

Writes ``noise.json`` or ``decisions.json`` next to the runs.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats

from analyze_experiment3a import compare
from analyze_experiment3b import load_runs
from jcm.mcb.scores import PATTERN_ALPHA, PATTERN_BETA, area_weights

PILOT_ARMS = ("uncontrolled", "fixed", "pi", "pi_slow", "adaptive",
              "plan_naive", "plan_oracle", "plan_learn", "plan_learn_cautious",
              "plan_learn_global")
SWITCHES = (("pi_slow", "pi"), ("plan_learn_cautious", "plan_learn"),
            ("plan_learn_global", "plan_learn"))
SWITCH_SE = 2.0
MEMBER_CHOICES = (6, 8, 10)
MDE_FRACTION = 0.20
EVAL_MACROS, EVAL_BRANCHES = 16, 3
TAU_MACRO = 0.0005
WINDOW = (98, 182)


def noise_rule(runs_dir):
    """Return ``noise_k`` and its inputs from the oracle's forecast misses."""
    ms = []
    for js in sorted((Path(runs_dir) / "plan_oracle").glob("ic*.json")):
        for log in json.loads(js.read_text())["planner_logs"]:
            ms += [r["innovation_ms"] for r in log if "innovation_ms" in r]
    if not ms:
        raise SystemExit("no oracle forecast misses found")
    ms = np.asarray(ms)
    return {"noise_k": float(np.sqrt(np.median(ms))), "n": int(ms.size),
            "ms_quartiles": np.percentile(ms, [25, 50, 75]).tolist()}


def profile_j(member_profiles, target_profile, lat_w, members):
    """``J_zonal`` of the mean of the chosen members' window-mean profiles."""
    s, e = WINDOW
    z = (member_profiles[list(members), s:e].mean(axis=(0, 1))
         - target_profile)
    mean = float(np.sum(lat_w * z))
    return PATTERN_ALPHA * mean ** 2 + PATTERN_BETA * float(
        np.sum(lat_w * (z - mean) ** 2))


def difference_variance(runs_dir, refs_dir, arm_a, arm_b):
    """Variance over states of ``J_A - J_B`` with 1..M members (matched seeds)."""
    with np.load(Path(refs_dir) / "grid.npz") as g:
        lats, ocean = g["latitudes_rad"], np.asarray(g["ocean_mask"], float)
    # The score's own ocean area weights, summed over each latitude.
    lat_w = np.asarray(area_weights(lats, ocean), float).sum(axis=0)
    counts = np.maximum(ocean.sum(axis=0), 1.0)
    per_m = {}
    for js in sorted((Path(runs_dir) / arm_a).glob("ic*.json")):
        idx = js.stem[2:]
        fa = js.with_suffix(".fields.npz")
        fb = Path(runs_dir) / arm_b / f"ic{idx}.fields.npz"
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
    # Per subset size: the variance over states, averaged over the subsets.
    return {m: float(np.mean([np.var(v, ddof=1) for v in subsets.values()]))
            for m, subsets in per_m.items()}


def fit_noise(var_by_m):
    """Fit ``Var(M) = A + B / M`` with ``A >= 0`` by least squares."""
    m = np.array(sorted(var_by_m), float)
    v = np.array([var_by_m[k] for k in sorted(var_by_m)])
    design = np.stack([np.ones_like(m), 1.0 / m], axis=1)
    coef, *_ = np.linalg.lstsq(design, v, rcond=None)
    if coef[0] < 0:
        coef = np.array([0.0, float(np.sum(v / m) / np.sum(1.0 / m ** 2))])
    return {"A": float(coef[0]), "B": float(coef[1])}


def minimum_detectable_effect(var_state, n_macro=EVAL_MACROS,
                              branches=EVAL_BRANCHES, tau=TAU_MACRO,
                              alpha=0.05, power=0.8):
    """MDE of a paired t test on macro means (two-sided)."""
    sd_macro = np.sqrt(var_state / branches + tau ** 2)
    df = n_macro - 1
    t = stats.t.ppf(1 - alpha / 2, df) + stats.t.ppf(power, df)
    return float(t * sd_macro / np.sqrt(n_macro))


def decide(runs_dir, refs_dir):
    """Apply the registered pilot rules; return the decisions."""
    per_arm = load_runs(runs_dir, [refs_dir], arms=PILOT_ARMS)
    states = sorted(set.intersection(*(set(v) for v in per_arm.values()
                                       if v)))
    macros = [i // 100 for i in states]

    def col(arm, key="J_zonal"):
        return np.array([per_arm[arm][i][key] for i in states])

    means = {arm: {k: float(np.mean(col(arm, k))) for k in
                   ("J_zonal", "J_bias_term", "J_pattern_term", "bias_K",
                    "cap_share")} for arm in PILOT_ARMS if per_arm.get(arm)}
    switches = {}
    for variant, default in SWITCHES:
        d = col(variant) - col(default)
        se = float(d.std(ddof=1) / np.sqrt(d.size))
        switches[variant] = {"default": default, "mean_diff": float(d.mean()),
                             "se": se, "switch": bool(d.mean()
                                                      < -SWITCH_SE * se)}
    learn = ("plan_learn_cautious" if switches["plan_learn_cautious"]["switch"]
             else "plan_learn")
    if switches["plan_learn_global"]["switch"] and \
            np.mean(col("plan_learn_global")) < np.mean(col(learn)):
        learn = "plan_learn_global"
    pi = "pi_slow" if switches["pi_slow"]["switch"] else "pi"
    var_by_m = difference_variance(runs_dir, refs_dir, learn, pi)
    fit = fit_noise(var_by_m)
    j_pi = float(np.mean(col(pi)))
    power = {}
    for m in MEMBER_CHOICES:
        var_m = fit["A"] + fit["B"] / m
        mde = minimum_detectable_effect(var_m)
        power[m] = {"var_state": var_m, "mde": mde,
                    "mde_fraction": mde / j_pi}
    members = next((m for m in MEMBER_CHOICES
                    if power[m]["mde_fraction"] <= MDE_FRACTION),
                   MEMBER_CHOICES[-1])
    effects = {name: compare(col(a), col(b), macros, n_boot=2000)
               for name, a, b in (("H1", learn, pi),
                                  ("H2", learn, "plan_naive"),
                                  ("H3", pi, "fixed"),
                                  ("naive_vs_oracle", "plan_naive",
                                   "plan_oracle"),
                                  ("learn_vs_oracle", learn, "plan_oracle"))}
    curves = [per_arm[learn][i].get("learning_curve") for i in states]
    curves = [c for c in curves if c]
    return {"states": states, "arm_means": means, "switches": switches,
            "chosen": {"learner": learn, "pi": pi, "members": members},
            "noise_by_members": var_by_m, "noise_fit": fit,
            "power": power, "pilot_effects": effects,
            "learning_curve": (np.mean(curves, axis=0).tolist()
                               if curves else None)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("stage", choices=("noise", "decide"))
    p.add_argument("--runs-dir", required=True)
    p.add_argument("--references-dir", required=True)
    args = p.parse_args(argv)
    if args.stage == "noise":
        result = noise_rule(args.runs_dir)
        out = Path(args.runs_dir) / "noise.json"
        print(f"noise_k = {result['noise_k']:.5f} K from {result['n']} "
              f"forecast misses")
    else:
        result = decide(args.runs_dir, args.references_dir)
        out = Path(args.runs_dir) / "decisions.json"
        for arm, m in sorted(result["arm_means"].items(),
                             key=lambda kv: kv[1]["J_zonal"]):
            print(f"  {arm:>20}  J_zonal {m['J_zonal']:.5f}  bias "
                  f"{m['bias_K']:+.4f} K  cap {m['cap_share']:.2f}")
        for v, s in result["switches"].items():
            print(f"  {v} vs {s['default']}: {s['mean_diff']:+.5f} "
                  f"(SE {s['se']:.5f}) -> {'switch' if s['switch'] else 'keep'}")
        for m, pw in result["power"].items():
            print(f"  M={m}: MDE {pw['mde']:.5f} = {pw['mde_fraction']:.0%} "
                  f"of J_pi")
        print(f"  chosen: {result['chosen']}")
    out.write_text(json.dumps(result, indent=2, default=float))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
