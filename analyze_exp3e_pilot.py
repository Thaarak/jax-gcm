#!/usr/bin/env python
"""Experiment 3e's pilot rules (Amendment 9 revision 4; MCB_PROJECT_REPORT.md Part 29).

The pilot runs the oracle planner (told the true strength) with analysis
sensing at each level of revision 4 (``--analysis-error`` 0.1, 0.3 and 1.0)
on 3d's 16 branch-1 training states, 3 members each, in the fixed-land model.
Its rules were written before it ran:

``noise_k`` (per level)
    The learner's noise level: 3b's rule (``analyze_exp3b_pilot.noise_rule``),
    unchanged, on that level's oracle runs.

``kappa`` (per level)
    The scale of revision 4's detection statistic. For every forecast miss
    ``r`` of the oracle (pure forecast error, since it knows the strength),
    with ``g`` the forecast's sensitivity to one strength factor shared by all
    bands (the sum of the bands' sensitivities) and ``W`` the planner's
    latitude weights, ``kappa^2 = median((g'Wr)^2 / (g'Wg)) / 0.4549``, where
    0.4549 is the median of a chi-square variable with one degree of freedom.
    Then, over a run's re-plans, ``Z = sum(g'Wz) / (kappa sqrt(sum(g'Wg)))``
    with ``z = miss + G belief`` is about standard normal when brightening
    does nothing. ``z_null_sd`` checks this on the pilot's own misses.

Also recorded (descriptive): how far each level's background had drifted from
the true weather when it was corrected (the median root-mean-square
difference of each atmosphere and land state array, on day 0 and later), the
oracle's own latitude-pattern score, and the noise curve with 3d's
exact-sensing level.

Example (GX10):
    python analyze_exp3e_pilot.py --pilot-dir mcb_experiments_gpu/exp3e/pilot
Writes ``summary.json`` in the pilot folder.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from analyze_exp3b_pilot import noise_rule

# --- Registered constants (revision 4) -----------------------------------------
LEVELS = {"a10": 0.1, "a30": 0.3, "a100": 1.0}
CHI2_1_MEDIAN = 0.4549364231195724        # scipy.stats.chi2(1).median()
EXACT_NOISE_K = 0.00069104                # 3d's oracle pilot (exact sensing)
# -------------------------------------------------------------------------------


def detection_terms(path):
    """Return ``(g'Wz, g'Wg, g'Wr)``, each ``(members, re-plans)``, from one innovations file."""
    with np.load(path) as f:
        obs = f["observed"].astype(np.float64)
        pred = f["predicted"].astype(np.float64)
        sens = f["sensitivity"].astype(np.float64)
        belief = f["belief"].astype(np.float64)
        w = f["profile_weights"].astype(np.float64)
    r = obs - pred                                   # (members, n, il)
    g = sens.sum(axis=-1)                            # one shared factor
    z = r + np.einsum("mnik,mnk->mni", sens, belief)
    return (w * g * z).sum(-1), (w * g * g).sum(-1), (w * g * r).sum(-1)


def kappa(paths):
    """Return the detection scale and the number of misses it comes from."""
    ratios = []
    for p in paths:
        _, v, n = detection_terms(p)
        ratios.append((n * n / v).ravel())
    ratios = np.concatenate(ratios)
    return float(np.sqrt(np.median(ratios) / CHI2_1_MEDIAN)), int(ratios.size)


def detection_z(u, v, scale):
    """Return the cumulative detection statistic over re-plans (the last axis)."""
    return np.cumsum(u, axis=-1) / (scale * np.sqrt(np.cumsum(v, axis=-1)))


def z_null_sd(paths, scale):
    """Spread of the statistic built from pure forecast error at the last re-plan."""
    finals = []
    for p in paths:
        _, v, n = detection_terms(p)
        finals.append(detection_z(n, v, scale)[..., -1].ravel())
    return float(np.std(np.concatenate(finals), ddof=1))


def drift(runs):
    """Median background drift per state array: on day 0, and at later re-plans."""
    day0, later = {}, {}
    for js in runs:
        for log in json.loads(js.read_text())["planner_logs"]:
            for record in log:
                d = record.get("background_error")
                if not d:
                    continue
                into = day0 if record["segment"] == 0 else later
                for k, value in d.items():
                    into.setdefault(k, []).append(value)
    return ({k: float(np.median(v)) for k, v in sorted(day0.items())},
            {k: float(np.median(v)) for k, v in sorted(later.items())})


def summarize(pilot_dir, levels=LEVELS, exact_noise=EXACT_NOISE_K):
    """Apply the pilot's rules to every level that has runs."""
    out = {}
    for name, error in levels.items():
        folder = Path(pilot_dir) / name
        runs = sorted((folder / "plan_oracle").glob("ic*.json"))
        if not runs:
            continue
        innovations = sorted((folder / "plan_oracle").glob("ic*.innovations.npz"))
        scale, n_scale = kappa(innovations) if innovations else (None, 0)
        day0, later = drift(runs)
        scores = [json.loads(js.read_text())["objective"]["controlled"]
                  ["zonal_pattern"] for js in runs]
        out[name] = {"analysis_error": error, **noise_rule(folder),
                     "kappa": scale, "kappa_n": n_scale,
                     "z_null_sd": (z_null_sd(innovations, scale)
                                   if innovations else None),
                     "runs": len(runs), "drift_day0": day0,
                     "drift_later": later,
                     "oracle_zonal_pattern_mean": float(np.mean(scores))}
    return {"levels": out, "exact_noise_k": exact_noise,
            "chi2_1_median": CHI2_1_MEDIAN,
            "created_utc": datetime.now(timezone.utc).isoformat()}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--pilot-dir", required=True)
    p.add_argument("--levels", nargs="+", default=sorted(LEVELS),
                   choices=sorted(LEVELS))
    args = p.parse_args(argv)
    result = summarize(args.pilot_dir, {k: LEVELS[k] for k in args.levels})
    if not result["levels"]:
        raise SystemExit(f"no pilot runs in {args.pilot_dir}")
    print(f"  {'level':>5} {'f':>5} {'noise_k (K)':>12} {'x exact':>8} "
          f"{'kappa':>10} {'Z null sd':>9} {'misses':>7} {'oracle J':>9}")
    for name, r in sorted(result["levels"].items(),
                          key=lambda kv: kv[1]["analysis_error"]):
        print(f"  {name:>5} {r['analysis_error']:5.2f} {r['noise_k']:12.5f} "
              f"{r['noise_k'] / EXACT_NOISE_K:8.1f} "
              f"{r['kappa'] if r['kappa'] is not None else float('nan'):10.3g} "
              f"{r['z_null_sd'] if r['z_null_sd'] is not None else float('nan'):9.2f} "
              f"{r['n']:7d} {r['oracle_zonal_pattern_mean']:9.5f}")
    out = Path(args.pilot_dir) / "summary.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
