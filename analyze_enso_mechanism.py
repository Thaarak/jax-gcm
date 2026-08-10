#!/usr/bin/env python
"""POST-HOC mechanism analysis for the ENSO campaign (exploratory).

NOT the pre-registered analysis. `analyze_enso.py` is the frozen, pre-committed
primary analysis (Amendment 6) and is never edited after the freeze; this
script adds the diagnostics that interpretation requires but that were not
pre-specified, and every number it prints is EXPLORATORY.

Three questions it answers:

1. DISTURBANCE REJECTION (the physically meaningful effect size). Regress each
   arm's signed per-IC error on the hidden amplitude A. The slope is the arm's
   sensitivity to the disturbance in K per K of Nino3.4 anomaly. Feedback should
   flatten it. Crucially this measure is INVARIANT to any constant calibration
   offset shared by the arms, so it is immune to the deficit documented below.

2. CALIBRATION DECOMPOSITION. The G-cal gate measured the static pattern with
   ENSO OFF; its distance from the target is a calibration deficit shared by all
   arms (the 180-day rescale factor was derived from a different base state).
   Split each arm's error into that shared deficit plus the ENSO-driven part,
   and re-run the primary comparisons on debiased errors as a sensitivity check:
   if the conclusions flip when the shared offset is removed, they were an
   artifact of miscalibration rather than a fact about feedback.

3. RESIDUAL BUDGET. After removing the A-dependence, is the leftover per-IC
   scatter just chaos? Compare it with the micro-ensemble standard error
   (member sd / sqrt(k)). Scatter at the chaos level means the controller has
   removed essentially all of the disturbance-driven variance.

Usage:
    python analyze_enso_mechanism.py \
        --eval mcb_experiments_gpu/enso_eval.pkl \
        --gcal mcb_experiments_gpu/enso_gcal.pkl
"""

import argparse
import pickle

import numpy as np

from jcm.mcb.gates_stats import paired_report


def pool(cells, arm_names, key="dsst_10d"):
    """Per-IC values: mean over members, then over the named arms (seeds)."""
    return np.stack([np.asarray(cells[a][key]).mean(axis=1)
                     for a in arm_names]).mean(axis=0)


def linregress(x, y):
    """Least-squares slope/intercept with slope se, r and two-sided p.

    Pure NumPy (SciPy is not a project dependency); the t-distribution tail
    is taken from the repo's own gates_stats.t_sf, keeping every p-value in
    this project on one implementation.
    """
    from jcm.mcb.gates_stats import t_sf
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = x.size
    xm, ym = x.mean(), y.mean()
    sxx = np.sum((x - xm) ** 2)
    slope = np.sum((x - xm) * (y - ym)) / sxx
    intercept = ym - slope * xm
    resid = y - (intercept + slope * x)
    dof = n - 2
    s2 = np.sum(resid ** 2) / dof
    se = np.sqrt(s2 / sxx)
    t = slope / se if se > 0 else np.inf
    p = 2.0 * t_sf(abs(t), dof)
    r = (np.corrcoef(x, y)[0, 1] if np.std(x) > 0 and np.std(y) > 0
         else float("nan"))
    return {"slope": float(slope), "intercept": float(intercept),
            "se": float(se), "t": float(t), "p": float(p), "r": float(r),
            "resid_sd": float(resid.std(ddof=2))}


def analyze(results, gcal=None, target=-0.1):
    cells = results["cells"]
    amps = np.asarray(results["enso_amps"], float)
    arms = list(cells)
    imit = sorted(a for a in arms if a.startswith("imitation"))
    groups = {a: [a] for a in arms}
    if imit:
        groups["imitation(pooled)"] = imit

    k = np.asarray(cells[arms[0]]["dsst_10d"]).shape[1]
    out = {"n_ics": int(amps.size), "k": int(k), "target": target,
           "mean_amplitude": float(amps.mean()), "arms": {}}

    # Shared calibration deficit from the ENSO-off gate.
    bias = None
    if gcal is not None:
        cal = float(np.asarray(
            gcal["per_ic"]["static"]["dsst_10d"]).mean())
        bias = cal - target
        out["calibration"] = {
            "gcal_static_dsst": cal,
            "deficit_mK": bias * 1000,
            "note": "measured with ENSO OFF; shared by every arm",
        }

    per_ic = {}
    for name, members in groups.items():
        d = pool(cells, members)
        per_ic[name] = d
        err = d - target
        reg = linregress(amps, err)
        force = pool(cells, members, "mean_mcb_forcing")
        freg = (linregress(amps, force) if np.std(force) > 1e-12
                else {"slope": 0.0, "r": float("nan"), "p": float("nan"),
                      "se": 0.0, "t": 0.0, "intercept": float(force[0]),
                      "resid_sd": 0.0})
        member_sd = np.mean([np.asarray(cells[a]["dsst_10d"]).std(
            axis=1, ddof=1) for a in members])
        out["arms"][name] = {
            "mean_dsst": float(d.mean()),
            "signed_err_mK": float(err.mean() * 1000),
            "abs_err_mK": float(np.abs(err).mean() * 1000),
            "abs_err_debiased_mK": (float(np.abs(err - bias).mean() * 1000)
                                    if bias is not None else None),
            "per_ic_sd_mK": float(d.std(ddof=1) * 1000),
            "sensitivity_mK_per_K": reg["slope"] * 1000,
            "sensitivity_se_mK_per_K": reg["se"] * 1000,
            "sensitivity_p": reg["p"],
            "residual_sd_after_A_mK": reg["resid_sd"] * 1000,
            "chaos_se_of_ic_mean_mK": float(member_sd / np.sqrt(k) * 1000),
            "forcing_mean": float(force.mean()),
            "forcing_vs_A_r": freg["r"],
            "forcing_vs_A_p": freg["p"],
            "adapts": bool(np.std(force) > 1e-12),
        }

    # Sensitivity check: do the primary verdicts survive debiasing?
    if bias is not None and "static" in per_ic:
        out["debiased_comparisons"] = {}
        pairs = [("pienso", "static"), ("pienso", "ffmean"),
                 ("ffmean", "static"), ("imitation(pooled)", "static"),
                 ("imitation(pooled)", "pienso")]
        for a, b in pairs:
            if a in per_ic and b in per_ic:
                ea = np.abs(per_ic[a] - target - bias)
                eb = np.abs(per_ic[b] - target - bias)
                out["debiased_comparisons"][f"{a} vs {b}"] = paired_report(
                    ea, eb)

    # Disturbance-rejection summary vs the ENSO-blind reference.
    if "static" in out["arms"]:
        s0 = out["arms"]["static"]["sensitivity_mK_per_K"]
        for name, a in out["arms"].items():
            a["sensitivity_reduction_pct"] = (
                100.0 * (1.0 - a["sensitivity_mK_per_K"] / s0)
                if s0 != 0 else float("nan"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="mcb_experiments_gpu/enso_eval.pkl")
    ap.add_argument("--gcal", default="mcb_experiments_gpu/enso_gcal.pkl")
    ap.add_argument("--target", type=float, default=-0.1)
    args = ap.parse_args()

    with open(args.eval, "rb") as f:
        results = pickle.load(f)
    gcal = None
    if args.gcal and args.gcal != "none":
        with open(args.gcal, "rb") as f:
            gcal = pickle.load(f)

    out = analyze(results, gcal, target=args.target)

    print("=" * 78)
    print("ENSO MECHANISM ANALYSIS — POST-HOC / EXPLORATORY (not "
          "pre-registered)")
    print("=" * 78)
    print(f"n = {out['n_ics']} ICs, k = {out['k']}, target {out['target']} K, "
          f"mean |A| = {out['mean_amplitude']:.2f} K")
    if "calibration" in out:
        c = out["calibration"]
        print(f"\nShared calibration deficit (G-cal, ENSO OFF): static reached "
              f"{c['gcal_static_dsst']:+.4f} K -> {c['deficit_mK']:+.1f} mK "
              f"short of target; every arm inherits it.")

    print("\nPer-arm error and disturbance sensitivity:")
    print(f"  {'arm':>18} {'|err|':>7} {'debias':>7} {'sens':>16} "
          f"{'reduction':>10} {'resid':>7} {'chaos':>7} {'cmd~A':>7}")
    for name, a in out["arms"].items():
        deb = (f"{a['abs_err_debiased_mK']:6.1f}"
               if a["abs_err_debiased_mK"] is not None else "     -")
        r = a["forcing_vs_A_r"]
        print(f"  {name:>18} {a['abs_err_mK']:6.1f}  {deb} "
              f"{a['sensitivity_mK_per_K']:+7.1f}+/-"
              f"{a['sensitivity_se_mK_per_K']:4.1f} "
              f"{a['sensitivity_reduction_pct']:9.0f}% "
              f"{a['residual_sd_after_A_mK']:6.1f} "
              f"{a['chaos_se_of_ic_mean_mK']:6.1f} "
              f"{'   n/a' if not a['adapts'] else f'{r:+6.2f}'}")
    print("  (|err|,debias,resid,chaos in mK; sens in mK per K of Nino3.4; "
          "cmd~A = corr(commanded forcing, hidden amplitude))")

    if "debiased_comparisons" in out:
        print("\nSensitivity check — primary comparisons on DEBIASED errors:")
        for name, rep in out["debiased_comparisons"].items():
            sig = "SIGNIFICANT" if rep["significant"] else "n.s."
            print(f"  {name:>30}: {rep['mean'] * 1000:+7.1f} mK "
                  f"(se {rep['se'] * 1000:.1f}, p={rep['p_t']:.3g}) {sig} "
                  f"|eq|<{rep['equivalence_bound_95'] * 1000:.1f} mK")

    out_path = args.eval.replace(".pkl", "_mechanism.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    print(f"\nmechanism analysis -> {out_path}")


if __name__ == "__main__":
    main()
