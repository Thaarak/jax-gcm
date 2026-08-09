#!/usr/bin/env python
"""Pre-committed primary analysis for the ENSO campaign (Amendment 6).

Frozen BEFORE any campaign data exists (the Tier-2/2b discipline). Reads
the run_confirmatory_eval.py output pickle from the ENSO evaluation and
scores the pre-registered hypotheses on the registered metric
(final-60-day time-mean dSST vs the member's NO-ENSO baseline).

Primary (Holm over the pair, direction-gated):
  H6: the classical pi-enso controller beats the rescaled static pattern.
  H7: the imitation-fc14 controller (pooled seeds) beats static.
Secondary (reported with TOST bounds where null):
  imitation vs pi-enso (does learning match/exceed the hand law?);
  pi-enso vs ff-mean (feedback beats mis-specified mean-feedforward);
  ff-mean vs static (mean compensation helps at all);
  per-seed imitation vs pi-enso; amplitude-stratified H7 (A >= median).

Usage:
    python analyze_enso.py mcb_experiments_gpu/enso_eval.pkl
"""

import pickle
import sys

import numpy as np

from analyze_tier2 import holm
from jcm.mcb.gates_stats import paired_report


def group_err(per_ic, arm_names, target):
    """Per-IC |dsst_10d - target|, averaged over an arm group (seeds)."""
    errs = np.stack([np.abs(np.asarray(per_ic[a]["dsst_10d"]) - target)
                     for a in arm_names])
    return errs.mean(axis=0)


def analyze(results, target=-0.1):
    per_ic = results["per_ic"]
    arms = sorted(per_ic.keys())
    imit_arms = sorted(a for a in arms if a.startswith("imitation"))
    assert "static" in arms and "pienso" in arms, arms

    err_st = group_err(per_ic, ["static"], target)
    err_pi = group_err(per_ic, ["pienso"], target)
    h6 = paired_report(err_pi, err_st)
    out = {
        "n_ics": int(err_st.size),
        "imitation_arms": imit_arms,
        "arm_mean_err_mK": {
            a: float(group_err(per_ic, [a], target).mean() * 1000)
            for a in arms},
        "primary": {},
        "secondary": {},
    }

    if imit_arms:
        err_im = group_err(per_ic, imit_arms, target)
        h7 = paired_report(err_im, err_st)
        adj = holm({"H6_pienso_vs_static": h6["p_t"],
                    "H7_imitation_vs_static": h7["p_t"]})
        out["primary"]["H6_pienso_vs_static"] = {
            **h6, "p_holm": adj["H6_pienso_vs_static"],
            "significant_holm": bool(adj["H6_pienso_vs_static"] < 0.05
                                     and h6["mean"] < 0)}
        out["primary"]["H7_imitation_vs_static"] = {
            **h7, "p_holm": adj["H7_imitation_vs_static"],
            "significant_holm": bool(adj["H7_imitation_vs_static"] < 0.05
                                     and h7["mean"] < 0)}
        out["secondary"]["imitation_vs_pienso"] = paired_report(err_im,
                                                                err_pi)
        for a in imit_arms:
            out["secondary"][f"per_seed[{a} vs pienso]"] = paired_report(
                group_err(per_ic, [a], target), err_pi)
    else:
        out["primary"]["H6_pienso_vs_static"] = {
            **h6, "p_holm": h6["p_t"],
            "significant_holm": bool(h6["p_t"] < 0.05 and h6["mean"] < 0)}

    if "ffmean" in arms:
        err_ff = group_err(per_ic, ["ffmean"], target)
        out["secondary"]["pienso_vs_ffmean"] = paired_report(err_pi, err_ff)
        out["secondary"]["ffmean_vs_static"] = paired_report(err_ff, err_st)

    amps = np.asarray(results.get("enso_amps", []))
    if amps.size == err_st.size and imit_arms:
        hi = amps >= np.median(amps)
        if hi.sum() >= 3 and (~hi).sum() >= 3:
            out["secondary"]["stratified_H7[A>=median]"] = paired_report(
                err_im[hi], err_st[hi])
            out["secondary"]["stratified_H7[A<median]"] = paired_report(
                err_im[~hi], err_st[~hi])
    return out


def fmt(name, rep, extra=""):
    sig = rep.get("significant_holm", rep.get("significant", False))
    p_extra = (f" p_holm={rep['p_holm']:.4g}" if "p_holm" in rep else "")
    return (f"  {name:>34}: {rep['mean'] * 1000:+7.1f} mK "
            f"(se {rep['se'] * 1000:.1f}, p_t={rep['p_t']:.4g}{p_extra}, "
            f"p_w={rep['p_wilcoxon']:.4g}) "
            f"{'SIGNIFICANT' if sig else 'n.s.'} "
            f"|eq|<{rep['equivalence_bound_95'] * 1000:.1f} mK{extra}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "mcb_experiments_gpu/enso_eval.pkl"
    with open(path, "rb") as f:
        results = pickle.load(f)
    target = float(results["config"].get("target_cooling", -0.1))
    out = analyze(results, target=target)

    print("=" * 76)
    print(f"ENSO CAMPAIGN PRIMARY ANALYSIS (n={out['n_ics']} ICs, "
          f"target {target} K)")
    print("=" * 76)
    print("Arm mean |miss| (mK):")
    for a, v in sorted(out["arm_mean_err_mK"].items(),
                       key=lambda kv: kv[1]):
        print(f"  {a:>16}: {v:6.1f}")
    print("\nPrimary (Holm, direction-gated):")
    for name, rep in out["primary"].items():
        print(fmt(name, rep))
    print("\nSecondary:")
    for name, rep in out["secondary"].items():
        print(fmt(name, rep))

    out_path = path.replace(".pkl", "_analysis.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    print(f"\nanalysis -> {out_path}")


if __name__ == "__main__":
    main()
