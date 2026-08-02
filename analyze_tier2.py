#!/usr/bin/env python
"""Pre-committed Tier-2 primary analysis (PREREGISTRATION.md Amendment 3).

Committed BEFORE tier2_eval.pkl exists so the primary inference cannot be
shaped after unblinding (2026-07-31 adversarial review: the pooled-over-seeds
Holm test was registered but unimplemented, and 'pooled' was ambiguous —
concatenating 60 correlated (IC, seed) pairs would pseudo-replicate the
single static arm and inflate dof).

POOLING RULE (frozen): the unit of analysis is the IC (n = 20). For each IC
i, an arm-group's error is the mean over its seed arms of
|mean-over-members dsst_10d - target|:
    err_NN(i)  = mean_s |dsst_10d(feedback_s, i) - target|
    err_OL(i)  = mean_s |dsst_10d(openloop_s, i) - target|
    err_ST(i)  =        |dsst_10d(static, i)     - target|
PRIMARY (Holm over the two p-values, alpha = 0.05):
    H2: paired t of err_NN vs err_ST over the 20 ICs
    H3: paired t of err_NN vs err_OL over the 20 ICs
SECONDARY (reported, ungated): per-seed tests, NN vs PI, PI vs static,
G2 band per arm-group, TOST equivalence bounds, eta-stratified (eta<1 vs
eta>1) means.

Usage:
    python analyze_tier2.py mcb_experiments_gpu/tier2_eval.pkl
"""

import pickle
import sys

import numpy as np

from jcm.mcb.gates import cooling_gate
from jcm.mcb.gates_stats import paired_report


def group_err(per_ic, arm_names, target):
    """Per-IC error for an arm group: mean over arms of |dsst_10d - target|."""
    errs = np.stack([np.abs(np.asarray(per_ic[a]["dsst_10d"]) - target)
                     for a in arm_names])
    return errs.mean(axis=0)


def holm(pvals):
    """Holm step-down adjusted p-values (dict name -> p)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted, running_max = {}, 0.0
    for rank, (name, p) in enumerate(items):
        adj = min(1.0, (m - rank) * p)
        running_max = max(running_max, adj)
        adjusted[name] = running_max
    return adjusted


def analyze(results, target=-0.1, band=(-0.12, -0.08)):
    per_ic = results["per_ic"]
    arms = sorted(per_ic.keys())
    feedback_arms = sorted(a for a in arms if a.startswith("feedback"))
    openloop_arms = sorted(a for a in arms if a.startswith("openloop"))
    assert "static" in arms, arms
    assert feedback_arms and openloop_arms, arms

    err_nn = group_err(per_ic, feedback_arms, target)
    err_ol = group_err(per_ic, openloop_arms, target)
    err_st = group_err(per_ic, ["static"], target)

    h2 = paired_report(err_nn, err_st)
    h3 = paired_report(err_nn, err_ol)
    adj = holm({"H2_nn_vs_static": h2["p_t"], "H3_nn_vs_openloop": h3["p_t"]})

    out = {
        "pooling_rule": "per-IC mean over seed arms; n = num ICs",
        "n_ics": int(err_nn.size),
        "feedback_arms": feedback_arms,
        "openloop_arms": openloop_arms,
        "primary": {
            "H2_nn_vs_static": {**h2, "p_holm": adj["H2_nn_vs_static"],
                                "significant_holm":
                                    bool(adj["H2_nn_vs_static"] < 0.05
                                         and h2["mean"] < 0)},
            "H3_nn_vs_openloop": {**h3, "p_holm": adj["H3_nn_vs_openloop"],
                                  "significant_holm":
                                      bool(adj["H3_nn_vs_openloop"] < 0.05
                                           and h3["mean"] < 0)},
        },
        "secondary": {},
    }

    # Secondary: per-seed primaries, PI comparisons, G2, eta stratification.
    for a in feedback_arms:
        e = np.abs(np.asarray(per_ic[a]["dsst_10d"]) - target)
        out["secondary"][f"per_seed[{a} vs static]"] = paired_report(e, err_st)
    if "pi" in arms:
        err_pi = group_err(per_ic, ["pi"], target)
        out["secondary"]["nn_vs_pi"] = paired_report(err_nn, err_pi)
        out["secondary"]["pi_vs_static"] = paired_report(err_pi, err_st)
    for name, group in [("static", ["static"]),
                        ("feedback(pooled)", feedback_arms),
                        ("openloop(pooled)", openloop_arms)] + (
                            [("pi", ["pi"])] if "pi" in arms else []):
        dsst = np.stack([np.asarray(per_ic[a]["dsst_10d"])
                         for a in group]).mean(axis=0)
        out["secondary"][f"G2[{name}]"] = cooling_gate(dsst, band=band)
    etas = np.asarray(results.get("efficacies", []))
    if etas.size == err_nn.size:
        for label, mask in [("eta<1", etas < 1.0), ("eta>=1", etas >= 1.0)]:
            if mask.sum() >= 2:
                out["secondary"][f"stratified[{label}]"] = {
                    "n": int(mask.sum()),
                    "err_static": float(err_st[mask].mean()),
                    "err_nn": float(err_nn[mask].mean()),
                    "err_ol": float(err_ol[mask].mean()),
                }
    return out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "mcb_experiments_gpu/tier2_eval.pkl"
    with open(path, "rb") as f:
        results = pickle.load(f)
    target = results["config"].get("target_cooling", -0.1)
    band = tuple(results["config"].get("band", (-0.12, -0.08)))
    out = analyze(results, target=target, band=band)

    print("=" * 72)
    print("TIER-2 PRIMARY ANALYSIS (Amendment 3, pre-committed)")
    print("=" * 72)
    print(f"n_ics={out['n_ics']} | pooling: {out['pooling_rule']}")
    for name, rep in out["primary"].items():
        print(f"\n{name}: mean diff {rep['mean']:+.5f} +/- {rep['se']:.5f}")
        print(f"  p_t={rep['p_t']:.4f}  p_holm={rep['p_holm']:.4f}  "
              f"p_wilcoxon={rep['p_wilcoxon']:.4f}  "
              f"SIGNIFICANT(Holm)={rep['significant_holm']}")
        print(f"  TOST equivalence bound: |diff| < "
              f"{rep['equivalence_bound_95']:.5f} @95%")
    print("\nSECONDARY:")
    for name, rep in out["secondary"].items():
        if "p_t" in rep:
            print(f"  {name}: mean {rep['mean']:+.5f} p_t={rep['p_t']:.4f} "
                  f"|equiv|<{rep['equivalence_bound_95']:.5f}")
        elif "verdict" in rep:
            print(f"  {name}: {rep['mean']:+.4f} +/- {rep['se']:.4f} "
                  f"[{rep['verdict']}]")
        else:
            print(f"  {name}: {rep}")

    out_path = path.replace(".pkl", "_analysis.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
