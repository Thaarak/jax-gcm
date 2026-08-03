#!/usr/bin/env python
"""Pre-committed Tier-2b primary analysis (PREREGISTRATION.md Amendment 4).

Committed BEFORE tier2b_eval.pkl exists. Pooling rule as Amendment 3: the
unit of analysis is the IC (n = 20); the fine-tuned group's per-IC error is
the mean over the 3 fine-tuned seed arms of |dsst_10d - target|.

PRIMARY (Holm over the two p-values, alpha = 0.05, NN-favorable direction):
    H4: fine-tuned NN (pooled) beats PI
    H5: fine-tuned NN (pooled) beats static
SECONDARY (reported, ungated): imitation-only vs PI (TOST: did distillation
transfer?), fine-tuned vs imitation-only (did fine-tuning add anything?),
eta<1-stratified fine-tuned vs PI (widening must concentrate there),
per-seed values, G2 band per group.

Usage:
    python analyze_tier2b.py mcb_experiments_gpu/tier2b_eval.pkl
"""

import pickle
import sys

import numpy as np

from analyze_tier2 import group_err, holm
from jcm.mcb.gates import cooling_gate
from jcm.mcb.gates_stats import paired_report


def analyze(results, target=-0.1, band=(-0.12, -0.08)):
    per_ic = results["per_ic"]
    arms = sorted(per_ic.keys())
    ft_arms = sorted(a for a in arms if a.startswith("finetuned"))
    imit_arms = sorted(a for a in arms if a.startswith("imitation"))
    assert "static" in arms and "pi" in arms and ft_arms, arms

    err_ft = group_err(per_ic, ft_arms, target)
    err_pi = group_err(per_ic, ["pi"], target)
    err_st = group_err(per_ic, ["static"], target)

    h4 = paired_report(err_ft, err_pi)
    h5 = paired_report(err_ft, err_st)
    adj = holm({"H4_ft_vs_pi": h4["p_t"], "H5_ft_vs_static": h5["p_t"]})

    out = {
        "pooling_rule": "per-IC mean over fine-tuned seed arms; n = num ICs",
        "n_ics": int(err_ft.size),
        "finetuned_arms": ft_arms,
        "imitation_arms": imit_arms,
        "primary": {
            "H4_ft_vs_pi": {**h4, "p_holm": adj["H4_ft_vs_pi"],
                            "significant_holm":
                                bool(adj["H4_ft_vs_pi"] < 0.05
                                     and h4["mean"] < 0)},
            "H5_ft_vs_static": {**h5, "p_holm": adj["H5_ft_vs_static"],
                                "significant_holm":
                                    bool(adj["H5_ft_vs_static"] < 0.05
                                         and h5["mean"] < 0)},
        },
        "secondary": {},
    }

    if imit_arms:
        err_im = group_err(per_ic, imit_arms, target)
        out["secondary"]["imitation_vs_pi"] = paired_report(err_im, err_pi)
        out["secondary"]["finetuned_vs_imitation"] = paired_report(
            err_ft, err_im)
    for a in ft_arms:
        e = np.abs(np.asarray(per_ic[a]["dsst_10d"]) - target)
        out["secondary"][f"per_seed[{a} vs pi]"] = paired_report(e, err_pi)
    etas = np.asarray(results.get("efficacies", []))
    if etas.size == err_ft.size:
        lo = etas < 1.0
        if lo.sum() >= 2:
            out["secondary"]["stratified_ft_vs_pi[eta<1]"] = paired_report(
                err_ft[lo], err_pi[lo])
            out["secondary"]["stratified_means"] = {
                "n_low": int(lo.sum()),
                "eta<1": {"pi": float(err_pi[lo].mean()),
                          "ft": float(err_ft[lo].mean()),
                          "static": float(err_st[lo].mean())},
                "eta>=1": {"pi": float(err_pi[~lo].mean()),
                           "ft": float(err_ft[~lo].mean()),
                           "static": float(err_st[~lo].mean())},
            }
    groups = [("static", ["static"]), ("pi", ["pi"]),
              ("finetuned(pooled)", ft_arms)]
    if imit_arms:
        groups.append(("imitation", imit_arms))
    for name, group in groups:
        dsst = np.stack([np.asarray(per_ic[a]["dsst_10d"])
                         for a in group]).mean(axis=0)
        out["secondary"][f"G2[{name}]"] = cooling_gate(dsst, band=band)
    return out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "mcb_experiments_gpu/tier2b_eval.pkl"
    with open(path, "rb") as f:
        results = pickle.load(f)
    target = results["config"].get("target_cooling", -0.1)
    band = tuple(results["config"].get("band", (-0.12, -0.08)))
    out = analyze(results, target=target, band=band)

    print("=" * 72)
    print("TIER-2b PRIMARY ANALYSIS (Amendment 4, pre-committed)")
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
        if isinstance(rep, dict) and "p_t" in rep:
            print(f"  {name}: mean {rep['mean']:+.5f} p_t={rep['p_t']:.4f} "
                  f"|equiv|<{rep['equivalence_bound_95']:.5f}")
        elif isinstance(rep, dict) and "verdict" in rep:
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
