"""Retrospective side-effect analysis: Amazon/Sahel precipitation across
Tier-1/2/2b campaign evaluations.

Closes two documented gaps (MCB_META_AUDIT.md section 2.4):
  1. amazon_mm_day / sahel_mm_day were recorded per (IC, member, arm) by
     run_confirmatory_eval.py in every campaign but never aggregated or
     reported anywhere.
  2. The v3 teleconnection gate was computed, came out False (FAIL), and
     went unreported. It is reported here, with the R4 softplus-offset
     caveat that applies to those v3 loss values.

Honest framing (disclosed up front, also in the printed header):
  - The stored quantity is a DAY-60 SNAPSHOT of the regional precip change
    vs the paired no-MCB baseline. The registered metric
    (PREREGISTRATION.md section 3) is an interval mean, which was never
    recorded; this analysis is therefore exploratory / best-available, and
    the section-3 resolvability rule governs whether G5 can be scored at
    all: if the static arm is statistically indistinguishable from the
    do-nothing constant (0 mm/day), the gate stays WITHDRAWN and only
    equivalence bounds are reported.
  - Region masks are plain lat-lon boxes NOT intersected with the land
    mask (jcm/mcb/mcb_regions.py TELECONNECTION_REGIONS), so "Amazon"
    includes some Atlantic cells at T31.
  - Single-day regional precip is chaos-dominated; the per-run chaos sd
    is estimated from the within-(arm, IC) micro-ensemble member spread
    and reported alongside every mean.
  - All one-sample tests form one Holm family (as do all vs-static
    comparisons): with ~24 looks at alpha=0.05, roughly one nominal
    "significant" is expected from noise alone.

Statistics reuse the registered machinery in jcm.mcb.gates_stats
(paired t, exact Wilcoxon, TOST equivalence bounds) and the Holm
correction from analyze_tier2.

Usage:
    python analyze_precip_sideeffects.py \
        --dir mcb_experiments_gpu --json-out precip_sideeffects.json
"""

import argparse
import json
import pickle

import numpy as np

from analyze_tier2 import holm
from jcm.mcb.gates_stats import paired_report, paired_t_test, \
    wilcoxon_signed_rank, equivalence_bound

REGIONS = ("amazon", "sahel")

# (pickle filename, campaign label, seed-pooled arm groups)
CAMPAIGNS = [
    ("confirmatory_eval.pkl", "Tier-1 confirmatory", {
        "static": ["static"],
        "retrain": ["retrain"],
        "openloop": ["openloop"],
    }),
    ("tier2_eval.pkl", "Tier-2 (eta-uncertainty)", {
        "static": ["static"],
        "pi": ["pi"],
        "feedback(3s)": ["feedback_s42", "feedback_s43", "feedback_s44"],
        "openloop(3s)": ["openloop_s42", "openloop_s43", "openloop_s44"],
    }),
    ("tier2b_eval.pkl", "Tier-2b (PI-imitation)", {
        "static": ["static"],
        "pi": ["pi"],
        "imitation": ["imitation_s52"],
        "finetuned(3s)": ["finetuned_s52", "finetuned_s53", "finetuned_s54"],
    }),
]


def per_ic_and_chaos(cells, arm_names, region):
    """Seed-pooled per-IC means and per-run chaos sd for one arm group.

    Per-IC value = mean over micro-ensemble members (and over seeds for
    multi-seed groups) — the same pooling rule as the frozen
    analyze_tier2/analyze_tier2b analyses. Chaos sd = member-spread sd
    within (arm, IC), pooled over arms and ICs (Bessel-corrected per cell).
    """
    per_arm = [np.asarray(cells[a][f"{region}_mm_day"], dtype=float)
               for a in arm_names]                       # each (n_ic, k)
    stacked = np.stack(per_arm)                          # (n_arm, n_ic, k)
    per_ic = stacked.mean(axis=(0, 2))                   # (n_ic,)
    k = stacked.shape[2]
    chaos_sd = float(np.sqrt(np.mean(stacked.var(axis=2, ddof=1)))) \
        if k > 1 else float("nan")
    return per_ic, chaos_sd, k


def one_sample_report(values):
    """Test per-IC values against the do-nothing constant (0 mm/day)."""
    d = np.asarray(values, dtype=float)
    t = paired_t_test(d)
    w = wilcoxon_signed_rank(d)
    return {
        "mean": t["mean"], "se": t["se"], "n": t["n"], "ci95_t": t["ci95"],
        "p_t": t["p"], "p_wilcoxon": w["p"],
        "equivalence_bound_95": equivalence_bound(d),
    }


def analyze_campaign(results, label, groups):
    """Analyze one campaign's cells dict. Pure; testable on synthetic data."""
    cells = results["cells"]
    out = {"label": label, "n_ic": None, "regions": {}}
    for region in REGIONS:
        reg = {"arms": {}, "vs_static": {}}
        per_ic_cache = {}
        for gname, arms in groups.items():
            per_ic, chaos_sd, k = per_ic_and_chaos(cells, arms, region)
            per_ic_cache[gname] = per_ic
            out["n_ic"] = int(per_ic.size)
            rep = one_sample_report(per_ic)
            rep["per_run_chaos_sd"] = chaos_sd
            rep["k_members"] = k
            reg["arms"][gname] = rep
        for gname, per_ic in per_ic_cache.items():
            if gname == "static":
                continue
            reg["vs_static"][gname] = paired_report(
                per_ic, per_ic_cache["static"])
        out["regions"][region] = reg
    return out


def apply_holm_families(campaigns):
    """Holm-correct the one-sample family and the vs-static family.

    Annotates each report in place with p_holm and significant_holm and
    returns the two adjusted p-value dicts.
    """
    onesample, diffs = {}, {}
    for camp in campaigns:
        for region, reg in camp["regions"].items():
            for gname, rep in reg["arms"].items():
                onesample[(camp["label"], region, gname)] = rep["p_t"]
            for gname, rep in reg["vs_static"].items():
                diffs[(camp["label"], region, gname)] = rep["p_t"]
    adj_one = holm(onesample)
    adj_diff = holm(diffs)
    for camp in campaigns:
        for region, reg in camp["regions"].items():
            for gname, rep in reg["arms"].items():
                p = adj_one[(camp["label"], region, gname)]
                rep["p_holm"] = p
                rep["significant_holm"] = bool(p < 0.05)
            for gname, rep in reg["vs_static"].items():
                p = adj_diff[(camp["label"], region, gname)]
                rep["p_holm"] = p
                rep["significant_holm"] = bool(p < 0.05)
    return adj_one, adj_diff


def resolvability(reg):
    """PREREGISTRATION section-3 rule on the static arm (Holm-corrected)."""
    return bool(reg["arms"]["static"].get(
        "significant_holm", reg["arms"]["static"]["p_t"] < 0.05))


def analyze_v3_breach(results):
    """Report the v3 teleconnection gate that failed and went unreported.

    v3 loss values predate the Tier-1 softplus-offset fix, so each regional
    penalty carries the R4 constant floor softplus(0)/100 = 0.0069315;
    values are reported raw AND offset-subtracted.
    """
    offset = float(np.log(2.0) / 100.0)
    gate = results["gates"].get("3_teleconnection_protection")
    arms = sorted({a for (a, _i) in results["cells"]})
    per_arm = {}
    for arm in arms:
        rows = [(i, c) for (a, i), c in results["cells"].items() if a == arm
                and c.get("split") == "heldout"]
        rows.sort()
        per_arm[arm] = {
            reg: np.array([c["precip_losses"][reg] for _, c in rows])
            for reg in REGIONS
        }
    result = {"stored_gate_verdict": gate,
              "softplus_offset_R4": offset, "arms": {}}
    for arm in arms:
        result["arms"][arm] = {
            reg: {
                "n": int(per_arm[arm][reg].size),
                "mean_raw": float(per_arm[arm][reg].mean()),
                "mean_offset_subtracted":
                    float(per_arm[arm][reg].mean() - offset),
            } for reg in REGIONS
        }
    if "stage5-realistic" in per_arm and "stage1-static" in per_arm:
        result["policy_vs_static"] = {
            reg: paired_report(per_arm["stage5-realistic"][reg],
                               per_arm["stage1-static"][reg])
            for reg in REGIONS
        }
    return result


def fmt_arm(name, rep):
    sig = "SIG" if rep["significant_holm"] else "n.s."
    return (f"    {name:>14}: {rep['mean']:+7.3f} ± {rep['se']:.3f} mm/day "
            f"(p_t={rep['p_t']:.3f}, p_holm={rep['p_holm']:.3f}, "
            f"p_w={rep['p_wilcoxon']:.3f}, {sig}) "
            f"|effect|<{rep['equivalence_bound_95']:.3f} @95% "
            f"[chaos sd/run {rep['per_run_chaos_sd']:.2f}, "
            f"k={rep['k_members']}]")


def fmt_diff(name, rep):
    sig = "SIG" if rep["significant_holm"] else "n.s."
    return (f"    {name:>14} − static: {rep['mean']:+7.3f} ± {rep['se']:.3f} "
            f"(p_t={rep['p_t']:.3f}, p_holm={rep['p_holm']:.3f}, {sig}) "
            f"|diff|<{rep['equivalence_bound_95']:.3f} @95%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="mcb_experiments_gpu")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    print("=" * 76)
    print("RETROSPECTIVE PRECIP SIDE-EFFECT ANALYSIS (day-60 snapshot, "
          "box masks;")
    print("registered interval-mean metric was never recorded — "
          "exploratory/best-available)")
    print("=" * 76)

    campaigns = []
    for fname, label, groups in CAMPAIGNS:
        with open(f"{args.dir}/{fname}", "rb") as f:
            results = pickle.load(f)
        campaigns.append(analyze_campaign(results, label, groups))
    apply_holm_families(campaigns)

    for camp in campaigns:
        print(f"\n--- {camp['label']} (n={camp['n_ic']} ICs) ---")
        for region in REGIONS:
            reg = camp["regions"][region]
            print(f"  {region.capitalize()} Δprecip vs paired baseline "
                  f"(mm/day, − = drying):")
            for gname, rep in reg["arms"].items():
                print(fmt_arm(gname, rep))
            for gname, rep in reg["vs_static"].items():
                print(fmt_diff(gname, rep))
            print(f"  G5 resolvability (static vs do-nothing 0): "
                  f"{'RESOLVABLE' if resolvability(reg) else 'NOT RESOLVABLE'
                  ' — G5 stays WITHDRAWN; equivalence bounds govern'}")

    print("\n--- v3 breach closure (eval_v3.pkl) ---")
    with open(f"{args.dir}/eval_v3.pkl", "rb") as f:
        v3_results = pickle.load(f)
    v3 = analyze_v3_breach(v3_results)
    print(f"  stored gate 3_teleconnection_protection = "
          f"{v3['stored_gate_verdict']}  (computed FAIL, never reported)")
    print(f"  NOTE: v3 losses carry the R4 softplus floor "
          f"{v3['softplus_offset_R4']:.5f}; offset-subtracted means below")
    for arm, regs in v3["arms"].items():
        vals = ", ".join(
            f"{reg} {r['mean_offset_subtracted']:+.4f}"
            for reg, r in regs.items())
        print(f"    {arm:>18} (n={regs['amazon']['n']}): {vals}")
    if "policy_vs_static" in v3:
        for reg, rep in v3["policy_vs_static"].items():
            sig = "SIG" if rep["significant"] else "n.s."
            print(f"    stage5 − static [{reg}]: {rep['mean']:+.4f} ± "
                  f"{rep['se']:.4f} (p_t={rep['p_t']:.3f}, {sig})")

    if args.json_out:
        def default(o):
            if isinstance(o, (np.floating, np.integer)):
                return o.item()
            if isinstance(o, np.ndarray):
                return o.tolist()
            raise TypeError(type(o))
        payload = {
            "campaigns": campaigns,
            "v3_breach": v3,
        }
        with open(args.json_out, "w") as f:
            json.dump(payload, f, indent=2, default=default)
        print(f"\nJSON written to {args.json_out}")


if __name__ == "__main__":
    main()
