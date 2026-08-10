#!/usr/bin/env python
"""Pre-committed primary analysis for the corrected ENSO campaign (Amendment 7).

Frozen BEFORE any Amendment-7 data exists. It replaces the Amendment-6
analysis, whose endpoint the audit showed to be degenerate
(MCB_META_AUDIT.md Addendum 5): with an unreachable target and a one-sided
disturbance, mean |dSST - target| reduces to "which arm cooled more", and a
retuned ENSO-BLIND constant gain tied the feedback controllers on it.

PRIMARY ENDPOINT: disturbance sensitivity — the slope of each arm's signed
per-IC error on the hidden amplitude A, in mK per K of Nino3.4. Tested as a
PAIRED slope: regress (arm - static) per-IC differences on A and test that
slope against zero. Why this endpoint cannot be gamed: rescaling a blind
controller by any constant shifts its mean error but leaves its slope
exactly unchanged, so a non-adaptive arm cannot manufacture a slope
reduction. Mean |miss| is retained as an explicitly bias-contaminated
secondary.

REGISTERED CONTROL ARM (computed, not simulated): the best possible
ENSO-blind controller — the static arm rescaled by one constant gain, the
gain chosen leave-one-IC-out so it never sees the case it is scored on.
Its physical model is dSST = g * mu_MCB + s * A + chaos, where chaos does
NOT scale with dose. Every feedback claim must beat it.

Hypotheses (Holm over the primary pair, direction-gated):
  H8: pienso's disturbance sensitivity is lower than static's.
  H9: the pooled imitation arms' sensitivity is lower than static's.
Secondary: imitation vs pienso (slope and mean, with TOST bounds); each
arm vs the LOO blind-gain control on BOTH endpoints; warm/cold asymmetry
(A >= 0 vs A < 0), pre-stated because the actuator floor binds on the cold
side at |A| = |target| / enso_effect_per_K.

Usage:
    python analyze_enso7.py mcb_experiments_gpu/enso7_eval.pkl
"""

import pickle
import sys

import numpy as np

from analyze_tier2 import holm
from jcm.mcb.gates_stats import paired_report, t_ppf, t_sf


def pool(cells, arms, key="dsst_10d"):
    return np.stack([np.asarray(cells[a][key]).mean(axis=1)
                     for a in arms]).mean(axis=0)


def slope_report(a, y, alpha=0.05):
    """Least-squares slope of y on a, with se, t, two-sided p and CI."""
    a = np.asarray(a, float)
    y = np.asarray(y, float)
    n = a.size
    am = a.mean()
    saa = np.sum((a - am) ** 2)
    slope = np.sum((a - am) * (y - y.mean())) / saa
    intercept = y.mean() - slope * am
    resid = y - (intercept + slope * a)
    dof = n - 2
    se = np.sqrt(np.sum(resid ** 2) / dof / saa)
    t = slope / se if se > 0 else np.inf
    p = 2.0 * t_sf(abs(t), dof)
    half = t_ppf(1.0 - alpha / 2.0, dof) * se
    return {"slope": float(slope), "intercept": float(intercept),
            "se": float(se), "t": float(t), "p": float(p), "n": int(n),
            "ci95": (float(slope - half), float(slope + half)),
            "resid_sd": float(resid.std(ddof=2))}


def blind_gain_control(static_per_ic, amps, target):
    """Best ENSO-blind arm: static rescaled by one LOO-tuned constant gain.

    Model: dSST(g) = g * mu_MCB + s * A + chaos, with mu_MCB the static
    arm's A=0 intercept, s its amplitude slope, and chaos its per-IC
    residual (which does NOT scale with dose — the error that made a first
    reconstruction of this control overstate its score).
    """
    static_per_ic = np.asarray(static_per_ic, float)
    amps = np.asarray(amps, float)
    n = amps.size
    grid = np.linspace(0.0, 5.0, 10001)
    out = np.empty(n)
    for i in range(n):
        keep = np.ones(n, bool)
        keep[i] = False
        s, mu = np.polyfit(amps[keep], static_per_ic[keep], 1)
        chaos = static_per_ic - (mu + s * amps)
        errs = [np.abs((g * mu + s * amps + chaos)[keep] - target).mean()
                for g in grid]
        g_star = grid[int(np.argmin(errs))]
        out[i] = g_star * mu + s * amps[i] + chaos[i]
    return out


def analyze(results, target=-0.1):
    cells = results["cells"]
    amps = np.asarray(results["enso_amps"], float)
    arms = list(cells)
    imit = sorted(a for a in arms if a.startswith("imitation"))
    assert "static" in arms and "pienso" in arms, arms

    groups = {a: [a] for a in arms}
    if imit:
        groups["imitation"] = imit
    per_ic = {name: pool(cells, members) for name, members in groups.items()}
    per_ic["blind_loo"] = blind_gain_control(per_ic["static"], amps, target)

    err = {n: v - target for n, v in per_ic.items()}
    out = {
        "n_ics": int(amps.size),
        "amplitude_range": [float(amps.min()), float(amps.max())],
        "amplitude_mean": float(amps.mean()),
        "target": target,
        "sensitivity": {n: slope_report(amps, e) for n, e in err.items()},
        "mean_abs_err_mK": {n: float(np.abs(e).mean() * 1000)
                            for n, e in err.items()},
        "primary": {},
        "secondary": {},
    }

    # PRIMARY: paired slope differences against static.
    h8 = slope_report(amps, err["pienso"] - err["static"])
    tests = {"H8_pienso_sensitivity_vs_static": h8["p"]}
    if imit:
        h9 = slope_report(amps, err["imitation"] - err["static"])
        tests["H9_imitation_sensitivity_vs_static"] = h9["p"]
    adj = holm(tests)
    out["primary"]["H8_pienso_sensitivity_vs_static"] = {
        **h8, "p_holm": adj["H8_pienso_sensitivity_vs_static"],
        "significant_holm": bool(
            adj["H8_pienso_sensitivity_vs_static"] < 0.05 and h8["slope"] < 0)}
    if imit:
        out["primary"]["H9_imitation_sensitivity_vs_static"] = {
            **h9, "p_holm": adj["H9_imitation_sensitivity_vs_static"],
            "significant_holm": bool(
                adj["H9_imitation_sensitivity_vs_static"] < 0.05
                and h9["slope"] < 0)}

    # SECONDARY: every arm against the blind control, on both endpoints.
    for name in ([n for n in ("pienso", "imitation") if n in per_ic]):
        out["secondary"][f"slope[{name} vs blind_loo]"] = slope_report(
            amps, err[name] - err["blind_loo"])
        out["secondary"][f"mean_abs[{name} vs blind_loo]"] = paired_report(
            np.abs(err[name]), np.abs(err["blind_loo"]))
    if imit:
        out["secondary"]["slope[imitation vs pienso]"] = slope_report(
            amps, err["imitation"] - err["pienso"])
        out["secondary"]["mean_abs[imitation vs pienso]"] = paired_report(
            np.abs(err["imitation"]), np.abs(err["pienso"]))
        for a in imit:
            e = pool(cells, [a]) - target
            out["secondary"][f"slope[{a} vs static]"] = slope_report(
                amps, e - err["static"])

    # Pre-stated warm/cold asymmetry (actuator floor binds on the cold side).
    warm = amps >= 0
    if warm.sum() >= 4 and (~warm).sum() >= 4:
        for name in [n for n in ("pienso", "imitation") if n in per_ic]:
            for lab, m in (("A>=0", warm), ("A<0", ~warm)):
                out["secondary"][f"stratified_mean_abs[{name} {lab}]"] = \
                    paired_report(np.abs(err[name][m]),
                                  np.abs(err["static"][m]))
    return out


def fmt_slope(name, r, extra=""):
    sig = r.get("significant_holm")
    tag = ("SIGNIFICANT" if sig else ("n.s." if sig is not None
                                      else ("sig" if r["p"] < 0.05
                                            else "n.s.")))
    ph = f" p_holm={r['p_holm']:.4g}" if "p_holm" in r else ""
    return (f"  {name:>40}: {r['slope'] * 1000:+7.1f} +/- "
            f"{r['se'] * 1000:4.1f} mK/K (p={r['p']:.4g}{ph}) {tag}{extra}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "mcb_experiments_gpu/enso7_eval.pkl"
    with open(path, "rb") as f:
        results = pickle.load(f)
    target = float(results["config"].get("target_cooling", -0.1))
    out = analyze(results, target=target)

    print("=" * 84)
    print(f"ENSO CAMPAIGN — AMENDMENT 7 PRIMARY ANALYSIS (n={out['n_ics']}, "
          f"A in [{out['amplitude_range'][0]:+.2f}, "
          f"{out['amplitude_range'][1]:+.2f}] K, mean "
          f"{out['amplitude_mean']:+.3f})")
    print("=" * 84)
    print("\nDisturbance sensitivity (PRIMARY endpoint) and mean |miss|:")
    print(f"  {'arm':>18} {'sensitivity mK/K':>20} {'|miss| mK':>10}")
    for n, r in out["sensitivity"].items():
        print(f"  {n:>18} {r['slope'] * 1000:+9.1f} +/- "
              f"{r['se'] * 1000:5.1f} {out['mean_abs_err_mK'][n]:9.1f}")

    print("\nPRIMARY (Holm, direction-gated) — paired slope vs static:")
    for name, r in out["primary"].items():
        print(fmt_slope(name, r))
    print("\nSECONDARY:")
    for name, r in out["secondary"].items():
        if "slope" in r:
            print(fmt_slope(name, r))
        else:
            sig = "SIGNIFICANT" if r["significant"] else "n.s."
            print(f"  {name:>40}: {r['mean'] * 1000:+7.1f} mK "
                  f"(se {r['se'] * 1000:.1f}, p={r['p_t']:.4g}) {sig} "
                  f"|eq|<{r['equivalence_bound_95'] * 1000:.1f} mK")

    out_path = path.replace(".pkl", "_analysis.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    print(f"\nanalysis -> {out_path}")


if __name__ == "__main__":
    main()
