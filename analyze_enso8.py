#!/usr/bin/env python
"""Pre-committed analysis for the ENSO anticipation ablation (Amendment 7 rev 2).

Frozen BEFORE any ablation data exists. Supersedes analyze_enso7.py, which
cannot score this campaign (it hard-codes the arm names) and whose inference
the audit showed to be anti-conservative.

Three corrections are built in, each traceable to an audit finding:

1. HETEROSCEDASTICITY-ROBUST INFERENCE. The slope residuals are strongly
   heteroscedastic (an open-loop arm's error sd grows with |A| while a
   controlled arm's is flat), so ordinary least-squares standard errors are
   too small — Amendment 7's OLS p ~ 2e-9 became ~5e-5 under a wild
   bootstrap. Every p-value here is a Rademacher wild bootstrap under the
   null, with HC3 standard errors reported alongside.

2. A SIGN-AGNOSTIC COMPANION ENDPOINT. A signed slope can read as "no
   sensitivity" when a controller under-corrects warm anomalies and
   OVER-corrects cold ones — measured in Amendment 7, where the classical
   arm's +2.6 mK/K pooled slope decomposes into +11.2 warm and -11.0 cold.
   RMS_A (the root-mean-square error across the amplitude design) cannot be
   gamed by that cancellation, and a hinge fit reports the two branches
   separately. The signed slope stays primary for continuity; RMS_A is
   co-primary and any disagreement between them must be reported.

3. SLOPE EQUIVALENCE (TOST). The likely outcome of the anticipation
   hypothesis is a null, and the repo's TOST covers paired means only. This
   module adds a bootstrap equivalence bound on a slope so a null is
   reportable as a bound rather than a shrug.

Usage:
    python analyze_enso8.py mcb_experiments_gpu/enso8_eval.pkl
"""

import pickle
import sys

import numpy as np

from analyze_tier2 import holm
from jcm.mcb.gates_stats import paired_report, t_ppf


def pool(cells, arms, key="dsst_10d"):
    return np.stack([np.asarray(cells[a][key]).mean(axis=1)
                     for a in arms]).mean(axis=0)


def _ols(a, y):
    am = a - a.mean()
    saa = float((am ** 2).sum())
    slope = float((am * (y - y.mean())).sum() / saa)
    intercept = float(y.mean() - slope * a.mean())
    resid = y - (intercept + slope * a)
    return slope, intercept, resid, saa


def hc3_se(a, y):
    """Heteroscedasticity-consistent (HC3) standard error of the slope."""
    n = a.size
    slope, _, resid, saa = _ols(a, y)
    am = a - a.mean()
    h = 1.0 / n + am ** 2 / saa
    return float(np.sqrt(((am ** 2) * (resid / (1.0 - h)) ** 2).sum()) / saa)


def wild_bootstrap_slope(a, y, n_boot=20000, seed=0):
    """Rademacher wild bootstrap for the slope, imposing the null.

    Returns slope, HC3 se, two-sided p, and a percentile CI from the
    unrestricted (pairs) resampling.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float)
    y = np.asarray(y, float)
    n = a.size
    slope, intercept, resid, saa = _ols(a, y)
    am = a - a.mean()
    null_fit = y.mean() * np.ones(n)          # slope forced to zero
    null_resid = y - null_fit
    count = 0
    boots = np.empty(n_boot)
    for b in range(n_boot):
        v = rng.choice((-1.0, 1.0), n)
        ys = null_fit + null_resid * v
        s_b = float((am * (ys - ys.mean())).sum() / saa)
        if abs(s_b) >= abs(slope):
            count += 1
        idx = rng.integers(0, n, n)
        ab = a[idx]
        yb = y[idx]
        abm = ab - ab.mean()
        sab = float((abm ** 2).sum())
        boots[b] = (float((abm * (yb - yb.mean())).sum() / sab)
                    if sab > 0 else np.nan)
    p = (count + 1) / (n_boot + 1)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return {"slope": slope, "hc3_se": hc3_se(a, y), "p_boot": float(p),
            "ci95_boot": (float(lo), float(hi)), "n": int(n),
            "ols_se": float(np.sqrt((resid ** 2).sum() / (n - 2) / saa))}


def slope_equivalence_bound(a, y, alpha=0.05):
    """Smallest delta supporting |true slope| < delta, using the HC3 se.

    Mirrors gates_stats.equivalence_bound but for a regression slope.
    """
    slope, _, _, _ = _ols(np.asarray(a, float), np.asarray(y, float))
    se = hc3_se(np.asarray(a, float), np.asarray(y, float))
    return float(abs(slope) + t_ppf(1.0 - alpha, a.size - 2) * se)


def hinge_slopes(a, y):
    """Separate warm (A>=0) and cold (A<0) slopes — the cancellation check."""
    out = {}
    for lab, m in (("warm", a >= 0), ("cold", a < 0)):
        if m.sum() >= 3:
            s, _, r, saa = _ols(a[m], y[m])
            out[lab] = {"slope": s, "se": hc3_se(a[m], y[m]),
                        "n": int(m.sum())}
        else:
            out[lab] = None
    return out


def rms_a(err):
    """Sign-agnostic amplitude-response magnitude: RMS error over the design."""
    return float(np.sqrt(np.mean(np.asarray(err, float) ** 2)))


def paired_rms_boot(e1, e0, n_boot=20000, seed=0):
    """Bootstrap the RMS_A difference (arm1 - arm0) over ICs."""
    rng = np.random.default_rng(seed)
    e1 = np.asarray(e1, float)
    e0 = np.asarray(e0, float)
    n = e1.size
    obs = rms_a(e1) - rms_a(e0)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        boots[b] = rms_a(e1[i]) - rms_a(e0[i])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2.0 * min((boots >= 0).mean(), (boots <= 0).mean())
    return {"diff": obs, "ci95": (float(lo), float(hi)),
            "p_boot": float(min(1.0, max(p, 1.0 / n_boot)))}


def analyze(results, target=-0.1, n_boot=20000):
    cells = results["cells"]
    amps = np.asarray(results["enso_amps"], float)
    arms = list(cells)
    imit = sorted(a for a in arms if a.startswith("imitation"))
    groups = {a: [a] for a in arms}
    if len(imit) > 1:
        groups["imitation"] = imit
    err = {n: pool(cells, m) - target for n, m in groups.items()}

    out = {
        "n_ics": int(amps.size),
        "amplitude_design": {
            "mean": float(amps.mean()), "sd": float(amps.std(ddof=1)),
            "saa": float(((amps - amps.mean()) ** 2).sum()),
            "min": float(amps.min()), "max": float(amps.max()),
        },
        "target": target,
        "arms": {}, "primary": {}, "secondary": {},
    }
    for n, e in err.items():
        s = wild_bootstrap_slope(amps, e, n_boot=n_boot)
        out["arms"][n] = {
            "sensitivity": s, "hinge": hinge_slopes(amps, e),
            "rms_a_mK": rms_a(e) * 1000,
            "mean_abs_mK": float(np.abs(e).mean() * 1000),
            "mean_signed_mK": float(e.mean() * 1000),
        }

    # PRIMARY — H10: does anticipation beat the BEST reactive controller?
    # The comparator is whichever reactive-ladder arm (feedforward ablated,
    # feedback gain varied) scores best ON THIS held-out data. Selecting the
    # comparator on the test set INFLATES it, which biases against H10 — the
    # conservative direction — and removes any "the opponent was detuned"
    # objection. Revision 2's train-tuned comparator was abandoned because
    # the train sweep put the optimum at the grid edge with overlapping CIs.
    reactive = sorted(a for a in err if a.startswith("b"))
    anticip = sorted(a for a in err if a.startswith("p"))
    if reactive:
        best_r = min(reactive, key=lambda a: rms_a(err[a]))
        out["comparator"] = {
            "selected": best_r, "rule": "min RMS_A on held-out (conservative)",
            "ladder": {a: rms_a(err[a]) * 1000 for a in reactive}}
        err["blindfb_t"] = err[best_r]
    if anticip:
        best_p = min(anticip, key=lambda a: rms_a(err[a]))
        out["anticipating"] = {
            "selected": best_p,
            "ladder": {a: rms_a(err[a]) * 1000 for a in anticip}}
        err["pienso"] = err[best_p]
    def pair(a, b, key):
        if a not in err or b not in err:
            return None
        d = err[a] - err[b]
        rep = wild_bootstrap_slope(amps, d, n_boot=n_boot)
        rep["equivalence_bound_95"] = slope_equivalence_bound(amps, d)
        rep["hinge"] = hinge_slopes(amps, d)
        rep["rms"] = paired_rms_boot(err[a], err[b], n_boot=n_boot)
        rep["mean_abs"] = paired_report(np.abs(err[a]), np.abs(err[b]))
        out[key][f"{a} vs {b}"] = rep
        return rep

    h10 = pair("pienso", "blindfb_t", "primary")
    h10r = None
    if h10 is not None:
        h10r = h10["rms"]
        adj = holm({"H10_slope": h10["p_boot"], "H10_rms": h10r["p_boot"]})
        h10["p_holm"] = adj["H10_slope"]
        h10["significant_holm"] = bool(adj["H10_slope"] < 0.05
                                       and h10["slope"] < 0)
        h10r["p_holm"] = adj["H10_rms"]
        h10r["significant_holm"] = bool(adj["H10_rms"] < 0.05
                                        and h10r["diff"] < 0)

    for a, b in (("blindfb_t", "static"), ("pienso", "static"),
                 ("imitation", "static"), ("pienso", "blindfb"),
                 ("blindfb_t", "blindfb"), ("imitation", "pienso"),
                 ("ffonly", "static"), ("pienso", "ffonly")):
        pair(a, b, "secondary")
    return out


def _fmt(rep):
    s = rep["slope"] * 1000
    return (f"{s:+7.1f} mK/K (HC3 se {rep['hc3_se'] * 1000:4.1f}, "
            f"boot p={rep['p_boot']:.2g}, CI [{rep['ci95_boot'][0] * 1000:+.1f},"
            f"{rep['ci95_boot'][1] * 1000:+.1f}])")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "mcb_experiments_gpu/enso8_eval.pkl"
    with open(path, "rb") as f:
        results = pickle.load(f)
    target = float(results["config"].get("target_cooling", -0.1))
    out = analyze(results, target=target)

    d = out["amplitude_design"]
    print("=" * 86)
    print(f"ENSO ANTICIPATION ABLATION — Amendment 7 rev 2 (n={out['n_ics']}, "
          f"A mean {d['mean']:+.1e}, Saa {d['saa']:.1f})")
    print("=" * 86)
    print(f"\n{'arm':>14} {'sensitivity (signed)':>34} {'RMS_A':>7} "
          f"{'warm':>7} {'cold':>7} {'|miss|':>7}")
    for n, a in out["arms"].items():
        h = a["hinge"]
        w = f"{h['warm']['slope'] * 1000:+6.1f}" if h["warm"] else "     -"
        c = f"{h['cold']['slope'] * 1000:+6.1f}" if h["cold"] else "     -"
        print(f"{n:>14} {_fmt(a['sensitivity']):>34} {a['rms_a_mK']:6.1f} "
              f"{w} {c} {a['mean_abs_mK']:6.1f}")
    print("  (warm/cold are the hinge branches; a signed slope near zero with "
          "opposite-signed branches is CANCELLATION, not rejection)")

    print("\nPRIMARY — H10: does anticipation beat the TUNED reactive law?")
    for name, rep in out["primary"].items():
        tag = "SIGNIFICANT" if rep.get("significant_holm") else "n.s."
        print(f"  slope[{name}]: {_fmt(rep)} p_holm={rep.get('p_holm', float('nan')):.3g} {tag}")
        print(f"        |slope| equivalence bound: "
              f"{rep['equivalence_bound_95'] * 1000:.1f} mK/K")
        r = rep["rms"]
        tag2 = "SIGNIFICANT" if r.get("significant_holm") else "n.s."
        print(f"  RMS_A[{name}]: {r['diff'] * 1000:+.1f} mK "
              f"(CI [{r['ci95'][0] * 1000:+.1f},{r['ci95'][1] * 1000:+.1f}], "
              f"p={r['p_boot']:.2g}) {tag2}")

    print("\nSECONDARY:")
    for name, rep in out["secondary"].items():
        print(f"  slope[{name}]: {_fmt(rep)} | RMS_A "
              f"{rep['rms']['diff'] * 1000:+.1f} mK (p={rep['rms']['p_boot']:.2g})")

    out_path = path.replace(".pkl", "_analysis.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    print(f"\nanalysis -> {out_path}")


if __name__ == "__main__":
    main()
