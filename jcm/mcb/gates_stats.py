"""Formal paired tests for the pre-registered gates (pure NumPy, no SciPy).

PREREGISTRATION.md section 4 promises "paired t / Wilcoxon, alpha=0.05
pre-set"; the 2026-07-29 meta-audit found neither was implemented — gates.py
uses a bare 2-s.e. rule, which at n=10 runs at alpha ~= 7.7% (t_{0.975,9} =
2.262, not 2). This module supplies the registered tests plus TOST
equivalence bounds, so "underpowered" verdicts can be upgraded to
statistically significant bounded-equivalence claims.

Everything is pure NumPy so the module is unit-testable without the model
(and without adding a SciPy dependency to jcm).
"""

from typing import Sequence

import numpy as np


# --- Student-t distribution via the regularized incomplete beta function ---

def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz)."""
    max_iter, eps, fpmin = 200, 3e-12, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    from math import exp, lgamma, log
    ln_front = (lgamma(a + b) - lgamma(a) - lgamma(b)
                + a * log(x) + b * log(1.0 - x))
    front = exp(ln_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_sf(t: float, df: float) -> float:
    """Student-t survival function P(T > t)."""
    x = df / (df + t * t)
    p = 0.5 * _betai(df / 2.0, 0.5, x)
    return p if t >= 0 else 1.0 - p


def t_ppf(q: float, df: float) -> float:
    """Student-t quantile via bisection on t_sf (q in (0, 1))."""
    if not 0.0 < q < 1.0:
        raise ValueError(f"q must be in (0,1), got {q}")
    lo, hi = -1e3, 1e3
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 1.0 - t_sf(mid, df) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --- Paired tests -----------------------------------------------------------

def paired_t_test(diffs: Sequence[float]) -> dict:
    """Two-sided paired t-test on per-IC differences.

    Returns mean, se, t, df, p (two-sided), and the proper t-based 95% CI.
    """
    d = np.asarray(diffs, dtype=float)
    n = int(d.size)
    if n < 2:
        return {"mean": float(d.mean()) if n else float("nan"),
                "se": float("inf"), "t": float("nan"), "df": n - 1,
                "p": float("nan"), "ci95": (float("nan"), float("nan")),
                "n": n}
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n))
    if se == 0.0:
        t = float("inf") if mean != 0 else 0.0
        p = 0.0 if mean != 0 else 1.0
    else:
        t = mean / se
        p = 2.0 * t_sf(abs(t), n - 1)
    tcrit = t_ppf(0.975, n - 1)
    return {"mean": mean, "se": se, "t": t, "df": n - 1, "p": float(p),
            "ci95": (mean - tcrit * se, mean + tcrit * se), "n": n}


def _signrank_cdf_table(n: int) -> np.ndarray:
    """Exact null distribution of the Wilcoxon signed-rank statistic W+.

    counts[w] = number of sign assignments with rank-sum w (DP over ranks).
    """
    max_w = n * (n + 1) // 2
    counts = np.zeros(max_w + 1, dtype=np.float64)
    counts[0] = 1.0
    for r in range(1, n + 1):
        shifted = np.zeros_like(counts)
        shifted[r:] = counts[:-r] if r > 0 else counts
        counts = counts + shifted
    return counts


def wilcoxon_signed_rank(diffs: Sequence[float]) -> dict:
    """Two-sided Wilcoxon signed-rank test (exact for n <= 25, else normal).

    Zero differences are dropped (standard practice); ties get midranks.
    The exact path assumes no ties among |d| (with float dSSTs, ties are
    essentially impossible); with ties the exact p is approximate.
    """
    d = np.asarray(diffs, dtype=float)
    d = d[d != 0.0]
    n = int(d.size)
    if n < 2:
        return {"W": float("nan"), "p": float("nan"), "n": n}
    order = np.argsort(np.abs(d))
    ranks = np.empty(n, dtype=float)
    abs_sorted = np.abs(d)[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs_sorted[j + 1] == abs_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    w_plus = float(ranks[d > 0].sum())
    max_w = n * (n + 1) / 2.0
    w = min(w_plus, max_w - w_plus)
    if n <= 25:
        counts = _signrank_cdf_table(n)
        total = counts.sum()
        # two-sided: 2 * P(W <= w) under the symmetric null
        p = 2.0 * counts[: int(np.floor(w)) + 1].sum() / total
        p = min(1.0, float(p))
    else:
        mu = max_w / 2.0
        sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
        z = (w - mu + 0.5) / sigma  # continuity-corrected
        p = min(1.0, 2.0 * _norm_sf(abs(z)))
    return {"W": w_plus, "p": float(p), "n": n}


def _norm_sf(z: float) -> float:
    from math import erfc, sqrt
    return 0.5 * erfc(z / sqrt(2.0))


# --- Equivalence (TOST) -----------------------------------------------------

def equivalence_bound(diffs: Sequence[float], alpha: float = 0.05) -> float:
    """Smallest delta for which TOST declares |mean diff| < delta at alpha.

    TOST rejects "|effect| >= delta" iff both one-sided t-tests reject, which
    at level alpha happens exactly when delta > |mean| + t_{1-alpha, n-1}*se.
    The returned bound therefore IS the demonstrable equivalence margin: e.g.
    a return of 0.005 supports "any true difference is inside +/-0.005 at 95%
    confidence" (equivalently, the 90% CI lies inside +/-0.005).
    """
    d = np.asarray(diffs, dtype=float)
    n = int(d.size)
    if n < 2:
        return float("inf")
    se = float(d.std(ddof=1) / np.sqrt(n))
    return float(abs(d.mean()) + t_ppf(1.0 - alpha, n - 1) * se)


def tost(diffs: Sequence[float], delta: float, alpha: float = 0.05) -> dict:
    """Two one-sided tests for equivalence within +/-delta."""
    d = np.asarray(diffs, dtype=float)
    n = int(d.size)
    if n < 2:
        return {"equivalent": False, "p": float("nan"), "delta": delta, "n": n}
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n))
    if se == 0.0:
        equivalent = abs(mean) < delta
        return {"equivalent": bool(equivalent),
                "p": 0.0 if equivalent else 1.0, "delta": delta, "n": n}
    t_lower = (mean + delta) / se   # H0: mean <= -delta
    t_upper = (mean - delta) / se   # H0: mean >= +delta
    p_lower = t_sf(t_lower, n - 1)          # reject if small
    p_upper = t_sf(-t_upper, n - 1)
    p = max(float(p_lower), float(p_upper))
    return {"equivalent": bool(p < alpha), "p": p, "delta": float(delta),
            "n": n}


# --- Combined report for gate consumers -------------------------------------

def paired_report(treatment: Sequence[float],
                  comparator: Sequence[float],
                  alpha: float = 0.05) -> dict:
    """Full pre-registered paired analysis of treatment - comparator.

    Convention matches gates.paired_stats: diffs = treatment - comparator,
    so negative means treatment is lower (better, for errors/losses).
    """
    tr = np.asarray(treatment, dtype=float)
    co = np.asarray(comparator, dtype=float)
    if tr.shape != co.shape:
        raise ValueError(f"paired arrays must align: {tr.shape} vs {co.shape}")
    d = tr - co
    t_res = paired_t_test(d)
    w_res = wilcoxon_signed_rank(d)
    significant = (t_res["p"] < alpha) if np.isfinite(t_res["p"]) else False
    return {
        "mean": t_res["mean"],
        "se": t_res["se"],
        "n": t_res["n"],
        "diffs": d.tolist(),
        "ci95_t": t_res["ci95"],
        "t": t_res["t"],
        "p_t": t_res["p"],
        "p_wilcoxon": w_res["p"],
        "significant": bool(significant),
        "equivalence_bound_95": equivalence_bound(d, alpha),
        "alpha": alpha,
    }
