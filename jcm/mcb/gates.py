"""Control-relative, significance-tested gates (PREREGISTRATION.md sections 4-5).

The 2026-07-12 audit found every stage verdict was a bare band or `<=` on an
n<=2 mean, decided inside the project's own noise floor. These gates instead
compare a policy to the stage1-static comparator on IDENTICAL initial
conditions via a PAIRED per-IC difference, and report mean +/- standard error
with an explicit 2-s.e. significance rule. A verdict whose margin is within
2 s.e. is reported as ``underpowered`` (neither PASS nor FAIL), never as a
result.

No SciPy dependency: for paired samples the test statistic is the mean of the
per-IC differences over their standard error (a paired t / z); at the small n
here the 2-s.e. rule is the honest, conservative call. Pure NumPy so the module
is unit-testable without the model.
"""

from typing import Sequence

import numpy as np


def paired_stats(policy: Sequence[float], static: Sequence[float]) -> dict:
    """Paired (policy - static) statistics over identical ICs.

    Returns mean, standard error, n, the per-IC diffs, the 2-s.e. confidence
    interval, and whether the mean differs from zero by more than 2 s.e.
    ``se`` is infinite for n < 2 (a single IC cannot support a verdict).
    """
    p = np.asarray(policy, dtype=float)
    s = np.asarray(static, dtype=float)
    if p.shape != s.shape:
        raise ValueError(f"paired arrays must align: {p.shape} vs {s.shape}")
    d = p - s
    n = int(d.size)
    mean = float(d.mean()) if n else float("nan")
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("inf")
    return {
        "mean": mean,
        "se": se,
        "n": n,
        "diffs": d.tolist(),
        "ci95": (mean - 2 * se, mean + 2 * se),
        "significant": abs(mean) > 2 * se,
    }


def cooling_gate(dsst_per_ic: Sequence[float],
                 band: tuple = (-0.12, -0.08)) -> dict:
    """G2: held-out per-IC dSST mean lies in ``band`` AND is powered.

    Powered means the 2-s.e. half-width is narrower than the band half-width;
    otherwise the ICs cannot resolve whether the mean is in the band and the
    gate is ``underpowered`` (the audit's Gate-2 was 0.036 s.e. from flipping).
    """
    x = np.asarray(dsst_per_ic, dtype=float)
    n = int(x.size)
    mean = float(x.mean()) if n else float("nan")
    se = float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else float("inf")
    lo, hi = band
    half = (hi - lo) / 2.0
    in_band = lo <= mean <= hi
    powered = (2 * se) < half
    if not powered:
        verdict = "underpowered"
    elif in_band:
        verdict = "PASS"
    else:
        verdict = "FAIL"
    return {"mean": mean, "se": se, "n": n, "band": band,
            "in_band": in_band, "powered": powered, "verdict": verdict}


def improvement_gate(policy_err: Sequence[float],
                     static_err: Sequence[float]) -> dict:
    """G3: does the policy beat static on cooling-error-to-target?

    ``*_err`` is |dSST - target| per IC (lower is better). Improvement =
    static_err - policy_err > 0. PASS only if the paired mean improvement
    exceeds 2 s.e.; within 2 s.e. is ``underpowered`` (the audit's "beats
    static" was a 0.14-sigma margin measured in-sample).
    """
    st = paired_stats(static_err, policy_err)  # static - policy; >0 = policy better
    if not np.isfinite(st["se"]) or st["se"] == 0:
        verdict = "underpowered"
    elif st["mean"] > 2 * st["se"]:
        verdict = "PASS"
    elif st["mean"] < -2 * st["se"]:
        verdict = "FAIL"  # policy significantly WORSE than static
    else:
        verdict = "underpowered"
    return {**st, "improvement": st["mean"], "verdict": verdict}


def no_worse_gate(policy_loss: Sequence[float],
                  static_loss: Sequence[float]) -> dict:
    """G4: policy held-out loss is no worse than static (paired).

    d = policy_loss - static_loss (lower is better). The policy FAILS only if
    it is SIGNIFICANTLY worse (mean - 2 s.e. > 0). If the policy is clearly
    better (mean + 2 s.e. < 0) that is a strong PASS; otherwise it is a
    (weak) PASS = "not significantly worse", flagged as such.
    """
    st = paired_stats(policy_loss, static_loss)  # policy - static; <=0 is good
    if not np.isfinite(st["se"]):
        verdict = "underpowered"
    elif st["mean"] + 2 * st["se"] < 0:
        verdict = "PASS"  # significantly better than static
    elif st["mean"] - 2 * st["se"] > 0:
        verdict = "FAIL"  # significantly worse than static
    else:
        verdict = "PASS (not sig. worse)"
    return {**st, "verdict": verdict}


def learned(loss_improvement: float, sigma_compile: float) -> dict:
    """Did training learn? Requires improvement > 2 * sigma_compile.

    ``sigma_compile`` is the noise floor from run_noise_floor.py. A
    non-improving noisy loss curve reports ``no`` (the audit found every
    "best epoch" was the minimum of a stationary noise sequence).
    """
    ok = loss_improvement > 2.0 * sigma_compile
    return {"loss_improvement": float(loss_improvement),
            "sigma_compile": float(sigma_compile),
            "threshold": 2.0 * float(sigma_compile),
            "learned": bool(ok)}
