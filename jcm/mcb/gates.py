"""Control-relative, significance-tested gates (PREREGISTRATION.md sections 4-5).

The 2026-07-12 audit found every stage verdict was a bare band or `<=` on an
n<=2 mean, decided inside the project's own noise floor. These gates instead
compare a policy to the stage1-static comparator on IDENTICAL initial
conditions via a PAIRED per-IC difference, and report mean +/- standard error
with an explicit 2-s.e. significance rule. A verdict whose margin is within
2 s.e. is reported as ``underpowered`` (neither PASS nor FAIL), never as a
result.

No SciPy dependency: pure NumPy so the module is unit-testable without the
model. The primary decision rule is the pre-registered 2-s.e. margin; note
that at n=10 this is slightly ANTI-conservative (t_{0.975,9}=2.262, so the
2-s.e. rule runs at alpha ~= 7.7%, not 5%). Every gate therefore also reports
the formal paired-t and exact Wilcoxon p-values plus the demonstrable TOST
equivalence bound (gates_stats.py, the tests PREREGISTRATION.md section 4
actually names) so verdicts can be checked at the registered alpha = 0.05.
"""

from typing import Sequence

import numpy as np

from jcm.mcb.gates_stats import (
    equivalence_bound,
    paired_t_test,
    wilcoxon_signed_rank,
)


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
    result = {
        "mean": mean,
        "se": se,
        "n": n,
        "diffs": d.tolist(),
        "ci95": (mean - 2 * se, mean + 2 * se),
        "significant": abs(mean) > 2 * se,
    }
    if n > 1:
        t_res = paired_t_test(d)
        w_res = wilcoxon_signed_rank(d)
        result.update({
            "p_t": t_res["p"],
            "p_wilcoxon": w_res["p"],
            "ci95_t": t_res["ci95"],
            "significant_t": bool(np.isfinite(t_res["p"])
                                  and t_res["p"] < 0.05),
            "equivalence_bound_95": equivalence_bound(d),
        })
    return result


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
    it is SIGNIFICANTLY worse (mean - 2 s.e. > 0) and PASSES only if it is
    significantly better. A margin within 2 s.e. is reported as
    "underpowered (not sig. worse)" per PREREGISTRATION.md section 5 ("any
    gate margin below 2 s.e. is reported as not significant / underpowered,
    never as PASS or FAIL") — the 2026-07-29 meta-audit found the previous
    "PASS (not sig. worse)" label converted absence of evidence into a pass,
    an unregistered non-inferiority framing with no margin. Use the reported
    ``equivalence_bound_95`` for a real non-inferiority claim.
    """
    st = paired_stats(policy_loss, static_loss)  # policy - static; <=0 is good
    if not np.isfinite(st["se"]):
        verdict = "underpowered"
    elif st["mean"] + 2 * st["se"] < 0:
        verdict = "PASS"  # significantly better than static
    elif st["mean"] - 2 * st["se"] > 0:
        verdict = "FAIL"  # significantly worse than static
    else:
        verdict = "underpowered (not sig. worse)"
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
