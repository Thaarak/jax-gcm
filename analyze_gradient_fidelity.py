#!/usr/bin/env python
"""Pre-registered analysis for Experiment 1 (PREREGISTRATION.md Amendment 9).

Reads ``<prefix>.npz`` / ``<prefix>.json`` from ``run_gradient_fidelity.py``
and writes ``<prefix>_analysis.json``. Frozen before any Experiment 1 data
exist; every threshold below is registered and must not be edited after
results are seen (log any change as an amendment).

Truth. For each realization (IC x member) and band k, the central difference
``(J(a0 + delta e_k) - J(a0 - delta e_k)) / (2 delta)`` of the tail-mean
objective. The truth vector is the mean over realizations.

Estimators. Reverse-mode Jacobians for each truncation window W (0 = full
BPTT), one per gradient realization. Metrics per (W, horizon, objective):

* ``angle_mean``  angle between the mean estimator and the truth (deg).
* ``ratio``       projection ratio (g_W . g) / (g . g).
* ``median_single_angle``  median over realizations of angle(g_W,r, g).
* ``noise_to_signal``  ||sd_r(g_W,r)|| / ||mean_r g_W,r||.
* ``sign_agreement``  share of (realization, resolved component) pairs with
  the truth's sign; a component is resolved when |g_k| / se_k >= 2.

Uncertainty: hierarchical bootstrap (resample ICs, then members within each
IC, independently for truth and estimator members) — primary. A flat
bootstrap over realizations is reported as a sensitivity check.

Verdict per estimator:
  useful       truth resolved (SNR >= 3), angle CI upper < 20 deg,
               ratio CI inside [0.6, 1.4], median single angle <= 45 deg.
  failed       angle CI lower > 20 deg, or ratio CI entirely outside
               [0.6, 1.4], or median single angle > 45 deg.
  inconclusive otherwise; ``unresolved_truth`` when truth SNR < 3.

Primary gate: objective T0 at horizon 60 d. Outcomes:
  U   truth unresolved at the primary endpoint (add truth members, rerun).
  A   some truncated W useful and full BPTT failed ("truncation rescues").
  A'  some truncated W useful and full BPTT inconclusive.
  B   full BPTT useful (ocean objectives keep a long gradient horizon).
  C   no estimator useful (stop: write up the characterization).
W* (carried into Experiments 2-3): the useful truncated window with the
smallest median single-realization angle; ties within 2 deg go to the larger
window. Registered prediction: at 60 d, W = 1 is ``failed`` on LAND or its
ratio lies outside [0.5, 2]; if the LAND truth is unresolved the prediction
is reported as untestable.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ANGLE_GATE_DEG = 20.0
RATIO_BAND = (0.6, 1.4)
SINGLE_ANGLE_MAX_DEG = 45.0
TRUTH_SNR_MIN = 3.0
SIGN_RESOLVED_Z = 2.0
W_TIE_DEG = 2.0
LAND_RATIO_BAND = (0.5, 2.0)
PRIMARY_OBJECTIVE = "T0"
PRIMARY_HORIZON = 60
N_BOOT = 2000
BOOT_SEED = 2026


# ---------------------------------------------------------------- basics ---

def angle_deg(u, v):
    """Angle between vectors (last axis) in degrees; NaN if either is 0."""
    u, v = np.asarray(u, float), np.asarray(v, float)
    nu = np.linalg.norm(u, axis=-1)
    nv = np.linalg.norm(v, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.sum(u * v, axis=-1) / (nu * nv)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def projection_ratio(u, truth):
    """(u . truth) / (truth . truth) along the last axis."""
    u, truth = np.asarray(u, float), np.asarray(truth, float)
    return np.sum(u * truth, axis=-1) / np.sum(truth * truth, axis=-1)


def tail_means(fd_series, horizons, tail_days):
    """(..., n_days, n_obj) daily series -> (..., n_H, n_obj) tail means."""
    return np.stack([fd_series[..., h - tail_days:h, :].mean(axis=-2)
                     for h in horizons], axis=-2)


def finite_differences(tm, labels, delta, k_bands):
    """Central differences: (n_ic, n_mem, n_H, n_obj, K)."""
    cols = [(tm[:, :, labels.index(f"plus_{k}")]
             - tm[:, :, labels.index(f"minus_{k}")]) / (2.0 * delta)
            for k in range(k_bands)]
    return np.stack(cols, axis=-1)


def nonlinearity(tm, labels, delta, k_bands):
    """|curvature * delta| / |slope| per band, pooled median (n_H, n_obj)."""
    center = tm[:, :, labels.index("center")]
    ratios = []
    for k in range(k_bands):
        plus = tm[:, :, labels.index(f"plus_{k}")]
        minus = tm[:, :, labels.index(f"minus_{k}")]
        slope = np.abs((plus - minus) / (2 * delta)).mean(axis=(0, 1))
        curv = np.abs((plus - 2 * center + minus) / delta ** 2).mean(
            axis=(0, 1))
        with np.errstate(invalid="ignore", divide="ignore"):
            ratios.append(curv * delta / slope)
    return np.median(np.stack(ratios), axis=0)


# ------------------------------------------------------------- bootstrap ---

def _resample(rng, n_ic, n_members, hierarchical):
    """Index arrays (ic_idx, member_idx) for one bootstrap replicate."""
    if hierarchical:
        ics = rng.integers(0, n_ic, n_ic)
        mem = rng.integers(0, n_members, (n_ic, n_members))
        return np.repeat(ics, n_members), mem.ravel()
    flat = rng.integers(0, n_ic * n_members, n_ic * n_members)
    return flat // n_members, flat % n_members


def bootstrap_means(fd, jac, n_boot, seed, hierarchical=True):
    """Bootstrap replicates of the truth and estimator means.

    fd:  (n_ic, k_fd, n_H, n_obj, K) central differences.
    jac: (n_ic, k_g, n_W, n_H, n_obj, K) estimator Jacobians.

    Returns (truth_b (B, n_H, n_obj, K), est_b (B, n_W, n_H, n_obj, K)).
    ICs are shared between truth and estimators in each replicate.
    """
    rng = np.random.default_rng(seed)
    n_ic, k_fd = fd.shape[:2]
    k_g = jac.shape[1]
    truth_b, est_b = [], []
    for _ in range(n_boot):
        if hierarchical:
            ics = rng.integers(0, n_ic, n_ic)
            fi = np.repeat(ics, k_fd)
            fm = rng.integers(0, k_fd, n_ic * k_fd)
            gi = np.repeat(ics, k_g)
            gm = rng.integers(0, k_g, n_ic * k_g)
        else:
            fi, fm = _resample(rng, n_ic, k_fd, False)
            gi, gm = _resample(rng, n_ic, k_g, False)
        truth_b.append(fd[fi, fm].mean(axis=0))
        est_b.append(jac[gi, gm].mean(axis=0))
    return np.stack(truth_b), np.stack(est_b)


# --------------------------------------------------------------- metrics ---

def estimator_metrics(truth, truth_se, est_real):
    """Point metrics for one (W, H, objective) cell.

    truth: (K,), truth_se: (K,), est_real: (n_real, K).
    """
    est_mean = est_real.mean(axis=0)
    sd = est_real.std(axis=0, ddof=1) if est_real.shape[0] > 1 else \
        np.full_like(est_mean, np.nan)
    resolved = np.abs(truth) >= SIGN_RESOLVED_Z * truth_se
    if resolved.any():
        agree = np.sign(est_real[:, resolved]) == np.sign(truth[resolved])
        sign_agreement = float(agree.mean())
    else:
        sign_agreement = float("nan")
    return {
        "angle_mean": float(angle_deg(est_mean, truth)),
        "ratio": float(projection_ratio(est_mean, truth)),
        "median_single_angle": float(np.median(angle_deg(est_real, truth))),
        "noise_to_signal": float(np.linalg.norm(sd)
                                 / np.linalg.norm(est_mean)),
        "sign_agreement": sign_agreement,
    }


def verdict(resolved, angle_ci, ratio_ci, median_single_angle):
    """Return the registered verdict for one estimator cell."""
    if not resolved:
        return "unresolved_truth"
    lo, hi = RATIO_BAND
    if (angle_ci[0] > ANGLE_GATE_DEG or ratio_ci[1] < lo or ratio_ci[0] > hi
            or median_single_angle > SINGLE_ANGLE_MAX_DEG):
        return "failed"
    if (angle_ci[1] < ANGLE_GATE_DEG and ratio_ci[0] >= lo
            and ratio_ci[1] <= hi):
        return "useful"
    return "inconclusive"


def choose_w_star(cells):
    """cells: list of (window, verdict, median_single_angle), truncated only."""
    useful = [(w, a) for w, v, a in cells if v == "useful"]
    if not useful:
        return None
    best = min(a for _, a in useful)
    return max(w for w, a in useful if a <= best + W_TIE_DEG)


def primary_outcome(truth_resolved, cells_truncated, full_verdict):
    """Outcome letter for the primary endpoint (see module docstring)."""
    if not truth_resolved:
        return "U"
    any_useful = any(v == "useful" for _, v, _ in cells_truncated)
    if full_verdict == "useful":
        return "B"
    if any_useful and full_verdict == "failed":
        return "A"
    if any_useful:
        return "A'"
    return "C"


def analyze(arrays, meta, n_boot=N_BOOT, seed=BOOT_SEED):
    """Full registered analysis; returns a JSON-serialisable dict."""
    cfg = meta["config"]
    labels = meta["run_labels"]
    horizons = list(meta["horizons"])
    windows = list(meta["windows"])
    objectives = list(meta["objective_names"])
    tail = int(cfg["tail_days"])
    delta = float(cfg["delta"])
    fd_series = np.asarray(arrays["fd_series"], float)
    jac = np.asarray(arrays["jacobians"], float)
    k_bands = jac.shape[-1]

    finished = int(meta.get("finished_ics", fd_series.shape[0]))
    fd_series, jac = fd_series[:finished], jac[:finished]

    tm = tail_means(fd_series, horizons, tail)
    fd = finite_differences(tm, labels, delta, k_bands)
    truth = fd.mean(axis=(0, 1))                            # (n_H, n_obj, K)
    truth_b, est_b = bootstrap_means(fd, jac, n_boot, seed, True)
    truth_se = truth_b.std(axis=0, ddof=1)
    flat_truth_b, flat_est_b = bootstrap_means(fd, jac, n_boot, seed + 1,
                                               False)

    out = {"n_ics": finished, "fd_members": fd.shape[1],
           "grad_members": jac.shape[1], "n_boot": n_boot,
           "boot_seed": seed, "cells": {}, "truth": {},
           "thresholds": {"angle_gate_deg": ANGLE_GATE_DEG,
                          "ratio_band": RATIO_BAND,
                          "single_angle_max_deg": SINGLE_ANGLE_MAX_DEG,
                          "truth_snr_min": TRUTH_SNR_MIN,
                          "w_tie_deg": W_TIE_DEG}}

    if "zero" in labels:
        resp = (tm[:, :, labels.index("center")]
                - tm[:, :, labels.index("zero")])      # (ic, mem, H, obj)
        out["operating_point_response"] = {
            f"{obj}@{h}": {
                "mean": float(resp[:, :, hi, oi].mean()),
                "se": float(resp[:, :, hi, oi].mean(axis=1).std(ddof=1)
                            / np.sqrt(resp.shape[0])) if resp.shape[0] > 1
                else float("nan")}
            for hi, h in enumerate(horizons)
            for oi, obj in enumerate(objectives)}
    nl = nonlinearity(tm, labels, delta, k_bands)
    out["nonlinearity_median"] = {
        f"{obj}@{h}": float(nl[hi, oi]) for hi, h in enumerate(horizons)
        for oi, obj in enumerate(objectives)}

    for hi, h in enumerate(horizons):
        for oi, obj in enumerate(objectives):
            g, se = truth[hi, oi], truth_se[hi, oi]
            snr = float(np.linalg.norm(g) / np.sqrt(np.sum(se ** 2)))
            resolved = snr >= TRUTH_SNR_MIN
            out["truth"][f"{obj}@{h}"] = {
                "vector": g.tolist(), "se": se.tolist(), "snr": snr,
                "resolved": bool(resolved)}
            for wi, w in enumerate(windows):
                real = jac[:, :, wi, hi, oi].reshape(-1, k_bands)
                cell = estimator_metrics(g, se, real)
                ang_b = angle_deg(est_b[:, wi, hi, oi], truth_b[:, hi, oi])
                rat_b = projection_ratio(est_b[:, wi, hi, oi],
                                         truth_b[:, hi, oi])
                fang = angle_deg(flat_est_b[:, wi, hi, oi],
                                 flat_truth_b[:, hi, oi])
                frat = projection_ratio(flat_est_b[:, wi, hi, oi],
                                        flat_truth_b[:, hi, oi])
                cell["angle_ci"] = np.nanpercentile(ang_b, [2.5, 97.5]).tolist()
                cell["ratio_ci"] = np.nanpercentile(rat_b, [2.5, 97.5]).tolist()
                cell["flat_angle_ci"] = np.nanpercentile(
                    fang, [2.5, 97.5]).tolist()
                cell["flat_ratio_ci"] = np.nanpercentile(
                    frat, [2.5, 97.5]).tolist()
                cell["verdict"] = verdict(resolved, cell["angle_ci"],
                                          cell["ratio_ci"],
                                          cell["median_single_angle"])
                out["cells"][f"W{w}|{obj}@{h}"] = cell

    key = f"{PRIMARY_OBJECTIVE}@{PRIMARY_HORIZON}"
    if PRIMARY_HORIZON in horizons and 0 in windows:
        cells_trunc = [(w, out["cells"][f"W{w}|{key}"]["verdict"],
                        out["cells"][f"W{w}|{key}"]["median_single_angle"])
                       for w in windows if w != 0]
        full_v = out["cells"][f"W0|{key}"]["verdict"]
        out["primary"] = {
            "endpoint": key,
            "outcome": primary_outcome(out["truth"][key]["resolved"],
                                       cells_trunc, full_v),
            "full_bptt_verdict": full_v,
            "truncated_verdicts": {str(w): v for w, v, _ in cells_trunc},
            "w_star": choose_w_star(cells_trunc),
        }
        land_key = f"W1|LAND@{PRIMARY_HORIZON}"
        if land_key in out["cells"]:
            lc = out["cells"][land_key]
            lo, hi_ = LAND_RATIO_BAND
            if not out["truth"][f"LAND@{PRIMARY_HORIZON}"]["resolved"]:
                # A prediction about an unmeasurable truth is untestable,
                # not confirmed or refuted.
                out["land_prediction_confirmed"] = "untestable"
            else:
                out["land_prediction_confirmed"] = bool(
                    lc["verdict"] == "failed"
                    or not lo <= lc["ratio"] <= hi_)
    else:
        out["primary"] = {"endpoint": key, "outcome": "not_computed",
                          "reason": "primary horizon or full BPTT missing"}
    return out


def print_table(result, windows, horizons, objectives):
    print(f"n_ics={result['n_ics']}  fd_members={result['fd_members']}  "
          f"grad_members={result['grad_members']}")
    for obj in objectives:
        for h in horizons:
            t = result["truth"][f"{obj}@{h}"]
            print(f"\n{obj}@{h}d  truth SNR {t['snr']:.1f}"
                  f"{'' if t['resolved'] else '  (UNRESOLVED)'}")
            for w in windows:
                c = result["cells"][f"W{w}|{obj}@{h}"]
                name = "full" if w == 0 else f"W={w}"
                print(f"  {name:>6}: angle {c['angle_mean']:5.1f} "
                      f"[{c['angle_ci'][0]:5.1f},{c['angle_ci'][1]:5.1f}]  "
                      f"ratio {c['ratio']:5.2f} "
                      f"[{c['ratio_ci'][0]:5.2f},{c['ratio_ci'][1]:5.2f}]  "
                      f"single {c['median_single_angle']:5.1f}  "
                      f"N/S {c['noise_to_signal']:5.2f}  -> {c['verdict']}")
    p = result["primary"]
    print(f"\nPRIMARY {p['endpoint']}: outcome {p['outcome']}  "
          f"(full BPTT {p.get('full_bptt_verdict')}, W* = "
          f"{p.get('w_star')})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prefix", help="Output prefix of run_gradient_fidelity.py")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args(argv)
    arrays = dict(np.load(args.prefix + ".npz"))
    with open(args.prefix + ".json") as f:
        meta = json.load(f)
    result = analyze(arrays, meta, n_boot=args.n_boot)
    result["input"] = {"prefix": args.prefix, "git": meta.get("git")}
    out_path = Path(args.prefix + "_analysis.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print_table(result, meta["windows"], meta["horizons"],
                meta["objective_names"])
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
