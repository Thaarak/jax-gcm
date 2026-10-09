#!/usr/bin/env python
"""Q-flux base climate: diagnose a monthly Q-flux, then settle and gate.

Pre-registered as PREREGISTRATION.md Amendment 9, revisions 0.2 (diagnose,
settle) and 0.3 (correct). Three stages:

``diagnose``
    Cold start of the coupled model on realistic terrain. After every coupled
    day, SST is restored toward the forcing.nc monthly climatology (tau = 5 d,
    ocean cells only). Spin up 365 days, record 1,460 days of the heat the
    restoring adds, and fit 12 mid-month Q-flux values by least squares
    (``jcm.mcb.qflux.fit_monthly_qflux``). Writes the Q-flux NetCDF plus
    diagnostics.

``correct``
    One Newton step on a settled free run: fit its deseasonalized ocean-mean
    SST with an exponential approach, take lambda = C_eff / tau, and add
    lambda * (observed - fitted end state) W m-2 to every ocean cell. Numpy
    only; no model run.

``settle``
    Cold start plus the diagnosed Q-flux, free slab (no restoring), for 3,650
    days. The last 1,460 days are scored with the registered gate:
    G1 |drift| < 0.02 K per 60 days (linear trend with 3 annual harmonics),
    G2 |annual-mean ocean SST - observed| < 0.5 K (cos-latitude weights over
    the slab's ocean cells). Saves the settled carry (the new base carry) and
    a JSON summary. Exits with status 3 when the gate fails.

Examples (CPU, this checkout; how the Q-flux in use was made):
    python run_qflux_base_climate.py diagnose --output-dir mcb_experiments/qflux
    python run_qflux_base_climate.py settle \
        --qflux mcb_experiments/qflux/qflux_monthly_t30.nc \
        --output-dir mcb_experiments/qflux \
        --base-carry-out mcb_experiments/qflux/base_carry.pkl  # attempt 1: FAIL
    python run_qflux_base_climate.py correct \
        --qflux mcb_experiments/qflux/qflux_monthly_t30.nc \
        --settle-dir mcb_experiments/qflux \
        --output mcb_experiments/qflux/qflux_monthly_t30_v2.nc
    python run_qflux_base_climate.py settle \
        --qflux mcb_experiments/qflux/qflux_monthly_t30_v2.nc \
        --output-dir mcb_experiments/qflux/attempt2 \
        --base-carry-out mcb_experiments/qflux/attempt2/base_carry.pkl  # PASS

Smoke (a few days; the gate numbers are meaningless at this length):
    python run_qflux_base_climate.py diagnose --spinup-days 2 \
        --record-days 4 --chunk-days 2 --output-dir /tmp/qflux_smoke
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
from jax import lax

from jcm.mcb import save_carry
from jcm.mcb.coupled_controller import create_coupled_step_fn
from jcm.mcb.coupled_train import ocean_mask_from_coupler
from jcm.mcb.land_climatology import land_climatology_mode
from jcm.mcb.qflux import (
    DAYS_PER_YEAR,
    SECONDS_PER_DAY,
    calendar_month,
    fit_monthly_qflux,
    load_qflux,
    ocean_heat_capacity,
    qflux_magnitude,
    save_qflux,
    set_qflux,
    wrap_step_fn_with_sst_restoring,
)
from run_coupled_training import coupler_workflow, setup_coupled_model
from run_gradient_fidelity import git_provenance
from run_stage5_training import START_DATE

DRIFT_MAX_K_PER_60D = 0.02
BIAS_MAX_K = 0.5
N_HARMONICS = 3
GATE_FAIL_EXIT = 3


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="stage", required=True)

    d = sub.add_parser("diagnose", help="restored run -> monthly Q-flux")
    d.add_argument("--tau-days", type=float, default=5.0)
    d.add_argument("--spinup-days", type=int, default=365)
    d.add_argument("--record-days", type=int, default=1460)
    d.add_argument("--chunk-days", type=int, default=73)
    d.add_argument("--output-dir", required=True)
    d.add_argument("--qflux-name", default="qflux_monthly_t30.nc")

    c = sub.add_parser("correct",
                       help="one Newton correction from a settle run "
                            "(revision 0.3)")
    c.add_argument("--qflux", required=True, help="Q-flux that was settled")
    c.add_argument("--settle-dir", required=True,
                   help="Directory with that run's settle_fields.npz and "
                        "settle_summary.json")
    c.add_argument("--fit-start-days", type=int, default=180)
    c.add_argument("--block-days", type=int, default=60)
    c.add_argument("--output", required=True, help="Corrected Q-flux NetCDF")

    s = sub.add_parser("settle", help="free slab + Q-flux, then the gate")
    s.add_argument("--qflux", required=True, help="Q-flux NetCDF")
    s.add_argument("--days", type=int, default=3650)
    s.add_argument("--eval-days", type=int, default=1460)
    s.add_argument("--chunk-days", type=int, default=73)
    s.add_argument("--output-dir", required=True)
    s.add_argument("--base-carry-out", required=True,
                   help="Where to save the settled carry (the base carry)")
    return p.parse_args(argv)


def validate_args(args):
    """Fail fast on lengths the stages cannot use."""
    if args.stage == "correct":
        if args.block_days < 1 or args.fit_start_days < 0:
            raise SystemExit("need --block-days >= 1 and --fit-start-days >= 0")
        return
    if args.chunk_days < 1:
        raise SystemExit("--chunk-days must be >= 1")
    if args.stage == "diagnose":
        for name in ("spinup_days", "record_days"):
            if getattr(args, name) % args.chunk_days:
                raise SystemExit(f"--{name.replace('_', '-')} must be a "
                                 f"multiple of --chunk-days")
        if args.record_days < 1 or not args.tau_days > 0:
            raise SystemExit("need --record-days >= 1 and --tau-days > 0")
    else:
        if args.days % args.chunk_days:
            raise SystemExit("--days must be a multiple of --chunk-days")
        if not 2 <= args.eval_days <= args.days:
            raise SystemExit("need 2 <= --eval-days <= --days")


# ----------------------------------------------------------------- set-up ---

def build_model():
    """Coupled model on realistic terrain with everything both stages need."""
    coupler, coords, terrain, _ = setup_coupled_model(
        jdt.to_datetime(START_DATE), jdt.to_timedelta(1, "day"),
        realistic_terrain=True)
    template = coupler.initialize()      # before the step fn (land clim)
    step_fn = create_coupled_step_fn(coupler, coupler_workflow(coupler),
                                     jitted=True)
    ocn = coupler.components["ocn"]
    raw = getattr(ocn, "raw_component", ocn)
    ocean = np.asarray(ocean_mask_from_coupler(coupler)) > 0.5
    lat = np.asarray(coords.horizontal.latitudes)
    lon = np.asarray(coords.horizontal.longitudes)
    weights = np.cos(lat)[None, :] * ocean
    return {
        "coupler": coupler, "template": template, "step_fn": step_fn,
        "ocean": ocean, "fmask_ocean": np.asarray(terrain.fmask) < 0.5,
        "weights": weights / weights.sum(), "lat_deg": np.degrees(lat),
        "lon_deg": np.degrees(lon),
        "sst_clim": np.asarray(raw.SST_clim, dtype=np.float32),
        "start_offset": float(raw._compute_start_day_offset()),
        "heat_capacity": np.asarray(ocean_heat_capacity(
            template["ocn"]["state"].mixed_layer_depth)),
    }


def ocean_mean(field, weights):
    """Area-weighted ocean mean of ``(..., ix, il)`` fields."""
    return np.tensordot(np.asarray(field, dtype=np.float64), weights,
                        axes=([-2, -1], [0, 1]))


def sim_time(carry):
    return float(carry["ocn"]["state"].sim_time)


def check_finite(values, what):
    if not np.all(np.isfinite(values)):
        raise SystemExit(f"non-finite {what}: the run blew up")


# --------------------------------------------------------------- diagnose ---

def diagnose(args, m):
    restored = wrap_step_fn_with_sst_restoring(
        m["step_fn"], m["sst_clim"], m["ocean"], m["heat_capacity"],
        args.tau_days, start_offset_seconds=m["start_offset"])
    idx = jnp.arange(args.chunk_days)

    @jax.jit
    def run_chunk(carry):
        return lax.scan(restored, carry, idx)

    carry = m["template"]
    n_spin = args.spinup_days // args.chunk_days
    n_rec = args.record_days // args.chunk_days
    heat_chunks, t_mid = [], []
    obs_annual = ocean_mean(m["sst_clim"].mean(axis=2), m["weights"])
    t0 = time.time()
    for c in range(n_spin + n_rec):
        t_before = sim_time(carry)
        carry, heat = run_chunk(carry)
        heat = np.asarray(heat)
        check_finite(heat, "restoring heat")
        sst = np.asarray(carry["ocn"]["state"].sea_surface_temperature)
        check_finite(sst, "SST")
        stage = "spin-up" if c < n_spin else "record"
        if c >= n_spin:
            heat_chunks.append(heat)
            t_mid.append(m["start_offset"] + t_before
                         + (np.arange(args.chunk_days) + 0.5)
                         * SECONDS_PER_DAY)
        day = (c + 1) * args.chunk_days
        print(f"[{stage:>7}] day {day:>5}: ocean SST "
              f"{ocean_mean(sst, m['weights']):.3f} K (obs annual "
              f"{obs_annual:.3f}) | mean restoring heat "
              f"{ocean_mean(heat, m['weights']).mean():+.2f} W m-2 "
              f"| {time.time() - t0:.0f}s", flush=True)

    heat = np.concatenate(heat_chunks)
    t_mid = np.concatenate(t_mid)
    q, rms = fit_monthly_qflux(heat, t_mid, m["ocean"])
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    git = git_provenance()
    attrs = {
        "source": "run_qflux_base_climate.py diagnose",
        "preregistration": "PREREGISTRATION.md Amendment 9 revision 0.2",
        "method": (f"SST restored to forcing.nc monthly climatology, tau "
                   f"{args.tau_days} d; spin-up {args.spinup_days} d, "
                   f"record {args.record_days} d; least-squares mid-month "
                   f"values, Q = -restoring heat"),
        "git_commit": str(git.get("commit")),
        "git_dirty": str(git.get("dirty")),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    qpath = out_dir / args.qflux_name
    save_qflux(qpath, q, m["lon_deg"], m["lat_deg"], attrs)

    monthly_mean = ocean_mean(np.moveaxis(q, 2, 0), m["weights"])
    zonal = (q.mean(axis=2) * m["ocean"]).sum(axis=0) / np.maximum(
        m["ocean"].sum(axis=0), 1)
    summary = {
        "stage": "diagnose", "config": vars(args), "git": git,
        "land_climatology": land_climatology_mode(),
        "qflux_file": str(qpath),
        "ocean_mean_qflux_by_month_wm2": monthly_mean.tolist(),
        "ocean_mean_qflux_annual_wm2": float(monthly_mean.mean()),
        "max_abs_qflux_wm2": float(np.abs(q).max()),
        "zonal_mean_annual_qflux_wm2": zonal.tolist(),
        "fit_residual_rms_ocean_mean_wm2": float(ocean_mean(rms,
                                                            m["weights"])),
        "final_ocean_sst_k": float(ocean_mean(
            carry["ocn"]["state"].sea_surface_temperature, m["weights"])),
        "observed_annual_ocean_sst_k": float(obs_annual),
        "wall_s": time.time() - t0,
    }
    with open(out_dir / "diagnose_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(out_dir / "diagnose_fields.npz", q_flux=q,
                        fit_residual_rms=rms, mean_heat=heat.mean(axis=0))
    print(f"\nQ-flux -> {qpath}\n  ocean-mean annual "
          f"{summary['ocean_mean_qflux_annual_wm2']:+.2f} W m-2 (upward), "
          f"max |Q| {summary['max_abs_qflux_wm2']:.1f} W m-2, fit residual "
          f"rms {summary['fit_residual_rms_ocean_mean_wm2']:.2f} W m-2")
    return 0


# ----------------------------------------------------------------- settle ---

def drift_per_60d(series, n_harmonics=N_HARMONICS):
    """Linear trend (K per 60 days) of a daily series with annual harmonics."""
    y = np.asarray(series, dtype=np.float64)
    t = np.arange(y.size, dtype=np.float64)
    cols = [np.ones_like(t), t]
    for k in range(1, n_harmonics + 1):
        arg = 2.0 * np.pi * k * t / DAYS_PER_YEAR
        cols += [np.cos(arg), np.sin(arg)]
    coef, *_ = np.linalg.lstsq(np.stack(cols, axis=1), y, rcond=None)
    return float(coef[1] * 60.0)


def harmonic_design(t, n_harmonics=N_HARMONICS):
    """Columns [1, cos, sin, ...] of the annual harmonics at days ``t``."""
    cols = [np.ones_like(t)]
    for k in range(1, n_harmonics + 1):
        arg = 2.0 * np.pi * k * t / DAYS_PER_YEAR
        cols += [np.cos(arg), np.sin(arg)]
    return np.stack(cols, axis=1)


def deseasonalize(series, fit_last_days, n_harmonics=N_HARMONICS):
    """Remove the seasonal cycle fitted on the final ``fit_last_days``."""
    y = np.asarray(series, dtype=np.float64)
    t = np.arange(y.size, dtype=np.float64)
    sl = slice(y.size - fit_last_days, y.size)
    coef, *_ = np.linalg.lstsq(harmonic_design(t[sl], n_harmonics), y[sl],
                               rcond=None)
    return y - harmonic_design(t, n_harmonics)[:, 1:] @ coef[1:]


def fit_exponential_approach(t, y, tau_grid=None):
    """Least-squares fit of ``y = t_inf + amplitude * exp(-t / tau)``.

    For each tau on a fine geometric grid the model is linear in
    (t_inf, amplitude); the tau with the smallest squared error wins.
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    grid = np.geomspace(30.0, 30000.0, 3000) if tau_grid is None else tau_grid
    best = None
    for tau in grid:
        x = np.stack([np.ones_like(t), np.exp(-t / tau)], axis=1)
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        sse = float(np.sum((x @ coef - y) ** 2))
        if best is None or sse < best[0]:
            best = (sse, float(tau), coef)
    sse, tau, coef = best
    return {"tau_days": tau, "t_inf": float(coef[0]),
            "amplitude": float(coef[1]), "sse": sse,
            "tau_at_grid_edge": bool(tau in (grid[0], grid[-1]))}


def block_means(series, block_days):
    """Means over consecutive blocks, with the block-centre days."""
    y = np.asarray(series, dtype=np.float64)
    n = y.size // block_days
    t = np.arange(n * block_days, dtype=np.float64)
    return (t.reshape(n, block_days).mean(axis=1),
            y[:n * block_days].reshape(n, block_days).mean(axis=1))


def newton_correction(daily_ocean_sst, eval_days, observed, heat_capacity,
                      fit_start_days=180, block_days=60):
    """Revision 0.3: uniform heat correction from one settle run.

    The run's deseasonalized ocean-mean SST approaches its own equilibrium
    ``t_inf`` with e-folding time tau. For a slab, ``lambda = C / tau`` is
    the heat (W m-2) per kelvin of ocean-mean change, so the uniform heat
    that moves the equilibrium onto the observations is
    ``lambda * (observed - t_inf)`` (positive = into the ocean).
    """
    tm, ym = block_means(deseasonalize(daily_ocean_sst, eval_days),
                         block_days)
    sel = tm >= fit_start_days
    fit = fit_exponential_approach(tm[sel], ym[sel])
    lam = heat_capacity / (fit["tau_days"] * SECONDS_PER_DAY)
    return {"fit": fit, "lambda_wm2_per_k": float(lam),
            "equilibrium_bias_k": float(fit["t_inf"] - observed),
            "heat_into_ocean_wm2": float(lam * (observed - fit["t_inf"]))}


def gate(drift, bias):
    """Return the registered verdict (revision 0.2) for the two gated numbers."""
    g1 = abs(drift) < DRIFT_MAX_K_PER_60D
    g2 = abs(bias) < BIAS_MAX_K
    return {"G1_drift_pass": bool(g1), "G2_bias_pass": bool(g2),
            "pass": bool(g1 and g2)}


def settle(args, m):
    q = load_qflux(args.qflux)
    carry = set_qflux(m["template"], q)
    step_fn = m["step_fn"]
    idx = jnp.arange(args.chunk_days)

    def body(c, i):
        c, _ = step_fn(c, i)
        return c, c["ocn"]["state"].sea_surface_temperature

    @jax.jit
    def run_chunk(c):
        return lax.scan(body, c, idx)

    n_chunks = args.days // args.chunk_days
    eval_start = args.days - args.eval_days
    daily_mean, eval_days_t = [], []
    ix, il = m["ocean"].shape
    sum_map = np.zeros((ix, il))
    month_sum = np.zeros((12, ix, il))
    month_n = np.zeros(12)
    sst_min, sst_max = np.inf, -np.inf
    land_range = None
    obs_annual_map = m["sst_clim"].mean(axis=2).astype(np.float64)
    obs_annual = ocean_mean(obs_annual_map, m["weights"])
    t0 = time.time()
    for c in range(n_chunks):
        t_before = sim_time(carry)
        carry, ssts = run_chunk(carry)
        ssts = np.asarray(ssts, dtype=np.float64)
        check_finite(ssts, "SST")
        means = ocean_mean(ssts, m["weights"])
        daily_mean.append(means)
        first_day = c * args.chunk_days
        t_end = (m["start_offset"] + t_before
                 + (np.arange(args.chunk_days) + 1.0) * SECONDS_PER_DAY)
        in_eval = first_day + np.arange(args.chunk_days) >= eval_start
        if in_eval.any():
            sel = ssts[in_eval]
            sum_map += sel.sum(axis=0)
            months = calendar_month(t_end[in_eval] - 0.5 * SECONDS_PER_DAY)
            for mo in range(12):
                hit = months == mo
                month_sum[mo] += sel[hit].sum(axis=0)
                month_n[mo] += hit.sum()
            oc = sel[:, m["ocean"]]
            sst_min, sst_max = min(sst_min, oc.min()), max(sst_max, oc.max())
            eval_days_t.append(t_end[in_eval])
        if "lnd" in carry:
            lt = np.asarray(carry["lnd"]["state"].land_surface_temperature)
            land_range = [float(lt[~m["ocean"]].min()),
                          float(lt[~m["ocean"]].max())]
        day = (c + 1) * args.chunk_days
        recent = np.concatenate(daily_mean)[-min(day, 730):]
        print(f"day {day:>5}: ocean SST {means[-1]:.3f} K (obs annual "
              f"{obs_annual:.3f}) | trend over last {recent.size} d "
              f"{drift_per_60d(recent) if recent.size >= 30 else float('nan'):+.4f}"
              f" K/60d | {time.time() - t0:.0f}s", flush=True)

    series = np.concatenate(daily_mean)
    window = series[eval_start:]
    drift = drift_per_60d(window)
    bias = float(window.mean() - obs_annual)
    verdict = gate(drift, bias)
    n_eval = args.eval_days
    annual_map = sum_map / n_eval
    err_map = np.where(m["ocean"], annual_map - obs_annual_map, 0.0)
    w = m["weights"]
    rms_annual = float(np.sqrt((w * err_map ** 2).sum()))
    covered = month_n > 0
    month_mean = month_sum[covered] / month_n[covered, None, None]
    month_obs = np.moveaxis(m["sst_clim"], 2, 0)[covered].astype(np.float64)
    seas_err = (month_mean - month_mean.mean(axis=0)) - (
        month_obs - month_obs.mean(axis=0))
    rms_seasonal = float(np.sqrt((w[None] * seas_err ** 2).sum(axis=(1, 2))
                                 .mean())) if covered.sum() > 1 else None
    fm = m["fmask_ocean"] & m["ocean"]
    summary = {
        "stage": "settle", "config": vars(args), "git": git_provenance(),
        "land_climatology": land_climatology_mode(),
        "qflux_file": args.qflux,
        "qflux_max_abs_wm2": qflux_magnitude(carry),
        "gate": {"drift_k_per_60d": drift, "drift_max": DRIFT_MAX_K_PER_60D,
                 "bias_k": bias, "bias_max": BIAS_MAX_K, **verdict},
        "eval_window_days": n_eval,
        "ocean_sst_eval_mean_k": float(window.mean()),
        "observed_annual_ocean_sst_k": float(obs_annual),
        "reported": {
            "rms_annual_mean_error_k": rms_annual,
            "rms_seasonal_cycle_error_k": rms_seasonal,
            "cells_abs_annual_error_gt_2k": int((np.abs(err_map) > 2.0)
                                                .sum()),
            "ocean_cells": int(m["ocean"].sum()),
            "sst_range_k": [float(sst_min), float(sst_max)],
            "land_temperature_range_k": land_range,
            "simple_mean_fmask_ocean_model_k": float(annual_map[fm].mean()),
            "simple_mean_fmask_ocean_observed_k": float(
                obs_annual_map[fm].mean()),
        },
        "daily_ocean_sst_first_last_k": [float(series[0]),
                                         float(series[-1])],
        "wall_s": time.time() - t0,
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "settle_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(out_dir / "settle_fields.npz",
                        daily_ocean_sst=series, annual_mean_map=annual_map,
                        annual_error_map=err_map,
                        monthly_mean_maps=month_sum / np.maximum(
                            month_n, 1)[:, None, None])
    base = Path(args.base_carry_out)
    base.parent.mkdir(parents=True, exist_ok=True)
    save_carry(carry, str(base))
    print(f"\nGATE: drift {drift:+.4f} K/60d (|.| < {DRIFT_MAX_K_PER_60D}: "
          f"{'PASS' if verdict['G1_drift_pass'] else 'FAIL'}) | bias "
          f"{bias:+.3f} K (|.| < {BIAS_MAX_K}: "
          f"{'PASS' if verdict['G2_bias_pass'] else 'FAIL'}) -> "
          f"{'PASS' if verdict['pass'] else 'FAIL'}")
    print(f"  RMS annual-mean error {rms_annual:.2f} K | settled carry -> "
          f"{base}")
    return 0 if verdict["pass"] else GATE_FAIL_EXIT


# ---------------------------------------------------------------- correct ---

def correct(args, m):
    """Revision 0.3: one uniform Newton correction of a settled Q-flux."""
    settle_dir = Path(args.settle_dir)
    with open(settle_dir / "settle_summary.json") as f:
        summary = json.load(f)
    series = np.load(settle_dir / "settle_fields.npz")["daily_ocean_sst"]
    eval_days = int(summary["eval_window_days"])
    observed = float(summary["observed_annual_ocean_sst_k"])
    c_eff = float(np.sum(m["heat_capacity"] * m["weights"]))
    result = newton_correction(series, eval_days, observed, c_eff,
                               args.fit_start_days, args.block_days)
    if result["fit"]["tau_at_grid_edge"]:
        raise SystemExit("exponential fit hit the edge of its tau grid: the "
                         "run did not approach an equilibrium; no correction")
    robustness = {
        str(start): newton_correction(series, eval_days, observed, c_eff,
                                      start, args.block_days)
        ["heat_into_ocean_wm2"] for start in (365, 730)}
    dq = result["heat_into_ocean_wm2"]
    q1 = load_qflux(args.qflux)
    # Upward-positive convention: more heat INTO the ocean lowers q_flux.
    q2 = np.where(m["ocean"][:, :, None], q1 - dq, 0.0).astype(np.float32)
    git = git_provenance()
    attrs = {
        "source": "run_qflux_base_climate.py correct",
        "preregistration": "PREREGISTRATION.md Amendment 9 revision 0.3",
        "method": (f"uniform Newton correction of {args.qflux}: "
                   f"{dq:+.3f} W m-2 into every ocean cell, every month"),
        "git_commit": str(git.get("commit")),
        "git_dirty": str(git.get("dirty")),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_qflux(out, q2, m["lon_deg"], m["lat_deg"], attrs)
    record = {"stage": "correct", "config": vars(args), "git": git,
              "input_qflux": args.qflux, "settle_dir": str(settle_dir),
              "observed_annual_ocean_sst_k": observed,
              "effective_heat_capacity_j_m2_k": c_eff, **result,
              "heat_into_ocean_by_fit_start_wm2": {
                  str(args.fit_start_days): dq, **robustness},
              "output_qflux": str(out)}
    with open(out.with_name(out.stem + "_correction.json"), "w") as f:
        json.dump(record, f, indent=2)
    fit = result["fit"]
    print(f"fit: tau {fit['tau_days']:.0f} d, equilibrium "
          f"{fit['t_inf']:.3f} K (bias {result['equilibrium_bias_k']:+.3f} "
          f"K), lambda {result['lambda_wm2_per_k']:.3f} W m-2 K-1")
    print(f"correction: {dq:+.3f} W m-2 into the ocean (fit start 365 d: "
          f"{robustness['365']:+.3f}, 730 d: {robustness['730']:+.3f}) -> "
          f"{out}")
    return 0


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    print(f"Q-FLUX BASE CLIMATE: {args.stage}  (JAX devices "
          f"{jax.devices()})", flush=True)
    m = build_model()
    stages = {"diagnose": diagnose, "correct": correct, "settle": settle}
    return stages[args.stage](args, m)


if __name__ == "__main__":
    sys.exit(main())
