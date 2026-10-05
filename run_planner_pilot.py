#!/usr/bin/env python
r"""Step 23: the planner pilot, on training states only (MCB_PROJECT_REPORT.md Part 18).

The pilot fixes the settings that revision 1 of Amendment 9 then freezes:
the warming, the scoring representation and window, the planner's
optimizer, iterations, copies and look-ahead, its objective, and its
penalty weights. It also checks the planner's response maps against
Experiment 1's brute-force maps. It never touches an evaluation state.

Three stages:

``offline`` (laptop, no GPU)
    * The maps check: Experiment 1's forward-mode response maps (W = 14 and
      plain backpropagation) against its brute-force maps, for the
      look-ahead's time-mean SST, over the whole ocean and far from each
      band, with the brute-force maps' own noise measured by splitting
      their members.
    * Signal and noise: from the training references (a steady 4 W m-2
      probe, five members), the warming signal and the weather noise of
      five-member means for every scored variable, domain and
      representation (grid-point map, zonal-mean profile, five band means,
      ocean mean), and the ramp warming each rule needs.

``ic`` (GPU, one training state per process)
    From one training state, the uncontrolled run with the pilot ramp
    warming is spun up to day ``--start-day`` (re-plan 9 of 13). From there
    the planner re-plans once in each candidate setting, and every
    resulting band setting is then held for 120 days in ``--members``
    weather samples (the same samples for every candidate), which gives
    its TRUE look-ahead objective. Candidates:
    - Gauss-Newton (GN), 60-day look-ahead, grid-point-map objective:
      iterates 1, 2 and 3 (3 copies);
    - GN 60 days, zonal-mean objective (3 iterations);
    - GN 120 days, map and zonal objectives (3 iterations);
    - Dubey et al.'s Adam, 60 days, map objective (15 steps);
    - from GN 60's last linearization (the response is nearly linear in
      the settings, Experiment 1 and the toy tests): each single copy's
      solution (the one-copy planner), the best uniform setting (where-blind),
      and the zonal solution with the translated penalty weights;
    - no brightening.

``analyze`` (laptop)
    Applies the rules below to all ``ic`` outputs and writes the decisions.

Decision rules, written before any ``ic`` output existed (2026-10-05).
``J`` is the true objective (alpha = 1, beta = 0.5) of the member-mean
time-mean SST error over the look-ahead; differences are paired over the
training states, with SE = sd / sqrt(n):

R1  Optimizer. GN, unless Adam's J60 (map) is lower than GN 60 (map, 3
    iterations) by more than 2 SE and by at least 5%. Iterations: the
    smallest k in {1, 2, 3} whose J60 is within 2% of iteration 3's or
    within 1 SE of it.
R2  Copies. One copy if the one-copy solutions' J60 is within 5% of the
    three-copy solution's (mean over states and copies) and the median
    noise-to-signal ratio at the first iterate is below 0.5; otherwise 3.
R3  Look-ahead of the long planner. 120 days if GN 120's J120 (in the
    chosen representation) is lower than GN 60's by more than 2 SE and by
    at least 5%; otherwise 60 days.
R4  Score representation. The finest of grid-point map, zonal-mean profile,
    band means and ocean mean whose ocean-SST signal is at least 3 times the
    noise (mean-square) in the scoring window with five members at the
    chosen warming. The planner's objective uses the same representation,
    unless the map-objective planner beats it on that representation's
    J60 by more than 2 SE.
R5  Warming. A ramp from 0 at day 0 to F_end at day 182 (Experiment 3a's
    "steadily growing warming"). F_end is the smallest of 4, 6 and 8 W m-2
    that meets R4's 3-to-1 rule in the zonal-mean profile; the pilot then
    reports the brightening it needs, as a share of the cap, which must
    stay below 40% at nominal strength so that a hidden strength of 0.5
    still leaves headroom.
R6  Scoring window [98, 182): the last 12 weeks of a 182-day episode;
    five members, from the references' seeds.
R7  Penalties. Dubey et al.'s weights translated into albedo units:
    mu = 0.01 s^2 and lam = 0.1 s^2, with s the mean over bands and states
    of the ocean-mean response (K per unit albedo) over the chosen
    look-ahead. Kept unless, in the linearized solution, mu costs more than
    10% of the objective reduction; then mu and lam are divided by 10 until
    it costs at most 10%.
R8  Rainfall, evaporation and land temperature are scored on a domain only
    if their zonal-mean signal is at least the noise (mean-square) at the
    chosen warming; otherwise they are reported descriptively.

Example (GX10, alongside vLLM):
    python run_planner_pilot.py ic --ic-dir mcb_experiments_gpu/ics_macro/exp3_train \
        --ic-position 0 --references \
        mcb_experiments_gpu/test_world_refs/exp3_train/ic0002_references.npz \
        --output-dir mcb_experiments_gpu/pilot_step23
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

EPISODE_DAYS = 182
SEGMENT_DAYS = 14
START_DAY = 112             # re-plan 9 of 13
TRUTH_DAYS = 120
LOOKAHEADS = (60, 120)
SCORE_WINDOW = (98, 182)
RAMP_CANDIDATES_WM2 = (4.0, 6.0, 8.0)
PROBE_WM2 = 4.0
TRUTH_SEED0 = 95000
ALPHA, BETA = 1.0, 0.5
DUBEY_MU, DUBEY_LAM = 0.01, 0.1


# --- Representations -------------------------------------------------------

def ocean_weights(latitudes_rad, ocean):
    w = np.cos(np.asarray(latitudes_rad))[None, :] * np.asarray(ocean)
    return w / w.sum()


def zonal_project(x, ocean):
    """Replace every ocean cell of ``x`` (..., ix, il) by its latitude's ocean mean."""
    ocean = np.asarray(ocean, np.float64)
    counts = np.maximum(ocean.sum(axis=0), 1.0)
    zonal = (np.asarray(x, np.float64) * ocean).sum(axis=-2) / counts
    return zonal[..., None, :] * ocean


def zonal_project_jacobian(jac, ocean):
    """``zonal_project`` for a Jacobian (..., ix, il, K)."""
    moved = np.moveaxis(np.asarray(jac, np.float64), -1, 0)
    return np.moveaxis(zonal_project(moved, ocean), 0, -1)


def objective(error, weights, alpha=ALPHA, beta=BETA):
    """``alpha <e>^2 + beta Var(e)`` over the last two axes (weights sum to 1)."""
    e = np.asarray(error, np.float64)
    mean = np.sum(weights * e, axis=(-2, -1))
    var = np.sum(weights * (e - mean[..., None, None]) ** 2, axis=(-2, -1))
    return alpha * mean ** 2 + beta * var


def represent(error, ocean, weights, kind):
    """Return ``error`` in a representation as (field, weights) for ``objective``."""
    if kind == "map":
        return error, weights
    if kind == "zonal":
        return zonal_project(error, ocean), weights
    raise ValueError(kind)


# --- Offline stage ------------------------------------------------------------

def maps_check(exp1_maps: str, band_patterns_fn) -> dict:
    """Experiment 1's gradient response maps against its brute-force maps."""
    z = np.load(exp1_maps)
    fd, jac = z["fd_sst_blocks"], z["jac_sst_blocks"]
    ocean, lats = z["ocean_mask"], z["latitudes_rad"]
    w = ocean_weights(lats, ocean)
    patterns = np.asarray(band_patterns_fn(lats, ocean))
    delta = 0.03                     # Experiment 1's central difference
    k = patterns.shape[0]
    truth = np.stack([(fd[:, :, 2 + 2 * j] - fd[:, :, 3 + 2 * j]) / (2 * delta)
                      for j in range(k)], axis=2)   # ic, member, band, block
    block_days = 5
    out = {}
    for horizon in LOOKAHEADS:
        nb = horizon // block_days
        t_members = truth[:, :, :, :nb].mean(axis=3)       # ic, mem, band
        t_mean = t_members.mean(axis=1)
        half = t_members.shape[1] // 2
        for w_idx, w_name in ((2, "W14"), (3, "BPTT")):
            grad = jac[:, 0, w_idx, :, :nb].mean(axis=2)    # ic, band
            rows = []
            for b in range(k):
                row = {"band": b}
                for region in ("ocean", "far"):
                    mask = ocean > 0 if region == "ocean" else (
                        (patterns[b] < 0.05) & (ocean > 0))
                    ww = w * mask
                    ww = ww / ww.sum()
                    gt, gg = t_mean[:, b], grad[:, b]
                    mt = np.sum(ww * gt, axis=(1, 2))
                    mg = np.sum(ww * gg, axis=(1, 2))
                    a, c = gt - mt[:, None, None], gg - mg[:, None, None]
                    r = np.sum(ww * a * c) / np.sqrt(np.sum(ww * a * a)
                                                     * np.sum(ww * c * c))
                    h1 = t_members[:, :half, b].mean(axis=1)
                    h2 = t_members[:, half:, b].mean(axis=1)
                    h1 = h1 - np.sum(ww * h1, axis=(1, 2))[:, None, None]
                    h2 = h2 - np.sum(ww * h2, axis=(1, 2))[:, None, None]
                    rh = np.sum(ww * h1 * h2) / np.sqrt(
                        np.sum(ww * h1 * h1) * np.sum(ww * h2 * h2))
                    reliability = 2 * rh / (1 + rh)
                    diff = mt - mg
                    row[region] = {
                        "pattern_corr": float(r),
                        "truth_reliability": float(reliability),
                        "corr_over_ceiling": (float(r / np.sqrt(reliability))
                                              if reliability > 0.05 else None),
                        "slope_truth_on_grad": float(
                            np.sum(ww * a * c) / np.sum(ww * c * c)),
                        "mean_truth": float(mt.mean()),
                        "mean_grad": float(mg.mean()),
                        "mean_diff_se": float(diff.std(ddof=1)
                                              / np.sqrt(diff.size)),
                    }
                rows.append(row)
            out[f"{horizon}d_{w_name}"] = rows
    return out


def signal_noise(refs_dir: str, window=SCORE_WINDOW) -> dict:
    """Probe signal and five-member noise (mean-square) per variable, domain, representation."""
    refs_dir = Path(refs_dir)
    grid = np.load(refs_dir / "grid.npz")
    ocean, land = grid["ocean_mask"], grid["land_mask"]
    lats = grid["latitudes_rad"]
    domains = {"ocean": ocean, "land": land, "global": np.ones_like(ocean)}
    pairs = [("sst", "ocean"), ("land_temperature", "land"),
             ("precipitation", "land"), ("precipitation", "global"),
             ("evaporation", "land"), ("evaporation", "global")]
    a, b = window
    files = sorted(refs_dir.glob("ic*_references.npz"))
    acc = {}
    probe_mean = []
    for f in files:
        z = np.load(f)
        for var, dom in pairs:
            mask = domains[dom]
            w = ocean_weights(lats, mask)
            counts = np.maximum(mask.sum(axis=0), 1.0)
            wz = np.cos(lats) * mask.sum(axis=0)
            wz = wz / wz.sum()
            sig = (z[f"warmed_{var}"][a:b] - z[f"normal_{var}"][a:b]).mean(0)
            mem = {s: z[f"{s}_members_{var}"][:, a // 5:b // 5].mean(axis=1)
                   for s in ("normal", "warmed")}
            for kind in ("map", "zonal", "mean"):
                if kind == "map":
                    def f_(x):
                        return x
                    wk = w
                elif kind == "zonal":
                    def f_(x):
                        return (x * mask).sum(axis=-2) / counts
                    wk = wz
                else:
                    def f_(x):
                        return np.sum(w * x, axis=(-2, -1))
                    wk = None
                var_m = 0.5 * sum(f_(mem[s]).var(axis=0, ddof=1)
                                  for s in mem)
                noise = (2 * var_m / 5 if wk is None
                         else np.sum(wk * 2 * var_m / 5))
                s2 = f_(sig) ** 2 if wk is None else np.sum(wk * f_(sig) ** 2)
                acc.setdefault((var, dom, kind), []).append(
                    (float(s2 - noise), float(noise)))
        w = ocean_weights(lats, ocean)
        probe_mean.append(np.sum(w * (z["warmed_sst"] - z["normal_sst"]),
                                 axis=(1, 2)))
    step = np.mean(probe_mean, axis=0) / PROBE_WM2     # K per (W m-2), daily
    ramp_unit = np.cumsum(step)                        # K per (W m-2 / day)
    probe_window = PROBE_WM2 * step[a:b].mean()
    table = {}
    for (var, dom, kind), vals in acc.items():
        s, n = np.mean(vals, axis=0)
        table[f"{var}@{dom}:{kind}"] = {"probe_signal_ms": s, "noise_ms": n,
                                        "probe_ratio": s / n}
    ramps = {}
    for f_end in RAMP_CANDIDATES_WM2:
        anomaly = f_end / EPISODE_DAYS * ramp_unit
        scale = (anomaly[a:b].mean() / probe_window) ** 2
        ramps[str(f_end)] = {
            "ocean_mean_anomaly_day182_K": float(anomaly[EPISODE_DAYS - 1]),
            "window_mean_anomaly_K": float(anomaly[a:b].mean()),
            "ms_scale_vs_probe": float(scale),
            "ratios": {key: v["probe_ratio"] * scale
                       for key, v in table.items()}}
    return {"window": list(window), "states": [f.name for f in files],
            "probe_wm2": PROBE_WM2, "probe": table, "ramps": ramps,
            "step_response_K_per_wm2": {str(d): float(step[d - 1])
                                        for d in (30, 60, 120, 182, 240)}}


def choose_warming(sn: dict) -> dict:
    """Apply R4 (representation), R5 (ramp end) and R8 (other variables)."""
    f_end = None
    for f in RAMP_CANDIDATES_WM2:
        if sn["ramps"][str(f)]["ratios"]["sst@ocean:zonal"] >= 3.0:
            f_end = f
            break
    if f_end is None:
        f_end = RAMP_CANDIDATES_WM2[-1]
    ratios = sn["ramps"][str(f_end)]["ratios"]
    representation = next((k for k in ("map", "zonal", "mean")
                           if ratios[f"sst@ocean:{k}"] >= 3.0), "mean")
    scored = {key.split(":")[0]: bool(ratios[key] >= 1.0) for key in ratios
              if key.endswith(":zonal") and not key.startswith("sst@")}
    return {"ramp_end_wm2": f_end,
            "ramp_wm2_per_day": f_end / EPISODE_DAYS,
            "representation": representation,
            "sst_ratios": {k: ratios[f"sst@ocean:{k}"]
                           for k in ("map", "zonal", "mean")},
            "other_variables_scored_zonal": scored,
            "other_variables_ratio_zonal": {
                key.split(":")[0]: ratios[key] for key in ratios
                if key.endswith(":zonal") and not key.startswith("sst@")}}


def stage_offline(args):
    from jcm.mcb.band_basis import gaussian_band_patterns
    import jax.numpy as jnp

    def bands(lats, ocean):
        return gaussian_band_patterns(jnp.asarray(lats), jnp.asarray(ocean))

    sn = signal_noise(args.refs_dir)
    out = {"stage": "offline", "rules": __doc__.split("Decision rules")[1],
           "created_utc": datetime.now(timezone.utc).isoformat(),
           "maps_check": maps_check(args.exp1_maps, bands),
           "signal_noise": sn, "warming_choice": choose_warming(sn)}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print_offline(out)
    print(f"-> {path}")


def print_offline(out):
    print("Maps check (time-mean SST response per band; truth = brute force):")
    for key, rows in out["maps_check"].items():
        cells = []
        for r in rows:
            o, fa = r["ocean"], r["far"]
            cells.append(f"b{r['band']} r={o['pattern_corr']:.2f}"
                         f" slope={o['slope_truth_on_grad']:.2f}"
                         f" far {fa['mean_truth']:+.3f}/{fa['mean_grad']:+.3f}"
                         f"±{fa['mean_diff_se']:.3f}")
        print(f"  {key}: " + " | ".join(cells))
    sn = out["signal_noise"]
    print(f"Signal/noise (mean-square), window {sn['window']}, 5 members:")
    for f, ramp in sn["ramps"].items():
        r = ramp["ratios"]
        print(f"  ramp to {f} W/m2: SST map {r['sst@ocean:map']:.2f}, zonal "
              f"{r['sst@ocean:zonal']:.2f}, mean {r['sst@ocean:mean']:.1f}; "
              f"window anomaly {ramp['window_mean_anomaly_K']:.3f} K")
    print("  warming choice:", json.dumps(out["warming_choice"]))


# --- GPU stage ----------------------------------------------------------------

def gn_solve(planner, copies, start, target_anomaly, weights, ocean, cap,
             iterations, project, damping=1e-3, mu=0.0):
    """Gauss-Newton as in ``Planner._gauss_newton``, optionally on zonal means.

    Returns every iterate and the per-copy linearization at the last point
    evaluated (residuals and their Jacobians), for linearized candidates.
    """
    from jcm.mcb.planner import gauss_newton_residuals, noise_to_signal

    a = np.clip(np.asarray(start, np.float64), 0.0, cap)
    zero = np.zeros_like(a)
    iterates, values, nts, last = [], [], None, None
    target = zonal_project(target_anomaly, ocean) if project else \
        np.asarray(target_anomaly, np.float64)
    for it in range(iterations):
        jacs, maps = planner.evaluate_copies(a, copies)
        if project:
            jacs = zonal_project_jacobian(jacs, ocean)
            maps = zonal_project(maps, ocean)
        lin, grads, total = [], [], 0.0
        for jac, mean_map in zip(jacs, maps):
            r, big_r = gauss_newton_residuals(mean_map - target, jac, weights,
                                              a, zero, ALPHA, BETA, mu, 0.0)
            lin.append((r, big_r))
            grads.append(2.0 * big_r.T @ r)
            total += float(r @ r)
        if it == 0:
            nts = noise_to_signal(grads)
        values.append(total / len(lin))
        normal = sum(br.T @ br for _, br in lin) / len(lin)
        rhs = sum(br.T @ r for r, br in lin) / len(lin)
        lm = damping * np.trace(normal) / a.size
        a = np.clip(a + np.linalg.solve(normal + lm * np.eye(a.size), -rhs),
                    0.0, cap)
        iterates.append(a.copy())
        last = {"point": iterates[-2] if it else np.asarray(start, np.float64),
                "lin": lin, "jacs": jacs}
    return {"iterates": iterates, "values": values, "noise_to_signal": nts,
            "last": last}


def linear_solve(lin, point, cap, mu=0.0, uniform=False, damping=1e-3):
    """Solve the linearized least squares around ``point`` (optionally uniform)."""
    point = np.asarray(point, np.float64)
    k = point.size
    rows_r = [r for r, _ in lin]
    rows_j = [br for _, br in lin]
    big_r = np.concatenate(rows_j) / np.sqrt(len(lin))
    r = np.concatenate(rows_r) / np.sqrt(len(lin))
    if uniform:
        d = big_r @ np.ones(k)
        u0 = float(point.mean())
        # residual at a = u * 1:  r + R (u 1 - point)
        base = r + big_r @ (u0 * np.ones(k) - point)
        du = -float(d @ base) / float(d @ d)
        return np.full(k, np.clip(u0 + du, 0.0, cap))
    normal = big_r.T @ big_r + mu * np.eye(k)
    rhs = big_r.T @ r + mu * point
    lm = damping * np.trace(normal) / k
    return np.clip(point + np.linalg.solve(normal + lm * np.eye(k), -rhs),
                   0.0, cap)


def stage_ic(args):
    import jax
    import jax.numpy as jnp

    from jcm.mcb.gradient_fidelity import MAP_REFERENCE_K
    from jcm.mcb.planner import Planner, PlannerConfig
    from jcm.mcb.test_world import (
        BRIGHTENING_CAP,
        EpisodeState,
        domain_weights,
        make_segment_fn,
        make_warming,
        perturb_member,
    )
    from run_gradient_fidelity import git_provenance, load_manifest_ics
    from run_test_world import build_model

    print(f"JAX devices: {jax.devices()}", flush=True)
    t_all = time.time()
    m = build_model()
    # Load only up to the needed state (each carry is about 24 MB).
    _, ics = load_manifest_ics(args.ic_dir, "all", args.ic_position + 1,
                               m["template"])
    entry, carry0 = ics[args.ic_position]
    ref_path = Path(args.references)
    if ref_path.name != f"ic{entry['index']:04d}_references.npz":
        raise SystemExit(f"references {ref_path.name} do not match IC "
                         f"{entry['index']}")
    if entry["split"] != "train" or "eval" in str(args.ic_dir):
        raise SystemExit("the pilot uses training states only")
    refs = np.load(ref_path, allow_pickle=False)
    normal_sst = refs["normal_sst"]
    las = tuple(args.lookaheads)
    tags = dict(zip(las, ("S", "L")))
    if args.truth_days < max(las):
        raise SystemExit("--truth-days must cover the longest look-ahead")
    if normal_sst.shape[0] < args.start_day + max(las):
        raise SystemExit("references too short for the look-ahead")
    ocean = np.asarray(m["ocean"], np.float64)
    weights = np.asarray(domain_weights(m["lats"], m["ocean"],
                                        m["land"])["ocean"], np.float64)
    q_base = carry0["ocn"]["forcing"].q_flux
    ramp = args.ramp_end_wm2 / EPISODE_DAYS
    warming = make_warming(carry0, m["ocean"], 0.0, ramp)
    cap = BRIGHTENING_CAP
    k = int(m["patterns"].shape[0])
    one = jnp.asarray(1.0, jnp.float32)
    timing = {}
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Spin up: the uncontrolled warmed run to the re-plan day.
    t0 = time.time()
    spin = make_segment_fn(m["step_fn"], m["patterns"], args.start_day)
    carry_d, _ = spin(carry0, jnp.zeros((k,), jnp.float32), one, q_base,
                      warming)
    jax.block_until_ready(carry_d)
    timing["spinup_s"] = time.time() - t0
    day = args.start_day

    def target_anomaly(lookahead):
        return (normal_sst[day:day + lookahead].mean(axis=0).astype(np.float64)
                - MAP_REFERENCE_K)

    candidates, info = {}, {}
    seed_offset = 100 * entry["index"]
    for lookahead in las:
        cfg = PlannerConfig(lookahead_days=lookahead, optimizer="gauss_newton",
                            iterations=3, copies=args.copies)
        planner = Planner(m["step_fn"], m["patterns"], weights, normal_sst,
                          q_base, warming, 1.0, cfg, seed_offset=seed_offset)
        copies = planner.prepare_copies(carry_d, day)
        start = np.full(k, cfg.first_guess * cap)
        for project in (False, True):
            name = f"gn{tags[lookahead]}_{'zonal' if project else 'map'}"
            t0 = time.time()
            res = gn_solve(planner, copies, start, target_anomaly(lookahead),
                           weights, ocean, cap, 3, project)
            timing[name + "_s"] = time.time() - t0
            info[name] = {"values": res["values"],
                          "noise_to_signal": res["noise_to_signal"],
                          "iterates": [a.tolist() for a in res["iterates"]]}
            candidates[name] = res["iterates"][-1]
            print(f"  {name}: {np.round(res['iterates'][-1], 4)} "
                  f"N/S {res['noise_to_signal']:.2f} "
                  f"[{timing[name + '_s']:.0f}s]", flush=True)
            if lookahead != las[0]:
                continue
            last = res["last"]
            jac_mean = np.mean(last["jacs"], axis=0)        # (ix, il, K)
            info[name]["band_ocean_mean_response"] = np.tensordot(
                weights, jac_mean, axes=([0, 1], [0, 1])).tolist()
            if not project:
                for it in (0, 1):
                    candidates[f"gnS_map_it{it + 1}"] = res["iterates"][it]
                for c, single in enumerate(last["lin"][:args.single_copies]):
                    candidates[f"gnS_map_copy{c}"] = linear_solve(
                        [single], last["point"], cap)
                candidates["uniformS"] = linear_solve(last["lin"],
                                                       last["point"], cap,
                                                       uniform=True)
                np.savez_compressed(
                    Path(args.output_dir)
                    / f"ic{entry['index']:04d}_gnS_jacobians.npz",
                    jacs=np.asarray(last["jacs"], np.float32),
                    point=last["point"])
            else:
                s = float(np.mean(np.abs(
                    info[name]["band_ocean_mean_response"])))
                mu = DUBEY_MU * s ** 2
                info["translated_penalties"] = {"s": s, "mu": mu,
                                                "lam": DUBEY_LAM * s ** 2}
                candidates["gnS_zonal_mu"] = linear_solve(
                    last["lin"], last["point"], cap, mu=mu)

    if not args.skip_adam:
        cfg = PlannerConfig(lookahead_days=las[0], optimizer="adam",
                            iterations=15, copies=args.copies,
                            batch_copies=True)
        planner = Planner(m["step_fn"], m["patterns"], weights, normal_sst,
                          q_base, warming, 1.0, cfg, seed_offset=seed_offset)
        t0 = time.time()
        state = EpisodeState(segment=0, day=day, carry=carry_d,
                             previous=np.zeros(k, np.float32),
                             last_fields=None)
        candidates["adamS_map"] = np.asarray(planner(state), np.float64)
        timing["adamS_map_s"] = time.time() - t0
        info["adamS_map"] = planner.log[-1]
        print(f"  adamS_map: {np.round(candidates['adamS_map'], 4)} "
              f"[{timing['adamS_map_s']:.0f}s]", flush=True)
    candidates["zero"] = np.zeros(k)

    # Truth: every candidate held for --truth-days in the same weather samples.
    seg = make_segment_fn(m["step_fn"], m["patterns"], args.truth_days)
    names = sorted(candidates)
    seeds = [TRUTH_SEED0 + 97 * entry["index"] + j
             for j in range(args.members)]
    members = [perturb_member(carry_d, s, 0.001) for s in seeds]
    store = {f: np.zeros((len(names), args.members, len(las))
                         + ocean.shape, np.float32)
             for f in ("sst", "land_temperature", "precipitation")}
    t0 = time.time()
    for i, name in enumerate(names):
        a = jnp.asarray(candidates[name], jnp.float32)
        for j, c in enumerate(members):
            _, fields = seg(c, a, one, q_base, warming)
            for f in store:
                x = np.asarray(fields[f], np.float64)
                for h, lookahead in enumerate(las):
                    store[f][i, j, h] = x[:lookahead].mean(axis=0)
        print(f"  truth {name} [{time.time() - t0:.0f}s]", flush=True)
    timing["truth_s"] = time.time() - t0
    targets = {f"target_{f}": np.stack([
        refs[f"normal_{f}"][day:day + la].mean(axis=0) for la in las])
        for f in store}
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"ic{entry['index']:04d}_pilot"
    np.savez_compressed(str(stem) + ".npz", names=np.array(names),
                        settings=np.stack([candidates[n] for n in names]),
                        weights=weights, ocean=ocean, **targets,
                        **{f"truth_{f}": v for f, v in store.items()})
    timing["total_s"] = time.time() - t_all
    meta = {"stage": "ic", "ic": entry, "config": vars(args),
            "start_day": day, "ramp_wm2_per_day": ramp,
            "lookaheads": list(las), "truth_days": args.truth_days,
            "truth_seeds": seeds, "candidates": {n: candidates[n].tolist()
                                                 for n in names},
            "info": info, "timing": timing, "git": git_provenance(),
            "command": " ".join(sys.argv),
            "finished_utc": datetime.now(timezone.utc).isoformat()}
    with open(str(stem) + ".json", "w") as f:
        json.dump(meta, f, indent=2, default=float)
    print(f"-> {stem}.npz/.json [{timing['total_s']:.0f}s]", flush=True)


# --- Analysis -------------------------------------------------------------------

def true_objectives(z) -> dict:
    """Per candidate: J (map, zonal) and bias for each look-ahead, member mean."""
    w, ocean = z["weights"], z["ocean"]
    out = {}
    for i, name in enumerate(z["names"]):
        res = {}
        for h, tag in enumerate(("S", "L")):
            err = z["truth_sst"][i, :, h].astype(np.float64).mean(axis=0) \
                - z["target_sst"][h]
            res[f"J{tag}_map"] = float(objective(err, w))
            res[f"J{tag}_zonal"] = float(objective(zonal_project(err, ocean),
                                                  w))
            res[f"bias{tag}_K"] = float(np.sum(w * err))
        out[str(name)] = res
    return out


def paired(values_a, values_b):
    """Mean and SE of a - b over states; relative to b's mean."""
    d = np.asarray(values_a) - np.asarray(values_b)
    se = float(d.std(ddof=1) / np.sqrt(d.size)) if d.size > 1 else float("nan")
    return {"diff": float(d.mean()), "se": se,
            "rel": float(d.mean() / np.mean(values_b))}


def stage_analyze(args):
    from jcm.mcb.test_world import BRIGHTENING_CAP

    in_dir = Path(args.input_dir)
    with open(args.offline) as f:
        offline = json.load(f)
    rep = offline["warming_choice"]["representation"]
    rep = "zonal" if rep == "mean" else rep   # a map objective needs a field
    states, metas = [], []
    for path in sorted(in_dir.glob("ic*_pilot.npz")):
        with open(str(path)[:-4] + ".json") as f:
            metas.append(json.load(f))
        states.append(true_objectives(np.load(path)))
    n = len(states)
    if n < 2:
        raise SystemExit("need at least two states")

    def col(name, key):
        return np.array([s[name][key] for s in states])

    d = {"states": [mt["ic"]["index"] for mt in metas], "n": n,
         "representation": rep}
    # R1: optimizer and iterations.
    if all("adamS_map" in s for s in states):
        r1 = paired(col("adamS_map", "JS_map"), col("gnS_map", "JS_map"))
        adam_better = (r1["diff"] < -2 * r1["se"] and r1["rel"] <= -0.05)
        d["R1_adam_vs_gn"] = r1
        d["optimizer"] = "adam" if adam_better else "gauss_newton"
    else:
        d["optimizer"] = "gauss_newton"
    iters = 3
    for kk, name in ((1, "gnS_map_it1"), (2, "gnS_map_it2")):
        p = paired(col(name, "JS_map"), col("gnS_map", "JS_map"))
        d[f"R1_it{kk}_vs_it3"] = p
        if p["rel"] <= 0.02 or p["diff"] <= p["se"]:
            iters = kk
            break
    d["iterations"] = iters
    # R2: copies.
    singles = [nm for nm in states[0] if nm.startswith("gnS_map_copy")]
    rels = [paired(col(nm, "JS_map"), col("gnS_map", "JS_map"))["rel"]
            for nm in singles]
    nts = [mt["info"]["gnS_map"]["noise_to_signal"] for mt in metas]
    d["R2_single_copy_rel"] = rels
    d["R2_median_noise_to_signal"] = float(np.median(nts))
    d["copies"] = 1 if (np.mean(rels) <= 0.05
                        and np.median(nts) < 0.5) else 3
    # R4 (planner objective) and R3 (look-ahead) in the chosen representation.
    if rep == "zonal":
        p = paired(col("gnS_map", "JS_zonal"), col("gnS_zonal", "JS_zonal"))
        d["R4_map_vs_zonal_objective"] = p
        d["objective"] = "map" if (p["diff"] < -2 * p["se"]) else "zonal"
    else:
        d["objective"] = "map"
    obj = d["objective"]
    r3 = paired(col(f"gnL_{obj}", f"JL_{rep}"),
                col(f"gnS_{obj}", f"JL_{rep}"))
    d["R3_long_vs_short_on_JL"] = r3
    d["R3_long_vs_short_on_JS"] = paired(col(f"gnL_{obj}", f"JS_{rep}"),
                                         col(f"gnS_{obj}", f"JS_{rep}"))
    la_s, la_l = metas[0]["lookaheads"]
    d["lookaheads_tested"] = [la_s, la_l]
    d["lookahead_days"] = la_l if (r3["diff"] < -2 * r3["se"]
                                   and r3["rel"] <= -0.05) else la_s
    # R7: penalties.
    pens = [mt["info"]["translated_penalties"] for mt in metas]
    s = float(np.mean([p["s"] for p in pens]))
    zero = col("zero", f"JS_{rep}")
    base = col("gnS_zonal", f"JS_{rep}")
    with_mu = col("gnS_zonal_mu", f"JS_{rep}")
    cost = float(np.mean(with_mu - base) / np.mean(zero - base))
    factor = 1.0
    while cost * factor > 0.10 and factor > 1e-3:
        factor /= 10.0   # first-order: the penalty's cost scales with mu
    d["R7"] = {"s_K_per_albedo": s, "mu_translated": DUBEY_MU * s ** 2,
               "lam_translated": DUBEY_LAM * s ** 2,
               "cost_share_of_reduction": cost, "factor": factor}
    d["mu"] = DUBEY_MU * s ** 2 * factor
    d["lam"] = DUBEY_LAM * s ** 2 * factor
    # Headroom (R5) and the value of choosing where.
    gn = np.array([s_[f"gnS_{obj}"] for s_ in
                   [mt["candidates"] for mt in metas]])
    d["R5_mean_setting_share_of_cap"] = float(gn.mean() / BRIGHTENING_CAP)
    d["R5_max_band_share_of_cap"] = float(gn.max() / BRIGHTENING_CAP)
    d["where_vs_uniform"] = paired(col(f"gnS_{obj}", f"JS_{rep}"),
                                   col("uniformS", f"JS_{rep}"))
    d["reduction_vs_zero"] = paired(col(f"gnS_{obj}", f"JS_{rep}"),
                                    col("zero", f"JS_{rep}"))
    d["timing_s"] = {key: float(np.mean([mt["timing"].get(key, np.nan)
                                         for mt in metas]))
                     for key in metas[0]["timing"]}
    table = {name: {key: float(np.mean(col(name, key)))
                    for key in states[0][name]} for name in states[0]}
    out = {"stage": "analyze", "decisions": d, "mean_true_objectives": table,
           "per_state": states,
           "created_utc": datetime.now(timezone.utc).isoformat()}
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"{'candidate':>16} " + " ".join(f"{k:>11}" for k in
                                             next(iter(table.values()))))
    for name, row in sorted(table.items(), key=lambda kv: kv[1]["JS_map"]):
        print(f"{name:>16} " + " ".join(f"{v:11.5f}" for v in row.values()))
    print(json.dumps(d, indent=2))
    print(f"-> {args.output}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="stage", required=True)
    o = sub.add_parser("offline")
    o.add_argument("--refs-dir", required=True)
    o.add_argument("--exp1-maps", required=True)
    o.add_argument("--output", required=True)
    i = sub.add_parser("ic")
    i.add_argument("--ic-dir", required=True)
    i.add_argument("--ic-position", type=int, required=True)
    i.add_argument("--references", required=True)
    i.add_argument("--output-dir", required=True)
    i.add_argument("--ramp-end-wm2", type=float, default=6.0)
    i.add_argument("--lookaheads", type=int, nargs=2, default=list(LOOKAHEADS),
                   help="Short and long look-ahead (days).")
    i.add_argument("--truth-days", type=int, default=TRUTH_DAYS)
    i.add_argument("--start-day", type=int, default=START_DAY)
    i.add_argument("--members", type=int, default=8)
    i.add_argument("--copies", type=int, default=3)
    i.add_argument("--single-copies", type=int, default=3)
    i.add_argument("--skip-adam", action="store_true")
    a = sub.add_parser("analyze")
    a.add_argument("--input-dir", required=True)
    a.add_argument("--offline", required=True)
    a.add_argument("--output", required=True)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    {"offline": stage_offline, "ic": stage_ic,
     "analyze": stage_analyze}[args.stage](args)


if __name__ == "__main__":
    main()
