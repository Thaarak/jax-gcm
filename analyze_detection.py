#!/usr/bin/env python
"""Detection with one Earth, from Experiment 3a's runs (exploratory).

MCB_PROJECT_REPORT.md Part 27. This is the advisor's question: with one
Earth, no same-weather twin and real weather noise, how soon could an
observer tell that a brightening strategy is cooling the ocean? The
design below was fixed before any detection statistic was computed.
Experiment 3a's skill scores were already known, so this is exploratory,
not registered.

*Signal.* For each 3a arm and evaluation state: the arm's 3-member-mean SST
minus the uncontrolled arm's, which runs with the same member seeds. It is
reduced to 5-day blocks of the ocean zonal-mean profile, averaged over the
24 states for the typical signal.

*Noise for one Earth.* What a single realization does that a forecast of
the no-intervention world cannot predict. There are three cases:
- *perfect-start forecast:* the spread of the references' 5 separate warmed
  members around their mean (the model's own ensemble forecast from the
  exact starting state), plus the 5-member forecast's own error. A best
  case: the members start 0.001 K apart;
- *saturated forecast:* the same noise, fixed at its end-of-episode level
  for every window. This is a forecast that has already lost its skill
  about the weather but still knows the ocean state, the realistic middle;
- *climatology:* an observer with no forecast also faces the spread of the
  normal climate across the 8 independent ocean states.

All are pooled over states and use the same 30-day averaging window.

*Statistics,* at each window end:
- *ocean-mean index:* SNR = |signal| / noise;
- *fingerprint:* the SNR of the optimal projection onto the expected
  latitude pattern, sqrt(s' C^-1 s). Here C is the noise covariance of the
  area-weighted latitude profile, shrunk toward a scaled identity by
  Ledoit-Wolf so that it is invertible. For any covariance this is at
  least the index's SNR (Cauchy-Schwarz), and the gap is what the pattern
  adds. Estimating C from 96 degrees of freedom biases it upward somewhat,
  so it is an upper estimate;
- *detection probability:* a one-sided test at 5% gives
  P = Phi(SNR - 1.645);
- *the reported times:* the days at which P first reaches 50% and 95%.

*Caveats.*
- The members start 0.001 K apart, so early forecast noise is a best case
  (a perfect starting state). A real forecast starts from an analysis with
  errors.
- Noise comes from runs without brightening, assuming brightening does
  not change the weather noise.
- The land-model defect (Part 26) is present in every run.

Stages:
``reduce`` (where the data are; numpy only)
    Writes ``detection_inputs.npz``: zonal profiles of the signals and of
    the reference members.
``analyze`` (laptop)
    Writes ``detection_analysis.json`` and ``fig_detection.png``.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BLOCK_DAYS = 5
EPISODE_DAYS = 182
WINDOW_BLOCKS = 6                        # 30-day observation window
FORECAST_MEMBERS = 5
Z_ONE_SIDED = 1.6448536269514722         # 5% one-sided
SIGNAL_ARMS = ("uniform_cancel", "linear_response", "plan14", "plan60",
               "plan120", "pi", "adaptive", "planner_average",
               "uniform_effort")


def zonal_profiles(field, ocean, used):
    """Return the ocean zonal-mean profile, ``(..., n_used_latitudes)``."""
    counts = ocean.sum(axis=0)
    z = (field * ocean).sum(axis=-2) / np.maximum(counts, 1.0)
    return z[..., used]


def blocks(daily, n_blocks):
    """Return 5-day block means of a daily series ``(days, ...)``."""
    d = np.asarray(daily, np.float64)[:n_blocks * BLOCK_DAYS]
    return d.reshape((n_blocks, BLOCK_DAYS) + d.shape[1:]).mean(axis=1)


def stage_reduce(args):
    runs, refs = Path(args.runs_dir), Path(args.references_dir)
    with np.load(refs / "grid.npz") as g:
        lats = np.asarray(g["latitudes_rad"], np.float64)
        ocean = np.asarray(g["ocean_mask"], np.float64)
    used = ocean.sum(axis=0) > 0
    area = np.cos(lats[used]) * ocean.sum(axis=0)[used]
    area = area / area.sum()
    n_blocks = EPISODE_DAYS // BLOCK_DAYS                       # 36
    states = sorted(int(p.name.split(".")[0][2:]) for p in
                    (runs / "uncontrolled").glob("ic*.fields.npz"))
    signal = np.full((len(SIGNAL_ARMS), len(states), n_blocks, used.sum()),
                     np.nan)
    members = np.full((len(states), FORECAST_MEMBERS, n_blocks, used.sum()),
                      np.nan)
    normal_mean = np.full((len(states), n_blocks, used.sum()), np.nan)
    for si, idx in enumerate(states):
        name = f"ic{idx:04d}"
        with np.load(runs / "uncontrolled" / f"{name}.fields.npz") as f:
            base = blocks(zonal_profiles(np.asarray(f["sst"], np.float64),
                                         ocean, used), n_blocks)
        for ai, arm in enumerate(SIGNAL_ARMS):
            with np.load(runs / arm / f"{name}.fields.npz") as f:
                z = blocks(zonal_profiles(np.asarray(f["sst"], np.float64),
                                          ocean, used), n_blocks)
            signal[ai, si] = z - base
        with np.load(refs / f"{name}_references.npz") as r:
            m = np.asarray(r["warmed_members_sst"][:, :n_blocks], np.float64)
            members[si] = zonal_profiles(m, ocean, used)
            normal_mean[si] = blocks(zonal_profiles(
                np.asarray(r["normal_sst"], np.float64), ocean, used),
                n_blocks)
        print(f"  {name}: done", flush=True)
    np.savez_compressed(args.output, signal=signal, members=members,
                        normal_mean=normal_mean, states=np.asarray(states),
                        arms=np.asarray(SIGNAL_ARMS), area=area,
                        lat_deg=np.degrees(lats[used]))
    print(f"-> {args.output}")


def window_means(x, w=WINDOW_BLOCKS):
    """Return trailing means over ``w`` blocks along axis -2 (block axis)."""
    c = np.cumsum(np.asarray(x, np.float64), axis=-2)
    out = np.full_like(c, np.nan)
    out[..., w - 1, :] = c[..., w - 1, :]
    out[..., w:, :] = c[..., w:, :] - c[..., :-w, :]
    return out / w


def ledoit_wolf(samples):
    """Return the Ledoit-Wolf shrunk covariance of centered ``(n, p)`` samples."""
    x = np.asarray(samples, np.float64)
    n, dim = x.shape
    cov = x.T @ x / n
    mu = np.trace(cov) / dim
    target = mu * np.eye(dim)
    delta = np.sum((cov - target) ** 2) / dim
    beta = sum(np.sum((np.outer(r, r) - cov) ** 2) for r in x) / n ** 2 / dim
    alpha = min(beta / delta, 1.0) if delta > 0 else 1.0
    return alpha * target + (1.0 - alpha) * cov


def noise_covariances(members, area):
    """Return the forecast noise covariance per window end, ``(B, L, L)``.

    Single-member deviations of the 30-day mean from the 5-member mean,
    pooled over states. They are shrunk (Ledoit-Wolf) in area-weighted
    space and inflated by ``(1 + 1/N)`` for the forecast's own error.
    Shrinkage keeps C invertible with 96 degrees of freedom for 46
    latitudes.
    """
    wm = window_means(members)                        # (S, M, B, L)
    dev = wm - wm.mean(axis=1, keepdims=True)
    s_count, m_count, n_blocks, n_lat = wm.shape
    sw = np.sqrt(area)
    out = np.full((n_blocks, n_lat, n_lat), np.nan)
    for b in range(n_blocks):
        x = dev[:, :, b].reshape(-1, n_lat)
        if not np.all(np.isfinite(x)):
            continue
        # Centering per state uses one degree of freedom of each state's M.
        x = x * np.sqrt(m_count / (m_count - 1.0))
        shrunk = ledoit_wolf(x * sw)
        out[b] = shrunk / np.outer(sw, sw)
    return out * (1.0 + 1.0 / FORECAST_MEMBERS)


def climatology_covariance(normal_mean, states):
    """Return the macro-state spread of the normal climate, ``(B, L, L)``."""
    wm = window_means(normal_mean)                    # (S, B, L)
    macros = np.asarray(states) // 100
    keys = sorted(set(macros))
    per_macro = np.stack([wm[macros == k].mean(axis=0) for k in keys])
    dev = per_macro - per_macro.mean(axis=0, keepdims=True)
    return np.einsum("kbi,kbj->bij", dev, dev) / (len(keys) - 1)


def index_snr(signal_window, cov, area):
    """Return |area-mean signal| / its noise, per block."""
    s = signal_window @ area                                    # (B,)
    var = np.einsum("i,bij,j->b", area, cov, area)
    return np.abs(s) / np.sqrt(var)


def fingerprint_snr(signal_window, cov):
    """Return sqrt(s' C^-1 s) per block (the optimal pattern detector)."""
    out = np.full(signal_window.shape[0], np.nan)
    for b in range(signal_window.shape[0]):
        if not (np.all(np.isfinite(signal_window[b]))
                and np.all(np.isfinite(cov[b]))):
            continue
        out[b] = float(np.sqrt(signal_window[b]
                               @ np.linalg.solve(cov[b], signal_window[b])))
    return out


def detection_probability(snr):
    """Return Phi(SNR - z) for a one-sided 5% test."""
    from math import erf
    return np.array([0.5 * (1.0 + erf((x - Z_ONE_SIDED) / np.sqrt(2.0)))
                     if np.isfinite(x) else np.nan for x in snr])


def first_day(prob, level):
    """Return the window-end day at which ``prob`` first reaches ``level``."""
    hits = np.flatnonzero(np.nan_to_num(prob) >= level)
    return None if hits.size == 0 else int((hits[0] + 1) * BLOCK_DAYS)


def stage_analyze(args):
    z = np.load(args.inputs)
    signal, members = z["signal"], z["members"]
    area, arms, states = z["area"], [str(a) for a in z["arms"]], z["states"]
    fc = noise_covariances(members, area)
    last = np.flatnonzero(np.all(np.isfinite(fc), axis=(1, 2)))[-1]
    sat = np.repeat(fc[last][None], fc.shape[0], axis=0)
    cl = sat + climatology_covariance(z["normal_mean"], states)
    mean_signal = window_means(np.nanmean(signal, axis=1))      # (A, B, L)
    days = (np.arange(signal.shape[2]) + 1) * BLOCK_DAYS
    cases = (("forecast", fc), ("saturated", sat), ("climatology", cl))
    noise_index = {f"{name}_K": np.sqrt(np.einsum("i,bij,j->b", area, cov,
                                                    area))
                   for name, cov in cases}
    result = {"days": days.tolist(), "window_days": WINDOW_BLOCKS * BLOCK_DAYS,
              "n_states": int(signal.shape[1]),
              "noise_index_K": {k: v.tolist() for k, v in noise_index.items()},
              "arms": {}}
    for ai, arm in enumerate(arms):
        s = mean_signal[ai]
        row = {"index_signal_K": (s @ area).tolist()}
        for sense, cov in cases:
            snr = {"index": index_snr(s, cov, area),
                   "fingerprint": fingerprint_snr(s, cov)}
            for name, values in snr.items():
                prob = detection_probability(values)
                row[f"{sense}_{name}"] = {
                    "snr": np.round(values, 3).tolist(),
                    "day_p50": first_day(prob, 0.5),
                    "day_p95": first_day(prob, 0.95)}
        result["arms"][arm] = row
    result["created_utc"] = datetime.now(timezone.utc).isoformat()
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(f"{'arm':>16} {'signal@180':>10} | day of 95% detection (index / "
          f"fingerprint): perfect-start | saturated | climatology")
    for arm in arms:
        r = result["arms"][arm]
        cells = " | ".join(f"{r[f'{c}_index']['day_p95']} / "
                           f"{r[f'{c}_fingerprint']['day_p95']}"
                           for c, _ in cases)
        print(f"{arm:>16} {r['index_signal_K'][-1]:+10.3f} | {cells}")
    if args.figure:
        plot(result, args.figure)
    print(f"-> {args.output}")


def plot(result, path):
    """Draw the noise curves and the detection times."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    days = np.asarray(result["days"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    for key, lab in (("forecast_K", "noise: perfect-start forecast"),
                     ("saturated_K", "noise: saturated forecast"),
                     ("climatology_K", "noise: climatology")):
        ax.plot(days, 1000 * np.asarray(result["noise_index_K"][key],
                                        dtype=float), label=lab)
    for arm in ("uniform_cancel", "plan14", "pi"):
        ax.plot(days, -1000 * np.asarray(result["arms"][arm]
                                         ["index_signal_K"]),
                "--", label=f"cooling signal: {arm}")
    ax.set_xlabel("window end (day)")
    ax.set_ylabel("mK (30-day mean, ocean average)")
    ax.set_yscale("log")
    ax.set_title("a   Signal and one-Earth noise", loc="left")
    ax.legend(fontsize=7)
    ax = axes[1]
    arms = list(result["arms"])
    y = np.arange(len(arms))
    for off, (key, lab) in enumerate((("forecast_index",
                                       "perfect-start forecast, index"),
                                      ("saturated_index",
                                       "saturated forecast, index"),
                                      ("saturated_fingerprint",
                                       "saturated forecast, fingerprint"))):
        vals = [result["arms"][a][key]["day_p95"] or np.nan for a in arms]
        ax.barh(y + 0.25 * off - 0.25, vals, height=0.25, label=lab)
    ax.set_yticks(y)
    ax.set_yticklabels(arms, fontsize=8)
    ax.set_xlabel("day by which detection is 95% likely")
    ax.set_title("b   Time to detect each strategy", loc="left")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    print(f"wrote {path}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="stage", required=True)
    r = sub.add_parser("reduce")
    r.add_argument("--runs-dir", required=True)
    r.add_argument("--references-dir", required=True)
    r.add_argument("--output", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("--inputs", required=True)
    a.add_argument("--output", required=True)
    a.add_argument("--figure", default=None)
    args = p.parse_args(argv)
    {"reduce": stage_reduce, "analyze": stage_analyze}[args.stage](args)


if __name__ == "__main__":
    main()
