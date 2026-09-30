#!/usr/bin/env python
"""Experiment 1: do atmosphere-truncated gradients match ensemble truth?

Pre-registered as PREREGISTRATION.md Amendment 9. For each initial condition
(IC) and micro-ensemble member this driver records:

* TRUTH (forward only). Rollouts at ``a0``, ``a0 +/- delta*e_k`` for each of
  the K ocean bands, and at ``a = 0`` for the operating-point response. Every
  rollout stores the daily series of the objectives (T0, T1, T2, LAND). The
  analysis (``analyze_gradient_fidelity.py``) turns central differences of
  tail means, averaged over members and ICs, into the ensemble-mean
  sensitivity. Paired baselines cancel in a central difference, so none are
  run.
* ESTIMATORS. Reverse-mode Jacobians of the tail-mean objectives at every
  horizon, for every truncation window W (0 = no truncation = ordinary
  backpropagation through time).

Window length and episode start are traced, so there is one compile for the
forward series plus one per horizon. Results go to ``<output>.npz`` (arrays)
and ``<output>.json`` (configuration, index maps, provenance) — no pickle.
Arrays are written after every IC, so a crash loses at most one IC.

Example (GPU):
    python run_gradient_fidelity.py \
        --ic-dir mcb_experiments_gpu/ics_macro/exp1 \
        --fd-members 4 --grad-members 1 --a0 0.03 --delta 0.03 \
        --horizons 15 30 60 120 --tail-days 10 --windows 1 7 14 0 \
        --output mcb_experiments_gpu/exp1_gradient_fidelity

CPU smoke (tiny):
    python run_gradient_fidelity.py --ic-dir <smoke ics> --max-ics 1 \
        --fd-members 1 --grad-members 1 --horizons 2 3 --tail-days 1 \
        --windows 1 0 --band-centers 20 -20 --output /tmp/exp1_smoke
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np

from jcm.mcb import load_carry
from jcm.mcb.band_basis import (
    DEFAULT_BAND_CENTERS_DEG,
    DEFAULT_BAND_WIDTH_DEG,
    OBJECTIVE_NAMES,
    gaussian_band_patterns,
    objective_weights,
    stack_objective_weights,
)
from jcm.mcb.coupled_controller import create_coupled_step_fn
from jcm.mcb.coupled_train import ocean_mask_from_coupler
from jcm.mcb.gradient_fidelity import make_jacobian_fn, make_series_fn
from jcm.mcb.gradient_truncation import NO_TRUNCATION_DAYS
from run_coupled_training import coupler_workflow, setup_coupled_model
from run_generate_ics_independent import perturb_sst
from run_stage5_training import START_DATE


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ic-dir", required=True,
                   help="Directory with manifest.json + carry files.")
    p.add_argument("--split", default="all",
                   choices=["all", "train", "heldout"])
    p.add_argument("--max-ics", type=int, default=None)
    p.add_argument("--fd-members", type=int, default=4,
                   help="Micro-ensemble members per IC for the truth.")
    p.add_argument("--grad-members", type=int, default=1,
                   help="Members per IC for the gradient estimators (the "
                        "first N of the truth members).")
    p.add_argument("--member-perturb-amp", type=float, default=0.001,
                   help="SST noise (K) seeding members >= 1.")
    p.add_argument("--member-seed0", type=int, default=91000)
    p.add_argument("--horizons", type=int, nargs="+",
                   default=[15, 30, 60, 120])
    p.add_argument("--tail-days", type=int, default=10)
    p.add_argument("--windows", type=int, nargs="+", default=[1, 7, 14, 0],
                   help="Truncation windows in days; 0 = none (full BPTT).")
    p.add_argument("--align", choices=["episode", "absolute"],
                   default="episode",
                   help="Align truncation windows to the episode start or "
                        "to absolute model days.")
    p.add_argument("--a0", type=float, default=0.03,
                   help="Operating-point amplitude for every band.")
    p.add_argument("--delta", type=float, default=0.03,
                   help="Central-difference half step per band (the minus "
                        "run switches that band off; a larger step buys "
                        "signal against ~20 mK of per-run chaos noise).")
    p.add_argument("--band-centers", type=float, nargs="+",
                   default=list(DEFAULT_BAND_CENTERS_DEG))
    p.add_argument("--band-width", type=float, default=DEFAULT_BAND_WIDTH_DEG)
    p.add_argument("--no-zero-run", dest="zero_run", action="store_false",
                   help="Skip the a = 0 rollout (operating-point response).")
    p.add_argument("--output", required=True,
                   help="Output prefix; writes <output>.npz and .json.")
    return p.parse_args(argv)


def validate_args(args):
    """Fail fast on settings the analysis cannot use."""
    if args.grad_members > args.fd_members:
        raise SystemExit("--grad-members must be <= --fd-members")
    if min(args.horizons) < args.tail_days or args.tail_days < 1:
        raise SystemExit("every horizon must be >= --tail-days >= 1")
    if args.a0 - args.delta < 0.0:
        raise SystemExit("a0 - delta must stay >= 0 (no darkening)")
    if any(w < 0 for w in args.windows):
        raise SystemExit("--windows must be >= 0 (0 = full BPTT)")


def run_labels(k_bands, zero_run):
    """Order of the forward runs stored per member: zero?, center, +/-k."""
    labels = (["zero"] if zero_run else []) + ["center"]
    for k in range(k_bands):
        labels += [f"plus_{k}", f"minus_{k}"]
    return labels


def amplitude_for(label, a0_vec, delta):
    """Amplitude vector for a run label from ``run_labels``."""
    if label == "zero":
        return jnp.zeros_like(a0_vec)
    if label == "center":
        return a0_vec
    sign, k = label.split("_")
    step = delta if sign == "plus" else -delta
    return a0_vec.at[int(k)].add(step)


def land_mask_from_coupler(coupler, shape):
    """Binary land mask of the slab land model (1 = land)."""
    lnd = coupler.components.get("lnd")
    if lnd is None:
        raise SystemExit("Experiment 1 needs the slab land model "
                         "(realistic terrain) for the LAND objective")
    raw = getattr(lnd, "raw_component", lnd)
    mask = jnp.asarray(raw.bmask_l, dtype=jnp.float32)
    if mask.shape != tuple(shape):
        raise SystemExit(f"land mask shape {mask.shape} != grid {shape}")
    return mask


def git_provenance():
    """Commit hash and dirty flag of the working tree (best effort)."""
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"],
                             capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"],
                                    capture_output=True, text=True,
                                    check=True).stdout.strip())
        return {"commit": sha, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def load_manifest_ics(ic_dir, split, max_ics, template):
    """Load IC carries (no baselines needed: they cancel in the truth)."""
    ic_dir = Path(ic_dir)
    with open(ic_dir / "manifest.json") as f:
        manifest = json.load(f)
    if manifest.get("start_date") != START_DATE:
        raise SystemExit(f"manifest start_date {manifest.get('start_date')} "
                         f"!= {START_DATE}")
    if not manifest.get("realistic_terrain", False):
        raise SystemExit("manifest was not generated on realistic terrain")
    entries = [e for e in manifest["ics"]
               if split == "all" or e["split"] == split]
    if max_ics is not None:
        entries = entries[:max_ics]
    if not entries:
        raise SystemExit(f"no ICs in split '{split}' of {ic_dir}")
    return manifest, [(e, load_carry(str(ic_dir / e["carry_file"]),
                                     template)) for e in entries]


def save_outputs(prefix, arrays, meta):
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(prefix) + ".npz", **arrays)
    with open(str(prefix) + ".json", "w") as f:
        json.dump(meta, f, indent=2)


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    t_start = time.time()

    print("=" * 72)
    print("EXPERIMENT 1: GRADIENT FIDELITY (PREREGISTRATION Amendment 9)")
    print("=" * 72)
    print(f"JAX devices: {jax.devices()}")

    coupler, coords, terrain, _ = setup_coupled_model(
        jdt.to_datetime(START_DATE), jdt.to_timedelta(1, "day"),
        realistic_terrain=True)
    workflow = coupler_workflow(coupler)
    # initialize() BEFORE building the step function: the slab land model
    # loads its climatology (stl_clim) inside initialize().
    template = coupler.initialize()
    step_fn = create_coupled_step_fn(coupler, workflow, jitted=True)
    manifest, ics = load_manifest_ics(args.ic_dir, args.split, args.max_ics,
                                      template)

    shape = coords.horizontal.nodal_shape
    lats = coords.horizontal.latitudes
    ocean_mask = ocean_mask_from_coupler(coupler)
    land_mask = land_mask_from_coupler(coupler, shape)
    patterns = gaussian_band_patterns(lats, ocean_mask,
                                      centers_deg=args.band_centers,
                                      width_deg=args.band_width)
    weight_stack = stack_objective_weights(
        objective_weights(lats, ocean_mask, land_mask))

    k_bands = len(args.band_centers)
    a0_vec = jnp.full((k_bands,), args.a0, dtype=jnp.float32)
    labels = run_labels(k_bands, args.zero_run)
    horizons = sorted(args.horizons)
    max_h = horizons[-1]
    windows = list(args.windows)
    window_values = [NO_TRUNCATION_DAYS if w == 0 else w for w in windows]

    series_fn = jax.jit(make_series_fn(step_fn, patterns, weight_stack,
                                       max_h))
    jac_fns = {h: make_jacobian_fn(step_fn, patterns, weight_stack, h,
                                   args.tail_days) for h in horizons}

    n_ic, n_obj = len(ics), len(OBJECTIVE_NAMES)
    arrays = {
        "fd_series": np.full((n_ic, args.fd_members, len(labels), max_h,
                              n_obj), np.nan, dtype=np.float64),
        "jacobians": np.full((n_ic, args.grad_members, len(windows),
                              len(horizons), n_obj, k_bands), np.nan,
                             dtype=np.float64),
        "ic_index": np.array([e["index"] for e, _ in ics]),
        "band_patterns": np.asarray(patterns),
        "objective_weights": np.asarray(weight_stack),
    }
    meta = {
        "experiment": "exp1_gradient_fidelity",
        "preregistration": "PREREGISTRATION.md Amendment 9",
        "config": vars(args),
        "objective_names": list(OBJECTIVE_NAMES),
        "run_labels": labels,
        "horizons": horizons,
        "windows": windows,
        "window_note": "0 = no truncation (full BPTT)",
        "ic_entries": [e for e, _ in ics],
        "ic_manifest": str(Path(args.ic_dir) / "manifest.json"),
        "manifest_extra": {k: manifest.get(k) for k in
                           ("macro_role", "seed0", "decorr_days",
                            "perturb_amp")},
        "member_seeds": {},
        "git": git_provenance(),
        "command": " ".join(sys.argv),
        "jax_devices": [str(d) for d in jax.devices()],
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "timing_s": {},
    }

    for i, (entry, carry) in enumerate(ics):
        t_ic = time.time()
        for m in range(args.fd_members):
            if m == 0:
                member = carry
                seed = None
            else:
                seed = args.member_seed0 + 97 * entry["index"] + m
                member = perturb_sst(carry, seed, args.member_perturb_amp)
            meta["member_seeds"][f"{i}:{m}"] = seed
            t0 = (member["ocn"]["state"].sim_time if args.align == "episode"
                  else jnp.asarray(0.0))

            t_fd = time.time()
            for r, label in enumerate(labels):
                series = series_fn(amplitude_for(label, a0_vec, args.delta),
                                   member)
                arrays["fd_series"][i, m, r] = np.asarray(series)
            fd_s = time.time() - t_fd

            grad_s = 0.0
            if m < args.grad_members:
                t_g = time.time()
                for h_idx, h in enumerate(horizons):
                    for w_idx, w in enumerate(window_values):
                        jac = jac_fns[h](a0_vec, member, jnp.asarray(w), t0)
                        arrays["jacobians"][i, m, w_idx, h_idx] = (
                            np.asarray(jac))
                grad_s = time.time() - t_g
            print(f"  IC {entry['index']:>3} member {m}: {len(labels)} "
                  f"forward runs {fd_s:.0f}s | gradients {grad_s:.0f}s",
                  flush=True)
            meta["timing_s"][f"{i}:{m}"] = {"forward": fd_s,
                                            "gradients": grad_s}
        meta["finished_ics"] = i + 1
        save_outputs(args.output, arrays, meta)
        print(f"IC {i + 1}/{n_ic} done in {time.time() - t_ic:.0f}s "
              f"-> {args.output}.npz", flush=True)

    meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
    meta["total_s"] = time.time() - t_start
    save_outputs(args.output, arrays, meta)
    print(f"DONE in {meta['total_s'] / 3600:.2f} h")


if __name__ == "__main__":
    main()
