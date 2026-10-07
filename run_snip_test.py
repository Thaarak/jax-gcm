#!/usr/bin/env python
"""The snip-window extension of Experiment 1 (Amendment 9 revision 1.1, Part A).

Reverse-mode Jacobians with the atmosphere snipped every 21 and 30 days, on
exactly Experiment 1's configuration:
- the 8 ``exp1`` starting states, member 0 (the state itself);
- the operating point a0 = 0.03 in each of the five bands;
- horizons 15, 30, 60 and 120 days, with tail means over the last 10 days;
- objectives T0, T1, T2 and LAND;
- windows aligned to the episode start.

The truth is Experiment 1's stored finite differences; nothing here re-runs
it. ``analyze_snip_test.py`` merges these Jacobians with Experiment 1's and
applies Experiment 1's frozen analysis.

Writes ``<output>.npz`` with ``jacobians`` laid out as Experiment 1's,
``(ic, member, window, horizon, objective, band)``, and ``<output>.json``.

Example (GX10):
    python run_snip_test.py --ic-dir mcb_experiments_gpu/ics_macro/exp1 \
        --output mcb_experiments_gpu/snip_test/snip_test
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REGISTERED_WINDOWS = (21, 30)
# Experiment 1's settings (Amendment 9; exp1_gradient_fidelity.json).
HORIZONS = (15, 30, 60, 120)
TAIL_DAYS = 10
A0 = 0.03


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ic-dir", required=True,
                   help="The exp1 role's IC directory (8 states).")
    p.add_argument("--windows", type=int, nargs="+",
                   default=list(REGISTERED_WINDOWS))
    p.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    p.add_argument("--max-ics", type=int, default=None,
                   help="Only for smoke tests.")
    p.add_argument("--output", required=True, help="Output prefix.")
    return p.parse_args(argv)


def validate_args(args):
    """Fail fast on settings the run cannot use."""
    if any(w < 1 for w in args.windows):
        raise SystemExit("--windows must be >= 1 (full BPTT is Experiment "
                         "1's)")
    if any(h < TAIL_DAYS for h in args.horizons):
        raise SystemExit(f"every horizon must be >= the {TAIL_DAYS}-day tail")


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    import jax
    import jax.numpy as jnp
    import jax_datetime as jdt

    from jcm.mcb.band_basis import (
        OBJECTIVE_NAMES,
        gaussian_band_patterns,
        objective_weights,
        stack_objective_weights,
    )
    from jcm.mcb.coupled_controller import create_coupled_step_fn
    from jcm.mcb.coupled_train import ocean_mask_from_coupler
    from jcm.mcb.gradient_fidelity import make_jacobian_fn
    from run_coupled_training import coupler_workflow, setup_coupled_model
    from run_gradient_fidelity import (
        git_provenance,
        land_mask_from_coupler,
        load_manifest_ics,
    )
    from run_stage5_training import START_DATE

    t_start = time.time()
    print(f"JAX devices: {jax.devices()}")
    coupler, coords, _, _ = setup_coupled_model(
        jdt.to_datetime(START_DATE), jdt.to_timedelta(1, "day"),
        realistic_terrain=True)
    template = coupler.initialize()
    step_fn = create_coupled_step_fn(coupler, coupler_workflow(coupler),
                                     jitted=True)
    manifest, ics = load_manifest_ics(args.ic_dir, "all", args.max_ics,
                                      template)
    if manifest.get("macro_role") != "exp1":
        raise SystemExit("the snip test reuses Experiment 1's truth: use the "
                         "exp1 role")
    lats = coords.horizontal.latitudes
    ocean = ocean_mask_from_coupler(coupler)
    land = land_mask_from_coupler(coupler, coords.horizontal.nodal_shape)
    patterns = gaussian_band_patterns(lats, ocean)
    weight_stack = stack_objective_weights(objective_weights(lats, ocean,
                                                             land))
    k = int(patterns.shape[0])
    a0 = jnp.full((k,), A0, dtype=jnp.float32)
    no_decay = jnp.asarray(1.0, dtype=jnp.float32)
    horizons = sorted(args.horizons)
    jac_fns = {h: make_jacobian_fn(step_fn, patterns, weight_stack, h,
                                   TAIL_DAYS) for h in horizons}
    jacobians = np.full((len(ics), 1, len(args.windows), len(horizons),
                         len(OBJECTIVE_NAMES), k), np.nan)
    timing = {}
    for i, (entry, carry) in enumerate(ics):
        t0_ic = time.time()
        t0 = carry["ocn"]["state"].sim_time          # episode alignment
        for hi, h in enumerate(horizons):
            for wi, w in enumerate(args.windows):
                jac = jac_fns[h](a0, carry, jnp.asarray(w), t0, no_decay)
                jacobians[i, 0, wi, hi] = np.asarray(jac)
        timing[str(entry["index"])] = round(time.time() - t0_ic, 1)
        print(f"  IC {entry['index']:>4}: windows {args.windows} x horizons "
              f"{horizons} [{timing[str(entry['index'])]:.0f}s]", flush=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out) + ".npz", jacobians=jacobians,
                        ic_index=np.array([e["index"] for e, _ in ics]))
    meta = {"experiment": "snip_test",
            "preregistration": "PREREGISTRATION.md Amendment 9 revision 1.1, "
                               "Part A",
            "windows": list(args.windows), "horizons": horizons,
            "objective_names": list(OBJECTIVE_NAMES), "tail_days": TAIL_DAYS,
            "a0": A0, "grad_members": 1, "align": "episode",
            "ic_entries": [e for e, _ in ics], "timing_s": timing,
            "git": git_provenance(), "command": " ".join(sys.argv),
            "jax_devices": [str(d) for d in jax.devices()],
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "total_s": round(time.time() - t_start, 1)}
    with open(str(out) + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"-> {out}.npz / .json")


if __name__ == "__main__":
    main()
