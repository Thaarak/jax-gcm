#!/usr/bin/env python
r"""Step 17: what one re-plan costs on the GPU (MCB_PROJECT_REPORT.md Part 18).

A re-plan is ``iterations`` repeats of one expensive call, plus cheap host
arithmetic. For Adam the call is one gradient per copy; for Gauss-Newton it
is one whole-map Jacobian per copy (``Planner.evaluate_copies``).

For each candidate setting this script builds the real planner on one
training IC and times that call after compilation, twice: once with the
copies run one after another, once with them side by side (one vmapped
call). Each (optimizer, look-ahead) runs in its own process, so it records
its own peak device memory, and an out-of-memory failure cannot sink the
sweep. The script also checks that side-by-side results equal sequential
ones.

Reported per setting:
- the compile time and the seconds per call;
- the seconds per re-plan at the registered iteration counts (Adam 15,
  Gauss-Newton 3);
- the hours per planned 182-day episode (13 re-plans of 14 days, plus the run
  itself);
- the GPU-hours per 100 planned episodes.

Example (GX10, alongside vLLM):
    python run_planner_cost.py \
        --ic-dir mcb_experiments_gpu/ics_macro/exp3_train --references \
        mcb_experiments_gpu/test_world_refs/exp3_train/ic0002_references.npz \
        --full-bptt-check --output mcb_experiments_gpu/planner_cost.json
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from jcm.mcb.gradient_truncation import NO_TRUNCATION_DAYS
from jcm.mcb.planner import OPTIMIZERS, W_STAR_DAYS

EPISODE_DAYS = 182            # 13 segments of 14 days: about six months
SEGMENT_DAYS = 14
# Measured on the GX10 (test-world references, 2026-10-02): s per model day.
FORWARD_S_PER_DAY = 0.068


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ic-dir", required=True)
    p.add_argument("--ic-position", type=int, default=0)
    p.add_argument("--references", required=True,
                   help="That IC's references (the normal-climate target).")
    p.add_argument("--optimizers", nargs="+", default=list(OPTIMIZERS),
                   choices=OPTIMIZERS)
    p.add_argument("--lookaheads", type=int, nargs="+", default=[14, 60, 120])
    p.add_argument("--window-days", type=int, default=W_STAR_DAYS,
                   help="Atmosphere snip in days (0 = full BPTT).")
    p.add_argument("--copies", type=int, default=3)
    p.add_argument("--repeats", type=int, default=2,
                   help="Timed calls after the compiling one.")
    p.add_argument("--adam-iterations", type=int, default=15)
    p.add_argument("--gn-iterations", type=int, default=3)
    p.add_argument("--full-bptt-check", action="store_true",
                   help="Also time Adam without the snip (one copy mode).")
    p.add_argument("--full-bptt-lookahead", type=int, default=60)
    p.add_argument("--warming-step-wm2", type=float, default=4.0)
    p.add_argument("--output", required=True, help="JSON results path.")
    p.add_argument("--single", default=None, help=argparse.SUPPRESS)
    p.add_argument("--sequential-only", action="store_true",
                   help=argparse.SUPPRESS)
    return p.parse_args(argv)


def validate_args(args):
    """Fail fast on settings the sweep cannot use."""
    if min(args.lookaheads) < 1 or args.copies < 1 or args.repeats < 1:
        raise SystemExit("--lookaheads, --copies and --repeats must be >= 1")
    if args.adam_iterations < 1 or args.gn_iterations < 1:
        raise SystemExit("iteration counts must be >= 1")
    if args.window_days < 0 or args.full_bptt_lookahead < 1:
        raise SystemExit("--window-days >= 0 and --full-bptt-lookahead >= 1")


def sweep_configs(args):
    """Return the ``(optimizer, look-ahead, window, sequential_only)`` grid."""
    grid = [(opt, la, args.window_days, False) for opt in args.optimizers
            for la in sorted(args.lookaheads)]
    if args.full_bptt_check:
        grid.append(("adam", args.full_bptt_lookahead, 0, True))
    return grid


def projections(call_s: float, iterations: int) -> dict:
    """Return the re-plan, episode and per-100-episode costs from one call's time."""
    replan_s = iterations * call_s
    replans = EPISODE_DAYS // SEGMENT_DAYS
    episode_h = (replans * replan_s
                 + EPISODE_DAYS * FORWARD_S_PER_DAY) / 3600.0
    return {"iterations": iterations, "replan_s": replan_s,
            "episode_h": episode_h, "gpu_h_per_100_episodes": 100 * episode_h}


def relative_difference(a, b) -> float:
    """Return the largest absolute difference over the largest entry of ``b``."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    scale = float(np.max(np.abs(b)))
    return float(np.max(np.abs(a - b))) / scale if scale > 0 else 0.0


def peak_memory_gb():
    """Peak device memory so far, in GB, or None where JAX cannot tell."""
    import jax
    try:
        stats = jax.devices()[0].memory_stats()
    except Exception:          # backends without statistics
        return None
    return None if not stats else stats.get("peak_bytes_in_use", 0) / 1e9


def run_single(args):
    """Time one (optimizer, look-ahead, window) in this process; print JSON."""
    import jax  # noqa: F401  (initializes the backend before timing)

    from jcm.mcb.planner import Planner, PlannerConfig, to_logits
    from jcm.mcb.test_world import domain_weights, make_warming
    from run_gradient_fidelity import load_manifest_ics
    from run_test_world import build_model

    opt, lookahead, window = args.single.split(":")
    lookahead, window = int(lookahead), int(window)
    m = build_model()
    _, ics = load_manifest_ics(args.ic_dir, "all", None, m["template"])
    entry, carry = ics[args.ic_position]
    refs = np.load(args.references, allow_pickle=False)
    ocean_w = domain_weights(m["lats"], m["ocean"], m["land"])["ocean"]
    warming = make_warming(carry, m["ocean"], args.warming_step_wm2)
    out = {"optimizer": opt, "lookahead_days": lookahead,
           "window_days": window, "copies": args.copies,
           "ic_index": entry["index"], "modes": {}}
    outputs = {}
    for batched in ((False,) if args.sequential_only else (False, True)):
        cfg = PlannerConfig(
            lookahead_days=lookahead, copies=args.copies, optimizer=opt,
            window_days=NO_TRUNCATION_DAYS if window == 0 else window,
            batch_copies=batched)
        planner = Planner(m["step_fn"], m["patterns"], ocean_w,
                          refs["normal_sst"], carry["ocn"]["forcing"].q_flux,
                          warming, 1.0, cfg)
        copies = planner.prepare_copies(carry, 0)
        start = np.full(planner.k, cfg.first_guess * cfg.cap)
        x = to_logits(start, cfg.cap, cfg.edge_margin) if opt == "adam" \
            else start
        extra = ((planner.target_mean(0), np.zeros(planner.k))
                 if opt == "adam" else ())
        t0 = time.time()
        planner.evaluate_copies(x, copies, *extra)
        first = time.time() - t0
        steady = []
        for _ in range(args.repeats):
            t0 = time.time()
            last = planner.evaluate_copies(x, copies, *extra)
            steady.append(time.time() - t0)
        call = float(np.mean(steady))
        mode = "side_by_side" if batched else "sequential"
        outputs[mode] = last
        out["modes"][mode] = {"first_call_s": first, "steady_calls_s": steady,
                              "call_s": call,
                              "compile_s": max(first - call, 0.0),
                              "peak_memory_gb": peak_memory_gb()}
    if len(outputs) == 2:
        out["side_by_side_vs_sequential_rel_diff"] = [
            relative_difference(b, s) for b, s in
            zip(outputs["side_by_side"], outputs["sequential"])]
    print(json.dumps(out))


def child_command(args, opt, lookahead, window, sequential_only):
    """Rebuild this script's command line for one configuration."""
    cmd = [sys.executable, str(Path(__file__).resolve()),
           "--ic-dir", args.ic_dir, "--ic-position", str(args.ic_position),
           "--references", args.references, "--copies", str(args.copies),
           "--repeats", str(args.repeats),
           "--warming-step-wm2", str(args.warming_step_wm2),
           "--output", args.output, "--single",
           f"{opt}:{lookahead}:{window}"]
    return cmd + (["--sequential-only"] if sequential_only else [])


def summarize(result: dict, args) -> list:
    """Return one projection row per copy mode of a finished configuration."""
    iters = (args.adam_iterations if result["optimizer"] == "adam"
             else args.gn_iterations)
    rows = []
    for mode, timing in result["modes"].items():
        rows.append({"optimizer": result["optimizer"],
                     "lookahead_days": result["lookahead_days"],
                     "window_days": result["window_days"], "mode": mode,
                     "call_s": timing["call_s"],
                     "compile_s": timing["compile_s"],
                     "peak_memory_gb": timing["peak_memory_gb"],
                     **projections(timing["call_s"], iters)})
    return rows


def main(argv=None):
    args = parse_args(argv)
    if args.single:
        run_single(args)
        return
    validate_args(args)
    meta = {"step": "Part 18 step 17: planning cost", "config": vars(args),
            "episode_days": EPISODE_DAYS, "segment_days": SEGMENT_DAYS,
            "forward_s_per_day": FORWARD_S_PER_DAY,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "results": [], "failures": [], "rows": []}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    for opt, lookahead, window, seq_only in sweep_configs(args):
        t0 = time.time()
        proc = subprocess.run(child_command(args, opt, lookahead, window,
                                            seq_only),
                              capture_output=True, text=True)
        label = f"{opt} {lookahead} d window {window or 'full'}"
        lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
        if proc.returncode != 0 or not lines:
            meta["failures"].append({"config": label,
                                     "returncode": proc.returncode,
                                     "stderr_tail": proc.stderr[-2000:]})
            print(f"FAILED {label} (exit {proc.returncode})", flush=True)
        else:
            result = json.loads(lines[-1])
            meta["results"].append(result)
            meta["rows"] += summarize(result, args)
            for row in summarize(result, args):
                print(f"{label:>28} {row['mode']:>12}: call "
                      f"{row['call_s']:7.1f} s | re-plan "
                      f"{row['replan_s'] / 60:6.1f} min | episode "
                      f"{row['episode_h']:5.2f} h | memory "
                      f"{row['peak_memory_gb'] or float('nan'):5.1f} GB "
                      f"[{time.time() - t0:.0f}s]", flush=True)
        with open(out, "w") as f:
            json.dump(meta, f, indent=2)
    meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
    with open(out, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
