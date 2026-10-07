#!/usr/bin/env python
"""Registered analysis of the snip-window extension (Amendment 9 revision 1.1, Part A).

Frozen with revision 1.1, before any snip-test Jacobian existed.

It merges Experiment 1's arrays with the snip test's Jacobians:
- Experiment 1 supplies the truth (central differences, 8 states x 4
  members), the Jacobians for W = 1, 7, 14 and full backpropagation, and
  the exploratory damped estimators.
- The snip test supplies the Jacobians for W = 21 and 30.

It then runs Experiment 1's frozen analysis (``analyze_gradient_fidelity.
analyze``) unchanged: the same metrics, verdict thresholds and hierarchical
bootstrap (2000 replicates, seed 2026). The bootstrap's draws depend only on
the numbers of states and members, so the old windows' intervals are
reproduced exactly and the new windows are resampled with the same draws.

Part A's registered summary, applied to the merged result:

* ``W_long``: among W = 14, 21 and 30, the windows that are ``useful`` for T0
  at 120 days, choose the one whose projection ratio is closest to 1. Ties
  within 0.05 go to the shorter window.
* *Outcome:*
  - **S-A**: ``W_long`` is 21 or 30. A longer snip removes part of the
    undercount at 120 days and stays useful.
  - **S-B**: ``W_long`` is 14. Fourteen days remains the best window at 120
    days.
  - **S-U**: no window is useful there. That would contradict Experiment 1's
    verdict for W = 14, so it is reported as an error, not an outcome.
* *Predictions* (reported as confirmed or refuted; no gate):
  - **P1**: the T0 ratio at 120 days rises with the window (14 < 21 < 30).
  - **P2**: so does the median single-realization angle (the price of a
    longer memory).
* Experiment 1's W* stays 14 for everything frozen before or with revision
  1.1. The merged analysis recomputes a W* over all truncated windows; it is
  reported for the record and changes nothing.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from analyze_gradient_fidelity import analyze

EXP1_PREFIX = "mcb_experiments_gpu/exp1/exp1_gradient_fidelity"
KEY = "T0@120"
COMPARED_WINDOWS = (14, 21, 30)
RATIO_TIE = 0.05
REPORTED_OBJECTIVES = ("T0", "T1", "T2", "LAND")
REPORTED_HORIZONS = (60, 120)


def load(prefix):
    """Return ``(arrays, meta)`` of an ``.npz`` / ``.json`` pair."""
    with np.load(str(prefix) + ".npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    with open(str(prefix) + ".json") as f:
        meta = json.load(f)
    return arrays, meta


def merge(exp1_arrays, exp1_meta, snip_arrays, snip_meta):
    """Return Experiment 1's arrays and meta with the snip windows inserted.

    Windows are ordered: Experiment 1's truncated windows, the snip test's,
    then full backpropagation (0), as Experiment 1 orders them.
    """
    if not np.array_equal(exp1_arrays["ic_index"], snip_arrays["ic_index"]):
        raise ValueError("the snip test must use Experiment 1's states, in "
                         "the same order")
    if list(snip_meta["horizons"]) != list(exp1_meta["horizons"]):
        raise ValueError("horizons differ from Experiment 1's")
    if list(snip_meta["objective_names"]) != list(
            exp1_meta["objective_names"]):
        raise ValueError("objectives differ from Experiment 1's")
    cfg = exp1_meta["config"]
    if (float(snip_meta["a0"]) != float(cfg["a0"])
            or int(snip_meta["tail_days"]) != int(cfg["tail_days"])):
        raise ValueError("operating point or tail differ from Experiment 1's")
    old = list(exp1_meta["windows"])
    new = list(snip_meta["windows"])
    if set(old) & set(new) or 0 in new:
        raise ValueError("snip windows must be new and truncated")
    order = sorted(w for w in old if w != 0) + sorted(new) + [0]
    old_jac = np.asarray(exp1_arrays["jacobians"], float)
    new_jac = np.asarray(snip_arrays["jacobians"], float)
    k_g = min(old_jac.shape[1], new_jac.shape[1])
    pieces = [old_jac[:, :k_g, old.index(w)] if w in old
              else new_jac[:, :k_g, new.index(w)] for w in order]
    arrays = dict(exp1_arrays)
    arrays["jacobians"] = np.stack(pieces, axis=2)
    meta = dict(exp1_meta)
    meta["windows"] = order
    return arrays, meta


def choose_w_long(cells):
    """Return ``W_long`` from ``{window: cell}`` at the key (None if none useful)."""
    useful = [(w, c["ratio"]) for w, c in cells.items()
              if c["verdict"] == "useful"]
    if not useful:
        return None
    best = min(abs(r - 1.0) for _, r in useful)
    return min(w for w, r in useful if abs(r - 1.0) <= best + RATIO_TIE)


def summarize(result):
    """Apply Part A's registered rules to the merged analysis."""
    cells = {w: result["cells"][f"W{w}|{KEY}"] for w in COMPARED_WINDOWS}
    w_long = choose_w_long(cells)
    outcome = ("S-U" if w_long is None else "S-B" if w_long == 14
               else "S-A")
    ratios = [cells[w]["ratio"] for w in COMPARED_WINDOWS]
    angles = [cells[w]["median_single_angle"] for w in COMPARED_WINDOWS]
    table = {}
    for obj in REPORTED_OBJECTIVES:
        for h in REPORTED_HORIZONS:
            for w in (14, 21, 30, 0):
                c = result["cells"].get(f"W{w}|{obj}@{h}")
                if c is not None:
                    table[f"W{w}|{obj}@{h}"] = {
                        k: c[k] for k in ("angle_mean", "angle_ci", "ratio",
                                          "ratio_ci", "median_single_angle",
                                          "noise_to_signal", "verdict")}
    return {"key": KEY, "w_long": w_long, "outcome": outcome,
            "P1_ratio_rises": bool(ratios[0] < ratios[1] < ratios[2]),
            "P2_single_angle_rises": bool(angles[0] < angles[1] < angles[2]),
            "ratios": dict(zip(map(str, COMPARED_WINDOWS), ratios)),
            "median_single_angles": dict(zip(map(str, COMPARED_WINDOWS),
                                             angles)),
            "recomputed_w_star_for_the_record": result.get(
                "primary", {}).get("w_star"),
            "cells": table}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--exp1-prefix", default=EXP1_PREFIX)
    p.add_argument("--snip-prefix", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    exp1_arrays, exp1_meta = load(args.exp1_prefix)
    snip_arrays, snip_meta = load(args.snip_prefix)
    arrays, meta = merge(exp1_arrays, exp1_meta, snip_arrays, snip_meta)
    result = analyze(arrays, meta)
    summary = summarize(result)
    out = {"analysis": "snip test (Amendment 9 revision 1.1, Part A)",
           "created_utc": datetime.now(timezone.utc).isoformat(),
           "inputs": {"exp1": args.exp1_prefix, "snip": args.snip_prefix},
           "windows": meta["windows"], "summary": summary,
           "merged_experiment1_analysis": result}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"outcome {summary['outcome']} (W_long = {summary['w_long']}); "
          f"T0@120 ratios {summary['ratios']}; P1 "
          f"{summary['P1_ratio_rises']}, P2 {summary['P2_single_angle_rises']}"
          f" -> {args.output}")


if __name__ == "__main__":
    main()
