"""Experiment 3f's registered analysis with Deviation D2 (Amendment 9 revision 5).

Deviation D2 (PREREGISTRATION.md, logged before any learn112_mpc run had
started) changes the target-extension check of F4, and nothing else:
  * the limit on the recorded mean absolute difference (``MAX_OVERLAP_K``)
    goes from 0.05 K to 0.20 K: two independent 10-member ensembles of the
    same state differ by about 0.13 K on it, which the registered limit
    assumed would not happen;
  * because that statistic cannot by itself tell weather from another climate,
    a run is also refused when the signed mean difference of the two files
    over the same 14 days exceeds 0.05 K in size (weather noise gave at most
    0.025 K; 3d's warmed climate differs from the normal one by +0.17 K).
This file applies both and runs ``analyze_experiment3f.py``'s ``main``
unchanged, with the same arguments. Run it from the repository's root (the
runs record the reference and extension files relative to it). The output
records the limits (``registered.max_overlap_k``, ``checks.d2_*``).

Usage: the same as analyze_experiment3f.py.
"""
import json
from pathlib import Path

import numpy as np

import analyze_experiment3f as registered

D2_MAX_OVERLAP_K = 0.20
D2_MAX_SIGNED_K = 0.05
registered.MAX_OVERLAP_K = D2_MAX_OVERLAP_K
# The registered function, even if this file is loaded twice.
_registered_check_runs = getattr(registered.check_runs, "__wrapped__",
                                 registered.check_runs)


def signed_difference(run: dict) -> float:
    """Extension minus references: grid-mean SST over their last 14 shared days."""
    ext = run["target_extension"]
    n = int(ext["own_days"])
    own = np.load(run["references"])["normal_sst"][:n]
    longer = np.load(ext["file"])["normal_sst"]
    return float(np.mean(longer[n - 14:n].mean(0) - own[n - 14:n].mean(0)))


def check_runs(runs_dir, runs_3d_dir, design, arms_3f=registered.ARMS_3F):
    """The registered checks at D2's limit, then D2's signed check."""
    checks = _registered_check_runs(runs_dir, runs_3d_dir, design, arms_3f)
    signed = []
    for arm in arms_3f:
        if arm not in registered.LONG:
            continue
        for js in sorted((Path(runs_dir) / arm).glob("ic*.json")):
            d = signed_difference(json.loads(js.read_text()))
            signed.append(d)
            if abs(d) > D2_MAX_SIGNED_K:
                raise SystemExit(f"{js}: the target extension's climate differs "
                                 f"from the references by {d:+.3f} K")
    checks.update({"d2_max_overlap_k": D2_MAX_OVERLAP_K,
                   "d2_max_signed_k": D2_MAX_SIGNED_K,
                   "d2_target_signed_k_max_abs": max(map(abs, signed), default=None)})
    return checks


check_runs.__wrapped__ = _registered_check_runs
registered.check_runs = check_runs

if __name__ == "__main__":
    registered.main()
