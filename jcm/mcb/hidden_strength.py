"""The hidden spraying strength of Experiment 3b (MCB_PROJECT_REPORT.md Part 24).

Every state of Experiment 3b gets a strength per band, ``e_k = g * r_k``,
which multiplies the brightening the controller asks for. Only the simulated
world sees it (``run_test_world.py --efficacy``); controllers start from the
nominal strength 1 unless they are the oracle.

* ``g``, the overall factor, is 0.5, 1 or 2. It rotates with
  ``(macro + branch) % 3``, so each evaluation ocean state (three weather
  branches) sees all three. A factor of 2 either way is deliberately milder
  than the roughly 20-fold spread across climate models in the sea salt
  needed for the same cooling (Hirasawa et al., GeoMIP G6-1.5K-MCB), and it
  keeps most runs inside the brightening cap.
* ``r_k``, the regional factors, are log-normal with standard deviation 0.5,
  re-centred to a geometric mean of 1, so ``g`` alone carries the overall
  strength. The strongest and weakest of five bands then typically differ
  about 3-fold, as susceptibility differs between cloud regions.

The draws come from NumPy's ``default_rng`` seeded per state. Because a
library update could in principle change a random stream, the registered
values are also written to a JSON table (``strength_table``), which the
campaigns read and the registration checksums.
"""

import argparse
import json
from typing import Dict, Iterable, List, Sequence

import numpy as np

OVERALL_STRATA = (0.5, 1.0, 2.0)
REGIONAL_SIGMA = 0.5
STRENGTH_SEED0 = 77000


def overall_factor(macro: int, branch: int) -> float:
    """Return ``g`` for a (macro state, weather branch) pair."""
    return OVERALL_STRATA[(int(macro) + int(branch)) % len(OVERALL_STRATA)]


def regional_factors(ic_index: int, k_bands: int = 5,
                     sigma: float = REGIONAL_SIGMA,
                     seed0: int = STRENGTH_SEED0) -> np.ndarray:
    """Return ``r`` (k,), log-normal with geometric mean exactly 1."""
    z = np.random.default_rng(seed0 + int(ic_index)).normal(size=k_bands)
    log_r = sigma * z
    return np.exp(log_r - log_r.mean())


def hidden_strength(macro: int, branch: int, k_bands: int = 5) -> np.ndarray:
    """Return the per-band strength ``g * r`` of one state (index 100 m + b)."""
    return overall_factor(macro, branch) * regional_factors(
        100 * int(macro) + int(branch), k_bands)


def strength_table(states: Iterable[Sequence[int]],
                   k_bands: int = 5) -> Dict[str, dict]:
    """Return ``{index: {macro, branch, overall, strength}}`` for (macro, branch) pairs."""
    table = {}
    for macro, branch in states:
        idx = 100 * int(macro) + int(branch)
        table[f"{idx:04d}"] = {
            "macro": int(macro), "branch": int(branch),
            "overall": overall_factor(macro, branch),
            "strength": [round(float(x), 6)
                         for x in hidden_strength(macro, branch, k_bands)]}
    return table


def manifest_states(manifest_path: str) -> List[tuple]:
    """Return the (macro, branch) pairs of an IC manifest, in its order."""
    with open(manifest_path) as f:
        ics = json.load(f)["ics"]
    return [(e["macro_index"], e["branch"]) for e in ics]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--manifests", nargs="+", required=True,
                   help="IC manifests whose states get a strength.")
    p.add_argument("--output", required=True, help="JSON table to write.")
    args = p.parse_args(argv)
    states = [s for m in args.manifests for s in manifest_states(m)]
    table = {"rule": {"overall_strata": list(OVERALL_STRATA),
                      "overall_index": "(macro + branch) % 3",
                      "regional_sigma": REGIONAL_SIGMA,
                      "seed": f"{STRENGTH_SEED0} + 100 * macro + branch",
                      "generator": "numpy.random.default_rng(seed).normal"},
             "numpy": np.__version__,
             "states": strength_table(states)}
    with open(args.output, "w") as f:
        json.dump(table, f, indent=2)
    print(f"wrote {len(states)} strengths -> {args.output}")


if __name__ == "__main__":
    main()
