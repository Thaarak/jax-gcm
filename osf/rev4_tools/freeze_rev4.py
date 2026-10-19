"""Turn Experiment 3e's pilot into revision 4.

Usage: python osf/rev4_tools/freeze_rev4.py [--design-only] [--smoke-date YYYY-MM-DD]
(in the repo, after copying the pilot from the GX10 without its field files:
exp3e/pilot/{a10,a30,a100}/noise.json, exp3e/pilot/summary.json and
exp3e/pilot/*/plan_oracle/ic*.{json,innovations.npz}; 3d's pilot and design
are already in the repo)

Writes mcb_experiments_gpu/exp3e/registered_design.json. Without
--design-only it also appends the filled revision 4 to PREREGISTRATION.md
(refuses if any placeholder is left, or if revision 4 is already there).
--design-only writes only the design, for the evaluation script's smoke test.
"""
import argparse
import datetime
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("REPO_ROOT", "/Users/thaaraksriram/workspace/jax-gcm"))
TOOLS = Path(__file__).parent
E = REPO / "mcb_experiments_gpu"
LEVELS = {"a10": 0.1, "a30": 0.3, "a100": 1.0}
LEARNING_DAY = 98

p = argparse.ArgumentParser()
p.add_argument("--design-only", action="store_true")
p.add_argument("--smoke-date", default=None,
               help="Date the evaluation script's smoke test passed.")
args = p.parse_args()


def load(path):
    return json.loads((E / path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


pilot = load("exp3e/pilot/summary.json")["levels"]
noise = {lv: load(f"exp3e/pilot/{lv}/noise.json") for lv in LEVELS}
design3d = load("exp3d/registered_design.json")
for lv, f in LEVELS.items():
    assert pilot[lv]["analysis_error"] == f and pilot[lv]["runs"] == 16, lv
    assert pilot[lv]["noise_k"] == noise[lv]["noise_k"], lv
design = {
    "experiment": "3e", "revision": "Amendment 9 revision 4",
    "land_climatology": "monthly", "sensing": "analysis",
    "members": 6, "reference_members": 10,
    "learner_mode": "bands", "learner_prior_sd": 1.0,
    "learner_bounds": [0.2, 5.0],
    "exact_learner_noise_k": design3d["learner_noise_k"],
    "levels": {lv: {"analysis_error": f,
                    "learner_noise_k": round(noise[lv]["noise_k"], 8),
                    "detection_kappa": round(pilot[lv]["kappa"], 8)}
               for lv, f in LEVELS.items()},
    "arms": ["plan_learn_a10", "plan_naive_a10", "plan_learn_a100",
             "plan_learn_a30"],
    "reused_from_3d": ["pi", "adaptive", "fixed", "uncontrolled",
                       "plan_learn", "plan_naive"],
    "pilot": {lv: {"noise": noise[lv],
                   **{k: pilot[lv][k] for k in (
                       "kappa", "kappa_n", "z_null_sd",
                       "oracle_zonal_pattern_mean", "drift_day0",
                       "drift_later")}}
              for lv in LEVELS},
}
d3e = E / "exp3e/registered_design.json"
d3e.write_text(json.dumps(design, indent=2) + "\n")
print(json.dumps({k: v for k, v in design.items() if k != "pilot"}, indent=2))
if args.design_only:
    raise SystemExit(0)
if args.smoke_date is None:
    raise SystemExit("give --smoke-date: the freeze needs the evaluation "
                     "script's smoke test first")


# --- Descriptive pilot numbers for the text ------------------------------------
def per_state(folder):
    return {q.stem: json.loads(q.read_text())["objective"]["controlled"]
            ["zonal_pattern"] for q in sorted(folder.glob("ic*[0-9].json"))}


exact = per_state(E / "exp3d/pilot/plan_oracle")
oracle = {}
for lv in LEVELS:
    j = per_state(E / f"exp3e/pilot/{lv}/plan_oracle")
    assert set(j) == set(exact), lv              # the same 16 states
    d = np.array([np.log(j[k] / exact[k]) for k in sorted(exact)])
    oracle[lv] = (d.mean(), d.mean() / (d.std(ddof=1) / np.sqrt(d.size)))


def shared_factor_sd(level, noise_k):
    """Median sd of one strength factor shared by all bands, by day 98.

    Fisher information sum(g'Wg) / noise^2 over the forecasts verified up to
    day 98 (g = the sum of the bands' sensitivities), per run and member.
    """
    sds = []
    for path in sorted((E / f"exp3e/pilot/{level}/plan_oracle").glob(
            "ic*.innovations.npz")):
        with np.load(path) as f:
            g = f["sensitivity"].astype(np.float64).sum(-1)
            w = f["profile_weights"].astype(np.float64)
            day = f["day"]
        info = (w * g * g).sum(-1)[:, day <= LEARNING_DAY].sum(-1) / noise_k ** 2
        sds.append(1.0 / np.sqrt(info))
    return float(np.median(np.concatenate(sds)))


exact_k = design3d["learner_noise_k"]
log_spread = float(np.std(np.log([s["strength"] for s in
                                  load("exp3d/hidden_strength.json")["states"].values()])))
temp = "atm[<flat index 2>]"                      # temperature variation
day0 = pilot["a10"]["drift_day0"][temp]
lines = []
for lv, f in LEVELS.items():
    r, nz = pilot[lv], noise[lv]
    lines.append(
        f"f = {f:g}: noise level {nz['noise_k']:.5f} K ({nz['noise_k'] / exact_k:.1f} times "
        f"exact sensing) from {nz['n']} forecast misses; kappa {r['kappa']:.5f} K from "
        f"{r['kappa_n']} forecasts with brightening; null spread {r['z_null_sd']:.2f}; the "
        f"oracle's latitude-pattern score {r['oracle_zonal_pattern_mean']:.5f}, against 3d's exact-sensing "
        f"oracle on the same states {oracle[lv][0]:+.3f} in mean log ratio (t {oracle[lv][1]:+.1f}); "
        f"background temperature drift at the later corrections {r['drift_later'][temp]:.2f} K; "
        f"sd of one shared strength factor by day 98 {shared_factor_sd(lv, nz['noise_k']):.2f}")
pilot_text = (
    "The oracle planner, told the true strength, ran with analysis sensing at each level "
    "(2026-10-19, 10:33-13:03 PDT; `run_campaign_exp3e_pilot.sh`) on 3d's 16 branch-1 training "
    "states, with 3 members and 13 segments, in the fixed-land model with 3d's planner settings, "
    "warming and hidden strengths. All 48 runs finished. Per level:\n"
    + "\n".join(f"  * {x};" for x in lines)
    + f"\n  * for comparison: exact sensing (3d's pilot) {exact_k:.5f} K, with a shared-factor sd "
    f"of about {shared_factor_sd('a10', exact_k):.3f} by day 98 (f = 0.1's sensitivities); "
    f"3c's ocean-only level in the old land model 0.04861 K; the background's temperature drift "
    f"on day 0 (another branch's weather) {day0:.2f} K.\n\n"
    "  The shared-factor sd is descriptive: it is computed from the pilot's sensitivities, "
    "not by the learner, which estimates each band separately. For scale, the log of the "
    f"hidden strengths has a standard deviation of {log_spread:.2f} across 3d's states "
    "and bands.")

# 3d and this pilot both ran about 0.98 member-runs per minute with the GPU
# busy (3d: 6 runs of 6 members in 35 min; the pilot: 5 of 3 in 14 min).
cost_h = 192 * 6 / 0.98 / 60.0
fill = {
    "FREEZE_DATE": datetime.datetime.now().strftime("%Y-%m-%d"),
    "SMOKE_DATE": args.smoke_date,
    "PILOT_SUMMARY": pilot_text,
    "NOISE_A10": f"{design['levels']['a10']['learner_noise_k']:.5f}",
    "NOISE_A30": f"{design['levels']['a30']['learner_noise_k']:.5f}",
    "NOISE_A100": f"{design['levels']['a100']['learner_noise_k']:.5f}",
    "DESIGN3E_SHA": sha(d3e),
    "COST": (f"about {cost_h:.0f} GPU-hours on the GX10 (192 runs of 6 members; about the "
             "same in wall time, since 5 runs at a time keep the GPU busy)"),
}
text = (TOOLS / "rev4_draft.md").read_text()
missing = [k for k in fill if not re.search(rf"\b{k}\b", text)]
if missing:
    raise SystemExit(f"placeholders not in the draft: {missing}")
for key in sorted(fill, key=len, reverse=True):     # longest first
    text = text.replace(key, fill[key])
left = [k for k in fill if re.search(rf"\b{k}\b", text)]
if left:
    raise SystemExit(f"placeholders left: {left}")
prereg = REPO / "PREREGISTRATION.md"
if "### Amendment 9, revision 4" in prereg.read_text():
    raise SystemExit("revision 4 is already in PREREGISTRATION.md")
prereg.write_text(prereg.read_text().rstrip("\n") + "\n" + text)
print(pilot_text)
