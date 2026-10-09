"""Turn the pilots' and the preparation's outputs into revision 3.

Usage: python osf/rev3_tools/freeze_rev3.py   (in the repo, after copying from
the GX10: exp3c/pilot/{noise,summary}.json; exp3d/qflux/*summary*.json and
*_correction.json; exp3d/hidden_strength.json; exp3d/train/ladder.json;
exp3d/pilot/noise.json; ics_macro3d/macro_bases_manifest.json)

Writes mcb_experiments_gpu/exp3c/registered_design.json and
mcb_experiments_gpu/exp3d/registered_design.json, and appends the filled
revision 3 to PREREGISTRATION.md (refuses if any placeholder is left).
"""
import datetime
import hashlib
import json
import os
import re
from pathlib import Path

REPO = Path(os.environ.get("REPO_ROOT", "/Users/thaaraksriram/workspace/jax-gcm"))
TOOLS = Path(__file__).parent
E = REPO / "mcb_experiments_gpu"


def load(path):
    return json.loads((E / path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --- Experiment 3c ------------------------------------------------------------
noise3c = load("exp3c/pilot/noise.json")
pilot3c = load("exp3c/pilot/summary.json")
design3c = {
    "experiment": "3c", "revision": "Amendment 9 revision 3, Part A",
    "sensing": "ocean", "members": 6, "reference_members": 10,
    "learner_mode": "bands", "learner_noise_k": round(noise3c["noise_k"], 8),
    "learner_prior_sd": 1.0, "learner_bounds": [0.2, 5.0],
    "reused_from_3b": ["pi", "adaptive", "fixed", "uncontrolled", "plan_learn",
                       "plan_naive", "plan_oracle"],
    "pilot": {"noise": noise3c,
              "power_at_6_members": pilot3c["power_at_6_members"]},
}
d3c = E / "exp3c/registered_design.json"
d3c.write_text(json.dumps(design3c, indent=2) + "\n")

means = pilot3c["arm_means"]
arm_lines = ", ".join(f"`{a}` {m['J_zonal']:.5f}" for a, m in
                      sorted(means.items(), key=lambda kv: kv[1]["J_zonal"]))
effects = "; ".join(f"{name} {e['rel']:+.0%} (t p {e['t_p']:.2g})"
                    for name, e in pilot3c["pilot_effects"].items())
pw = pilot3c["power_at_6_members"]
curve = pilot3c["learning_curves"].get("plan_learn_rs")
curve_text = (f" Learning curve of the ocean-only learner (mean |log(estimate / truth)| per "
              f"band, days 0, 56, 98, 168): {curve[0]:.2f}, {curve[4]:.2f}, {curve[7]:.2f}, "
              f"{curve[12]:.2f}." if curve else "")
pilot3c_text = (
    f"Pilot (3b's 16 pilot states, 3 members; arms without the suffix are 3b's pilot runs): "
    f"mean `J_zonal` {arm_lines}. Learner noise level (3b's rule on the ocean-only oracle): "
    f"{noise3c['noise_k']:.5f} K from {noise3c['n']} forecast misses (exact sensing in 3b: "
    f"0.00067 K). Pilot effects (training states, descriptive): {effects}.{curve_text} "
    f"Power for H1 at the fixed 6 members (3b's method, 16 macro states x 3 branches, "
    f"between-macro spread 0.0005 added): minimum detectable effect {pw['mde']:.5f} "
    f"({pw['mde_fraction_of_J_pi']:.0%} of `pi`'s pilot score). Members stay at 6 so that 3b's "
    f"runs of the reused arms stay valid; the power is recorded, not used to choose.")

# --- Experiment 3d ------------------------------------------------------------
noise3d = load("exp3d/pilot/noise.json")
ladder = load("exp3d/train/ladder.json")
macro = load("ics_macro3d/macro_bases_manifest.json")
fixed = [round(float(x), 6) for x in ladder["pooled"]["linear_response"]["amplitudes"]]
settles = []
for name in ("settle1", "settle2"):
    path = E / "exp3d/qflux" / name / "settle_summary.json"
    if path.exists():
        settles.append((name, json.loads(path.read_text())))
correction = E / "exp3d/qflux/qflux_monthly_t30_v2_correction.json"
base_used = settles[-1][0]
design3d = {
    "experiment": "3d", "revision": "Amendment 9 revision 3, Part B",
    "land_climatology": "monthly", "members": 6, "reference_members": 10,
    "learner_mode": "bands", "learner_noise_k": round(noise3d["noise_k"], 8),
    "learner_prior_sd": 1.0, "learner_bounds": [0.2, 5.0],
    "pi_closed_loop_days": 42.0, "fixed_amplitudes": fixed,
    "base_climate": {name: s["gate"] for name, s in settles},
    "base_carry": f"exp3d/qflux/{base_used}/base_carry.pkl",
    "pilot": {"noise": noise3d},
}
d3d = E / "exp3d/registered_design.json"
d3d.write_text(json.dumps(design3d, indent=2) + "\n")

diag3d = load("exp3d/qflux/diagnose_summary.json")
gate_lines = []
for name, s in settles:
    g = s["gate"]
    gate_lines.append(
        f"{name}: drift {g['drift_k_per_60d']:+.4f} K per 60 d, bias {g['bias_k']:+.3f} K -> "
        f"{'PASS' if g['pass'] else 'FAIL'}")
base_text = (
    "The Q-flux was diagnosed in the fixed-land model (cold start, SST restored to the "
    f"climatology, 365 + 1460 days; `{diag3d.get('qflux_file', 'qflux_monthly_t30.nc')}`), "
    "then settled for 3650 days with the free slab and scored by the registered gate (G1 "
    "|drift| < 0.02 K per 60 days, G2 |mean SST - observed| < 0.5 K). "
    + "; ".join(gate_lines) + "."
    + (" The first settle failed G2, so the one registered Newton correction was applied "
       "(revision 0.3) and the corrected Q-flux settled again, as happened for the original "
       "base climate in 2026-09." if correction.exists() and len(settles) > 1 else "")
    + f" The settled carry of {base_used} is the base of the new states.")
diag = macro["diagnostics"]
tr, bal = diag["trend_K_per_decade"], diag["split_balance"]
gen3d = (f"ocean-mean SST {min(diag['ocean_mean_sst_K']):.2f}-"
         f"{max(diag['ocean_mean_sst_K']):.2f} K; trend {tr['slope']:+.3f} +- {tr['se']:.3f} K "
         f"per decade; evaluation minus training {bal['eval_minus_train_K']:+.3f} K")

# Cost: planner runs about 6 min and classical runs about 1.4 min of GPU time
# at 6 members (3b's evaluation), plus about 4 h of 3d evaluation references.
cost_h = (96 * 6 + 96 * 6 + 192 * 1.4) / 60.0 + 4.0
fill = {
    "FREEZE_DATE": datetime.datetime.now().strftime("%Y-%m-%d"),
    "NOISE3C": f"{noise3c['noise_k']:.5f}",
    "PILOT3C_SUMMARY": pilot3c_text,
    "BASE_CLIMATE": base_text,
    "GENERATION_DIAGNOSTICS_3D": gen3d,
    "TABLE3D_SHA": sha(E / "exp3d/hidden_strength.json"),
    "FIXED3D_AMPLITUDES": "[" + ", ".join(f"{x:.4f}" for x in fixed) + "]",
    "NOISE3D": f"{noise3d['noise_k']:.5f}",
    "DESIGN3C_SHA": sha(d3c),
    "DESIGN3D_SHA": sha(d3d),
    "COST": (f"{cost_h:.0f} GPU-hours on the GX10 (both evaluations together; about the same "
             "in wall time, since they share the GPU)"),
}
text = (TOOLS / "rev3_draft.md").read_text()
missing = [k for k in fill if not re.search(rf"\b{k}\b", text)]
if missing:
    raise SystemExit(f"placeholders not in the draft: {missing}")
for key in sorted(fill, key=len, reverse=True):     # longest first
    text = text.replace(key, fill[key])
left = [k for k in fill if re.search(rf"\b{k}\b", text)]
if left:
    raise SystemExit(f"placeholders left: {left}")
prereg = REPO / "PREREGISTRATION.md"
if "### Amendment 9, revision 3" in prereg.read_text():
    raise SystemExit("revision 3 is already in PREREGISTRATION.md")
prereg.write_text(prereg.read_text().rstrip("\n") + "\n" + text)
print(json.dumps({"3c": design3c, "3d": design3d}, indent=2))
print(pilot3c_text)
print(base_text)
