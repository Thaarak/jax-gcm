"""Turn the pilot's outputs into revision 2's registered design and text.

Usage: python freeze_rev2.py   (run in the repo after copying exp3b/pilot and
exp3b/train/ladder.json and ics_macro3b/macro_bases_manifest.json locally)
Writes mcb_experiments_gpu/exp3b/registered_design.json and appends the
filled revision 2 to PREREGISTRATION.md (refuses if any placeholder is left).
"""
import datetime
import hashlib
import json
import re
from pathlib import Path

import os
REPO = Path(os.environ.get("REPO_ROOT", "/Users/thaaraksriram/workspace/jax-gcm"))
SCRATCH = Path(__file__).parent
E = REPO / "mcb_experiments_gpu"
noise = json.loads((E / "exp3b/pilot/noise.json").read_text())
dec = json.loads((E / "exp3b/pilot/decisions.json").read_text())
ladder = json.loads((E / "exp3b/train/ladder.json").read_text())
macro = json.loads((E / "ics_macro3b/macro_bases_manifest.json").read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731

chosen = dec["chosen"]
learner = chosen["learner"]
mode = "global" if learner == "plan_learn_global" else "bands"
noise_k = noise["noise_k"] * (4.0 if learner == "plan_learn_cautious" else 1.0)
pi_days = 84.0 if chosen["pi"] == "pi_slow" else 42.0
members = int(chosen["members"])
fixed = [round(float(x), 6) for x in
         ladder["pooled"]["linear_response"]["amplitudes"]]
design = {
    "experiment": "3b", "revision": "Amendment 9 revision 2",
    "members": members, "reference_members": 10,
    "learner_mode": mode, "learner_noise_k": round(noise_k, 8),
    "learner_prior_sd": 1.0, "learner_bounds": [0.2, 5.0],
    "pi_closed_loop_days": pi_days, "fixed_amplitudes": fixed,
    "pilot": {"noise_k_measured": noise["noise_k"], "noise_n": noise["n"],
              "switches": dec["switches"], "chosen": chosen},
}
dpath = E / "exp3b/registered_design.json"
dpath.write_text(json.dumps(design, indent=2) + "\n")

diag = macro["diagnostics"]
tr = diag["trend_K_per_decade"]
bal = diag["split_balance"]
gen = (f"ocean-mean SST {min(diag['ocean_mean_sst_K']):.2f}-"
       f"{max(diag['ocean_mean_sst_K']):.2f} K (states 0-15: 289.27-290.04 K); "
       f"trend {tr['slope']:+.3f} +- {tr['se']:.3f} K per decade; evaluation "
       f"minus training {bal['eval_minus_train_K']:+.3f} K")
pw = dec["power"]
fx = dec["pilot_effects"]
means = dec["arm_means"]
power_lines = "; ".join(
    f"{m} members: minimum detectable effect {float(p['mde']):.5f} "
    f"({float(p['mde_fraction']):.0%} of J_pi)" for m, p in pw.items())
switch_lines = "; ".join(
    f"`{v}` vs `{s['default']}` {s['mean_diff']:+.5f} (SE {s['se']:.5f}): "
    f"{'switched' if s['switch'] else 'kept the default'}"
    for v, s in dec["switches"].items())
arm_lines = ", ".join(f"`{a}` {m['J_zonal']:.5f}"
                      for a, m in sorted(means.items(),
                                         key=lambda kv: kv[1]["J_zonal"]))
power_summary = (
    f"Pilot (16 training states, 3 members; MCB_PROJECT_REPORT.md Part 24): mean `J_zonal` "
    f"{arm_lines}. Learner noise level (rule): {noise['noise_k']:.5f} K from "
    f"{noise['n']} forecast misses. Switches (2-SE rule): {switch_lines}. "
    f"Noise of the H1 difference by members (variance over states): "
    f"{ {int(k): float(v) for k, v in dec['noise_by_members'].items()} }, fitted A + B/M "
    f"with A = {dec['noise_fit']['A']:.3g}, B = {dec['noise_fit']['B']:.3g}. "
    f"Power for H1 (16 macro states x 3 branches, between-macro spread 0.0005 added): "
    f"{power_lines}. Chosen: **{members} members**. Pilot effects (training states, "
    f"descriptive): H1 {fx['H1']['rel']:+.0%}, H2 {fx['H2']['rel']:+.0%}, H3 "
    f"{fx['H3']['rel']:+.0%}, naive vs oracle {fx['naive_vs_oracle']['rel']:+.0%}, "
    f"learner vs oracle {fx['learn_vs_oracle']['rel']:+.0%}.")
planner_minutes = 3.7 * members
cost_h = (48 * (3 * planner_minutes + 4 * 0.6 * members)) / 60.0
text = (SCRATCH / "rev2_draft.md").read_text()
fill = {
    "FREEZE_DATE": datetime.datetime.now().strftime("%Y-%m-%d"),
    "GENERATION_DIAGNOSTICS": gen,
    "MEMBERS": str(members),
    "TABLE_SHA": sha(E / "exp3b/hidden_strength.json"),
    "LEARNER_MODE": "each band's" if mode == "bands" else "one shared",
    "NOISE_K": f"{noise_k:.5f}",
    "PI_DAYS": f"{pi_days:.0f}",
    "FIXED_AMPLITUDES": ", ".join(f"{x:.4f}" for x in fixed),
    "DESIGN_SHA": sha(dpath),
    "POWER_SUMMARY": power_summary,
    "COST": f"{cost_h:.0f} (about {cost_h / 4:.0f} hours wall time, 4 at a time)",
}
draft_keys = set(re.findall(r"\b(?:FREEZE_DATE|GENERATION_DIAGNOSTICS|MEMBERS|"
                            r"TABLE_SHA|LEARNER_MODE|NOISE_K|PI_DAYS|"
                            r"FIXED_AMPLITUDES|DESIGN_SHA|POWER_SUMMARY|COST)\b",
                            text))
missing = draft_keys - set(fill)
if missing:
    raise SystemExit(f"no value for {sorted(missing)}")
for key in sorted(fill, key=len, reverse=True):     # longest first
    text = text.replace(key, fill[key])
left = [k for k in fill if re.search(rf"\b{k}\b", text)]
if left:
    raise SystemExit(f"placeholders left: {left}")
prereg = REPO / "PREREGISTRATION.md"
prereg.write_text(prereg.read_text().rstrip("\n") + "\n" + text)
print(json.dumps(design, indent=2))
print(power_summary)
