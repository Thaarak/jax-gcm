"""Turn Experiment 3f's pilot into revision 5.

Usage: python osf/rev5_tools/freeze_rev5.py --smoke-date YYYY-MM-DD [--design-only]
(in the repo, after copying from the GX10 without field files:
exp3f/pilot/summary.json, exp3f/pilot/*/ic*.json and
exp3f/registered_design.json, which the pilot script writes)

Checks the design against the pilot summary, then appends the filled
revision 5 (rev5_draft.md) to PREREGISTRATION.md. Refuses if a placeholder is
left, if the pilot reported a problem, or if revision 5 is already there.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path

REPO = Path(os.environ.get("REPO_ROOT",
                           "/Users/thaaraksriram/workspace/jax-gcm"))
TOOLS = Path(__file__).parent
E = REPO / "mcb_experiments_gpu"
HEADER = "### Amendment 9, revision 5 — Experiment 3f"
# Revision 5 says "the results of Experiments 3a-3e, logged above".
RESULT_3E = "**Amendment 9, revision 4 — Experiment 3e RESULT"
HYPOTHESIS = {"learn56_mpc": "H2", "learn112_hold": "H3"}
EXTENSION_H = 1.0          # the 48 target extensions, before the runs


def signed(x, digits=4):
    """A signed number with a true minus sign, as in the rest of the file."""
    return f"{x:+.{digits}f}".replace("-", "−")

p = argparse.ArgumentParser()
p.add_argument("--smoke-date", required=True,
               help="Date the evaluation script's smoke test passed.")
p.add_argument("--design-only", action="store_true",
               help="Only check the design and print the filled text.")
args = p.parse_args()

summary = json.loads((E / "exp3f/pilot/summary.json").read_text())
design = json.loads((E / "exp3f/registered_design.json").read_text())
if summary["problems"]:
    raise SystemExit(f"the pilot reported problems: {summary['problems']}")
if design["arms"] != summary["eval_arms"]:
    raise SystemExit("the design's arms are not the budget rule's")
if design["arms"][:2] != ["learn14", "learn112_mpc"]:
    raise SystemExit("learn14 and learn112_mpc must lead the arms")

prereg = REPO / "PREREGISTRATION.md"
current = prereg.read_text()
if RESULT_3E not in current:
    raise SystemExit("log the Experiment 3e result in PREREGISTRATION.md first")

arms = summary["arms"]
gap = max(a["target_overlap_k_max"] or 0 for a in arms.values())
feas = (f"all {sum(a['runs'] for a in arms.values())} runs finished with "
        f"finite scores, and the largest target-extension difference was "
        f"{gap:.4f} K"
        + (": over their shared days, the extensions repeat the references "
           "exactly" if gap == 0 else ""))
dropped = summary["dropped"]
budget = (f"measured speed-up {summary['speedup']:.2f} (the pilot's summed "
          "run time over its wall time, four runs at a time); projected "
          "evaluation "
          + ", ".join(f"`{a}` {h:.1f} h"
                      for a, h in summary["projected_eval_h"].items())
          + f"; total for the kept arms {summary['projected_total_h']:.1f} h "
          f"(budget {summary['budget_h']:.0f} h)"
          + (f". Dropped: {' and '.join(f'`{a}`' for a in dropped)}. So "
             f"{' and '.join(HYPOTHESIS[a] for a in dropped)} "
             f"{'are' if len(dropped) > 1 else 'is'} not tested, and the "
             "descriptive comparisons with a dropped arm are not made"
             + ("; H1 is the only hypothesis"
                if set(dropped) == set(HYPOTHESIS) else "")
             if dropped else "; nothing was dropped"))
scores = "; ".join(
    f"`{a}` J {r['feasibility_only']['J_zonal_mean']:.5f}, bias "
    f"{signed(r['feasibility_only']['bias_K_mean'])} K, cap share "
    f"{r['feasibility_only']['cap_share_mean']:.2f}, "
    f"{r['replan_s_median']:.1f} s per re-plan"
    for a, r in arms.items())
total_h = summary["projected_total_h"]
cost = (f"about {total_h + EXTENSION_H:.0f} hours of wall time on the GX10: "
        f"the budget rule's projection for the evaluation arms ({total_h:.1f} "
        f"h, {48 * len(design['arms'])} runs of {design['members']} members), "
        "plus about an hour for the target extensions")
text = (TOOLS / "rev5_draft.md").read_text()
fill = {"FREEZE_DATE": datetime.date.today().isoformat(),
        "SMOKE_DATE": args.smoke_date, "PILOT_FEASIBILITY": feas,
        "PILOT_BUDGET": budget,
        "PILOT_ARMS": ", ".join(f"`{a}`" for a in design["arms"]),
        "PILOT_SCORES": scores,
        "DESIGN3F_SHA": hashlib.sha256(
            (E / "exp3f/registered_design.json").read_bytes()).hexdigest(),
        "COST": cost}
for k, v in fill.items():
    text = text.replace(k, v)
left = [k for k in fill if k in text]
if left:
    raise SystemExit(f"placeholders left: {left}")
if args.design_only:
    print(text)
    raise SystemExit(0)
if HEADER in current:
    raise SystemExit("revision 5 is already in PREREGISTRATION.md")
prereg.write_text(current.rstrip("\n") + "\n\n" + text.rstrip("\n") + "\n")
print(f"appended revision 5 to {prereg} ({len(text)} characters)")
