"""Post Amendment 9 revision 5 to OSF: cover note + files, then a registration.

Usage: python osf_post_rev5.py <commit>  [--dry-run]
Run only after the user has approved the posting.
Reads the token from ~/.osf_token and never prints it.
"""
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request

REPO = "/Users/thaaraksriram/workspace/jax-gcm"
NODE = "pqabf"
COMMIT = sys.argv[1]
DRY = "--dry-run" in sys.argv
FROZEN_CODE = ["analyze_experiment3f.py", "analyze_exp3f_pilot.py",
               "analyze_experiment3e.py", "analyze_exp3e_pilot.py",
               "analyze_experiment3b.py", "analyze_experiment3a.py",
               "exp3f_common.sh", "run_campaign_exp3f_pilot.sh",
               "run_campaign_exp3f_eval.sh",
               "jcm/mcb/planner.py", "jcm/mcb/land_climatology.py",
               "jcm/mcb/test_world.py", "jcm/mcb/strength_estimator.py",
               "jcm/mcb/hidden_strength.py", "jcm/mcb/scores.py",
               "run_test_world.py",
               "mcb_experiments_gpu/exp3f/registered_design.json",
               "mcb_experiments_gpu/exp3f/pilot/summary.json",
               "mcb_experiments_gpu/exp3d/registered_design.json",
               "mcb_experiments_gpu/exp3d/hidden_strength.json"]
SUBJECTS = ["584240d954be81056ceca9a1", "584240d954be81056ceca9de",
            "584240db54be81056cecacd9", "584240da54be81056cecaad4"]


def git(*args):
    return subprocess.run(["git", "-C", REPO, *args], check=True,
                          capture_output=True).stdout


full = git("rev-parse", COMMIT).decode().strip()
when = git("show", "-s", "--format=%ci", full).decode().strip()
prereg = git("show", f"{full}:PREREGISTRATION.md")
text = prereg.decode()
start = text.index("**Amendment 9, revision 4 — Experiment 3e RESULT")
section = text[start:]
sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
code_lines = "\n".join(f"  - `{p}`: `{sha(git('show', f'{full}:{p}'))}`"
                       for p in FROZEN_CODE)
now = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
cover = f"""# Pre-registration: Amendment 9 revision 5 (Experiment 3f), with the result of Experiment 3e

This file is the OSF posting of revision 5 of Amendment 9 of this project's pre-registration
(`PREREGISTRATION.md`), together with everything logged since revision 4's posting, in file
order:
* the result of Experiment 3e (revision 4);
* revision 5 (Experiment 3f).
Everything below the horizontal line is the registered text exactly as frozen in the public
repository; nothing in it has been edited.

## Frozen source

- Repository: https://github.com/Thaarak/jax-gcm (branch `fable-version`)
- Commit: `{full}`, committed {when}
- SHA-256 of `PREREGISTRATION.md` at that commit: `{sha(prereg)}`
- SHA-256 of the section reproduced below (from "Amendment 9, revision 4 — Experiment 3e RESULT" to
  the end of that file): `{sha(section.encode())}`
- SHA-256 of the frozen code, designs and pilot summary at that commit:
{code_lines}
- Earlier postings: Amendment 9 with revisions 0.1-0.4 (https://osf.io/2bs8p/); revision 1
  (Experiment 3a; https://osf.io/7bwe4/); revision 1.1 (Experiment 2 and the snip test;
  https://osf.io/pvwx9/); revision 2 (Experiment 3b; https://osf.io/g7y2e/); revision 3
  (Experiments 3c and 3d; https://osf.io/fyt9r/); revision 4 (Experiment 3e; https://osf.io/94euq/).

## Timeline

- 2026-10-19/20: Experiment 3e's evaluation ran (15:52 to 11:27 PDT) and was analysed; its runs and
  analysis were committed in 80ade71c (logged below).
- 2026-10-22: Experiment 3f was designed, and the planner's block schedules and target extension
  were written, with tests.
- 2026-10-23: the smoke test of Experiment 3f's pilot script, which also built the training states'
  target extensions (01:05-01:30 PDT); then the pilot, on 3d's training states only (10:43-15:44
  PDT).
- 2026-10-26: the smoke test of the evaluation script, on one training state (its first attempt
  stopped in the analysis because of a smoke-only setting, which was corrected; the repeat passed);
  a memory probe on the same training state; Experiment 3e's result was logged, and revision 5 was
  frozen in commit {full[:8]}.
- {now}: this posting, before any evaluation run of Experiment 3f.

---

{section}"""
out = os.path.join(REPO, "osf", "REVISION5_OSF.md")
prereg_out = os.path.join(REPO, "osf", f"PREREGISTRATION_at_{full[:8]}.md")
open(out, "w").write(cover)
open(prereg_out, "wb").write(prereg)
print("wrote", out, "and", prereg_out)
if DRY:
    sys.exit(0)

TOKEN = open(os.path.expanduser("~/.osf_token")).read().strip()
if not re.fullmatch(r"[A-Za-z0-9]{40,100}", TOKEN):
    sys.exit("token file malformed; not sending")


def call(method, url, body=None, content_type="application/vnd.api+json"):
    data, headers = None, {"Authorization": f"Bearer {TOKEN}"}
    if body is not None:
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        headers["Content-Type"] = content_type
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code} on {method} {url}:", e.read().decode()[:800])
        sys.exit(1)


# Upload names are unique in the project's storage: revision 2 already holds
# Experiment 3b's `registered_design.json`, so 3f's design and pilot summary get
# the experiment's prefix.
UPLOADS = [(out, None), (prereg_out, None),
           (os.path.join(REPO, "analyze_experiment3f.py"), None),
           (os.path.join(REPO, "analyze_exp3f_pilot.py"), None),
           (os.path.join(REPO, "mcb_experiments_gpu/exp3f/registered_design.json"),
            "exp3f_registered_design.json"),
           (os.path.join(REPO, "mcb_experiments_gpu/exp3f/pilot/summary.json"),
            "exp3f_pilot_summary.json")]
existing, url = {}, f"https://api.osf.io/v2/nodes/{NODE}/files/osfstorage/?page[size]=100"
while url:
    page = call("GET", url)
    for item in page["data"]:
        a = item["attributes"]
        existing[a["name"]] = (a.get("extra", {}).get("hashes", {}).get("sha256"))
    url = (page.get("links") or {}).get("next")
plan = []
for path, name in UPLOADS:
    blob = open(path, "rb").read()
    name = name or os.path.basename(path)
    if name in existing and existing[name] != sha(blob):
        sys.exit(f"{name} already exists on OSF with different content; nothing uploaded")
    plan.append((name, blob, name in existing))
for name, blob, already in plan:
    if already:           # an identical copy from an interrupted earlier run
        print(f"already on OSF, identical: {name}")
        continue
    url = (f"https://files.osf.io/v1/resources/{NODE}/providers/osfstorage/"
           f"?kind=file&name={urllib.parse.quote(name)}")
    attrs = call("PUT", url, blob, "application/octet-stream")["data"]["attributes"]
    osf_sha = attrs.get("extra", {}).get("hashes", {}).get("sha256")
    print(f"uploaded {name}: {attrs.get('size')} bytes, sha256 match: "
          f"{osf_sha == sha(blob)} ({sha(blob)[:16]})")

SUMMARY = (
    "This registration freezes revision 5 of Amendment 9 of the pre-registration for a study of "
    "long-horizon gradients and receding-horizon control in a differentiable coupled "
    "atmosphere-slab-ocean model (JAX-GCM with jax-esm). It logs the result of Experiment 3e, and "
    "freezes Experiment 3f in full. Experiment 3f asks whether a planner that learns an unknown "
    "cloud-brightening strength does better when it plans 16 weeks ahead, as model predictive "
    "control with the atmosphere's gradients truncated every 30 days, than when it plans two "
    "weeks ahead, as in Dubey et al. (2026). On Experiment 3d's 48 held-out states, its primary "
    "hypothesis compares the 16-week planner with the two-week planner, run again alongside it. "
    "The pilot's budget rule, fixed before the pilot ran, dropped the 8-week and hold-one-setting "
    "arms, so the primary hypothesis is the only one tested. Endpoints, tests, verdicts and "
    "interpretations are fixed in advance.\n\n"
    f"The registered text is in REVISION5_OSF.md, exactly as frozen in commit {full[:12]} of "
    "https://github.com/Thaarak/jax-gcm, with SHA-256 checksums of the text, code and designs.")
schemas, url = [], "https://api.osf.io/v2/schemas/registrations/?page[size]=100"
while url:
    page = call("GET", url)
    schemas += page["data"]
    url = (page.get("links") or {}).get("next")
active = [s for s in schemas if s["attributes"].get("name") == "Open-Ended Registration"
          and s["attributes"].get("active", True)]
schema = max(active, key=lambda s: s["attributes"].get("schema_version", 0))
blocks = call("GET", f"https://api.osf.io/v2/schemas/registrations/{schema['id']}/schema_blocks/")
keys = [b["attributes"].get("registration_response_key") for b in blocks["data"]
        if b["attributes"].get("registration_response_key")]
draft = call("POST", "https://api.osf.io/v2/draft_registrations/", {"data": {
    "type": "draft_registrations",
    "relationships": {
        "branched_from": {"data": {"type": "nodes", "id": NODE}},
        "registration_schema": {"data": {"type": "registration-schemas",
                                         "id": schema["id"]}}}}})["data"]
D = draft["id"]
call("PATCH", f"https://api.osf.io/v2/draft_registrations/{D}/", {"data": {
    "type": "draft_registrations", "id": D,
    "attributes": {"title": "Long-horizon gradients in a coupled climate model: "
                            "pre-registration, Amendment 9 revision 5 (Experiment 3f)",
                   "registration_responses": {keys[0]: SUMMARY}}}})
call("PUT", f"https://api.osf.io/v2/draft_registrations/{D}/relationships/subjects/",
     {"data": [{"type": "subjects", "id": i} for i in SUBJECTS]})
r = call("POST", "https://api.osf.io/v2/registrations/", {"data": {"type": "registrations",
         "attributes": {"draft_registration": D, "registration_choice": "immediate"}}})["data"]
a = r["attributes"]
print("registration:", r["id"], r["links"].get("html"))
for k in ("date_registered", "public", "pending_registration_approval"):
    print(f"  {k}: {a.get(k)}")
open(os.path.join(REPO, "osf", "osf_registration_rev5_id.txt"), "w").write(r["id"] + "\n")
