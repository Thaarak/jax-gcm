"""Post Amendment 9 revision 2 to OSF: cover note + files, then a registration.

Usage: python osf_post_rev2.py <commit>  [--dry-run]
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
FROZEN_CODE = ["analyze_experiment3b.py", "analyze_experiment3a.py",
               "analyze_exp3b_pilot.py", "run_campaign_exp3b_eval.sh",
               "run_campaign_exp3b_train.sh", "run_campaign_exp3b_pilot.sh",
               "jcm/mcb/planner.py", "jcm/mcb/strength_estimator.py",
               "jcm/mcb/hidden_strength.py", "jcm/mcb/feedback.py",
               "jcm/mcb/ladder.py", "jcm/mcb/scores.py", "jcm/mcb/test_world.py",
               "run_test_world.py", "run_controllers.py",
               "run_generate_macro_ics.py",
               "mcb_experiments_gpu/exp3b/hidden_strength.json",
               "mcb_experiments_gpu/exp3b/registered_design.json",
               "mcb_experiments_gpu/exp3b/train/ladder.json",
               "mcb_experiments_gpu/exp3b/pilot/decisions.json"]
SUBJECTS = ["584240d954be81056ceca9a1", "584240d954be81056ceca9de",
            "584240db54be81056cecacd9", "584240da54be81056cecaad4"]


def git(*args):
    return subprocess.run(["git", "-C", REPO, *args], check=True,
                          capture_output=True).stdout


full = git("rev-parse", COMMIT).decode().strip()
when = git("show", "-s", "--format=%ci", full).decode().strip()
prereg = git("show", f"{full}:PREREGISTRATION.md")
text = prereg.decode()
start = text.index("**Amendment 9, revision 1 — Deviation D1")
section = text[start:]
sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
code_lines = "\n".join(f"  - `{p}`: `{sha(git('show', f'{full}:{p}'))}`"
                       for p in FROZEN_CODE)
now = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
cover = f"""# Pre-registration: Amendment 9 revision 2 (Experiment 3b), with the Experiment 3a result

This file is the OSF posting of revision 2 of Amendment 9 of this project's pre-registration
(`PREREGISTRATION.md`), together with what was logged after revision 1's posting: Deviation D1 and
the Experiment 3a result.
Everything below the horizontal line is the registered text exactly as frozen in the public
repository; nothing in it has been edited.

## Frozen source

- Repository: https://github.com/Thaarak/jax-gcm (branch `fable-version`)
- Commit: `{full}`, committed {when}
- SHA-256 of `PREREGISTRATION.md` at that commit: `{sha(prereg)}`
- SHA-256 of the section reproduced below (from "Amendment 9, revision 1 — Deviation D1" to the
  end of that file): `{sha(section.encode())}`
- SHA-256 of the frozen code and designs at that commit:
{code_lines}
- Earlier postings: Amendment 9 with revisions 0.1-0.4 (https://osf.io/2bs8p/); revision 1
  (Experiment 3a; https://osf.io/7bwe4/).

## Timeline

- 2026-10-06/07: Experiment 3a ran on its own evaluation states and was analysed (logged below).
- 2026-10-07: fresh macro states 16-47 were generated for Experiment 3b; its training side and pilot
  ran on training states only.
- {now}: this posting, before any evaluation reference, run or score of Experiment 3b exists.

---

{section}"""
out = os.path.join(REPO, "osf", "REVISION2_OSF.md")
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


for path in (out, prereg_out, os.path.join(REPO, "analyze_experiment3b.py"),
             os.path.join(REPO, "mcb_experiments_gpu/exp3b/hidden_strength.json"),
             os.path.join(REPO, "mcb_experiments_gpu/exp3b/registered_design.json")):
    blob = open(path, "rb").read()
    name = os.path.basename(path)
    url = (f"https://files.osf.io/v1/resources/{NODE}/providers/osfstorage/"
           f"?kind=file&name={urllib.parse.quote(name)}")
    attrs = call("PUT", url, blob, "application/octet-stream")["data"]["attributes"]
    osf_sha = attrs.get("extra", {}).get("hashes", {}).get("sha256")
    print(f"uploaded {name}: {attrs.get('size')} bytes, sha256 match: "
          f"{osf_sha == sha(blob)} ({sha(blob)[:16]})")

SUMMARY = (
    "This registration freezes revision 2 of Amendment 9 of the pre-registration for a study of "
    "long-horizon gradients and receding-horizon control in a differentiable coupled "
    "atmosphere-slab-ocean model (JAX-GCM with jax-esm). It logs the Experiment 3a result and a "
    "disclosed analysis deviation, and freezes Experiment 3b in full: on 48 fresh held-out starting "
    "states, the strength of marine cloud brightening is hidden from every controller (an overall "
    "factor of 0.5, 1 or 2 times log-normal regional factors). A 14-day gradient-based planner that "
    "learns the strength from its own forecast misses is compared with the GLENS-style feedback "
    "controller (primary hypothesis), with a planner assuming nominal strength, an oracle planner, "
    "a classical adaptive law and a fixed design, with pre-registered endpoints, tests, power "
    "analysis and interpretation.\n\n"
    f"The registered text is in REVISION2_OSF.md, exactly as frozen in commit {full[:12]} of "
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
                            "pre-registration, Amendment 9 revision 2 (Experiment 3b)",
                   "registration_responses": {keys[0]: SUMMARY}}}})
call("PUT", f"https://api.osf.io/v2/draft_registrations/{D}/relationships/subjects/",
     {"data": [{"type": "subjects", "id": i} for i in SUBJECTS]})
r = call("POST", "https://api.osf.io/v2/registrations/", {"data": {"type": "registrations",
         "attributes": {"draft_registration": D, "registration_choice": "immediate"}}})["data"]
a = r["attributes"]
print("registration:", r["id"], r["links"].get("html"))
for k in ("date_registered", "public", "pending_registration_approval"):
    print(f"  {k}: {a.get(k)}")
open(os.path.join(REPO, "osf", "osf_registration_rev2_id.txt"), "w").write(r["id"] + "\n")
