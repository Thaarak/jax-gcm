"""Post Deviation D2 of Amendment 9 revision 5 to OSF: cover note + files, then a registration.

Usage: python osf_post_d2.py <commit>  [--dry-run]
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
D2 = "mcb_experiments_gpu/exp3f/deviation_d2"
FILES = ["analyze_experiment3f_d2.py", "analyze_experiment3f_d2_test.py",
         f"{D2}/check_extensions.py", f"{D2}/extensions_check.json",
         f"{D2}/rerun_10100.py", f"{D2}/rerun_10100.json",
         "analyze_experiment3f.py", "run_campaign_exp3f_eval.sh", "exp3f_common.sh"]
SUBJECTS = ["584240d954be81056ceca9a1", "584240d954be81056ceca9de",
            "584240db54be81056cecacd9", "584240da54be81056cecaad4"]


def git(*args):
    return subprocess.run(["git", "-C", REPO, *args], check=True,
                          capture_output=True).stdout


full = git("rev-parse", COMMIT).decode().strip()
when = git("show", "-s", "--format=%ci", full).decode().strip()
prereg = git("show", f"{full}:PREREGISTRATION.md")
text = prereg.decode()
start = text.index("**Amendment 9, revision 5 — Deviation D2")
section = text[start:]
sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
file_lines = "\n".join(f"  - `{p}`: `{sha(git('show', f'{full}:{p}'))}`" for p in FILES)
now = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
cover = f"""# Pre-registration: Amendment 9 revision 5, Deviation D2 (Experiment 3f's target-extension check)

This file is the OSF posting of Deviation D2 of revision 5 of Amendment 9 of this project's
pre-registration (`PREREGISTRATION.md`), and of the conduct note logged with it. Experiment 3f's
evaluation was running when they were logged; no `learn112_mpc` run had started, and no score of the
evaluation had been read. Everything below the horizontal line is the registered text exactly as
committed in the public repository; nothing in it has been edited.

## Source

- Repository: https://github.com/Thaarak/jax-gcm (branch `fable-version`)
- Commit: `{full}`, committed {when}
- SHA-256 of `PREREGISTRATION.md` at that commit: `{sha(prereg)}`
- SHA-256 of the section reproduced below (from "Amendment 9, revision 5 — Deviation D2" to the end
  of that file): `{sha(section.encode())}`
- SHA-256 of the deviation's code and evidence, and of the registered files it leaves unchanged, at
  that commit:
{file_lines}
- Revision 5 itself (Experiment 3f frozen in full) is https://osf.io/eum9h/; the earlier postings are
  listed there.

## Timeline (2026-10-26, PDT)

- 19:26:06 revision 5 registered (eum9h); 19:26:30 Experiment 3f's evaluation started on the GX10.
- 19:27–22:06 the 48 target extensions were built; 22:06:29 the `learn14` runs started.
- 22:10 a check of the extensions alone found them 0.108–0.151 K from 3d's references (limit 0.05 K).
- 22:13–22:17 state 10100 was run again to find the cause.
- 23:14 and 23:17 Deviation D2 was decided (8 of 48 `learn14` runs finished, none of their scores
  read; no `learn112_mpc` run started).
- 23:23:38 the campaign was stopped for lack of memory; 23:25:12 it was restarted by hand (conduct
  note below).
- {when}: the commit above. {now}: this posting.

---

{section}"""
out = os.path.join(REPO, "osf", "DEVIATION_D2_OSF.md")
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


# Upload names are unique in the project's storage, so the evidence files get
# the experiment's prefix.
UPLOADS = [(out, None), (prereg_out, None),
           (os.path.join(REPO, "analyze_experiment3f_d2.py"), None),
           (os.path.join(REPO, D2, "check_extensions.py"), "exp3f_d2_check_extensions.py"),
           (os.path.join(REPO, D2, "extensions_check.json"), "exp3f_d2_extensions_check.json"),
           (os.path.join(REPO, D2, "rerun_10100.py"), "exp3f_d2_rerun_10100.py"),
           (os.path.join(REPO, D2, "rerun_10100.json"), "exp3f_d2_rerun_10100.json")]
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
    "This registration logs Deviation D2 of revision 5 of Amendment 9 of the pre-registration for "
    "a study of long-horizon gradients and receding-horizon control in a differentiable coupled "
    "atmosphere-slab-ocean model (JAX-GCM with jax-esm), with a conduct note. Experiment 3f's "
    "16-week planner aims at a normal-climate target that, after day 200, comes from a longer run "
    "of each state (a target extension). The registered analysis refuses a run whose extension "
    "differs from the scoring references by more than 0.05 K, a limit set when the pilot's "
    "extensions repeated their references exactly. The evaluation's references cannot be "
    "reproduced bit for bit on the GPU, and its extensions differ from them by 0.11-0.15 K, which "
    "is the weather noise between two independent 10-member ensembles, with no climate offset. "
    "Deviation D2, decided before any 16-week run had started and before any score was read, "
    "raises that limit to 0.20 K and adds a check of the signed difference (0.05 K) that does "
    "separate another climate; nothing else changes. The conduct note records a restart of the "
    "evaluation after an out-of-memory stop.\n\n"
    f"The registered text is in DEVIATION_D2_OSF.md, exactly as committed in {full[:12]} of "
    "https://github.com/Thaarak/jax-gcm, with SHA-256 checksums of the text, code and evidence.")
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
                            "pre-registration, Amendment 9 revision 5, Deviation D2 "
                            "(Experiment 3f)",
                   "registration_responses": {keys[0]: SUMMARY}}}})
call("PUT", f"https://api.osf.io/v2/draft_registrations/{D}/relationships/subjects/",
     {"data": [{"type": "subjects", "id": i} for i in SUBJECTS]})
r = call("POST", "https://api.osf.io/v2/registrations/", {"data": {"type": "registrations",
         "attributes": {"draft_registration": D, "registration_choice": "immediate"}}})["data"]
a = r["attributes"]
print("registration:", r["id"], r["links"].get("html"))
for k in ("date_registered", "public", "pending_registration_approval"):
    print(f"  {k}: {a.get(k)}")
open(os.path.join(REPO, "osf", "osf_registration_d2_id.txt"), "w").write(r["id"] + "\n")
