#!/usr/bin/env python
"""Build the project's Zenodo deposit (MCB_PROJECT_REPORT.md Part 18 step 33).

Version 1 holds everything behind the finished results (Experiments 1 and
3a, the step-23 pilot, the planning-cost test) and the starting states of
Experiments 1-3a.

Groups:
``climate``
    The settled Q-flux base climate: the base state as NetCDF, the
    settling summary and fields, and the monthly Q-flux file with its
    correction record.
``states``
    Every starting state of ``ics_macro`` (the macro bases and the roles
    exp1, exp2_train, exp2_eval, exp3_train and exp3_eval) as NetCDF, one
    file per state, with each role's manifest. The older paired-baseline
    pickles (``*_baseline*.pkl``) are not used by Experiments 1-3 and are
    left out.
``references``
    The test-world reference ensembles (``.npz``) with their manifests.
``results``
    Experiment 1's arrays and analysis, the pilot's outputs, Experiment
    3a's runs and analysis, and the planning-cost measurements.
``code``
    A ``git archive`` of the source at a given commit (``--code-tarball``,
    made where the git repository is).

**States are converted, not copied.** Pickles are tied to the code that
wrote them and are unsafe to open from an untrusted source. Each state is
loaded with the model's template, written with ``jcm.mcb.state_netcdf``,
read back and compared leaf by leaf with the pickle. Every comparison is
recorded in ``VERIFICATION.json``, and a single mismatch stops the build.

Output (``--output-dir``):
- ``staging/<group>/...``: the files, each group with its own
  ``CHECKSUMS.sha256``;
- ``upload/``: one ``.tar`` per group (states one per role), plus
  ``README.md``, ``zenodo.json``, ``VERIFICATION.json`` and
  ``CHECKSUMS.sha256`` of the upload files. This folder is what goes to
  Zenodo; nothing here uploads or publishes anything.

Runs on the CPU (``JAX_PLATFORMS=cpu``), so it can share a GPU machine with
running experiments. Resumable: finished files are skipped.

Example (GX10, from an export of the repository):
    JAX_PLATFORMS=cpu nice -n 19 python make_zenodo_archive.py \
        --data-root ~/workspace/jax-gcm/mcb_experiments_gpu \
        --repo-root . --code-tarball jax-gcm-<sha>.tar.gz \
        --output-dir ~/zenodo_archive
"""

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

GROUPS = ("climate", "states", "references", "results", "code")
STATE_ROLES = ("macro_bases", "exp1", "exp2_train", "exp2_eval", "exp3_train",
               "exp3_eval")
REFERENCE_SETS = ("test_world_refs", "test_world_refs_ramp6")
RESULT_DIRS = ("exp3a", "pilot_step23")
RESULT_FILES_FROM_REPO = (
    "mcb_experiments_gpu/exp1/exp1_gradient_fidelity.npz",
    "mcb_experiments_gpu/exp1/exp1_gradient_fidelity.json",
    "mcb_experiments_gpu/exp1/exp1_gradient_fidelity_analysis.json",
    "mcb_experiments_gpu/exp1/campaign_step0_exp1.log",
    "mcb_experiments_gpu/exp1/macro_bases_manifest.json",
    "mcb_experiments_gpu/planner_cost.json",
)
CLIMATE_FROM_REPO = (
    "mcb_experiments/qflux/qflux_monthly_t30_v2.nc",
    "mcb_experiments/qflux/qflux_monthly_t30_v2_correction.json",
)
REGISTRATIONS = {"Amendment 9 (Step 0, Experiment 1)": "https://osf.io/2bs8p/",
                 "Revision 1 (Experiment 3a)": "https://osf.io/7bwe4/",
                 "Revision 1.1 (Experiment 2, snip test)":
                     "https://osf.io/pvwx9/"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True,
                   help="The mcb_experiments_gpu folder holding the data.")
    p.add_argument("--repo-root", default=".",
                   help="The repository (or its export): committed files.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--groups", nargs="+", default=list(GROUPS),
                   choices=GROUPS)
    p.add_argument("--code-tarball", default=None,
                   help="git archive of the source (needed for 'code').")
    p.add_argument("--max-states", type=int, default=None,
                   help="Per role; only for smoke tests.")
    p.add_argument("--dry-run", action="store_true",
                   help="List what would be archived, with sizes.")
    return p.parse_args(argv)


def sha256(path) -> str:
    """Return a file's SHA-256 checksum (streamed)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def size_of(paths) -> int:
    """Return the total size of files, in bytes."""
    return sum(Path(p).stat().st_size for p in paths)


def inventory(args) -> dict:
    """Return ``{group: [(source, destination relative to the group)]}``."""
    data, repo = Path(args.data_root).expanduser(), Path(args.repo_root)
    inv = {g: [] for g in GROUPS}
    eq = data / "equilibrated_qflux"
    inv["climate"] += [(eq / "base_carry.pkl", "base_state.nc")]
    inv["climate"] += [(eq / n, n) for n in ("settle_summary.json",
                                             "settle_fields.npz")
                       if (eq / n).exists()]
    inv["climate"] += [(repo / p, Path(p).name) for p in CLIMATE_FROM_REPO]
    macro = data / "ics_macro"
    for role in STATE_ROLES:
        role_dir = macro / role
        if role == "macro_bases":
            carries = sorted(role_dir.glob("macro_*_carry.pkl"))
            manifest = macro / "macro_bases_manifest.json"
        else:
            entries = json.loads((role_dir / "manifest.json").read_text())
            carries = [role_dir / e["carry_file"] for e in entries["ics"]]
            manifest = role_dir / "manifest.json"
        if args.max_states is not None:
            carries = carries[:args.max_states]
        inv["states"] += [(manifest, f"{role}/{manifest.name}")]
        inv["states"] += [(c, f"{role}/{c.name.replace('_carry.pkl', '.nc')}")
                          for c in carries]
    for ref_set in REFERENCE_SETS:
        for f in sorted((data / ref_set).rglob("*")):
            if f.is_file():
                inv["references"] += [(f, f"{ref_set}/{f.relative_to(data / ref_set)}")]
    for d in RESULT_DIRS:
        for f in sorted((data / d).rglob("*")):
            if f.is_file():
                inv["results"] += [(f, f"{d}/{f.relative_to(data / d)}")]
    inv["results"] += [(repo / p, f"{Path(p).parent.name}/{Path(p).name}")
                       for p in RESULT_FILES_FROM_REPO]
    if args.code_tarball:
        inv["code"] += [(Path(args.code_tarball),
                         Path(args.code_tarball).name)]
    missing = [str(src) for g in args.groups for src, _ in inv[g]
               if not Path(src).exists()]
    if missing:
        raise SystemExit("missing inputs:\n  " + "\n  ".join(missing))
    return {g: inv[g] for g in args.groups}


def convert_state(src: Path, dst: Path, template, record: dict):
    """Write one pickled state as NetCDF and prove the round trip exact."""
    from jcm.mcb import load_carry
    from jcm.mcb.state_netcdf import load_state, save_state, states_equal

    carry = load_carry(str(src), template)
    save_state(dst, carry, {"source_file": src.name,
                            "source_sha256": sha256(src)})
    back = load_state(dst, template)
    ok = states_equal(carry, back)
    record.update({"source": src.name, "source_sha256": sha256(src),
                   "netcdf": dst.name, "netcdf_sha256": sha256(dst),
                   "round_trip_exact": ok})
    if not ok:
        raise SystemExit(f"round trip NOT exact for {src}; stopping")


def build_staging(args, inv) -> dict:
    """Fill ``staging/``: convert states, copy the rest; return verification."""
    out = Path(args.output_dir).expanduser() / "staging"
    verification, template = {"states": []}, None
    for group, items in inv.items():
        t0 = time.time()
        for src, rel in items:
            dst = out / group / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if str(src).endswith("_carry.pkl"):
                done = dst.with_suffix(".nc.ok")
                if done.exists():
                    verification["states"].append(json.loads(done.read_text()))
                    continue
                if template is None:
                    from run_experiment2 import build_model
                    template = build_model()["template"]
                record = {"group": group, "destination": rel}
                convert_state(Path(src), dst, template, record)
                done.write_text(json.dumps(record))
                verification["states"].append(record)
            elif not dst.exists() or dst.stat().st_size != src.stat().st_size:
                shutil.copy2(src, dst)
        lines = [f"{sha256(f)}  {f.relative_to(out / group)}"
                 for f in sorted((out / group).rglob("*"))
                 if f.is_file() and not f.name.endswith(".ok")
                 and f.name != "CHECKSUMS.sha256"]
        (out / group / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n")
        print(f"  {group}: {len(items)} items [{time.time() - t0:.0f}s]",
              flush=True)
    return verification


def pack(args, groups) -> list:
    """Write one tar per group (states: one per role) into ``upload/``."""
    out = Path(args.output_dir).expanduser()
    staging, upload = out / "staging", out / "upload"
    upload.mkdir(parents=True, exist_ok=True)
    tars = []
    for group in groups:
        units = ([p for p in sorted((staging / "states").iterdir())
                  if p.is_dir()] if group == "states"
                 else [staging / group])
        for unit in units:
            name = (f"states_{unit.name}.tar" if group == "states"
                    else f"{group}.tar")
            tar_path = upload / name
            with tarfile.open(tar_path, "w") as tar:
                for f in sorted(unit.rglob("*")):
                    if f.is_file() and not f.name.endswith(".ok"):
                        tar.add(f, arcname=str(f.relative_to(staging)))
                if group == "states":
                    tar.add(staging / "states" / "CHECKSUMS.sha256",
                            arcname="states/CHECKSUMS.sha256")
            tars.append(tar_path)
    return tars


def readme(args, verification, tars) -> str:
    """Return the deposit's README."""
    n_states = len(verification["states"])
    exact = sum(r["round_trip_exact"] for r in verification["states"])
    regs = "\n".join(f"- {k}: {v}" for k, v in REGISTRATIONS.items())
    files = "\n".join(f"- `{t.name}` ({t.stat().st_size / 1e6:,.0f} MB)"
                      for t in tars)
    return f"""# Long-horizon gradients and cloud-brightening control in a coupled JAX climate model: data, version 1

Data behind the project's pre-registered experiments in JAX-GCM (T30, SPEEDY physics) coupled to
jax-esm slab ocean and land models with a monthly Q-flux. Source code:
https://github.com/Thaarak/jax-gcm (branch `fable-version`). The plain-language report is
`MCB_PROJECT_REPORT.md` in that repository.

## Pre-registrations (OSF)

{regs}

## Files

{files}

Each `.tar` unpacks to a folder with its own `CHECKSUMS.sha256` (`sha256sum -c CHECKSUMS.sha256`).
`CHECKSUMS.sha256` next to this README covers the `.tar` files themselves.

- **climate**: the settled base climate (`base_state.nc`), its settling run (`settle_summary.json`,
  `settle_fields.npz`), and the monthly Q-flux (`qflux_monthly_t30_v2.nc`) with how it was derived.
- **states_<role>**: every starting state, one NetCDF file each, with the role's `manifest.json`
  (seeds, macro state, branch). Roles: `macro_bases` (16 ocean states 730 days apart), `exp1`,
  `exp2_train`, `exp2_eval`, `exp3_train`, `exp3_eval`.
- **references**: the test-world reference ensembles (normal climate and uncontrolled warming;
  5-member means and per-member block means), `.npz`.
- **results**: Experiment 1 (gradient fidelity), the step-23 pilot, Experiment 3a (every run's
  summary and member-mean daily fields, and the registered analysis), and the planning-cost
  measurements.
- **code**: the repository's source at the archived commit (named in the file), without the data
  folders archived above. A name ending in `-with-archive-tools` adds the state reader
  (`jcm/mcb/state_netcdf.py`), this builder and the jax-esm bug report from the working tree.

## Reading a state

Each NetCDF state stores one variable per leaf of the model's state pytree (`v0000`, `v0001`, ...),
with the leaf's key path in its `path` attribute and all paths, in order, in the file's `paths`
attribute. To rebuild a state, build the coupled model as the repository does and call
`jcm.mcb.state_netcdf.load_state(path, template)`. It refuses a file whose structure differs from
the template's. All {n_states} states were converted from the original pickles and read back;
{exact} of {n_states} round trips are exact, leaf by leaf (`VERIFICATION.json`).

## Known issue: the slab land model's 12-day year

jax-esm 0.1.0 reads its 12-month land climatology one month per model day, so land temperature in
these runs follows a 12-day "annual cycle" (20 K peak to peak on average). Ocean-temperature
comparisons between controllers share it and stay fair; land temperature and rainfall are not
physical. Report: `upstream_reports/jax-esm-monthly-climatology/` in the source repository.

## License

Data: CC BY 4.0. Code: Apache 2.0 (as the repository).
"""


def zenodo_metadata() -> dict:
    """Return the deposit's Zenodo metadata (a draft for the author to edit)."""
    return {
        "upload_type": "dataset",
        "title": ("Long-horizon gradients and cloud-brightening control in a "
                  "coupled JAX climate model: data (version 1)"),
        "creators": [{"name": "Sriram, Thaarak",
                      "affiliation": "TODO: add affiliation"}],
        "description": (
            "Data behind pre-registered experiments on gradient fidelity "
            "and receding-horizon control of an idealized ocean "
            "cloud-albedo intervention in JAX-GCM coupled to jax-esm slab "
            "ocean and land models: the settled base climate, every "
            "starting state as NetCDF, the reference ensembles, and the "
            "outputs of Experiments 1 and 3a. See README.md."),
        "access_right": "open",
        "license": "cc-by-4.0",
        "keywords": ["differentiable climate model", "JAX", "adjoint",
                     "gradient fidelity", "model predictive control",
                     "cloud brightening", "pre-registration"],
        "related_identifiers": (
            [{"identifier": "https://github.com/Thaarak/jax-gcm",
              "relation": "isSupplementTo", "scheme": "url"}]
            + [{"identifier": url, "relation": "isDocumentedBy",
                "scheme": "url"} for url in REGISTRATIONS.values()]),
        "notes": ("TODO before publishing: confirm the author list and "
                  "affiliations (an open decision for the advisor), and "
                  "the AI-assistance disclosure."),
    }


def main(argv=None):
    args = parse_args(argv)
    sys.path.insert(0, str(Path(args.repo_root).resolve()))
    if "code" in args.groups and not args.code_tarball:
        raise SystemExit("--code-tarball is needed for the 'code' group")
    inv = inventory(args)
    for group, items in inv.items():
        print(f"{group:>10}: {len(items):4d} files, "
              f"{size_of([s for s, _ in items]) / 1e9:6.2f} GB (sources)")
    if args.dry_run:
        return
    t0 = time.time()
    verification = build_staging(args, inv)
    tars = pack(args, list(inv))
    out = Path(args.output_dir).expanduser() / "upload"
    verification["created_utc"] = datetime.now(timezone.utc).isoformat()
    verification["all_exact"] = all(r["round_trip_exact"]
                                    for r in verification["states"])
    (out / "VERIFICATION.json").write_text(json.dumps(verification, indent=2))
    (out / "README.md").write_text(readme(args, verification, tars))
    (out / "zenodo.json").write_text(json.dumps(zenodo_metadata(), indent=2))
    lines = [f"{sha256(t)}  {t.name}" for t in tars]
    (out / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n")
    print(f"-> {out}: {len(tars)} tar files, "
          f"{size_of(tars) / 1e9:.2f} GB, {len(verification['states'])} "
          f"states, all exact: {verification['all_exact']} "
          f"[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
