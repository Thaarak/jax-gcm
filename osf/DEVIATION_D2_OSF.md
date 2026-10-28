# Pre-registration: Amendment 9 revision 5, Deviation D2 (Experiment 3f's target-extension check)

This file is the OSF posting of Deviation D2 of revision 5 of Amendment 9 of this project's
pre-registration (`PREREGISTRATION.md`), and of the conduct note logged with it. Experiment 3f's
evaluation was running when they were logged; no `learn112_mpc` run had started, and no score of the
evaluation had been read. Everything below the horizontal line is the registered text exactly as
committed in the public repository; nothing in it has been edited.

## Source

- Repository: https://github.com/Thaarak/jax-gcm (branch `fable-version`)
- Commit: `02243afa4e8a1f48c39924d9d41cbd6cb35492b5`, committed 2026-10-26 23:28:57 -0700
- SHA-256 of `PREREGISTRATION.md` at that commit: `90db117ef6d781b86617e65a3ee713fa1f21366fa96f4ad3376aea934d9610fc`
- SHA-256 of the section reproduced below (from "Amendment 9, revision 5 — Deviation D2" to the end
  of that file): `a6ca52caa60eb5959161368977d0e2d8ad74aa48359a2be0d952dfa09d9cb3ec`
- SHA-256 of the deviation's code and evidence, and of the registered files it leaves unchanged, at
  that commit:
  - `analyze_experiment3f_d2.py`: `a32bb748d689d646255a040ecf8b5aa6118b918ac60ad45a33e8bf2f6943c76d`
  - `analyze_experiment3f_d2_test.py`: `9300f3ff977c8b6adc32c5a4186d13918df9b13619d89cca5b6fd20134020122`
  - `mcb_experiments_gpu/exp3f/deviation_d2/check_extensions.py`: `70530a25549a971b588b4a458221b74c2be5d596b6084254c924b4db6f3f5326`
  - `mcb_experiments_gpu/exp3f/deviation_d2/extensions_check.json`: `26ddd1e67351e89c7b5490a1a554d678688e1b8b2a9bfccc84f16899ba3a24b9`
  - `mcb_experiments_gpu/exp3f/deviation_d2/rerun_10100.py`: `24b2281396a0a85db4386779488dffb1381b433d4709c74319020c8f0aa4c7bf`
  - `mcb_experiments_gpu/exp3f/deviation_d2/rerun_10100.json`: `1491ed607549f9a967e3073661aa00f76758b8ec1075b10145d5c6ae4cefcccb`
  - `analyze_experiment3f.py`: `1496d3305fe84fdd4a5dd769a0bb93140ea406fa660368b8050f54fcc3c76e3b`
  - `run_campaign_exp3f_eval.sh`: `2f256d48c0f2ca910cfdfd21dfd8474499ce2f9c4f5e7c490776643e50868c3d`
  - `exp3f_common.sh`: `19fcb8372d593877d2202366e67fe625c707ce4c3b4b0bc6ac24659115526c9b`
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
- 2026-10-26 23:28:57 -0700: the commit above. 2026-10-27 18:33 PDT: this posting.

---

**Amendment 9, revision 5 — Deviation D2 (logged 2026-10-26 23:28 PDT, before any `learn112_mpc`
run had started and before any score of Experiment 3f's evaluation had been read).** F4 has the
analysis refuse a run whose target extension differs from its scoring references by more than 0.05 K:
the mean absolute difference of the two files' SST, averaged over their last 14 shared days. That
limit came from the pilot, whose extensions repeated 3d's training references exactly (0.0000 K). The
48 evaluation extensions were built as F4 says (2026-10-26, 19:27–22:06 PDT), but they do not repeat
3d's evaluation references: they differ by 0.108–0.151 K (median 0.126 K). The registered analysis
would therefore refuse every `learn112_mpc` run and stop. A check of the extensions alone found this
at 22:10 PDT.

* *Cause.* 3d's evaluation references (built 2026-10-11 17:05 to 2026-10-12 00:46 PDT, while
  Experiment 3c's evaluation shared the GPU) cannot be reproduced bit for bit. Every extension equals
  its references on the first day and differs from the second day on. State 10100 was run again on
  2026-10-26 (22:13–22:17 PDT) with 3d's own 200-day program and two of its members. Member 0, the
  initial state itself with no perturbation, differs from 3d's stored copy from the first 5-day block
  (by up to 0.012 K) and by up to 3.2 K at day 200; member 1 behaves the same. Members run one at a
  time, so a member's trajectory does not depend on the number of members, the stored blocks or the
  warmed run: the GPU, not a setting, makes the difference. F10 already says that GPU runs are not
  bit-reproducible; F4 assumed, from the pilot, that reference runs were.
* *Size.* The differences are those of two independent 10-member ensembles of the same state. From
  3d's own members (members 0–4 against 5–9 over days 185–199, scaled to 10 against 10), the same
  statistic is 0.105–0.149 K (median 0.129 K); observed over expected has median 0.98 (0.77–1.30).
  The member seeds, the land model and the lengths are as registered. There is no climate offset: the
  signed mean difference is +0.003 K (sd 0.010 K, at most 0.025 K in size; positive in 29 of 48
  states). The extensions are the normal climate with different weather.
* *Change.* The limit on the recorded mean absolute difference becomes 0.20 K, above every value
  here. That statistic cannot by itself tell weather from another climate: 3d's warmed references
  differ from its normal ones by 0.18–0.23 K on it. So the analysis also refuses a run whose signed
  mean difference over the same 14 days exceeds 0.05 K in size: weather gave at most 0.025 K, and the
  warmed climate gives +0.15 to +0.19 K. Nothing else changes. The runs use the extensions as built,
  and scoring never uses them (F4). They steer only the `learn112_mpc` re-plans whose look-ahead
  passes day 200 (those from day 88 on), with the same kind of weather noise that 3d's references
  carry on every day.
* *How it is applied.* The campaign script runs the registered analysis at the end; it will stop at
  the old limit ("ANALYSIS FAILED"). The analysis is then run, with the same arguments, as
  `analyze_experiment3f_d2.py`, which applies the two limits and otherwise runs
  `analyze_experiment3f.py` unchanged. Its output records the limits (`registered.max_overlap_k` and
  `checks.d2_*`). Tests: `analyze_experiment3f_d2_test.py`.
* *Timing.* The 0.20 K limit was decided at 23:14 PDT, and the signed check at 23:17 PDT, after the
  warmed references had been measured on the same statistic. At that point 8 of the 48 `learn14` runs
  had finished, none of their scores had been read, and no `learn112_mpc` run had started. Both
  limits come from the extension and reference statistics above, which involve no planner run.
* *Files.* `mcb_experiments_gpu/exp3f/deviation_d2/` holds `check_extensions.py` with its output
  `extensions_check.json`, and `rerun_10100.py` with `rerun_10100.json` (the commands, times and
  comparisons). The reruns' fields are on the GX10 (`~/workspace/exp3f_extdiag`).

**Amendment 9, revision 5 — conduct note (logged 2026-10-26 23:28 PDT).** At 23:23:38 PDT the
GX10's kernel stopped the evaluation campaign for lack of memory. The cause was a check made for
Deviation D2 that loaded the analysis without restricting JAX to the CPU, so JAX claimed most of the
machine's memory, which its GPU shares. Nine `learn14` runs had finished. The four that were running
(states 10700, 10701, 10702 and 10900) were killed in their first attempt. The campaign was restarted
by hand at 23:25:12 PDT with the same command, as F10 says. It kept the nine finished runs and is
running the four killed ones again as their second attempt (attempts were not reset); a run that
fails again is dropped, as registered.
