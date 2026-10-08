# Pre-registration: Amendment 9 revision 2 (Experiment 3b), with the Experiment 3a result

This file is the OSF posting of revision 2 of Amendment 9 of this project's pre-registration
(`PREREGISTRATION.md`), together with everything logged after revision 1's posting, in file order:
Deviation D1, the Experiment 3a result, and revision 1.1 (Experiment 2 and the snip test), which
was frozen and posted separately as https://osf.io/pvwx9/ and is reproduced here only because it
sits between them in the file.
Everything below the horizontal line is the registered text exactly as frozen in the public
repository; nothing in it has been edited.

## Frozen source

- Repository: https://github.com/Thaarak/jax-gcm (branch `fable-version`)
- Commit: `676ce331e7b0562c825ab7b81c46d69ee8fc4046`, committed 2026-10-07 22:47:21 -0700
- SHA-256 of `PREREGISTRATION.md` at that commit: `63c85b6ad7b1e79ef7f78e928d7759d6c038001c9aebc0500ab6f241be34a8a5`
- SHA-256 of the section reproduced below (from "Amendment 9, revision 1 — Deviation D1" to the
  end of that file): `10324b3f6a2c6ed7c82a2416f2b5dcc77cdc10f0a2713c95338152219dc050c6`
- SHA-256 of the frozen code and designs at that commit:
  - `analyze_experiment3b.py`: `a2b9816cad63ce27e93afc1881ee2e13ef0ee350bb49b0cb75a85201ed8dbca2`
  - `analyze_experiment3a.py`: `875bb7500033ee9e30cd6315ed3739c5bdb64101337254ca087249736e1693ca`
  - `analyze_exp3b_pilot.py`: `f12413c10a2b08978101bbc6d69a227d8db1b654098d6b9cf31090e8243e0873`
  - `run_campaign_exp3b_eval.sh`: `98b3860a15891473cb13be87f1c87946c2bf68b2a7be4de324b3a748b1b058af`
  - `run_campaign_exp3b_train.sh`: `6970956fa1e61574c8bf2c704558ecaf1ba5edef43a7bfcc5f5c9d02b73ff527`
  - `run_campaign_exp3b_pilot.sh`: `adb04aa5db6cfe96a4837b7500a161aedb3525332602c0427dc5c4103ca7565c`
  - `jcm/mcb/planner.py`: `f533558e63ca2a91f27f6d350bc5e8d861ce365035f8ca0a4febfbbf4483788f`
  - `jcm/mcb/strength_estimator.py`: `01a6884d78116693b04e61960e7db13d10a9ea7fed9f435b922e708f2a1fb276`
  - `jcm/mcb/hidden_strength.py`: `9d334a656e3f40bd25c89f77c5f98e99cad2a86b2283d854b5b895a78cd80140`
  - `jcm/mcb/feedback.py`: `67d7f449b2529eda2338f0a7d469b158583b844f27dd8af7e062cf5345b16039`
  - `jcm/mcb/ladder.py`: `bc644fff67ac56b62ea0e43b4a16b23eae9bf94dc3891263b4489bc1fd973824`
  - `jcm/mcb/scores.py`: `4ff68a872ec6ea54813918986951d73b86ac79f847701f60cb60360c84f58d06`
  - `jcm/mcb/test_world.py`: `adfa255b009ab2095b3538af4e1057d88191f5debdf0c2f6f93014030a9db857`
  - `run_test_world.py`: `e6411fe61127ed86b36addfb07894e884b2d966e5c01f7a3a36588f8afe39120`
  - `run_controllers.py`: `0452dfb5aa7ff62b16e200ec0e2ef3d5ba25f61686cc02b8aacd661c4723c869`
  - `run_generate_macro_ics.py`: `afefe8edb38e63c933375afb94399068354dcb507c974cc50e8967d92c1644d1`
  - `mcb_experiments_gpu/exp3b/hidden_strength.json`: `87e135a35da75de1d49d75b3fe06d9833989f3353f4c2a355473ade47edc92af`
  - `mcb_experiments_gpu/exp3b/registered_design.json`: `e179912c4241ba08588db7d808ef4a32fc17bdbfab0575705e04cb64100c19d4`
  - `mcb_experiments_gpu/exp3b/train/ladder.json`: `29f2a2789dace9ac23cf81f07d49ee77515006262c9df1bc55059f910340416b`
  - `mcb_experiments_gpu/exp3b/pilot/decisions.json`: `8946235f1c3c5bddd2f726f3dbdbc4723f0183b5134b830275041200738107cb`
- Earlier postings: Amendment 9 with revisions 0.1-0.4 (https://osf.io/2bs8p/); revision 1
  (Experiment 3a; https://osf.io/7bwe4/); revision 1.1 (Experiment 2 and the snip test;
  https://osf.io/pvwx9/).

## Timeline

- 2026-10-06/07: Experiment 3a ran on its own evaluation states and was analysed (logged below).
- 2026-10-07: fresh macro states 16-47 were generated for Experiment 3b; its training side and pilot
  ran on training states only.
- 2026-10-07 22:49 PDT: this posting, before any evaluation reference, run or score of Experiment 3b exists.

---

**Amendment 9, revision 1 — Deviation D1 (logged 2026-10-07 12:10 PDT, before any Experiment 3a
analysis output or score existed).** All 240 evaluation runs finished (05:27 PDT; none failed). The
registered `analyze_experiment3a.py` then stopped with `KeyError: 'schedules'` while loading the runs.
The five fixed-pattern arms (`uncontrolled`, `uniform_cancel`, `uniform_effort`, `planner_average`,
`linear_response`) record their single constant pattern as `amplitudes`, not as a per-segment
`schedules` table. The pre-launch check had loaded only planner files.

The fix reads that constant pattern as the same setting in every segment, which is what those arms
applied, and adds a unit test. It touches only two secondary descriptives (`cap_share` and
`max_band_share`). The primary endpoint, every test, the verdicts and the gains are computed from the
saved fields and are unchanged. No other line of the registered analysis changed.

**Amendment 9, revision 1 — Experiment 3a RESULT (logged 2026-10-07; evaluation 2026-10-06 12:22 to
2026-10-07 05:27 PDT; the registered `analyze_experiment3a.py` was applied with Deviation D1 only).**
All 24 states and 10 arms are complete; no run failed and no state was dropped. Primary endpoint
`J_zonal`, `plan120` vs comparator (Holm family of four):
* H1 vs `plan14`: rel +47% (90% CI +8% to +99%), t p 0.037, Holm p 0.11 → **inconclusive**.
* H2 vs `linear_response`: rel +21% (−7% to +64%), Holm p 0.21 → **inconclusive**.
* H3 vs `pi`: rel +35% (+1% to +86%), t p 0.050, Holm p 0.11 → **inconclusive**.
* H4 vs `planner_average`: rel −56% (−65% to −43%), Holm p 0.0051 → **better**.

The Wilcoxon test disagrees with the t test only on H1 (0.055 vs 0.037), and neither survives Holm.
Every planner and classical arm reduces `J_zonal` by 77-84% against `uncontrolled`. `plan120`
overcools (ocean bias −0.031 K, against −0.008 K for `plan14`). Secondary: `plan60` vs `plan120` −33%
(p 0.011), `plan60` vs `plan14` −2% (p 0.91). The full output is
`mcb_experiments_gpu/exp3a/exp3a_analysis.json`, and the plain-language account is
`MCB_PROJECT_REPORT.md` Part 23. By section 7: the H1 claim is not supported, H2+H4 does not establish
"changing over time beats the best fixed pattern", and the H3 reading is "in a perfect model, feedback
on three indices is enough", which moves the planner's case to Experiment 3b.

### Amendment 9, revision 1.1 — Experiment 2 in full, and the snip-window extension of Experiment 1 (frozen 2026-10-07, BEFORE any of their data: no snip-test Jacobian, and no Experiment-2 reference, design run or score exists)

Revision 1 deferred Experiment 2 to "revision 1.1, frozen before any of its data". This is that
revision. It also freezes the short snip-window test that MCB_PROJECT_REPORT.md Part 21.3 added.
Experiment 3b is frozen separately (revision 2), and the two touch no common state.

**What existed when this was frozen.**
* *Experiment 1's result* (outcome A', W* = 14). The arrays are
  `mcb_experiments_gpu/exp1/exp1_gradient_fidelity.npz` (SHA-256
  `fe7624b64197efb29af92dac8786d9c12b227b5b576274d1cc2a18987e607322`) and `.json` (SHA-256
  `954bb84386edf23621fe9142f7f7b2d5dacf29f96ef5f29113fb4346ea5a1533`). At T0 and 120 days, W = 14 is
  `useful`: angle 8.2 deg, ratio 0.69 (95% interval 0.65-0.73). Full backpropagation is `failed`
  (95 deg).
* *Experiment 3a's result* (revision 1; report Part 23).
  * The 120-day planner, one Gauss-Newton step with the 14-day snip, overcooled the ocean by
    0.031 K. That fits the snip's undercount (post hoc; not a registered finding).
  * The five-band linear-response design scored `J_zonal` 0.0041 on `exp3_eval`, against 0.0049
    for one uniform setting and 0.021 with no brightening.
* *No run on any Experiment 2 role.*
  * `exp2_train` has only the 2026-10-02 step-probe references (4 W m^-2). They are training data
    and are not used here.
  * `exp2_eval` has never been run.
  * Experiment 3b uses newly generated macro states (16 and up, `ics_macro3b`).
* *The code frozen below* was tested on the test world's toy model and smoke-tested on the coupled
  model: one `exp2_train` state and one `exp1` state, 4-day episodes (`--smoke`). Smoke outputs are
  flagged, and the analysis refuses them.

**Changes from what was declared** (Amendment 9; Part 18 steps 25-27; Part 21.3).
* *Target.* Amendment 9's declared target (T0 = -0.1 K at a 60-day tail, January) is replaced by
  revision 1's test world. The design must cancel a growing warming, scored on the latitude profile,
  as Part 18 planned.
* *New arms* (Part 21.3):
  * the sunlight-formula opponent;
  * a many-knob layout as a main test;
  * a brute-force arm given the gradient's model time.
* *Iterated gradient design.* The gradient design takes four Gauss-Newton steps. The snipped
  Jacobian is about 30% too small at these horizons (Experiment 1). One step with it therefore
  overshoots (by about 1/0.7), while repeated steps with exact forward errors shrink the error by
  about |1 - 1/c| per step. A toy test confirms this (`jcm/mcb/design_test.py`). The choice is
  fixed here; it is not tested.

**Part A — the snip test (does a longer snip shrink the undercount?).**
* *Estimators:* reverse-mode Jacobians with the atmosphere snipped every **21** and **30** days.
  Everything else is Experiment 1's:
  * its 8 `exp1` states, member 0;
  * a0 = 0.03 in each band;
  * horizons 15, 30, 60 and 120 days, with tail means over the last 10 days;
  * T0, T1, T2 and LAND;
  * windows aligned to the episode start (`run_snip_test.py`).
* *Truth:* Experiment 1's stored central differences, unchanged.
* *Analysis* (`analyze_snip_test.py`). The new Jacobians are merged with Experiment 1's arrays, and
  Experiment 1's frozen `analyze()` runs unchanged: the same metrics, verdict thresholds and
  hierarchical bootstrap (2000 replicates, seed 2026). The bootstrap's draws depend only on the
  numbers of states and members, so Experiment 1's own numbers are reproduced exactly (tested in
  `analyze_snip_test_test.py`).
* *Registered outcome.* `W_long` is chosen among W = 14, 21 and 30 from the windows that are
  `useful` for T0 at 120 days: the one with the projection ratio closest to 1 (ties within 0.05 go
  to the shorter window).
  * **S-A**: `W_long` is 21 or 30. A longer snip removes part of the undercount and stays useful.
  * **S-B**: `W_long` is 14. Fourteen days remains the best window at 120 days.
  * **S-U**: no window is useful. That contradicts Experiment 1, and is reported as an error.
* *Predictions* (confirmed or refuted; no gate):
  * **P1**: the T0 ratio at 120 days rises with the window (14 < 21 < 30).
  * **P2**: the median single-realization angle rises too (the price of a longer memory).
* *Consequences.* W* stays 14 for everything frozen before or with this revision, including Part B.
  The merged analysis's recomputed W* is reported for the record only. Part A informs the gradient
  paper and any later planner registration.
* *Cost:* about 1 GPU-hour (Experiment 1 measured about 190 s per state and window).

**Part B — Experiment 2: can the gradient design one fixed pattern?**

*1. Test world* (revision 1's, unchanged):
* the Q-flux climate;
* a ramp from 0 at day 0 to 6 W m^-2 at day 182 (0.03296703 W m^-2 per day);
* efficacy 1;
* the target is the normal-climate (no warming) 5-member mean of each state's references;
* scoring over days 98-182;
* every setting in [0, 0.15].

Every design is ONE setting, held from day 0 to day 182.

*2. States and references.*
* *Training (design):* the 8 `exp2_train` states (macro states 0, 2, ..., 14, branch 1; indices 1,
  201, ..., 1401).
* *Evaluation:* the 16 `exp2_eval` states (macro states 1, 3, ..., 15, branches 0 and 1; indices
  100, 101, 300, 301, ..., 1500, 1501).
* *References* (`run_experiment2.py references`):
  * normal and warmed runs with no brightening, 5 members (seeds from 93000, member 0 = the state),
    182 days;
  * also each member's window-mean SST and, for the normal runs, the daily cloud cover.
* *Order:* evaluation references are built only after `designs.json` (step 6) exists and its SHA-256
  is logged.

*3. Knob layouts* (ocean-masked Gaussian latitude bands, `jcm.mcb.design.BAND_LAYOUTS`):
* `b5`: Dubey et al.'s 45N, 20N, 0, 20S, 45S with width 10 deg;
* `b13`: 60N to 60S every 10 deg with width 5 deg.

*4. The design problem.* Find `a` in [0, 0.15]^K that minimizes the mean over the 8 training states
of `J_zonal(e_s(a)) + mu |a|^2`, with mu = 6.325e-4 (revision 1).
* `e_s(a)` is the window-mean SST of the run with setting `a`, minus the target's window mean.
* `J_zonal` is `<z>^2 + 0.5 Var(z)` (area-weighted, ocean) of the error's zonal-mean profile `z`.

Under a linear model of `e_s(a)` this is an exact bounded least-squares problem
(`jcm.mcb.design.zonal_design`, scipy BVLS). Every method below solves it, and they differ only in
where the linear model comes from.

*5. Arms* (11; `run_experiment2.ARMS`):

| Arm | Layout | How the setting is found |
|---|---|---|
| `uncontrolled` | b5 | zero |
| `uniform` | b5 | ladder rung 1: one setting in every band cancelling the pooled ocean-mean warming under `brute5`'s linear model |
| `brute5`, `brute13` | b5, b13 | brute force: every band alone at +0.1 for 182 days in the 5 reference members, with the warming. Response maps are member-mean (band run - warmed run) / 0.1; the warmed error is the 5-member warmed run minus the target. Solved exactly. |
| `brute5_eq`, `brute13_eq` | b5, b13 | brute force at equal cost: the same runs, using only their first `M_eq` members (both the band and warmed runs) |
| `grad5`, `grad13` (PRIMARY) | b5, b13 | the gradient: 4 Gauss-Newton steps, see below |
| `bptt5` | b5 | the same as `grad5`, but without the snip (plain backpropagation) |
| `sunlight5`, `sunlight13` | b5, b13 | the sunlight formula: no model run of the brightening or the warming, see below |

*The gradient design* (`jcm.mcb.design.gauss_newton_design`):
* It starts at `a = 0`.
* Each of the 4 steps runs member 0 of every training state once, for 182 days at the current
  setting, with the warming.
  * Forward mode (`jax.jacfwd`) gives the window-mean SST map and its Jacobian over the K settings.
  * The atmosphere is snipped every 14 days from the episode start (W*); `bptt5` uses no snip.
* The pooled problem, linearized at the current setting with the exact forward errors, is then
  solved as in 4.
* A state whose map or Jacobian is not finite sits out that step; if all do, the setting is kept.
* The last iterate is the design.

*Equal cost:* `M_eq = min(5, max(1, floor(4 t_J(K) / ((K + 1) t_F))))`.
* `t_F` is one 182-day forward run.
* `t_J(K)` is one forward-mode Jacobian run with K tangents.
* Both are medians of 2 timed runs after compilation, on the first training state
  (`run_experiment2.py timing`).
* The gradient design costs 4 Jacobian runs per training state; brute force costs K + 1 runs per
  state per member.

*The sunlight formula* (`jcm.mcb.design.sunlight_design`; Duncan's back-of-envelope, report Part
20.2). For band k and an ocean cell:
* *Response:* `-profile_k x I / C`.
  * `I` is the window mean (days 98-182) of the accumulated daily-mean top-of-atmosphere
    insolation times the daily total cloud cover (J m^-2).
  * `C` is the slab's own heat capacity, 1025 x 3985 x (60 + (40 - 60) cos^3 lat) J m^-2 K^-1.
* *Warming:* the accumulated ramp flux over `C`, on the ocean. No feedbacks are included in either.
* *Insolation:* the standard daily-mean formula (S0 = 1361 W m^-2; declination -23.44 deg x
  cos(2 pi (day + 10) / 365.2425)), from the training states' mean calendar start day.
* *Cloud cover:* the 8 training states' normal-climate daily total cloud cover (SPEEDY `cloudc`,
  5-member mean), averaged over states.
* *Solve:* the same bounded least squares.

*6. Frozen designs.* The design stage writes every arm's setting to
`mcb_experiments_gpu/exp2/designs.json`. Its SHA-256 is logged before any evaluation reference
exists, and the evaluation reads only that file.

*7. Evaluation.* Every arm runs on all 16 `exp2_eval` states:
* 5 members, the references' seeds, identical across arms;
* 13 segments of 14 days, setting held, efficacy 1 (`run_experiment2.py episode`).

*8. Endpoints.*
* *Primary:* `J_zonal` of the 5-member-mean SST over days 98-182 against the target (no penalties).
* *Secondary:*
  * `J_map` and the ocean-mean bias;
  * the effort (each arm's own band profiles) and the cap share;
  * the gains G against `uncontrolled` (SST, land temperature, rainfall; descriptive, by
    revision 1's rule R8).
* *Costs* (descriptive): the gradient design's model time against brute force's (5 members and
  `M_eq`), per layout.

*9. Hypotheses* (one Holm family of four, all on `J_zonal`).
* **H1** `grad13` vs `brute13_eq`: with 13 knobs and the same model time, does the gradient design
  beat brute force?
* **H2** `grad13` vs `sunlight13`: does the model's gradient know more than the sunlight formula?
* **H3** `grad5` vs `bptt5`: does snipping the atmosphere make gradient design work at these
  horizons?
* **H4** `grad13` vs `grad5`: do 13 knobs beat 5?

*Secondary comparisons* (unadjusted):
* grad13 vs brute13;
* grad5 vs brute5;
* grad5 vs brute5_eq;
* brute13 vs brute5;
* brute13 vs brute13_eq;
* brute5 vs brute5_eq;
* grad13 vs uniform;
* brute5 vs uniform;
* grad5 vs sunlight5;
* sunlight13 vs uniform;
* sunlight5 vs uniform;
* every arm vs uncontrolled.

*10. Tests and verdicts.* These are revision 1's own functions, imported from
`analyze_experiment3a.py` at this commit by `analyze_experiment2.py`:
* the macro state is the unit (8; branches averaged);
* a paired two-sided t test, with the exact Wilcoxon test alongside;
* a hierarchical bootstrap of `rel = mean(d) / mean(J_B)` (10000 replicates, seed 2026);
* better if Holm p < 0.05 and rel <= -5%; worse if Holm p < 0.05 and rel >= +5%; equivalent if the
  90% interval lies inside [-10%, +10%]; negligible if p < 0.05 and |rel| < 5%; otherwise
  inconclusive.

*11. Interpretation, fixed in advance.*
* **H1**:
  * *better:* "with many knobs, the gradient is the better use of model time than brute force";
  * *equivalent or negligible:* "the gradient design matches brute force at equal model time";
  * *worse:* "at equal model time, brute force wins";
  * *inconclusive:* the efficiency claim is not established. The interval and the cost ratio are
    reported.
* **H2**:
  * *better:* "the model's gradient knows more than geometry about where to brighten";
  * *equivalent, negligible or worse:* "the sunlight formula is as good". The gradient paper's
    design claim then rests on the cost results alone.
* **H3 better:** "snipping is what makes gradient design possible at 3-6-month horizons" (Experiment
  1's finding, carried over to design).
* **H4**:
  * *better:* "finer latitude control helps";
  * *equivalent or negligible:* "five bands suffice for the latitude profile".
* Every arm, every verdict, the costs and the training logs are reported, whatever the outcome.

*Expectations, for the record (not hypotheses).*
* `brute5` should score close to Experiment 3a's `linear_response` (about 0.004).
* By Part 20.2, the sunlight formula overstates each band's cooling about twice.
* Experiment 1 predicts that the `bptt5` Jacobians are noise at 98-182 days.

*12. Failures.*
* A run that fails is re-run once with identical settings.
* An evaluation state that fails twice is dropped from every arm. Losing more than 4 states is a
  deviation.
* A training-side stage that fails twice drops the arms that need it. Their hypotheses are reported
  as not computed, and Holm runs over the hypotheses that remain.
* Non-finite gradient values are not failures; they follow the sit-out rule.
* Nothing is re-tuned after any evaluation run exists.

*13. Code, cost and isolation.*
* *Frozen code:*
  * `jcm/mcb/design.py`, `run_experiment2.py`, `analyze_experiment2.py`;
  * `run_snip_test.py`, `analyze_snip_test.py`;
  * `run_campaign_exp2.sh`;
  * and what they import, at this commit: `jcm/mcb/ladder.py`, `jcm/mcb/planner.py` (its
    `gauss_newton_residuals`), `jcm/mcb/scores.py`, `jcm/mcb/test_world.py`,
    `jcm/mcb/gradient_fidelity.py`, `jcm/mcb/gradient_truncation.py`, `analyze_experiment3a.py` and
    `analyze_gradient_fidelity.py`.
* *Commit:* recorded at posting.
* *Estimated cost:* about 9 GPU-hours.
  * Part A: about 1.
  * Training: references about 0.3, one-band runs about 2.5, gradient designs about 1.5.
  * Evaluation: references about 0.5, 176 episodes of 5 members about 3.
* *Isolation.* The campaign runs on the GX10 from a `git archive` export of this commit in its own
  directory, not from the working copy that Experiment 3b deploys into. It runs only when it cannot
  starve 3b's runs of memory, with every process's memory capped (the GX10 rebooted once from an
  uncapped process).

*14. Posting.* Before any snip-test Jacobian or Experiment-2 run exists, this revision and its frozen
commit are posted to OSF (project https://osf.io/pqabf/), as revision 1 was.

### Amendment 9, revision 2 — Experiment 3b frozen in full (frozen 2026-10-07, BEFORE any evaluation reference, run or score of Experiment 3b)

**What existed when this was frozen.** Nothing ran on `exp3b_eval` beyond generating its starting
states, which uses no outcome. Everything else is on training states or earlier experiments:
* *Experiment 3a's result* (revision 1, logged above). It ran on `exp3_eval` states, none of which
  3b uses.
* *Fresh starting states* (`run_generate_macro_ics.py --plan exp3b`, 2026-10-07):
  * 32 macro states (16-47) continue the same control run from macro state 15, every 365 days.
  * They are interleaved: even states train (`exp3b_train`, branches 0-1) and odd states evaluate
    (`exp3b_eval`, branches 0-2).
  * They never reuse a state that Experiment 3a's results or its noise analysis (2026-10-07) used.
  * Generation diagnostics: ocean-mean SST 289.08-289.93 K (states 0-15: 289.27-290.04 K); trend +0.076 +- 0.042 K per decade; evaluation minus training -0.009 K.
* *Training side* (`run_campaign_exp3b_train.sh`, 2026-10-07):
  * ramp references on the 32 training states;
  * one-band response runs on the 16 branch-0 training states;
  * the fixed design.
* *The pilot* (`run_campaign_exp3b_pilot.sh`, 2026-10-07/08). It ran on the 16 branch-1 training
  states, and its rules were written into `analyze_exp3b_pilot.py` before it ran. Results are in 2
  and 8 below.
* *Evaluation references for `exp3b_eval` did not exist.* `run_test_world.py references` refuses
  evaluation roles without `--allow-eval-roles`.

**Scope.**
* *Frozen in full here:* Experiment 3b, control when the spraying's strength is hidden (Part 18
  steps 29-30 as re-planned in MCB_PROJECT_REPORT.md Part 24).
* *Dropped from 3b:* the copying student and the direct-training race. This follows the 2026-10-02
  advisor meeting: no machine learning for its own sake.
* *Still deferred:* Experiment 2, in its own revision.

**1. Test world.** As revision 1 (Experiment 3a), except for the hidden strength in 3:
* the Q-flux climate, and the warming ramp to 6 W m^-2 at day 182;
* 13 segments of 14 days, scored on days 98-182;
* the cap of 0.15.
* *Target:* the normal-climate mean of the state's reference ensemble. For the evaluation states
  that ensemble has **10 members** (seeds from 93000) over 200 days, which halves the target's own
  noise compared with 3a's 5.

**2. States and members.**
* *States:* all 48 `exp3b_eval` states (macro states 17, 19, ..., 47 x branches 0, 1, 2).
* *Members:* every arm runs **6 members**, the references' first 6 member seeds, identical
  across arms. The number comes from the pilot's registered power rule (8).

**3. Hidden strength** (`jcm/mcb/hidden_strength.py`; table
`mcb_experiments_gpu/exp3b/hidden_strength.json`, SHA-256 87e135a35da75de1d49d75b3fe06d9833989f3353f4c2a355473ade47edc92af).
* *What it is:* each band's brightening is multiplied by `e_k = g x r_k`.
* *The overall factor:* `g` is 0.5, 1 or 2 by `(macro + branch) % 3`, so every evaluation ocean
  state has one branch at each.
* *The regional factors:* `r_k` are log-normal with standard deviation 0.5. They are re-centred to
  a geometric mean of 1, and seeded by `77000 + 100 x macro + branch`.
* *Who sees it:* only the simulated world (`--efficacy`). No controller receives it except the
  oracle.
* *Why this range:* factors of 2 either way are milder than the roughly 20-fold spread across models
  in the sea salt needed for the same cooling (Hirasawa et al., GeoMIP G6-1.5K-MCB). The regional
  spread stands for susceptibility that differs between cloud regions.
* *The strength is constant in time,* an idealization noted in 11.

**4. Arms (7).** Every planner is revision 1's 14-day planner (`short14`):
* Gauss-Newton, 1 step, 1 copy (copy amplitude 0.001 K);
* the zonal objective, alpha = 1, beta = 0.5, mu = 6.325e-4 and lam = 6.325e-3;
* a first guess of half the cap.

| Arm | What it is |
|---|---|
| `plan_learn` (PRIMARY) | starts at nominal strength; before every re-plan it compares its last 14-day forecast of the ocean latitude profile with what happened and updates each band's strength by regularized least squares (`jcm.mcb.strength_estimator`; prior sd 1, noise level 0.00067 K from the pilot's rule, bounds [0.2, 5]) |
| `plan_naive` | assumes nominal strength 1 throughout |
| `plan_oracle` | is told the true strengths. This is perfect knowledge, not necessarily the best possible score: in the pilot it scored worse than the learner (see 8), probably because the 14-day planner over-corrects. It is reported descriptively |
| `pi` | revision 1's GLENS-style PI on T0-T2 with feedforward, closed-loop time 42 d; sensitivities fitted over days 1-60 of the 16 branch-0 training response runs (nominal strength) |
| `adaptive` | revision 1's Tier 2 law (relaxation 0.5), learning one overall strength and scaling the fixed design |
| `fixed` | the best fixed pattern inside the cap under the linear model from the 16 training response runs (zonal objective, days 98-182, mu as above): 0.0245, 0.0303, 0.0125, 0.0132, 0.0165 |
| `uncontrolled` | no brightening |

All registered values are collected in `mcb_experiments_gpu/exp3b/registered_design.json` (SHA-256
e179912c4241ba08588db7d808ef4a32fc17bdbfab0575705e04cb64100c19d4), which the evaluation script reads.

**5. Endpoints.**
* *Primary:* `J_zonal`, exactly as revision 1. It is `alpha <e>^2 + beta Var(e)` of the ocean
  latitude profile of `e`, the member-mean SST over days 98-182 minus the target's mean over the
  same days.
* *Secondary:*
  * `J_zonal`'s two terms (bias and pattern) separately;
  * `J_map` and the ocean-mean bias;
  * the effort, the mean cap share and the largest band's cap share;
  * the gains G against the matched uncontrolled arm;
  * for `plan_learn`, the learning curve `|log(estimate / truth)|` by segment;
  * every endpoint by overall factor.

**6. Hypotheses.**
* **H1 (primary, alone at alpha = 0.05):** `plan_learn` vs `pi`. Does a planner that learns the
  strength beat the classical feedback controller?
* **Secondary family (Holm over two):**
  * **H2:** `plan_learn` vs `plan_naive`. Does learning help the planner?
  * **H3:** `pi` vs `fixed`. Is feedback essential when the strength is unknown?
* *Descriptive (no test claims):*
  * every arm vs `plan_oracle` and vs `uncontrolled`;
  * `plan_naive` vs `pi`, `adaptive` vs `pi`, and `plan_learn` vs `adaptive`;
  * all of it by overall factor.

**7. Tests and outcome grid** (`analyze_experiment3b.py`, frozen with this revision; it imports the
unchanged functions of `analyze_experiment3a.py`). As revision 1:
* `d = J_A - J_B` per state, and `rel = mean(d) / mean(J_B)`;
* macro states as the unit (the 3 branches, one per overall factor, averaged first);
* the paired t test, two-sided, and the exact Wilcoxon test on the 16 macro means;
* the hierarchical bootstrap (10000, seed 2026), with 90% and 95% intervals;
* verdicts: better, worse, equivalent (90% interval inside +-10%), negligible or inconclusive, with
  the 5% threshold. H1 uses its own p; H2 and H3 use Holm-adjusted p.

**8. Power** (the pilot's rule, `analyze_exp3b_pilot.py decide`). Pilot (16 training states, 3 members; MCB_PROJECT_REPORT.md Part 24): mean `J_zonal` `plan_learn` 0.00370, `plan_naive` 0.00397, `plan_learn_cautious` 0.00409, `plan_learn_global` 0.00412, `plan_oracle` 0.00471, `pi` 0.00643, `pi_slow` 0.00655, `adaptive` 0.00675, `fixed` 0.01348, `uncontrolled` 0.02027. Learner noise level (rule): 0.00067 K from 576 forecast misses. Switches (2-SE rule): `pi_slow` vs `pi` +0.00012 (SE 0.00075): kept the default; `plan_learn_cautious` vs `plan_learn` +0.00039 (SE 0.00052): kept the default; `plan_learn_global` vs `plan_learn` +0.00042 (SE 0.00046): kept the default. Noise of the H1 difference by members (variance over states): {1: 4.445272102494732e-05, 2: 1.3885145156810649e-05, 3: 7.203155664371001e-06}, fitted A + B/M with A = 0, B = 3.95e-05. Power for H1 (16 macro states x 3 branches, between-macro spread 0.0005 added): 6 members: minimum detectable effect 0.00117 (18% of J_pi); 8 members: minimum detectable effect 0.00103 (16% of J_pi); 10 members: minimum detectable effect 0.00094 (15% of J_pi). Chosen: **6 members**. Pilot effects (training states, descriptive): H1 -42%, H2 -7%, H3 -52%, naive vs oracle -16%, learner vs oracle -22%.

**9. Interpretation, fixed in advance.**
* *H1 better:* "a planner that learns how strongly and where brightening works, from its own forecast
  misses and using the model's gradients, beats the field's standard feedback controller when the
  strength is unknown".
* *H1 equivalent or negligible:* "classical feedback is as good as a learning planner under strength
  uncertainty". With revision 1's result, the planner's value must then come from elsewhere (many
  knobs; targets beyond three indices; Experiment 2).
* *H1 worse:* "learning planning loses to classical feedback". The cause (learning noise,
  aggressiveness, cap) is described from the descriptive endpoints.
* *H2 better:* learning the strength is what the planner needs. *H2 not better:* the planner's state
  feedback already absorbs the hidden strength.
* *H3 better:* without feedback, a design made at nominal strength fails under realistic strength
  uncertainty. This replicates Tier 2 over a far wider range.
* *Learning curve:* how many fortnights one Earth needs to reveal the strength. This is reported as
  a lower bound, because the planners know the current state exactly (11).
* Every verdict, arm and descriptive comparison is reported, whatever the outcome.

**10. Failures.** As revision 1:
* a failed run is re-run once;
* a state that still fails is dropped from every arm;
* losing more than 6 of 48 states is a deviation to log.

**11. Idealizations, stated in advance.**
* *Perfect state and model, except the strength.* The planners forecast from the exact current
  state. Their two-week forecast misses are therefore small (about 0.0007 K on the latitude profile
  in the smoke test), and learning the strength is easier than it would be from real, noisy
  observations. Realistic sensing is a separate experiment.
* *Constant strength.* It does not drift with season or time.
* *One overall factor per branch.* The factors are balanced within every macro state, so each
  macro-state mean averages the same mix.

**12. Code and cost.**
* *Frozen code:* `analyze_experiment3b.py`, `analyze_experiment3a.py`, `run_campaign_exp3b_eval.sh`,
  `jcm/mcb/planner.py`, `jcm/mcb/strength_estimator.py`, `jcm/mcb/hidden_strength.py`,
  `jcm/mcb/feedback.py`, `jcm/mcb/test_world.py`, `jcm/mcb/scores.py`, `run_test_world.py` and
  `run_controllers.py`. Also frozen: the strength table, the design file and the fixed design, with
  their SHA-256 in the OSF posting.
* *Cost:* about 65 GPU-hours of runs on the GX10 (about 16 hours of wall time, 4 runs at a time), after about 3 hours for the 10-member evaluation references.
* *Posting:* this revision is posted to OSF (project pqabf) before any evaluation reference is built.
