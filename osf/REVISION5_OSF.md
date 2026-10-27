# Pre-registration: Amendment 9 revision 5 (Experiment 3f), with the result of Experiment 3e

This file is the OSF posting of revision 5 of Amendment 9 of this project's pre-registration
(`PREREGISTRATION.md`), together with everything logged since revision 4's posting, in file
order:
* the result of Experiment 3e (revision 4);
* revision 5 (Experiment 3f).
Everything below the horizontal line is the registered text exactly as frozen in the public
repository; nothing in it has been edited.

## Frozen source

- Repository: https://github.com/Thaarak/jax-gcm (branch `fable-version`)
- Commit: `fd4740bd9e46310cf466d1c5e8b185d3b5a8b56e`, committed 2026-10-26 16:35:02 -0700
- SHA-256 of `PREREGISTRATION.md` at that commit: `1a491a0e3dd129329fb514bdcd30e5ed8fd2788d25d71e6296cc486386cb10b6`
- SHA-256 of the section reproduced below (from "Amendment 9, revision 4 — Experiment 3e RESULT" to
  the end of that file): `3cc81efa60b0bee2167221682201ed358ee8b0960d1b277a364b3c8312058ce9`
- SHA-256 of the frozen code, designs and pilot summary at that commit:
  - `analyze_experiment3f.py`: `1496d3305fe84fdd4a5dd769a0bb93140ea406fa660368b8050f54fcc3c76e3b`
  - `analyze_exp3f_pilot.py`: `a02a3c115fcdb01bacc40c67f59c775a75cf4ec355a4a1b7299419666c267a14`
  - `analyze_experiment3e.py`: `eed44366b6761cab711d793897ba1daa915fd3866fa134f1205ab35166c48aa9`
  - `analyze_exp3e_pilot.py`: `c8b15cc5c128dbcaf001b81287947303ffc5fdb3bf7f064cbce5539adb392857`
  - `analyze_experiment3b.py`: `ace60e6575b112eaf019566d6fbfcbeeb34865889a1a89256971146352dc4cc2`
  - `analyze_experiment3a.py`: `875bb7500033ee9e30cd6315ed3739c5bdb64101337254ca087249736e1693ca`
  - `exp3f_common.sh`: `19fcb8372d593877d2202366e67fe625c707ce4c3b4b0bc6ac24659115526c9b`
  - `run_campaign_exp3f_pilot.sh`: `9be3232811ff38cc4334b15b9125cc085707621fe9e5732c01509f546dd5ccab`
  - `run_campaign_exp3f_eval.sh`: `2f256d48c0f2ca910cfdfd21dfd8474499ce2f9c4f5e7c490776643e50868c3d`
  - `jcm/mcb/planner.py`: `e13f8152928f57cd35eb5862c39855397501bbf335a992b3d3c86a3d8ac2b575`
  - `jcm/mcb/land_climatology.py`: `e04dc41f02bcc060d686fb16e47520e938ce6d617766e4559c85d1e7d8da6f2c`
  - `jcm/mcb/test_world.py`: `0a9fa4f145410d506a6a43731aa21672d3dcdf2b1a4802859488c409823f252e`
  - `jcm/mcb/strength_estimator.py`: `01a6884d78116693b04e61960e7db13d10a9ea7fed9f435b922e708f2a1fb276`
  - `jcm/mcb/hidden_strength.py`: `9d334a656e3f40bd25c89f77c5f98e99cad2a86b2283d854b5b895a78cd80140`
  - `jcm/mcb/scores.py`: `4ff68a872ec6ea54813918986951d73b86ac79f847701f60cb60360c84f58d06`
  - `run_test_world.py`: `87e3739e6aaa1f57aa3a60a9d33266ed96d533fdf4c7976ce72c5bd7a2752961`
  - `mcb_experiments_gpu/exp3f/registered_design.json`: `4e967770908a9aa8c17ff23d371f5b739f2a99110458e23c1ddf470be479cd24`
  - `mcb_experiments_gpu/exp3f/pilot/summary.json`: `9b647471185412d8fd29bf3983693d53c913c8a545f2dde8e152ca0670abf278`
  - `mcb_experiments_gpu/exp3d/registered_design.json`: `7e2782406001ee52f00488d18ce55e81fdb8e9ed4f039e85d28d86fcd0dcb2cd`
  - `mcb_experiments_gpu/exp3d/hidden_strength.json`: `bb00df59d405e00c5ca357b214508d4fe9c5cad42290d8bcfd788301364f8cc8`
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
  frozen in commit fd4740bd.
- 2026-10-26 19:25 PDT: this posting, before any evaluation run of Experiment 3f.

---

**Amendment 9, revision 4 — Experiment 3e RESULT (logged 2026-10-26; evaluation 2026-10-19 15:52 to
2026-10-20 11:27 PDT, after revision 4's posting at 15:41; the registered `analyze_experiment3e.py` was
applied unchanged).**
All 192 runs (4 arms × 48 states, 6 members each, against 3d's 10-member references) finished at the
first attempt: no run failed or was re-run, no state was dropped, and there was no deviation. 3d's runs
of the six reused arms were reused as registered (E4), and the analysis's checks of land model,
sensing, analysis error, learner noise level and member count passed. The 81 frozen code and design
files on the GX10 were re-checked against 79d9494f after the run. The analysis ran on 2026-10-20, and
the runs and its output were committed on 2026-10-23 (80ade71c); this entry was written on
2026-10-26, before revision 5 was frozen. Primary endpoint `J_zonal`, on the means of the 16 macro
states:
* **H1** `plan_learn_a10` vs `pi`: rel −43.5% (90% CI −52.4% to −33.3%), t p 8.4×10⁻⁶, Wilcoxon p
  6.1×10⁻⁵ (15 of 16 macro states favour `plan_learn_a10`) → **better**.
* **H2** `plan_learn_a10` vs `plan_naive_a10`: rel −4.9% (−15.7% to +7.1%), Holm p 0.74 (Wilcoxon
  0.43; 9 of 16) → **inconclusive**.
* **H3** `plan_learn_a10` vs 3d's `plan_learn`: rel −2.1% (−15.5% to +12.5%), Holm p 0.78 (Wilcoxon
  0.90; 8 of 16) → **inconclusive**. The interval reaches beyond ±10%, so equivalence is not shown.
* **H4** strength error on day 98, `plan_learn_a10` vs `plan_learn_a100`: 0.299 against 0.559, rel
  −46.4% (−49.5% to −43.4%), Holm p 1.2×10⁻¹¹ (Wilcoxon 3.1×10⁻⁵; 16 of 16) → **better**.

*Arm means:* `plan_learn_a10` 0.00278, `plan_learn` 0.00284, `plan_naive_a10` 0.00292, `plan_naive`
0.00319, `plan_learn_a30` 0.00338, `plan_learn_a100` 0.00378, `adaptive` 0.00388, `pi` 0.00492,
`fixed` 0.01506, `uncontrolled` 0.01836 (the arms without an `_a` suffix are 3d's runs).

*The dose-response* (each learner vs `pi`, with its 90% CI, then its strength error on day 98):
* f = 0 (3d's `plan_learn`): −42.3% (−51.4% to −30.4%); 0.075;
* f = 0.1: −43.5% (−52.4% to −33.3%); 0.299;
* f = 0.3: −31.3% (−40.3% to −20.9%); 0.499;
* f = 1: −23.2% (−34.8% to −8.9%); 0.559.

All four 95% intervals lie below zero.

*Descriptive* (unadjusted p):
* `plan_naive_a10` vs `pi`: −40.6% (−49.4% to −29.9%, t p 4.8×10⁻⁵).
* `plan_learn_a10` vs `adaptive`: −28.3% (−38.9% to −14.7%, t p 0.0026).
* `plan_naive_a10` vs 3d's `plan_naive`: −8.3% (−19.5% to +4.4%, t p 0.21).
* `plan_learn_a30` vs `plan_learn_a10`: +21.5% (+5.1% to +39.6%, t p 0.0044); `plan_learn_a100` vs
  `plan_learn_a10`: +35.9% (+17.0% to +58.0%, t p 5.8×10⁻⁴).
* *3c in the fixed-land model:* `plan_learn_a100` vs `pi` is −23.2% (95% CI −36.6% to −5.9%), against
  3c's −15.0%. They point the same way, and 3c's estimate lies inside 3e's 95% interval. The strength
  error on day 98 is 0.559, against 3c's 0.568.
* By overall factor 0.5 / 1 / 2: `plan_learn_a10` vs `pi` is −36% / −18% / −58%; vs `plan_naive_a10`
  +1% / +3% / −14%; vs 3d's `plan_learn` +1% / −6% / −1%.
* Learning curves (mean |log(estimate/truth)| per band at days 0, 28, 56, 98 and 168):
  * f = 0.1: 0.59, 0.57, 0.44, 0.30, 0.18;
  * f = 0.3: 0.59, 0.58, 0.56, 0.50, 0.42;
  * f = 1: 0.59, 0.58, 0.58, 0.56, 0.51;
  * 3d's exact sensing: 0.59, 0.35, 0.15, 0.08, 0.04.
* *The forecast errors' size* on the evaluation states, against the registered noise levels: f = 0.1
  0.00586 K (`plan_naive_a10` 0.00584; registered 0.00569), f = 0.3 0.02025 K (0.02028), f = 1
  0.04565 K (0.04703).
* *Detection* (the share of (state, member) pairs with Z ≥ 1.645, on days 28 / 56 / 98 / 168):
  * f = 0.1: 93% / 100% / 100% / 100% (`plan_naive_a10`: 91% / 100% / 100% / 100%);
  * f = 0.3: 75% / 96% / 99.7% / 100%;
  * f = 1: 48% / 83% / 93% / 98%. On day 28 it is 34% / 35% / 75% at overall factor 0.5 / 1 / 2.
* Gains G (no brightening over controlled; > 1 is better) of `plan_learn_a10`: land temperature 1.07,
  land precipitation 0.96, global precipitation 1.05 (`pi`: 1.02, 0.96, 0.94).

The full output is `mcb_experiments_gpu/exp3e/exp3e_analysis.json` (SHA-256 73d7b4d6…daecf0).

*By E9:*
* H1 supports "the learning planner's advantage over classical feedback holds when it knows the
  weather only through a realistic analysis".
* H2 is not better and H4 is better, so the registered reading is revision 2's: "the strength is
  learned but the planner's state feedback absorbs most of its effect". As in 3d, the verdict is
  inconclusive, not equivalent: the interval does not exclude a modest benefit of learning.
* H3 has no registered reading for an inconclusive verdict, so it is reported as found: no cost of
  the realistic analysis against exact knowledge of the weather was detected (−2.1%), but the 90%
  interval (−15.5% to +12.5%) is wider than the ±10% that equivalence needs.
* H4 supports "a realistic analysis restores learning: knowing the weather even with a 10% error lets
  one Earth reveal the strength within the first 98 days, which 3c could not". By day 98 the error is
  about half of f = 1's, and four times exact sensing's.
* *The dose-response and detection:* the advantage over `pi` is kept at f = 0.1, shrinks at f = 0.3,
  and is smallest, but still present, at f = 1. Learning is fastest with exact knowledge of the
  weather, slower at f = 0.1, slow at f = 0.3 and nearly absent at f = 1. How soon one Earth would
  tell that brightening works falls the same way: by day 28, in 93%, 75% and 48% of (state, member)
  pairs.

### Amendment 9, revision 5 — Experiment 3f (a months-long look-ahead for the learning planner) frozen in full (frozen 2026-10-26, BEFORE any evaluation run of Experiment 3f)

**What existed when this was frozen.**
* *The results of Experiments 3a–3e*, logged above. In particular:
  * 3a (strength known): the 60-day planner scored like the 14-day one, and the 120-day planner was
    47% worse (inconclusive) and overcooled by 0.031 K;
  * 3b, 3d and 3e: the learning planner, which plans two weeks ahead, beat the classical controller by
    38%, 42% and 43.5%.
* *The block-schedule planner* (`jcm.mcb.planner`, `schedule = "hold" | "linear"`;
  `jcm.mcb.test_world.make_schedule_sst_fn`) and the planner's target extension
  (`run_test_world.py plan --target-extension`), with tests (`jcm/mcb/planner_schedule_test.py`):
  * with `schedule = "none"`, the default, the planner's code path and results are unchanged, and
    the existing planner tests pass;
  * `hold` plans exactly as the original planner;
  * one Gauss–Newton step reaches the toy model's optimum;
  * a planner that plans 3 blocks ahead learns a hidden strength from its first segment, as 3d's
    planner does.
* *Experiment 3f's pilot* (F9; 2026-10-23), on four of 3d's branch-1 training states only, and its
  normal-only target extensions for the 16 branch-1 training states.
* *Smoke tests,* each on one training state with one member and two segments:
  * of the pilot script (2026-10-23). It also built the pilot's target extensions;
  * of the evaluation script (2026-10-26). It ran both evaluation arms' exact commands next to 3d's
    smoke runs of the reused arms, then the registered analysis in its smoke mode. The first attempt
    ran both arms, but its analysis stopped: in smoke mode the script passed it only branch 1's
    training references, and 3d's smoke runs are on branch 0. That smoke-only setting was corrected
    (it has no counterpart outside smoke mode), and the repeat passed.

  Their scores are not used.
* *A memory probe* (2026-10-26): one `learn112_mpc` run on the same training state, with one member
  and two segments, to choose how many runs go at once. Its score is not used.
* *No evaluation run of Experiment 3f exists,* and no target extension of an evaluation state.

**Scope.** Experiment 3f is frozen in full here. It uses revision 2's endpoint, tests and verdicts
through the same code (`analyze_experiment3b.analyze`, `analyze_experiment3a.verdict`), in
`analyze_experiment3f.py`.

**F1. Question.** Dubey et al. (2026) plan two weeks ahead, and suggest model predictive control,
which "optimizes several segments ahead and commits only the first", as an extension. In 3a, with
the strength known, looking further ahead did not help. 3b–3e's winning planner looks two weeks
ahead. *When the strength is hidden and must be learned, does planning 16 (or 8) weeks ahead, with
the 30-day atmosphere-truncated gradient, beat planning two weeks ahead?* This is the one cell of the
design (strength known or hidden × two weeks or months) that no experiment has tested.

**F2. The planners.** Every arm is 3d's learning planner, exactly, except for what it plans over:
* *The same in every arm:*
  * Gauss–Newton, 1 step, 1 copy, the zonal objective, 3d's penalties (μ = 6.325×10⁻⁴,
    λ = 6.325×10⁻³);
  * exact sensing;
  * the per-band learner (`--learn-strength bands`, prior sd 1, bounds [0.2, 5]) at 3d's registered
    noise level, 0.00069104 K;
  * the learner's input: the two-week forecast of the segment the planner is about to apply,
    compared with the observed segment-mean latitude profile, exactly as in 3d. In the long arms this
    forecast is the first block of the look-ahead, which no truncation touches (the first cut is at
    day 30).
* *The arms:*
  * `learn14`: a 14-day look-ahead with ordinary backpropagation. This is 3d's `plan_learn`, run
    again alongside the new arms, so the primary comparison is between runs made with the same
    software and GPU at the same time.
  * `learn112_mpc`: 112 days (16 weeks; eight blocks of 14 days) as model predictive control. Block
    j holds `clip(level + trend × j, 0, cap)` per band. Every block's zonal error is scored with
    weight 1/8. The amplitude penalty is the mean of μ|a_j|² over the blocks, and the movement penalty
    is λ|level − previous|² plus λ(7/8)|trend|². Only `level` is applied, and the next re-plan
    warm-starts from it and from the planned trend. The trend is bounded to ±cap/7. The
    atmosphere's derivatives are cut every 30 days (the window that revision 1.1 registered for
    future experiments).
  * `learn56_mpc`: the same over 56 days (four blocks), with the trend bounded to ±cap/3.
  * `learn112_hold`: 112 days with one setting held for the whole look-ahead and scored on its
    look-ahead mean. This is 3a's long-planner objective, here with the 30-day cut, so that it
    differs from `learn112_mpc` only in the schedule.
* *What the long arms forecast beyond day 182:* the episode ends on day 182, but a re-plan on day
  168 looks to day 280. The forecast simply continues the registered warming ramp.

**F3. States, strengths, references and members.** All are 3d's, exactly:
* the fixed land model;
* the 48 `exp3d_eval` states;
* `exp3d/hidden_strength.json`;
* the 10-member evaluation references (scoring, days 98–182);
* 6 members with the same seeds;
* 13 segments of 14 days.

**F4. The long arms' target.** 3d's references cover 200 days. Before the runs, the evaluation script
builds a normal-climate-only reference for every evaluation state to day 280 (`run_test_world.py
references --no-warmed --days 280`), with 3d's 10 member seeds. The planner's daily target is the
scoring references' normal climate on days 0–199, then this file's on days 200–279. Scoring never
uses the extension. Every run records the mean absolute difference of the two files' SST over their
last 14 shared days. The analysis refuses a run where it exceeds 0.05 K.

**F5. Reused arms.** 3d's runs of `uncontrolled`, `pi`, `adaptive`, `fixed`, `plan_learn` and
`plan_naive` are reused as they are: none of them changes in 3f.

**F6. Hypotheses** (`J_zonal`, macro-state means; 3b's paired t-test, Wilcoxon check, hierarchical
bootstrap and verdict grid: better / worse need p < 0.05 and |rel| ≥ 5%; equivalent needs the 90%
interval inside ±10%):
* **H1 (primary, alone at α = 0.05):** `learn112_mpc` vs `learn14`.
* **Secondary family (Holm over the two):**
  * H2: `learn56_mpc` vs `learn14`;
  * H3: `learn112_mpc` vs `learn112_hold` ("does planning a schedule fix a long look-ahead?").
* **Descriptive:**
  * every arm against `pi`;
  * `learn14` against 3d's `plan_learn`, the same planner run again (a check of drift between
    software, GPU and dates);
  * `learn112_mpc` vs `learn56_mpc`, and `learn112_hold` vs `learn14`;
  * each arm's mean bias (K), effort, cap share, and the learner's strength error on day 98;
  * compute: median seconds per re-plan and per run, and their ratio to `learn14`. Runs share the
    GPU, so these are costs under contention, and they depend on how many runs share it. Before
    this freeze, with at most one other run on the GPU (the evaluation smoke and the memory probe),
    a re-plan took 7.5 s for `learn14` and 47 s for `learn112_mpc` (6.3×). In the pilot, four runs
    at a time, it took 13.8 s and 205 s (14.8×). Both are reported with the result.

**F7. Predictions** (written before any run):
* *H1:* inconclusive or worse is more likely than better, because 3a found no gain from a longer
  look-ahead with the strength known.
  * Two things could make it better here. The learner corrects the believed strength every two
    weeks. And a schedule, unlike 3a's held setting, does not have to overcool now to cancel later
    warming.
  * Against it, the 30-day truncated gradient still undercounts the response at 16 weeks (its size
    ratio was 0.80 at 120 days in the snip test). The plan for later blocks therefore over-brightens,
    which pulls the applied level up.
* *Bias:* the long arms overcool more than `learn14` (a more negative mean bias), and
  `learn112_hold` overcools most.
* *H3:* better (MPC beats holding one setting), for the reason above.
* *H2:* inconclusive.
* *Compute:* `learn112_mpc` costs roughly 10–15× `learn14` per re-plan (10 forward-mode tangents over
  112 days, against 5 over 14).

**F8. Power, and how an inconclusive H1 is read.** 3d's and 3e's runs give a standard error of about
6.5–7.5% of `J_zonal` for a paired comparison of two planners over 16 macro states. So the smallest
effect this design detects with 80% power is about 20%, and an equivalence verdict (±10%) is out of
reach. An inconclusive H1 is reported with its 90% interval as the bound on how much a 16-week
look-ahead can help or hurt. Members do not tighten it; only more macro states would.

**F9. Pilot and budget rule** (run before this freeze; `run_campaign_exp3f_pilot.sh`,
`analyze_exp3f_pilot.py`).
* *What it ran* (2026-10-23, 10:43–15:44 PDT, four runs at a time): every arm on four branch-1
  training states (macro states 100, 108, 116 and 124), with 3 members, 13 segments and 3d's training
  references. The normal-only target extensions had been built first (01:05–01:30 PDT, by the pilot
  script's smoke run).
* *Feasibility:* every run finishes with finite scores, and every target extension is within 0.05 K
  of its references. *Result:* all 16 runs finished with finite scores, and the largest target-extension difference was 0.0000 K: over their shared days, the extensions repeat the references exactly.
* *The budget rule:* project the evaluation's wall-clock time from the pilot's median run time,
  scaled to 6 members and 48 states, at the pilot's measured speed-up from parallel runs. Then drop
  arms from the end of the list `learn14`, `learn112_mpc`, `learn56_mpc`, `learn112_hold` until the
  projection is at most 60 hours. `learn14` and `learn112_mpc` always stay. A dropped arm's
  hypothesis is not tested.
  * *Result:* measured speed-up 3.91 (the pilot's summed run time over its wall time, four runs at a time); projected evaluation `learn14` 5.3 h, `learn112_mpc` 55.9 h, `learn56_mpc` 28.5 h, `learn112_hold` 31.2 h; total for the kept arms 61.2 h (budget 60 h). Dropped: `learn56_mpc` and `learn112_hold`. So H2 and H3 are not tested, and the descriptive comparisons with a dropped arm are not made; H1 is the only hypothesis.
  * *Evaluation arms:* `learn14`, `learn112_mpc`.
* *No setting is chosen from the pilot's scores.* They are logged here for feasibility only:
  `learn14` J 0.00527, bias −0.0032 K, cap share 0.07, 13.8 s per re-plan; `learn112_mpc` J 0.00458, bias −0.0129 K, cap share 0.03, 204.9 s per re-plan; `learn56_mpc` J 0.00502, bias −0.0139 K, cap share 0.07, 102.7 s per re-plan; `learn112_hold` J 0.00568, bias −0.0200 K, cap share 0.04, 116.4 s per re-plan.

**F10. Conduct.**
* Runs go in the order of the arm list above, all states of one arm before the next.
* A failed run is re-run once (`run_campaign_exp3f_eval.sh` counts attempts). If the GPU disappears,
  the script stops, and it is restarted by hand once the GPU is back; it skips finished runs. A run
  that fails twice is logged, and the analysis uses the states that every arm completed.
* GPU runs are not bit-reproducible run to run (logged with 3d), so `learn14` against 3d's
  `plan_learn` measures that spread; it is descriptive.
* The registered analysis is applied unchanged. Anything else is a logged deviation.

**F11. What each outcome will mean** (written now, for the paper):
* *H1 better:* "With the strength hidden and learned, planning 16 weeks ahead with
  atmosphere-truncated gradients improved on two-week receding-horizon planning by X%, at Y× the
  compute." The months-long gradient then earns a place in the headline.
* *H1 inconclusive:* "Extending the look-ahead from 2 to 16 weeks changed the score by between a% and
  b% (90% interval). Two-week planning, as in Dubey et al., is enough for this problem, at 1/Y of the
  compute." The months-long gradient stays a validated tool (Experiments 1 and 2), not a source of
  control skill here.
* *H1 worse:* "A 16-week look-ahead was X% worse, even as model predictive control." Its bias tells
  whether the truncated gradient's undercount is the reason.
* *H3,* whatever H1 says, tells whether 3a's long-planner failure came from holding one setting
  (H3 better) or from the look-ahead itself (H3 not better).

#### Common rules

* *Order:* after this posting, the GX10 builds the 48 evaluation states' target extensions (one
  process per branch), then runs the evaluation arms in F9's order, all 48 states of one arm before
  the next, four runs at a time (`run_campaign_exp3f_eval.sh`).
  * Four at a time is the pilot's setting, whose speed-up the budget rule used.
  * The memory probe found about 3.7 GB per `learn112_mpc` run, with about 41 GB of the GX10's memory
    free.
  * More runs at a time would not be faster. Four `learn112_mpc` runs at once made each re-plan
    about 4.4 times slower than one alone, so the GPU is the limit.
  * The script is resumable.
* *Frozen code:*
  * the analyses: `analyze_experiment3f.py`, `analyze_exp3f_pilot.py`, `analyze_experiment3e.py`,
    `analyze_exp3e_pilot.py`, `analyze_experiment3b.py` and `analyze_experiment3a.py`;
  * the campaign scripts: `exp3f_common.sh`, `run_campaign_exp3f_pilot.sh` and
    `run_campaign_exp3f_eval.sh`;
  * the model: `jcm/mcb/planner.py`, `jcm/mcb/land_climatology.py`, `jcm/mcb/test_world.py`,
    `jcm/mcb/strength_estimator.py`, `jcm/mcb/hidden_strength.py` and `jcm/mcb/scores.py`;
  * the driver: `run_test_world.py`;
  * the design: `exp3f/registered_design.json` (SHA-256 4e967770908a9aa8c17ff23d371f5b739f2a99110458e23c1ddf470be479cd24), with the pilot's summary
    (`exp3f/pilot/summary.json`). 3d's design, strength table, states and references are revision
    3's.
* *Cost:* about 62 hours of wall time on the GX10: the budget rule's projection for the evaluation arms (61.2 h, 96 runs of 6 members), plus about an hour for the target extensions.
* *Posting:* this revision is posted to OSF (project pqabf) before any evaluation run of 3f, and
  before any target extension of an evaluation state.
