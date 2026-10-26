### Amendment 9, revision 5 — Experiment 3f (a months-long look-ahead for the learning planner) frozen in full (frozen FREEZE_DATE, BEFORE any evaluation run of Experiment 3f)

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
  * of the evaluation script (SMOKE_DATE). It ran both evaluation arms' exact commands next to 3d's
    smoke runs of the reused arms, then the registered analysis in its smoke mode. The first attempt
    ran both arms, but its analysis stopped: in smoke mode the script passed it only branch 1's
    training references, and 3d's smoke runs are on branch 0. That smoke-only setting was corrected
    (it has no counterpart outside smoke mode), and the repeat passed.

  Their scores are not used.
* *A memory probe* (SMOKE_DATE): one `learn112_mpc` run on the same training state, with one member
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
  of its references. *Result:* PILOT_FEASIBILITY.
* *The budget rule:* project the evaluation's wall-clock time from the pilot's median run time,
  scaled to 6 members and 48 states, at the pilot's measured speed-up from parallel runs. Then drop
  arms from the end of the list `learn14`, `learn112_mpc`, `learn56_mpc`, `learn112_hold` until the
  projection is at most 60 hours. `learn14` and `learn112_mpc` always stay. A dropped arm's
  hypothesis is not tested.
  * *Result:* PILOT_BUDGET.
  * *Evaluation arms:* PILOT_ARMS.
* *No setting is chosen from the pilot's scores.* They are logged here for feasibility only:
  PILOT_SCORES.

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
  * the design: `exp3f/registered_design.json` (SHA-256 DESIGN3F_SHA), with the pilot's summary
    (`exp3f/pilot/summary.json`). 3d's design, strength table, states and references are revision
    3's.
* *Cost:* COST.
* *Posting:* this revision is posted to OSF (project pqabf) before any evaluation run of 3f, and
  before any target extension of an evaluation state.
