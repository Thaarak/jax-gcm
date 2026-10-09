
### Amendment 9, revision 3 — Experiments 3c (the planner cannot see the weather) and 3d (the fixed land model) frozen in full (frozen FREEZE_DATE, BEFORE any evaluation run of Experiment 3c and any evaluation reference, run or score of Experiment 3d)

**What existed when this was frozen.**
* *Experiment 3b's result* (revision 2, logged above): a planner that learns the strength beat
  the classical controller (H1 better, −38%); learning helped (H2 better); feedback beat the fixed
  design (H3 better).
* *Experiment 2's and the snip test's results* (revision 1.1, logged above).
* *Experiment 3c* has no evaluation run. Its pilot ran on 3b's pilot states only (`exp3b_train`
  branch 1); the pilot's other arms are 3b's pilot runs, reused.
* *A real-model smoke test* (2026-10-09, one training state, one member):
  * an oracle planner that sees the exact state misses its two-week forecast of the ocean
    latitude profile by about 0.0006–0.0009 K rms;
  * one that sees only the ocean misses by about 0.04–0.06 K.
* *Experiment 3d* uses no evaluation data yet. The following exist, all on training states:
  * the land fix (commit 5f5abc13);
  * its base climate (3 below);
  * the 32 new macro states and their branches;
  * the hidden strengths;
  * the training side;
  * the oracle pilot.

  Evaluation references for `exp3d_eval` did not exist.

**Scope.**
* *Frozen in full here:*
  * Experiment 3c (Part A), the realistic-information version of 3b;
  * Experiment 3d (Part B), 3b's tests repeated with the land-model defect fixed.

  They are MCB_PROJECT_REPORT.md Part 28.
* *Shared rules:* both use revision 2's endpoint, tests and verdicts, through the same code.
  `analyze_experiment3b.analyze` now takes the hypotheses as arguments. With revision 2's defaults
  it reproduces 3b's logged output exactly (checked on the GX10 on 2026-10-09).

#### Part A — Experiment 3c: the planner sees the ocean, not the weather

**A1. Question.** 3b's planners forecast from the exact current state, weather included (revision
2, §11). 3c asks whether 3b's result survives when the planners cannot see the weather.

**A2. What the planners know** (`run_test_world.py plan --sensing ocean`; `jcm.mcb.planner`).
* *What they see:* the true ocean, the slab state, observed exactly, as every controller's
  observation was in 3b.
* *What they do not see:* the atmosphere and the land. Each forecast starts from the planner's
  background:
  * after day 0, the atmosphere and land of its own previous 14-day forecast;
  * on day 0, those of the next weather branch of the same macro state at the same date:
    branch (b + 1) mod 3 for evaluation states, and (b + 1) mod 2 for training states.
* *Everything else is 3b's planner:*
  * the 14-day Gauss–Newton step (1 step, 1 copy);
  * the zonal objective and the same penalties;
  * the same comparison of the observed segment-mean latitude profile with its forecast.
* *The classical arms* (PI, the adaptive law), the fixed design and no brightening never used the
  weather. What they know is unchanged.

**A3. States, strengths, references and members.** All are 3b's, exactly:
* the 48 `exp3b_eval` states;
* `exp3b/hidden_strength.json`;
* the 10-member evaluation references;
* 6 members with the same seeds;
* 13 segments of 14 days, scored on days 98–182.

**A4. Arms.**
* *New (run here):*
  * `plan_learn_rs`: the learning planner. It learns each band's strength with prior sd 1 and
    bounds [0.2, 5]. Its noise level, NOISE3C K, comes from 3b's rule (`analyze_exp3b_pilot.py
    noise`) applied to the ocean-only oracle in the pilot.
  * `plan_naive_rs`: nominal strength.
* *Reused unchanged from 3b's evaluation (`exp3b/eval`):*
  * `pi`, `adaptive`, `fixed` and `uncontrolled`;
  * for comparison, 3b's exact-sensing `plan_learn`, `plan_naive` and `plan_oracle`.

  Reuse is valid because these runs depend only on the state, members, strength and controller,
  none of which changes. The analysis checks every run's sensing and member count.

**A5. Endpoints.** As revision 2 (§5).

**A6. Hypotheses.**
* **H1 (primary, alone at alpha = 0.05):** `plan_learn_rs` vs `pi`. Does the learning planner
  still beat the classical controller when it cannot see the weather?
* **Secondary family (Holm over two):**
  * **H2:** `plan_learn_rs` vs `plan_naive_rs`. Does learning still help?
  * **H3:** `plan_learn_rs` vs 3b's `plan_learn`. What does not seeing the weather cost?
* *Descriptive (no test claims):*
  * `plan_naive_rs` vs `pi`, and `plan_learn_rs` vs `adaptive`;
  * `plan_naive_rs` vs `plan_naive`;
  * `plan_learn_rs` and `plan_naive_rs` vs `plan_oracle`;
  * both learners' learning curves;
  * everything by overall factor.

**A7. Tests.** `analyze_experiment3c.py`: revision 2's tests and verdict grid, unchanged.

**A8. Pilot and power.** PILOT3C_SUMMARY

**A9. Interpretation, fixed in advance.**
* *H1 better:* "the learning planner's advantage over classical feedback survives when it cannot
  see the weather".
* *H1 equivalent or negligible:* "without the weather, a planner on one forecast is only as good as
  classical feedback". 3b's advantage then needs knowledge of the weather, or a forecast ensemble.
* *H1 worse:* "without the weather, planning on a single forecast loses to classical feedback".
* *H1 inconclusive:* the advantage is not established under realistic information. The interval
  is reported.
* *H2 better:* learning still helps without the weather.
* *H2 not better:* two-week forecast misses dominated by weather do not reveal the strength within
  one episode. This is the one-Earth detection limit of Part 27, seen from inside a controller.
* *H3:* the cost of not seeing the weather. It is expected to be worse; its size is reported.

**A10. Failures.** As revision 2 (§10).

#### Part B — Experiment 3d: 3b's tests with the land model fixed

**B1. Why.** Every experiment up to revision 2 ran with jax-esm's slab land reading its 12 monthly
climatology entries one per day, so the land runs a year every 12 days (MCB_PROJECT_REPORT.md Part
26). The scores use only ocean temperatures and every arm shares the defect, but it shapes the
atmosphere. 3d repeats 3b's tests in the model with the land fixed.

**B2. The fix.**
* *What it is:* `jcm.mcb.land_climatology.MonthlyClimatologySlabLandModel` (commit 5f5abc13),
  selected by `JCM_LAND_CLIMATOLOGY=monthly`. The 12 entries are read as months, with mid-month
  linear interpolation (as the Q-flux). Everything else is jax-esm's land step.
* *Safeguards:* states and references record the land model they were made with, and the runners
  refuse the other one. `analyze_experiment3d.py` refuses any run not made with the fixed land.

**B3. Base climate** (revisions 0.2–0.3: procedure and gate unchanged). BASE_CLIMATE

**B4. States and strengths.**
* *States:* `run_generate_macro_ics.py --plan exp3d` makes 32 macro states (100–131), continued from
  the settled carry every 365 days. Even states train (`exp3d_train`, branches 0–1); odd states
  evaluate (`exp3d_eval`, branches 0–2). Diagnostics: GENERATION_DIAGNOSTICS_3D.
* *Hidden strengths:* 3b's rule and distribution (`jcm.mcb.hidden_strength`). Table:
  `exp3d/hidden_strength.json`, SHA-256 TABLE3D_SHA.

**B5. Training side** (3b's procedure, on the fixed-land training states).
* references with 5 members;
* one-band response runs on the 16 branch-0 states, which give PI's and the adaptive law's
  sensitivities;
* the fixed design (ladder rung 4, zonal, days 98–182): FIXED3D_AMPLITUDES.

**B6. Arms.** 3b's, without the oracle:
* `plan_learn`, with the noise level NOISE3D K from 3b's rule applied to the oracle pilot on the 16
  branch-1 training states (3 members);
* `plan_naive`;
* `pi` (a 42-day loop with feedforward);
* `adaptive`;
* `fixed`;
* `uncontrolled`.

Evaluation references have 10 members (seeds from 93000). Runs have 6 members: this is 3b's design
repeated, so there is no new power rule.

**B7. Hypotheses.**
* *Tests:* 3b's H1–H3 unchanged. H1 (`plan_learn` vs `pi`) stands alone at alpha = 0.05. H2
  (`plan_learn` vs `plan_naive`) and H3 (`pi` vs `fixed`) are Holm over two.
* *Descriptive:*
  * `plan_naive` vs `pi`, `adaptive` vs `pi`, and `plan_learn` vs `adaptive`;
  * the learning curve;
  * everything by overall factor;
  * *the replication:* for each hypothesis, 3b's and 3d's relative differences, whether they point
    the same way, and whether 3b's estimate lies inside 3d's 95% interval.

**B8. Tests.** `analyze_experiment3d.py`: revision 2's tests and verdict grid, unchanged.

**B9. Interpretation, fixed in advance.**
* *H1 better:* "3b's main result does not depend on the land-model defect".
* *H1 not better:* "3b's main result does not replicate with the land fixed". This is described
  from the descriptive endpoints.
* *H2 and H3:* as in revision 2.

**B10. Failures.** As revision 2 (§10).

#### Common to both

* *Order:* after this posting, the GX10 runs 3c's evaluation (2 arms × 48 states) and 3d's (the
  references, then 6 arms × 48 states) side by side, 3 runs at a time each. Both are resumable.
* *Idealizations:*
  * 3c keeps a perfect model and an exactly known ocean;
  * 3d keeps revision 2's idealizations (§11);
  * neither drifts the strength over time.
* *Frozen code:*
  * the analyses: `analyze_experiment3c.py`, `analyze_experiment3d.py`, `analyze_experiment3b.py`,
    `analyze_experiment3a.py`, `analyze_exp3b_pilot.py` and `analyze_exp3c_pilot.py`;
  * the campaign scripts: `run_campaign_exp3c_pilot.sh`, `run_campaign_exp3c_eval.sh`,
    `run_campaign_exp3d_prep.sh` and `run_campaign_exp3d_eval.sh`;
  * the model: `jcm/mcb/planner.py`, `jcm/mcb/land_climatology.py`, `jcm/mcb/test_world.py`,
    `jcm/mcb/strength_estimator.py`, `jcm/mcb/hidden_strength.py`, `jcm/mcb/feedback.py`,
    `jcm/mcb/scores.py` and `jcm/mcb/qflux.py`;
  * the drivers: `run_coupled_training.py`, `run_test_world.py`, `run_controllers.py`,
    `run_generate_macro_ics.py` and `run_qflux_base_climate.py`;
  * the designs: `exp3c/registered_design.json` (SHA-256 DESIGN3C_SHA) and
    `exp3d/registered_design.json` (SHA-256 DESIGN3D_SHA), with their pilot records.
* *Cost:* about COST.
* *Posting:* this revision is posted to OSF (project pqabf) before any evaluation run of 3c and
  any evaluation reference of 3d.
