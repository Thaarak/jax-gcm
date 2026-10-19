
### Amendment 9, revision 4 — Experiment 3e (the planner corrects its forecast toward the true weather) frozen in full (frozen FREEZE_DATE, BEFORE any evaluation run of Experiment 3e)

**What existed when this was frozen.**
* *The results of Experiments 3c and 3d* (revision 3, logged above):
  * 3c: without the weather, the learning planner still beat the classical controller (H1 better,
    −15.0%), but it could not learn the strength (H2 inconclusive; its strength error on day 98 was
    0.57, against 0.08 when it saw the weather);
  * 3d: with the land fixed, 3b's result replicated (H1 better, −42.3%).
* *Analysis sensing in the planner* (commit 8f4af5cd), with tests. At analysis error 0 it is exactly
  3d's planner, and at 1 exactly 3c's. Each planner run now also saves every forecast miss
  (`<run>.innovations.npz`).
* *Read-only diagnostics* (2026-10-18; MCB_PROJECT_REPORT.md Part 29.1). They used existing
  no-brightening reference ensembles, including those of 3b's and 3d's evaluation states, and 3d's
  training response runs. They estimated how a two-week forecast's error grows with the error of its
  starting weather. They contain no planner run, no score and nothing of 3e's evaluation.
* *Experiment 3e's pilot* (commits 88be5787 and c6b9e76f; E8), on 3d's 16 branch-1 training states
  only.
* *Smoke tests,* each on one training state with one member and two segments:
  * of the pilot script (2026-10-19);
  * of the evaluation script (SMOKE_DATE). It ran every new arm's exact command next to 3d's smoke
    runs of the reused arms, then the registered analysis in its smoke mode.

  Their scores are not used.
* *No evaluation run of Experiment 3e exists.* 3d's evaluation runs, which 3e reuses, and their
  results are logged above.

**Scope.** Experiment 3e is frozen in full here (MCB_PROJECT_REPORT.md Part 29). It uses revision 2's
endpoint, tests and verdicts, through the same code (`analyze_experiment3b.analyze`), and adds one
endpoint, for H4.

**E1. Question.** 3b's and 3d's planners forecast from the exact state, weather included. 3c's saw
only the ocean, and could not learn the strength. Between those ends, how well must the planner know
the weather to beat the classical controller, and to learn the strength?

**E2. What the planners know** (`run_test_world.py plan --sensing analysis --analysis-error f`;
`jcm.mcb.planner.analysis_carry`).
* *The ocean:* the true slab state, observed exactly, as in 3b–3d.
* *The atmosphere and land:* before each forecast the planner starts from an analysis, x_a = (1 − f)
  × truth + f × background, for every state array of the atmosphere and the land. Values the truth
  and the background share exactly, and integer counters, are the truth's.
  * *Background:* after day 0, where the planner's own previous 14-day forecast ended. On day 0, the
    next weather branch of the same macro state at the same date (branch (b + 1) mod 3), as in 3c.
  * *The ends:* f = 0 is 3d's planner (the exact state), and f = 1 is 3c's (the ocean only).
  * *Its errors:* the analysis keeps the share f of the background's error, so its errors are the
    model's own forecast errors.
* *Everything else is 3d's planner:*
  * the 14-day Gauss–Newton step (1 step, 1 copy);
  * the zonal objective and the same penalties;
  * the same comparison of the observed segment-mean latitude profile with its forecast.
* *The classical arms* (PI, the adaptive law), the fixed design and no brightening never used the
  weather. What they know is unchanged.

**E3. States, strengths, references and members.** All are 3d's, exactly:
* the fixed land model (`JCM_LAND_CLIMATOLOGY=monthly`);
* the 48 `exp3d_eval` states (macro states 101–131, 3 branches each);
* `exp3d/hidden_strength.json`;
* the 10-member evaluation references built by 3d;
* 6 members with the same seeds;
* 13 segments of 14 days, scored on days 98–182.

**E4. Arms.**
* *New (run here).* Each learning arm uses its level's noise level, from 3b's rule applied to that
  level's oracle pilot (E8). The learner is 3b's: each band's strength, with prior sd 1 and bounds
  [0.2, 5].
  * `plan_learn_a10`: the learning planner at f = 0.1, the primary level (noise level NOISE_A10 K);
  * `plan_naive_a10`: the planner that assumes nominal strength, at f = 0.1;
  * `plan_learn_a30`: the learning planner at f = 0.3 (NOISE_A30 K);
  * `plan_learn_a100`: the learning planner at f = 1, which is 3c's information in the fixed-land
    model (NOISE_A100 K).
* *Reused unchanged from 3d's evaluation (`exp3d/eval`):*
  * `pi`, `adaptive`, `fixed` and `uncontrolled`;
  * 3d's `plan_learn` and `plan_naive`, the f = 0 end.

  Reuse is valid because these runs depend only on the state, members, strength and controller, none
  of which changes.
  * *Checks:* the analysis checks every run's land model, sensing, analysis error, learner noise level
    and member count.
  * *Code:* the reused runs were made with revision 3's code. The planner's exact-sensing path has not
    changed since, and the tests check it.
  * *The system update:* 3d's planners ran before the GX10's system update of 2026-10-12, and 3e's
    runs come after it. 3d's conduct note found the update's effect indistinguishable from ordinary
    run-to-run variation, which is weather-sized and unbiased. GX10 runs are reproducible
    statistically, not bit for bit.

**E5. Endpoints.**
* *Primary:* `J_zonal`, as revision 2 (§5).
* *H4's endpoint:* the learner's strength error on day 98, the first scored day.
  * It is |log(estimate / truth)| for the estimate the planner plans with on that day, averaged over
    the 5 bands and then over the members.
  * This is the learning curve's value on that day.
* *Secondary and descriptive:* as revision 2 (§5), and E6's.

**E6. Hypotheses.**
* **H1 (primary, alone at alpha = 0.05):** `plan_learn_a10` vs `pi`. Does the learning planner beat
  the classical controller when it knows the weather only through a realistic analysis?
* **Secondary family (Holm over three):**
  * **H2:** `plan_learn_a10` vs `plan_naive_a10`. Does learning the strength help?
  * **H3:** `plan_learn_a10` vs 3d's `plan_learn`. What does a realistic analysis cost, against
    knowing the weather exactly?
  * **H4:** `plan_learn_a10` vs `plan_learn_a100`, on the strength error on day 98. Does a realistic
    analysis let the planner learn the strength?
* *Descriptive (no test claims):*
  * *the dose-response:* each learner (f = 0, 0.1, 0.3 and 1) against `pi`, with its strength error
    on day 98 and its learning curve;
  * `plan_naive_a10` vs `pi` and vs 3d's `plan_naive`; `plan_learn_a10` vs `adaptive`;
    `plan_learn_a30` and `plan_learn_a100` vs `plan_learn_a10`;
  * *3c in the fixed-land model:* `plan_learn_a100` vs `pi`, and its strength error on day 98,
    against 3c's logged H1 (−15.0%) and strength error (0.57). Do they point the same way, and does
    3c's estimate lie inside 3e's 95% interval?
  * *the forecast errors' size* on the evaluation states, per new arm. This is the root of the
    median weighted mean square of each miss, less its part from the strength error,
    G (truth − belief). It is compared with the registered noise level.
  * *detection* (Part 27, now from real forecasts), per new arm:
    * *the statistic:* Z = Σ g'Wz / (κ √(Σ g'Wg)) over each run's re-plans. Here z = miss + G × belief,
      g = G 1 (the sum of the bands' sensitivities), W is the planner's latitude weights, and κ is
      the level's scale from the pilot (E8).
    * *what is reported:* brightening works in every state, so the share of (state, member) pairs
      with Z ≥ 1.645 (one-sided 5%) on days 28, 56, 98 and 168 says how often one Earth would tell
      by then. Each day is read from that day's look alone. The shares are also given by overall
      factor.
  * everything by overall factor.

**E7. Tests.** `analyze_experiment3e.py` applies revision 2's tests and verdict grid, unchanged
(`analyze_experiment3a`'s functions), to all four hypotheses, H4 on its own endpoint. H1 uses its own
p; H2–H4 use Holm-adjusted p.

**E8. Pilot.** PILOT_SUMMARY

*The pilot's rules,* fixed before it ran (`run_campaign_exp3e_pilot.sh`, `analyze_exp3e_pilot.py`):
* each level's noise level is 3b's rule (`analyze_exp3b_pilot.py noise`) on that level's oracle runs;
* its detection scale is κ² = median((g'Wr)² / (g'Wg)) / 0.4549 over the oracle's misses r, whose
  only cause is the forecast's error.

*One repair, made before this freeze.* When the pilot ran, κ came out undefined. In 5, 9 and 45 of
each level's 576 forecasts the planner did not brighten, so g = 0 and the ratio is 0/0. Those
forecasts carry no information on the strength and add nothing to Z, so they are left out (commit
c6b9e76f). The noise levels did not depend on this.

*Power.* There is no power rule. Members are 6 so that 3d's runs of the reused arms stay valid, as
in 3c. For orientation, 3d's H1 had a 90% interval 21 points wide at 6 members (−51.4% to −30.4%).

**E9. Interpretation, fixed in advance.**
* *H1 better:* "the learning planner's advantage over classical feedback holds when it knows the
  weather only through a realistic analysis".
* *H1 equivalent or negligible:* "with a realistic analysis, the learning planner is only as good as
  classical feedback".
* *H1 worse:* "with a realistic analysis, the learning planner loses to classical feedback".
* *H1 inconclusive:* the advantage is not established with a realistic analysis. The interval is
  reported.
* *H2 better:* learning the strength helps with a realistic analysis.
* *H2 not better:* this is read together with H4.
  * If H4 is better, the strength is learned but the planner's state feedback absorbs most of its
    effect (revision 2's reading of H2).
  * If H4 is not better, the misses do not reveal the strength in time.
* *H3 equivalent or negligible:* "a realistic analysis is as good as knowing the weather exactly".
* *H3 worse:* the cost of a realistic analysis. Its size is reported.
* *H3 better:* not expected. It is reported as found.
* *H4 better:* "a realistic analysis restores learning: knowing the weather even with a 10% error lets
  one Earth reveal the strength within the first 98 days, which 3c could not".
* *H4 not better:* even a 10% analysis is not enough to learn the strength within 98 days.
* *The dose-response and detection* are descriptive. They say where, between knowing the weather
  exactly and not at all, the planner's advantage and its learning are lost, and how soon one Earth
  would tell that brightening works.
* Every verdict, arm and descriptive comparison is reported, whatever the outcome.

**E10. Failures.** As revision 2 (§10):
* a failed run is re-run once;
* a state that still fails is dropped from every arm;
* losing more than 6 of 48 states is a deviation to log.

The evaluation script counts each run's attempts, and it makes the automatic re-run only while the GPU
is visible. Otherwise it stops, and the re-run is made by starting it again by hand.

#### Common rules

* *Order:* after this posting the GX10 runs the four new arms in the order `plan_learn_a10`,
  `plan_naive_a10`, `plan_learn_a100`, `plan_learn_a30`, about 5 runs at a time. The script is
  resumable.
* *Idealizations:*
  * the model is perfect except for the strength, and the ocean is known exactly;
  * *the analysis has one parameter.* Its error is the same share of the planner's own forecast
    error in every variable and place. It is not an observation network with its own errors.
    * With a fully decorrelated background, f = 0.1 gives analysis errors of about 2% of the
      weather's natural variance, and f = 0.3 about 18%.
    * Calling f = 0.1 roughly as good as today's analyses of large-scale weather is a judgment,
      not a measurement;
  * the strength is constant in time.
* *Frozen code:*
  * the analyses: `analyze_experiment3e.py`, `analyze_exp3e_pilot.py`, `analyze_experiment3b.py`,
    `analyze_experiment3a.py` and `analyze_exp3b_pilot.py`;
  * the campaign scripts: `run_campaign_exp3e_pilot.sh` and `run_campaign_exp3e_eval.sh`;
  * the model: `jcm/mcb/planner.py`, `jcm/mcb/land_climatology.py`, `jcm/mcb/test_world.py`,
    `jcm/mcb/strength_estimator.py`, `jcm/mcb/hidden_strength.py` and `jcm/mcb/scores.py`;
  * the driver: `run_test_world.py`;
  * the design: `exp3e/registered_design.json` (SHA-256 DESIGN3E_SHA), with its pilot record. 3d's
    design, strength table and references are revision 3's.
* *Cost:* COST.
* *Posting:* this revision is posted to OSF (project pqabf) before any evaluation run of 3e.
