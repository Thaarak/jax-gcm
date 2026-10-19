# Pre-registration: Amendment 9 revision 4 (Experiment 3e), with the results of Experiments 3c and 3d

This file is the OSF posting of revision 4 of Amendment 9 of this project's pre-registration
(`PREREGISTRATION.md`), together with everything logged since revision 3's posting, in file
order:
* the result of Experiment 3c (revision 3, Part A);
* the result of Experiment 3d (revision 3, Part B), with its conduct note on a system update during
  the evaluation and the registered single re-run;
* revision 4 (Experiment 3e).
Everything below the horizontal line is the registered text exactly as frozen in the public
repository; nothing in it has been edited.

## Frozen source

- Repository: https://github.com/Thaarak/jax-gcm (branch `fable-version`)
- Commit: `79d9494f9968fb1ea5a4e8536a029eb9ffb18faf`, committed 2026-10-19 14:34:42 -0700
- SHA-256 of `PREREGISTRATION.md` at that commit: `1a706c2806189f2925052f932c001c9333b4416b69275a7ef4918f4486bfb848`
- SHA-256 of the section reproduced below (from "Amendment 9, revision 3 — Experiment 3c RESULT" to
  the end of that file): `229e1bc08eeb39905543fa88f644fdec17b9857e5931e4d7795d23b2c88a8d01`
- SHA-256 of the frozen code, designs and pilot summary at that commit:
  - `analyze_experiment3e.py`: `eed44366b6761cab711d793897ba1daa915fd3866fa134f1205ab35166c48aa9`
  - `analyze_exp3e_pilot.py`: `c8b15cc5c128dbcaf001b81287947303ffc5fdb3bf7f064cbce5539adb392857`
  - `analyze_experiment3b.py`: `ace60e6575b112eaf019566d6fbfcbeeb34865889a1a89256971146352dc4cc2`
  - `analyze_experiment3a.py`: `875bb7500033ee9e30cd6315ed3739c5bdb64101337254ca087249736e1693ca`
  - `analyze_exp3b_pilot.py`: `f12413c10a2b08978101bbc6d69a227d8db1b654098d6b9cf31090e8243e0873`
  - `run_campaign_exp3e_pilot.sh`: `d17391802a53164914472be7071cad93e5b2236f0fa381e6ca3b8eb9a11a6961`
  - `run_campaign_exp3e_eval.sh`: `c5b35fcf514a185758692121c45e671f66f8fbdf7ce004d5b10e54f4e6ceb05d`
  - `jcm/mcb/planner.py`: `92ce2a4a88cfea16e545d960d08d3af7c7d48701931e142d97d061a170746f11`
  - `jcm/mcb/land_climatology.py`: `e04dc41f02bcc060d686fb16e47520e938ce6d617766e4559c85d1e7d8da6f2c`
  - `jcm/mcb/test_world.py`: `005911120c9babfc111b4bc251cb3d54e470a9856e495fb5206a119d0aba8fe0`
  - `jcm/mcb/strength_estimator.py`: `01a6884d78116693b04e61960e7db13d10a9ea7fed9f435b922e708f2a1fb276`
  - `jcm/mcb/hidden_strength.py`: `9d334a656e3f40bd25c89f77c5f98e99cad2a86b2283d854b5b895a78cd80140`
  - `jcm/mcb/scores.py`: `4ff68a872ec6ea54813918986951d73b86ac79f847701f60cb60360c84f58d06`
  - `run_test_world.py`: `357ac06ae898dda0ccb5e05f4198eda393712a242165eed5583e32f1e3b0880e`
  - `mcb_experiments_gpu/exp3e/registered_design.json`: `03f4d36997659648f65068a39db10385bde3cbfce6d2c00f84d0d320708947f2`
  - `mcb_experiments_gpu/exp3e/pilot/summary.json`: `eb48ec1b038743649ef1c5f34006daf54273c5ec3c1cd9fc9914477e156fa5a1`
  - `mcb_experiments_gpu/exp3d/registered_design.json`: `7e2782406001ee52f00488d18ce55e81fdb8e9ed4f039e85d28d86fcd0dcb2cd`
  - `mcb_experiments_gpu/exp3d/hidden_strength.json`: `bb00df59d405e00c5ca357b214508d4fe9c5cad42290d8bcfd788301364f8cc8`
- Earlier postings: Amendment 9 with revisions 0.1-0.4 (https://osf.io/2bs8p/); revision 1
  (Experiment 3a; https://osf.io/7bwe4/); revision 1.1 (Experiment 2 and the snip test;
  https://osf.io/pvwx9/); revision 2 (Experiment 3b; https://osf.io/g7y2e/); revision 3
  (Experiments 3c and 3d; https://osf.io/fyt9r/).

## Timeline

- 2026-10-11/12: Experiments 3c and 3d's evaluations ran and were analysed (logged below).
- 2026-10-18: read-only diagnostics on existing reference runs, and analysis sensing added to the
  planner (commit 8f4af5cd).
- 2026-10-19: Experiment 3e's pilot ran on 3d's training states only (10:33-13:03 PDT); the smoke
  test of its evaluation script ran on one training state; revision 4 was frozen in commit
  79d9494f.
- 2026-10-19 15:40 PDT: this posting, before any evaluation run of Experiment 3e.

---

**Amendment 9, revision 3 — Experiment 3c RESULT (logged 2026-10-12; evaluation 2026-10-11 16:59 to
2026-10-12 12:36 PDT, sharing the GX10 with 3d; the registered `analyze_experiment3c.py` was applied
unchanged).**
All 96 runs (2 arms × 48 states, 6 members each, against 3b's 10-member references) finished. No run
failed, no state was dropped, and there was no deviation. 3b's runs of the other seven arms were reused
as registered (A4), and the analysis's checks of sensing and member counts passed. The 225 frozen code
and design files on the GX10 were re-checked against b8152e3f after the run. Primary endpoint
`J_zonal`, on the means of the 16 macro states:
* **H1** `plan_learn_rs` vs `pi`: rel −15.0% (90% CI −24.1% to −4.7%), t p 0.0033, Wilcoxon p
  0.0027 (14 of 16 macro states favour `plan_learn_rs`) → **better**.
* **H2** `plan_learn_rs` vs `plan_naive_rs`: rel −2.1% (−14.0% to +12.0%), Holm p 0.73 (Wilcoxon
  0.82) → **inconclusive**.
* **H3** `plan_learn_rs` vs 3b's `plan_learn`: rel +37.0% (+15.1% to +63.0%), Holm p 0.0066
  (Wilcoxon 0.0042) → **worse**.

*Arm means:* `plan_learn` 0.00276, `plan_oracle` 0.00295, `plan_naive` 0.00341, `plan_learn_rs`
0.00378, `plan_naive_rs` 0.00386, `adaptive` 0.00391, `pi` 0.00445, `fixed` 0.01405, `uncontrolled`
0.01900 (the arms without the suffix are 3b's runs).

*Descriptive* (unadjusted p):
* `plan_naive_rs` vs `pi`: −13.2% (−23.9% to −1.8%, t p 0.049; Wilcoxon 0.083).
* `plan_learn_rs` vs `adaptive`: −3.3% (−18.3% to +14.6%, p 0.70).
* `plan_naive_rs` vs `plan_naive`: +13.2% (−4.3% to +31.4%, t p 0.11; Wilcoxon 0.025).
* `plan_learn_rs` vs `plan_oracle`: +28.1% (+10.3% to +48.8%, p 0.0039); `plan_naive_rs` vs
  `plan_oracle`: +30.8% (+12.9% to +51.8%, p 0.0012).
* By overall factor 0.5 / 1 / 2: `plan_learn_rs` vs `pi` is −29% / −9% / −10%. At factor 2,
  `plan_learn_rs` overcools by 0.041 K (`pi` by 0.026 K).
* Learning curve of `plan_learn_rs` (mean |log(estimate/truth)| per band at days 0, 28, 56, 98 and
  168): 0.59, 0.59, 0.59, 0.57, 0.51. 3b's `plan_learn`: 0.59, 0.35, 0.15, 0.08, 0.03.

The full output is `mcb_experiments_gpu/exp3c/exp3c_analysis.json` (SHA-256 bd9462e7…7f6d72), and the
plain-language account is `MCB_PROJECT_REPORT.md` Part 28.6.

*By A9:*
* H1 supports "the learning planner's advantage over classical feedback survives when it cannot see
  the weather".
* H2 is not better: "two-week forecast misses dominated by weather do not reveal the strength within
  one episode. This is the one-Earth detection limit of Part 27, seen from inside a controller."
* H3 gives the cost of not seeing the weather: 37% for the learning planner.

**Amendment 9, revision 3 — Experiment 3d RESULT (logged 2026-10-12; evaluation references 2026-10-11
17:03 to 2026-10-12 00:47, runs to 2026-10-12 22:46 PDT, sharing the GX10 with 3c until 12:36; the
registered `analyze_experiment3d.py` was applied unchanged).**
All 288 runs (6 arms × 48 states, 6 members each, against 10-member references made with the fixed
land) finished, and no state was dropped. 170 runs failed once because of an operating-system update
and were re-run once, as §10 provides (conduct note below). The analysis's land-model checks passed,
and the 225 frozen code and design files on the GX10 were re-checked against b8152e3f after the run.
Primary endpoint `J_zonal`, on the means of the 16 macro states:
* **H1** `plan_learn` vs `pi`: rel −42.3% (90% CI −51.4% to −30.4%), t p 4.6×10⁻⁵, Wilcoxon p
  3.1×10⁻⁴ (15 of 16 macro states favour `plan_learn`) → **better**.
* **H2** `plan_learn` vs `plan_naive`: rel −10.9% (−21.9% to +2.7%), Holm p 0.12 (Wilcoxon 0.058;
  12 of 16) → **inconclusive**.
* **H3** `pi` vs `fixed`: rel −67.3% (−74.0% to −58.7%), Holm p 5.3×10⁻⁷ (Wilcoxon 3.1×10⁻⁵; 16 of
  16) → **better**.

*Replication (B7):* 3b's estimates (−38.0%, −19.1%, −68.3%) point the same way as 3d's, and each lies
inside 3d's 95% interval (H1 −52.9% to −27.9%; H2 −23.7% to +5.6%; H3 −75.2% to −56.8%). The verdicts
go from 3b to 3d as follows: H1 better → better, H2 better → inconclusive, H3 better → better.

*Arm means:* `plan_learn` 0.00284, `plan_naive` 0.00319, `adaptive` 0.00388, `pi` 0.00492, `fixed`
0.01506, `uncontrolled` 0.01836.

*Descriptive* (unadjusted p):
* `plan_naive` vs `pi`: −35.2% (−42.6% to −26.2%, t p 1.1×10⁻⁵).
* `adaptive` vs `pi`: −21.2% (−34.3% to −5.9%, t p 0.0043).
* `plan_learn` vs `adaptive`: −26.8% (−38.1% to −12.0%, t p 0.0060).
* By overall factor 0.5 / 1 / 2:
  * `plan_learn` vs `pi`: −36% / −12% / −57%;
  * `plan_learn` vs `plan_naive`: −14% / +8% / −20%;
  * `fixed` scores 0.0062 / 0.0045 / 0.035.
* Learning curve (mean |log(estimate/truth)| per band at days 0, 28, 56, 98 and 168): 0.59, 0.35,
  0.15, 0.08, 0.04.
* Gains G (no brightening over controlled; > 1 is better), with the land now physical:
  * land temperature: `plan_learn` 1.09, `plan_naive` 0.96, `pi` 1.02, `adaptive` 1.14, `fixed`
    0.97;
  * land precipitation: 0.92–0.99;
  * global precipitation: 0.87–0.99.

The full output is `mcb_experiments_gpu/exp3d/exp3d_analysis.json` (SHA-256 af614aef…40606f), and the
plain-language account is `MCB_PROJECT_REPORT.md` Part 28.7.

*Conduct note: the outage and the re-run.*
* *The update.* At 17:05 PDT on 2026-10-12, the GX10's system updater was run at the machine's own
  screen, not by the campaign. It upgraded about 500 packages:
  * it removed the running kernel's NVIDIA module and installed kernel 7.0.0-1019-nvidia;
  * it updated the driver from 580.159.03 to 580.178.04;
  * it updated the system Python's patch build (3.12.3, Ubuntu build 0.15 → 0.17).

  The machine restarted at 17:09 and 17:29. The project's Python packages (JAX and jaxlib 0.10.2,
  with their CUDA 12.9 wheels) did not change.
* *The failures.* From 17:07:25, every new run failed at start ("No visible GPU devices"): 170 runs
  (`pi` 26; `adaptive`, `fixed` and `uncontrolled` 48 each).
  * The 118 runs finished before (`plan_learn` 48, `plan_naive` 48, `pi` 22) were kept.
  * The wrapper's automatic re-run stopped at its land-model check for the same reason, so no run was
    re-run.
  * The analysis refused, printing only "arms with no runs: ['uncontrolled', 'adaptive', 'fixed']".
    No result existed or was seen.
* *The re-run.* At 18:42:19, one pass of the frozen `run_campaign_exp3d_eval.sh` (which skips finished
  runs) was started by hand, with `REVISION3_COMMIT` = b8152e3f. This is §10's single re-run of each
  failed run. All 170 finished, so no state was dropped and §10 was followed.
* *Reproducibility.* GX10 GPU runs are not repeatable bit for bit. The same run, repeated, diverges
  after its first control step into another weather sample, because nothing forces deterministic GPU
  arithmetic.
  * `pi` on state 11300, made before the update, was repeated twice after it.
  * The runner's zonal pattern error was 0.0081 before, and 0.0070 and 0.0061 after (ocean bias
    −0.022, −0.034 and −0.026 K).
  * The two post-update repeats differ from each other as much as from the original. So the update's
    effect cannot be told from ordinary run-to-run variation, which is weather-sized and unbiased.

  Reproducibility of these experiments is statistical, not bitwise. The repeats' run summaries (not
  used by the analysis) are `mcb_experiments_gpu/exp3d/repro/after_update_{1,2}/pi/ic11300.json`.
* *Exploratory check (not registered).* The macro states were split by whether `pi` ran before
  (101–113) or after (117–131) the update.
  * H1 is −51% and −41% in the two halves.
  * Between the same halves, `plan_learn` vs `plan_naive` (both run entirely before the update)
    moves similarly: −20% and −12%.
  * `adaptive` vs `pi` (`adaptive` always ran after) does not move: −20% and −23%.

*By B9 and §9:*
* H1 supports "3b's main result does not depend on the land-model defect".
* H2 is not better. Its registered reading is "the planner's state feedback already absorbs the
  hidden strength".
  * The verdict is inconclusive, not equivalent, so 3d establishes neither reading.
  * 3b's estimate lies inside 3d's interval.
  * Pooling 3b's and 3d's 32 macro states (exploratory, not registered): −15.2% (90% CI −23.8% to
    −5.4%).
* H3 says again that, without feedback, a design made at nominal strength fails under realistic
  strength uncertainty.
* The learning curve is again a lower bound on how fast one Earth reveals the strength (§11).

### Amendment 9, revision 4 — Experiment 3e (the planner corrects its forecast toward the true weather) frozen in full (frozen 2026-10-19, BEFORE any evaluation run of Experiment 3e)

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
  * of the evaluation script (2026-10-19). It ran every new arm's exact command next to 3d's smoke
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
  * `plan_learn_a10`: the learning planner at f = 0.1, the primary level (noise level 0.00569 K);
  * `plan_naive_a10`: the planner that assumes nominal strength, at f = 0.1;
  * `plan_learn_a30`: the learning planner at f = 0.3 (0.02028 K);
  * `plan_learn_a100`: the learning planner at f = 1, which is 3c's information in the fixed-land
    model (0.04703 K).
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

**E8. Pilot.** The oracle planner, told the true strength, ran with analysis sensing at each level (2026-10-19, 10:33-13:03 PDT; `run_campaign_exp3e_pilot.sh`) on 3d's 16 branch-1 training states, with 3 members and 13 segments, in the fixed-land model with 3d's planner settings, warming and hidden strengths. All 48 runs finished. Per level:
  * f = 0.1: noise level 0.00569 K (8.2 times exact sensing) from 576 forecast misses; kappa 0.00158 K from 571 forecasts with brightening; null spread 0.98; the oracle's latitude-pattern score 0.00383, against 3d's exact-sensing oracle on the same states -0.002 in mean log ratio (t -0.0); background temperature drift at the later corrections 0.16 K; sd of one shared strength factor by day 98 0.22;
  * f = 0.3: noise level 0.02028 K (29.3 times exact sensing) from 576 forecast misses; kappa 0.00782 K from 567 forecasts with brightening; null spread 0.72; the oracle's latitude-pattern score 0.00389, against 3d's exact-sensing oracle on the same states +0.001 in mean log ratio (t +0.0); background temperature drift at the later corrections 0.25 K; sd of one shared strength factor by day 98 0.71;
  * f = 1: noise level 0.04703 K (68.1 times exact sensing) from 576 forecast misses; kappa 0.01454 K from 531 forecasts with brightening; null spread 0.88; the oracle's latitude-pattern score 0.00498, against 3d's exact-sensing oracle on the same states +0.244 in mean log ratio (t +1.7); background temperature drift at the later corrections 0.29 K; sd of one shared strength factor by day 98 1.58;
  * for comparison: exact sensing (3d's pilot) 0.00069 K, with a shared-factor sd of about 0.027 by day 98 (f = 0.1's sensitivities); 3c's ocean-only level in the old land model 0.04861 K; the background's temperature drift on day 0 (another branch's weather) 0.20 K.

  The shared-factor sd is descriptive: it is computed from the pilot's sensitivities, not by the learner, which estimates each band separately. For scale, the log of the hidden strengths has a standard deviation of 0.74 across 3d's states and bands.

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
  * the design: `exp3e/registered_design.json` (SHA-256 03f4d36997659648f65068a39db10385bde3cbfce6d2c00f84d0d320708947f2), with its pilot record. 3d's
    design, strength table and references are revision 3's.
* *Cost:* about 20 GPU-hours on the GX10 (192 runs of 6 members; about the same in wall time, since 5 runs at a time keep the GPU busy).
* *Posting:* this revision is posted to OSF (project pqabf) before any evaluation run of 3e.
