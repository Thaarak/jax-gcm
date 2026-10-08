
### Amendment 9, revision 2 — Experiment 3b frozen in full (frozen FREEZE_DATE, BEFORE any evaluation reference, run or score of Experiment 3b)

**What existed when this was frozen.** Nothing ran on `exp3b_eval` beyond generating its starting
states, which uses no outcome. Everything else is on training states or earlier experiments:
* *Experiment 3a's result* (revision 1, logged above). It ran on `exp3_eval` states, none of which
  3b uses.
* *Fresh starting states* (`run_generate_macro_ics.py --plan exp3b`, 2026-10-07):
  * 32 macro states (16-47) continue the same control run from macro state 15, every 365 days.
  * They are interleaved: even states train (`exp3b_train`, branches 0-1) and odd states evaluate
    (`exp3b_eval`, branches 0-2).
  * They never reuse a state that Experiment 3a's results or its noise analysis (2026-10-07) used.
  * Generation diagnostics: GENERATION_DIAGNOSTICS.
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
* *Members:* every arm runs **MEMBERS members**, the references' first MEMBERS member seeds, identical
  across arms. The number comes from the pilot's registered power rule (8).

**3. Hidden strength** (`jcm/mcb/hidden_strength.py`; table
`mcb_experiments_gpu/exp3b/hidden_strength.json`, SHA-256 TABLE_SHA).
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
| `plan_learn` (PRIMARY) | starts at nominal strength; before every re-plan it compares its last 14-day forecast of the ocean latitude profile with what happened and updates LEARNER_MODE strength by regularized least squares (`jcm.mcb.strength_estimator`; prior sd 1, noise level NOISE_K K from the pilot's rule, bounds [0.2, 5]) |
| `plan_naive` | assumes nominal strength 1 throughout |
| `plan_oracle` | is told the true strengths. This is perfect knowledge, not necessarily the best possible score: in the pilot it scored worse than the learner (see 8), probably because the 14-day planner over-corrects. It is reported descriptively |
| `pi` | revision 1's GLENS-style PI on T0-T2 with feedforward, closed-loop time PI_DAYS d; sensitivities fitted over days 1-60 of the 16 branch-0 training response runs (nominal strength) |
| `adaptive` | revision 1's Tier 2 law (relaxation 0.5), learning one overall strength and scaling the fixed design |
| `fixed` | the best fixed pattern inside the cap under the linear model from the 16 training response runs (zonal objective, days 98-182, mu as above): FIXED_AMPLITUDES |
| `uncontrolled` | no brightening |

All registered values are collected in `mcb_experiments_gpu/exp3b/registered_design.json` (SHA-256
DESIGN_SHA), which the evaluation script reads.

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

**8. Power** (the pilot's rule, `analyze_exp3b_pilot.py decide`). POWER_SUMMARY

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
* *Cost:* about COST GPU-hours on the GX10.
* *Posting:* this revision is posted to OSF (project pqabf) before any evaluation reference is built.
