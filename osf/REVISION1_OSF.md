# Pre-registration: Amendment 9 revision 1 (Experiment 3a), and the Experiment 1 result

This file is the OSF posting of revision 1 of Amendment 9 of this project's pre-registration
(`PREREGISTRATION.md`), together with the logged Experiment 1 result that revision 1 builds on.
Everything below the horizontal line is the registered text exactly as frozen in the public
repository; nothing in it has been edited.

## Frozen source

- Repository: https://github.com/Thaarak/jax-gcm (branch `fable-version`)
- Commit: `5e11b0827ca4b7afd0a7fafc0c2afbd9a0a4d374`, committed 2026-10-06 11:50:36 -0700
- SHA-256 of `PREREGISTRATION.md` at that commit: `da25232e8b8517a5f2885406c07ba48d1caf2106af86a3957011ea7de09e4a69`
- SHA-256 of the section reproduced below (from "Amendment 9 — Experiment 1 RESULT" to the end of
  that file): `700f7d1ede9b6d85bfaa7f421005a5f6444ba4479d1ad1315df81c39d118de12`
- SHA-256 of the frozen code and designs at that commit:
  - `analyze_experiment3a.py`: `fe73794193c77ece2ee49bd1ebbdf04fb4d8753ae3f5b89cf1ca0fb04bc42282`
  - `run_campaign_exp3a_eval.sh`: `52f81957ad3f1d3cc5a3b1998749a6aa34060072bd86203a2dc49cb140f9c455`
  - `run_campaign_exp3a_train.sh`: `c67e2579a83caf42c1e78aa2feeff453a8f8351f65020a42cd74764f54e0d3be`
  - `jcm/mcb/planner.py`: `42dfb89c0041e1d8bec1283fa7d6a3e0397edf5740ea3e6bab79c16e46d517e5`
  - `jcm/mcb/feedback.py`: `67d7f449b2529eda2338f0a7d469b158583b844f27dd8af7e062cf5345b16039`
  - `jcm/mcb/ladder.py`: `bc644fff67ac56b62ea0e43b4a16b23eae9bf94dc3891263b4489bc1fd973824`
  - `jcm/mcb/scores.py`: `4ff68a872ec6ea54813918986951d73b86ac79f847701f60cb60360c84f58d06`
  - `jcm/mcb/test_world.py`: `adfa255b009ab2095b3538af4e1057d88191f5debdf0c2f6f93014030a9db857`
  - `run_test_world.py`: `6a4f1e19ddc6730a964f77cc9d7505d1091e36baeb1d14c04ce5e8b70aea0548`
  - `run_controllers.py`: `1ab8bb3fec9cffda10a0f223d158e653977ccc5fd6514516a2bdf94bbec687ba`
  - `mcb_experiments_gpu/exp3a/train/ladder.json`: `4362fb580cbfc1e45a635e0abd757805fc7c53d19ad3a825472f684a057df0a0`
- Earlier postings: Amendment 9 with revisions 0.1-0.4 (https://osf.io/2bs8p/).

## Timeline

- 2026-10-01: Experiment 1 ran (17:48-20:34 PDT); outcome A', W* = 14.
- 2026-10-05: the step-23 pilot (training states only) fixed the planner's settings; this revision's
  training-side runs built the opponents' designs on training states.
- 2026-10-06 11:52 PDT: this posting, before any evaluation reference, run or score of Experiment 3a exists.

---

**Amendment 9 — Experiment 1 RESULT (logged 2026-10-05; the campaign ran 2026-10-01 17:48-20:34 PDT
and the frozen `analyze_gradient_fidelity.py` was applied unchanged).**

* *Registered outcome:* **A'**. At the primary endpoint (T0 at 60 days) the snipped windows W = 1, 7 and
  14 days are all `useful`, and full BPTT is `inconclusive`. **W\* = 14 days.**
* *Registered prediction on LAND:* **untestable**. The LAND truth is unresolved (signal-to-noise
  0.3-1.2 beyond 15 days).
* *Reported, not gated* (MCB_PROJECT_REPORT.md Part 20):
  * At 60 days, each W\* = 14 run lies within 14-16% of the truth and reads 86-90% of its size.
    Single full-BPTT runs are off by 42-74%.
  * At 120 days, W\* = 14 runs are within 30-33% and read 69-75% of the truth's size. Full BPTT blows
    up by 10^4-10^5.
* *Files:* `mcb_experiments_gpu/exp1_gradient_fidelity{.json,.npz,_analysis.json}`, committed in
  fbfa4b8d.
* *Consequence, as registered:* Experiments 2-3 use W\* = 14.

### Amendment 9, revision 1 — Experiment 3a frozen in full; Experiments 2 and 3b deferred (frozen 2026-10-05, BEFORE any evaluation reference, run or score of Experiment 3a, and before any Experiment-2 or Experiment-3b data)

**What existed when this was frozen.** Everything below is on training states (exp3_train, macro
states 0, 2, ..., 14). Nothing ran on `exp3_eval` or on any `exp2_*` role.

* *The step-23 pilot* (MCB_PROJECT_REPORT.md Part 22):
  * It ran on 2026-10-05 on the 8 branch-2 training states.
  * Its decision rules R1-R8 were written into `run_planner_pilot.py` before any of its data existed.
  * Its closed-loop check ran on training states 0002 and 0202.
* *This revision's training-side runs* (`run_campaign_exp3a_train.sh`, 2026-10-05):
  * ramp references for the 16 training states (branches 2 and 3);
  * one-band response runs on those 16 states;
  * the 120-day planner on the 8 branch-3 states;
  * the ladder designs.
* *Evaluation references for `exp3_eval` did not exist.* `run_test_world.py references` refuses evaluation
  roles without `--allow-eval-roles`.

**Scope.**
* *Frozen in full here:* Experiment 3a, planning with a perfect model (Part 18 step 28).
* *Deferred:* Experiment 2 (fixed-pattern design) and Experiment 3b (hidden strength, students).
  * Each gets its own revision (1.1, 1.2), frozen before any of its data.
  * The hidden-strength range, the student, the copying loop and the direct-training budgets are
    therefore not fixed here.
* *Why split:* the 2026-10-02 advisor meeting (Part 21) put the controller comparison, especially
  **where and how much to brighten**, first, and machine learning for its own sake last.

**1. Test world** (fixed by the pilot's rules R4-R6).
* *Base climate:* the Q-flux climate (revisions 0.2-0.3).
* *Warming:* a Q-flux ramp from 0 at day 0 to 6 W m^-2 at day 182 (0.03296703 W m^-2 per day; no
  step).
* *Strength:* efficacy 1 (perfect model).
* *Episode:* 13 segments of 14 days from each state's (January) start, with decisions every 14 days.
* *Scoring window:* days 98-182.
* *Target:* the normal-climate (no warming) 5-member mean of the state's reference ensemble (member seeds
  from 93000, 300 days).
* *Cap:* every band setting lies in [0, 0.15].

**2. States and members.**
* *States:* all 24 `exp3_eval` states (macro states 1, 3, ..., 15 x branches 2, 3, 4).
* *Members:* every arm runs **3 members**, the references' first three member seeds, identical across
  arms, so the arms' weather noise matches (the fairness rule of Part 18 step 11).
  * This replaces the pilot's R6 (5 members), to roughly halve the cost (about 30 GPU-hours instead of
    about 50).
  * Paired differences between arms are unbiased with any common member count.
  * The uncontrolled arm also runs the same 3 members, so the gains G are matched too.

**3. Arms (10).** Every planner uses:
* Gauss-Newton with 1 step and 1 copy (copy amplitude 0.001 K, seeds from 97000);
* the zonal-mean objective (`representation="zonal"`);
* alpha = 1, beta = 0.5, mu = 6.325e-4 and lam = 6.325e-3 (the pilot's R7: Dubey et al.'s 0.01 and 0.1
  times s^2, with s = 0.25 K per unit albedo);
* cap 0.15, and a first guess of half the cap.

| Arm | What it is |
|---|---|
| `plan120` (PRIMARY) | 120-day look-ahead, atmosphere snipped every 14 days (W\*) |
| `plan60` | 60-day look-ahead, W\* = 14 |
| `plan14` | 14-day look-ahead, no truncation (Dubey et al.'s myopic planner; at 14 days the snip changes nothing) |
| `uncontrolled` | no brightening |
| `uniform_cancel` | ladder rung 1: one setting in every band, sized on the training states to cancel the ocean-mean warming over days 98-182 |
| `uniform_effort` | rung 2: one setting in every band with the training planner runs' mean effort |
| `planner_average` | rung 3: the training planner runs' time-mean pattern, held fixed |
| `linear_response` | rung 4: the best fixed pattern inside the cap under the linear model from the 16 training response runs (zonal objective, mu as above), solved by bounded least squares |
| `pi` | the GLENS-style controller (`jcm.mcb.feedback.IndexPIController`) |
| `adaptive` | the Tier-2 adaptive law with relaxation 0.5, scaling the rung-4 pattern |

The `pi` arm holds T0, T1 and T2 on target with:
* proportional and integral action;
* a closed-loop time scale of 42 days and damping 1;
* anti-windup;
* feedforward of the forecast warming;
* sensitivities fitted over days 1-60 of the 16 training response runs.

The `adaptive` arm uses the same sensitivities. Both feedback arms are given the forecast because the
planners are given the warming.

**3a. Frozen designs (built on training states only, 2026-10-05 20:55 to 2026-10-06 00:15 PDT).**
The training phase (`run_campaign_exp3a_train.sh`) built, on the 16 `exp3_train` states (macro states
0, 2, ..., 14 x branches 2, 3) under the same ramp-6 test world:
* references (5 members x 300 days each);
* one-band response runs (setting +0.1 in one band, 5 members x 182 days, all 5 bands, all 16 states);
* the `plan120` planner on the 8 branch-3 states, 1 member each, with exactly the arm's settings above.

The ladder (`run_controllers.py ladder`, zonal objective over days 98-182, mu as above, planner runs
pooled) froze these amplitudes, bands ordered south to north, in
`mcb_experiments_gpu/exp3a/train/ladder.json` (SHA-256 `4362fb580cbfc1e45a635e0abd757805fc7c53d19ad3a825472f684a057df0a0`):

| Rung | Amplitudes | Objective predicted by the linear model |
|---|---|---|
| `uniform_cancel` | 0.0185, 0.0185, 0.0185, 0.0185, 0.0185 | 0.00273 |
| `uniform_effort` | 0.0390, 0.0390, 0.0390, 0.0390, 0.0390 | 0.03023 |
| `planner_average` | 0.0207, 0.0310, 0.0149, 0.0338, 0.0447 | 0.01069 |
| `linear_response` | 0.0249, 0.0201, 0.0106, 0.0176, 0.0201 | 0.00231 |

The evaluation script reads these numbers from that file; it is committed with this revision and
is not rebuilt. The `pi` and `adaptive` arms fit their sensitivities from the same 16 response
files (`mcb_experiments_gpu/exp3a/train/responses/`), against the training references.

For the record, before any evaluation data exist, the 8 training planner runs (1 member each) gave:
* mean ocean bias -0.034 K (uncontrolled +0.128 K): every run overshoots slightly;
* zonal objective 0.0111 (uncontrolled 0.0188), 41% lower on average, though 2 of 8 single-member
  runs scored worse than uncontrolled;
* effort 0.029-0.043, and at most 2% of settings at the cap.

The linear model predicts `linear_response` at 0.0023, well below these planner runs. H2 is
therefore a live test, and a `worse` verdict for the planner there is a real possible outcome, not a
failure of the experiment. None of these numbers changes any registered setting.

**4. Endpoints.**
* *Primary:* `J_zonal`, which is `alpha <e>^2 + beta Var(e)` with no penalties, of the ocean zonal-mean
  profile of `e`. Here `e` is the 3-member-mean SST over days 98-182 minus the target's mean over the
  same days.
* *Secondary:*
  * `J_map` (the same on the grid-point map) and the ocean-mean bias;
  * Dubey et al.'s effort, and the mean setting as a share of the cap;
  * the gains G against the matched uncontrolled arm for SST (ocean), land temperature (land) and
    rainfall (land and global).
* *Descriptive only:* land temperature and rainfall, because the pilot's R8 found their signal below the
  noise.

**5. Hypotheses** (all compare `plan120` with a comparator on `J_zonal`; a Holm family of four).
* **H1** vs `plan14`: does looking 120 days ahead through the ocean beat two-week planning?
* **H2** vs `linear_response`: does the planner beat the best fixed pattern (the classical design)?
* **H3** vs `pi`: does the planner beat the classical feedback controller?
* **H4** vs `planner_average`: does changing the pattern over time beat the same planner's average
  pattern held fixed?

*Secondary comparisons, unadjusted:*
* plan60 vs plan120;
* plan60 vs plan14;
* linear_response vs uniform_cancel (choosing where, for a fixed pattern);
* linear_response vs pi;
* adaptive vs pi;
* planner_average vs uniform_effort;
* every arm vs uncontrolled.

**6. Tests and outcome grid** (`analyze_experiment3a.py`, frozen with this revision). For arm A and
comparator B, `d = J_A - J_B` per state, and `rel = mean(d) / mean(J_B)`.
* *Unit of replication:* the macro state (8). Branches share a macro state's ocean, so they are averaged
  first.
  * Paired t test, two-sided, on the 8 macro means.
  * Exact Wilcoxon signed-rank test, reported alongside; a disagreement is reported, not resolved.
* *Interval:* a hierarchical bootstrap of `rel` (macro states, then branches within each; 10000
  replicates, seed 2026), with 90% and 95% intervals.
* *Verdicts:*

| Verdict | Condition |
|---|---|
| better | Holm-adjusted p < 0.05 and rel <= -5% |
| worse | Holm-adjusted p < 0.05 and rel >= +5% |
| equivalent | 90% interval inside [-10%, +10%] (TOST at 5%) |
| negligible | p < 0.05 but \|rel\| < 5% |
| inconclusive | otherwise |

**7. Interpretation, fixed in advance.**
* *H1 better:* "planning months ahead through the ocean's memory beats two-week planning". This is the
  claim Dubey et al.'s atmosphere-only setting cannot make.
* *H1 equivalent or negligible:* "with re-planning every two weeks, a two-week look-ahead suffices
  here". It is reported as such, and the cheaper planner is recommended.
* *H2 and H4 better:* "changing where and how much over time beats the best fixed pattern".
* *H4 not better while H2 is better:* the gain comes from the pattern, not from changing it.
* *H3 better:* "the planner beats the field's standard feedback controller even with a perfect model".
* *H3 equivalent, worse or negligible:* "in a perfect model, feedback on three indices is enough", which
  moves the planner's case to Experiment 3b (uncertainty) and to targets beyond three indices.
* Every verdict, every arm and every secondary comparison is reported, whatever the outcome.

**8. Failures.**
* A run that fails is re-run once with identical settings.
* If it fails again, that state is dropped from every arm and reported, and nothing else is excluded.
* Losing more than 4 states is reported as a deviation.
* No arm is re-tuned after any evaluation run exists.

**9. Code and cost.**
* *Frozen code:*
  * `run_campaign_exp3a_eval.sh` (runs everything above, then the analysis);
  * `analyze_experiment3a.py`;
  * `jcm/mcb/planner.py` with the zonal objective;
  * `jcm/mcb/feedback.py` and `jcm/mcb/ladder.py`;
  * `run_test_world.py` and `run_controllers.py`.
* *Commit:* recorded at posting.
* *Estimated cost:* about 30 GPU-hours (the planners dominate; measured in Part 22: about 90 s per
  re-plan at 120 days).

**10. Posting.** Before any evaluation reference is built, this revision, the Experiment 1 result, the
frozen designs and the frozen commit are posted to OSF (project https://osf.io/pqabf/), as Amendment 9
was.
