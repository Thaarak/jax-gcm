# Pre-registration: Amendment 9 revision 1.1 (Experiment 2 and the snip-window test)

This file is the OSF posting of revision 1.1 of Amendment 9 of this project's pre-registration
(`PREREGISTRATION.md`). Revision 1 (https://osf.io/7bwe4/) deferred Experiment 2 to this revision.
Everything below the horizontal line is the registered text exactly as frozen in the public
repository; nothing in it has been edited.

## Frozen source

- Repository: https://github.com/Thaarak/jax-gcm (branch `fable-version`)
- Commit: `66e2c04e69f3994de8c150758c53d90005ecdb3c`, committed 2026-10-07 15:37:15 -0700
- SHA-256 of `PREREGISTRATION.md` at that commit: `fe9d68c1e2256e29bafe675020d6a43fef927bdb6563fa0b84939da2c26cef82`
- SHA-256 of the section reproduced below (from "### Amendment 9, revision 1.1" to the end of that file):
  `573548c8d2be9948a34138949e57a9b22d20587edc4a48cad24302903ba57857`
- SHA-256 of the frozen code and inputs at that commit:
  - `jcm/mcb/design.py`: `9eb8858133d6560318c0096c45be596370185cda137c47e0d86ae8e711aca17e`
  - `run_experiment2.py`: `47209ba7510d1185c91dd6f96850ea531acfcc74fa4b010ca2627c29e66a9b16`
  - `analyze_experiment2.py`: `6f4354f5324545370b1cb0f9c9126196066d0b389175c9ad3c8c83d0f10406a1`
  - `run_snip_test.py`: `a7462aeaeee22a7fbce43ead5c4db23af6f488027fe3284a193f9de73bdf9fb1`
  - `analyze_snip_test.py`: `a53a52939b4ecb4ebbd0bffc21b77b2e9c4b09a36edcb90b78ac7d9de8c1924b`
  - `run_campaign_exp2.sh`: `e7681afb15d77f1dae4bde6d655a2aff09220dfa515a6bd2a91fb54d0e4bfd52`
  - `jcm/mcb/ladder.py`: `bc644fff67ac56b62ea0e43b4a16b23eae9bf94dc3891263b4489bc1fd973824`
  - `jcm/mcb/planner.py`: `f533558e63ca2a91f27f6d350bc5e8d861ce365035f8ca0a4febfbbf4483788f`
  - `jcm/mcb/scores.py`: `4ff68a872ec6ea54813918986951d73b86ac79f847701f60cb60360c84f58d06`
  - `jcm/mcb/test_world.py`: `adfa255b009ab2095b3538af4e1057d88191f5debdf0c2f6f93014030a9db857`
  - `jcm/mcb/gradient_fidelity.py`: `532d21b5a1a1395a3d346ece6880ae2e3dc6400b42fcc3d1b0260aba8f66c01c`
  - `jcm/mcb/gradient_truncation.py`: `12bbd74282183de6a8e341a1345ba97847802940b50f6de08fd61c49662bb216`
  - `analyze_experiment3a.py`: `875bb7500033ee9e30cd6315ed3739c5bdb64101337254ca087249736e1693ca`
  - `analyze_gradient_fidelity.py`: `d099c9ccb50432319225203f7bb071ca9750ee285a5acfe7fc09ef749dfad26f`
  - `mcb_experiments_gpu/exp1/exp1_gradient_fidelity.npz`: `fe7624b64197efb29af92dac8786d9c12b227b5b576274d1cc2a18987e607322`
  - `mcb_experiments_gpu/exp1/exp1_gradient_fidelity.json`: `954bb84386edf23621fe9142f7f7b2d5dacf29f96ef5f29113fb4346ea5a1533`
- Earlier postings: Amendment 9 with revisions 0.1-0.4 (https://osf.io/2bs8p/); revision 1
  (https://osf.io/7bwe4/). Entries logged between revision 1 and this revision (Deviation D1 and the
  Experiment 3a result) are in the full file at the commit, attached as
  `PREREGISTRATION_at_66e2c04e.md`.

## Timeline

- 2026-10-01: Experiment 1 ran; outcome A', W* = 14.
- 2026-10-06/07: Experiment 3a ran on its own evaluation states (revision 1).
- 2026-10-07 15:37 PDT: this posting. No snip-test Jacobian and no Experiment-2 reference, design run or score
  exists; no Experiment-2 state has been run.

---

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
