# Pre-registration: Amendment 9 revision 3 (Experiments 3c and 3d), with the results of Experiments 2 and 3b

This file is the OSF posting of revision 3 of Amendment 9 of this project's pre-registration
(`PREREGISTRATION.md`), together with everything logged since revision 2's posting, in file
order:
* the result of Experiment 2 and the snip test (revision 1.1);
* revision 2 itself, already posted as https://osf.io/g7y2e/ and reproduced here only because it
  sits between the results in the file;
* the result of Experiment 3b (revision 2);
* revision 3.
Everything below the horizontal line is the registered text exactly as frozen in the public
repository; nothing in it has been edited.

## Frozen source

- Repository: https://github.com/Thaarak/jax-gcm (branch `fable-version`)
- Commit: `b8152e3fc821cbd43b38e26c7b654179978037c9`, committed 2026-10-11 16:50:03 -0700
- SHA-256 of `PREREGISTRATION.md` at that commit: `ddc2b2abafe331aa69f110f7beb297f16c5c86c1210b60352e3f5472ef1f57bb`
- SHA-256 of the section reproduced below (from "Amendment 9, revision 1.1 — snip test and
  Experiment 2 RESULT" to the end of that file): `d067a31f09cc66263daa17ca759fd59d9c0e036edb7dad1ad2b3d15a200cfed5`
- SHA-256 of the frozen code and designs at that commit:
  - `analyze_experiment3c.py`: `6a808c9b6ac2beac7cf04d7e2dd48f898242852bb59f4071ddd51fa0cbf85ba2`
  - `analyze_experiment3d.py`: `bae983c778d21d98ea7a7c944bcceb36817a441d7a42be0a535bab492e315a04`
  - `analyze_experiment3b.py`: `ace60e6575b112eaf019566d6fbfcbeeb34865889a1a89256971146352dc4cc2`
  - `analyze_experiment3a.py`: `875bb7500033ee9e30cd6315ed3739c5bdb64101337254ca087249736e1693ca`
  - `analyze_exp3b_pilot.py`: `f12413c10a2b08978101bbc6d69a227d8db1b654098d6b9cf31090e8243e0873`
  - `analyze_exp3c_pilot.py`: `0ee64168a117a6d9dc0c7d55e74c2575f4ca38ff570fea75ab8300f5a3045a32`
  - `run_campaign_exp3c_pilot.sh`: `0e6a04788d3788c56c7cc93e652175442f1a614fa0daec627aa69d4bb0052f2b`
  - `run_campaign_exp3c_eval.sh`: `e903cb73cd8b9962f82768d0260faea3a4391dec1e6cbc0205ac3f848c2814ac`
  - `run_campaign_exp3d_prep.sh`: `acc3411ffd9b8f5ec37ed1506cc6caa34263ab2203d7a87762c24fdb5d3a0c9e`
  - `run_campaign_exp3d_eval.sh`: `c1b0c865378fde67f51dbf76c65cf1521a7bcef0c06f2a4b78b211977342a61e`
  - `jcm/mcb/planner.py`: `33049743682210dbbafa68e5787305b1dd2e8ef8d68ce2c71fc0397e90138b0d`
  - `jcm/mcb/land_climatology.py`: `e04dc41f02bcc060d686fb16e47520e938ce6d617766e4559c85d1e7d8da6f2c`
  - `jcm/mcb/test_world.py`: `005911120c9babfc111b4bc251cb3d54e470a9856e495fb5206a119d0aba8fe0`
  - `jcm/mcb/strength_estimator.py`: `01a6884d78116693b04e61960e7db13d10a9ea7fed9f435b922e708f2a1fb276`
  - `jcm/mcb/hidden_strength.py`: `9d334a656e3f40bd25c89f77c5f98e99cad2a86b2283d854b5b895a78cd80140`
  - `jcm/mcb/feedback.py`: `67d7f449b2529eda2338f0a7d469b158583b844f27dd8af7e062cf5345b16039`
  - `jcm/mcb/scores.py`: `4ff68a872ec6ea54813918986951d73b86ac79f847701f60cb60360c84f58d06`
  - `jcm/mcb/qflux.py`: `18fa93f87e3ddeaee696549c7a877bc68fe650a982df797d0f23cb00d3d0768b`
  - `run_coupled_training.py`: `c74bcb17ed4cb9ba64f322e62f8f851178d2fc8c5ee38a3275c7cb46d53ce571`
  - `run_test_world.py`: `f6d5e786f831ba961219a54dd0024c52964b2ad74f3a78525a084936455d8991`
  - `run_controllers.py`: `0452dfb5aa7ff62b16e200ec0e2ef3d5ba25f61686cc02b8aacd661c4723c869`
  - `run_generate_macro_ics.py`: `8296f2df30f7e2ef92b413ce6f39d247c99a637d3e65d6768fe1810dc93ca68e`
  - `run_qflux_base_climate.py`: `85146b7bf99751f7dffdf521bce40d168e7c5f8c391a9aee0bb2f12ee042ff50`
  - `mcb_experiments_gpu/exp3c/registered_design.json`: `098571f6834429b6128cd80bf14f5bb25e8eba15c225f5de8505f71eb2867d2a`
  - `mcb_experiments_gpu/exp3d/registered_design.json`: `7e2782406001ee52f00488d18ce55e81fdb8e9ed4f039e85d28d86fcd0dcb2cd`
  - `mcb_experiments_gpu/exp3d/hidden_strength.json`: `bb00df59d405e00c5ca357b214508d4fe9c5cad42290d8bcfd788301364f8cc8`
  - `mcb_experiments_gpu/exp3d/train/ladder.json`: `3948c3f16192eea0b19aff5393abe8388d81c49b91562fd303bb56d1f8525af8`
- Earlier postings: Amendment 9 with revisions 0.1-0.4 (https://osf.io/2bs8p/); revision 1
  (Experiment 3a; https://osf.io/7bwe4/); revision 1.1 (Experiment 2 and the snip test;
  https://osf.io/pvwx9/); revision 2 (Experiment 3b; https://osf.io/g7y2e/).

## Timeline

- 2026-10-07/08: Experiment 3b's evaluation ran and was analysed (logged below).
- 2026-10-08/09: Experiment 2 and the snip test ran and were analysed (logged below).
- 2026-10-09: the land fix, Experiment 3d's base climate, states, training side and oracle pilot,
  and Experiment 3c's pilot ran, on training states only.
- 2026-10-11: the smoke test of Experiment 3d's evaluation script ran, on one training state;
  revision 3 was frozen in commit b8152e3f.
- 2026-10-11 16:57 PDT: this posting, before any evaluation run of Experiment 3c and any evaluation reference,
  run or score of Experiment 3d.

---

**Amendment 9, revision 1.1 — snip test and Experiment 2 RESULT (logged 2026-10-09; run 2026-10-08
21:55 to 2026-10-09 07:10 PDT on the GX10 from an export of 66e2c04e; the registered analyses were
applied unchanged).**
Every stage and all 176 evaluation episodes finished. Nothing failed, nothing was re-run, and there was
no deviation. Two operational settings do not affect results: the snip test ran alone with a larger
memory cap (0.25), and the evaluation episodes ran 3 at a time. The SHA-256 of `designs.json`,
ddf62344…ecacd, was logged at 03:08, before any evaluation reference existed.
* **Part A:** outcome **S-A**, `W_long` = 30.
  * T0@120 ratios are 0.69 (W 14), 0.74 (21) and 0.80 (30), all `useful`; full backpropagation is
    `failed` (95°).
  * **P1 confirmed:** the ratio rises, 14 < 21 < 30.
  * **P2 refuted:** the median single angles are 8.9°, 7.8° and 9.7°.
  * The merged re-analysis's W* is 30, for the record only; W* stays 14 for this revision.
* **Part B** (`J_zonal`, 8 macro states, Holm over four):
  * H1 `grad13` vs `brute13_eq`: rel +6.4% (90% CI −21% to +39%), Holm p 1.0 → **inconclusive**.
  * H2 `grad13` vs `sunlight13`: rel −7.8% (−32% to +24%), Holm p 1.0 → **inconclusive**.
  * H3 `grad5` vs `bptt5`: rel −77.7% (−84% to −70%), Holm p 5.0×10⁻⁵ → **better**.
  * H4 `grad13` vs `grad5`: rel +27.0% (−4% to +71%), Holm p 0.58 → **inconclusive**.
* *Arm means:* `uniform` 0.00412, `brute5_eq` 0.00415, `brute5` 0.00424, `grad5` 0.00424, `brute13`
  0.00494, `brute13_eq` 0.00506, `grad13` 0.00539, `sunlight13` 0.00584, `sunlight5` 0.00763,
  `bptt5` 0.01898, `uncontrolled` 0.01973.
* *Costs:* `M_eq` = 2 for both layouts. Brute force with 5 members costs 2.0× (b5) and 2.1× (b13) the
  gradient design's model time.
* *Secondary* (unadjusted):
  * `grad5` vs `brute5`: +0.1% (p 0.99).
  * `grad5` vs `sunlight5`: −44% (p 0.0004).
  * `brute5` vs `uniform`: +2.8% (p 0.82).
  * `sunlight5` vs `uniform`: +85% (p 0.002).
  * `grad13` vs `uniform`: +31% (p 0.16).
* *Training logs:* `bptt5`'s design never left zero. `grad13`'s last step put 0.136 (90% of the cap)
  in the 60°N band.
* *Expectations:*
  * `brute5` (0.0042) matches 3a's `linear_response` (0.0041).
  * The sunlight formula under-brightens: about half of brute force's settings, leaving a bias of
    +0.06 K.
  * `bptt5`'s Jacobians are useless.

The full outputs are `mcb_experiments_gpu/snip_test/snip_test_analysis.json` and
`mcb_experiments_gpu/exp2/exp2_analysis.json`, and the plain-language account is
`MCB_PROJECT_REPORT.md` Part 25.4.

*By section 11:*
* H1: the efficiency claim is not established; the interval and the cost ratio are reported.
* H2: it is not established that the gradient knows more than the sunlight formula with 13 bands.
* H3: "snipping is what makes gradient design possible at 3-6-month horizons".
* H4: it is not established that finer latitude control helps.

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

**Amendment 9, revision 2 — Experiment 3b RESULT (logged 2026-10-08; evaluation 2026-10-07 22:51 to
2026-10-08 21:54 PDT; the registered `analyze_experiment3b.py` was applied unchanged).**
All 48 states and 7 arms are complete (6 members each, against 10-member references). No run failed,
no state was dropped, and there was no deviation. The 13 frozen files on the GX10 were re-checked
against 676ce331 after the run. Primary endpoint `J_zonal`, on the means of the 16 macro states:
* **H1** `plan_learn` vs `pi`: rel −38.0% (90% CI −46.4% to −28.2%), t p 6.5×10⁻⁶, Wilcoxon p
  3.1×10⁻⁵ (all 16 macro states favour `plan_learn`) → **better**.
* **H2** `plan_learn` vs `plan_naive`: rel −19.1% (−31.7% to −5.2%), Holm p 0.0095 (Wilcoxon
  0.005) → **better**.
* **H3** `pi` vs `fixed`: rel −68.3% (−74.5% to −59.3%), Holm p 3.4×10⁻⁷ (Wilcoxon 3.1×10⁻⁵) →
  **better**.

*Arm means:* `plan_learn` 0.00276, `plan_oracle` 0.00295, `plan_naive` 0.00341, `adaptive` 0.00391,
`pi` 0.00445, `fixed` 0.01405, `uncontrolled` 0.01900.

*Descriptive* (unadjusted p):
* `plan_learn` vs `plan_oracle`: −6.5% (−21% to +10%, p 0.41).
* `plan_naive` vs `plan_oracle`: +15.6% (p 0.039).
* `plan_naive` vs `pi`: −23% (p 1.5×10⁻⁵).
* `adaptive` vs `pi`: −12% (p 0.026).
* `plan_learn` vs `adaptive`: −29% (p 1.5×10⁻⁴).
* By overall factor 0.5 / 1 / 2: `plan_learn` vs `pi` is −33% / −44% / −37%, and `fixed` scores
  0.0074 / 0.0044 / 0.030.
* Learning curve (mean |log(estimate/truth)| per band at days 0, 28, 56, 98 and 168): 0.59, 0.35,
  0.15, 0.08, 0.03.

The full output is `mcb_experiments_gpu/exp3b/exp3b_analysis.json` (SHA-256 d4bee14e…a83e18), and
the plain-language account is `MCB_PROJECT_REPORT.md` Part 24.7.

*By section 9:*
* H1 supports "a planner that learns how strongly and where brightening works, from its own forecast
  misses and using the model's gradients, beats the field's standard feedback controller when the
  strength is unknown".
* H2 says learning the strength is what the planner needs.
* H3 says that, without feedback, a design made at nominal strength fails under realistic strength
  uncertainty.
* The learning curve is a lower bound on how fast one Earth reveals the strength, because the
  planners know the current state exactly (§11).

### Amendment 9, revision 3 — Experiments 3c (the planner cannot see the weather) and 3d (the fixed land model) frozen in full (frozen 2026-10-11, BEFORE any evaluation run of Experiment 3c and any evaluation reference, run or score of Experiment 3d)

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
* *Smoke tests of both evaluation scripts* (3c on 2026-10-09, 3d on 2026-10-11). Each ran every
  arm's exact command on one training state (two segments, one member), then the registered analysis
  in its smoke mode. They check that the commands run end to end; their scores are not used.
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
    bounds [0.2, 5]. Its noise level, 0.04861 K, comes from 3b's rule (`analyze_exp3b_pilot.py
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

**A8. Pilot and power.** Pilot (3b's 16 pilot states, 3 members; arms without the suffix are 3b's pilot runs): mean `J_zonal` `plan_learn` 0.00370, `plan_naive` 0.00397, `plan_naive_rs` 0.00459, `plan_oracle` 0.00471, `plan_learn_rs` 0.00484, `plan_oracle_rs` 0.00499, `pi` 0.00643, `adaptive` 0.00675, `fixed` 0.01348, `uncontrolled` 0.02027. Learner noise level (3b's rule on the ocean-only oracle): 0.04861 K from 576 forecast misses (exact sensing in 3b: 0.00067 K). Pilot effects (training states, descriptive): H1 learn_rs vs pi -25% (t p 0.069); H2 learn_rs vs naive_rs +5% (t p 0.73); H3 learn_rs vs learn (exact) +31% (t p 0.088); naive_rs vs pi -29% (t p 0.025); oracle_rs vs naive_rs +9% (t p 0.53); learn_rs vs oracle_rs -3% (t p 0.8). Learning curve of the ocean-only learner (mean |log(estimate / truth)| per band, days 0, 56, 98, 168): 0.58, 0.57, 0.56, 0.50. Power for H1 at the fixed 6 members (3b's method, 16 macro states x 3 branches, between-macro spread 0.0005 added): minimum detectable effect 0.00125 (19% of `pi`'s pilot score). Members stay at 6 so that 3b's runs of the reused arms stay valid; the power is recorded, not used to choose.

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

**B3. Base climate** (revisions 0.2–0.3: procedure and gate unchanged). The Q-flux was diagnosed in the fixed-land model (cold start, SST restored to the climatology, 365 + 1460 days; `mcb_experiments_gpu/exp3d/qflux/qflux_monthly_t30.nc`), then settled for 3650 days with the free slab and scored by the registered gate (G1 |drift| < 0.02 K per 60 days, G2 |mean SST - observed| < 0.5 K). settle1: drift -0.0115 K per 60 d, bias -1.289 K -> FAIL; settle2: drift +0.0087 K per 60 d, bias -0.361 K -> PASS. The first settle failed G2, so the one registered Newton correction was applied (revision 0.3) and the corrected Q-flux settled again, as happened for the original base climate in 2026-09. The settled carry of settle2 is the base of the new states.

**B4. States and strengths.**
* *States:* `run_generate_macro_ics.py --plan exp3d` makes 32 macro states (100–131), continued from
  the settled carry every 365 days. Even states train (`exp3d_train`, branches 0–1); odd states
  evaluate (`exp3d_eval`, branches 0–2). Diagnostics: ocean-mean SST 288.99-289.92 K; trend -0.235 +- 0.024 K per decade; evaluation minus training -0.021 K.
* *Hidden strengths:* 3b's rule and distribution (`jcm.mcb.hidden_strength`). Table:
  `exp3d/hidden_strength.json`, SHA-256 bb00df59d405e00c5ca357b214508d4fe9c5cad42290d8bcfd788301364f8cc8.

**B5. Training side** (3b's procedure, on the fixed-land training states).
* references with 5 members;
* one-band response runs on the 16 branch-0 states, which give PI's and the adaptive law's
  sensitivities;
* the fixed design (ladder rung 4, zonal, days 98–182): [0.0285, 0.0208, 0.0120, 0.0153, 0.0206].

**B6. Arms.** 3b's, without the oracle:
* `plan_learn`, with the noise level 0.00069 K from 3b's rule applied to the oracle pilot on the 16
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
  * the designs: `exp3c/registered_design.json` (SHA-256 098571f6834429b6128cd80bf14f5bb25e8eba15c225f5de8505f71eb2867d2a) and
    `exp3d/registered_design.json` (SHA-256 7e2782406001ee52f00488d18ce55e81fdb8e9ed4f039e85d28d86fcd0dcb2cd), with their pilot records.
* *Cost:* about 28 GPU-hours on the GX10 (both evaluations together; about the same in wall time, since they share the GPU).
* *Posting:* this revision is posted to OSF (project pqabf) before any evaluation run of 3c and
  any evaluation reference of 3d.
