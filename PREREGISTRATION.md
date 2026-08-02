# MCB Controller — Pre-Registration (frozen before the post-fix GPU campaign)

**Purpose.** The 2026-07-12 audit found every stage-level verdict was decided inside the project's own
noise floor, on n≤2 held-out ICs, with gates and bands written *after* the results they judged (a
garden of forking paths). This document freezes the analysis plan **before** the post-fix regeneration
+ retraining campaign, so every verdict is pre-committed. Edits after the first GPU run of this
campaign must be logged as amendments with justification, not silent changes.

Frozen: 2026-07-13. Applies to: all runs on the post-fix code (R1–R7 fixes + P2 rebuild). Nothing
generated before 2026-07-13 is comparable (artifacts are stale — see the plan's R2/P1 notes).

---

## 1. Fixed experimental configuration

- **Physics:** cloud-top-albedo MCB (R6), slab ocean + **prognostic slab land** (R7a), sea/land flux
  split (R7b), climatology ocean init (R5). Realistic T30 terrain.
- **Horizon:** 60-day rollout, 2×30-day control intervals (the proven-trainable horizon; 180-day is
  known to explode).
- **Precision:** decided by P0.4 (x64 vs float32), settled from the noise-floor A/B **before** training
  — whichever is chosen is then fixed for the whole campaign.
- **Equilibration:** ICs are sampled only from a coupled state spun to `|60-day ocean drift| < 0.02 K`
  (`run_equilibrate.py`). If that criterion is not met, the campaign does not proceed — no training on
  a transient.

## 2. Initial conditions (fixes R5 / experimental-design findings)

- **N ≥ 10 training ICs and N ≥ 10 held-out ICs**, each an **independent trajectory** — branch the
  equilibrated state with distinct perturbation seeds, then spin each independently. NOT snapshots of
  one run (the fatal flaw of the old {0,45,90,…}-day scheme).
- **No temporal overlap:** no held-out 60-day evaluation window may share simulated days with any
  training window.
- **Seeds:** ≥ 3 policy-init seeds per configuration; every reported number is mean ± s.e. over seeds.
  Seeds and the IC-generation seed set are recorded in each checkpoint's metadata.

## 3. Metrics (all measured on the post-fix, equilibrated model)

- **Cooling:** per-IC day-60 dSST = area-(cos-lat)-weighted, ocean-masked SST change vs the **paired
  no-MCB baseline regenerated with the same code** (never a cached baseline). Report the **time-mean
  over the final 10 days**, not a single day-60 snapshot (lower variance; slab SST is inertial).
- **Teleconnection (only if the slab-land teleconnection is shown to be resolvable):** per-region
  **interval-mean** (not instantaneous) precipitation change vs baseline, in physical units
  (mm/day), with the `softplus(0)` offset removed. If a control run shows the region-precip metric is
  statistically indistinguishable from the do-nothing constant, this gate is **withdrawn**, not
  reweighted (R4).
- **Effort:** ocean-mean MCB forcing and resulting TOA forcing (W/m²), reported for sanity vs the
  published −1…−5 W/m² regional MCB range.

## 4. Gates — ALL control-relative and significance-tested (fixes the coin-flip finding)

The comparator is the **regenerated stage1-static pattern** evaluated on the **identical** ICs. No
gate is a bare band on an n≤2 mean.

| Gate | Statistic | PASS rule |
|---|---|---|
| G1 Terrain | \|truncated orography\| | > 0 (binary) |
| G2 Cooling | per-IC dSST | held-out mean within [−0.12, −0.08] K **and** the CI half-width < the band half-width (else "underpowered", not PASS) |
| G3 Controller adds value | paired per-IC (policy − static) dSST error vs target | mean improvement > **2·s.e.** (paired t / Wilcoxon, α=0.05 pre-set) |
| G4 Generalization | paired per-IC (policy − static) held-out **loss** | mean ≤ 0 at **2·s.e.** (policy no worse than static on unseen ICs) |
| G5 Teleconnection | paired per-IC region-precip change | only if §3 shows it resolvable; else WITHDRAWN |

## 5. Decision rules (from the measured noise floor, fixes R2/R3)

- **"Learned" requires** the training-loss improvement to exceed **2·σ_compile** (measured by
  `run_noise_floor.py` at the campaign's precision/horizon). A non-improving noisy loss curve is
  reported as **no learning**, never as a "best epoch N".
- **Checkpoint selection + early stopping on HELD-OUT loss**, evaluated every epoch — not training
  loss (fixes the selection error). `best_params` must reproduce its recorded loss (off-by-one fixed).
- **Any gate margin below 2·s.e. is reported as "not significant / underpowered"**, never as PASS or
  FAIL. Cross-run comparisons over *different* IC sets are forbidden (the Option-A refutation error).
- **Report all runs**, including failures, against this fixed plan. No metric is chosen after seeing
  the result.

## 6. Ablations that gate the central claim (run before any "it works" statement)

1. **Open-loop control:** a time-only policy (no climate-state features). If it matches the
   state-dependent policy at 2·s.e., the "feedback controller" contribution is unsupported — report as
   such. (The current setup exercises feedback exactly once per 2-interval rollout; increase the number
   of control intervals so feedback is actually tested.)
2. **No warm-start:** train from random init. The trained policy must beat its own warm-start
   initialization on held-out ICs by > 2·s.e. to claim the NN adds anything over the static pattern.

---

## AMENDMENTS

### Amendment 1 — retroactive disclosure of deviations in campaigns v1-v3 (logged 2026-07-30)

The 2026-07-29 meta-audit (`MCB_META_AUDIT.md`) found the following unlogged deviations from the
frozen plan. They are recorded here as the amendment rule requires; none is retroactively "approved".

1. **Seeds (§2, violated by v1, v2, v3):** every arm ran exactly one policy-init seed (hardcoded 42 /
   PRNGKey(0)); no seed was recorded in checkpoint metadata; no number was reported mean±s.e. over
   seeds. All v3 controller-level conclusions are single-training-realization statements.
2. **Metric (§3, violated everywhere):** the registered cooling metric (time-mean over the final 10
   days) was never implemented; every training objective, gate, noise floor, and ablation used the
   forbidden day-60 snapshot.
3. **Tests (§4, violated):** neither the paired t nor Wilcoxon was implemented; a bare 2·s.e. rule
   was used (anti-conservative at n=10, alpha ≈ 7.7%). Under the registered tests, v1's "G3 PASS —
   controller beats static" was never significant (paired t p=0.0615, Wilcoxon p=0.0645).
4. **G4 verdict (§5, violated):** within-noise margins were labeled "PASS (not sig. worse)" instead
   of "underpowered".
5. **G5 (§3, violated):** the teleconnection gate was dropped without the required resolvability
   control or formal withdrawal, and the one regional-precip comparison computed in v3 (legacy gate)
   FAILED and went unreported, breaching §5's report-all-runs rule.
6. **Configuration changes without amendment:** max_perturbation 0.15→0.09 (v2); loss_mode
   summed→terminal_dsst (v3; permitted on the letter of the plan, unlogged); control interval 15 d
   vs §1's "2×30-day"; the §1 x64-vs-f32 A/B never ran (float32 by default; blocked by a lax.scan
   carry-dtype mismatch); v3 reused v2's cached ICs/baselines (§3 requires regenerated baselines).
7. **Noise floor (§5):** sigma_compile was measured with in-process recompiled reps, which are
   bitwise identical on the GPU — 0.0 by construction. Cross-process runs of the identical
   computation differ by ~0.014-0.017 K per IC (chaos amplification of compile-level differences),
   so the "learned > 2·sigma_compile" rule was vacuous and per-run noise was unmodeled.
8. **Held-out exhaustion (design flaw in this plan):** §5 mandates held-out model selection while §4
   gates on the same held-out set, with no third split. Across v1/v2/v3 the same 10 held-out ICs
   (seeds 1010-1019) absorbed 13 gate tests, 5 gradient-probe looks, and ~123 per-epoch selection
   looks. **Those 10 ICs are retired for all confirmatory use.**

Consequences adopted: the v3 "held-out RMS halved (0.020→0.011 K)" claim is RETRACTED (min-of-39
selection artifact; does not replicate on re-evaluation); v3's G2 PASS and G4 PASS are downgraded to
exploratory; the null verdicts (G3 tie, feedback ≈ open-loop, warm-start ≈ random) stand — nulls are
not manufactured by forking paths — but are scoped to a single-season, single-ocean-state,
weather-noise-only task distribution in which the feedback-headroom ceiling (~0.006 K) was below the
detection floor by design.

### Amendment 2 — confirmatory protocol (frozen 2026-07-30, BEFORE any new GPU campaign)

All future confirmatory claims use this protocol. Changes after the first confirmatory GPU run must
be logged here as further amendments.

1. **Fresh ICs, third-split discipline.** A new independent-trajectory IC set is generated with
   `run_generate_ics_independent.py` using a previously unused seed range (`--seed0 3000`+). ICs used
   for any model selection or exploratory look are never used for confirmatory gates. ICs seeds
   1000-1019 are retired from confirmatory use.
2. **Registered metric, implemented.** The cooling metric is the area-weighted, ocean-masked dSST
   time-mean over the final 10 days vs the paired baseline (`final_sst_change_10d`,
   `evaluate_coupled_policy(tail_mean_days=10)`).
3. **Micro-ensembles.** Each (arm, IC) is evaluated as the mean over k=8 members (member 0
   unperturbed; members 1-7 seeded 0.001 K SST perturbations). Every member's paired baseline is
   regenerated IN THE SAME PROCESS as the policy runs (`run_confirmatory_eval.py`); cached baselines
   are never used for confirmatory numbers.
4. **Tests.** Primary rule remains the 2·s.e. margin for continuity, but every gate also reports the
   paired t and exact Wilcoxon p-values at the registered alpha = 0.05, and verdicts must be
   consistent with the paired t; disagreements are reported, not resolved silently. Within-noise
   results are reported as "underpowered", never PASS/FAIL, together with the TOST
   `equivalence_bound_95` (the demonstrable |effect| bound), which converts a powered null into a
   bounded-equivalence claim.
5. **Primary comparisons (pre-specified, Holm-corrected as a family of two):**
   (a) trained controller vs stage1-static on |dSST_10d − target|;
   (b) trained controller vs time-only open-loop on the same metric.
   All other comparisons are secondary/exploratory and labeled as such.
6. **Seeds.** Any training-based arm requires ≥3 policy-init seeds (`--seed`, recorded in checkpoint
   metadata); arm-level numbers are reported per-seed AND pooled mean ± s.e. over seeds.
7. **Noise floor.** The per-run noise is measured with `run_noise_floor.py --cross-process`
   (fresh-process reps), on the SAME split the gates use, before any gate is interpreted.
8. **Effort metric.** Ocean-mean MCB albedo forcing and its TOA W/m² equivalent are reported per §3.
9. **Teleconnection metric.** Regional precip is reported in physical units (mm/day,
   `region_precip_change_mm_day`, offset-free); the G5 gate stays WITHDRAWN unless the §3
   resolvability control is run and passes, and all computed regional-precip comparisons are
   reported regardless of outcome.

### Amendment 3 — Tier-2 experiment: feedback under uncertain MCB efficacy (frozen 2026-07-31, BEFORE any Tier-2 GPU run)

**Motivation.** The Tier-1 confirmatory campaign closed the original feedback question with a bounded
null (any feedback effect within ±4.5 mK at 95%), and the design-headroom analysis showed the task
gave feedback nothing to correct (perfect-controller ceiling ~6 mK). Tier-2 tests feedback where it
has real, physically motivated headroom: **per-episode uncertain seeding efficacy** — the dominant
real-world MCB uncertainty, and the classic argument for feedback control of SRM (cf. the
Kravitz/MacMartin explicit-feedback SAI literature).

**Mechanism (implemented, tested).** Each episode draws an unobserved efficacy
η ~ Uniform[0.6, 1.4]; the applied cloud-albedo perturbation is η × the (cap-clipped) command.
Policies never observe η directly; feedback arms can infer it from the realized cooling in the
paired-anomaly features. A static pattern's expected gate error is E|η−1| × 0.1 K ≈ 20 mK — ~13×
the Tier-1 detection floor.

**Splits (all fresh; 3000–3019 are now spent by Tier-1 gates):**
train seeds 4000-4009 (n=10), validation-for-selection seeds 4010-4015 (n=6, the held-out
split of the same generation run; used for early stopping /
checkpoint selection only), confirmatory eval seed0=5000 (n=20, touched exactly once by the final
gates). Eval-time η draws use seed 920; training-time draws seed 910+run-seed; the eval η stream is
never used in training.

**Arms.** (1) stage1-static (η-blind); (2) open-loop time-only, trained under η-randomization,
3 seeds; (3) NN feedback (fc13), trained under η-randomization (domain randomization, warm-started
from the static pattern), 3 seeds; (4) PI: hand-designed deadbeat efficacy compensator on the static
pattern (no training; analytically verified to recover the target under η≠1 in the linear test
model; cap saturation limits its authority — a reportable property, not a bug).

**Metric & mechanics.** Registered final-10-day time-mean dSST; k=4 micro-ensemble members per
(arm, IC) with per-member in-process baselines; η shared across arms and members of an IC (exactly
paired); 60-day horizon, 15-day control intervals, cap 0.09, target −0.1 K.

**Manipulation check (gates the expensive steps).** Before any training, the static arm is evaluated
under η-randomization on the validation ICs: its mean |dSST_10d − target| must exceed 3× its Tier-1
(η=1) value. If not, the disturbance is too weak to matter and the campaign stops for redesign —
reported either way.

**Hypotheses (primary family, Holm-corrected, α=0.05, paired t on per-IC |dSST_10d − target|,
pooled over seeds with per-seed values reported):**
- H2: NN feedback beats static.
- H3: NN feedback beats the trained open-loop schedule.
**Secondary (reported, not gated):** NN vs PI (does learning beat the hand-designed compensator?);
PI vs static; per-seed spread; G2 band per arm; TOST equivalence bounds for any null; precip
mm/day per region.

**Decision rules.** Verdicts follow the paired t at α=0.05 (the 2·s.e. rule is reported alongside;
disagreements reported). Any null is reported with its TOST equivalence bound. All runs reported,
including failures and the manipulation check. Seeds, η draws, and IC seeds recorded in artifacts.

**Amendment 3 revisions (2026-07-31, from the pre-launch adversarial review; still BEFORE any
Tier-2 GPU run, so the freeze is intact):**
1. **Training objective = tail_dsst** (squared error of the final-10-day time-mean dSST — the
   registered gate metric exactly), not the day-60 terminal snapshot: in the measured ramp-like
   response regime (Tier-1 tail/terminal ratio 0.932 ≈ the linear-ramp 0.925), a terminal-targeting
   controller systematically under-corrects the tail metric by up to ~half the available effect.
2. **Training efficacies are redrawn every epoch** (epoch-seeded); with one fixed η per IC, the
   absolute-SST features fingerprint the IC and memorizing the IC→η map strictly dominates learning
   the feedback law (10 draws/seed cannot identify U[0.6,1.4]). Validation efficacies remain FIXED
   (selection metric comparable across epochs) and are drawn ANTITHETICALLY (pairs x, 2−x) so both
   η directions are covered during checkpoint selection.
3. **Eval efficacies are antithetic** (10 draws + mirrors, seed 920): the plain draw was skewed
   (mean 1.07, 13/20 above 1), flattering the PI arm (which has downward-only authority on the
   bang-bang pattern) and under-exercising the NN's low-η direction.
4. **Training baselines are regenerated in-process** (`--regen-baselines`): cached cross-program
   baselines carry ~0.01 K mismatch — the same order as the η signal the anomaly features must carry.
5. **Manipulation-check threshold = 0.0162 K** (3× Tier-1, as originally frozen; the campaign
   script briefly said 0.012 — reconciled to the frozen value).
6. **Primary analysis pooling rule pinned and pre-committed** (`analyze_tier2.py`, unit-tested):
   the unit of analysis is the IC (n=20); an arm-group's per-IC error is the mean over its 3 seed
   arms of |dSST_10d − target|; H2/H3 are paired t over ICs, Holm-corrected; significance requires
   the NN-favorable direction. Concatenating (IC, seed) pairs is forbidden (pseudo-replication).
7. **Widened-deployment efficiency probe** added to the pre-training step (uniform ocean fields at
   0.0265 and 0.0442 mean forcing, η=1, validation ICs): measures the cooling efficiency of cells
   outside the optimized pattern — the mechanism the NN needs for low-η compensation. Reported,
   ungated.
8. **PI reference calibration recorded:** the linear-ramp reference is validated by the Tier-1
   measurement (tail/terminal = 0.932 ± 0.071 vs ramp prediction 0.925); PI's cap-limited upward
   authority on the bang-bang pattern (η<1 episodes) is a pre-registered reportable property.
   Expected effects on the realized antithetic draws: static mean|err| ~20 mK; a perfect
   compensator leaves ~2 mK; detection floor ~2 mK (k=4, n=20).
