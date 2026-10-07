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

### Amendment 4 — Tier-2b: PI-imitation initialization (frozen 2026-08-02, BEFORE any Tier-2b GPU run)

**Motivation.** Tier-2 (Addendum 2 of MCB_META_AUDIT.md) showed classical deadbeat feedback halves
efficacy-uncertainty error (10.1 vs 20.2 mK) while BPTT-trained NNs stay at static level — a verified
TRAINING failure (flat losses at the static-under-η floor, gradient coherence ~0.45), not an
information or authority limit. Tier-2b tests the diagnosis's remedy: initialize the NN by
supervised imitation of the PI law (a noiseless regression onto the same features), then fine-tune
with BPTT. The open scientific question: can a learned policy EXCEED PI by widening deployment on
low-η episodes, where the cap blocks PI's pattern-scaling but the widening probe measured flat
cooling efficiency (−3.4 to −3.6 K per unit forcing)?

**Imitation (run_pi_imitation.py).** Target = clip(pattern × pi_gain(f0, f10), 0, cap) over synthetic
feature batches covering f0 (realized dSST) ∈ [−0.25, 0.05], f10 (time fraction) ∈ {0, .25, .5, .75},
other features sampled broadly (teaches initial invariance); MSE, Adam; 3 init seeds (52, 53, 54);
fidelity acceptance: mean |NN − PI| output error < 0.002 albedo on a held-out feature grid, and a
FakeCoupler closed-loop check within 1 mK of PI.

**Gate (before fine-tuning spend).** Imitation-only policy evaluated under η-randomization on the
Tier-2 validation ICs (seeds 4010–4015, k=2): mean |dSST_10d − target| must be < 15 mK (PI reference
~10–12; static ~20). Failure = distillation failed; stop and report.

**Fine-tuning.** run_stage5_training from each imitation checkpoint: loss_mode=tail_dsst,
select-on-heldout (validation = 4010–4015, antithetic η), per-epoch η redraws, in-process baselines,
learning-rate 1e-3 (10× lower than Tier-2: protect the init), 30 epochs, patience 15, cap 0.09,
CI 15, 60 d. Training ICs = seeds 4000–4009 (reused; training-data reuse is permitted).

**Evaluation.** FRESH confirmatory ICs seed0=6000 (n=20; the 5000s are spent by Tier-2 gates); FRESH
antithetic η stream seed 930; k=4 members, in-process member baselines; registered 10-day tail
metric; arms: static, pi, imitation_s52 (no fine-tune), finetuned_s52/s53/s54.

**Hypotheses (primary family, Holm, α=0.05, paired t on per-IC |dSST_10d − target|, pooled = per-IC
mean over the 3 fine-tuned seeds; significance requires the NN-favorable direction;
analyze_tier2b.py pre-committed):**
- H4: fine-tuned NN (pooled) beats PI.
- H5: fine-tuned NN (pooled) beats static.
**Secondary (reported, ungated):** imitation-only vs PI (TOST equivalence: did distillation
transfer?); fine-tuned vs imitation-only (did fine-tuning add anything?); η<1-stratified fine-tuned
vs PI (the widening mechanism must concentrate gains there); per-seed values; G2 band; forcing-vs-η
response correlations.

**Decision rules.** As Amendment 3 (paired t governs; TOST bounds for nulls; all runs reported).
Interpretation grid pre-stated: H4 sig + η<1-concentrated = learning exceeds hand design via
widening; H4 null + imitation≈PI = distillation transfers but gradients add nothing (chaos-gradient
bottleneck persists even from a good basin); imitation-only ≉ PI = distillation itself fails
(feature-space mismatch between synthetic and rollout distributions).

**Amendment 4, revision 1 (2026-08-02, after the imitation gate fired; BEFORE any fine-tuning or
confirmatory-eval spend — the seed-6000 eval set was generated but never evaluated on):** the first
campaign attempt stopped at the pre-registered imitation gate (static 19.8 / PI 12.1 / imitation-only
19.4 mK — distillation did not transfer; the net behaved exactly like static). Root cause, measured:
the real model's absolute-SST features sit at f11 ≈ −1.83 and f12 ≈ −4.23, i.e. −3.7σ / −8.5σ outside
the synthetic sampling (N(0, 0.5)), where the MLP extrapolated to a frozen gain ≈ 1. Fix (pre-launch
of attempt 2): (a) nuisance-feature sampling anchored at the measured operating point with generous
widths (f11 ± 1.5, f12 ± 2.0; anomaly features widened to cover mid-rollout scales incl. heat flux
± 5σ); (b) distillation trains on standardized features with the standardization FOLDED into the
first-layer weights afterwards (mathematically identical network on raw features; required because
raw-scale nuisance inputs made the wide-range regression unoptimizable); (c) new regression test
pins fidelity at the measured operating point. The PI-vs-static replication observed at the gate
(12.1 vs 19.8 mK on the validation ICs, fresh η stream) is noted as corroborating Tier-2's headline
on an independent draw. Gate threshold, hypotheses, splits, and η streams unchanged.

### Amendment 5 — retrospective precipitation side-effect analysis (logged 2026-08-07; exploratory,
no new GPU data)

**What.** `run_confirmatory_eval.py` recorded per-(IC, member, arm) Amazon and Sahel precipitation
changes (`amazon_mm_day` / `sahel_mm_day`, day-60 snapshot vs the paired baseline) in every
Tier-1/2/2b campaign, but nothing ever aggregated or reported them (meta-audit §2.4). This
amendment logs their first analysis (`analyze_precip_sideeffects.py` + tests, committed with this
amendment; JSON output `precip_sideeffects.json`).

**Status: exploratory, not confirmatory.** (a) The recorded quantity is a single-day snapshot; the
§3-registered metric is an interval mean, which was never implemented in the recording path — so G5
cannot be scored as registered from this data. (b) The eval ICs were already consumed by each
campaign's primary analyses; this is a different variable on the same runs. No confirmatory claim
is made from this analysis.

**Protocol (fixed before results were seen — the analysis script is the protocol).** Per-IC pooling
identical to the frozen Tier-2/2b analyses (mean over members, then over seeds); paired t + exact
Wilcoxon + TOST equivalence bounds from `jcm.mcb.gates_stats`; two Holm families (all 24 one-sample
arm-vs-zero tests; all 16 arm-vs-static comparisons); per-run precip chaos sd estimated from the
within-(arm, IC) micro-ensemble spread.

**§3 resolvability verdict: NOT RESOLVABLE — G5 remains WITHDRAWN.** In every campaign and both
regions the static arm is statistically indistinguishable from the do-nothing constant (0 mm/day)
after Holm (min raw p = 0.044 → Holm p = 0.965). Equivalence bounds govern instead: any true
regional effect of any arm is within |0.56–1.21| mm/day (Amazon) and |0.06–0.21| mm/day (Sahel) at
95%. Measured per-run snapshot chaos sd: 2.7–3.3 mm/day (Amazon), 0.4–0.8 mm/day (Sahel).

**Breach closure (Amendment-1 scope).** The v3 legacy Gate 3 ("stage5 weighted region-precip
penalty ≤ stage4-warmstart, bare means over all 20 ICs, no significance test, R4-floored softplus
metric") was computed FALSE (1.23e-3 vs 1.07e-3) and went unreported. Now reported. Honest
reanalysis: the paired difference is +0.0033 ± 0.0049 (p_t = 0.51, p_w = 0.50) — a coin flip inside
noise on a metric the 2026-07-12 audit had already deprecated; on held-out ICs alone stage5 is
numerically (not significantly) BETTER on both regions.

**Known caveats, disclosed.** Region masks are plain lat-lon boxes not intersected with the land
mask (`jcm/mcb/mcb_regions.py`), so "Amazon" includes some Atlantic cells at T31; episodes span
Jan–Feb (Amazon wet season; Sahel dry season, which partly explains the tight Sahel bounds).

**Forward rule (binding on any future precipitation gate).** Before any controller comparison on
precipitation may be scored: (1) implement and record the §3 interval-mean metric; (2) measure the
precip noise floor at the experiment's horizon; (3) intersect the region masks with the land mask.

### Pre-amendment note — ENSO scoping run (logged 2026-08-08; exploratory calibration, no gates)

Next experiment: closed-loop MCB feedback against imposed ENSO variability (the 2026-08-03
verified direction; no published study closes this loop — Lee et al. 2025 GRL is feedforward+PI
against warming only). Before any design freeze, `run_enso_scoping.py` measures the two unknowns
the design depends on: (a) this model's GMST-per-Nino3.4 sensitivity (no published SPEEDY+slab
number; obs reference ~0.11 K/K at ~3-month lag, Trenberth et al. 2002) and (b) the long-horizon
chaos noise floor from control-member spread. Mechanism: Molteni-2024-style pacemaker —
SST relaxation (tau = 5 d) toward a commanded step anomaly (+/-2 K, 30-day ramp) inside a tapered
Nino3.4 mask, relaxing against a paired no-ENSO control trajectory (relaxation, not q-flux, per
Dommenget 2010 slab-amplification; `jcm/mcb/enso.py`, phase driven by sim_time). This run is
EXPLORATORY instrument calibration (like the Tier-1 noise floor): no gates, no hypotheses, and
its ICs/outputs will not be reused for any confirmatory claim. The ENSO experiment itself will be
frozen as a numbered amendment (arms, hypotheses, splits, n x k, horizon) BEFORE any gated GPU
campaign, informed by these calibration numbers.

### Amendment 6 — ENSO feedback experiment (frozen 2026-08-08, BEFORE any gated ENSO GPU campaign)

**Question.** Can a feedback controller demonstrably cancel the global temperature effect of imposed
ENSO variability while delivering the MCB cooling target — the loop no published study has closed
(Lee et al. 2025 GRL is feedforward+PI against warming; Wan 2026 / Xing 2025 are open-loop)?

**Design (all constants below are calibration-measured, 2026-08-08 scoping runs; exploratory
pickles `enso_scoping.pkl`, `mcb_authority_scoping.pkl`).**
- Episodes: **180 days**, control interval 15 d (12 decisions), daily coupling. Registered metric:
  **final-60-day time-mean global-ocean dSST vs the member's NO-ENSO no-MCB baseline**; target
  −0.1 K. The baseline (regenerated in-process per member) doubles as the pacemaker's relaxation
  reference. Measured at this horizon/window: El Nino(2 K) GMST effect **+105 mK**, chaos floor
  **13 mK**, actuator ceiling (uniform cap 0.09) **−0.64 K** (3× the required −0.21 K; the cap
  will not bind), dose linearity 1.05.
- Disturbance: pacemaker El Nino (`jcm/mcb/enso.py`: relaxation, tau 5 d, tapered Nino3.4 mask,
  30-day ramp then hold; phase from sim_time), applied to ARM rollouts only, never the baseline.
  Hidden per-IC amplitude **A ~ U[0.5, 2.0] K, seed 940, antithetic**; shared across arms and
  members (paired). **La Nina excluded**: measured response is weak and non-monotonic
  (−0.026 ± 0.040 K at −2 K, vs +0.242 ± 0.040 K at +2 K) — an asymmetry reported as a finding.
- Arms: **static** (Tier-1 pattern × **0.3668** = 0.10/0.2726, the measured 180-d rescale);
  **ffmean** (open-loop schedule compensating the MEAN amplitude 1.25 K — the Kravitz-2014-style
  mis-specified feedforward, the fair non-feedback control); **pienso** (classical feedforward +
  proportional law, `make_enso_pi_policy_fn`, enso_effect_per_K = 0.0525 measured); **imitation_s62/
  s63/s64** (fc14 MLP distilled from the pienso law — 3 seeds per the ≥3-seed rule; feature 13 =
  paired Nino3.4 box anomaly). **No BPTT anywhere**: gradients through 180-d rollouts diverge
  (grad norms 1e7–1e13 at 180 d, meta-audit) — a pre-stated exclusion, and Tier-2b showed the
  imitation pipeline needs none.
- ICs: fresh set seed0 **7000** (6 train + 20 held-out, decorr 30 d, horizon 180). Train split
  only for gates/probes; held-out only for the single final evaluation. k = **4** members
  (member-seed0 77000). Efficacy uncertainty OFF (one axis at a time; the eta × ENSO composition
  is a separate future amendment).
- Gates (each stops the campaign, in order): **G-cal** static without ENSO on train ICs lands in
  [−0.12, −0.08]; **G-manip** static under fixed A = 2.0 misses by > 0.05 K; **G-imit**
  distillation fidelity mean|err| < 0.002 albedo (per Amendment 4) AND on train ICs (amp seed 941,
  a stream never reused) the imitation arm's mean miss ≤ 1.5× pienso's.
- Hypotheses and decision rules are the pre-committed `analyze_enso.py` (+ tests), verbatim:
  primary **H6 pienso-vs-static** and **H7 imitation-vs-static** (Holm over the pair, paired t
  governs, direction-gated, Wilcoxon co-reported); secondary imitation-vs-pienso (+TOST),
  pienso-vs-ffmean, ffmean-vs-static, per-seed, amplitude-stratified H7. All runs reported.
- Power: expected static mean miss ≈ 0.0525 × E[A] ≈ 66 mK vs a ~4–5 mK paired comparison s.e.
  (per-run paired sd ~33 mK, k=4, n=20) — the design is deliberately overpowered; the interesting
  quantities are the residual misses and the imitation-vs-pienso equivalence bound.
- Cost: ~3.5 GPU-h (`run_campaign_enso.sh`, gated and resumable).

### Amendment 6 POST-MORTEM (logged 2026-08-09, AFTER the campaign and its audit) — and the
### specification for the corrected re-run (Amendment 7, to be frozen before any new GPU spend)

The Amendment-6 campaign executed exactly as registered (verified: horizon, interval, metric, cap,
n=20 held-out ICs 6-25, k=4, arms, amplitude streams 940 eval / 941 gates, zero train-heldout
overlap, no BPTT, `analyze_enso.py` unmodified since the freeze). Its pre-registered primaries
came out at 8 sigma. A five-lens adversarial audit then showed **the primaries are degenerate**
(MCB_META_AUDIT.md Addendum 5). Four design errors, all mine, all logged here rather than quietly
fixed:

1. **La Nina was excluded on an off-horizon measurement.** The justification ("weak, non-monotonic
   cold response") came from the 365-day / tail-90 scoping numbers (+241 vs -27 mK). At the
   registered 180-day / tail-60 horizon the same scoping data show near-symmetry (+104.5 vs
   -98.7 mK). Excluding it made the disturbance one-sided, so its MEAN was non-zero, so a single
   retuned scalar gain on the ENSO-blind pattern absorbs it: a leave-one-out-tuned blind arm
   scores 31.4 mK vs pienso 30.5 (p = 0.85). H6/H7 could not have failed.
2. **Plant constants were measured on the wrong variable.** `enso_effect_per_K = 0.0525` came from
   atmospheric global-mean surface air temperature; the registered metric is global-ocean dSST,
   whose measured sensitivity is 0.0716 K/K (1.36x). This crippled the ffmean control arm
   (correctly specified it scores 27.5 not 52.7 mK, and pienso's advantage becomes +3.0 mK,
   p = 0.58), and it is the likely source of the target shortfall.
3. **The control law's reference does not match the scored metric.** The deadbeat law regulates
   INSTANTANEOUS dSST onto a ramp reaching target at day 180; the metric averages days 121-180. A
   perfect tracker therefore scores -0.0836 K — a structural +16.4 mK miss (observed A-independent
   miss: pienso +16.6 mK). Most of the "no arm hits the target" outcome is this, not physics.
4. **The registered endpoint cannot measure the phenomenon.** With an unreachable target and a
   one-sided disturbance, no per-IC value crosses the target, so |dSST - target| is exactly linear
   and the primaries reduce to "which arm cooled more". Disturbance rejection — the actual
   question — was never a registered endpoint.

Also to correct: the "chaos floor 13 mK" in Amendment 6 was a 3-dof estimate from one base IC; the
realized per-run figure is 35-40 mK (open-loop arms) / ~18 mK (feedback arms), so the power
calculation was optimistic by ~1.8x in s.e. And a gate (G-imit) already showed the target was
unreachable before the held-out run; the campaign proceeded anyway.

**Amendment 7 specification (binding on the re-run; freeze before any GPU spend):**
- **PRIMARY endpoint = disturbance sensitivity**, the slope of per-IC error on the hidden
  amplitude (mK per K of Nino3.4), tested as a paired slope difference against the blind
  comparator. It is invariant to any constant recalibration, so it cannot be gamed by retuning.
  Mean |miss| is demoted to a secondary, explicitly bias-contaminated, metric.
- **Zero-mean disturbance:** A ~ U[-2, +2] K (El Nino AND La Nina), antithetic. This removes the
  mean-offset degeneracy by construction.
- **Registered control arm: the retuned ENSO-blind constant gain**, tuned leave-one-IC-out. Any
  feedback claim must beat it.
- **Plant constants re-derived on the registered metric** (ocean dSST, >= 6 ICs, k >= 4), not on
  GMST and not from n=1 IC; ffmean rebuilt with the corrected constant.
- **Control-law reference matched to the scored window** (regulate the tail-60 mean, not the
  instantaneous ramp), so a perfect controller can actually score 0.
- **Gates:** k >= 4 (a k=2 gate produced a >120 mK outlier here); and the gate must require the
  ARM UNDER TEST, not only the comparator, to land in band.
- Everything else (fresh ICs, micro-ensembles, >= 3 seeds, held-out separation, no BPTT, Holm,
  TOST bounds, all runs reported) carries over unchanged.

### Amendment 7 — corrected ENSO experiment (frozen 2026-08-09, BEFORE any Amendment-7 GPU run)

Supersedes Amendment 6, whose four design errors are documented in the post-mortem above and in
MCB_META_AUDIT.md Addendum 5. Same scientific question — can feedback cancel the global-mean
temperature effect of imposed ENSO variability while delivering the MCB target? — with a design
that can actually answer it.

**PRIMARY ENDPOINT (changed): disturbance sensitivity.** The slope of each arm's signed per-IC
error on the hidden amplitude A, in mK per K of Nino3.4, tested as a PAIRED slope (regress the
per-IC difference arm-minus-static on A; test the slope against zero). Rationale: rescaling a
non-adaptive controller by any constant shifts its mean error but leaves its slope EXACTLY
unchanged (unit-tested), so this endpoint cannot be satisfied by retuning. Mean |dSST - target|
is retained only as an explicitly bias-contaminated secondary.

**Hypotheses** (Holm over the primary pair, direction-gated; paired t governs; Wilcoxon
co-reported; TOST bounds on nulls; all runs reported):
- **H8:** pienso's disturbance sensitivity is lower than static's.
- **H9:** the pooled imitation arms' sensitivity is lower than static's.
Secondary: each arm vs the blind control on BOTH endpoints; imitation vs pienso (slope and mean,
with equivalence bounds); per-seed; warm/cold stratification (pre-stated because the actuator
floor is one-sided).

**Registered control arm — the retuned ENSO-BLIND gain.** Computed, not simulated: the static arm
rescaled by one constant gain chosen leave-one-IC-out, under the physical model
dSST = g*mu + s*A + chaos (chaos does NOT scale with dose). This is the best any blind controller
could do, and it TIED the feedback arms on Amendment 6's endpoint. Any feedback claim must beat it.

**Disturbance: zero-mean.** A ~ U[-A_max, +A_max] (El Nino AND La Nina), antithetic, seed 950
(gates use 951). A_max = min(|target| / s, 2.0) with s measured in step 2 below — i.e. sized to the
actuator floor, since gain is clipped below at 0 and a cold anomaly beyond |target|/s cannot be
compensated even by spraying nothing. With the expected s this gives A_max ~ 1.4 K, a zero-mean
disturbance with ~1.9x the amplitude spread of Amendment 6 (better slope power). La Nina is
INCLUDED: the Amendment-6 exclusion rested on a 365 d / tail-90 measurement, whereas at this
campaign's 180 d / tail-60 horizon the scoping data show near-symmetry (+104.5 / -98.7 mK).

**Plant constants measured in METRIC SPACE** (`run_calibrate_enso_plant.py`, train ICs only,
k=4, on the registered final-60-day mean ocean dSST): mu = MCB authority per unit gain (sets the
pattern rescale = target/mu) and s = ENSO sensitivity (sets the law's feedforward gain). Amendment
6 used 0.0525 measured on atmospheric GMST from one base state; the campaign-measured ocean-dSST
value was 0.0716 (1.36x), which crippled its open-loop arm.

**Control-law reference matched to the scored window.** The law's ramp reference is scaled by
`tail_reference_scale(180, 60) = 1.1960` so a perfect tracker scores the target rather than
structurally undershooting by (1-f)|target| = 16.4 mK.

**Arms:** static (rescaled, ENSO-blind), pienso (corrected law), imitation_s72/s73/s74 (fc14
distilled from the corrected law). ffmean is DROPPED — with a zero-mean disturbance a
mean-feedforward schedule is identically the static arm, and its Amendment-6 role is taken by the
blind-gain control. No BPTT.

**ICs and gates:** fresh set seed0 8000 (6 train + 20 held-out, horizon 180), k=4 everywhere
(Amendment 6's k=2 gates produced a >120 mK outlier). Gates, in order: **G-cal** static with ENSO
off lands in [-0.12, -0.08]; **G-arm** — new, from the Amendment-6 lesson — the ARM UNDER TEST
(pienso) must itself land in band on train ICs before any held-out spend; distillation fidelity
< 0.002 albedo as in Amendment 4.

**Analysis:** the pre-committed `analyze_enso7.py` (+ tests), frozen with this amendment.
**Cost:** ~4 GPU-h (`run_campaign_enso7.sh`, gated and resumable).

### Amendment 7, revision 1 — the ENSO-blind outcome-feedback ablation (frozen 2026-08-10,
### AFTER the Amendment-7 eval but BEFORE the ablation arm is run)

The Amendment-7 campaign passed both primaries (H8/H9) and, unlike Amendment 6, its result is NOT
reproducible by rescaling a blind controller. But the adversarial audit identified that H8/H9 as
registered compare feedback against an OPEN-LOOP arm, and therefore test "feedback of any kind vs
no feedback" — NOT whether OBSERVING ENSO helps. Feature 0 (the realized global-ocean dSST anomaly
against the no-ENSO baseline) already carries the ENSO signal, so a controller with no Nino
observation can reject the disturbance through ordinary outcome feedback. Two independent surrogate
calculations (mine and the auditor's, agreeing) put such an arm at 12-18 mK/K, i.e. ~63-77%
rejection versus the observed 95%, and passing H8 at p ~ 1e-5. Registering that arm now, before
running it.

Two further audit findings are logged here and will be reported as corrections rather than
re-litigated: (a) the `blind_loo` control is slope-identical to static BY the invariance theorem
that motivates it, so it demonstrates the endpoint's non-gameability by rescaling but is NOT an
independent competitor — the "every feedback claim must beat it" framing is withdrawn; (b) the
slope residuals are strongly heteroscedastic (static's error sd grows 25 -> 71 mK across the
amplitude range while pienso's is flat at 9 -> 11), so OLS p-values are anti-conservative:
HC3/wild-bootstrap inference gives p ~ 5e-5 for H8/H9 rather than 2e-9/3.8e-8. All Amendment-7
p-values on the primary endpoint are hereby superseded by their wild-bootstrap values.

**New arm.** `blindfb` = the identical pi-enso law with its feedforward gain set to ZERO
(`--arm blindfb=pi-enso@0=<pattern>`), i.e. deadbeat outcome feedback on the realized dSST and time
only, with the Nino3.4 observation contributing nothing. It is otherwise byte-identical to the
pienso arm.

**Run design.** One new evaluation containing `static`, `pienso` and `blindfb` together, so all
three are paired WITHIN one process (avoiding the ~16 mK cross-process chaos offset), on the SAME
held-out ICs (indices 6-25 of ics_enso7), the SAME amplitude stream (seed 950, antithetic, range
+/-1.827) and the SAME member seeds as the Amendment-7 eval. Nothing else changes.

**Hypothesis H10 (pre-stated, direction-gated, wild-bootstrap inference):** the Nino3.4 observation
reduces disturbance sensitivity beyond outcome feedback alone — i.e. the paired slope
(pienso - blindfb) on A is negative. Secondary: blindfb vs static (does outcome feedback alone
reject?), and the same contrasts on mean |miss|, with TOST bounds if null.

**Interpretation grid, pre-stated.** (i) H10 significant => the ENSO observation carries genuine
incremental control value, and the headline may say the controller exploits its observation of the
disturbance. (ii) H10 null with a tight equivalence bound => outcome feedback explains the
rejection and the headline must be stated as "feedback rejects the disturbance; observing it adds
no measurable benefit at this horizon" — a cleaner and more surprising result than the original
framing, and one that materially changes what a deployment would need to measure. (iii) blindfb
not different from static => the surrogates are wrong and the Nino term is doing all the work.
All three outcomes are reportable; none is a failure.

**Cost:** ~1.5 GPU-h (3 arms x 20 ICs x k=4 at 180 d, baselines shared).

### Amendment 7, revision 2 — the ANTICIPATION ablation, with a tuned comparator
### (frozen 2026-08-10, SUPERSEDES revision 1, BEFORE any ablation GPU run)

Revision 1 is withdrawn UNRUN. A pre-run design review found that the ablation it specified would
have manufactured a win from a detuned comparator — the Amendment-6 error in mirror image — and
three further defects. All were reproduced independently before acting on them.

**Why revision 1 was withdrawn.** Its comparator `blindfb` was the pi-enso law with the
feedforward term deleted while the FEEDBACK gain stayed frozen at the value chosen when
feedforward was carrying the disturbance. A surrogate sweep of that single constant (transcribed
from `make_enso_pi_policy_fn`) shows a purely reactive controller improving from +12.5 mK/K at
fb_weight 1 to +1.8 at fb_weight 6 — i.e. retuning one number nearly closes the gap to the
anticipating law's +1.5. Testing against fb_weight = 1 would have produced a "significant" result
that says nothing except that we detuned the opponent.

**Three further corrections, all logged rather than silently fixed.**
1. *The signed slope is sign-degenerate.* Amendment 7's pienso slope of +2.6 mK/K is a
   CANCELLATION: hinge fit gives warm +11.2 and cold -11.0 (imitation +6.2 / -2.5). A signed
   contrast credits a controller for wrong-signed error. A sign-agnostic co-primary (RMS_A) and a
   mandatory hinge report are added. Measured RMS_A: static 52.6, pienso 16.3, imitation 12.0 mK —
   the rejection is real (69-77%) but smaller than "95%" implied.
2. *The committed inference did not exist.* `analyze_enso7.py` cannot score this campaign and the
   repo had no wild-bootstrap/HC3 code and no slope-TOST, so revision 1's promised inference was
   unimplemented. `analyze_enso8.py` (+ 10 tests) is frozen with this amendment.
3. *The proposed amplitude seed was lucky.* Seed 960 sits at the 99.3rd percentile of design-matrix
   spread over 5000 seeds (Saa 38.3 vs median 22.0). Randomness is removed entirely.

**Design.**
- **Amplitudes: DETERMINISTIC, no seed.** n/2 mirrored pairs at |A_j| = A_max*sqrt((j-0.5)/(n/2)),
  A_max = 1.8273 K (frozen from `enso7_plant.json`, not recomputed). Exactly zero mean, no draw
  luck, Saa = 40.07 vs a median random draw's 22.0 (slope s.e. x0.74 for free). Sign-symmetric, so
  the pooled estimand is comparable with Amendment 7.
- **n = 24 held-out ICs (12 mirrored pairs) + 6 train, fresh set seed0 9000, horizon 180, k = 4.**
  Fresh because the Amendment-7 held-out ICs have now carried ~22 reported statistics, and because
  fresh ICs additionally give an INDEPENDENT REPLICATION of H8/H9.
- **fb_weight tuning, train ICs only, before any held-out rollout:** grid
  fb in {1, 1.5, 2, 3, 4, 6} for BOTH the reactive law (ff=0) and the anticipating law (ff=1),
  k = 2, 6 train ICs. Selection: minimise RMS_A; ties to the SMALLER fb. Both selected values are
  frozen and the full sweep is reported. Tuning only the comparator would make superiority
  conservative but equivalence anti-conservative, and equivalence is the likely landing zone.
- **Arms:** `static`; `pienso` (ff=1, fb=1 — unchanged, for continuity and H8 replication);
  `blindfb` (ff=0, fb=1 — the untuned arm, DESCRIPTIVE ONLY, never a hypothesis comparator);
  `blindfb_t` (ff=0, fb=FBb — **the H10 comparator**); `ffonly` (ff=1, fb=0 — pure anticipation,
  giving the additivity check); `imitation_s72` (one seed; pooling three moved the s.e. only
  4.9 -> 4.7 mK/K because the residual is chaos common to all three — seed chosen by lowest index,
  registered as an index rule).
- **Inference:** Rademacher wild bootstrap under the null with HC3 standard errors, on BOTH the
  signed slope and RMS_A; Holm over the two co-primaries; direction-gated; slope equivalence bounds
  on any null. All runs reported.

**H10 (pre-stated):** anticipation — the Nino3.4 feedforward term — reduces disturbance
sensitivity beyond a RETUNED reactive controller, i.e. the paired slope (pienso - blindfb_t) on A
is negative, and/or its RMS_A difference is negative.

**Interpretation grid, pre-stated; every outcome is reportable and none is a failure.**
(i) H10 significant on both co-primaries => anticipation carries genuine control value beyond
reaction; the headline may say the controller exploits its observation of the disturbance.
(ii) H10 null with a tight equivalence bound => a well-tuned reactive controller matches an
anticipating one at this horizon; the headline becomes "feedback rejects the disturbance, and
OBSERVING it adds no measurable benefit once the reactive gain is tuned" — which is more useful for
deployment (it says what must be measured) and is the outcome the surrogates predict.
(iii) The two co-primaries disagree (signed slope vs RMS_A) => report both and treat the
cancellation structure in the hinge fit as the finding.
(iv) blindfb_t not better than blindfb => the tuning grid was inadequate; report and do not claim
H10 either way.
(v) blindfb_t not better than static => the surrogates are wrong and the feedforward term is doing
all the work; H10 becomes trivially true and must be reported as such.

**Cost:** tuning sweep ~35 min (12 arms x 6 train ICs x k=2, baselines shared); final eval ~2.1 h
(6 arms x 24 ICs x k=4, baselines shared). **Total ~2.7 GPU-h.**

### Amendment 7, revision 3 — comparator selected on held-out (frozen 2026-08-11,
### BEFORE any held-out ablation data exists; supersedes revision 2's tuning protocol)

Revision 2's train-IC tuning sweep RAN and is reported, but its selection rule cannot be used.
Two reasons, both visible in the sweep and neither anticipated:

1. **The reactive optimum sits at the GRID EDGE.** RMS_A over fb in {1, 1.5, 2, 3, 4, 6} was
   23.1 / 18.5 / 16.0 / 15.0 / 15.3 / **9.5** mK — still falling at the largest gain tested.
   Declaring fb = 6 "the best reactive controller" would risk exactly the strawman the tuning step
   exists to prevent.
2. **The selection is noise-limited.** Bootstrap CIs over the 6 train ICs overlap throughout
   (fb=3 [7.8, 20.5], fb=4 [7.2, 21.5], fb=6 [6.6, 11.6]) at k=2.

Extending the grid would chase a noisy optimum with more GPU time and still leave the choice
contestable. Instead the comparator is chosen where it cannot be disputed:

**The held-out evaluation carries a LADDER of reactive gains, and the H10 comparator is defined as
whichever ladder arm scores best (min RMS_A) ON THE HELD-OUT DATA ITSELF.** Selecting the
comparator on the test set INFLATES it. That is the CONSERVATIVE direction — it biases against
H10, the hypothesis being advocated — and it forecloses any "the opponent was detuned" objection.
The same rule is applied symmetrically to the anticipating arms so neither side is privileged, and
the full ladder is reported either way. This is registered as a deliberate, disclosed departure
from ordinary practice (one normally avoids test-set selection); it is acceptable here precisely
because its bias runs against the claim.

**Arms (6):** `static`; reactive ladder `b6` (fb=6) and `b12` (fb=12, extending past the train-sweep
edge); anticipating `p1` (ff=1, fb=1 — the Amendment-7 controller exactly, giving an independent
replication of H8/H9 on fresh ICs) and `p2` (ff=1, fb=2 — the train sweep's interior optimum);
`imitation_s72`. FF/FB are WEIGHTS (0 = ablate, 1 = designed strength), pinned by regression test.

**Everything else carries over from revision 2 unchanged:** fresh ICs `ics_enso8` (seed0 9000,
24 held-out), k = 4, deterministic amplitude design (12 mirrored pairs, A_max 1.8273 K, zero mean,
Saa 40.07), registered metric, wild-bootstrap/HC3 inference, RMS_A co-primary with the signed
slope, hinge reporting, slope equivalence bounds, and the `analyze_enso8.py` interpretation grid
in which every outcome of H10 is reportable.

**Additional pre-stated read-out:** if `b12` beats `b6` the reactive optimum is still not bracketed,
and any H10 result must be reported as an UPPER BOUND on the value of anticipation rather than an
estimate of it.

**Cost:** ~3.1 GPU-h.

### Amendment 8 — a GROWING disturbance (frozen 2026-08-12, BEFORE any GPU spend)

**The transient-CO2 experiment is NOT run, and this records why.** The dormant CO2 path was
enabled, plumbed (`setup_coupled_model(co2_rate=, co2_year_ref=)`), and verified live — rate-0 and
rate-2000 rollouts are bit-identical on days 1-2 and diverge from day 3, so the forcing does reach
the ocean. It was then rejected on a STRUCTURAL ground found in design review and confirmed by
inspection: `increase_co2` lives in the `Parameters` closure on `SpeedyPhysics`, so it is present in
BOTH the arm rollout and its paired no-MCB baseline (`run_confirmatory_eval.py:563` passes the same
`step_fn` to `compute_baseline_trajectory`; the ENSO pacemaker escapes this only because it is
applied through `step_fn_transform` to the arm alone). The registered metric is a paired difference
and every controller input is a paired anomaly, so a CO2 trend cancels to EXACTLY zero in the score
and in the controller's only input. The experiment was guaranteed to measure nothing — the Tier-1
"null baked in" failure. Secondarily, longwave band 1 is already near-opaque at the reference
absorptivity (column optical depth 6.0), so the forcing saturates around a 2x multiplier and can
reverse under a stratospheric inversion; "amplify the rate to substitute for a longer run" is false.
Making CO2 usable would require threading the rate through the carry AND running unforced baselines
— a different experiment, not a fix. The plumbing and `run_co2_scoping.py` are kept as a record.

**A correction to the previous framing, also recorded.** Amendment 7's disturbance was described as
a bounded oscillation. It is not: `EnsoConfig.period_days` defaults to 0, so A(t) ramps over 30 days
and then HOLDS at full amplitude for the remaining 150. The 85-89% rejection result is already
**persistent-step** rejection. The genuine untested increment is step (type-0) -> **monotonically
growing ramp** (type-1) — the case where classical control predicts proportional action leaves a
steady-state error — and the existing pacemaker delivers it by setting the ramp to the full episode.
Unlike CO2 it is applied to arms only, so the disturbance is visible in score and input alike.

**Question.** Does closed-loop rejection, and the Amendment-7-revision-3 finding that OBSERVING the
disturbance is worth nothing, survive when the disturbance never stops growing?

**Design.** Identical to the previous campaign except `--enso-ramp-days 180` (ramp across the whole
episode) and n, so any difference is attributable to the disturbance SHAPE. 180-day episodes
(the 365-day alternative was rejected: measured paired tail-mean sd 32.2 mK at 180 d versus
85.2 mK at 365 d, with 184 mK of control drift). n = 32 held-out ICs (16 mirrored pairs, fresh set
seed0 10000 — the enso7/enso8 held-out sets are exhausted), k = 4, deterministic zero-mean
amplitude design with A_max set to the measured actuator floor.

**Arms.** `static` (manipulation check); reactive ladder `b6`, `b12` (feedforward ablated, feedback
gain varied — comparator = best ON HELD-OUT, conservative, per revision 3); anticipating ladder
`p1`, `p15` (feedforward weight 1.0 and 1.5); `imitation_s72`. The FF LADDER is mandatory, not
decorative: under a ramp the feedforward term reads the INSTANTANEOUS index while the tail-window
index is f*A with f = 0.8361, so the plant slope must be fed as `s_ramp / f`; feeding the raw slope
would under-dose feedforward by 16% and MANUFACTURE an H10 null — confirming the previous headline
by detuning the arm under test, this project's recurring failure mode pointed at confirmation bias.
The ladder makes the conclusion robust to that constant.

**Endpoints and inference.** Unchanged from `analyze_enso8.py`: disturbance sensitivity (paired
slope on the hidden amplitude, invariant to constant rescaling) co-primary with the sign-agnostic
RMS_A, wild-bootstrap/HC3 inference, hinge warm/cold reporting, slope equivalence bounds.

**Freeze gate before the eval:** the measured actuator floor A_max must land in 2.5-5.0 K. Above
5 K the disturbance is no longer ENSO-shaped and the framing must become "a generic tropical SST
trend" rather than ENSO.

**Interpretation grid, pre-stated.** (i) Rejection holds AND observing still adds nothing => the
previous finding generalises from a step to a growing trend; the deployment implication (measure
what you control, not what disturbs it) strengthens materially. (ii) Rejection holds but the
anticipating arms now BEAT the reactive ones => the type-1 prediction is confirmed: a growing
disturbance is exactly where anticipation earns its keep, and the previous null was specific to a
bounded disturbance. (iii) Rejection degrades sharply for ALL arms => a growing disturbance exceeds
what this actuator can track, which is a real limit worth reporting. (iv) Reactive and anticipating
both fail while static does not => something is wrong with the ramp plumbing; report, do not claim.
All four are reportable.

**Cost:** scoping 0.3 + ICs 0.4 + calibration 0.5 + eval 5.5 = **~6.7 GPU-h**. No new code: every
flag already exists.

### Amendment 9 — long-horizon gradients in the coupled model: Step 0 + Experiments 1-3 (Step 0 and Experiment 1 frozen 2026-09-29, BEFORE any GPU spend; Experiments 2-3 declared here and frozen in full as revision 1 BEFORE any of their data exist)

**Why a new line of work.** The 2026-09-29 review (MCB_PROJECT_REPORT.md Part 16) found that the
planned methods paper's central claim, "gradients design, they do not train", is untested on both
halves: the Stage-1 v2 pattern was the iteration-1 pick of a loss that was 99.5% weather noise, and
the Tier-2 training failure confounds a 60-day horizon, a ~1.25M-parameter output and 40 updates. It
also found a same-model competitor: Dubey, Abbot & Chattopadhyay (2026, arXiv:2609.12528) design
interventions in atmosphere-only JAX-GCM with receding-horizon backpropagation and measure the
gradient breakdown (gradient-vs-finite-difference correlation 0.91 / 0.87 / 0.49 at 7 / 14 / 28 d).
With an interactive ocean, today's forcing keeps acting for months, so two-week planning is myopic.
This amendment tests whether cutting only the atmosphere's chaotic memory, while keeping the ocean's,
yields gradients that stay useful for months.

**Estimator (Step 0, implemented and tested).** `jcm/mcb/gradient_truncation.py` stops gradients
through `carry["atm"]["state"]` (the atmosphere's dynamical memory) at the start of every coupling
step whose day index d satisfies d % W == 0, reading d from the OCEAN clock (never the scan index).
It deliberately leaves `atm.derived` (the previous day's fluxes, handed to the ocean by the coupler at
the start of the next step, and the actuator field), `atm.forcing` (SST and land temperature arriving
from the slow components), and the ocean and land states untouched. Forward values are bit-identical
for every W; W = full means ordinary BPTT. Unit tests against an independent forward-mode recursion
on a toy with the coupler's exact data flow: `jcm/mcb/gradient_truncation_test.py`.

**Related work to read in full before any novelty claim** (full texts were not retrievable by tool):
Sugiura et al. 2008 (JGR, coupled 4D-Var with months-long windows: how is the atmospheric adjoint
handled?) and Lu & Hsieh 1998 (Tellus A, coupled toy-model adjoint). Also cite: Lyu et al. 2018
(JAMES, 2-month adjoint windows by chaos synchronization), ECCO (ocean-only adjoints), Lea et al.
2000 / Eyink et al. 2004 (ensemble adjoint), Wang et al. 2014 (least-squares shadowing), Pires et al.
1996 (quasi-static variational assimilation), Metz et al. 2021, Suh et al. 2022, List et al. 2024,
the multi-step-penalty training of chaotic neural ODEs (arXiv:2407.00568), FESOM2-JAX
(arXiv:2608.01546). Terminology rules from PAPER_PLAN.md apply: "idealized ocean cloud-albedo
intervention", never "MCB", in any title or abstract; "weather realizations" / "macro states", never
"climates".

**Step 0: macro x micro starting states** (`run_generate_macro_ics.py`, registered plan in its
`DEFAULT_PLAN`). Continue the equilibrated control run from `equilibrated/base_carry.pkl` and save
16 macro states every 730 days (same season each time; macro 0 = the base carry). Branch each macro
state into weather trajectories exactly as `run_generate_ics_independent.py` does (0.05 K seeded SST
perturbation, 30-day decorrelation spin, paired baseline), with branch seed = 12000 + 100 * macro +
branch and IC index = 100 * macro + branch. Roles, disjoint in (macro, branch) and with evaluation on
macro states no other role touches:

| role | macro states | branches | split | baseline horizon |
|---|---|---|---|---|
| `exp1` | 0-7 | 0 | heldout | 120 d |
| `exp2_train` | 0-7 | 1 | train | 60 d |
| `exp2_eval` | 8-15 | 0, 1 | heldout | 60 d |
| `exp3_train` | 0-7 | 2, 3 (train); 4 (heldout = validation) | train / heldout | 60 d |
| `exp3_eval` | 8-15 | 2, 3, 4 | heldout | 60 d |

The script reports each macro state's ocean-mean SST and the anomaly pattern correlation and RMS
difference between neighbouring macro states. These are reported, not gated: they document how
different the ocean states really are. Micro members (0.001 K) are drawn at evaluation time by the
harnesses, as before.

**Experiment 1 — do truncated gradients match ensemble truth?** (`run_gradient_fidelity.py`,
`analyze_gradient_fidelity.py`, both frozen with this amendment; the analysis constants at the top of
`analyze_gradient_fidelity.py` are the registered thresholds.)

* *Controls.* Five Gaussian latitude bands over the ocean, centred at 45N, 20N, 0, 20S, 45S with 10 deg
  width (Dubey et al.'s layout), applied through the cloud-albedo actuator. Operating point
  a0 = 0.03 in every band.
* *Objectives* (daily series, tail mean over the final 10 days of each horizon): T0 ocean-mean SST;
  T1 and T2 the interhemispheric and equator-to-pole Legendre contrasts over the ocean (centred, so a
  uniform change projects on T0 only); LAND the slab-land mean temperature, which the actuator can
  reach only through the atmosphere. Horizons 15, 30, 60, 120 d.
* *Truth.* For each IC in `exp1` (8 ICs) and each of 4 micro members (member 0 unperturbed, members
  1-3 perturbed 0.001 K with seed 91000 + 97 * index + m): forward rollouts at a = 0, a0, and
  a0 +/- delta e_k with delta = 0.03 (the minus run switches that band off). The truth vector is the
  mean over the 32 realizations of the central differences. Paired baselines cancel exactly in a
  central difference.
* *Estimators.* Reverse-mode Jacobians at a0 on member 0 of each IC (8 realizations) for W = 1, 7,
  14 d and full BPTT, windows aligned to the episode start.
* *Metrics* (per window x horizon x objective): angle between the mean estimator and the truth;
  projection ratio; median single-realization angle; noise-to-signal ratio; sign agreement on
  resolved components. 95% CIs from a hierarchical bootstrap (ICs, then members within IC; 2000
  replicates, seed 2026) are primary; a flat bootstrap over realizations is reported alongside.
* *Verdicts.* **useful** = truth resolved (SNR >= 3) AND angle CI upper < 20 deg AND ratio CI inside
  [0.6, 1.4] AND median single-realization angle <= 45 deg. **failed** = angle CI lower > 20 deg, OR
  ratio CI entirely outside [0.6, 1.4], OR median single-realization angle > 45 deg.
  **inconclusive** otherwise.
* *Primary endpoint:* T0 at 60 d. *Outcome grid and actions:* **U** truth unresolved -> add truth
  members (no design change) and rerun the truth only. **A** some truncated W useful AND full BPTT
  failed -> "truncation rescues the gradient"; run Experiments 2-3 with W*. **A'** some truncated W
  useful, full BPTT inconclusive -> run Experiments 2-3 with W*; report full BPTT as inconclusive.
  **B** full BPTT useful -> "ocean objectives keep a long gradient horizon" (a contrast with
  Dubey et al.'s land objective); run Experiments 2-3 with full BPTT as the primary estimator and W*,
  if any, as secondary. The old training failure is then attributed to parameterization and budget,
  which Experiment 3 tests. **C** no estimator useful -> STOP all GPU spend on Experiments 2-3 and
  write up the gradient-fidelity characterization.
* *W\*:* the useful truncated window with the smallest median single-realization angle at the primary
  endpoint; ties within 2 deg go to the larger window.
* *Registered prediction:* at 60 d, W = 1 is `failed` on LAND or its ratio lies outside [0.5, 2]
  (the method's domain boundary). If the LAND truth is unresolved (SNR < 3), the prediction is
  reported as untestable. All other cells are reported without gates.
* *Cost:* about 8 GPU-h (truth ~3.5 h: 8 ICs x 4 members x 12 rollouts of 120 d; gradients ~4.8 h:
  8 x 4 windows x (15+30+60+120) d). The macro starting states take about 1.4 GPU-h.

**Experiment 2 — can the gradient design a spatial forcing? (declared; frozen as revision 1 after
Experiment 1).** The target is T0 = -0.1 K on the 60-day tail mean with no change in T1 or T2.
Uniform brightening cannot meet it, because January sunlight is weighted to the southern hemisphere,
so the design has to be spatial. Arms:
* the gradient design with the Experiment-1 estimator;
* a full-BPTT design;
* the classical linear-response design solved from finite-difference sensitivities computed on
  `exp2_train`;
* a uniform map matched on T0.

A higher-dimensional basis variant then shows how cost scales with the number of knobs. Evaluation is
on `exp2_eval`, paired with micro-ensembles (k = 4), using the Amendment-2 statistics. Revision 1 must
fix the objective weights, the compute budget given to each arm, the endpoints and the tests before
any Experiment-2 rollout.

**Experiment 3 — can the gradient train a controller? (declared; frozen as revision 1 after
Experiment 1).** The task is the Tier-2 hidden-efficacy problem with a LOW-dimensional policy (at most
~100 parameters), trained at equal GPU budget by three methods:
* full BPTT;
* the Experiment-1 estimator;
* ensemble Kalman inversion (the standard derivative-free comparator).

The policies are compared with the classical adaptive law and the static map on `exp3_eval`. An
optional greedy 14-day receding-horizon arm shows what myopia costs once the ocean remembers.
Revision 1 must fix:
* the efficacy range, with the choice justified against the ~20x cross-model spread reported for
  G6-1.5K-MCB;
* the policy architecture, the budgets and the stopping rules;
* the hypotheses and the equivalence bounds.

**Posting.** Before Experiment 1 runs, this amendment and the commit hash that freezes the Step-0 and
Experiment-1 code are posted to OSF for an independent timestamp.

### Amendment 9, revision 0.1 — related work read in full; exploratory damped estimators added to Experiment 1 (frozen 2026-09-29, BEFORE any Step-0 or Experiment-1 data)

**Related work, now read in full.** The PDFs are kept locally under `literature/` and are not
committed.

* **Sugiura et al. (2008)**, JGR 113, C10017, doi:10.1029/2008JC004741.
  * *What they did:* 4D-Var coupled data assimilation in the CFES coupled GCM (T42 atmosphere,
    1-degree ocean), with 9-month windows and 1.5-month margins.
  * *Controls and data:* ocean initial conditions plus bulk flux adjustment factors (latent heat,
    sensible heat, momentum) at every grid point every 10 days; observations assimilated as 10-day
    means.
  * *How they tame atmospheric chaos:* an APPROXIMATE adjoint. It is tangent-linearized about
    10-day-mean ("coarse-grained") fields, adds an artificial adjoint damping
    Gamma = diag{a_i lambda_i^2}, and approximates nonlinear and subgrid terms semilinearly. The
    forward model is unchanged.
  * *Validation:* the approximation is asserted to "affect the efficiency but not necessarily the
    direction" of the optimization. It is validated only indirectly, against observations, and never
    compared with finite differences.
* **Lu & Hsieh (1998)**, Tellus 50A, 534-544, doi:10.3402/tellusa.v50i4.14531.
  * *What they did:* the full, exact coupled adjoint of a LINEAR, damped shallow-water equatorial
    model over 40-day windows. There is no chaos, so the gradient cannot blow up.
  * *Setup:* daily coupling with the forcing held fixed within the day. Atmospheric initial
    conditions are fixed at zero (an equilibrium atmosphere), and they retrieve 3 ocean initial
    fields and 6 parameters.
  * *Caveat:* the authors warn that their conclusions may not hold for longer windows in realistic
    models.

**Consequence for the novelty claim (binding on every write-up).** Months-long gradients in a coupled
model are NOT new: Sugiura et al. obtained approximate ones. The claim is restricted to three things:

1. Exact automatic differentiation that removes only the atmosphere's dynamical memory
   (`atm.state`), with no tuned damping, no approximated adjoint and a bit-identical forward model.
2. The first direct validation of such a long-window coupled gradient against ensemble
   finite-difference truth. Experiment 1 tests exactly the property Sugiura et al. asserted.
3. Its use for intervention design and control, not state estimation.

Cite Sugiura et al. as the closest precedent, and Lu & Hsieh as the exact-adjoint, non-chaotic case.

**Added to Experiment 1: exploratory damped estimators D_tau.** A reviewer will ask why we cut instead
of damp.

* *Definition.* No truncation (W = full). At the start of every coupling step, the tangents and
  cotangents of `carry["atm"]["state"]` are multiplied by gamma = exp(-1/tau) per day (see
  `scale_tangent`, `damp_atmosphere_gradient` and the `atm_decay` argument in
  `jcm/mcb/gradient_truncation.py`). Forward values are bit-identical, and the rest of the carry is
  left untouched, exactly as for truncation. This is a constant-rate analogue of Sugiura et al.'s
  adaptive damping.
* *Values.* tau in {3, 7} days.
* *Stated expectation (not a gate).* Damping tames the chaos only if 1/tau exceeds the growth rate of
  the gradient noise. Dubey et al.'s noise-to-signal ratio rises from 0.07 at 14 days to 1.2 at 30
  days, about 0.18 per day. So tau = 3 d (0.33 per day) should stay stable, and tau = 7 d (0.14 per
  day) is marginal.
* *Data.* Computed on the same ICs, member and horizons as the registered estimators, and stored as
  `jacobians_damped`.
* *Analysis.* `damped_cells` in `analyze_gradient_fidelity.py` uses the same metrics, the same
  thresholds and the same bootstrap resampling (same seed, identical truth replicates). Results are
  reported under `exploratory_damped`.
* *Role.* NOT part of the outcome grid (U/A/A'/B/C) and NOT eligible for W*. Experiments 2-3
  (revision 1) may carry a damped estimator only as a secondary arm.

**What this changes, and doesn't, in the registered analysis:**

* The registered windows are now evaluated with an explicit decay factor of exactly 1, which leaves
  their gradients unchanged. Tested in `DampedJacobianTest.test_decay_one_equals_the_registered_estimators`
  (bitwise) and in `DampedGradientTest`.
* The registered code path of the analysis is unchanged. Tested in
  `ExploratoryDampedTest.test_registered_results_are_unchanged`.

**Cost.** About +2.4 GPU-h (1,800 gradient-days per estimator). Experiment 1 is now about 10.7 GPU-h,
and about 12 with the macro starting states.

**Still to decide before Step 0 runs** (MCB_PROJECT_REPORT.md Part 18, steps 1-3):

* a Q-flux for the slab ocean;
* interleaved train and evaluation macro states;
* saving full maps from Experiment 1's truth runs.

Each one adopted will be logged as a further revision BEFORE any data exist.

**Posting.** The OSF posting required above will include this revision and the commit hash that
contains it.

### Amendment 9, revision 0.2 — base climate with a monthly Q-flux (frozen 2026-09-30, BEFORE the Q-flux diagnosis, the settling run and any Step-0 or Experiment-1 data)

**Decision.** The user chose to fix the free slab's cold bias before Step 0 (MCB_PROJECT_REPORT.md
Part 18, step 1). A Q-flux is a fixed, seasonal heat source or sink per ocean cell that stands in for
the ocean heat transport a slab lacks.

**Two facts found while planning (reported upstream, not patched there):**

* *jax-esm indexes both climatology modes by DAY.* Its `forcing_method="Qflux"` computes the index as
  floor(day) mod (cycle length), although the field's dimension is named "month" and holds 12
  entries. A monthly Q-flux would therefore advance one month per day and repeat every 12 days.
  `"relaxation"` indexes its SST climatology the same way. Neither mode is used.
* *Over sea ice, `forcing.nc` "sst" is the ice surface temperature* (minimum 236.6 K). SPEEDY's surface
  fluxes use `sea_surface_temperature` directly, with no separate ice temperature, so these are the
  temperatures the atmosphere is built to see. Sub-freezing values in ice regions are therefore not
  an error in this model; the missing ocean heat transport is. No freezing floor is added.

**Model change.**

* *The new class.* `MonthlyQfluxSlabOceanModel` (`jcm/mcb/qflux.py`) is the jax-esm free slab plus a
  monthly Q-flux read from `carry["ocn"]["forcing"].q_flux`: shape (lon, lat, 12), in W m-2,
  upward-positive. It is linearly interpolated between mid-month anchors on a 365.2425-day year and
  evaluated at mid-step.
* *Where it is used.* `setup_coupled_model` now always builds this class.
* *Nothing changes without a Q-flux.* With a zero Q-flux (every cold start, and every carry saved
  before this revision) it reproduces the old free slab bit for bit. This is tested on the component
  (`MonthlyQfluxSlabTest`) and on the full coupled model (`CoupledBitIdentityTest`, slow).
* *It travels with the state.* The Q-flux lives in the carry, so every state branched from a Q-flux
  base carry inherits it.

**Diagnosis** (`run_qflux_base_climate.py diagnose`).

* *Start.* Cold start (the usual `initialize()`: observed January SST, resting atmosphere), coupled
  model on realistic terrain.
* *Restoring.* After every coupled day, the SST of the slab's ocean cells is restored toward the
  `forcing.nc` monthly SST climatology (same interpolation), with tau = 5 days.
* *Record.* Spin up for 365 days (discarded), then record 1,460 days.
* *Fit.* The daily restoring heat is fitted by least squares with the interpolation weights at each
  step's midpoint. Q = minus the fit (upward-positive); land cells get 0.
* *Output.* Written to `mcb_experiments/qflux/qflux_monthly_t30.nc` with diagnostics: fit residual,
  area-weighted mean Q, zonal mean.

**Settling run** (`run_qflux_base_climate.py settle`). Cold start plus the diagnosed Q-flux, as a free
slab (no restoring), for 3,650 days. The final state is the new base carry.

**Acceptance gate.** It is evaluated on the last 1,460 days of the settling run, with cos(latitude)
area weights over the slab's ocean cells.

* **G1 drift.** The linear trend of daily ocean-mean SST, fitted together with 3 annual harmonics, must
  have |trend| < 0.02 K per 60 days. This is the original equilibration criterion.
* **G2 bias.** The mean ocean-mean SST must be within 0.5 K of the observed annual mean.
* *Reported, not gated:*
  * the RMS error of the annual-mean SST map;
  * the RMS error of the seasonal cycle;
  * the number of cells with |annual-mean error| > 2 K;
  * the SST range and the land temperature range;
  * simple averages over `fmask < 0.5` cells, for comparison with the original 283.9 -> 280.1 K.
* *If G1 or G2 fails:*
  * the Q-flux climate is NOT used;
  * Step 0 keeps the original base carry, and the cold bias stays a documented limitation;
  * any new attempt is discussed first. No silent retries.

**Where it runs.**

* *On a CPU, locally:* the diagnosis and a validation settle. Their numbers go in the report, and the
  small Q-flux NetCDF is committed.
* *In the GPU campaign* (`run_campaign_step0_exp1.sh`): it re-runs the settle stage from the
  committed Q-flux file, applies the same gate, and aborts if it fails. Step 0 then branches from
  that GPU-settled base carry (`mcb_experiments_gpu/equilibrated_qflux/base_carry.pkl`, used instead
  of `equilibrated/base_carry.pkl`). `run_generate_macro_ics.py --require-qflux` refuses a base carry
  without one.

**Consequences.**

* The campaigns in Parts 8-15 ran in the no-Q-flux climate. They are not re-run, and results are
  never pooled across the two climates.
* Cost: about 5,500 model-days in all, which is about 0.5 GPU-h on the GPU or a few hours on a CPU.

**Amendment 9, revision 0.2 — RESULT (logged 2026-09-30, after the runs; the rule above is unchanged).**
Both runs were done on a laptop CPU (`mcb_experiments/qflux/`).

* **Diagnosis** (27 min).
  * The restored run tracked the observed climatology almost exactly: its mean minus observed at the
    same time of year is -0.02 K (range -0.07 to +0.04 K, 20 samples).
  * The fitted Q-flux's ocean-mean annual value is -11.13 W m-2 (upward-positive), meaning about 11
    W m-2 of heat INTO the ocean on average.
  * The largest values, up to about 1,300 W m-2, are seasonal and sit in the Northern-Hemisphere
    seasonal sea-ice zone (147 of the 148 cells above 1,000 W m-2, at 54-87 N). There, 40-60 m of
    water is made to follow the ice surface's 18-37 K annual swing; the annual mean over ice cells is
    +1.2 W m-2.
  * In ice-free cells, the median of each cell's largest monthly |Q| is 109 W m-2, and 99% of cells
    stay below 354 W m-2.
* **Settling run** (51 min, 3,650 days).
  * **G1 drift = -0.0004 K per 60 days: PASS.**
  * **G2 bias = -1.62 K: FAIL** (limit 0.5 K).
  * **Verdict: FAIL.**
* **Reported numbers.**
  * RMS annual-mean error 2.06 K; RMS seasonal-cycle error 0.62 K; 1,159 of 3,411 cells with
    |annual error| > 2 K.
  * Simple mean over `fmask < 0.5` cells: model 283.23 K vs observed 284.93 K, where the old free slab
    was about 3.8 K cold.
  * The error is nearly uniform: -1.1 to -2.2 K in every 20-degree latitude band. Ice-free cells are
    at -1.57 K and seasonal-ice cells at -1.81 K (22% of the global bias, close to their 20% area
    share).
  * The free run cooled slowly for about 7 years, then levelled off.
* **Consequence (per the rule).**
  * This Q-flux climate is NOT used, and the original base carry stays the Step-0 base.
  * No new attempt is run before it is discussed with the user and registered as a further revision.
  * The files are kept as the record of the attempt: the Q-flux NetCDF and both summaries.

### Amendment 9, revision 0.3 — one Newton correction of the Q-flux (frozen 2026-09-30, AFTER the revision-0.2 result and BEFORE the correction's settling run)

**Decision.** After the revision-0.2 gate failed (bias -1.62 K), the user chose ONE correction attempt.
This is the registered "discuss first" step.

**Why the method changed before any run.** The option put to the user was to re-measure the missing
heat under loose restoring. Checking it before registration showed two problems:

* It recovers only a fraction, (C / tau_r) / (lambda + C / tau_r), of a state-independent deficit.
* It recovers less still if the restoring suppresses the variability that causes the deficit.

The expected residual was -0.3 to -0.9 K. It is replaced by a direct Newton step on the settled free
run, which needs no new measuring run:

* **Fit.** Take attempt 1's daily area-weighted ocean-mean SST. Deseasonalize it with 3 annual
  harmonics fitted on its last 1,460 days, average in 60-day blocks, and fit from day 180:
  T(t) = T_inf + A exp(-t / tau). Tau is searched on a fine geometric grid, and the model is linear in
  T_inf and A.
* **Sensitivity.** lambda = C_eff / tau, where C_eff is the area-weighted mean rho c_p h over the
  slab's ocean cells (the gate's weights).
* **Correction.** dQ = lambda (T_obs - T_inf) W m-2 into every ocean cell in every month, so
  Q2 = Q1 - dQ (upward-positive). It is uniform because attempt 1's bias is nearly uniform and the
  gate is on the global mean.
* **Numbers already seen** (from attempt-1 data only):
  * tau about 1,255 d (fits starting at 365 and 730 d give 1,231 and 1,194 d);
  * T_inf about 288.61 K, i.e. a bias of -1.78 K;
  * C_eff about 1.96e8 J m-2 K-1, so lambda is about 1.80 W m-2 K-1 and dQ about +3.2 W m-2.

  `run_qflux_base_climate.py correct` reproduces the computation, and
  `qflux_monthly_t30_v2_correction.json` records it.

**Test.** Run `settle` with Q2 from a cold start for 3,650 days. The gate and code are the SAME as in
revision 0.2 (G1: |drift| < 0.02 K per 60 days; G2: |bias| < 0.5 K). Outputs go to
`mcb_experiments/qflux/attempt2/`.

**Outcomes.**

* **PASS.** Q2 (`mcb_experiments/qflux/qflux_monthly_t30_v2.nc`, committed) becomes the Q-flux. The
  GPU campaign re-settles from it under the same gate, and Step 0 branches from that base carry.
* **FAIL.** There are no further attempts. Step 0 uses the original base climate, the cold bias is a
  documented limitation, and the GPU script is switched back.

**Amendment 9, revision 0.3 — RESULT (logged 2026-09-30, after the run; the rule above is unchanged).**
Run on a laptop CPU from a cold start with Q2 (`mcb_experiments/qflux/attempt2/`).

* **Settling run** (3,650 days; 3.3 h of wall time because the laptop was throttled on battery,
  against 51 min for attempt 1).
  * **G1 drift = -0.0044 K per 60 days: PASS.**
  * **G2 bias = -0.26 K: PASS** (limit 0.5 K).
  * **Verdict: PASS.**
* **Reported numbers** (attempt 1 in brackets).
  * RMS annual-mean error 1.25 K [2.06]; RMS seasonal-cycle error 0.56 K [0.62]; 363 of 3,411 cells
    with |annual error| > 2 K [1,159].
  * Simple mean over `fmask < 0.5` cells: model 284.79 K vs observed 284.93 K [283.23].
  * By 20-degree band, south to north: +0.17, +0.28, +0.23, -0.05, -0.88, -0.73, -0.29, -0.17,
    +0.07 K [-2.17 to -1.23 K]. The uniform heating warmed the high latitudes about twice as much as
    the tropics, so the leftover error is a pattern: tropics 0.7-0.9 K too cold, Southern Ocean about
    0.25 K too warm.
  * Cells with sea ice in any month (`icec`, 20% of the area): +0.21 K [-1.81]; ice-free: -0.38 K
    [-1.57].
* **Checks (not part of the gate).**
  * *No hidden approach.* The exponential fit used for the Newton step finds no approach curve here
    (tau runs to the grid edge for fit starts 180, 365 and 730 d): the run started near where it
    settled. Yearly means sit between -0.05 and -0.31 K (SD 0.086 K, lag-1 autocorrelation 0.65); the
    10-year mean is -0.19 K.
  * *The correction under-shot slightly.* Adding 3.21 W m-2 raised the gate-window mean by 1.36 K
    (1.52 K relative to attempt 1's fitted end state), against the 1.78 K aimed for. The realized
    sensitivity is 2.1-2.4 W m-2 K-1 rather than the fitted 1.80.
* **Consequence (per the rule).**
  * Q2 (`mcb_experiments/qflux/qflux_monthly_t30_v2.nc`) is the Q-flux. Attempt 1's files stay as the
    record. No further correction is made.
  * `run_campaign_step0_exp1.sh` re-settles from Q2 on the GPU under the same gate, and Step 0
    branches from that base carry. The GPU run is a second weather realization; with a window-mean
    standard error of about 0.07 K, a fail there is very unlikely, but the gate still applies.
  * The tropical cold / Southern Ocean warm pattern is a documented limitation of the base climate.

### Amendment 9, revision 0.4 — interleaved training and evaluation states; maps from Experiment 1 (frozen 2026-10-01, BEFORE any Step-0 or Experiment-1 data)

**Decision.** The user adopted MCB_PROJECT_REPORT.md Part 18 steps 2 and 3, the last two items that
revision 0.1 left open. No Step-0 or Experiment-1 data exist, and nothing has run on the GPU.

**1. Interleaved macro states (Step 0).**

* *Change.* Experiment 1 and every training or validation role use the even macro states (0, 2, ...,
  14); evaluation uses the odd ones (1, 3, ..., 15). Branches, seeds (12000 + 100 * macro + branch), IC
  indices, horizons and role sizes are unchanged. Amendment 9's table becomes:

| role | macro states | branches | split | baseline horizon |
|---|---|---|---|---|
| `exp1` | 0, 2, ..., 14 | 0 | heldout | 120 d |
| `exp2_train` | 0, 2, ..., 14 | 1 | train | 60 d |
| `exp2_eval` | 1, 3, ..., 15 | 0, 1 | heldout | 60 d |
| `exp3_train` | 0, 2, ..., 14 | 2, 3 (train); 4 (heldout = validation) | train / heldout | 60 d |
| `exp3_eval` | 1, 3, ..., 15 | 2, 3, 4 | heldout | 60 d |

* *Why.*
  * The first-half/second-half split put training in the first 15 years of the 30-year control run
    and evaluation in the last 15, so any slow wander of the ocean's mean state would have separated
    the two sides. The Q-flux settling run shows such wander: its yearly means moved over about
    0.26 K, with a lag-1 autocorrelation of 0.65 (revision 0.3 RESULT).
  * Interleaving keeps the states on each side four years apart instead of two, so they are closer
    to independent. Experiment 1 benefits directly: its hierarchical bootstrap resamples ICs as
    independent units, and under the old plan its eight ICs were consecutive states two years apart.
* *Alternatives considered.*
  * An ABBA order (train, eval, eval, train, ...) would cancel a linear trend exactly, but it puts
    pairs of same-side states two years apart, which costs independence. The measured trend is too
    small to be worth that (attempt-2 drift -0.0044 K per 60 days; no trend in the yearly means).
  * A random assignment would add nothing over a fixed rule.
* *The cost of the choice.* Each evaluation state now has training neighbours two years away.
  Regional SST anomalies in the slab decay with e-folding times of about 60-570 days, so this link
  should be weak. It is measured, not assumed (next bullet).
* *Reported, not gated* (in the log and `macro_bases_manifest.json`), in addition to Amendment 9's
  per-state means and neighbour similarities, which now straddle the split:
  * the similarity of states two spacings apart (same side);
  * the least-squares trend of the ocean-mean SST across the 16 states, with its standard error
    (Part 18 step 8);
  * the split balance: the mean ocean SST of the training-side and of the evaluation-side states, and
    their difference.
* *Enforced in code.* `validate_plan` now rejects any plan in which an evaluation role (`*_eval`)
  shares a macro state with another role. Before, only a unit test checked this, and only for the
  registered plan.

**2. Maps from Experiment 1 (secondary outputs).**

* *Truth maps.* Every truth rollout (unchanged: 8 ICs x 4 members x 12 runs of 120 d) also returns its
  slab-ocean SST and slab-land temperature maps as 5-day block means over all 120 days, relative to
  288 K, from the same rollout (`make_series_and_maps_fn`). The 5-day blocks reproduce every
  registered tail window exactly (days 6-15, 21-30, 51-60 and 111-120) and any look-ahead average that
  revision 1 may choose.
* *Map Jacobians.* For each registered window (W = 1, 7, 14 d and full) on member 0 of each IC, one
  forward-mode pass through the same truncated rollout (decay exactly 1) gives the derivative of every
  block-mean map cell with respect to each band (`make_map_jacobian_fn`). The damped estimators get
  none.
* *Why the map Jacobians too.* Part 18 step 3 names a check on the planner's map-shaped objective,
  which needs both sides; truth maps alone give only the brute-force response. With both stored:
  * the gradient of any map-shaped objective can be checked offline against brute force;
  * step 23's comparison of the planner's response maps with brute-force maps is paired on the same
    ICs;
  * the forward-mode pass is the planner's preferred gradient (Part 18: Gauss-Newton with a
    forward-mode gradient). Experiment 1 therefore also validates it, runs step 16's
    forward-versus-backward agreement test on the real model, and times it on the GPU for step 17.
* *Two checks, logged in `<output>.json`, reported and not gated.*
  * (a) Each truth run's maps, weighted like the objectives, must reproduce the tail means of its own
    objective series (`maps.truth_alignment_max_abs_K`).
  * (b) The forward-mode map Jacobians, weighted the same way, must reproduce the registered
    reverse-mode Jacobians of the same window, IC and member (`map_jacobian_check`, per objective).
  * *Expected values, measured on a laptop CPU* (real model, 2-10 days):
    * (a) a few 1e-6 K: 0.8e-6 K over 10 days in 5-day blocks, 3.4e-6 K with 1-day tails. This is
      float32 round-off in the objectives' weighted sums; a one-day misalignment would show as at
      least about 1e-2 K.
    * (b) about 1e-3 of each objective's largest entry, worst 3.3e-3 (LAND at 3 days), similar for
      W = 1, 7 and full. This is float32 round-off over the model's sub-steps.
    * Over long untruncated windows chaos may amplify round-off differently in the two modes; that
      would be reported as a property of untruncated gradients.
* *Storage.* `<output>_maps.npz`, float32, about 0.48 GB uncompressed, next to the registered
  `<output>.npz`.
* *Role.* Secondary outputs. `analyze_gradient_fidelity.py` never reads them, and they enter neither
  the outcome grid (U/A/A'/B/C) nor the choice of W*. Their uses are fixed in revision 1, before any
  Experiment-2 or Experiment-3 data: the classical linear-response design, the check of the planner's
  response maps and map-shaped objective, and the pilot.

**What this changes, and doesn't, in the registered Experiment 1.**

* Unchanged: the ICs (now drawn from the even macro states), members, seeds, amplitudes, horizons,
  windows, damped arms, metrics, thresholds, outcome grid and W*, and also `analyze_gradient_fidelity.py`
  and every array it reads.
* The truth's objective series now come from the function that also returns the maps. On the laptop
  CPU, over 10 days with the registered 5 bands, the two functions gave bit-identical series at the
  same speed. `MapsTest` (toy) and `RealModelMapsTest` (real model, slow) check this to round-off.
* End to end, the CPU smoke of the campaign run with and without maps gave bit-identical
  `fd_series`, `jacobians` and `jacobians_damped`, and identical output from the registered
  analysis.
* The registered reverse-mode Jacobians are computed exactly as before; the map Jacobians run after
  them.

**Cost.**

* Truth maps: no measurable extra time on the laptop.
* Map Jacobians: one 120-day forward-mode pass per IC and window (32 passes). On the laptop a
  forward-mode day cost 6.0-6.6 truth-rollout days, against 10.9-12.6 for the registered reverse mode.
  Scaled to the registered gradients' 4.8 GPU-h, that is about 1.5 GPU-h.
* Experiment 1 is now about 12.2 GPU-h; Step 0 plus Experiment 1, with the Q-flux re-settle, about
  14 GPU-h.

**Posting.** The OSF posting includes revisions 0.1-0.4 and the commit that contains them.

**Amendment 9 — OSF posting (logged 2026-10-01).** Amendment 9 with revisions 0.1-0.4 was posted to
OSF on 2026-10-01:

* the public project https://osf.io/pqabf/, with files uploaded at 18:42 PDT whose SHA-256 match commit
  512c85dc;
* the registration https://osf.io/2bs8p/, registered at 18:44 PDT and pending the admin's approval at
  the time of writing (OSF then makes it public).

**Deviation.** The amendment required this posting before Experiment 1 ran; it came 54 minutes after
Experiment 1 started (17:48 PDT). The registered text and code were public on GitHub, in commit
512c85dc, from about 16:30 PDT, before the campaign began.

**What had been viewed before the posting:**
* the GPU re-settle's registered check and the Step-0 diagnostics (both reported, not gated);
* for the first Experiment-1 IC only, revision 0.4's forward-versus-backward check and the size of
  the registered gradient estimates.

No truth estimate, metric or outcome had been computed or viewed.

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

**Amendment 9, revision 1 — Deviation D1 (logged 2026-10-07 12:10 PDT, before any Experiment 3a
analysis output or score existed).** All 240 evaluation runs finished (05:27 PDT; none failed). The
registered `analyze_experiment3a.py` then stopped with `KeyError: 'schedules'` while loading the runs.
The five fixed-pattern arms (`uncontrolled`, `uniform_cancel`, `uniform_effort`, `planner_average`,
`linear_response`) record their single constant pattern as `amplitudes`, not as a per-segment
`schedules` table. The pre-launch check had loaded only planner files.

The fix reads that constant pattern as the same setting in every segment, which is what those arms
applied, and adds a unit test. It touches only two secondary descriptives (`cap_share` and
`max_band_share`). The primary endpoint, every test, the verdicts and the gains are computed from the
saved fields and are unchanged. No other line of the registered analysis changed.

**Amendment 9, revision 1 — Experiment 3a RESULT (logged 2026-10-07; evaluation 2026-10-06 12:22 to
2026-10-07 05:27 PDT; the registered `analyze_experiment3a.py` was applied with Deviation D1 only).**
All 24 states and 10 arms are complete; no run failed and no state was dropped. Primary endpoint
`J_zonal`, `plan120` vs comparator (Holm family of four):
* H1 vs `plan14`: rel +47% (90% CI +8% to +99%), t p 0.037, Holm p 0.11 → **inconclusive**.
* H2 vs `linear_response`: rel +21% (−7% to +64%), Holm p 0.21 → **inconclusive**.
* H3 vs `pi`: rel +35% (+1% to +86%), t p 0.050, Holm p 0.11 → **inconclusive**.
* H4 vs `planner_average`: rel −56% (−65% to −43%), Holm p 0.0051 → **better**.

The Wilcoxon test disagrees with the t test only on H1 (0.055 vs 0.037), and neither survives Holm.
Every planner and classical arm reduces `J_zonal` by 77-84% against `uncontrolled`. `plan120`
overcools (ocean bias −0.031 K, against −0.008 K for `plan14`). Secondary: `plan60` vs `plan120` −33%
(p 0.011), `plan60` vs `plan14` −2% (p 0.91). The full output is
`mcb_experiments_gpu/exp3a/exp3a_analysis.json`, and the plain-language account is
`MCB_PROJECT_REPORT.md` Part 23. By section 7: the H1 claim is not supported, H2+H4 does not establish
"changing over time beats the best fixed pattern", and the H3 reading is "in a perfect model, feedback
on three indices is enough", which moves the planner's case to Experiment 3b.

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
