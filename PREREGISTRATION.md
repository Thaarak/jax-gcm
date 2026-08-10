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
