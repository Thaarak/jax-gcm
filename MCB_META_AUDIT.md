# MCB Meta-Audit: Independent Validation of the Entire Project (2026-07-29)

**What this document is.** A full, independent, adversarial validation of every stage of the MCB
project — the original staged plan, the 2026-07-12 audit, the P0–P2 rebuild, and campaigns v1/v2/v3 —
performed without trusting any project document. Every headline statistic was recomputed from the raw
GPU artifacts (pulled from `diya:~/workspace/jax-gcm/mcb_experiments_gpu/`), every claimed code fix was
verified by reading and (where cheap) executing the actual code, and an 8-lens multi-agent review
(objective alignment, gate statistics, training/selection, physics R6/R7, IC independence, experimental
design headroom, pre-registration compliance, result robustness) was run over the codebase and
artifacts (~1.1M tokens of checking; most findings verified by execution).

**Companion documents:** `MCB_PROJECT_REPORT.md` (plain-language account), `MCB_IMPLEMENTATION_PLAN.md`
(engineering log), `PREREGISTRATION.md` (frozen analysis plan). This document **corrects** specific
claims in the first two.

---

## TL;DR

The project's bookkeeping is impeccable — every recorded gate number reproduces from the raw per-IC
data to 1e-12 — and the headline *negative* result (neural feedback ≈ static pattern ≈ open-loop
schedule) is solid; in fact it is **stronger than the project's own documents state**. But three
things the project currently believes are wrong:

1. **The "held-out RMS halved (0.020→0.011 K), real but underpowered" claim is a selection artifact
   and must be retracted.** It is the minimum of 39 noisy held-out evaluations — the very metric used
   for model selection — and it reverses on independent re-evaluation.
2. **The noise floor is mismeasured.** `σ_compile = 0.0` was guaranteed by the harness's construction
   (within-process reps only). Real cross-process per-run chaos noise is ~0.014–0.017 K — larger than
   σ_IC and 3× the effects under test. The same checkpoint on the same ICs differs by up to 0.049 K
   per IC between two steps of the same campaign.
3. **The feedback-vs-static null was baked into the experimental design.** A mathematically *perfect*
   feedback controller could only ever have been worth ~6 mK on the gate metric — below the n=10
   detection floor. "More ICs and seeds" cannot fix this; a cheap design change can.

There is currently **no statistically significant positive scientific result** in the project — but
there are two or three genuinely achievable ones (Part 4).

---

## Part 1 — What was validated and CONFIRMED

All verified by execution unless noted.

- **The rebuild is real.**
  - R1 area weights: `compute_area_weights` produces cos-lat weights, pole/equator ratio 20.17 on T30.
  - R6 cloud rewire: MCB enters `shortwave_radiation.py:74-83` as a clipped, ocean-masked cloud-albedo
    perturbation whose SW effect scales as `pert × cloudc` (correct Twomey direction). The
    discriminating 3-region regression test (cloudy ocean / clear ocean / land) passes and would fail
    on the old surface-albedo physics.
  - R7 flux split: `JCM.py:131-133` routes sea-slab flux (`-hfluxn[...,1]`) to the ocean and land-slab
    flux (`-hfluxn[...,0]`) to a prognostic slab land. v3 provably ran realistic terrain (explicit
    flag, campaign log lines, orography eval gate, IC-manifest assertion) — no silent aquaplanet.
  - Selection machinery: `select_on_heldout` gates best_params + early stopping on held-out loss
    evaluated every epoch on the pre-update (shipped) params; the off-by-one is fixed and its
    regression test has teeth. Warm start reproduces the static pattern **bit-exactly** (max diff 0.0).
  - Objective alignment: `loss_mode="terminal_dsst"` (`coupled_controller.py:357-374`) is
    line-for-line the same scalar the gates score (same final step, weights, ocean mask, paired
    baseline); the forcing regularizer is a measured 0.3–1% of the loss.
  - 107/107 `jcm/mcb` tests pass; equilibration drift criterion credibly met; no train/held-out
    leakage (disjoint seeds 1000–1019, no shared post-branch segments).

- **The recorded v3 numbers are faithful.** G2 −0.0973±0.0077 (PASS, in band, powered);
  G3 −0.0048±0.0063 (underpowered); G4 +0.000538±0.000444; feedback −0.0054±0.0039; nn-vs-random
  +0.0033±0.0041 — all reproduce from per-IC data to 1e-12. (Sign convention:
  improvement = comparator_err − treatment_err, positive = treatment better. Note both G3 and the
  feedback gate point estimates run **against** the neural controller.)

- **The v3 null verdicts are robust** to proper paired t (p = 0.20–0.46), exact Wilcoxon
  (p = 0.19–0.70), 10k bootstrap (all 95% CIs straddle 0), and (mostly) leave-one-out. The ties are
  genuine ties.

- **v3 training genuinely descended** — the first campaign ever to do so (train-loss trend p<0.001;
  grad norm 0.21 → 0.004). The machinery finally works. (What the descent *bought* is another matter —
  see Part 2.1.)

---

## Part 2 — What was found that the project documents do NOT know

### 2.1 The "held-out loss halved" claim is a selection artifact — RETRACT

`MCB_PROJECT_REPORT.md:172` and the plan's v3 section claim the retrain "genuinely halved" held-out
RMS dSST error (0.020→0.011 K), "real but below the significance bar." Refuted on four independent
grounds (all executed):

- 0.011 K = √(0.000120), the **minimum of 39 per-epoch held-out evaluations** — the very metric used
  for selection and early stopping. The post-warmup held-out curve has **no trend** (Spearman p=0.69
  for epochs ≥5; the linear p=0.002 is driven entirely by epochs 1–4, where training first *degraded*
  the warm start to ~0.0010 and then recovered).
- The plateau (0.000359 ± 0.000120) is statistically identical to the static warm start (0.000394,
  paired p=0.86). A simulated min-of-39-draws null puts the observed "best" at its **58.5th
  percentile** — exactly what pure noise produces. This is audit root-cause R2 reincarnated on
  held-out data.
- Because warm-start = static bit-exactly and selection takes the min over all epochs *including
  epoch 0*, "best ≤ static" was **guaranteed by construction**.
- Cross-checks: the random-init ablation's "best" (9.9e-5) *beat* the warm-started run's (1.2e-4);
  and on both independent re-evaluations of the selected checkpoint (campaign steps 5 and 6), the
  retrain policy measured **worse than static** on the exact trained metric (terminal RMS error
  0.0231 K and 0.0176 K vs static's 0.0136 K). The "static = 0.020" baseline in the claim was itself
  the inflated epoch-0 training-process draw; the actual static pattern scores 0.0136 K RMS.

**Honest restatement:** *no detectable held-out improvement over static at any point in training.*
This strengthens, not weakens, the project's own negative conclusion.

### 2.2 The noise floor is wrong: unmeasured ~0.014–0.017 K per-run chaos noise

- `run_noise_floor.py` loops reps **inside one process** (same compiled program) — the 8 "reps" per IC
  are **bit-identical** in the pickle, so σ_compile = 0.0 was guaranteed by construction. Cross-process
  variance was never sampled; the old "~15% XLA variation" concern was never actually laid to rest.
- Measured cross-process reality: the **same retrain_v3 checkpoint on the same 10 held-out ICs**
  gives per-IC dSSTs differing by up to **0.049 K** (sd 0.024, mean shift 0.007 K) between campaign
  steps 5 and 6 (eval mean −0.0973 vs ablation mean −0.1043 — the docs quote both without flagging
  it). The same static pattern differs RMS 0.0134 K (max 0.032 K) per IC between the v2 and v3 evals.
  Mechanism: 60 days of chaos amplifies any compile-level bit difference to weather-noise scale.
- Consequence for the statistics: the G3 per-IC paired scatter (sd 0.0198) is **statistically
  indistinguishable from what two runs of the SAME policy produce via chaos alone** (F=0.66, p=0.55).
  Pairing on ICs buys almost nothing (policy–static per-IC correlation 0.41); the G3 "tie" is a tie
  between noise realizations. Within-script paired gates remain internally valid (the empirical s.e.
  absorbs this noise), but **every cross-script/cross-campaign number comparison in the docs carries
  ~±0.008 K contamination**, and the "learned requires > 2·σ_compile" rule is vacuous at σ_compile=0.
- float32 itself is NOT the problem for measurement: dSST values are quantized at exactly the f32 ULP
  (2⁻¹⁵ K ≈ 3e-5 K), ~160× finer than the effects; within-process runs are bit-deterministic. The G3
  null is chaos/power-limited, not precision-limited. (The x64/P0.4 loose end remains open only for
  BPTT gradient quality, blocked by the lax.scan f32/f64 carry mismatch.)

### 2.3 The feedback null was baked in by design — the deepest finding

- **Episodes are climatologically identical.** All 20 ICs branch from ONE equilibrated carry with
  0.05 K noise + a 30-day spin; all share ONE calendar date (2000-01-01 + 30d ≈ late January — single
  boreal-winter season, and the model DOES run a real seasonal cycle, so this matters); the rollout
  is fully deterministic (no stochastic terms anywhere in the step path). The slab's SST e-folding
  time (60–570 days, computed from the model's own ρ·cp·h and plausible feedback strengths) is 2–19×
  the 30-day spin, so the ICs sample independent *weather* around **one** ocean state — effective
  n=10 for weather noise, **n=1 for climate/season**.
- **The controller cannot see anything.** At the first control decision, **11 of 13 policy features
  are exactly zero by construction** (paired anomalies vs each IC's own baseline, plus time=0); the
  two absolute-SST features spread ~1e-4 K across ICs. Measured cross-IC action spread at interval 0:
  ≤5.2e-4 albedo vs a 0.09 cap. The only exploitable signal is per-episode chaotic divergence.
- **The ceiling is below the floor.** Executed Monte Carlo with the measured quantities: a PERFECT
  controller (cancelling all correctable variance given the last action at day 45 and slab inertia)
  is worth ~**0.006 K** on the G3 metric — detected only **46%** of the time at n=10 under the
  pre-registered rule; a realistic partial controller needs n≈400. "Underpowered" was the
  near-certain a-priori outcome.
- **The data actively indicate zero realized feedback value.** The trained feedback policy has
  1.5–2× static's cross-IC dispersion (sd 0.018–0.024 vs 0.0129) — a noise **amplifier** — while the
  feature-blind time-only open-loop schedule is numerically the **best** policy overall (mean gate
  error 0.0089 K vs static 0.0122–0.0166 and retrain 0.0142–0.0231, per-process variation). All three
  training configs (warm-start, open-loop, random-init) converge to the same ~1e-4 held-out band, the
  open-loop noise floor.
- The gradient probe's "a gate-improving GENERALIZING direction exists (0.01531→0.01439)" claim
  **fails its own significance standard**: best step improvement +0.0009 ± 0.0046, **p=0.85**; all
  four step sizes p>0.5.
- **Bottom line:** "feedback adds nothing demonstrable" is a property of THIS task distribution, not
  a finding about MCB feedback control. The docs' closing claim that resolving it "would require more
  ICs/seeds, not more code" (`MCB_PROJECT_REPORT.md:190`) is **wrong in the most important way** —
  more samples of the current design can at best bound a ~6 mK ceiling that exists only because the
  environment gave feedback nothing to observe or correct.

### 2.4 Pre-registration compliance is substantially overstated

- **Seeds (worst violation):** PREREGISTRATION §2 requires ≥3 policy-init seeds per configuration
  with numbers reported mean±s.e. over seeds. Every campaign ran **one** seed (hardcoded 42 /
  PRNGKey(0); no `--seed` flag exists; no seed in checkpoint metadata; "seed" absent from the entire
  campaign log). Every controller-level conclusion is a statement about one optimization run.
- **Metric:** §3 froze the dSST metric as the **time-mean over the final 10 days**, "not a single
  day-60 snapshot." The snapshot is what everything uses — training, gates, noise floor, ablations.
  The 10-day-mean was never implemented anywhere. No amendment logged.
- **Tests:** §4 names "paired t / Wilcoxon, α=0.05." Neither is implemented; the code uses a bare
  2·s.e. rule, which at n=10 runs at α≈7.7% (anti-conservative — the docstring's "conservative" claim
  is backwards). Re-running the registered tests on the archived diffs: **v1's celebrated "G3 PASS —
  controller beats static" was never significant** (paired t p=0.0615, Wilcoxon p=0.0645) — part of
  the v1→v2 "reversal" narrative was manufactured by the liberal rule. All v3 verdicts survive both
  tests.
- **G5/teleconnections:** silently dropped — no resolvability control was run, no formal withdrawal
  as §3 requires. Worse, the one precip comparison v3 did compute (legacy gate) **FAILED and went
  unreported**, breaching §5's report-all-runs rule. The "without harming Amazon/Sahel" half of the
  project's stated objective has no v3 verdict at all (and the precip metric is still R4-contaminated
  — see 2.5).
- **Held-out exhaustion:** the identical 10 held-out ICs (seeds 1010–1019, byte-identical across v1
  and v3 manifests) absorbed **13 formal gate tests** (v1/v2/v3) + **5 gradient-probe looks** (which
  chose v3's objective) + **~123 per-epoch selection looks** (which chose v3's checkpoints), then
  scored the final gates. Familywise false-positive exposure ≈49% at nominal α. v3's G2 PASS is a
  third try at the same band after two result-informed redesigns. **Any future "significant" result
  on these 10 ICs is uninterpretable; fresh held-out trajectories are mandatory.** The prereg itself
  contains the design flaw: it mandates held-out selection while gating on the same held-out set,
  with no third split.
- Other unlogged deviations: cap 0.15→0.09, CI=15 vs §1's "2×30-day intervals," x64 A/B never run
  (f32 chosen by default), TOA-W/m² effort metric never computed, no amendment log at all. Nuance in
  the project's favor: v1/v2 violated §5's held-out-selection rule, so the v3 re-run was partly
  *compliance repair*, not pure forking; and the headline v3 conclusions are nulls, which forking
  paths cannot manufacture — the final negative survives.

### 2.5 New code bugs found (none previously documented)

| Bug | Location | Consequence |
|---|---|---|
| `create_latitude_band_mask` compares **degree bounds to radian latitudes** | `state_features.py:177-191` | "Tropical" (−30,30) mask covers **100% of the globe** (executed); (30,60) band is empty. Two of the 13 policy features are exact duplicates of the global anomalies; the "tropics" loss/eval term is actually global; any future latitude-band analysis would be corrupted. |
| Legacy surface-albedo MCB path still live | `mcb_forcing.py`, `speedy/forcing.py:31-41`, `speedy_physics.py mcb_config` | Dormant in v3, but passing `mcb_config` (the more discoverable API) **silently reintroduces root cause R6**. Delete or rewire. |
| Teleconnection precip metric still has the softplus(0) offset | `coupled_loss.py:193-196` | R4 recurring: the metric sits at the do-nothing constant 0.006931 for most cells; plan item 0.5 was never completed despite "fixed" framing. |
| G4 scored on the old summed loss, not the trained metric | `run_stage5_eval.py:100-109` (loss_mode never set → default "summed") | "Gate-aligned training" is true only for G2/G3; G4 judges a quantity of which the trained term is 1–5%. |
| G4 converts "underpowered" into "PASS (not sig. worse)" | `gates.py:104-111` | Contradicts prereg §5 ("never as PASS or FAIL"); v3's G4 mean is numerically *worse* and is one IC (18) from flipping to significantly-worse. The feedback gate is likewise one IC (11) from flipping to "significantly worse than open-loop." |
| Noise floor measured on train ICs, applied to held-out gates | `run_noise_floor.py:135-137` + `run_campaign.sh` | The "gates are powered" calibration was derived from the wrong split (happens to hold empirically). |

### 2.6 Physical-scope caveats (for any writeup)

- Hitting −0.1 K in 60 days required ~3–4.6 W/m² sustained ocean-mean forcing (slab heat-capacity
  math closes) — **at or beyond the maximum published MCB deployment** (~−1 W/m² moderate, −3.7 W/m²
  Latham-type max). ~25% of grid cells sit pinned at the 0.09 cap (near-bang-bang pattern,
  Southern-Ocean-heavy); local instantaneous forcing ~10–15 W/m² is 2–3× the published regional band.
- SPEEDY's **stratiform** cloud albedo term (`albcls·clstr` — the analog of the marine stratocumulus
  real MCB targets) is **not perturbed**; only the convective/total deck at cloud top is. The
  optimizer found the *model*-optimal pattern, which is not necessarily the real-world MCB pattern.
- Scope of every v3 result: single season (boreal winter), single ocean state (n=1), weather-noise-only
  IC variation, 60-day horizon on a shallow slab (near-maximally responsive by construction).

---

## Part 3 — What can honestly be claimed today

1. **Genuine and defensible:** a fully differentiable coupled atmosphere–ocean–land model through
   which BPTT/gradient optimization works end-to-end (Stage-0 AD-vs-FD check; survived everything),
   plus a corrected, honest experimental framework (paired control-relative gates, underpowered
   verdicts, noise-floor thinking — even where execution fell short of the plan).
2. **Defensible with caveats (exploratory, not confirmatory):** differentiable optimization finds a
   physically sensible, on-target static MCB pattern (−0.097/−0.104 K in both realizations, in band)
   — on an unregistered metric, a third look at the same band, single season/ocean state, at a
   forcing scale beyond published MCB feasibility, with the stratiform caveat.
3. **Solid negative (understated by the docs):** the neural feedback controller adds nothing over a
   static pattern or an open-loop schedule; the "real but underpowered improvement" life-raft should
   be retracted; every point estimate runs against the controller; the trained controller amplifies
   variance. AND: the experiment could not have shown otherwise (Part 2.3) — the null is a design
   property, which is itself the most interesting lesson.
4. **Not claimable:** real-world MCB efficacy or deployment guidance; teleconnection safety (metric
   broken, one computed comparison failed unreported); aerosol dose-response/cloud adjustments;
   anything ENSO/AMOC/decadal (structurally absent physics).

---

## Part 4 — Paths to a statistically significant result (ranked)

**Why not just scale the current design:** at the observed ~5 mK effects (which point the *wrong*
way) the power calculation demands ~126–134 fresh held-out ICs for G3 (~43–48 for the feedback gate),
and the perfect-controller ceiling is ~6 mK anyway. Pouring compute into the existing task buys, at
best, a tighter null.

**Cost basis (measured):** 60-day rollout ≈ 16 s on diya post-compile; full 60-cell eval ≈ 16 min;
training epoch (10 ICs, BPTT) ≈ 6 min; fresh IC ≈ 30 s (spin + baseline).

### Tier 1 — Fix the measurement channel (do regardless; ~1 day of code + a few GPU-hours)

1. **Micro-ensembles per (arm, IC):** average k=8 tiny-perturbation replicate rollouts per cell
   (each with its own paired baseline). Since per-IC paired diffs are ~pure chaos noise
   (σ_run ≈ 0.014 K), paired sd drops 0.0198 → ~0.007; detection floor at n=10 becomes ~4.4 mK
   (~3.1 mK at n=20). Full 3-arm + baseline eval over 20 ICs ≈ 3 GPU-hours. **This alone converts
   every "underpowered" verdict into either a real decision or a tight equivalence bound.**
2. **TOST equivalence testing:** with current data the demonstrable bound is only ±12.5 mK (feedback,
   n=10) / ±16 mK (G3); with micro-ensembles it tightens to ~±5 mK — i.e., a *statistically
   significant* bounded-negative: "any feedback benefit is <5% of the target, at 95% confidence."
3. **Fresh confirmatory IC set** (mandatory — the current 10 are exhausted), **≥3 seeds per arm**
   (~36 GPU-h for all arms), the registered **10-day terminal mean** (only ~×0.85–0.95 sd, but it is
   what was frozen), a **cross-process noise-floor harness** (separate processes, not
   `jax.clear_caches`), and the Part-2.5 bug fixes. Log everything as prereg amendments; add a third
   IC split (selection / gating / confirmation).

### Tier 2 — The decisive feedback experiment (~1 week; answers the project's actual question)

4. **Per-episode randomized OBSERVABLE disturbances.** Draw a solar-constant perturbation or
   hemispheric SST anomaly per episode; apply it identically to the policy run and its paired
   baseline (the `mcb_perturbation` carry channel is the exact injection pattern; the absolute-SST
   features already make it observable). The disturbance changes MCB efficacy per episode, so any
   fixed schedule MUST err by the disturbance-response spread — which the experimenter dials to
   5–10× σ_IC — while a controller that reads the state can compensate. The information-theoretic
   asymmetry is airtight: open-loop cannot cancel randomness it cannot see. n=10–20 with
   micro-ensembles is fully powered. Train with domain randomization over the disturbances; choose
   disturbances the slab responds to within 60 days. **Either outcome is significant and
   publishable** — "feedback provably adds value once there is something to correct," or "even with
   dialed-in headroom, BPTT-trained feedback fails to realize it."
5. **Season-staggered episodes** (~5-line change in `run_generate_ics_independent.py`; setup already
   takes `start_datetime`): insolation geometry makes the optimal pattern time-dependent; also fixes
   the single-season scoping. Composes with #4.
6. **Per-episode varying cooling targets** appended to the features (trivial; `target_cooling` is a
   config scalar): "controller tracks a commanded target" is a genuine capability a static pattern
   cannot match; effect size = target spread, dialable to 50 mK. Closer to a capability demo than
   climate science, but near-certain significance.

### Tier 3 — Reframed positive claims (high probability of significance; mostly eval-only)

7. **Optimized pattern vs canonical MCB regions at matched effort:** paired comparison against the
   three literature stratocumulus boxes (SE Pacific, SE Atlantic, NE Pacific) scaled to identical
   total forcing. Expected effect ≫ noise floor → significant at n=10–20. Claim: "gradient-based
   deployment optimization through a differentiable coupled model beats hand-placed literature
   patterns per unit effort, in-model." **Fix the stratiform-albedo gap first** (it likely biases
   against the canonical regions); quantifying how the optimal pattern shifts under that fix is
   itself a publishable physics-sensitivity result.
8. **Dose-response curves and adjoint sensitivity maps** (cooling per unit cloud-albedo by region at
   short horizons where adjoints are well-conditioned), compared against GeoMIP-style efficacy
   rankings — deterministic-derivative results with no statistical-power problem.
9. **A methods paper the project already owns:** "chaos noise, not compile noise, sets the power
   floor for short-horizon geoengineering experiments in differentiable climate models —
   within-process determinism is a false reassurance; micro-ensemble averaging restores power." The
   project's own artifacts are the dataset (σ_run ≈ 0.014–0.017 K finding, σ_compile=0 trap, power
   analysis). Genuinely useful to the differentiable-climate community.

### Recommendation

Do **Tier 1 immediately** (cheap; everything else depends on it), then **#4 as the flagship** — it is
the only path on which the project's original question gets a real answer rather than a foregone
null — with **#7 as the near-certain positive result** to anchor a writeup. In any account of the
work so far: retract the loss-halving claim, correct the noise-floor claim, report the precip-gate
result, and reframe the feedback null as "no headroom by design." That reframing is not a weakness —
it is the most interesting scientific lesson the project has produced.

---

## Appendix A — Key measured quantities

| Quantity | Value | Source |
|---|---|---|
| Per-run cross-process chaos noise (day-60 dSST) | ~0.014–0.017 K/IC (max 0.049 K) | same checkpoint, steps 5 vs 6 of campaign v3; v2-vs-v3 static arms |
| σ_IC (cross-IC, fixed pattern) | 0.0129–0.0142 K | noise-floor pickles |
| σ_compile (within-process) | 0.0 exactly (bit-identical reps) | noise_floor_v2_f32.pkl — artifact of harness design |
| G3 paired-diff sd / feedback-gate sd | 0.0198 / 0.0123 K | eval_v3, ablation_compare_v3 |
| Effects under test | ~0.005 K (against the controller) | G3 −0.0048, feedback −0.0054, nn +0.0033 |
| n for 80% power at 5 mK | ~126–134 (G3), ~43–48 (feedback) | noncentral-t, executed |
| Perfect-controller ceiling on G3 | ~0.006 K (46% detection at n=10) | Monte Carlo, executed |
| Trained-feedback vs static cross-IC dispersion | 0.018–0.024 vs 0.0129 K | eval_v3 + ablation |
| Best policy by mean gate error | open-loop 0.0089 < static 0.0122–0.0166 < retrain 0.0142–0.0231 K | per-process |
| Rollout / epoch / IC-gen cost (GB10) | 16 s / ~6 min / ~30 s | campaign_v3.log timings |
| Micro-ensemble k=8 detection floor | ≥4.4 mK (n=10), ≥3.1 mK (n=20) | σ√(2/k)/√n |
| TOST bound achievable now → with micro-ensembles | ±12.5 mK → ~±5 mK (feedback) | executed/hand-verified |
| Held-out IC looks consumed (v1–v3) | 13 gate tests + 5 probe + ~123 selection | prereg-compliance lens |

## Addendum — Confirmatory campaign results (2026-07-31)

The Tier-1 confirmatory campaign (`run_campaign_confirm.sh`, PREREGISTRATION.md Amendment 2) ran
2026-07-30/31 on diya (2.74 h for the main eval): 20 FRESH ICs (seeds 3000–3019, never previously
touched), k=8 micro-ensembles per (arm, IC) with per-member in-process baselines, the registered
final-10-day time-mean dSST metric, and formal paired t / Wilcoxon / TOST. All numbers below were
independently recomputed from `confirmatory_eval.pkl` member-level data and match the campaign's
recorded gates exactly.

**The measurement fix worked as designed.** Arm-mean standard errors dropped from ~0.007 K (v3) to
**0.0014 K** — a 5× improvement, exactly the k=8 × n=20 prediction. The micro-ensembles directly
measured the per-run chaos noise at 0.0129–0.0144 K, and the cross-process noise floor
(`noise_floor_crossproc.pkl`) measured **σ_run = 0.0160 K** where the old harness reported 0.0 —
both confirming the meta-audit's 0.014–0.017 K diagnosis. Residual real IC-to-IC response
heterogeneity is ~0.004 K.

**Results on the registered metric (n=20 fresh ICs):**

| Arm | dSST (10-day mean) | G2 | mean \|err\| |
|---|---|---|---|
| stage1-static | **−0.1023 ± 0.0014** | PASS (powered) | 0.0054 |
| retrain (v3 NN feedback) | −0.0938 ± 0.0015 | PASS | 0.0070 |
| open-loop (time-only) | −0.0914 ± 0.0014 | PASS | 0.0091 |

**Primary comparisons (Holm-corrected family of two):** retrain vs static p_t=0.274 (Holm 0.32);
retrain vs open-loop p_t=0.161 (Holm 0.32). Both ties — but now with **tight TOST equivalence
bounds: any feedback effect vs static is within ±0.0043 K, and vs the open-loop schedule within
±0.0045 K, at 95% confidence** (≤4.5% of the target signal). Combined with the design-headroom
ceiling (~0.006 K for a perfect controller), the feedback question is now CLOSED for this task
distribution, quantitatively: neural feedback cannot and does not add measurable value here.

**Secondary:** open-loop vs static +0.0037, p_t=0.057 / p_w=0.058 — the 2·s.e. gate labels this
FAIL but the registered paired t does not reject at α=0.05 (the known anti-conservative
discrepancy; per Amendment 2 the t-test governs and the disagreement is reported). Directionally,
the static pattern had the smallest error of all three arms. Leave-one-out: the retrain-vs-static
tie is fully stable; retrain-vs-openloop can flip to significant (retrain better) on dropping one
IC — noise around zero, consistent with the equivalence bounds.

**The two defensible, protocol-clean results of the project are therefore:**
1. **Confirmatory positive:** gradient-based optimization through the differentiable coupled model
   yields an MCB pattern that hits the −0.1 K target on never-touched ICs on the registered metric
   with real statistical power (−0.1023 ± 0.0014 K, G2 PASS). Caveats: single season, single ocean
   state, in-model forcing scale beyond published MCB feasibility, stratiform deck unperturbed.
2. **Significant bounded-negative:** any advantage of the neural feedback controller over a static
   pattern or an open-loop schedule is smaller than ~4.5 mK (95% CI), i.e. under ~4.5% of the
   target effect — in an environment whose theoretical feedback ceiling is ~6 mK. The Tier-2
   disturbance-injection experiment remains the path to testing feedback where it has real headroom.

Remaining protocol caveat: the trained arms are the single-seed v3 checkpoints (re-evaluated, not
retrained); Amendment 2's ≥3-seed rule applies to any future training-based claim.

## Addendum 2 — Tier-2 campaign results: feedback under uncertain efficacy (2026-08-01/02)

The Tier-2 campaign (`run_campaign_tier2.sh`, PREREGISTRATION.md Amendment 3 + pre-launch revisions)
ran 2026-07-31→08-01 on diya (~26 h): per-episode unobserved efficacy η ~ U[0.6, 1.4] (antithetic,
mean exactly 1.0), 20 fresh confirmatory ICs (seeds 5000–5019), k=4 micro-ensembles with in-process
member baselines, registered 10-day tail metric, 8 arms. The manipulation check passed its
pre-registered gate (static degrades to 19.8 mK mean error under η-uncertainty vs 5.4 mK at η=1).
All numbers below were independently recomputed from member-level raw data and adversarially
verified (2-lens review, wf_7a76e9e1; the PI headline additionally survived LOO over all 20 ICs
(max p=0.0013), removal of the 3 most extreme η draws, permutation and bootstrap tests, the
alternative snapshot metric (p=1e-5), and Bonferroni over all 5 secondary comparisons (p=0.0029)).

**Results on mean |dSST_10d − target| (n=20 paired ICs):**

| Arm | mean error | vs static |
|---|---|---|
| static pattern (η-blind) | 20.2 ± 2.5 mK | — |
| open-loop schedule (trained, pooled 3 seeds) | 19.7 ± 2.2 mK | n.s. |
| NN feedback (trained, pooled 3 seeds) | 18.7 ± 2.2 mK | −1.6 mK, p=0.42 (H2 NULL) |
| **PI/deadbeat (hand-designed, no training)** | **10.1 ± 1.6 mK** | **−10.2 mK, p=5.9e-4** |

**Primary hypotheses (Holm): both NULL.** H2 (NN vs static): −1.6 mK, p=0.42, |effect| < 4.8 mK
(TOST 95%). H3 (NN vs open-loop): −1.0 mK, p=0.065 (Holm 0.13), |effect| < 1.9 mK.

**The two significant Tier-2 findings (both pre-registered secondaries, reported as such):**

1. **Feedback control demonstrably compensates uncertain MCB efficacy — the project's first
   significant feedback win.** The hand-designed deadbeat controller halves the error
   (20.2 → 10.1 mK; p=5.9e-4, Wilcoxon 7.1e-4, Bonferroni×5 = 0.0029; 16/20 ICs), with the
   mechanism physically verified: its command correlates −0.94 with η (monotone down-modulation, no
   overshoot, member spread 11 vs 15 mK). The pre-registered asymmetric-authority prediction
   reproduced exactly: near-perfect 4.5 mK on η≥1 episodes (where reducing the command suffices)
   vs 15.7 mK on η<1 (where the cap on the bang-bang pattern blocks up-modulation — an actuator
   constraint, not physics: the widening probe measured cooling efficiency flat at −3.4 to −3.6
   K per unit forcing across deployments). Scope caveat: the controller reads a noiseless
   ocean-mean dSST vs the paired counterfactual baseline — an idealized observer — so 10.1 mK is
   an upper bound on deployable performance; this bounds external validity only, and the NN arms
   consumed the same features.

2. **The BPTT-trained neural controller is significantly WORSE than the hand-designed controller**
   (+8.6 mK, p=7.0e-3) and indistinguishable from static. Diagnosis (verified from training
   histories + behavior): the failure is in TRAINING, not information or authority — PI achieves
   10.1 mK from two features the NN also receives. All six training runs' losses sat flat at the
   static-under-η noise floor (~5–10e-4 vs the PI-equivalent 2e-4; cross-IC gradient coherence
   ~0.45; per-epoch η redraws added ~24 mK loss variance on top of ~12 mK rollout chaos, burying
   the feedback gradient). Held-out selection over 6 fixed ICs added winner's curse (an open-loop
   arm, structurally incapable of feedback, was "selected" at held-out loss 8.5e-5 yet evaluated at
   21.4 mK). The resulting policies modulate in the CORRECT direction (command-vs-η r = −0.64 to
   −0.74 in all seeds — genuine but vestigial feedback) at 5–10× too little amplitude; only seed 43
   (~26% η-rejection) beat static (14.4 mK). A 2–6% warm-start-inherited thermal over-gain (shared
   by the open-loop arms) explains the stratified better-at-low-η/worse-at-high-η pattern.

**Honest headline for the project:** *In a differentiable coupled GCM with pre-registered protocol,
classical feedback control halves the error induced by realistic seeding-efficacy uncertainty, while
gradient-through-the-model trained neural controllers — the project's founding bet — fail to realize
the same gains, not for lack of information or actuator authority but because BPTT through 60-day
chaotic rollouts does not converge at practical budgets.* Combined with Tier-1 (static pattern
on-target; feedback worthless when there is nothing to correct, bounded within ±4.5 mK), the project
now has three defensible, statistically significant results and a mechanistic account of each.

**Follow-up with the highest leverage (from the verified diagnosis):** initialize the NN by
supervised imitation of the PI law (a noiseless regression onto the same features — bypasses the
chaotic-rollout gradient entirely, lands the policy in the 10 mK basin), then fine-tune with
η-marginalized objectives and a much larger selection set; the open question it answers is whether
a learned policy can exceed PI by widening deployment on low-η episodes (the physical headroom the
probe confirmed exists and the cap denies to pattern-scaling controllers).

Artifacts: `diya:mcb_experiments_gpu/{tier2_eval.pkl, tier2_eval_analysis.pkl,
t2_manipulation_check.pkl, t2_widening_probe.pkl, t2_feedback_s4*, t2_openloop_s4*, ics_t2_train,
ics_t2_eval}`, `campaign_tier2.log`; local copies + verification transcripts in the session
scratchpad (`tier2/`, workflow wf_7a76e9e1).

## Addendum 3 — Tier-2b results: PI-imitation initialization (2026-08-02/03)

The Tier-2b campaign (Amendment 4 + revision 1) tested the verified Tier-2 diagnosis by initializing
the NN via supervised distillation of the PI law. Attempt 1 was stopped in 23 minutes by the
pre-registered imitation gate: the distilled net behaved exactly like static (19.4 mK) because the
real model's absolute-SST features (measured f11 = −1.83, f12 = −4.23) sat 3.7–8.5σ outside the
synthetic sampling. After the fix (sampling anchored at the measured operating point; training on
standardized features folded back into the first-layer weights; regression test at the real
operating point), attempt 2 passed every gate — imitation-only scored 10.7 mK on validation,
already at PI level — and ran to completion (~11.5 h). Eval: 20 FRESH ICs (seeds 6000–6019, first
confirmatory use), fresh antithetic η stream (seed 930), k=4 members, registered tail metric.
All numbers independently recomputed from per-IC data (match to the recorded analysis).

**Results, mean |dSST_10d − target| (n=20):** static 19.5 ± 2.7 mK; **PI 9.3 ± 1.6**;
**imitation-only 8.2 ± 1.5**; **fine-tuned NN (pooled 3 seeds) 7.9 ± 1.2** (per-seed 6.9/8.5/8.2 —
tight, unlike Tier-2's direct-BPTT spread).

**Primary hypotheses (Holm):**
- **H5 — the fine-tuned neural controller SIGNIFICANTLY beats the static pattern: −11.6 mK,
  Holm p = 2.5e-4** (Wilcoxon 2.1e-4). The project's founding claim — a neural feedback controller
  demonstrably outperforming a static deployment — is finally supported, with the strongest
  statistics of the entire effort.
- **H4 — fine-tuned NN vs PI: NULL** (−1.5 mK, p = 0.20; TOST |effect| < 3.4 mK). Learning did not
  demonstrably exceed the hand-designed law.

**Secondaries (pre-registered):** imitation-only ≈ PI (−1.2 mK, n.s., equivalence < 3.2 mK) —
distillation transferred fully; **fine-tuned ≈ imitation-only (−0.3 mK, n.s., equivalence
< 2.4 mK) — gradient fine-tuning added essentially nothing**, the pre-stated outcome "the
chaos-gradient bottleneck persists even from a good basin." The η<1-stratified fine-tuned-vs-PI
comparison (the widening hypothesis) is suggestive but not significant (−2.9 mK, p = 0.053).
Behaviorally, all learned arms carry full-strength feedback (command-vs-η r = −0.91 to −0.95,
matching PI's −0.93 — versus direct BPTT's vestigial −0.64 to −0.74 in Tier-2). PI-beats-static
replicated for the third time on a third independent IC set and η stream (9.3 vs 19.5 mK).

**Tier-2b's contribution to the honest headline:** *a neural feedback controller CAN significantly
beat static deployment under efficacy uncertainty — but the capability was put there by imitating a
classical controller, not by gradient training through the climate model. Differentiable-simulator
gradients neither found the feedback law from scratch (Tier-2) nor measurably improved upon it once
given it (fine-tuning ≈ imitation, bounded within 2.4 mK).* The differentiable-GCM's demonstrated
optimization value remains spatial-pattern design (Tier-1); its BPTT policy-gradient value at these
horizons is, on this evidence, bounded near zero — with the ~12 mK/run chaos noise as the measured
mechanism.

Artifacts: `diya:mcb_experiments_gpu/{tier2b_eval.pkl, tier2b_eval_analysis.pkl,
t2b_imitation_gate.pkl, t2b_imitation_s5*.pkl, t2b_finetuned_s5*, ics_t2b_eval}`,
`campaign_tier2b.log`; local copies in the session scratchpad (`tier2b/`).

## Addendum 4 — Retrospective precipitation side-effect analysis (2026-08-07)

Closes the two §2.4 precipitation gaps: the per-(IC, member, arm) `amazon_mm_day` / `sahel_mm_day`
values recorded by every Tier-1/2/2b evaluation but never aggregated, and the v3 legacy Gate 3 that
was computed FAIL and went unreported. Protocol and status (exploratory, not confirmatory — the
recorded quantity is a day-60 snapshot, not the §3-registered interval mean) are frozen as
PREREGISTRATION.md Amendment 5; analysis in `analyze_precip_sideeffects.py` (+ tests), zero new
GPU-hours.

**Headline: no detectable rainfall side-effect anywhere, and G5 stays WITHDRAWN by its own rule.**
Across all three campaigns (n=20 fresh ICs each), both regions, and every arm — static, PI,
BPTT-trained, imitation, fine-tuned — no regional precipitation change is distinguishable from
zero after Holm correction (24-test family, min raw p = 0.044 → Holm p = 0.965; the one nominal
hit was a Tier-2 *wetting*, direction opposite to harm). No controller differs from static on
either region (16-test family, all n.s.). The §3 resolvability rule therefore keeps G5 withdrawn;
what the data support instead are equivalence bounds:

| Campaign | Amazon |effect| bound @95% | Sahel |effect| bound @95% |
|---|---|---|
| Tier-1 (k=8) | 0.64–0.76 mm/day | 0.06–0.07 mm/day |
| Tier-2 (k=4) | 0.60–1.21 mm/day | 0.10–0.18 mm/day |
| Tier-2b (k=4) | 0.56–0.87 mm/day | 0.08–0.21 mm/day |

Measured per-run snapshot chaos noise: **2.7–3.3 mm/day (Amazon), 0.4–0.8 mm/day (Sahel)** — the
first quantified precip noise floor in the project, and the quantitative reason any future precip
gate needs the registered interval-mean metric plus micro-ensembles (a single-day regional snapshot
carries ~½ σ of pure weather at k=4–8). Context for the bounds: Jan–Feb episodes put the Amazon box
in wet season (several mm/day climatology — the bounds are ~10–20% of the regional mean, honest but
not tight) and the Sahel box in dry season (near-zero climatology — the tight bounds partly reflect
that). Masks are lat-lon boxes not intersected with land (some Atlantic cells in both).

**v3 breach closed.** The legacy Gate 3 rule was "stage5's weighted regional precip penalty ≤
stage4-warmstart's" — bare means over all 20 ICs (train and held-out mixed), no significance test,
on the R4-floored softplus metric this audit's predecessor had already deprecated. It evaluated
FALSE (1.23e-3 vs 1.07e-3) and was never reported. Reanalysis: the paired difference is
+0.0033 ± 0.0049 (p_t = 0.51, p_w = 0.50) — a coin flip inside noise; on the held-out split alone,
stage5 is numerically (not significantly) better than the control on BOTH regions (Amazon
−0.0024 ± 0.0064, p = 0.72; Sahel −0.0014 ± 0.0012, p = 0.25). The unreported FAIL concealed no
real effect; the breach was in the non-reporting, not in any hidden harm.

Artifacts: `diya:mcb_experiments_gpu/{confirmatory_eval.pkl, tier2_eval.pkl, tier2b_eval.pkl,
eval_v3.pkl}` (pulled locally to `mcb_experiments_gpu/`), output `precip_sideeffects.json`.

## Addendum 5 — ENSO campaign results, and the audit that overturned its headline (2026-08-08/09)

The Amendment-6 ENSO campaign ran clean (~3.5 h, all gates passed) and its pre-registered
primaries came out overwhelmingly significant. A five-lens adversarial verification then
established that **the primaries are degenerate** and the honest result is a different quantity
entirely. Every claim below was independently recomputed from the raw member-level cells; where
the audit and the campaign disagree, the audit wins. Artifacts: `enso_eval.pkl`,
`enso_eval_analysis.pkl`, `enso_eval_mechanism.pkl`, `enso_{gcal,gmanip,gimit}.pkl`,
`enso_scoping.pkl`, `mcb_authority_scoping.pkl`.

**What ran.** 180-day episodes, 12 control decisions, registered metric = final-60-day mean
global-ocean dSST vs a paired NO-ENSO baseline, target −0.1 K. A relaxation pacemaker imposed an
El Nino of hidden per-episode amplitude A ~ U[0.5, 2.0] K on the arm rollouts only. Arms: static
(ENSO-blind), ffmean (open-loop schedule compensating the mean amplitude), pienso (classical
feedforward + deadbeat proportional), imitation x3 seeds (fc14 MLP distilled from pienso). n = 20
fresh held-out ICs (seed0 7000, indices 6–25), k = 4; gates used the disjoint train ICs 0–5 and a
separate amplitude stream (941 vs 940). No BPTT.

**Reported result (all arithmetic verified correct, scipy vs the repo's stats agreeing to machine
precision):** mean |miss| static 87.8, ffmean 52.7, pienso 30.5, imitation 29.1 mK; H6
pienso-vs-static −57.3 mK (Holm p = 4.6e-8); H7 imitation-vs-static −58.7 mK (Holm p = 3.3e-8);
imitation-vs-pienso null (equivalence < 4.6 mK); pienso-vs-ffmean −22.2 mK (p = 0.0033). H6/H7
survive every robustness attack (worst leave-one-out p = 1.7e-7; bootstrap CI [−69.9, −44.7] mK;
the unregistered snapshot metric strengthens them). The numbers are right. The *inference* is not.

### 5.1 The primaries are degenerate — an ENSO-BLIND controller ties the feedback controllers

Fit a blind counterfactual: the same static pattern at one retuned scalar gain, carrying no ENSO
information whatsoever, with its gain tuned leave-one-IC-out (no test-set peeking) and its chaos
residual held fixed (dose scales the MCB response, not the weather noise). It scores **31.4 mK**,
statistically tied with pienso (30.5; diff −0.9 mK, p = 0.85) and with imitation (29.1; −2.3 mK,
p = 0.67). The registered mean-|miss| metric therefore **cannot distinguish adaptive feedback from
a better-tuned open loop.**

The mechanism is the metric's blindness to overshoot combined with a one-sided disturbance. Not
one of the 20 per-IC values for static, pienso or any imitation seed crosses the target (ffmean
2/20), so |dSST − target| is exactly linear and H6/H7 are algebraically identical to a paired test
on raw dSST — "which arm cooled more". Because the imposed El Nino always warms, its **mean** is
non-zero, and compensating that mean by a constant is sufficient. H6/H7 were guaranteed at design
time: static's structural bias is ~4.8x the minimum detectable effect. This is the mirror image of
the Tier-1 finding that a null was baked into the design — here a **win** was baked in.

### 5.2 Root cause: excluding La Nina, on a justification measured at the wrong horizon

Amendment 6 excluded La Nina because the scoping run showed a weak, non-monotonic cold response.
That measurement was taken at **365 days / tail-90** (El Nino +241 mK vs La Nina −27 mK). At the
campaign's **own registered horizon**, 180 days / tail-60, the scoping data show a near-symmetric
response: **El Nino +104.5 mK, La Nina −98.7 mK.** The exclusion was not justified at the horizon
actually used, and it is the design-level cause of §5.1: a zero-mean disturbance would have left
no mean offset for a constant gain to absorb, making H6/H7 genuine tests of adaptation.

### 5.3 The one non-tautological comparison does not survive either

pienso-vs-ffmean (−22.2 mK, p = 0.0033) rests on ffmean having been handed the wrong plant
constant. `enso_effect_per_K = 0.0525` was measured on atmospheric **global-mean surface air
temperature**, while the registered metric is **global-ocean dSST**, whose campaign-measured
sensitivity is **0.0716 K/K — 1.36x larger**. Re-specifying ffmean in metric space moves it from
52.7 to 27.5 mK and erases pienso's advantage: **+3.0 mK, p = 0.58.** Consistently, the as-run
advantage lives entirely in the high-amplitude half (−45.0 mK, p = 8.6e-5) and is exactly null in
the low half (+0.7 mK, p = 0.89) — the fingerprint of a one-sided gain deficit, not of per-episode
adaptivity. **The claim "feedback beats a mis-specified feedforward schedule" must be withdrawn**
and restated narrowly: feedback is *robust to* a 36% plant-model error that cripples an open loop.

### 5.4 The residual miss is a law/metric mismatch, not physics

No arm reaches the target, and this is not an authority or lag limitation (the cap never binds;
dose linearity 1.05; 3x headroom). The deadbeat law regulates the **instantaneous** dSST onto a
linear ramp reaching the target at day 180, while the metric averages days 121–180. A perfect
tracker of that reference scores −0.1 x mean(t/180) = **−0.0836 K, a structural +16.4 mK miss.**
Observed A-independent level: pienso −0.0834 K (+16.6 mK), imitation −0.0881 K (+11.9 mK). The
prediction matches to 0.2 mK. Fixing the reference to the tail-mean is a few lines.

Relatedly, the "~20 mK shared calibration deficit" is **not established**: the G-cal gate (6 train
ICs) gives static-without-ENSO at −0.0803 K, but the campaign's own 20-IC extrapolation to A = 0
gives −0.1016 ± 0.0118 K. Two disjoint IC sets disagree by 21 mK. Either way it does not inflate
H6/H7 (it cancels in the paired difference; H6 moves from −57.3 to −56.6 mK when re-scored at the
achievable target).

### 5.5 What survives: disturbance rejection

Regress each arm's per-IC error on the hidden amplitude. The slope is the arm's sensitivity to the
disturbance, and **no blind arm at any gain can change it** (verified: gains 1.0–3.0 leave static's
slope at +71.6 to +76.5 mK/K). This measure is immune to the mean-offset degeneracy, to the
calibration dispute, and to the |·| linearity:

| Arm | disturbance sensitivity (mK per K of Nino3.4) | rejection |
|---|---|---|
| static (blind) | +71.6 ± 8.9 | — |
| ffmean (open-loop schedule) | +81.1 ± 8.7 | none |
| **pienso (classical feedback)** | **+11.1 ± 4.3** | **84%** |
| **imitation (distilled MLP, 3 seeds)** | **+13.8 ± 4.0** | **81%** |

Paired slope difference pienso-vs-static: **−60.5 ± 7.6 mK/K, p = 2.6e-7.** The residual
amplitude dependence is itself significant (+11.1 ± 4.3, p = 0.019), so rejection is 84%, not
complete. Cross-episode dispersion falls 34.8 → 9.2 mK (F = 14.3, p = 3e-7) to the chaos floor
(8.9 mK), i.e. essentially all amplitude-driven variance is gone. Note that **ffmean rejects
nothing** (+81.1, indistinguishable from static): compensating the average disturbance lowers the
mean error while leaving episode-to-episode tracking untouched. That contrast — not H6/H7 — is the
campaign's scientific content.

**Mechanism, verified directly.** The pacemaker delivered as commanded (realized-vs-commanded
r = 0.998–0.999, slope 0.97–0.99, arm-independent). Commanded forcing tracks the hidden amplitude
at r = +0.91 (pienso) and +0.86–0.89 (imitation seeds), and is exactly constant for static and
ffmean. Extrapolated to A = 0 the feedback arms command static's own dose (0.0095 vs 0.0097), so
they are not simply dosing more. The distilled networks match the hand law within ±2.9–4.6 mK —
the fourth independent replication that imitation transfers a classical law faithfully and does
not exceed it.

### 5.6 Other audited findings

- **The chaos floor was understated ~3x.** Amendment 6 assumed 13 mK/run (a 3-dof estimate from
  one base IC); the campaign's realized per-run figure is 35–40 mK for open-loop arms and ~18 mK
  for feedback arms, making the realized comparison s.e. 1.8x the design assumption. This did not
  threaten the 8-sigma primaries but is why the one interesting comparison landed at ~3.4 sigma.
- **Both statistical traps this project has fallen into before are absent**: the analysis clusters
  at the IC level (a run-level s.e. would have been 1.4–1.6x too small) and applies |·| after
  member averaging, not before.
- **A pre-run gate already showed the target was unreachable** (G-imit: pienso at −0.062 K) and the
  campaign proceeded regardless. Future amendments must gate on the arm under test landing in
  band, not only the comparator.
- **k = 2 gates are one bad draw from a wrong verdict**: one G-cal no-ENSO static run landed at
  +0.0243 K, a >120 mK excursion. Gates should use k >= 4.
- **Precipitation** (now available for 180-day episodes with a large disturbance): no Amazon or
  Sahel effect survives correction for any arm, consistent with Addendum 4.
- Pre-registration compliance was otherwise exact: horizon, interval, metric, cap, n, k, arms,
  amplitude streams (940 eval / 941 gates), held-out/train IC separation (zero overlap), no BPTT,
  and `analyze_enso.py` unmodified since the freeze commit.

### 5.7 Honest status

**Retracted:** "a feedback controller significantly outperforms static deployment under ENSO
variability" (H6/H7 — degenerate; a retuned blind gain ties it) and "feedback beats a
mean-feedforward schedule" (pienso-vs-ffmean — an artifact of a wrong-variable constant).

**Supported:** *classical and distilled feedback controllers reject ~81–84% of an imposed ENSO's
effect on global-mean ocean temperature (slope +71.6 → +11.1 mK/K, p = 2.6e-7), while an open-loop
schedule rejects none — and the distilled neural controller matches the classical law to within
±3 mK.* This is a real, mechanism-verified, non-degenerate result, and it is the first closed-loop
MCB-against-interannual-variability measurement in the literature.

**The corrected experiment is cheap** (~4 GPU-h) and is specified in the Amendment 6 post-mortem
in PREREGISTRATION.md: include La Nina (zero-mean disturbance, killing the mean-offset
degeneracy), re-derive the plant constants on the registered metric, fix the deadbeat reference to
the tail window, pre-register disturbance sensitivity as the PRIMARY endpoint, and add the
retuned-blind-gain arm as a registered control.

## Appendix B — Provenance

- Raw artifacts pulled 2026-07-29 from `diya:~/workspace/jax-gcm/mcb_experiments_gpu/` (eval_final/v2/v3,
  ablation_compare_v2/v3, gradient_probe_v3, noise floors, retrain_v3 + ablation histories, stage1_v2
  pattern, campaign_v3.log) to the session scratchpad (`diya_artifacts/`).
- Validation: 8-lens multi-agent workflow (run wf_7fb7bebb) + inline recomputation; most findings
  verified by executing code against artifacts (venv `.venv/bin/python`, jax 0.10.0, numpy 2.4.4).
- The planned ideation subagents were interrupted by a session limit + infrastructure outage; the
  future-experiment analysis in Part 4 was synthesized inline from the completed lenses' quantitative
  findings (esp. the design-headroom lens's executed Monte Carlo and codebase-support assessments).
- Full per-lens findings: session transcript
  `subagents/workflows/wf_7fb7bebb-87f/` and the workflow output file; condensed working notes in the
  scratchpad (`inline_stats_findings.md`, `synthesis_draft.md`).
