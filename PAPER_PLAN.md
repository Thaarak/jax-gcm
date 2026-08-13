# Paper Plan (frozen 2026-08-13)

*Produced by a four-lens assessment (literature/novelty, claims audit, hostile
review, framing) plus independent re-verification of every load-bearing number
against the pickles in `mcb_experiments_gpu/`. Two of the assessment's new
technical claims were re-derived from scratch before adoption (§2 below).*

---

## 1. Decision

**ONE paper. Venue: JAMES.** Methods-framed, not MCB-framed.

**Title:** *Gradients Design, They Do Not Train: Chaos Noise Sets the Power
Floor for Control Experiments in a Differentiable Climate Model*

**Central claim (one sentence):** In a differentiable climate model, per-run
chaos noise — not compile-time nondeterminism, which is exactly zero by
construction — sets a hard statistical floor of 12–17 mK per 60-day
integration that governs what control experiments can resolve; at that floor,
first-order gradients through the model successfully *design* a static forcing
field but neither discover nor improve a feedback policy, while a
two-parameter classical law does both.

**Why not the other framings.** "Differentiable GCM for MCB": the actuator
perturbs the CONVECTIVE cloud-top albedo (`albcl_mcb * cloudc`,
`shortwave_radiation.py:82`) and leaves the stratocumulus analog
(`albcls * clstr`, line 85) untouched — verified in code; nothing simulated is
specific to MCB. "Closed-loop control of ENSO": see §2. "First feedback
control of climate intervention": Lee et al. 2025 owns it.

**Tier-A novel contributions (no prior art found):** (i) the measured chaos
noise floor and the in-process-harness pitfall; (ii) the countermeasure
package (micro-ensembles, paired counterfactual baselines, wild-bootstrap/HC3,
equivalence bounds, the rescaling-invariant slope endpoint); (iii) the
five-exhibit degeneracy catalogue; (iv) the design/train asymmetry of
first-order gradients. **Not novel (cite, do not claim):** feedback beats
mis-specified feedforward (Kravitz et al. 2014); feedback-regulated MCB
(Lee et al. 2025, GRL 52, e2024GL113728); ML controllers for SRM in a GCM
(Quan et al. 2025, JGR-Atmos, zeroth-order RL — the contrast with our
first-order failure is exactly Suh et al. 2022's prediction and is worth a
paragraph); open-loop MCB vs ENSO (Wan 2026 Sci. Adv.; Xing 2025 Earth's
Future). Nearest differentiable-atmosphere precedent: Whittaker & Di Luca
2025 (arXiv:2506.10660), NeuralGCM initial-condition optimization, lr ≤ 1e-9
for stability — an unquantified hint of the pathology we measure.

## 2. Two verified reframing facts (fold into every result statement)

1. **The pacemaker patch is ~2.06% of global area, so its DIRECT contribution
   to the ocean-mean metric is ~29 mK per K of box anomaly** — ~53% of the
   measured step sensitivity (55.3 mK/K) and ~69% of the ramp's (35.5 mK/K).
   Re-derived independently from `nino_pattern` + area weights. The
   disturbance must be described as "a prescribed tropical SST anomaly of
   hidden amplitude", and the paper must publish the decomposition
   (direct vs teleconnected remainder) and report rejection against both.
   The arm COMPARISONS are unaffected (every arm faces the same disturbance);
   the "ENSO teleconnection" language is not defensible.
2. **The actuator is an idealized ocean cloud-albedo intervention, not MCB.**
   `cloudc` global mean 0.713, stratocumulus analog 0.0086, corr −0.43; the
   perturbation lands harder on the warm pool than the Peru deck. Present as
   a characterization result with its own main-text figure (F8), not a buried
   caveat. Cap 0.09 on base 0.43 = +21% relative cloud-top albedo,
   ~3.2–4.7 W m⁻² ocean-mean sustained.

Language rules that follow: "weather realizations", never "climates"
(all ICs branch one equilibrated carry at one 2000-01-01 start;
effective climate n = 1); "idealized ocean albedo intervention", never "MCB",
in title/abstract; every equivalence bound scoped "conditional on a single
ocean state and start date"; result 6 restated as "with a perfect outcome
sensor, feedforward on the disturbance is redundant" (the deployment slogan
needs the observation-noise experiment first); "85–89% rejection" replaced by
the two-measure statement (signed 85.0/67.0%, RMS_A 83.2/66.9%; the 89% was a
signed value — the sign-agnostic range is 69.0–83.2%).

## 3. Abstract (draft, 238 words — verified numbers only)

> Differentiable climate models promise gradient-based design and policy
> learning, but the statistical resolution of experiments run inside them has
> not been measured. Using a fully differentiable JAX atmosphere–slab-ocean
> model (SPEEDY physics, spectral dynamical core, T30) driven by an idealized
> ocean albedo actuator, we measure the noise floor directly. Repeating an
> identical 60-day integration within one process reproduces results
> bit-for-bit; repeating it across processes yields a standard deviation of
> 16.0 mK in the 60-day ocean temperature response, with per-arm per-run
> values of 12.9–14.4 mK. A harness that repeats experiments in-process
> therefore reports the floor as exactly zero and licenses conclusions the
> data cannot support. We show what this floor costs. Gradient descent
> through the model calibrated a static forcing field that transferred to
> twenty held-out weather realizations at −0.1023 ± 0.0014 K against a
> −0.1 K target. But backpropagation through 60-day rollouts did not learn a
> feedback policy: trained controllers were statistically indistinguishable
> from the static field (−1.6 ± 1.9 mK, equivalence bound 4.8 mK) and
> significantly worse than a two-parameter classical law (+8.6 mK, p = 0.007),
> and fine-tuning from a known-good initialization added nothing (−0.3 mK,
> equivalence bound 2.4 mK). We give the countermeasures that restore power
> and a catalogue of five control-experiment designs that are unfalsifiable
> by construction, each caught by auditing our own campaigns.

## 4. Structure (10 sections + supplement)

1. Introduction — contributions (i)–(iv); Metz 2021 / Suh 2022 / List 2024
   predict the pathology qualitatively; nobody has priced it in a coupled GCM.
2. Model, actuator, and plant — **the honesty section, early**: F8, the +21%
   cap, the flux budget, plant identification (scalar, near-integrating,
   measured DC gains). Concede the 1/(1+L) arithmetic here: residual
   8.31/55.26 = 15.0% ≈ 1/(1+6); then lead with what the algebra does NOT
   predict (loop-gain non-monotonicity b12 > b6; warm/cold hinge; cap
   asymmetry).
3. The noise floor (F2) — the paper's core.
4. Countermeasures and their measured effect (F3) — s.e. ×5 reduction;
   bootstrap p-values reported as `p < 5×10⁻⁵ (20 000 draws)`, never `=`.
5. Case study A — design transfers (−0.1023 ± 0.0014 K), demoted to
   *amplitude calibration*: the widening probe shows flat cooling efficiency
   across regions, and the retracted Arctic-peaked pattern also hit target.
   One optimizer run; claim is about this pattern, not the procedure.
6. Case study B — gradients do not train (F6): primaries were null; the PI
   win is a pre-registered secondary (−10.15 ± 2.47 mK, Bonferroni×5
   p = 2.9e-3, replicated on disjoint ICs and η-stream); fine-tune adds
   −0.32 mK (equiv 2.36 mK); imitation arm is n = 1 seed — say so.
7. Case study C — the invariant endpoint (F5, F7): slope invariance
   (blind_loo r = 0.99994), RMS_A co-primary (NOT rescaling-invariant —
   ffmean improved RMS_A 33.8% while rejecting nothing by slope; report both,
   always). Step 85.0/83.2%, ramp 67.0/66.9%; observing the disturbance:
   step bound 2.68 mK/K, ramp 2.89 mK/K.
8. The degeneracy catalogue (F4): no-headroom null; one-sided-disturbance win;
   asymmetric-ablation comparator; mean-|err| dose/adaptation confound; the
   paired-metric-cancelling disturbance (why transient CO2 was never run).
9. Limitations — a scope statement, not a hedge list. Precipitation null to
   supplement (the configuration cannot form the teleconnections it tests).
10. Conclusions.
Supplement: endpoint-evolution table S1 (every hypothesis, every revision,
whether results were visible — the five retractions priced); per-seed spreads;
full prereg + amendment log.

## 5. Figures

F1 plant schematic (gains from `mcb_authority_scoping.pkl`, `*_plant.json`);
**F2 the floor** (in-process σ=0 vs cross-process 16.0 mK;
`noise_floor_crossproc.pkl` + `confirmatory_eval.pkl['noise']`);
F3 power ledger (s.e. vs k with effect-size bands);
F4 degeneracy catalogue, 5 panels;
F5 invariant endpoint (built: `figures/fig1_disturbance_rejection.png` —
relabel "climates" → "weather realizations", retitle per §2);
F6 design-vs-train (built: `figures/fig2_efficacy_uncertainty.png` — mark
imitation n=1 seed, add per-seed points);
F7 both measures × both disturbances (extend `figures/fig3_effects_summary.png`
with RMS_A columns and campaign metadata);
F8 what the actuator touches (`coupled_baseline_atm.nc`: cloudc/cloudstr maps,
r = −0.43, Peru 0.046 / California 0.069 / warm pool 0.727);
S1 endpoint-evolution timeline.
Generator: `make_paper_figures.py` (all panels from pickles; no retyped
numbers).

## 6. The three dangerous reviewer attacks, and the prepared responses

1. **"Your negative measures your optimizer budget"** (40 Adam updates, 1.25M
   params, k=1 gradients while k=8 was used only for eval). Response:
   (a) scope the claim "at this budget" with the budget in the abstract;
   (b) RUN the killer experiment — micro-ensembled-gradient BPTT (2.83×
   gradient SNR) + matched-wall-clock ES control. Every outcome publishable;
   BPTT-fails/ES-succeeds would be the paper's best finding (Suh 2022,
   Quan 2025 contrast).
2. **"Your four load-bearing nouns are not what the code implements"**
   (MCB / El Niño / climates / neural controller). Response: concede all
   four IN the title, abstract, and §2, before the reviewer does; publish the
   area decomposition and F8 as findings. No contribution requires any of the
   four nouns.
3. **"Effective climate n = 1; every p-value is pseudoreplicated."**
   Response: re-scope all inference as across weather realizations (the floor
   measurement and the degeneracy catalogue are unaffected); run
   season-staggered ICs (~5-line change, 4 start dates) — highest
   value-per-GPU-hour item remaining.

## 7. To-do before submission

**(a) No simulation needed (blocking first):**
1. Recover from diya: `stage1_v2/stage1_optimized_pattern.pkl` (loaded by
   every post-rebuild campaign; missing locally), `noise_floor_v2_f32.pkl`,
   `t2b_imitation_gate.pkl`, `t2_widening_probe.pkl`; re-create the ~6 mK
   perfect-controller Monte Carlo as a script (currently single-sourced to a
   document).
2. Purge retracted Amendment-6 numbers from quotable text (report line 428's
   p = 3e-7 is OLS on a retracted design); keep `enso_eval*` only as
   degeneracy exhibits.
3. ~~Correct MCB_META_AUDIT "69–89% sign-agnostic" → 69.0–83.2%~~ (done with
   this commit).
4. Standardize bootstrap p reporting; never quote the stale OLS values in
   `enso7_eval_analysis.pkl`.
5. Rewrite report Part 11 "What's still open" (stale on three counts).
6. ENSO area decomposition as a paper table (§2.1 — arithmetic done, make it
   a table + repo script).
7. Produce F8; build S1; add per-seed spreads to pooled figures
   (enso7 imitation slopes +5.59/+1.79/+7.10 mK/K; enso8/ramp reuse s72 only).
8. Obtain Lee et al. 2025 PDF (gold OA; every fetch got 402/403). The claim
   "no published MCB controller tests mis-specified efficacy" is load-bearing
   and currently rests on absence of evidence. Also verify Xing 2025's
   ENSO-amplitude figure from the paper itself.

**(b) Needs GPU (ranked by claim-load per hour):**
1. Micro-ensembled BPTT + matched ES control (defends the headline; every
   outcome publishable).
2. Matched-forcing pattern comparator (optimized vs uniform-ocean vs canonical
   boxes at equal integrated forcing) — the only test of whether the spatial
   pattern carries information; if uniform ties, C5 becomes amplitude
   calibration, which the flat-efficiency probe already suggests.
3. Season-staggered ICs (4 start dates).
4. Observation noise on feature 0 + H10 re-run (either confirms the
   redundancy result under realistic sensing or reverses it — a reversal
   would be the better finding).
5. x64 gradient-quality check (fix the `lax.scan` carry-dtype mismatch).

Sequencing: (a)1–5 and (b)1–2 before a first draft; (b)3–4 parallel with
drafting; (b)5 optional if C4's scope language is honest.
