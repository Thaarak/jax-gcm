# Retrospective: the advisor-guided campaign arc (2026-08-03 → 2026-08-13)

*An evaluation of everything attempted since the professor's guidance was
assessed on 2026-08-03: what each piece of advice became, what worked, what
failed, why, and what should happen next. Companion to `PAPER_PLAN.md` (the
forward-looking document); this is the backward-looking one.*

---

## 1. The advice → outcome ledger

| Professor's suggestion | What we did | Outcome | Verdict |
|---|---|---|---|
| **ENSO experiment** ("best idea" per the 08-03 assessment) | Four campaigns: pacemaker built and calibrated; Amendment 6 (retracted — win baked in by the one-sided disturbance); Amendment 7 (corrected: zero-mean disturbance, non-gameable slope endpoint); the anticipation ablation (rev 1 withdrawn unrun as a strawman; rev 3 ran); the growing-ramp variant (Amendment 8). | The most productive arc of the project — but not as an ENSO result. Final standing: feedback rejects 67–85% (signed) / 67–83% (sign-agnostic) of the disturbance; **observing the disturbance adds nothing** (bounded < 2.7–2.9 mK/K in both regimes); a growing disturbance costs ~2× the residual. Paper prep then showed ~53–69% of the "ENSO effect" is the nudged patch itself in the area average, and the actuator never touched the stratocumulus deck — so the deliverable mutated from "MCB-ENSO control" into a **methods paper**. | **Right direction, wrong destination — and the destination is better.** The professor's instinct (give the controller a real disturbance) produced the project's sharpest findings. The MCB/ENSO *framing* did not survive contact with our own audits. |
| **Transient CO2** | Enabled the dormant code path, verified it live, quantified the 1950-reference trap (+28.4% instant step), then **rejected it structurally before any GPU spend**: the forcing sits in the physics closure, so it cancels *exactly* in the paired metric and in every controller input. Replaced by the growing-ramp experiment, which asks the same type-0→type-1 question. | Zero GPU wasted; one reusable plumbing commit; the replacement produced result 7. The 08-03 assessment had already flagged the professor's "controller should beat the fixed baseline over time" as a preordained win — the replacement design guarded against exactly that. | **Failed informatively, at zero cost.** The idea was good; the model's experimental machinery cannot express it. Worth telling the professor precisely why. |
| **Precipitation targets** | The "free first step" ran (Amendment 5): the never-analyzed Amazon/Sahel data from three campaigns, paired and Holm-corrected. No detectable effect anywhere; the withdrawn gate stays withdrawn by its own rule; first precip noise floor measured (2.7–3.3 mm/day/run Amazon). | Closed a documented integrity breach. But paper prep concluded the null **carries no information** — the slab configuration structurally cannot form the teleconnections the test targets — so it moves to the supplement. The Pareto-frontier reframe (the genuinely publishable version) was never pursued. | **Done as hygiene, not as science.** The advice's deeper version (T–P trade-off frontier) remains untouched and requires model work first. |
| **PM2.5 / air-quality term** | Not attempted (assessed 08-03 as impossible as stated: the model has no aerosols). The salvageable "exposure constraint" version never rose to the top of the queue. | — | **Correctly deprioritized.** Revisit only after an actuator that touches the right cloud deck exists. |
| **Multi-objective weighting (D-metric)** | Not attempted. | — | **Deferred, reasonably** — it presupposes the precip/teleconnection machinery the model lacks. |

## 2. What went well

1. **The audit-before-write discipline caught five wrong headlines**, three of
   them *before* any GPU was spent once design reviews moved pre-launch:
   Amendment 6's baked-in win (caught post-hoc, ~3.5 GPU-h — the only
   post-spend catch); Amendment 7's inflated p-values and signed-slope
   cancellation (post-hoc, but pre-writeup); the rev-1 strawman comparator
   (pre-spend); the 18× feedforward-coefficient bug (caught on train ICs by a
   gate that existed for a different reason); the CO2 structural cancellation
   (pre-spend, at design review).
2. **Every campaign was reportable regardless of outcome** because
   interpretation grids were pre-stated. No run was wasted even when its
   hypothesis died — Amendment 6's retracted campaign became two degeneracy
   exhibits; the failed tuning sweep validated the bug-fix and measured the
   gain curve.
3. **Instruments compounded.** The pacemaker, the `@FF@FB` arm grammar, the
   deterministic amplitude designs, the wild-bootstrap/HC3 module, the
   held-out comparator selection — each built once, reused by every later
   campaign, each pinned by tests (200+ passing).
4. **Cheap kills.** CO2 died at zero GPU; rev 1 died unrun; the ff bug died on
   six training climates. The expensive failure mode (publishing, then
   retracting) never occurred.
5. **The negative results are the assets.** "Observing the disturbance is
   redundant given outcome feedback" and "first-order gradients design but do
   not train" are more novel than any of the positive control results, per
   the verified literature check.

## 3. What went badly

1. **Amendment 6 launched without a pre-launch design review.** The one-sided
   disturbance flaw was findable by inspection — the review that would have
   caught it was only instituted *afterwards*. Cost: ~3.5 GPU-h plus a full
   writeup-and-retraction cycle. (Process fixed from rev 1 onward.)
2. **Recurring framing drift, all mine, all caught late:** an "oscillation"
   that was a step; "MCB" that is convective-cloud albedo; "climates" that are
   weather realizations; "95% rejection" that is 69–83% sign-agnostic; a
   "wrong-variable constant" that was actually nonlinearity. Each correction
   is now logged, but each was caught by audit rather than at writing time.
   The pattern: **nouns get assigned at design time and never re-verified
   against the code.**
3. **Single-threaded evidence in places that matter:** one equilibrated ocean
   state and one start date under every p-value (climate n = 1); one imitation
   seed reused across the last two campaigns; the stage1_v2 pattern pickle
   missing from the local checkout entirely.
4. **The central negative is still exposed.** Micro-ensembling fixed the
   *measurement* channel but was never applied to the *training gradients* —
   so "gradients do not train" is, today, "gradients did not train at the
   budget we gave them." The defense experiment is specified but not run.
5. **A load-bearing citation was never read.** Every fetch of Lee et al. 2025
   was blocked; the novelty claim ("no published MCB controller tests
   mis-specified efficacy") rests on abstracts and absence of evidence.

## 4. The scoreboard

- GPU spent on the advice-driven arc: **~19 GPU-hours** (scoping 0.2, A-6 3.5,
  A-7 ~4, ablation ~4, ramp 6.7, misc ~0.5). Earlier tiers: ~40.
- Significant results standing: **7** (down-scoped per the claims audit; the
  durable Tier-A core is the noise floor, the countermeasures, the degeneracy
  catalogue, and the design/train asymmetry).
- Headlines retracted or materially corrected by our own audits: **5** — none
  after external exposure.
- Experiments killed before spend: **3** (CO2, rev-1 ablation, and the
  365-day horizon option).

## 5. What would most improve the research (ranked, from §7 of PAPER_PLAN.md)

1. **Micro-ensembled-gradient BPTT + evolution-strategies control at matched
   cost.** Converts the central negative from "at our budget" into either a
   sharp first-order-vs-zeroth-order result or a demonstrated fix. Every
   outcome is publishable. (~GPU, the one that defends the headline.)
2. **Season-staggered initial conditions** (4 start dates, ~5-line change).
   The cheapest possible widening of climate n = 1.
3. **Matched-forcing pattern comparator** (optimized vs uniform-ocean vs
   canonical boxes). Decides whether the one undisputed positive (the
   designed pattern) is spatial information or amplitude calibration.
4. **Observation noise on the sensor + re-run of the redundancy ablation.**
   Decides whether "watching the disturbance is redundant" survives realistic
   sensing — a reversal would be the *better* finding.
5. **Artifact recovery + Lee 2025 PDF** (no compute; blocking for
   reproducibility and the novelty claim respectively).
6. **Longer term:** an actuator on the stratocumulus term (`albcls * clstr`)
   would make "MCB" honest and is the gateway back to the professor's precip /
   multi-objective agenda — but it is model development, i.e. a second paper.

## 6. Questions for the professor

**The two that matter most:**

1. **Framing:** Our audits showed the code implements an idealized
   ocean-albedo intervention (convective deck, not stratocumulus) and the
   "ENSO effect" is ~half direct patch arithmetic. We therefore plan a
   *methods* paper — the chaos-noise floor, the countermeasure kit, the
   catalogue of unfalsifiable designs, and "gradients design but don't
   train" — rather than an MCB paper. **Do you endorse dropping the MCB/ENSO
   framing for paper 1, with the stratocumulus-actuator fix as paper 2 — or
   would you rather we fix the actuator first and keep one MCB-named paper?**
2. **The optimizer-budget defense:** Before claiming gradients can't train
   controllers, we must run micro-ensembled-gradient training plus a
   gradient-free baseline at matched cost. **Do you consider the negative
   publishable without that experiment, and can we get the compute for it?**

**Worth asking if there's time:**

3. Is season-staggering (4 start dates) sufficient to answer the "one climate,
   n = 1" objection at a venue like JAMES, or do we need genuinely independent
   equilibrated ocean states?
4. Given Lee et al. 2025 is the direct predecessor and our result is partly
   "their controller class is all you need — observing the disturbance is
   redundant," is it worth contacting the MacMartin group before submission?
5. For the disturbance experiments: does reporting rejection against the
   *teleconnected remainder* (after subtracting the nudged patch's direct
   share) satisfy the attribution concern, or should the pacemaker be replaced
   with something whose global effect is entirely indirect?
