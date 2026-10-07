# The MCB Project: A Plain-Language Report

*A beginner-friendly account of the whole effort — what we set out to do, what we built, what worked, what didn't, and what we honestly know now. No prior background assumed. Every technical term is explained the first time it appears.*

*Companion to the detailed engineering log in `MCB_IMPLEMENTATION_PLAN.md` and the independent validation report `MCB_META_AUDIT.md`. Written 2026-07-28; updated 2026-08-03 to cover the second audit and the Tier-1, Tier-2, and Tier-2b campaigns (Parts 7–11); updated 2026-08-07 with the rainfall side-effect analysis (Part 12); updated 2026-08-09 with the ENSO campaign and its audit (Part 13); updated 2026-08-12 with the anticipation ablation and the growing-disturbance test (Parts 14-15); updated 2026-09-29 with a fresh-eyes review of the whole project and the two closest published papers (Part 16), and with the new research direction — months-long gradients — and its Step 0 preparation (Part 17); and, the same day, with the step-by-step plan for the experiments and code that follow from Dubey et al.'s approach (Part 18); updated 2026-09-30 with the Q-flux that fixes the too-cold ocean: the first attempt failed its pre-written check, and the one registered correction passed (Part 17, Step 0); updated 2026-10-01 with interleaved training and evaluation states and the maps from Experiment 1 (Part 18 steps 2–3; Amendment 9 revision 0.4), and with a map of every publishable idea, how finished it is, and how they combine into papers (Part 19); updated 2026-10-01 with the test world for Experiments 2-3 (Part 18 steps 10-12); updated 2026-10-02 with the planner (Part 18 steps 13-16); updated 2026-10-02 with Experiment 1's result and four follow-up checks (Part 20) and with the advisor meeting and the current paper plan (Part 21); updated 2026-10-05 with measured planning costs (Part 18 step 17) and with the methods of steps 18–22, built but not yet run; updated 2026-10-05 with the step-23 pilot, which fixes the planner's settings (Part 22); updated 2026-10-07 with Experiment 3a, the pre-registered test of the planner against its opponents (Part 23).*

---

## The one-paragraph version

> **Latest status (2026-10-07):** Experiment 3a (**Part 23**, registered at https://osf.io/7bwe4/)
> finished. Every sensible controller removes about 80% of the warming's latitude pattern, but the
> 120-day planner does **not** beat a 14-day planner, the classical feedback controller or the best fixed
> pattern. It overshoots, and the data lean against it, though not past the pre-registered bar. Next is
> Experiment 3b (uncertain brightening strength).
>
> *Earlier (2026-10-05):* Experiment 1 passed (outcome A′; Part 20). After the advisor meeting, the
> current plan is in **Part 21**: write up the controller/methods paper first (draft skeleton done), then
> the gradient paper, with real MCB on the right clouds as a later third paper. The planner pilot
> (**Part 22**) fixed the planner's settings: choosing *where* to brighten beats brightening everywhere
> equally by about 30%, and the cheapest settings are enough.


We tried to teach a small AI to fight global warming inside a computer simulation of Earth's climate. The specific idea is **Marine Cloud Brightening (MCB)**: making low ocean clouds slightly whiter so they reflect more sunlight and cool the planet a little. We built the AI as a "controller" that watches the simulated climate and decides where and how much to brighten clouds, aiming to cool the ocean by a small target amount without wrecking rainfall in sensitive places like the Amazon. Along the way we discovered — twice — that our own results couldn't be trusted: an earlier version of the work was riddled with bugs and wishful statistics, and even the careful rebuild turned out to be measuring a 5-thousandths-of-a-degree signal with an instrument that jittered by 15, in an experiment accidentally designed so the AI could never win. So we tore it down twice, fixed the measurement, and finally gave the controller a problem worth solving: uncertainty about how strongly the cloud-seeding actually works. **The honest ending, in three acts:** the corrected method *does* find a good, on-target cloud-brightening plan (now confirmed with real statistical power); under realistic uncertainty a feedback controller *demonstrably* beats the fixed plan — our strongest statistical result, and finally the project's founding claim; but the winning neural controller got every bit of its skill by *imitating a hundred-year-old control-engineering formula* — training it through the climate simulation itself, the project's founding bet, never worked, and we measured exactly why. That's a real, defensible, and genuinely interesting scientific ending — just not the one we set out to find. **Then we did it twice more.** We gave the controller a genuine climate disturbance to fight — an artificial El Niño — and got the best-looking numbers of the whole project, only for our own audit to show the test had been rigged by our design once again: because we imposed only warm events, a blind controller that simply sprayed harder by a fixed amount scored just as well. We rebuilt it, and this time the controllers genuinely cancel **85–89%** of the El Niño's effect — replicated on two independent sets of fresh climates, and apparently the first time anyone has closed a control loop on cloud brightening against real year-to-year climate variability. Then we asked the sharper question: does the controller actually need to *watch* the El Niño? It does not. One that never looks at the Pacific does just as well, because the ocean temperature it is already measuring carries the signal by itself — a more useful answer than the win we set out to claim, and one we only reached by noticing that the obvious version of that experiment would have been rigged in our favour too.

---

# Part 1 — The Big Idea: what are we even trying to do?

## The climate problem, and one proposed response

Greenhouse gases trap heat and warm the planet. The permanent fix is to stop emitting them, but that's slow. So scientists also study **geoengineering** — deliberate, large-scale interventions to cool the climate — as a possible emergency backstop. One branch, called **Solar Radiation Management (SRM)**, cools the planet not by removing greenhouse gases but by reflecting a small fraction of incoming sunlight back to space before it can warm anything.

**Marine Cloud Brightening (MCB)** is one SRM idea, first proposed by physicist John Latham in 1990. The plan: use ships to spray a fine mist of sea salt into the low, flat clouds that hang over cool ocean regions. Those clouds — called **marine stratocumulus** — already blanket places like the ocean off Peru, Namibia, and California, and they already reflect a lot of sunlight. Make them a little brighter, and you reflect a little more.

## Why spraying salt brightens a cloud (the Twomey effect)

This is the one piece of physics worth really understanding, because a lot of the story hinges on it.

A cloud is made of tiny water droplets. Water needs something to condense onto — a speck of dust or salt, called a **cloud condensation nucleus**. If you add *lots* of extra specks (like sprayed sea salt), the same amount of cloud water gets divided among *many more, but smaller* droplets. And here's the key: **many small droplets reflect sunlight better than a few large ones.** So the cloud becomes brighter — more reflective — without holding any more water. This is the **Twomey effect**, and it's the engine that makes MCB work. Remember this: *more clouds → stronger brightening effect.* We'll come back to it, because getting this backwards caused one of the biggest bugs in the project.

**Albedo** is the word for "how reflective something is," on a scale from 0 (black) to 1 (white). Brightening a cloud raises its albedo.

## The catch: you can't cool one place in isolation

Earth's atmosphere and oceans are one connected system. Cool one patch of ocean, and the effects ripple thousands of miles away — shifting winds, storm tracks, and especially rainfall. These long-distance ripple effects are called **teleconnections**. A realistic worry is that cooling the wrong patch could dry out the Amazon rainforest or Africa's Sahel. So the real goal is not just *"cool the planet"* but *"cool it by the right amount, in the right places, without causing harmful side effects somewhere else."*

## Why do this in a computer — and why bring in AI?

You obviously can't experiment on the real atmosphere to find the best spraying plan. So you do it inside a **climate simulation**. But even in a simulation, "where and how much should I spray, moment to moment, to hit a target without side effects?" is a hard search problem. This is called an **inverse problem**: instead of running the model forward and seeing what happens, you work *backwards* from the outcome you want to the inputs that produce it.

The project's bet was that a small **neural network** — an AI **controller** (also called a **policy**) — could learn to solve this: look at the current climate state and output a map of how much to brighten clouds where, updating that decision as the climate evolves. Think of it as an automatic thermostat for the planet. Whether that adaptive "look at the state and react" ability actually helps — versus just following a fixed pre-planned schedule — became *the* central scientific question of the whole project.

---

# Part 2 — The Tools: how the simulation works

## What is a climate model?

A **General Circulation Model (GCM)** is a physics program that simulates the atmosphere. You give it a starting snapshot of the air — winds, temperature, humidity, pressure everywhere on a grid around the globe — and it steps forward in time, computing the next moment, then the next, for days or months of simulated weather. Our model is called **JAX-GCM** (the software package is `jcm`).

It has two halves:

- A **dynamical core** (borrowed from a package called **Dinosaur**) that handles the big-picture fluid motion of the air — the winds and pressure systems. "Spectral" just refers to a clever mathematical trick for representing the globe efficiently; you don't need the details.
- A **physics** module (a re-implementation of a well-known simple model called **SPEEDY**) that handles everything too small or detailed to track directly: sunlight heating, clouds, rain, evaporation, and turbulence near the surface. These are called **parameterizations** — simplified formulas that stand in for complicated small-scale processes.

## The superpower: the model is "differentiable"

Here's what makes this project unusual. JAX-GCM is written in **JAX**, a system that provides **automatic differentiation (autodiff)**. In plain terms: the computer can automatically calculate **how much any output would change if you nudged any input** — a quantity called a **gradient**.

Why does that matter? Because gradients tell you *which direction to adjust things to improve them.* If you know that "brightening this region a bit more cools the ocean a bit more," you can systematically search for the best plan instead of blindly guessing. Traditional climate models are old Fortran programs that are effectively black boxes — you can run them, but you can't get gradients through them. A fully differentiable climate model is genuinely novel, and it's the core capability that makes AI-based MCB optimization possible at all. **This capability was tested and confirmed to work, and it survived everything that follows.**

## The "coupled" Earth: atmosphere + ocean + land

The atmosphere alone isn't enough — MCB is about cooling the *ocean*. So the atmosphere is connected to a simple ocean and land, stepped forward together by a small piece of software called the **coupler** (`jax-esm`, nicknamed "jem").

The ocean and land here are deliberately crude: a **slab ocean** and a **slab land**. A "slab" is a single well-mixed layer — imagine the ocean as one bucket of water with a temperature that rises or falls depending on how much heat flows in or out. No currents, no depth, no ocean weather. This is cheap to run and easy to differentiate, at the cost of realism. (This simplicity, we'll see, later turned out to matter a lot.) Each coupling step is one simulated day: step the atmosphere, hand the heat flowing into the sea to the slab ocean and the heat flowing into the ground to the slab land, then feed the updated ocean and land temperatures back to the atmosphere as its new surface conditions.

## The AI controller and how it learns

The **policy** is a small neural network (a **multilayer perceptron**, or **MLP** — the simplest standard kind). Every so often (every 15 or 30 simulated days) it looks at features of the current climate and outputs a map of MCB brightening to apply.

Training it uses **Backpropagation Through Time (BPTT)**. In plain English: run the whole 60-day simulation forward with the controller acting at each step; measure how far the final result is from the goal (this gap is the **loss** — lower is better); then use the differentiable model to trace that error *backward* through all 60 days and figure out exactly how to adjust the network's internal settings ("weights") to do better next time. Repeat thousands of times. The forward run is wrapped in an efficient loop (`lax.scan`), and JAX computes the whole backward pass automatically.

> **Mini-glossary so far:** *MCB* = brighten ocean clouds to reflect sunlight. *Twomey effect* = more particles → more, smaller droplets → brighter cloud. *Teleconnection* = a far-away side effect. *GCM* = atmosphere physics simulator. *Differentiable / gradient* = the model can tell you which way to nudge inputs to change outputs. *Slab ocean/land* = a crude single-layer ocean/land. *Policy / controller* = the AI that decides where to brighten. *Loss* = how wrong the result is. *BPTT* = the training method that runs forward then traces error backward.

---

# Part 3 — The First Attempt: the staged plan (Stages 0–5)

Because the goal was ambitious, the work was broken into a "ladder" of stages, each meant to prove one thing before climbing to the next. Between stages were **gates** — pass/fail success criteria. Here's the story *as it was told at the time* (Part 4 will explain why most of it didn't hold up).

| Stage | What it tried | What it claimed |
|---|---|---|
| **0 — Plumbing** | Confirm that turning the MCB knob actually cools the model, and that the automatic gradient matches a brute-force check | **Passed, and genuinely valid.** Turning on MCB cooled the ocean; the autodiff gradient (−0.752) matched a manual finite-difference estimate (−0.740) to 1.6%. The differentiable machinery is real. |
| **1 — Static pattern** | Skip the AI; directly optimize a fixed brightening *map* to hit the cooling target | Claimed it hit ≈ −0.097 K (target −0.1 K) and found a "spatially structured" pattern |
| **2 — Fix the bookkeeping** | Compare each run against a paired "no-MCB" baseline run so natural drift cancels out | Verified the accounting was clean |
| **3 — Train the AI** | Train the MLP policy, starting from Stage 1's pattern | Claimed the AI slightly beat the static pattern |
| **4 — Make it adaptive** | Train across *many different starting climates* so the controller must actually react to state, not memorize one run | Cooling gate **failed** — the controller overshot the target on unseen climates |
| **5 — Realistic Earth** | Add real continents and mountains, and re-add penalties for disturbing Amazon/Sahel rainfall | **2 of 4 gates failed** — the AI generalized *worse* than the simple static pattern |
| **Option A** | A follow-up trying to fix Stage 4/5's failures by rearranging which climates were used for training vs. testing | **Refuted** — the fix didn't help; the AI still overcooled unseen cases |

The pattern by the end was discouraging: the fancy adaptive controller kept doing *worse* on climates it hadn't been trained on than a dumb fixed pattern did. At the time this was diagnosed as an AI "generalization" problem. The real explanation was deeper — and that's what the audit uncovered.

---

# Part 4 — The Reckoning: the independent audit

On 2026-07-12 an independent audit re-examined everything — and crucially, **re-ran the actual code** instead of trusting the write-ups. Its verdict: essentially none of the reported numbers from Stage 1 onward could be trusted. It found **seven separate root causes**, labeled R1–R7. Here they are in plain language.

**R1 — The "global average" was broken.** To compute an average temperature over the globe, you must weight each grid cell by its actual area (cells near the poles are tiny; cells near the equator are large). A units bug made every cell count equally, over-weighting the poles ~20×. This quietly corrupted every "global mean" number — and it's why Stage 1's "optimal" brightening pattern pointed at the Arctic (+68.7°N) instead of the sunny subtropics where MCB actually makes sense.

**R2 — The AI never actually learned.** When you plot the training loss over time, it should trend downward as the model improves. Here the loss curves were statistically flat — the wiggles were just random noise, not learning (the downward "trend" was no more real than a coin-flip streak). On top of that, the code that saved the "best" model had an off-by-one bug and was picking the luckiest random moment, and the whole thing ran in low-precision arithmetic. In short: the "trained" models weren't trained.

**R3 — A key conclusion was just luck.** A headline finding ("Option A refuted") was reproduced 72% of the way by a control that *didn't change anything*, and rested on a sample of just two test cases. It was noise dressed up as a result.

**R4 — A scorecard was measuring a constant.** One of the gates (Gate 3) was, on inspection, essentially reporting a fixed mathematical constant (about 0.0069) no matter what the AI did. It couldn't pass or fail meaningfully.

**R5 — The starting conditions were quietly drifting.** The "different climates" used for training weren't independent, settled climate states. They were snapshots of a single simulation that hadn't reached equilibrium and was drifting by ~1.4 K over 60 days — about 14× larger than the tiny 0.1 K cooling signal being hunted. You can't measure a whisper next to a jackhammer.

**R6 — It wasn't even simulating MCB.** This is the big one. The code brightened the **ocean surface**, not the **clouds**. It never touched a single cloud property, so it didn't implement the Twomey effect at all. Worse: the coded surface effect got *weaker* where there were more clouds, while real MCB gets *stronger* with more clouds (remember Part 1). So the model was doing the **opposite** of MCB — suppressing the effect exactly in the cloudy stratocumulus decks that are MCB's whole target. And the forcing was cranked ~5–7× too strong.

**R7 — The Earth system couldn't produce the side effects being studied.** The slab ocean has no currents and the land temperature was frozen to a fixed climatology, so the model *structurally cannot* form the ocean- and land-driven teleconnections (shifted monsoons, El Niño–like patterns) that the "protect the Amazon/Sahel" penalties were supposedly guarding against. The penalties were policing an effect the model was incapable of producing. There was also a wiring bug in the coupler that fed the ocean the land and sea heat mixed together, corrupting roughly 35% of the final-stage temperature signal.

## What survived

The audit was careful to say what *did* hold up — and it's the important part:

- **The differentiable machinery is real** (the Stage-0 gradient check). This is the project's genuine core achievement.
- The basic cooling mechanism works (an albedo change does cool the coupled model sensibly).
- Several pieces of infrastructure (ensemble gradient averaging, gradient clipping, the initial-condition tooling) were correct and reusable.

In other words: the *engine* was sound; everything built on top of it was standing on sand.

---

# Part 5 — The Rebuild: fixing the foundations

Rather than patch over the problems, the project rebuilt the experiment in three phases (called P0, P1, P2). A few new *concepts* were introduced here that are worth understanding, because they're what make the final results trustworthy.

**Fixing the physics (the R6/R7 repairs).** MCB was rewired to brighten **cloud albedo** (scaled by how much cloud is present) — real Twomey physics, in the right direction, at a realistic strength. The slab **land model was actually turned on**, and the coupler's heat-flux wiring bug was fixed so the ocean gets only the sea heat and the land gets only the land heat. The ocean was started from real observed sea-surface temperatures and **equilibrated** — run for a simulated 10 years until it stopped drifting (drift fell from 1.4 K to under 0.02 K per 60 days), so the tiny cooling signal is no longer drowned out.

**Fixing the measurement (R1–R5 repairs), plus three key ideas:**

- **The noise floor.** Before you can claim a result is real, you must know how big the model's *random* run-to-run wiggle is. So we ran the *same fixed* plan many times and measured the spread. This "noise floor" turned out to be about 0.014 K. Any claimed effect smaller than roughly twice that is indistinguishable from luck. Measuring this was never done before — it's what turns "it looks like it worked" into "it's bigger than noise." (Keep an eye on this one: Part 7 reveals the measurement had a serious blind spot.)

- **Held-out selection.** When you pick your "best" model, you must judge it on **held-out** data — climates it was *not* trained on — not on the training data itself. Otherwise you're just picking whatever memorized the training set best. (The old code picked the best on training data, which is R2.)

- **Pre-registration and control-relative gates.** We wrote down the exact pass/fail rules **before** seeing any results (in a frozen file, `PREREGISTRATION.md`), so we couldn't unconsciously move the goalposts. And every test is **paired and control-relative**: we compare the AI head-to-head against a simple baseline on the *same* climates, and require the difference to beat a **2× standard-error** bar (a standard statistical significance threshold) to count. If the difference is within that bar, the verdict is **"underpowered"** — meaning "too small to call either way," which is honest rather than a false PASS or FAIL.

> **Two more terms:** A **standard error** measures how uncertain an average is; requiring an effect to exceed *2 standard errors* is a common bar for "probably not just chance." An **ablation** is an experiment where you deliberately remove one ingredient (e.g., take away the AI's ability to see the climate state) to test how much that ingredient actually mattered.

---

# Part 6 — The First Three Campaigns (v1–v3)

With the foundations rebuilt, we ran three big experiments on a GPU (a fast computer chip). Each is a comparison between the **AI feedback controller** and a **static pattern** (the same brightening map applied every time, with no adaptation). The question throughout: *does the adaptive AI actually beat the simple fixed plan?*

## Campaign v1 — an apparent win (that wasn't)

At a strong brightening setting, the AI controller *appeared* to beat the static pattern. But it was overcooling badly (about −0.14 K, well past the −0.1 K target). It turned out the "win" was an illusion: when everything is overcooling, the adaptive controller just had easy room to look better by pulling back from the overshoot. Not a real advantage.

## Campaign v2 — a clean-looking negative

At a realistic brightening setting, the picture flipped. Now the AI controller was **significantly *worse*** than the static pattern (by −0.0093 K, beyond the noise), it still overcooled slightly, and its "adaptivity" was statistically **indistinguishable from a fixed schedule**. This looked like a decisive negative result: *the fancy AI adds nothing, and even hurts.*

## Campaign v3 — the correction

Before accepting that negative, we stress-tested it — and found **v2's negative was itself a bug**, not a fact about feedback. Two problems:

1. **The AI was being trained on the wrong goal.** The quantity it was minimizing was **less than 0.2% aligned** with what the pass/fail gate actually measured, and it was quietly *rewarding overcooling*. Evidence: the v2 controller moved the ocean temperature hard (from −0.084 to −0.119 K) while its training loss barely budged — proof it was optimizing something almost unrelated to the real target.
2. **It was still picking the "best" model using noisy training data**, not held-out data.

We fixed both — trained the AI **directly on the gate's own metric** and selected on **held-out** data — and re-ran everything. We also built a diagnostic (`run_gradient_probe.py`) that confirmed a genuine, generalizing improvement direction *exists* (so this wasn't a hopeless flat wall), and that the problem really was the wrong objective, not a deeper math issue.

**The v3 results (the trustworthy ones):**

| Test | v2 (buggy) | v3 (fixed) | Meaning |
|---|---|---|---|
| **On-target cooling** | FAIL — overcooled to −0.121 K | **PASS — on target, −0.097 K** | The corrected training hits the goal |
| **AI vs. static pattern** | FAIL — AI *worse* | **Tie (underpowered), −0.0048 ± 0.0063 K** | AI is no longer worse — but no better either |
| **AI feedback vs. fixed schedule** | underpowered | **underpowered, −0.0054 ± 0.0039 K** | Adaptivity is indistinguishable from a fixed schedule |
| **Trained-from-a-head-start vs. from scratch** | FAIL | **underpowered, +0.0033 ± 0.0041 K** | The head start doesn't measurably help |

At the raw-loss level the fixed training *appeared* to genuinely improve things — roughly halving the error on unseen climates (from 0.020 K to 0.011 K), real but below the significance bar at our sample size of 10 test cases. That, at least, is what we wrote at the time. Part 7 tells what happened when this claim, too, went under the microscope.

---

# Part 7 — The Audit of the Audit: the second reckoning

The story above — "the AI adds nothing provable, and that's our defensible negative result" — is where the project stood in late July. Before writing it up for the world, we did one last thing: a **second full audit** (2026-07-29), this time of the *rebuilt* project, done the same way as the first — trust nothing, recompute every number from the raw GPU outputs, re-run the actual code, and actively try to break every conclusion. The full findings live in `MCB_META_AUDIT.md`.

The good news first: the rebuild's bookkeeping is impeccable. Every recorded number reproduces from the raw data to twelve decimal places, the v3 "tie" verdicts survive every statistical test thrown at them, and v3's training really did learn — the first campaign ever to show genuine descent. But three things the project *believed* turned out to be wrong, and the third one changed the course of everything that followed.

## 7.1 The consolation prize was a mirage

Part 6 ended with a life-raft: "the fixed training genuinely halved the error on unseen climates, real but too small to prove." **Retracted.** That flattering 0.011 K was the *minimum of 39 noisy measurements* — and the same number had been used to pick the "best" model in the first place. Picking your luckiest dice roll out of 39 and calling it skill is exactly the mistake (R2) that sank the original project, reincarnated on new data; a simulation confirmed that pure noise routinely produces a "best" that good. And when the selected model was independently re-evaluated — twice — it scored *worse* than the static pattern both times. The honest restatement: **training never produced any detectable improvement over the static pattern at all.**

## 7.2 The noise floor had a blind spot: meet chaos noise

Part 5's noise floor (~0.014 K) measured how much results wiggle across *different starting climates*. A second kind of noise — re-running the very same experiment — had been measured as exactly zero. That zero was an illusion of how the harness worked: it repeated the experiment **inside one program run**, where a computer is perfectly deterministic, so bit-for-bit identical answers were guaranteed by construction. Run the same experiment in a *fresh* program run, and the compiler may order the arithmetic microscopically differently — differences around the fifteenth decimal place. In most software that's irrelevant. In a climate model it is not, because the atmosphere is **chaotic**: any tiny difference doubles and redoubles (the famous "butterfly effect") until, after 60 simulated days, it has grown to the size of real weather variability.

Measured directly: the same controller on the same starting climates, run twice, differs by up to 0.049 K per climate — typically **0.014–0.017 K per run**. From here on it helps to talk in **millikelvins (mK)** — thousandths of a degree. The cooling target is 100 mK. The chaos noise is ~14–17 mK *per individual run*. The effects the project had been hunting were ~5 mK. We had been trying to read a 5 mK signal with a 15 mK-jittery instrument, reassured by a noise meter that was blind to the jitter.

## 7.3 The deepest finding: the "tie" was baked into the design

Why did the adaptive AI never beat the fixed plan? The audit's answer: **because the experiment gave it literally nothing to adapt to.** All 20 test climates were grown from one parent ocean state, at one time of year, in a model with no random weather — they differ only in tiny atmospheric wiggles. At the moment of the controller's first decision, **11 of its 13 input features were exactly zero by construction**, and the remaining two differed across climates by less than a ten-thousandth of a degree. The controller was a thermostat installed in a house where the temperature never changes.

A Monte Carlo calculation (a simulation of the experiment itself) made it quantitative: even a mathematically *perfect* feedback controller could beat the static pattern by at most **~6 mK** in this setup — below the ~8–13 mK smallest effect the experiment could detect. The "underpowered" verdicts were preordained before a single run. And the earlier conclusion that resolving the question "needs more test climates and seeds, not more code" was exactly backwards: no amount of data can detect headroom that the design removed.

## 7.4 And the rulebook hadn't actually been followed

The pre-registration — the frozen rulebook meant to keep us honest — had been quietly violated in several ways: every experiment used **one** training seed where the rules required at least three; results were scored on a day-60 snapshot where the rules froze a final-10-day average (never implemented anywhere); the significance check in the code was a home-made "2 standard errors" shortcut instead of the registered t-test — and under the proper test, v1's celebrated "AI beats static" moment *was never significant in the first place*; one rainfall comparison was computed, failed, and went unreported; and the same 10 held-out climates had been peeked at roughly 141 times across selection, probing, and gating — meaning any *future* "significant" result on them would be uninterpretable, like grading students on exam questions they had already seen.

The audit's blunt summary: **at that moment the project had no statistically significant positive result at all.** But it also produced something better than a verdict — a costed, ranked plan for getting one. Step one: fix the measuring instrument (**Tier 1**). Step two: give feedback a problem it can actually solve (**Tier 2**). The rest of this report is what happened when we executed that plan.

> **New terms:** *Chaos / the butterfly effect* = in a chaotic system, microscopic differences grow exponentially until they're as big as the weather itself. *Millikelvin (mK)* = a thousandth of a degree; the cooling target is 100 mK. *Selection artifact (winner's curse)* = when you pick the best-looking of many noisy options, its score is inflated by luck, so it disappoints on re-measurement.

---

# Part 8 — Tier 1: rebuilding the measuring instrument

Tier 1 (built in about a day, run 2026-07-30/31) attacked the measurement problem directly, with five fixes:

- **Micro-ensembles.** Instead of measuring each (controller, climate) pair with a single rollout, run **eight near-clones** — each nudged by an imperceptible 0.001 K so chaos sends them down different weather paths, each with its own paired no-MCB baseline — and average them. Averaging 8 noisy readings shrinks random error by √8 ≈ 2.8×. (Weather forecasters have used this "ensemble" trick for decades; we borrowed it for measurement.)
- **Fresh test climates.** A brand-new set of 20 starting climates that no one — human or algorithm — had ever looked at, retiring the exhausted old ten. Every later experiment got its own fresh set too.
- **The registered metric, actually implemented.** Results are now scored on the average over the final 10 days, as the rulebook always required — not a single-day snapshot.
- **Real statistics.** The registered paired t-test and the Wilcoxon test (a backup that uses only rankings, so one weird case can't swing it) — plus a new tool, **equivalence testing (TOST)**. Ordinary tests can only *fail to find* a difference, which is not the same as showing there is none; an equivalence test can positively demonstrate "whatever difference exists is smaller than X." It converts an "underpowered" shrug into a hard bound.
- **Bug fixes and an amendment log.** The audit's newly found code bugs were fixed (the best one: a degrees-versus-radians mix-up that made the "tropical band" feature cover the entire globe), and every deviation from the rulebook is now formally logged as a numbered amendment in `PREREGISTRATION.md`.

The campaign took under three GPU-hours and worked exactly as predicted: measurement uncertainty on each arm dropped **5×** (from ±7 mK to ±1.4 mK), the micro-ensembles directly measured the chaos noise at 13–14 mK per run, and the corrected cross-process harness measured 16 mK where the old one had reported zero.

**Tier-1 results (20 fresh climates, registered 10-day metric):**

| Arm | Cooling achieved | Verdict |
|---|---|---|
| **Static pattern** | **−0.1023 ± 0.0014 K** | **PASS — on target, with real statistical power** |
| AI feedback controller (v3) | −0.0938 ± 0.0015 K | in band |
| Open-loop schedule | −0.0914 ± 0.0014 K | in band |

The AI-versus-static and AI-versus-open-loop comparisons came out as ties once more — but this time the ties have teeth: **any feedback effect is within ±4.5 mK at 95% confidence**, under 5% of the target signal, in an environment whose theoretical ceiling for a *perfect* controller was ~6 mK (Part 7.3). The feedback question for *this* task was now closed quantitatively rather than shrugged at.

Tier 1 took the project's tally from zero significant results to two:

1. **A confirmatory positive:** gradient-based optimization through the differentiable model produces a brightening pattern that hits the −0.1 K target on never-before-seen climates (−0.1023 ± 0.0014 K).
2. **A significant bounded negative:** in an environment with nothing to correct, feedback of any kind is worth less than 4.5 mK — and Part 7.3 explains *why*.

(One protocol footnote, duly logged: the trained arms here were the existing single-seed v3 networks, re-evaluated on the new instrument. The ≥3-seeds rule kicks in from Tier 2 onward, where new training-based claims are made.)

---

# Part 9 — Tier 2: giving the controller something to correct

A thermostat is pointless in a house whose temperature never changes. Tier 2 (run 2026-07-31 → 08-01, ~26 GPU-hours) finally built the house with drafts — by injecting a realistic uncertainty that a real MCB deployment would face and that a fixed plan *cannot* handle.

**The uncertainty: seeding efficacy.** How much brightening does a given spraying effort actually deliver? In the real world this is the biggest question mark of all (aerosol–cloud interaction is the largest stated uncertainty in climate projection). So each 60-day episode now draws a hidden **efficacy multiplier η** ("eta"), uniformly between 0.6 and 1.4. The controller *commands* a brightening; the world silently delivers η × command; η is never revealed. A fixed plan tuned for η = 1 must now miss the target roughly in proportion to how far η landed from 1 — while a feedback controller can watch the *realized* cooling and compensate. A pre-registered "manipulation check" confirmed the knob works: the static pattern's mean miss grows from 5.4 to 19.8 mK under η-uncertainty.

**A new competitor: the PI controller.** Before asking whether the *neural network* could exploit this, we added the honest yardstick — a **proportional–integral (PI) controller**, the hundred-year-old workhorse of control engineering, the law inside thermostats and cruise control. Ours is a "deadbeat" variant a few lines long: *from the cooling realized so far, estimate what η must be; divide the command by that estimate.* No training, no learning, no gradients.

**Tier-2 results — mean miss of the −0.1 K target (20 fresh climates, hidden η per episode):**

| Controller | Mean miss (mK) | vs static |
|---|---|---|
| Static pattern (η-blind) | 20.2 ± 2.5 | — |
| Open-loop schedule (trained, 3 seeds) | 19.7 ± 2.2 | tie |
| Neural feedback (trained, 3 seeds) | 18.7 ± 2.2 | tie (p = 0.42) |
| **PI controller (no training)** | **10.1 ± 1.6** | **halves the error — p = 0.0006** |

Two significant findings, both pre-registered:

1. **Feedback control demonstrably works here — the project's first significant feedback win.** The PI controller halves the error (better on 16 of 20 climates; the result survives every robustness check we could throw at it). Its mechanism was verified directly: its command tracks the hidden efficacy almost perfectly (correlation −0.94). A pre-registered asymmetry appeared exactly as predicted: when the world over-delivers (η > 1), backing off is easy and the controller is near-perfect (4.5 mK); when the world under-delivers (η < 1), the 0.09 brightening cap blocks pushing harder (15.7 mK) — an actuator limit, not physics.
2. **The trained neural network is significantly *worse* than the PI controller** (p = 0.007) and indistinguishable from static. The diagnosis, verified from the training records: a **training** failure, not an information problem — the PI controller gets its skill from two input features the network also receives. Gradients backpropagated through 60 days of chaotic weather simply never converge: all six training runs sat flat at the noise floor, and the resulting networks nudge in the *correct direction* but at 5–10× too little strength — vestigial feedback.

The uncomfortable moral of Tier 2: given a genuinely solvable feedback problem, the differentiable climate model's gradients — the project's founding technology — lost, decisively, to a few lines of 1920s control theory.

---

# Part 10 — Tier 2b: if the AI can't learn it, teach it

Tier 2's diagnosis suggested its own remedy. If the neural network fails only because *training through chaos* doesn't converge — not because it lacks the inputs or the capacity — then bypass the chaos: **teach the network the PI law directly.** This is called **distillation** (or imitation learning): generate a large batch of synthetic "situations," compute the PI law's answer for each, and train the network by ordinary supervised regression to copy it. No climate model in the loop, no chaos, no noise — it takes minutes. Then **fine-tune** the copy with BPTT through the climate model, starting from inside a known-good solution. Two pre-registered questions: **H5** — does the neural controller now beat the static pattern? And the more ambitious **H4** — can learning *exceed* the classical law? (There was real room to: the PI law can only scale one fixed pattern up and down, so when efficacy is low it slams into the brightening cap — a smarter policy could recruit *more ocean area* instead.)

**Attempt 1: a gate earns its keep.** The campaign's pre-registered quality gate killed the first attempt after 23 minutes: the distilled network behaved exactly like the static pattern. The measured cause is a lesson in itself — the synthetic practice situations had been sampled around zero, but two of the real climate's input features sit far from zero (3.7 and 8.5 standard deviations outside the practice range). Confronted with numbers unlike anything in its training data, the network froze its correction at "change nothing." The fix: measure the real model's actual operating point, generate the practice data around it, and add a numerical-conditioning trick (train on rescaled features, then fold the rescaling back into the network's first layer so the saved network is unchanged in form). Logged as a formal amendment; pinned by a regression test.

**Attempt 2 passed every gate** — the copy was near-perfect (mean copying error under 0.002 albedo), and on validation climates the imitation-only network already scored at PI level. The full campaign (~11.5 hours) then fine-tuned three seeds and evaluated everything on 20 brand-new climates with a fresh η stream.

**Tier-2b results — mean miss of the target (20 fresh climates):**

| Controller | Mean miss (mK) |
|---|---|
| Static pattern | 19.5 ± 2.7 |
| PI controller | 9.3 ± 1.6 |
| Imitation-only network | 8.2 ± 1.5 |
| **Fine-tuned network (3 seeds pooled)** | **7.9 ± 1.2** |

- **H5 — SIGNIFICANT: the headline of the entire project.** The fine-tuned neural controller beats the static pattern by **−11.6 mK (p = 0.00025** after multiple-comparison correction; the rank-based test agrees at p = 0.0002). It wins on 18 of 20 climates; the verdict survives dropping any single climate (worst case p = 0.0003), and resampling the data 10,000 ways puts the true effect between −16 and −7 mK. **The project's founding claim — a neural feedback controller demonstrably outperforming a static deployment — is finally supported**, with the strongest statistics of the entire effort.
- **H4 — null.** The fine-tuned network does *not* beat the PI controller (−1.5 mK, p = 0.20; provably within ±3.4 mK). Learning did not exceed the hand-designed law. The "recruit more area when efficacy is low" hope showed a suggestive trend (p = 0.053) — real enough to motivate a follow-up, not real enough to claim.
- **The kicker.** Fine-tuned ≈ imitation-only, provably within ±2.4 mK: the gradient fine-tuning — BPTT through the differentiable climate model, the founding technology — **added essentially nothing**, even when started inside a known-good solution. Every bit of the network's demonstrated skill was put there by copying the classical controller. (This was one of the pre-stated possible outcomes: "the chaos-gradient bottleneck persists even from a good basin.")
- And the PI controller beat static for the **third time, on a third independent climate set** — that result is now as replicated as anything in the project.

---

# Part 11 — The Honest Bottom Line

## What genuinely worked

- **The differentiable climate model is real and valuable.** Gradients flow correctly through a coupled atmosphere–ocean–land simulation. This survived two audits and every campaign.
- **Differentiable optimization designs a good static plan — now confirmed with power.** The optimized pattern hits the cooling target on never-touched climates: −0.1023 ± 0.0014 K against a −0.1 K goal (Tier 1).
- **A neural feedback controller can provably beat a static plan under realistic uncertainty.** Under hidden efficacy variation, the imitation-initialized network wins by −11.6 mK, p = 0.00025 — the founding claim, finally supported (Tier 2b).

## What didn't

- **Training the controller *through the simulation*.** BPTT through 60-day chaotic rollouts neither discovered the feedback law from scratch (Tier 2: trained networks ≈ static, significantly worse than PI) nor improved it when handed it on a plate (Tier 2b: fine-tuning ≈ imitation, within ±2.4 mK). The measured mechanism: ~12–17 mK of per-run chaos noise buries the gradient signal at any practical budget.
- **Both generations of wishful results.** The original Stages 1–5 dissolved under the first audit (seven root causes); the rebuild's own consolation claims — the "halved loss," the zero noise floor, the "generalizing improvement direction" — dissolved under the second. What survived is what was pre-registered, replicated, and adversarially re-derived.

## The seven significant results

1. **Confirmatory positive (Tier 1):** the gradient-optimized static pattern hits the target on fresh climates, −0.1023 ± 0.0014 K.
2. **Significant bounded negative (Tier 1):** with nothing to correct, feedback of any kind is worth less than 4.5 mK — and the design analysis explains why (a ~6 mK ceiling).
3. **Classical feedback halves the efficacy-uncertainty error (Tier 2, replicated 3×):** 20 → 10 mK, p = 0.0006 — while directly-trained neural controllers fail to realize the same gain (a measured training failure, not an information limit).
4. **An imitation-initialized neural controller significantly beats static deployment (Tier 2b):** −11.6 mK, p = 0.00025 — with fine-tuning contributing provably nothing beyond the imitation.
5. **Feedback rejects 85–89% of an imposed El Niño's effect on global temperature (Parts 13–14):** disturbance sensitivity falls from 55.3 to 5.8–8.3 mK per K of Niño3.4 (p < 1.2×10⁻⁴), replicated on two independent sets of fresh climates, while an open-loop plan rejects none. This replaced the first ENSO campaign's own pre-registered headline, which the audit showed a retuned blind controller could match.
6. **Observing the disturbance is worth nothing measurable (Parts 14–15):** a controller that never sees the Niño index matches one that does — bounded under 2.7 mK/K against a disturbance that levels off, and under 2.9 mK/K against one that never stops growing, in both cases under 5% of the disturbance. What matters is closing the loop on the realized outcome, not measuring the driver.
7. **A growing disturbance is materially harder to reject (Part 15):** the uncancelled share rises from ~15% to ~33% of the disturbance — the penalty classical control predicts for a trend — though this is a cross-campaign contrast rather than a paired test.

## The one-sentence headline

**A neural network controller *can* demonstrably steer a climate intervention under realistic uncertainty — but in this project every bit of that ability came from imitating hundred-year-old control theory, and none of it from gradient training through the climate model itself.** The differentiable model's proven value is *design* (finding the spatial pattern); its value for *training controllers* through long chaotic rollouts is, on this evidence, bounded near zero — and we can say exactly why.

## What's still open (honestly)

*(This list was written after Part 11 and has been updated as later parts closed items; see
Parts 12–15 for what happened to each.)*

- **Was the training failure just an under-resourced optimizer?** The strongest remaining objection
  to the central negative: the trainings used small budgets and single-rollout gradients, while the
  micro-ensemble trick that fixed *measurement* was never applied to the *gradients*. The decisive
  experiment — micro-ensembled-gradient training plus a gradient-free search at matched cost — is
  specified in `PAPER_PLAN.md` and is the most important run still to do.
- **Can learning ever exceed the classical law?** Five independent replications now say imitation
  *matches* the classical law and never beats it. The low-efficacy "recruit more area" trend
  (p = 0.053) remains the one unexplored opening.
- **The idealized observer.** All feedback controllers read a noiseless measurement of the realized
  cooling against a paired counterfactual baseline — something no real deployment could have.
  Adding observation noise is the external-validity test still to run, and it directly decides
  whether Part 14's "watching the disturbance is redundant" survives realistic sensing.
- **One ocean state, one start date.** Every "climate" is a small perturbation of a single
  equilibrated state started on the same calendar day, so all statistics generalize across weather,
  not climates. Season-staggered starting points are a five-line change and the cheapest way to
  widen the claim.
- **Scope.** Everything holds for a slab ocean with no currents, 60–180 day horizons, an in-model
  forcing at or beyond published MCB feasibility, and with the model's stratocumulus-analog cloud
  deck unperturbed — the actuator brightens the *convective* cloud population, so this is an
  idealized ocean-albedo intervention, not MCB proper (Part 15 and `PAPER_PLAN.md` take the
  consequence: the paper drops the MCB framing). These are statements about control and
  optimization in a differentiable climate model — not deployment guidance.

## The scientific lesson

Twice this project tore down its own results — and both teardowns made the final product stronger. The first audit found broken code; the second found something subtler: *correct* code measuring the wrong thing, a noise meter blind to the dominant noise, and an experiment whose null result was guaranteed by its own design. The way out was not more compute but better *measurement* (micro-ensembles, equivalence bounds, fresh test sets) and a better *question* (inject the very uncertainty that feedback exists to fight). The final story is smaller than "AI controls the climate," but it is coherent, mechanistic, replicated — and true.

---

# Part 12 — Closing the books on rainfall

One loose end remained after Part 11, and it was an integrity debt as much as a science question. Remember the original worry from Part 1: cooling the ocean in the wrong place might dry out the Amazon or the Sahel. Every campaign since Tier 1 had quietly *recorded* the Amazon and Sahel rainfall change of every single run — and nothing had ever analyzed those numbers. Worse, the second audit found that back in v3 a rainfall comparison had been computed, had come out as a FAIL, and had simply gone unreported. Before any writeup, both had to be dealt with.

So we did the cheapest experiment of the whole project: we analyzed data we already had (no new simulation time at all), with the same statistical machinery as everything else — paired tests, equivalence bounds, and a correction for the fact that we were making 40 comparisons at once (make enough comparisons and one will look "significant" by luck; the correction accounts for that).

**The verdict, in three parts:**

1. **No detectable rainfall harm, anywhere.** Across all three campaigns, both regions, and every controller — static pattern, PI, and all the neural variants — not a single rainfall change was distinguishable from zero, and no controller differed from the static pattern. The one nominally "significant" number in the pile (before correction) was a slight *wetting*, not a drying, and it dissolves under the correction exactly as a fluke should.
2. **With honest bounds on what "no harm" means.** Whatever true effect exists is smaller than about 0.6–1.2 mm/day for the Amazon and 0.06–0.21 mm/day for the Sahel (at 95% confidence). The Amazon bound is honest but not tight — roughly 10–20% of its rainy-season rainfall — because a single day of regional rain fluctuates chaotically by ±3 mm/day between otherwise-identical runs. That measured noise number is itself new: it says any *future* rainfall claim needs time-averaged metrics and ensembles, exactly as the frozen rulebook already required.
3. **The buried FAIL was a coin flip.** Re-analyzed properly, the v3 comparison that went unreported shows a difference of +0.0033 ± 0.0049 — statistically nothing, on a metric the first audit had already condemned (it carried the R4 constant-floor bug). On the cleanest subset of that data, the "failing" controller was actually numerically *better* on both regions. The sin was the silence, not any hidden harm.

The books are now clean: the rulebook has a formal amendment (number 5) logging the analysis and its limits, the withdrawn rainfall gate stays withdrawn *by its own pre-registered rule* rather than by neglect, and the last known unreported result in the project's history is reported. One design note carried forward: the region boxes used for "Amazon" and "Sahel" are plain rectangles that include a sliver of Atlantic ocean, and the episodes all run January–February (Amazon wet season, Sahel dry season) — both fine for a bounded "no detectable harm" statement, both to fix before any paper leans on regional rainfall.

---

# Part 13 — The ENSO experiment: a third teardown, and the result that survived it

## Giving the thermostat real weather to fight

Every previous chapter shared one weakness: the simulated world barely changed. Part 7 diagnosed
it precisely — the controller was "a thermostat installed in a house where the temperature never
changes." Tier 2 fixed that with a *hidden* uncertainty (how strongly the spraying works). This
chapter fixes it with something more real: **El Nino**.

El Nino is the biggest year-to-year wobble in Earth's climate — a periodic warming of the tropical
Pacific that shifts weather worldwide and raises global temperature for months. It is exactly the
kind of disturbance a real deployment would have to cope with, and no published study had ever
closed a control loop against it. So we built a **pacemaker**: a way to command an El Nino of a
chosen strength inside the simulation, by gently pulling the sea temperature in one Pacific box
toward a target and letting the rest of the model respond on its own.

Two calibration runs came first, because the project's rule is now measure-before-you-design. They
established that the model's global temperature responds to a commanded El Nino at **0.123 K per
K** — almost exactly the real-world figure (0.11) — and that the spraying had roughly three times
the strength needed to fight it. Then each 180-day episode drew a *hidden* El Nino strength, and
four kinds of controller had to hit the −0.1 K cooling target anyway: the fixed pattern (blind to
El Nino), a fixed *schedule* that compensates the average El Nino, the classical control formula,
and the neural network taught to imitate it.

## The result that looked spectacular

The campaign ran clean and the pre-registered tests came back overwhelming: the classical
controller and the neural network each beat the blind fixed pattern by about **58 thousandths of a
degree, with p-values around 1 in 30 million**. On paper, the strongest numbers the project had
ever produced.

Every number was correct. The *conclusion* was not — and finding that out took a five-lens
adversarial audit, run before a word of this was written.

## Why the headline collapsed

**The winning margin didn't require any knowledge of El Nino at all.** The audit built a
counterfactual: the same fixed pattern, spraying harder by one constant amount, knowing nothing
about El Nino, with that constant tuned honestly (never using the case it was scored on). It
scored 31.4 — statistically **tied** with the classical controller's 30.5 and the network's 29.1.

The reason is a design mistake of mine, and it is instructive. We only imposed El Ninos — warm
events. A disturbance that always pushes one way has a non-zero *average*, and anything with a
non-zero average can be cancelled by a fixed adjustment. The scoreboard we had registered measured
only "how close to the target did you land," which a bigger constant dose achieves just as well as
genuine responsiveness. **We had built the mirror image of the Part 7 failure**: there, a null
result was guaranteed by the design; here, a *win* was.

Worse, the reason we left out La Nina — the cold counterpart — doesn't hold up. We excluded it
because the cold response looked weak, but that measurement came from a 365-day window, while the
experiment ran on a 180-day one. At the length actually used, the responses are nearly mirror
images (+104 vs −99 thousandths). Including La Nina would have made the disturbance average out to
zero, and no fixed dose could have faked its way to a win.

**The one comparison that wasn't circular also failed.** The fixed *schedule* was supposed to be
the fair opponent — it compensates the average El Nino, so beating it would prove that reacting
*per episode* matters. The classical controller did beat it, by 22 thousandths. But the audit found
the schedule had been handed a wrong number: its assumed strength of the El Nino effect was
measured on *air* temperature while the experiment scored *ocean* temperature — a 36% error. Given
the right number, the schedule scores 27.5 instead of 52.7 and the advantage evaporates (3
thousandths, p = 0.58). That claim is withdrawn.

**And nobody hit the target** — not because of any physical limit, but because of a mismatch
between what the control formula aims at and what the scoreboard measures. The formula steers the
temperature on the *final day*; the scoreboard averages the *last sixty* days. A flawless
controller of that formula can only score −0.084 instead of −0.100. Predicted shortfall: 16.4
thousandths. Observed: 16.6. A few lines of code, not physics.

## What survived — and it is the real finding

Ask a different question, the one the scoreboard couldn't: **when the El Nino is stronger, how much
further off target does each controller drift?** That slope is the honest measure of fighting a
disturbance, and — crucially — **no amount of retuning a blind controller can change it.** A bigger
constant dose shifts your average; it cannot make you track something you cannot see.

| Controller | drift per unit of El Nino strength | disturbance rejected |
|---|---|---|
| Fixed pattern (blind) | 71.6 | — |
| Fixed schedule (average compensation) | 81.1 | **none** |
| **Classical feedback** | **11.1** | **84%** |
| **Neural network (imitation)** | **13.8** | **81%** |

The feedback controllers cancel **more than four-fifths** of El Nino's effect on global ocean
temperature (p < 0.0001 — the simpler p-value first computed here was later found overstated by
Part 14's audit, because the scatter grows with the disturbance; the corrected, noise-robust value
is what's quoted), and what's left of their episode-to-episode scatter is
indistinguishable from the model's own chaotic weather noise. The fixed schedule, by contrast,
rejects *nothing* — it lowers the average error while leaving the tracking untouched. That contrast
is the actual scientific content: **compensating the average is not the same as reacting, and only
one of them is control.**

The mechanism is confirmed rather than inferred: the commanded spraying tracks the hidden El Nino
strength at a correlation of 0.86–0.91 for every feedback controller and is flat, by construction,
for the blind ones — and with no El Nino present they spray exactly what the fixed pattern does, so
they are not merely dosing more. And for the fourth independent time, the network taught by
imitation matches the hundred-year-old formula to within about 3 thousandths of a degree, and does
not beat it.

## The lesson, again

This is the third time the project has dismantled its own headline, and the pattern is now
unmistakable. The first audit found broken code. The second found correct code measuring the wrong
thing. This one found a correct experiment asking a question whose answer was fixed in advance — and
it found it *because the result looked too good*, which is now the trigger for scrutiny rather than
celebration.

The fix is specified and cheap (about four GPU-hours): include La Nina so the disturbance averages
to zero, measure the controller's constants on the quantity actually being scored, point the
control formula at the window being measured, register the *slope* as the primary test, and add the
retuned-blind controller as an official opponent that any feedback claim must beat. That
specification is now frozen in the rulebook as Amendment 7, alongside a written post-mortem of all
four design errors.

What we can honestly claim today is narrower than the campaign first suggested and more solid than
anything before it: **a feedback controller — classical, or a neural network that learned by
imitating it — cancels about 84% of a hidden El Nino's effect on global temperature, where a fixed
schedule cancels none.** As far as the literature shows, that is the first time anyone has closed a
control loop on marine cloud brightening against real interannual climate variability.

---

# Part 14 — What the thermostat actually needs to see

Part 13 ended with a corrected experiment and one nagging gap. The controllers were rejecting most
of an artificial El Niño's warming — but the pre-registered test had compared them against a
controller that did **nothing at all**. Beating "no control" only shows that control works. It says
nothing about the more interesting question: **does it help to *watch* the El Niño?**

That question matters practically. A real deployment could either monitor the Pacific and act on
what it sees coming, or ignore the forecast and simply react to the temperature it is trying to
hold. The first is more expensive and more fragile. Which one you need is a design decision worth
knowing.

## Why the obvious experiment would have cheated

The natural test is to take the controller apart: keep the half that reacts to the error, delete
the half that watches the Pacific, and see how much worse it does. We nearly ran exactly that, and
it would have been rigged in our favour.

The reason is subtle and worth understanding, because it's the same shape of mistake as Parts 7 and
13. The controller has two knobs: how hard it reacts to error, and how much it trusts its El Niño
observation. Those knobs were set *together* — the reaction knob was turned down low precisely
because the watching knob was doing much of the work. Delete the watching half and leave the
reaction knob where it was, and you haven't built "the same controller without foresight." You've
built a controller with its remaining knob set wrong. Our own simulation said so: retuning that one
number takes the reactive controller from clearly worse to essentially equal.

So before the real experiment, we ran a cheap one on practice climates to find each controller's
best setting. That sweep did two useful things. It confirmed the concern — the untuned reactive
controller looked far worse than it really was — and it caught a bug of mine, where a "1" in a
configuration string was being read as a physical constant eighteen times too large. Both were
caught on practice data, before anything that counts.

The sweep then failed on its own terms: the reactive controller kept improving all the way to the
largest setting we tried, so we couldn't honestly name a best one. Rather than keep hunting, we
changed the rules in the safest possible direction — put a *range* of settings into the real
experiment and let the opponent be whichever one performed best **on the very data used to judge**.
That deliberately flatters the opponent. If our controller still won, nobody could say we had
handicapped its rival.

## The answer: watching the Pacific buys nothing

| Controller | drift per unit of El Niño | rejected |
|---|---|---|
| Fixed plan (no control) | 55.3 | — |
| **Reacts to error only — never sees the El Niño** | **8.3** | **85%** |
| Watches the El Niño *and* reacts | 7.9 | 86% |
| Neural network (taught by imitation) | 5.8 | 89% |

All three controllers cancel roughly six-sevenths of the disturbance. And the difference between
watching and not watching is **−0.4 ± 1.3**, statistically indistinguishable from zero — with the
effect bounded below 2.7, under 5% of the disturbance being fought. On the second, cancellation-proof
measure the watching controller is *slightly worse*. The null holds however we pair the arms, holds
when any single climate is dropped, and holds on an alternative metric.

The finding, plainly: **a controller that never looks at the Pacific does just as well as one that
does.** The realized ocean temperature — the thing it is already measuring in order to know whether
it's on target — turns out to carry the El Niño's signature all by itself. Watching the driver adds
nothing once you are properly reacting to the outcome.

That is genuinely useful, and more actionable than the win we set out to claim. It says a deployment
should invest in measuring **what it is controlling**, not in forecasting **what is disturbing it**.

Two honest limits. The rejection is large but *not* complete — every controller keeps a small,
statistically real residual, so 85–89% is the number, not "essentially all". And the earlier
experiment's headline of "95%" was too generous: it came from a signed measure in which
under-correcting warm events and over-correcting cold ones partly cancel. Corrected, the same
experiment shows 69–77%.

The good news alongside it: the Part 13 result **replicated** on twenty-four climates it had never
seen, at the same strength.

## Three teardowns later

This is the fourth time the project has dismantled a result of its own, and the pattern has become
almost mechanical: the more decisive a finding looks, the harder we now hunt for the design choice
that guaranteed it. In Part 7 a null was baked in. In Part 13 a win was. Here a win *would* have
been, and we caught it before spending the compute rather than after publishing the claim.

What is left standing is smaller than "AI controls the climate" and considerably more solid: a
feedback controller — classical, or a neural network that learned by copying one — cancels most of
a hidden El Niño's effect on global temperature; the neural version matches the hundred-year-old
formula and never beats it, now five times over; and the ability to *observe* the disturbance,
which sounded essential, is worth nothing you can measure.

---

# Part 15 — The experiment we couldn't run, and the one we ran instead

## Setting out to add greenhouse warming

The obvious next step was carbon dioxide. Every result so far fought a disturbance that eventually
levels off; real greenhouse warming doesn't. And the simulator turned out to have a CO2 knob built
in that nobody in this project had ever switched on.

Switching it on took an afternoon, and it worked — I could show the model warming. Then, planning
how to measure it, the whole thing collapsed for a reason that had nothing to do with CO2 physics.

**The measurement is built as a comparison.** Every result in this report comes from running the
climate twice — once with cloud brightening, once without — and subtracting. That subtraction is
what removes the model's own drift and leaves only what the intervention did. But the CO2 setting
lives in the *model*, not in the *intervention*, so it would have applied to **both** runs. Subtract
them and the greenhouse warming vanishes exactly: not reduced, not noisy — mathematically zero. And
because the controller also sees only differences, it wouldn't even have noticed the warming it was
supposed to fight.

We would have spent a day of computing to measure a quantity guaranteed in advance to be zero. It's
the same trap as Part 7, where the answer was fixed by the design before any simulation ran. This
time it was caught while planning, at a cost of nothing.

(There was a second, independent reason, which would have killed it anyway: the model's greenhouse
band is already nearly opaque, so turning the knob harder stops adding warming and eventually
*reverses* it. The trick I'd planned — crank it up to squeeze five years of warming into one — was
never available.)

## The question survived, and got sharper

Losing CO2 forced a useful re-examination, and turned up something embarrassing. I had been
describing the El Niño disturbance as an *oscillation* — something that swings up and comes back
down. It never did. Looking at the actual configuration, it ramps up over a month and then **holds
steady** for the remaining five. Everything we had concluded was already about a disturbance that
arrives and stays.

So the untested case was never "persistent versus temporary". It was **steady versus
still-growing** — and that distinction has real teeth in control theory. A controller that reacts to
error can settle out a disturbance that stops changing. Against one that keeps growing, it is
permanently chasing: by the time it corrects for where the world was, the world has moved further.

That we could test immediately, by making the imposed disturbance grow across the whole episode
instead of levelling off. No new physics, no new code — one setting.

## Two answers

**First: a growing disturbance really is harder.** The share the controllers fail to cancel rises
from about a seventh to about a third. That is the predicted penalty, and it is not subtle. (Fair
warning on this one: it compares two campaigns run on different climates rather than a head-to-head
test, so treat it as a solid indication rather than a precise measurement.)

**Second, and this is the one that matters: watching the disturbance still buys nothing.**

| Controller | drift per unit disturbance | cancelled |
|---|---|---|
| Fixed plan | 35.5 | — |
| **Reacts to error only — never watches** | **11.7** | **67%** |
| Watches the disturbance and reacts | 12.8 | 64% |
| Neural network | 12.3 | 65% |

The controller that watches is, if anything, slightly *worse* — the difference is +1.1 ± 1.1,
indistinguishable from zero and bounded under 2.9. The blind controller is also the only one that
actually lands on target (−0.1004 against a −0.1 goal).

This was the case where anticipation had the strongest theoretical claim to being necessary, and it
still isn't. The finding from Part 14 doesn't just repeat — it survives the harder test.

One guard worth mentioning, because it nearly bit. Under a growing disturbance the watching
controller needs its sensitivity constant scaled up by about 20%, and feeding the raw measured value
would have weakened it by 16% — producing a null that merely reflected our own miscalibration and
conveniently confirming what we'd already published. So the campaign ran the watching controller at
two different strengths. The stronger one did better, exactly as expected, and still lost.

## Where that leaves it

Across a disturbance that arrives and stays, and one that never stops growing, the same answer holds:
**closing the loop on the temperature you're trying to control is what matters; measuring the thing
pushing it around adds nothing you can detect.** For anyone designing such a system, that's a
statement about where to spend the instrumentation budget.

What a growing disturbance does cost is accuracy — a third of it goes uncancelled instead of a
seventh. Since anticipation isn't the fix, the honest next question is whether explicit *integral*
action is. The expectation should be modest: the control law already has integral-like structure,
and the simulated ocean is itself close to an integrator.

---

# Part 16 — A fresh-eyes review, and the two papers closest to ours

On 2026-09-29 the whole project got one more end-to-end review. It re-read this report, the paper
plan (`PAPER_PLAN.md`), the retrospective and both audits; re-checked the code and the committed
campaign data; ran the test suite (all 216 fast tests pass); and searched the literature for anything
published since August. The two audits had hunted for errors inside individual campaigns. This review
asked a wider question: *if a journal reviewer read everything tomorrow, what would they say?*

The answer has three parts:

1. **The feedback result stands, but only with a perfect sensor.** Every controller reads the exact
   difference between its own run and a no-brightening twin run with identical weather, a
   measurement no real deployment could have.
2. **"Gradients design, they do not train" has not been shown yet, on either half** (16.2 and 16.3).
3. **Two recent papers do something very similar**, and one of them uses the same climate model (16.1).

## 16.1 The two papers closest to ours

**Dubey, Abbot & Chattopadhyay (2026), "Optimizing Geoengineering Interventions Using Differentiable
Climate Models"** (arXiv:2609.12528, posted 11 September 2026; UC Santa Cruz and the University of
Chicago) uses JAX-GCM, the same differentiable model this project is built on, to design a climate
intervention with gradients. They warm the whole ocean by 4 K, then ask how much to cool five broad
latitude bands of ocean (centred on 45°N, 20°N, the equator, 20°S and 45°S) so that temperatures over
land return to the unwarmed climate. They sidestep chaos with a trick from weather forecasting and
turbulence control called **receding-horizon control**: look only 8 to 14 days ahead, compute the
gradient for that short stretch, apply the best cooling, run the model forward, and re-plan from
wherever the climate actually ended up. No gradient ever spans more than two weeks. Stitched
together, this removes 92% of the land warming over two years, holds it in a three-year run, and
still works when the same cooling schedule is replayed in two AI climate emulators (LUCIE and
NeuralGCM). Their supplement measures exactly the effect we ran into: their gradients agree with
brute-force checks at 7 and 14 days (correlations 0.91 and 0.87) but only loosely at 28 days (0.49).
Our controllers were trained on 60-day runs, twice that horizon, so our training failure is what
their numbers predict. The differences matter, though. They set the ocean temperature directly, so
there is no ocean that responds and no sunlight-reflecting actuator; their "controller" is an
optimizer re-run at every step rather than a trained policy; and they test no hidden uncertainty, no
varying disturbances and no classical feedback controller. Their own discussion lists ocean adjustment
and radiative forcing as missing, and reports that weather noise (0.5 K² per segment) was too large to
tell whether their time-varying schedule beats a constant one. That is precisely the kind of question
our paired, micro-ensemble measurement was built to settle.

**Quan, Koll, Lutsko & Yuval (2025), "Solar Geoengineering Strategies Based on Reinforcement
Learning"** (*Journal of Geophysical Research: Atmospheres* 130, e2025JD044319) is the closest
precedent for the *AI controller* half of this project. They let a **reinforcement learning**
algorithm (an AI that learns by trial and error from a score, without any gradients through the
climate model) decide how to distribute stratospheric aerosol in an idealized climate model. That is
the other main sunlight-reflection idea: reflective particles high in the stratosphere instead of
brighter clouds. They present it explicitly as an alternative to the linear feedback controllers of
earlier work, one that includes feedback naturally. Within "several dozen" climate simulations it
learned stable, plausible strategies, and it discovered on its own that the best strategy depends on
when the intervention starts, which they explain with a simple energy-balance model. (The preprint
version adds that it found the "kicking the can down the road" effect: warming can be reversed faster
by varying the aerosol amount over time.) Their AI learned from scores alone (a **zeroth-order**
method), whereas ours learned from gradients passed back through 60 days of chaotic weather (a
**first-order** method), so their success next to our failure is an interesting contrast. It is not yet
a fair one. Only their abstract was readable (the full text sits behind a website that blocks
automated access), so we do not know how many knobs their AI turned or how long its scoring windows
were, and both matter: scores averaged over years suppress chaos noise, and a controller with a
handful of knobs is far easier to learn than ours, which output a whole map (about 1.25 million
adjustable numbers) and got 40 training steps.

A third paper is already in the paper plan and worth keeping in view: **Lee et al. (2025,
*Geophysical Research Letters*)** ran the first classical feedback controller for marine cloud
brightening, in the full CESM2 climate model. Each year it decides how much ocean area to brighten,
the same "recruit more area" lever as Part 10's H4, and the authors report that it converged more
slowly than intended.

## 16.2 The biggest surprise: the "designed" pattern was never really designed

Every campaign since Tier 1 uses one fixed brightening map: the "gradient-optimized pattern" from
Stage 1 (`mcb_experiments_gpu/stage1_v2/stage1_optimized_pattern.pkl`). It is the static arm in every
comparison and the map the classical and imitation controllers scale up and down. Its saved
optimization history shows it was not produced by gradient descent in any meaningful sense:

- **The kept map is the one after step 1 of 150.** It was kept because it had the lowest loss of the
  150. That is the "luckiest of many noisy tries" selection Part 7.1 retracted for the controllers.
- **Almost none of that loss was about cooling.** In a typical step, 99.5% of the loss was a
  "uniformity" penalty (how patchily the ocean cooled) measured on one 60-day run from one starting
  climate. Over 60 days chaotic weather makes the ocean patchy all by itself, so this term mostly
  measured weather; the error against the cooling target was typically about a hundredth as large.
  Step 1 won mainly because its weather happened to be less patchy: its loss was 2.2 standard
  deviations below the average of the later steps.
- **The map is a single step away from where it started.** Adam, the optimizer, moves every setting
  by roughly one learning rate on its first step, and 85% of ocean cells sit at exactly the starting
  value 0.02 plus or minus that step (0.05). About half of the ocean ended at 0.07 and nearly all the
  rest at zero: a map of which way the very first gradient pointed.
- **The optimizer then stalled.** The gradient faded from 0.13 to around 0.00000001, consistent with
  cells being pushed outside the allowed 0–0.09 range, where they receive no gradient at all.

So the Tier-1 success (−0.1023 ± 0.0014 K on 20 fresh climates) is a real property of *this map*, and
of a first step that happened to be about the right size. It is not evidence that gradient descent
designs good patterns. The paper plan had already demoted the result to "amplitude calibration"; the
history says even the amplitude came from the step size, not from optimization. Neither half of the
planned title, "Gradients Design, They Do Not Train", has been tested yet. The fix is cheap: rerun
Stage 1 with the cooling target as the objective, averaged over several starting climates and scored
on the registered final-10-day window, with a map whose cells cannot get permanently stuck, picked on
held-out climates, and compared with a uniform map using the same total brightening.

## 16.3 Other problems found

- **The "gradients are correct" claim rests on an old check.** Parts 2, 4 and 11 lean on Stage 0's
  single test: a 10-day run, one number (a uniform brightening level), on the *old* surface-albedo
  actuator that the first audit replaced. It has never been repeated for the cloud-albedo actuator,
  over 60 days, or region by region.
- **A planned paper figure would come from the wrong run.** The paper plan builds figure F8 ("what the
  actuator touches") and its numbers from `coupled_baseline_atm.nc`, a May 27 run from before the
  rebuild. Its starting ocean had the same temperature all the way around each latitude and its land
  was frozen at 288 K, so it does not show the climate the campaigns used.
- **The simulated climate is about 3.8 K too cold.** During the 10-year settling run the ocean cooled
  from the observed 283.9 K to about 280.1 K (simple averages over ocean grid cells). The slab ocean
  has no **Q-flux** (the standard fixed heating term that stands in for ocean currents) and nothing
  stops its water cooling past freezing. The paired design cancels this bias out of every comparison,
  but any future stratocumulus experiment needs the realistic cold water off Peru and Namibia that
  makes those clouds form. *Fixed on 2026-09-30 with a Q-flux (Part 17, Step 0): the ocean now settles
  0.26 K below observed.*
- **"Watching El Niño adds nothing" is partly built in.** The feedback controller's main input is the
  average ocean temperature change, and that average *includes* the patch where the pacemaker holds El
  Niño. By the paper plan's own arithmetic, the patch alone is 53% (steady) to 69% (growing) of the
  whole disturbance effect. So more than half of the disturbance lands in the feedback sensor almost
  at once, with no noise, leaving a forecast of El Niño little to add. This belongs in the catalogue of
  self-fulfilling designs, next to Parts 7 and 13.
- **The training failure has three causes tangled together.** The controllers were trained on 60-day
  runs (beyond the two to four weeks over which Dubey et al. found gradients useful); they output a
  full map with about 1.25 million adjustable numbers to learn what is essentially a one- or two-knob
  rule; and they got 40 training steps. As it stands, the negative result means "this setup at this
  budget", not "gradients cannot train controllers".
- **"PI controller" is the wrong name.** The "I" in PI is integral action: a running memory of past
  error. Neither controller keeps one. The efficacy controller estimates how strongly the spraying is
  working and divides by that estimate (an **adaptive controller**, from the self-tuning regulators of
  the 1970s), and the El Niño controller combines a proportional correction with a forecast term.
  Calling them "hundred-year-old PI" will draw a control engineer's objection.
- **Staggered start seasons will hand feedback a free win unless the fixed map is re-tuned.** The map
  leans south and tropical (56% of its area-weighted brightening is in the southern hemisphere), was
  tuned for January (southern summer), and sits at 0.07 against the 0.09 cap, so it can only be
  scaled up about 1.3×. Starting in July changes how well it works. The fair opponent is a fixed map
  re-tuned for each start month, written into the rules in advance. Done that way, seasons become a
  natural test of "recruit more area", because *moving* the spraying beats *scaling* it.
- **The efficacy uncertainty is small next to the real one.** Tier 2 hid a multiplier between 0.6 and
  1.4. A 2026 multi-model protocol for cloud brightening (Hirasawa et al., GeoMIP "G6-1.5K-MCB")
  found that the sea-salt emissions needed for the same cooling differ about 20-fold across three
  major climate models.
- **The "noise floor" will not look new to climate scientists.** Weather noise between runs, and
  ensembles started from imperceptible nudges, are standard practice; the large CESM ensembles call
  these **micro perturbations**. What this project lacks is the matching **macro** half (starts from
  genuinely different ocean states, taken from different years of a long control run), which is the
  paper plan's "one climate, n = 1" worry.

## 16.4 What to run next

Ranked by how much each experiment decides per GPU-hour. Costs are rough, scaled from the
meta-audit's planning numbers.

| # | Experiment | What it decides | ~GPU-h |
|---|---|---|---|
| 1 | Re-check the gradients in the current setup against brute-force finite differences, region by region, at 10, 30 and 60 days from several starting climates; also try blocking the gradient through the atmosphere so it flows only along the direct sunlight-to-ocean path | Whether the gradients are right today, and whether that direct path keeps them useful longer than in Dubey et al.'s land-temperature setup | 1–2 |
| 2 | Rerun Stage 1 properly (16.2) and compare with a uniform map of equal total brightening | Whether "gradients design" is true at all | 3–5 |
| 3 | Use gradients to tune only the classical controller's two knobs (the tuning sweeps already give the answer), then a controller with only a few knobs | Whether training failed because of bad gradients or because of the huge map and tiny budget | 3–5 |
| 4 | Train with gradients cut off after 8–14 days, or through the direct path only | Whether "gradients don't train" narrows to "gradients through chaos don't train" | 4–8 |
| 5 | Dubey-style short-horizon optimization under hidden efficacy: the optimizer's internal model assumes efficacy 1, the world doesn't; compare with the classical controller and the fixed map | Whether plan-as-you-go survives a wrong model; new, and engages Dubey et al. directly | 8–16 |
| 6 | Starts from genuinely different oceans (one per year of a longer control run), then staggered start seasons with re-tuned fixed maps | Fixes "one climate"; sets up a real test of "recruit more area" | 6–10 |
| 7 | A realistic sensor (the controller compares the actual temperature with a climatology, not with a same-weather twin), with the El Niño patch left out of both the sensor and the score; re-run the "watch El Niño?" test | Whether the anticipation null survives realistic sensing | 3–6 |

The paper plan's own "killer experiment" (gradient training with micro-ensembles against a
gradient-free search at the same cost) is still worth running, with two additions: run it on the
few-knob controller as well, and include **ensemble Kalman inversion**, the climate community's
standard gradient-free method for tuning chaotic models, next to evolution strategies.

## 16.5 What it means for the paper

- **Change the lead.** With Dubey et al. out, "gradients through chaos are only useful for a couple
  of weeks" and "gradients can design an intervention in JAX-GCM" are no longer new. What is still
  ours: an ocean that responds, a sunlight-reflecting actuator, hidden uncertainty and disturbances,
  learned versus classical controllers, and a measurement protocol strong enough to settle the
  question Dubey et al. could not. Lead with experiments 1–4 and the closed-loop results; keep the
  noise floor and the catalogue of self-fulfilling designs as the methods backbone.
- **Move quickly.** Dubey et al. name model predictive control and higher-dimensional forcing as their
  next steps. A preprint once experiments 1–4 are done protects what is distinctly ours.
- **Consider getting in touch** with Dubey, Abbot and Chattopadhyay (same model; experiment 5 is a
  natural joint project) and with the MacMartin group behind Lee et al.
- **Citations to update.** Cite the model paper (Davenport, Madan et al. 2026, "JCM v1.1",
  *Geoscientific Model Development* 19:6451). Whittaker & Di Luca is now published (*Weather and
  Climate Dynamics* 7:393–410, 2026). For Lee et al. 2025, a free preprint exists on ESS Open Archive
  (doi:10.22541/essoar.172107989.92419868/v1), but it has to be downloaded by hand.

## 16.6 Housekeeping

- The settled base climate (`equilibrated/base_carry.pkl`) and every set of starting climates
  (`ics_*`) exist only on the GPU machine, and every campaign depends on them. Archive them, convert
  the pickle files to NetCDF with checksums, and deposit them (for example on Zenodo) for the
  journal's open-data requirement.
- Put the remaining pre-registrations on OSF, a public registry, so they carry timestamps no one
  controls.
- Get a human check. The same AI-assisted workflow designed, ran and audited these campaigns, and
  several of the problems above got past more than one of its audits. Have the advisor or a colleague
  recompute the headline numbers from the raw data with their own code, and disclose the AI
  assistance as the journal requires.
- Stop spending on more imitation or fine-tuning repeats (five already agree), on rainfall analyses in
  this configuration, and on new El Niño variants until the sensor problem in 16.3 is fixed.

## 16.7 What this part corrects in earlier parts

This report keeps its history, so earlier parts stay as written. Read them with these corrections:

- **The one-paragraph version, Part 11 (result 5) and Part 13** say the controllers cancel "85–89%"
  of El Niño's effect, and call it "the first time anyone has closed a control loop … against real
  year-to-year climate variability". The disturbance is a prescribed tropical temperature patch that
  ramps up and then holds (Part 15), and more than half of its measured effect is the patch itself
  (16.3). The paper plan's corrected figures are 85.0% steady / 67.0% growing (signed) and
  83.2% / 66.9% (sign-agnostic).
- **Part 11 (result 6) and Parts 14–15** say observing the disturbance is worth nothing. That holds
  for a perfect sensor that already contains most of the disturbance (16.3), not in general.
- **Parts 9, 11, 13 and 14** call the classical controllers "PI" and "hundred-year-old" (see 16.3).
- **Part 10** calls H5 "the headline of the entire project". It shows the classical controller beating
  the fixed map, reached by copying the classical controller; the paper plan already demotes it.
- **Parts 8 and 11** say differentiable optimization designed a good static plan and that design is
  the model's "proven value" (see 16.2).
- **Parts 2, 4 and 11** say the gradients were tested and confirmed; that rests on the 10-day,
  one-number check on the old actuator (16.3).

> **New terms:** *Receding-horizon control* = plan only a short way ahead, act, then re-plan from
> where you actually are. *Reinforcement learning* = an AI that learns by trial and error from a
> score, without gradients through the model. *Zeroth-order vs first-order* = learning from scores
> alone vs learning from gradients. *Finite differences* = the brute-force check of a gradient: nudge
> an input, rerun, see how the output moves. *Q-flux* = a fixed heating term standing in for ocean
> currents in a slab ocean. *Adaptive controller* = one that estimates how strongly its actuator works
> and adjusts for it. *Macro vs micro perturbations* = starting runs from genuinely different ocean
> states vs from the same state nudged imperceptibly.

---

# Part 17 — The new direction: gradients that last for months

## Why this, and why now

Part 16 left two facts side by side. First, the claim this project was about to publish —
"gradients design, they do not train" — had not really been tested on either half. Second, a
paper posted in September 2026 (Dubey, Abbot & Chattopadhyay) uses the very same model to steer a
climate simulation with gradients, but only by never looking more than two weeks ahead, and in a
setup with no ocean that responds.

That second fact points at the opening. Think of a gradient as a long chain of "if I nudge this,
that changes" links running backward through every simulated day. The links that pass through the
**atmosphere** amplify noise: weather is chaotic, so after a few weeks the chain is mostly noise
(Dubey et al. measured it: their gradients agree with brute-force checks at 7 and 14 days, and only
loosely at 28). The links that pass through the **ocean** do the opposite — a slab ocean soaks up
heat slowly and forgets slowly, so its links shrink gently instead of exploding. Two weeks of
planning is fine when nothing remembers for longer than two weeks; with an ocean, today's cloud
brightening keeps cooling the water for months, and a planner that only looks two weeks ahead is
short-sighted.

So the new question is: **can we snip the gradient's chain through the atmosphere every few days,
keep its chain through the ocean intact, and get gradients that stay useful for months?** If yes,
that is new in a specific way. Sugiura et al. (2008) already got *approximate* months-long gradients
in a coupled model by blurring them and damping them by hand. Nobody has cut only the atmosphere
while keeping the gradients exact, checked the result against brute force, or used it to design an
intervention. It also turns this project's biggest failure into its motivation, and it would matter
to anyone building differentiable climate models with an ocean. If no, measuring exactly where and
why it fails is still publishable.

## Step 0 — getting everything ready (status as of 2026-09-29)

Step 0 is preparation: nothing in it produces a scientific result, but every later experiment
depends on it.

- **The gradient cut** (`jcm/mcb/gradient_truncation.py`). Every W days it stops the gradient from
  flowing back through the atmosphere's dynamical state — its memory from one day to the next — and
  leaves everything else alone. That "everything else" matters: the coupled model hands the previous
  day's heat flux to the ocean at the start of each day, and the cloud-brightening field itself
  travels in the same part of the model's state, so cutting too much would silently cut the very
  path from brightening to ocean cooling. The cut changes only the backward pass — every simulated
  temperature is bit-for-bit identical with and without it. **Done and tested**: 14 tests check it
  against an independent hand-derived calculation on a toy with the real model's data flow. Three
  more tests on the real coupled model confirm three things: simulated values are bit-identical with
  and without the cut; a window longer than the run reproduces ordinary backpropagation exactly; and
  the same-day-only gradient still carries the direct brightening-to-cooling effect.
- **Controls and objectives** (`jcm/mcb/band_basis.py`). Five brightening "knobs", each a band of
  ocean around one latitude (45°N, 20°N, the equator, 20°S, 45°S — the same layout as Dubey et al., so
  the two studies can be compared), and four things to measure: the average ocean temperature (T0),
  the north–south difference (T1), the equator-to-pole difference (T2), and the average land
  temperature (LAND), which the brightening can only reach *through* the atmosphere. **Done and
  tested.** One precision problem was found and fixed on the way: averaging ~290 K numbers in the
  model's single-precision arithmetic loses about a ten-thousandth of a degree, which matters when
  the signals are a few hundredths, so every average now subtracts a 288 K reference first.
- **Experiment 1's harness and analysis** (`run_gradient_fidelity.py`,
  `analyze_gradient_fidelity.py`). The analysis — with every pass/fail threshold — is written and
  frozen *before* any data exist, as for every campaign since Tier 1. **Done, tested on synthetic
  data, and run end to end on a laptop** on a tiny version of the experiment. That end-to-end run
  caught one real bug before any GPU time was spent (the model must be initialized before its step
  function is built, or the land component has no climatology).
- **Starting states from genuinely different oceans** (`run_generate_macro_ics.py`). Every earlier
  campaign started from one ocean state on one calendar day ("climate n = 1", Part 7). This script
  continues the settled control run and saves the ocean every two years — 16 different ocean states,
  all in the same season — then branches each into independent weather runs: 80 starting states in
  five roles (Experiment 1; design training and evaluation; controller training and evaluation), with
  every evaluation set drawn from ocean states nothing else ever touches. Training and Experiment 1
  get the even-numbered ocean states and evaluation the odd-numbered ones, so both sides are spread
  over the whole 30 years (Amendment 9 revision 0.4). The script refuses any plan that lets an
  evaluation state be used by anything else, and it reports whether the two sides come out balanced
  and whether the 16 states drift. **Written and tested on a laptop; the full run needs the GPU (~1.4
  hours).**
- **A realistic ocean temperature: the Q-flux** (added 2026-09-30; `jcm/mcb/qflux.py`,
  `run_qflux_base_climate.py`; Amendment 9 revisions 0.2 and 0.3). The simple ocean has no
  currents, so left alone it settles about 3.8 K colder than the real ocean. The standard fix is a
  **Q-flux**: a fixed, seasonal heating or cooling in each ocean cell that stands in for the missing
  currents.
  - *How it is measured.* The model runs for five years while its ocean is gently held to the
    observed temperatures, and the heat that holding adds is recorded, month by month.
  - *How it is checked.* The model then runs ten free years with that heat added. It must pass a
    check written down in advance: drift under 0.02 K per 60 days, and an average within 0.5 K of the
    observed ocean.
  - *Surprise 1.* The coupler's own Q-flux option reads its twelve monthly values as if they were
    days, so it would run through a whole year of heating every twelve days. The project now uses its
    own ocean class, which reads them as months. With no Q-flux it matches the old ocean exactly (bit
    for bit), so nothing earlier changes.
  - *Surprise 2.* Over sea ice, the observed "sea temperature" is actually the ice surface, as cold as
    237 K. That is what the model's atmosphere is built to see, so "water below freezing" there was
    never the real problem (Part 16.3); the missing currents were.
  - **Status (2026-09-30): in use. The first attempt failed its pre-written check; the one registered
    correction (Amendment 9 revision 0.3) passed.**
    - *Measuring* took 27 minutes on a laptop. The ocean needs about 11 W/m² of extra heat on average
      to stay at observed temperatures. In the sea-ice zone the heat swings enormously with the seasons,
      because there 40–60 m of water is made to follow the ice surface's yearly swing of up to 37 K.
    - *Attempt 1:* the ten free years settled without drifting (-0.0004 K per 60 days: **pass**), but
      about 1.6 K colder than the observed ocean, against a limit of 0.5 K (**fail**). That is roughly
      half the old cold bias. The chill was spread almost evenly over every latitude and was not a
      sea-ice effect. The held run had matched the observations to 0.02 K, so the measurement was
      accurate; an ocean left to vary freely simply loses a few W/m² more heat than a held one.
    - *The correction:* how fast attempt 1 cooled, and where it was heading, showed the ocean was
      about 3.2 W/m² short. That much was added evenly to every ocean cell. Only this one try was
      allowed, and it was written into the pre-registration before it ran.
    - *Attempt 2* (ten more free years with the corrected Q-flux): drift -0.004 K per 60 days
      (**pass**), and 0.26 K colder than the observed ocean (**pass**). The typical error of one grid
      cell's yearly average fell from 2.1 K to 1.2 K, and the cells off by more than 2 K fell from 1,159
      to 363 (of 3,411). The run started close to where it settled, and its yearly averages stayed
      0.05–0.31 K below observed with no trend, so the pass is not a run caught halfway.
    - *What is left:* the even heating warmed the poles roughly twice as much as the tropics. So the
      tropical ocean between 10°S and 30°N is still 0.7–0.9 K too cold, while the ocean south of 30°S
      is about 0.2–0.3 K too warm. The check is on the global average, so this is recorded as a
      limitation, not tuned further.
    - *What happens now:* the GPU script repeats this ten-year run from the corrected file
      (`qflux_monthly_t30_v2.nc`) and checks it again before Step 0 uses it. Its weather will differ,
      but the four-year average wobbles by only about 0.07 K, far less than the 0.24 K margin, so it
      should pass there too.
- **Pre-registration** (`PREREGISTRATION.md` Amendment 9). Step 0 and Experiment 1 are frozen in
  full; Experiments 2 and 3 are declared and get frozen in full after Experiment 1, before any of
  their data exist. **Written; still to be posted to OSF** (a public registry that timestamps it
  independently of this repository).
- **Literature check.** Done. Related ideas exist — training on short windows, nudging a model's
  weather toward observations to tame chaos (Lyu et al. 2018), averaging gradients over ensembles,
  ocean-only gradient systems (ECCO), and a coupled data-assimilation system with months-long windows
  (Sugiura et al. 2008) — but none cuts only the atmosphere in a differentiable coupled model and
  checks the result against ensemble truth. **The two papers that had to be read in full were read on
  2026-09-29** (the PDFs are kept locally in `literature/`, not committed).
  - *Sugiura et al. (2008)* is the closest precedent. They ran 9-month optimizations through a full
    coupled climate model and hit our exact problem: the exact gradient breaks down because of the
    atmosphere's chaos. They tamed it differently. They worked with 10-day averages, added
    artificial damping and simplified parts of the gradient by hand. They never checked the result
    against brute force, only asserting that the approximation "affects the efficiency but not
    necessarily the direction".
  - *Lu & Hsieh (1998)* used the full, exact gradient in a simple linear toy model with no chaos, so
    it worked over 40 days. They warn that this may not hold for longer windows in realistic models.
    Their model trades heat between atmosphere and ocean once a day, as ours does.
  - *What this changes:* we can no longer say months-long gradients in a coupled model are new. The
    claim is narrower: exact gradients with only the atmosphere cut, the first direct test against
    brute force, and use for designing interventions (logged as Amendment 9 revision 0.1).

**Status (2026-10-01):**
- *The GPU run:* approved and started at 16:45 PDT on the GPU machine, an ASUS GX10.
- *OSF:* Amendment 9 (with revisions 0.1 to 0.4) went up at 18:42–18:44 PDT: project
  https://osf.io/pqabf/ and registration https://osf.io/2bs8p/. That was 54 minutes after Experiment
  1 started, so the posting came late. The commit was already public on GitHub before the run began,
  and the deviation is logged in `PREREGISTRATION.md`.

## Experiment 1 — does the snipped gradient match the truth?

**The truth** is measured the slow, brute-force way: nudge one brightening knob up, then down, run
the full model forward each time, and see how much the ocean temperature moves. One pair of runs is
drowned in weather noise, so this is repeated on 32 weather runs (8 different ocean states × 4
imperceptibly nudged copies) and averaged. **The candidates** are gradients computed with the
atmosphere snipped every 1, 7 or 14 days, and with no snipping at all (ordinary backpropagation). All
are compared at 15, 30, 60 and 120 days.

**The pass rule** (registered): a candidate is *useful* if its direction is within 20° of the truth,
its size is between 0.6× and 1.4× the truth's, and a single weather run's gradient is usually within
45° (a gradient that is only right on average is no use for training). The deciding comparison is
the average ocean temperature at 60 days. The possible outcomes, all pre-stated:

- **A — snipping rescues the gradient:** some snipped version passes and ordinary backpropagation
  fails. This is the result the new paper is built around; Experiments 2 and 3 go ahead.
- **B — no rescue needed:** ordinary backpropagation already passes for ocean temperature. Also new
  (a clean contrast with Dubey et al.'s land objective), and it means the old training failure came
  from the oversized network and tiny budget — which Experiment 3 then tests.
- **C — nothing passes:** stop spending GPU time; write up the map of where gradients break down.
- **U — the truth itself is too noisy to judge:** add more brute-force runs and repeat the truth only.

One prediction is written down in advance: the snipped gradient should *fail* on land temperature,
because the only way the ocean brightening reaches land is through the atmosphere we snipped. Seeing
that failure would show the method's limits are understood, not hidden.

**One extra comparison, added after reading Sugiura et al. (Amendment 9 revision 0.1).** A reviewer
will ask why we *snip* the atmosphere's part of the gradient instead of *damping* it, as Sugiura et al.
did. So Experiment 1 also computes a **damped gradient**. Nothing is snipped; the atmosphere's memory
simply fades a little every day, with a fade time of 3 days in one version and 7 days in the other.
It is scored with exactly the same measures, but it does not count toward the pass/fail rule or the
choice of window, so adding it cannot change the registered answer.

The expectation is written down in advance. Fading only calms the chaos if it is faster than the
chaos grows (about 0.18 per day, judging from Dubey et al.'s data). So the 3-day fade should stay
stable, and the 7-day fade is borderline. If the damped version does as well as snipping, both are
reasonable choices. If it needs its fade time tuned just right, that is a point in snipping's favour.

**The maps, added before anything runs (Amendment 9 revision 0.4).** The brute-force runs now also
keep their full ocean and land temperature maps, as 5-day averages over all 120 days. So each knob's
response *map*, which the classical designs in Experiments 2 and 3 need, comes free from runs that
are made anyway. On the gradient side, one extra pass per snipping window, run forward alongside the
model, gives each knob's whole response map at once. That is how the planner itself is meant to
compute its gradient, so the planner's map-shaped objective can later be checked against brute force
without new runs. Two automatic checks guard the new files: the maps must reproduce the four numbers
of their own runs, and the forward-pass maps must reproduce the registered backward-pass gradients.
None of this touches the pass/fail rule or the frozen analysis. On a laptop, keeping the maps added
no measurable time and gave bit-identical numbers, and the forward pass cost about 6 plain runs per
simulated day against about 11 for the registered backward pass.

**Cost:** about 12.2 GPU-hours (8.3 for the registered comparison, 2.4 for the damped one, about 1.5
for the forward-pass maps), plus 1.4 for the starting states.

## Experiment 2 — can the gradient design a pattern? (after Experiment 1)

The target: cool the average ocean by 0.1 K over 60 days while leaving the north–south and
equator-to-pole temperature differences unchanged — the kind of target used in stratospheric
aerosol studies. Brightening everything evenly cannot do it, because January sunlight falls mostly
on the southern hemisphere, so the design has to be genuinely spatial (unlike Part 16's finding that
the old design was really just a brightness setting). Four designs compete, judged on fresh ocean
states: the snipped-gradient design, the ordinary-backpropagation design, the classical design that
climate groups use today (built from brute-force sensitivities), and even brightening. A second
round with many more knobs shows the payoff: the gradient's cost stays flat as knobs are added,
while the brute-force approach grows with every knob.

## Experiment 3 — can the gradient train a controller? (after Experiment 1)

The same hidden-efficacy task as Tier 2, but with a small controller (at most about a hundred
adjustable numbers instead of 1.25 million), trained three ways with the same GPU budget: ordinary
backpropagation, the snipped gradient, and a standard gradient-free method (ensemble Kalman
inversion — the climate community's usual tool for tuning chaotic models, and the fair answer to
"why not just do what Quan et al. did?"). The trained controllers face the classical adaptive law
and the fixed map on fresh ocean states. An optional fifth contender plans two weeks ahead the way
Dubey et al. do, to measure what short-sightedness costs once the ocean remembers.

## What success would look like — and what failure would still give us

The strongest paper: Experiment 1 comes out A (or B), the gradient design matches the classical
design at a fraction of the cost and beats even brightening, and the controller trained with the
snipped gradient matches or beats the classical law where ordinary backpropagation failed. Even the
weakest outcome (C) gives a measured map of when gradients through a coupled climate model can and
cannot be trusted, which does not exist today. Whatever happens is decided by rules written down
before the data.

**Total cost:** roughly 25–35 GPU-hours across all three experiments; the first real answer
(Experiment 1) about 1–2 weeks after the GPU run is approved.

> **New terms:** *Truncation window (W)* = how many days of atmospheric history a snipped gradient
> keeps. *Ground truth (here)* = the brute-force, many-runs answer the gradients are checked against.
> *Macro vs micro starting states* = different ocean states from a long control run vs the same state
> nudged imperceptibly. *OSF* = the Open Science Framework, a public registry that timestamps a
> pre-registration so nobody can say it was written after the fact. *Damped gradient* = instead of
> cutting the atmosphere's part of the gradient every few days, let it fade a fixed fraction each day.
> *4D-Var* = the weather-forecasting method that fits a model to observations over a time window
> using gradients; Sugiura et al.'s system is a coupled version of it.

---

# Part 18 — The plan from here: every step, and why

*Written 2026-09-29, after reading Dubey et al. in full. Nothing in this part has run yet. It replaces
Part 17's sketches of Experiments 2 and 3 and its cost estimate. Those experiments were declared but
never frozen, so changing them now is within the rules. Step 0 and Experiment 1 stay as Part 17
describes them, apart from three small changes made before anything runs (steps 1–3).*

## The idea in four sentences

1. Use the model's gradient only where Experiment 1 shows it can be trusted.
2. Plan two weeks at a time and re-plan from wherever the simulated climate actually ends up, as
   Dubey et al. do, but let every plan look months ahead through the ocean, which is exactly what the
   snipped gradient is for.
3. Teach the small AI by letting it copy the planner, the way it already learned by copying the
   classical controller (Part 10), instead of training it through months of chaotic weather.
4. Judge everything the way Dubey et al. do: add warming only to the runs being controlled, measure
   against the model's own normal climate instead of a same-weather twin, and also score things the
   objective never saw, such as land temperature and rainfall.

## How the pieces fit

- **Experiment 1** decides which gradient can be trusted, and how far ahead.
- **A new test world** (steps 10–12): warming added only to controlled runs, a "normal climate"
  target, and a map-shaped score.
- **Experiment 2:** can gradients design one fixed brightening pattern?
- **Experiment 3a:** does planning months ahead beat planning two weeks ahead, the best fixed pattern,
  and the classical feedback controller?
- **Experiment 3b:** when the true spraying strength is hidden, can a small network that copies the
  planner beat the classical controller?
- **Experiment 4 (optional):** does any of it survive in a changed version of the model?

## The steps

### Phase A — before anything runs (this week, no GPU)

1. **Settle the base climate:** *done on 2026-09-30* (Part 17, Step 0; Amendment 9 revisions 0.2 and
   0.3). The first Q-flux left the ocean 1.6 K too cold and failed the pre-written 0.5 K check; the
   one registered correction passed (0.26 K too cold, no drift), so Step 0 starts from the Q-flux
   climate.
2. **Alternate training and test ocean states:** *done 2026-10-01* (Amendment 9 revision 0.4).
   Training gets the even-numbered ocean states and evaluation the odd-numbered ones, instead of the
   first eight and the last eight, because the ocean's average temperature wanders slowly (in the
   Q-flux settling run its yearly averages moved over a range of about 0.25 K, staying high or low
   for a few years at a time), so over Step 0's 30 simulated years a first-half/second-half split
   could train and test on measurably different climates. It also keeps the states on each side four
   years apart instead of two, so they are closer to independent, which Experiment 1's statistics
   assume.
3. **Keep the maps from Experiment 1:** *done 2026-10-01* (Amendment 9 revision 0.4). Experiment
   1's brute-force runs also save their full ocean and land temperature maps (5-day averages),
   without touching its frozen analysis, because the same runs then give each knob's response map,
   which the classical designs in Experiments 2 and 3 need, instead of repeating those runs later
   (about 3.5 GPU-hours). The gradient side gets the matching maps from one forward pass per window
   (about 1.5 GPU-hours), because checking the planner's map-shaped objective needs both sides, and
   the forward pass is how the planner itself is meant to compute its gradient.
4. **Post the rules publicly:** *done 2026-10-01, but late.* Amendment 9 with revisions 0.1 to 0.4
   is on OSF (project https://osf.io/pqabf/, registration https://osf.io/2bs8p/), posted 54 minutes
   after Experiment 1 started instead of before the GPU run. The commit was public on GitHub before
   the run, and the deviation is logged in `PREREGISTRATION.md`. Revision 1 should be posted before
   any Experiment-2 or Experiment-3 run, because an outside timestamp is what makes "decided before
   the data" believable to a reviewer.
5. **Read the two blocked papers:** *done 2026-09-29.* Both papers support the premise. Sugiura et al.
   turned out to be the closest precedent, so the novelty claim was narrowed and Experiment 1 gained a
   damped-gradient comparison (Part 17; Amendment 9 revision 0.1).
6. **Decide about contacting Dubey et al.:** ask the advisor whether to write to Dubey, Abbot and
   Chattopadhyay now, because their paper names "optimizing several segments ahead" as their own next
   step, which overlaps with the planner here, so early contact avoids a race or opens a collaboration.

### Phase B — Step 0 and Experiment 1 (GPU, about 14 GPU-hours)

7. **Run it:** once approved, run the laptop smoke test and then `run_campaign_step0_exp1.sh` on the
   GPU machine, because every step after Phase B depends on which gradient Experiment 1 says can be
   used.
8. **Check the sixteen ocean states:** before anything uses them, check how different the 16 states
   are and whether their average temperature trends across the 30 years, because starting from
   different oceans only helps if the states really differ and are not drifting.
9. **Follow the pre-written rule:** run the frozen analysis and act on its outcome exactly as
   registered (A or A′: the planner uses the best snipped window; B: it uses ordinary
   backpropagation; U: add brute-force runs; C: stop and write up the gradient map), because the
   choice of gradient must come from a rule written before the data, not from looking at the numbers.

### Phase C — build the new test world (code, in parallel with Phase B)

10. **Warming only where we control:** *built 2026-10-01.* Add a steady extra heat input to the slab
    ocean, only in the runs being controlled, using the same fixed-heating slot a Q-flux uses, because
    this warms the ocean steadily the way greenhouse gases do, and unlike the CO2 knob (Part 15) it does
    not cancel out of the comparison. The warming can be steady, growing, or both. On the real model,
    zero warming changes nothing bit for bit, and 20 W/m² warms the ocean by the textbook amount.
11. **The normal-climate target:** *built 2026-10-01.* For every starting state, average five runs
    with no warming and no brightening into one smooth "normal climate" path, because controllers then
    steer toward, and are judged against, the model's own normal climate while living with their own
    weather, which replaces the perfect same-weather-twin sensor that Part 16 put first among its
    concerns. The same code also builds the uncontrolled warmed run that the gain compares with.
12. **A map-shaped objective and fair scores:** *built 2026-10-01.* Implement Dubey et al.'s pattern
    objective for ocean temperature (the average error squared, plus half of the error map's variance,
    plus small penalties on effort and on sudden changes) and their gain and effort scores, and apply
    the gain also to land temperature, rainfall and evaporation, which the objective never sees,
    because a map-shaped target forces genuinely spatial designs and out-of-objective scores catch side
    effects. The objective can also be differentiated through the 14-day snip, ready for the planner.

**The test world as built** (`jcm/mcb/test_world.py`, `jcm/mcb/scores.py`, `run_test_world.py`;
2026-10-01):

- *What it is.* Warming in the controlled runs only; the normal-climate and warmed references for each
  starting state; Dubey et al.'s objective, gain and effort; and an "episode" runner that asks a
  controller for its five band settings every few days. Fixed patterns, the feedback controller, the
  planner and the student all plug into the same runner. A "hidden spraying strength" setting is
  included for Experiment 3b.
- *Checked:*
  - 24 tests on a small stand-in model, and 4 on the real coupled model with the corrected Q-flux.
  - Rainfall in the settled Step-0 climate averages 2.9–3.5 mm/day, close to Earth's, which
    confirms the units.
  - Our five bands reproduce Dubey et al.'s stated 0.925 area mean of the summed band profile.
  - An end-to-end smoke run on a laptop worked.
- *A fairness rule found while testing.* Scoring one weather sample against a five-sample average
  made rainfall look about twice as damaged as it was, because the average had shed most of its
  weather noise and the single sample hadn't. Each scored run is now averaged over as many weather
  samples as the references, started from the same seeds.
- *A guard.* The code refuses to build references for evaluation states unless told to, and they are
  built only after revision 1 is frozen, so no design choice can be tuned on them.
- *Training-state references, built 2026-10-02 on the GPU* (`run_campaign_test_world_refs.sh`; 32
  starting states; 1.4 hours, run alongside the vLLM server without stopping it):
  - A steady 4 W/m² probe warming heats the ocean by 0.10 K after 60 days, 0.19 K after 120 and
    0.36 K after 240 (0.31–0.39 across states), within a few percent of the simple slab estimate.
  - The normal climate rains 3.26 mm/day and evaporates 3.27 mm/day.
  - Weather noise in a five-sample average is about 15 thousandths of a degree, so the warming
    stands about 24 times above the noise by day 240. The pilot can scale this probe linearly to
    whatever warming it chooses.
- *Left to the pilot (step 23) and revision 1:*
  - how strong the warming is and whether it grows;
  - run and scoring-window lengths;
  - how many weather samples to use;
  - the effort and change penalty weights;
  - whether rainfall and evaporation are scored over land, the globe, or both.
13. **The planner:** *built 2026-10-02* (`jcm/mcb/planner.py`; run it with `run_test_world.py
    plan`). Every 14 days, starting from wherever the simulated climate actually is, choose the five
    band settings that minimize the objective over a 60-day look-ahead using the gradient Experiment
    1 validated (averaged over three slightly nudged copies), apply them for 14 days, then re-plan,
    because this is the new method: short, trustworthy steps that still see the ocean's
    months-long memory. Both optimizers in the settings table are built: Dubey et al.'s Adam steps,
    and Gauss–Newton steps from the forward pass, which see the whole map's response at once.
14. **Planner safeguards:** *built 2026-10-02.* Keep every band setting inside its allowed range
    through a smooth transformation, log the gradient's noise-to-signal ratio at every re-plan, and
    always apply the optimizer's last step rather than its best-looking one, because these guard
    against the three failures behind the old "optimized" pattern: knobs stuck at a limit with no
    gradient, a gradient that faded to nothing unnoticed, and keeping the luckiest of many noisy tries
    (Part 16.2).
15. **Two comparison planners:** *built 2026-10-02* as presets of the same planner (`short14`,
    `bptt60`; the main one is `snipped60`). Build the same planner with a 14-day look-ahead (Dubey
    et al.'s setting) and with a 60-day look-ahead using ordinary backpropagation, because together
    with step 13 they show the dilemma and its fix: a short look-ahead is short-sighted, a long one
    without snipping is noisy, and snipping is meant to give the long view without the noise.
16. **Test the planner:** check that it finds the known best answer on a toy model, and that its
    forward-mode and backward-mode gradients agree on the real model, because a planner bug would
    silently corrupt every later result. Experiment 1 already logs this agreement on the real model
    for every window (revision 0.4), so this step mainly tests the planner's own code. *Done on
    2026-10-02:*
    - on a stand-in model whose best answer is known exactly, one Gauss–Newton step lands on it and
      Adam converges to it;
    - one re-plan also runs on the real model with both optimizers.
17. **Measure what planning costs:** time one re-plan on the GPU for each candidate setting (copies
    run side by side; a forward-mode gradient with two or three Gauss–Newton steps versus Dubey et
    al.'s 15 Adam steps; a 14- versus a 60-day look-ahead), because at Dubey et al.'s settings a year of
    60-day planning could take roughly 20–50 GPU-hours, so the budget in the rules has to come from
    measured costs, not guesses. *Done 2026-10-05 on the GX10* (`run_planner_cost.py`; results in
    `mcb_experiments_gpu/planner_cost.json`):

    | One re-plan, 3 copies | 14-day look-ahead | 60-day | 120-day |
    |---|---|---|---|
    | Gauss–Newton (3 steps) | 0.5 min | 2.1 min | 4.3 min |
    | Adam (15 steps), copies side by side | 2.4 min | 10.3 min | 21.0 min |
    | Adam (15 steps), copies one after another | 4.6 min | 19.7 min | 39.3 min |

    - *Per planned six-month run* (13 re-plans): about 0.5 GPU-hours with Gauss–Newton and a 60-day
      look-ahead, against 2.2 (side by side) to 4.3 hours with Adam. A year of 60-day Adam planning
      costs 4.5–8.5 GPU-hours, well under the 20–50 guessed above.
    - *Running copies side by side* halves Adam's cost but does nothing for Gauss–Newton, whose
      five forward-mode directions already fill the GPU.
    - *Snipping is free:* 79.0 versus 78.7 seconds per call without and with it.
    - *Memory* stays under 8 GB, so all of it runs alongside the vLLM server.
    - *The two copy modes give statistically equivalent results, not identical ones:* after about
      two weeks the GPU's chaotic weather makes them different weather samples, as forward and
      backward mode were in Experiment 1.
18. **The fixed-pattern ladder:** build Dubey et al.'s four fixed opponents (uniform brightening tuned
    to cancel the average warming, uniform brightening at the planner's effort, the planner's own
    average pattern held constant, and the classical linear-response design), because each rung
    isolates where a win comes from: how much is brightened, where, when it changes, or how the design
    was found. *Built 2026-10-05: see "Steps 18–22 as built" below.*
19. **The classical feedback controller:** build a controller that holds three numbers on target (the
    ocean average, the north–south difference and the equator-to-pole difference) with true integral
    action, as in the GLENS stratospheric-aerosol simulations, and keep the old adaptive controller for
    continuity, because this is the feedback method the field actually uses and it answers Part 15's
    open question about integral action. *Built 2026-10-05 (below).*
20. **The student network:** build a tiny network (about 100 adjustable numbers) that sees only what a
    real system could measure (ocean temperature anomalies against the normal climate, the season,
    and its own previous settings) and outputs the five band settings, because the goal is a
    controller that is cheap to run and needs no access to the model's insides. *Built 2026-10-05
    (below).*
21. **The copying loop, with a teacher that knows more:** train the student to copy the planner, then
    repeatedly let the student drive, ask the planner what it would have done in each situation the
    student reached, and retrain on the growing set (the DAgger method), with the planner told the
    hidden spraying strength and the student not, because plain copying breaks down once the
    student's small errors take it where the teacher never went, and a teacher that knows the answer
    lets the student learn to infer it from what it sees. *Built 2026-10-05 (below).*
22. **Direct-training comparisons:** train the same tiny network straight through the simulation with
    ordinary backpropagation, with the snipped gradient and with ensemble Kalman inversion, each at
    the same GPU budget as the copying route (whose budget includes its teacher), because this
    re-tests the project's founding bet, which failed in Part 9, now with a small network and
    possibly trustworthy gradients. *Built 2026-10-05 (below).*

**Steps 18–22 as built** (2026-10-05; code and tests only: nothing has run on the GPU, and no
evaluation state was touched). Part 21.3 deferred steps 20–22 "unless time allows". They were built
anyway, at no GPU cost, while the pilot ran, so revision 1 can decide whether they run.

- *The fixed-pattern ladder* (`jcm/mcb/ladder.py`, step 18). The four rungs are fixed band settings:
  1. the same brightening in every band, sized to cancel the ocean-average warming over the scoring
     window;
  2. the same brightening in every band, at the planner's effort;
  3. the planner's own time-average pattern;
  4. the classical linear-response design.

  Rung 4 is the best fixed setting inside the cap under a linear model built from runs that brighten
  one band at a time (`run_controllers.py responses`, about 5 GPU-minutes per training state). Under
  that model the objective is an exact sum of squares, so the design is an exact bounded
  least-squares fit, pooled over training states. Any response maps can be plugged in, so the same
  solver gives Experiment 2's sunlight-guess opponent (21.3) once its maps are written down. On the
  toy model, the linear model predicts a design's actual objective to round-off, and the design beats
  uniform brightening.
- *The classical feedback controllers* (`jcm/mcb/feedback.py`, step 19).
  - **GLENS-style.** It holds the ocean average, the north–south difference and the equator-to-pole
    difference on target. It uses proportional and true integral action, maps them onto the five
    bands through a sensitivity matrix, and has anti-windup, so the integral doesn't pile up while
    the bands are at a limit. A feedforward from the forecast warming, as GLENS has, is optional.
  - **Its sensitivities come free from Experiment 1's brute-force runs.** Per unit albedo held from
    day 0, the ocean average cools by 3, 9, 13, 13 and 14 thousandths of a degree a day for the bands
    at 45N, 20N, 0, 20S and 45S, nearly steadily, because the slab acts like an integrator. The
    north–south and equator-to-pole rows change sign across the bands, so the five bands can steer
    all three numbers.
  - **One tuning number.** The gains follow from a single closed-loop time scale instead of a tuning
    sweep. Each two-week ocean-average reading of a single weather sample is off by about 0.022 K
    (measured from the references). In a simulated integrator world with that noise and Experiment
    1's sensitivities, a 42-day time scale gave the smallest ocean-average error over days 98–182.
    The error was 0.019 K, against 0.022 K at 28 days, 0.026 K at 56 days, 0.050 K at 84 days and
    0.19 K with no control: faster gains chase noise and slower ones lag. 42 days is the default;
    revision 1 freezes the value.
  - **The old adaptive controller, ported for continuity.** It is one fixed pattern times one gain:
    it estimates how strongly the spraying works from realized versus expected cooling and divides
    that out. Its Tier 2 sensor (a same-weather twin) and target (a fixed cooling) don't exist in the
    test world, so it now measures realized cooling against the model's forecast of the uncontrolled
    warmed run and aims to cancel that forecast warming. In the integrator world it recovers hidden
    strengths of 0.5, 1 and 2 to within 20% and cancels a growing warming to within 10%.
  - **The tests check the textbook behaviour:**
    - integral action removes a steady miss that proportional control alone would keep;
    - the feedforward halves the error against a growing warming;
    - anti-windup halves the overshoot after a spell at the cap.
- *The student network* (`jcm/mcb/student.py`, step 20). It has 113 adjustable numbers.
  - *Inputs:* the ocean temperature anomaly under each band over the last two weeks, against the
    normal climate (5); the season (2); its own previous settings (5).
  - *Network:* one hidden layer of 6 units.
  - *Outputs:* five settings, always inside the cap.

  Feeding back its previous settings makes integral-like behaviour learnable. **A caveat for
  revision 1:** every macro state starts on the same calendar date, so "the season" also tells the
  student how long the episode has run, and so how much warming to expect. `--no-season` switches it
  off, and the July mini-replicate (21.3) would separate the two.
- *The copying loop* (`jcm/mcb/imitation.py`, step 21). This is DAgger with a privileged teacher.
  - In round 0 the planner, told the episode's hidden strength, drives.
  - In every later round the student drives. The planner is asked what it would have done in each
    state the student reached, and the student is refitted on everything collected.

  It was tested on the toy with the real planner as the teacher. Told a weaker spraying, the planner
  asks for more brightening, and the student's copying error falls across rounds.
- *Direct training* (`jcm/mcb/direct_training.py`, step 22).
  - **The closed loop inside JAX.** The student's whole episode is one differentiable function. It
    reproduces the ordinary episode runner's settings and temperatures to round-off, and its gradient
    matches finite differences.
  - **Three trainers:**
    - ordinary backpropagation;
    - the snipped gradient (W* = 14);
    - ensemble Kalman inversion. It uses forward runs only, with the ensemble run side by side. The
      Kalman gain is computed in the ensemble's own space, so a misfit vector the size of a map costs
      little.
  - **Equal budgets.** Each method stops when its budget of GPU-seconds is spent. The copying loop's
    log records its total, teacher included, which is the budget the direct methods get. All three
    lower the objective on the toy.
- *Runners.* `run_controllers.py` has the responses, ladder, feedback and student-episode stages;
  `run_student_training.py` has the copying loop and direct training. Both refuse evaluation states,
  and the training runner accepts only `*_train` roles. `run_test_world.py` now shares its episode
  options with them; its behaviour is unchanged.
- *Checked:* 60 new tests on the toy and on synthetic data, all passing, with lint clean.
  - *Smoke run:* all nine stages ran on one real training state on the laptop CPU (two segments of
    two days). In the copying loop's real-model smoke the student's copying error fell from 19% to
    12% of the cap after one round, and each direct method was handed exactly the copying loop's 208
    GPU-seconds.
  - *One fix came out of it.* Ensemble Kalman inversion first set its noise level per misfit
    component. With map-sized misfits and a small ensemble, that left the update undamped, and one
    step shrank the ensemble's spread from 0.1 to 0.0013, which ends its search. The noise is now
    scaled to the ensemble's own spread, and on the real model the spread narrows gently (0.10, 0.074,
    0.057, 0.042). A test guards it: on a problem of the same shape the old scaling keeps 3% of the
    spread after one step, the new one 55%.
- *Left for revision 1:*
  - the controller's time scale and whether it gets the forecast;
  - the ladder's design window;
  - the hidden-strength range (a placeholder of 0.5–2 here);
  - the copying loop's rounds and episodes;
  - the direct methods' step sizes and ensemble size;
  - whether steps 20–22 run at all.
- *GPU cost if they run* (step 17's speeds). One copying round of 8 episodes with the Gauss–Newton
  60-day teacher is 8 × 13 re-plans × 2.1 minutes, about 3.6 GPU-hours. Four rounds are about 15
  GPU-hours, which each direct method then also gets, so about 60 GPU-hours for steps 21–22
  together.

### Phase D — fix the rules for Experiments 2 and 3 (after Experiment 1, before any of their data)

23. **Pilot on training states only:** use training states to pick the warming strength (several
    times the weather noise, well inside what brightening can cancel), the long planner's look-ahead
    (60 or 120 days), its optimizer and number of copies, and the penalty weights, and to check the
    planner's per-knob response maps against Experiment 1's brute-force maps, because each of these
    must be fixed before any evaluation state is touched.
24. **Freeze revision 1:** write and freeze the full rules for Experiments 2 and 3 (arms, budgets, a
    run length of about six months so each run crosses from southern to northern summer, a
    hidden-strength range justified against the roughly 20-fold spread between climate models,
    endpoints, tests, equivalence bounds and the outcome grid), write the analysis script, and post
    both to OSF, because this project's recurring failure was a design whose answer was fixed, or
    picked, after the fact.

### Phase E — Experiment 2: can gradients design a fixed pattern? (about 10–15 GPU-hours)

25. **Design it four ways:** on the training states, design one fixed five-band pattern that cancels
    the warming over the validated horizon with the snipped gradient, with ordinary backpropagation,
    with the linear-response method and with uniform brightening, because "gradients design" is half
    of the planned paper's claim, and Part 16.2 showed it was never really tested.
26. **Judge on fresh states:** score all four designs on the 16 evaluation states with
    micro-ensembles, on the ocean map and on land temperature and rainfall, because a design only
    counts if it works on climates nothing was tuned on.
27. **More knobs (optional):** repeat the design with 7 bands and with 30 patches, recording GPU cost
    next to the linear-response method's, because the practical case for differentiable models is
    that the gradient's cost stays flat as knobs are added while the brute-force cost grows with every
    knob.

### Phase F — Experiment 3: planning and copying over months (the big one; cost set by step 17)

28. **3a, planning with a perfect model:** over about six months of steadily growing warming, run the
    three planners, the classical feedback controller and the fixed-pattern ladder on the evaluation
    states, because this measures whether looking months ahead through the ocean beats two-week
    planning, the best fixed pattern and the classical controller, and whether changing the pattern
    with the seasons pays off for a sunlight-driven actuator, a question Dubey et al. could not
    resolve.
29. **3b, train the students:** with the spraying strength hidden, record the all-knowing planner's
    decisions on the training states, then run the copying loop and the three direct-training
    methods, because this produces the networks to be compared.
30. **3b, the head-to-head:** on the evaluation states, compare the student with the classical
    feedback controller, a planner that does not know the hidden strength, the all-knowing planner
    (the ceiling), the fixed-pattern ladder and the directly trained networks, reporting GPU cost next
    to skill, because the paper's central question is whether a network that learned from the model's
    gradients, by copying a planner, can beat the classical controller under realistic uncertainty at
    a small fraction of the planner's cost.

### Phase G — Experiment 4 (optional): does it survive a changed model? (about 5–10 GPU-hours)

31. **Replay in changed models:** replay the fixed designs and the planner's recorded schedule, and run
    the feedback controllers live, in altered versions of the model (a shallower or deeper ocean,
    brightening applied to the stratocumulus clouds instead of the convective ones, a different
    resolution), because any real strategy would be designed in one model and used in another, Dubey
    et al.'s cross-model replay sets that standard, and our feedback results predict that closing the
    loop is what protects performance.

### Phase H — write-up and housekeeping

32. **Preprint quickly:** update `PAPER_PLAN.md` to this story and post a preprint as soon as
    Experiments 1–3 have their registered answers, because the paper plan still describes the old
    claim and Dubey et al. may reach planning ahead soon.
33. **Make it checkable:** archive the settled climate and every set of starting states as NetCDF with
    checksums (for example on Zenodo), have the advisor or a colleague recompute the headline numbers
    with their own code, disclose the AI assistance, and fix the names, figure and citations Part 16
    flagged, because every result depends on files that exist only on the GPU machine, and several
    problems got past more than one AI-run audit.

## Planner settings to start from (fixed for good in step 23; the fixed values are in Part 22.4)

| Setting | Starting value | Why |
|---|---|---|
| Knobs | Experiment 1's five Gaussian ocean bands | comparable with Experiment 1 and with Dubey et al. |
| Re-plan every | 14 days | Dubey et al.'s window, inside which gradients stay reliable |
| Look-ahead | 60 days (14 for the short-sighted planner) | the ocean remembers for months; 60 days is Experiment 1's deciding horizon |
| Gradient | Experiment 1's winning snipped window (ordinary backpropagation if the outcome is B) | only what Experiment 1 validated |
| Copies averaged | 3, or 1 if Experiment 1 shows single runs are reliable | weather noise; one copy costs a third as much |
| Optimizer | 2–3 Gauss–Newton steps with a forward-mode gradient, or 15 Adam steps at learning rate 0.1 (Dubey et al.), whichever the pilot shows is accurate at lower cost | with only five knobs, running the gradient forward alongside the model is cheap and gives each knob's whole response map at once |
| Objective | ocean map: average error squared + 0.5 × error-map variance + effort and change penalties | Dubey et al.'s pattern objective; their penalty weights (0.01 and 0.1) were set for cooling in kelvin, not for albedo, so ours are tuned in the pilot |
| Limits | each band between zero and the brightening cap, through a smooth transform | no knob can get stuck with zero gradient |
| What gets applied | the optimizer's last step | never the luckiest try |

Experiment 1's brute-force runs also give the classical feedback controller its sensitivity matrix (how
much each knob moves each of the three numbers it tracks), so the classical and gradient methods start
from the same information.

## Budget: a lean and a full version

The planner and the copying loop are new and expensive, so this plan costs more than Part 17's
25–35 GPU-hours. Step 17 replaces these estimates with measured numbers.

| Phase | Lean (GPU-hours) | Full (GPU-hours) |
|---|---|---|
| A: base-climate fix (done on a laptop; its GPU re-settle is counted in B) | 0 | 0 |
| B: Q-flux re-settle, Step 0 and Experiment 1 (with the damped comparison and the maps) | 14 | 14 |
| C–D: tests, scoping, pilot | 3 | 5 |
| E: Experiment 2 | 8 (no extra knobs) | 10–15 |
| F: Experiment 3 | 20–30 (one copy, 4-month runs, the snipped and short-sighted planners only, one copying round, snipped-gradient direct training only) | 70–170 |
| G: Experiment 4 | none | 5–10 |
| **Total** | **about 45–55** | **about 105–215** |

*Measured planning costs (step 17, 2026-10-05) change the Experiment 3 row.* One planned six-month
run costs about 0.5 GPU-hours with Gauss–Newton at a 60-day look-ahead (0.1 at 14 days), or 2.2 with
Adam. The biggest lever is how many weather samples each evaluation state gets:

| Experiment 3a planners | Gauss–Newton | Adam (side by side) |
|---|---|---|
| 24 states × 1 sample × 3 planners | about 25 GPU-hours | about 120 |
| 24 states × 5 samples × 3 planners | about 125 GPU-hours | about 600 |

The three planners are snipped60, short14 and bptt60. Five samples would match the references'
noise level. Revision 1 should weigh Gauss–Newton and the number of samples against the budget the
advisor chooses.

Rough calendar: Phase A this week; Experiment 1's answer a few days after the GPU run starts; Phase C
coded in the meantime; revision 1 about a week after Experiment 1; Experiments 2 and 3 over the
following two to four weeks, depending on the budget; a preprint roughly two months from now.

## If things go differently

| If … | then … |
|---|---|
| Experiment 1 comes out C (no gradient is trustworthy at 60 days) | Experiments 2 and 3 stop; the gradient map becomes a short paper on its own |
| Experiment 1 comes out B (ordinary backpropagation is fine) | the planners use it; snipping becomes a side result, and Experiment 3a is about look-ahead length alone |
| Planning costs far more than expected | the lean version, or fewer states, shorter runs and one copy, all fixed in revision 1 before any Experiment-3 data |
| The planner only ties the best fixed pattern or the classical controller | report the tie with equivalence bounds; Dubey et al. explain such ties by a flat loss near the optimum, and the case for gradients then rests on cost as knobs are added (step 27) |
| The student falls well short of its teacher | report it; the planner itself is still the method, and the gap measures what copying loses |

## Decisions for the advisor

1. **Add the Q-flux before Step 0 runs (step 1)?** *Decided yes and done on 2026-09-30.* The first
   attempt halved the cold bias but failed its pre-written check; the one registered correction
   passed (Part 17, Step 0). Nothing is left to decide here.
2. **Which budget,** lean (about 50 GPU-hours) or full (about 100–200), and on which machine?
3. **Contact Dubey et al. now (step 6)?**
4. **Is a steady extra heat input into the ocean an acceptable stand-in for greenhouse warming
   (step 10)?** Imposed ocean heating is a common device in idealized slab-ocean studies (for example
   Kang et al. 2008), but it heats the surface rather than the top of the atmosphere.

## Limits this plan does not fix

- The brightening still acts on the model's convective clouds, not the stratocumulus decks real cloud
  brightening would target; only Experiment 4 touches this.
- There are no aerosol particles, so air-quality questions (PM2.5) stay out of reach.
- The slab ocean has no currents, so there is no real El Niño, and far-away rainfall effects stay weak.
- The controllers are given the normal climate exactly; a real system would only have an estimate.
- Everything happens in one model; Experiment 4 tests changed versions of it, not a different model.

> **New terms:** *Planner / model predictive control* = choose the next settings by simulating ahead,
> apply only the first stretch, then re-plan (receding-horizon control, Part 16, is the same idea).
> *Look-ahead* = how far ahead each plan simulates. *Forward-mode gradient* = working out how every
> output responds to each knob by carrying the sensitivities forward alongside the simulation, which is
> cheap when there are few knobs. *Gauss–Newton* = an optimizer that treats the response as locally
> linear and solves for the best settings directly. *DAgger* = an imitation method in which the student
> drives and the teacher corrects it in the situations the student actually reaches. *Privileged
> teacher* = a teacher that sees information the student cannot, so the student learns to infer it.
> *Integral action* = a controller's running memory of past error, which removes steady misses.
> *Gain (G)* = the doing-nothing run's squared error against the normal climate divided by the
> strategy's (10 means ten times smaller). *Effort (E)* = how much brightening a strategy asks for.

---

## Where to look next in the codebase

- **`MCB_META_AUDIT.md`** — the second audit, and (in its addenda) the full Tier-1/2/2b campaign numbers behind Parts 7–10.
- **`MCB_IMPLEMENTATION_PLAN.md`** — the detailed engineering log of the original effort: the first audit (R1–R7), the rebuild, and campaigns v1/v2/v3.
- **`PREREGISTRATION.md`** — the frozen rules, with the formal amendment log (Amendments 1–8 plus three revisions) covering every design decision from Tier 1 through the growing-ramp campaign, and Amendment 9 freezing Part 17's Step 0 and Experiment 1.
- **`jcm/mcb/gates_stats.py`** — the paired t / Wilcoxon / equivalence (TOST) statistics behind every verdict, in pure NumPy with its own test suite.
- **`run_confirmatory_eval.py`** — the micro-ensemble evaluation harness (and the PI controller) used by all three campaigns.
- **`run_noise_floor.py --cross-process`** — the corrected chaos-noise measurement (Part 7.2).
- **`run_pi_imitation.py`** — the Tier-2b distillation of the PI law, including the feature-anchoring fix.
- **`run_campaign_confirm.sh` / `run_campaign_tier2.sh` / `run_campaign_tier2b.sh`** — the gated, resumable campaign scripts exactly as run.
- **`analyze_tier2.py` / `analyze_tier2b.py`** — the primary analyses, written and frozen before the data existed.
- **`jcm/mcb/enso.py` / `run_enso_scoping.py` / `analyze_enso.py`** — the ENSO pacemaker, its calibration run, and the frozen Amendment-6 analysis (Part 13).
- **`analyze_enso_mechanism.py`** — the post-hoc disturbance-rejection analysis that produced Part 13's surviving result.
- **`analyze_enso8.py`** — the pre-committed analysis for Parts 14-15: wild-bootstrap/HC3 inference, the sign-agnostic RMS_A endpoint, and the held-out comparator selection.
- **`run_campaign_ramp.sh`** — Part 15's growing-disturbance campaign; **`run_co2_scoping.py`** and the `co2_rate` plumbing are kept as the record of the CO2 route that could not work.
- **`analyze_precip_sideeffects.py`** — the Part-12 rainfall side-effect analysis (Amendment 5; meta-audit Addendum 4).
- **`jcm/physics/speedy/shortwave_radiation.py`** — where MCB correctly brightens cloud albedo (the R6 fix).
- **`mcb_experiments_gpu/stage1_v2/stage1_optimized_pattern.pkl`** (its `history`), **`run_stage0_plumbing_test.py`** and **`jcm/mcb/coupled_features.py`** (feature 0) — the evidence behind Part 16's three main code findings.
- **`jcm/mcb/gradient_truncation.py`** — Part 17's gradient cut (atmosphere snipped every W days, ocean kept) and the damped alternative (atmosphere's memory fades daily; Amendment 9 revision 0.1), with their tests.
- **`jcm/mcb/band_basis.py`** / **`jcm/mcb/gradient_fidelity.py`** — Experiment 1's five ocean-band knobs, its four objectives, and the truth-vs-gradient machinery, including the maps of revision 0.4 (brute-force map averages, forward-pass map gradients, and the two checks between them).
- **`run_gradient_fidelity.py`** / **`analyze_gradient_fidelity.py`** — Experiment 1's driver and its pre-committed analysis (every threshold registered in Amendment 9).
- **`run_generate_macro_ics.py`** — starting states from 16 genuinely different ocean states, in five disjoint roles (training on the even-numbered states, evaluation on the odd-numbered ones; revision 0.4).
- **`run_campaign_step0_exp1.sh`** — the gated GPU script for the Q-flux re-settle, the macro starting states and Experiment 1 (~14 GPU-h).
- **`jcm/mcb/planner.py`** — the receding-horizon planner of Part 18 steps 13–15: 14-day re-plans over a 60-day look-ahead with the 14-day snip, Adam or Gauss–Newton steps, three nudged copies, the safeguards, and the `snipped60` / `short14` / `bptt60` presets. `run_test_world.py plan` runs and scores it.
- **`jcm/mcb/test_world.py`** / **`jcm/mcb/scores.py`** / **`run_test_world.py`** — the test world for Experiments 2 and 3 (Part 18 steps 10–12): warming in the controlled runs only, the normal-climate and warmed references, Dubey et al.'s objective, gain and effort, the episode runner every controller plugs into, and the planner's differentiable look-ahead objective.
- **`jcm/mcb/ladder.py`** / **`jcm/mcb/feedback.py`** / **`run_controllers.py`** — Part 18 steps 18–19: the fixed-pattern ladder (with the exact bounded least-squares linear-response design), the GLENS-style controller with integral action and anti-windup, the ported Tier 2 adaptive law, and the stages that measure one-band responses, design the ladder and score the controllers.
- **`jcm/mcb/student.py`** / **`jcm/mcb/imitation.py`** / **`jcm/mcb/direct_training.py`** / **`run_student_training.py`** — Part 18 steps 20–22: the 113-number student, the copying loop (DAgger with a teacher told the hidden strength), and direct training through the closed loop by backpropagation, the snipped gradient or ensemble Kalman inversion, at equal GPU budgets.
- **`jcm/mcb/qflux.py`** / **`run_qflux_base_climate.py`** — the ocean class with a monthly Q-flux, how the Q-flux is measured, and the settling run with its pre-written check (Amendment 9 revision 0.2), plus the one-step correction (revision 0.3). The Q-flux in use is `mcb_experiments/qflux/qflux_monthly_t30_v2.nc`; how it was derived is in `qflux_monthly_t30_v2_correction.json`, and its passing check is in `attempt2/`. The first attempt's file and summaries stay next to it as the record.

---

# Part 19 — What could actually be published: every idea, how finished it is, and what's left

> **Superseded in part (2026-10-02).** Experiment 1 has now run (Part 20), and the advisor meeting
> changed the paper plan (Part 21). Part 21 holds the current plan; this part is kept as the record of
> the first sort. Corrections to it: #1 has run (outcome A′); #8's numbers are 83–89% (step) and
> 61–67% (growing), measured on new *weather realizations* from one ocean state, and "possibly the
> first feedback loop against El Niño" is dropped (the disturbance is a prescribed patch); "Paper B,
> mostly writing" overstated its standalone value; and #9 is harder than "fairly cheap" (Part 20.4).

*Added 2026-10-01.* The project has produced a lot of results, plus a lot of things that failed or
were taken apart. This part sorts all of it into **ideas that could become published research**,
says how finished each one is, and lists what's still needed to finish it. Think of each idea as an
**ingredient**: most aren't a paper on their own, but a few of them combined make a full one.

## The chart

| # | The idea (simple version) | What we already have | What's missing to finish it | How finished? | How interesting to others? |
|---|---|---|---|---|---|
| **1** | **Gradients that stay useful for months.** Cut the gradient's path through the chaotic weather every few days, keep its path through the slow ocean (Part 17). | Idea, setup and the Q-flux ocean fix all done. Literature search found only one older, rougher version (Sugiura 2008). | **Run Experiment 1** (~14 GPU-hours): check the new gradients against the brute-force true answer. Everything else depends on this. | Set up, not run | **High** — could be the main result of a paper |
| **2** | **Gradients can design a good brightening map.** | A map that hits the target (Tier 1), but it was picked by luck after step 1, not designed (16.2). | **Redo the design properly** (Experiment 2): score it on cooling, average over several starting climates, compare against a plain "same brightening everywhere" map. Dubey et al. already did something similar, so it needs our responding ocean to stand out. | Needs redoing | **Medium** — best as one piece of #1 |
| **3** | **Copying works, training through the simulation doesn't.** The AI got all its skill from copying the classical controller, never from gradient training (Parts 9–10). | Strong results, repeated 5 times. Dubey's numbers (gradients break down after ~2 weeks) explain why. | **Untangle the three possible causes:** 60-day runs (too long), an output map of ~1.25 million numbers (too big), only 40 training steps (too few). Test a few-knob controller, short training windows, and a trial-and-error (no-gradient) method at the same cost. | Result done, explanation missing | **High** — especially for AI/machine-learning readers |
| **4** | **Plan two weeks ahead, re-plan, and let the AI copy the planner** (Dubey-style planning plus our copying trick; Part 18). | Fully planned (Experiment 3). | **Everything:** build the planner, run it, train the student AI. The most expensive item (~20–170 GPU-hours). Test whether it still works when the planner's assumed spraying strength is wrong. | Just an idea | **High** — and a natural joint project with the Dubey group |
| **5** | **The "weather noise floor":** how much random wobble there is in experiments with a differentiable climate model, and how to measure through it (Parts 7–8). | Measured noise (~12–17 mK per 60-day run), the "fake zero noise" trap, and the fixes (averaging 8 runs, twin comparison runs, proper statistics). | **Mostly writing.** Show it holds at other run lengths. Frame it as advice for experiments in differentiable climate models, since climate scientists already know weather is noisy. | Mostly done | **Medium** — very useful, not flashy |
| **6** | **The "rigged experiment" catalogue:** five times a test's answer was decided by its design before it ran. | All five documented: feedback that could never win (Part 7), the warm-only El Niño win (Part 13), the untuned knob (Part 14), CO2 cancelling out (Part 15), the El Niño inside the sensor (16.3). | **Writing only.** Turn it into a checklist other researchers can use. | Done | **Medium** — pairs well with #5 |
| **7** | **Feedback beats a fixed plan when the spraying strength is hidden** (Tier 2). | Classical controller halves the error, repeated 3 times, strong statistics. | **A realistic sensor** (no perfect twin Earth). **A bigger uncertainty range** (models disagree ~20-fold, not 0.6–1.4×). Different ocean states and seasons. **Rename the controller** ("adaptive", not "PI"). Note "feedback beats a fixed plan" is already known (Kravitz 2014, Lee 2025). | Strong but narrow | **Low on its own** — a solid supporting result |
| **8** | **Feedback cancels most of an El Niño, without needing to watch it** (Parts 13–15). | 85% (steady) and about two-thirds (growing) cancelled, held up on new climates. "Watching El Niño adds nothing" held twice. | **Fix the built-in part:** take the El Niño patch out of what the controller watches and how it's scored, and use a realistic sensor (Experiment 7 in 16.4, ~3–6 GPU-hours). Ideally test with a real El Niño, not a fake patch. | Done, with a known flaw | **Medium** — possibly the first feedback loop against El Niño-style swings |
| **9** | **Real MCB: brighten the right clouds** (low stratocumulus instead of storm clouds). | Code understood: the brightening is added to the main/storm cloud term (`shortwave_radiation.py:82`); the stratocumulus term (line 85) is untouched. The change is small, and the Q-flux ocean is now realistic enough for these clouds to form. | **Map where the model's stratocumulus forms** and compare with satellite maps; add a switch (storm / stratocumulus / both); recalibrate; re-run the key results. If it works, the MCB name can come back. | Just an idea | **High** — turns a methods paper into an MCB paper |
| **10** | **"Spread out vs. turn up":** when spraying is weak, brighten *more area* instead of just brightening harder. | A hint only (p = 0.053, just short of significant; Tier 2b). Lee et al. use this same lever. | **Start runs in different seasons** with a fairly re-tuned fixed map for each, and a controller that can move where it sprays. | A hint | **Medium** |
| **11** | **The Q-flux ocean fix for JAX-GCM** (Part 17, Step 0). | Done: ocean went from 3.8 K too cold to 0.26 K off. | **Clean it up and share it** with the model's developers (a code contribution). Too small for a paper alone. | Done | **Low as a paper, high as a contribution** |
| **12** | **Rainfall side effects (Amazon/Sahel)** (Part 12). | "No detectable harm", plus a measured rain noise of about ±3 mm/day. | **A model that can actually create far-away rain changes** (ocean currents), plus longer averages and different seasons. | Weak | **Low for now** — keep as a paragraph |

**Not research (dead ends, but keep the lessons):** the original Stage 1–5 results (broken by bugs,
Part 4), the CO2 experiment (cancels out by design; it becomes an example for #6), and PM2.5/aerosols
(the model has no particles).

## How the ingredients combine into papers

| Paper | Ingredients | One-line pitch | What has to happen first |
|---|---|---|---|
| **Paper A (main)** | **#1 + #2 + #3** (+ #4 if budget allows) | "Gradients through a coupled climate model stay useful for months if you cut the chaotic weather out, and here's what that does for designing and training controllers." | Run Experiment 1. If it fails, it's still a short paper: "a map of when you can trust gradients." |
| **Paper B (methods)** | **#5 + #6 + #7 + #8** | "How to run control experiments in a chaotic differentiable climate model without fooling yourself, with worked examples." | Mostly writing, plus Experiment 7 (the realistic sensor) to fix the El Niño flaw. |
| **Paper C (later, MCB proper)** | **#9 + #10** (+ a better ocean, + #12) | "Feedback-controlled brightening of low marine clouds in a differentiable model." | The stratocumulus switch, different seasons and ocean states. A follow-up after A and B. |

## If you only remember three things

1. **The most finished work is about *method*:** the noise floor, the rigged-experiment catalogue,
   and feedback versus fixed plans. That's Paper B, and it mostly needs writing.
2. **The most exciting work hasn't run yet:** the months-long gradients (#1). One ~14 GPU-hour
   experiment decides whether Paper A exists.
3. **Getting the MCB name back** depends on #9 (brightening the right clouds), a fairly cheap fix
   worth doing after Experiment 1.

> **New terms:** *Ingredient vs. paper* = a single result (ingredient) is rarely enough to publish;
> a paper bundles several around one clear claim. *Stratocumulus* = the low, flat, gray cloud sheets
> over cold ocean that real MCB targets (Part 1). *Adaptive controller* = a controller that estimates
> an unknown quantity (here, how strongly the spraying works) and adjusts for it; the correct name for
> what earlier parts called "PI" (16.3).

---

# Part 20 — Experiment 1's answer, and four checks made after it

*Added 2026-10-02.* Experiment 1 (Part 17) ran on the GPU on 2026-10-01 from 17:48 to 20:34. That is
**2.8 hours instead of the estimated 12**: GPU work is about 4× cheaper than every plan so far assumed.
The data and the frozen analysis are committed (`mcb_experiments_gpu/exp1_gradient_fidelity*`).

## 20.1 The registered result: outcome A′, best window 14 days

Snipping the atmosphere every 1, 7 or 14 days passed the pre-written test at 60 days; ordinary
backpropagation was "inconclusive". So the frozen rule says: go ahead with Experiments 2–3 using the
14-day snip (W* = 14). The numbers below are "how far one gradient run lands from the brute-force
truth, as a share of the truth's size", and "what fraction of the true size it reports".

| Looking ahead | Ordinary gradient | Snipped every 14 days |
|---|---|---|
| 15–30 days | excellent: each run within 4–7% | same |
| 60 days | each run off by 42–74%; only the average of 8 runs is right | each run within 14–16%; reads 86–90% of the true size |
| 120 days | **blown up**: every run 10,000–100,000× too big | each run within 30–33%; reads 69–75% of the true size |

In plain words: the ordinary "what if I brighten here?" calculation is reliable for about a month on
ocean goals (about twice Dubey et al.'s 2–4 weeks for land) and turns to nonsense by four months.
Snipping keeps it usable for at least four months, but it comes out **steadily a little too small**,
because it misses slow air feedbacks that take longer than two weeks to build. The shortfall shrinks
as the window grows (1 → 7 → 14 days). The 7-day **fade** behaved like the 14-day snip, although the
plan had expected it to be borderline.

**The land prediction could not be tested.** Brightening the ocean bands moves land temperature too
little to see against land weather (truth signal-to-noise 0.3–1.2 beyond 15 days), so the registered
"snipping should fail on land" prediction is reported as untestable.

## 20.2 Check 1 (after the results, so exploratory): a no-model guess gets the direction almost free

With only five broad latitude bands, a guess that uses no model run at all (sunlight × cloud cover ×
band area) points within 3–9° of the truth for the ocean average and the equator-to-pole contrast,
about as close as the snipped gradient (2.5–9°). The gradient clearly wins on the **north–south
contrast** (5° vs 13° at 60 days) and on the **size** of the effect. So the pass test on the ocean
average was easy, and any paper must show what the model knows beyond geometry: the size, the
contrasts, the response maps, and designs with many knobs.

Duncan's own back-of-envelope (sunlight × cloud ÷ ocean heat capacity; the slab is 40 m deep at the
equator and 60 m at the poles) points 8–20° off and predicts about **2× too much** cooling, because it
ignores the air's feedbacks. The gradient beats it on both counts (0.3–10° off; 69–99% of the size).
It is the right sanity check, and the gradient demonstrably knows more.

## 20.3 Check 2: one gradient run is worth several brute-force runs

For five knobs, one snipped-gradient run costs about the same GPU time as one round of brute-force runs
(about 35 seconds each). Its accuracy equals **5–12 brute-force rounds at 1–2 months**, but only about
**1.5–6 rounds at 4 months**, because its undercount does not average away. Brute force needs two extra
runs per knob while the gradient's cost stays flat, so the advantage grows with the number of knobs.
This is the answer to Duncan's question of whether gradients can stand in for large ensembles.

## 20.4 Check 3: the model's low clouds, and why "real MCB" is a bigger job

A one-year run of the Q-flux climate on the laptop measured both cloud types. The stratocumulus layer
sits in the right places, but it is thin: about 19% average cover off Peru (54% in its strongest cell),
16% off Namibia, 5% off California, against roughly half the sky in real decks. The current actuator's
main cloud is anti-correlated with it (pattern correlation −0.64). Brightening only the low clouds
gives about **1/12 of the cooling per unit of brightening**, so a realistic signal over 60 days would be
about the size of the weather noise. Real MCB therefore needs months-to-year runs and bigger ensembles:
a later paper (Part 21), and a natural use of the months-long gradients.

## 20.5 Check 4: what "one Earth" looks like without the twin

Every controller so far read a perfect twin run. In 9 years of one free-running model Earth, the ocean's
average temperature wanders by about 83 mK on its own, and slowly (it barely changes over a month). A
before/after check of whether brightening worked therefore has noise of **33 mK at 2 months, 49 mK at 4
months and 88 mK at a year**: 2–3× the twin-based noise of all past results, and it grows the longer you
wait. The adaptive controller's win (shrinking a 20 mK miss to 10 mK) and its estimate of the hidden
spraying strength (worth up to about 40 mK) would be hard to see with one Earth, so they need
re-testing. Because the wander is slow, it is also predictable: a model forecast of "what would have
happened without brightening" could remove much of it.

## 20.6 Other Step 0 facts

The Q-flux climate passed its check again on the GPU (bias +0.08 K, drift −0.012 K per 60 days). The 16
starting ocean states really differ (neighbouring states differ by about 1 K in their temperature
patterns) but share a slow cooling trend of 0.15 ± 0.05 K per decade; the interleaved split balances
it (evaluation minus training +0.05 K).

> **New terms:** *A′ (A-prime)* = snipping passes and ordinary backpropagation is inconclusive.
> *Undercount* = the snipped gradient's steady habit of reporting less than the true effect.
> *One Earth* = measuring with a single realization and no twin, as a real deployment must.

---

# Part 21 — The advisor meeting and the current plan

*Added 2026-10-02 (meeting with Duncan Watson-Parris that day). This part replaces the paper plan in
Part 19. Parts 17–18 remain the experiment design; the changes to them are listed in 21.3.*

## 21.1 What Duncan advised, and the verdict on each point

| His advice | Verdict | Why |
|---|---|---|
| The weather noise is interesting in itself: can you even **detect** an intervention? | Strongly agree | It is what a real deployment faces, and his own field. Prior work exists on detection times (MacMartin et al. 2019; Diamond et al. 2022); the new angle is detection inside a feedback controller with one Earth (20.5) |
| Use the model's **gradients**, Dubey-style, for better or faster control **without big ensembles** | Agree | Experiment 1 shows gradients last longer than he assumed (a month plain, four months snipped), and one gradient run stands in for 5–12 brute-force runs (20.3) |
| Treat the model as a cheap **testbed for methods**, not for trusted predictions | Agree | Changes how Paper 1 is described (21.4) |
| The methods paper should focus on **where to intervene and how much** | Agree | No existing controller chooses *where*: the classical laws and copied networks only scale one fixed map, so this is open territory |
| A simple formula (sunlight × cloud × heat capacity) should roughly give the best places | Agree, as a sanity check | Tested in 20.2: right general direction, 2× too much cooling; the gradient knows more |
| No machine learning **for its own sake** | Agree | Copying a formula cannot beat the formula. The "train the AI four ways" race and the copying student are dropped |
| Rainfall is worth including (more nonlinear) but noisy | Partly | Even land temperature was undetectable after 2 weeks with 32 runs; rainfall goes into the detectability analysis first, not into control |
| A more advanced ocean: **not now** | Agree | More complexity than it solves at this stage |
| **Write up** methods and main results now; abstract last; a 4-page Climate Change AI workshop paper as a first target | Agree | Done for the methods paper (21.5) |

Two points of nuance: in the slab ocean heat cannot move sideways through the water (only through the
air), so "after long enough it doesn't matter where you brighten" may hold less well than in the real
ocean, and is testable; and because the slab's wander is slow, "wait longer to beat the noise" helps
less than for random noise (20.5).

Things said slightly wrong in the meeting, to correct with him: the model adds no rain (it only makes
clouds more reflective, and the storm-type clouds at that); the copying AI clearly beat the fixed map
(the AI trained through the simulation did not); the classical controller does not choose regions (the
southern-hemisphere map was the old fixed map); the 8 runs fixed the noise, while the 16 starting oceans
fixed the "same ocean" problem.

## 21.2 The current paper plan

| Paper | Contents | Status | Target |
|---|---|---|---|
| **Paper 2 — methods/controllers (write-up first)** | Fixed map vs adaptive classical law vs neural controllers (trained through the model, copied, copied + fine-tuned) under hidden efficacy and a hidden tropical SST patch (Parts 8–10, 13–15); the noise floor and the six rigged-design traps as methods; then the new "one Earth" experiment (realistic sensing, patch removed from sensor and score) and **where and how much** to brighten, with a gradient-based (Dubey-style, receding-horizon) controller | Draft skeleton written (21.5); new experiments not yet run | Write-up for Duncan now; 4-page Climate Change AI workshop; then JAMES |
| **Paper 1 — gradients** | Experiment 1 (20.1), gradients vs ensembles (20.3), the formula sanity check (20.2), and gradient design of maps with **many knobs** (Experiment 2), where gradients beat brute force | Experiment 1 done; Experiment 2 not yet run | JAMES or GMD; a write-up is not needed yet |
| **Paper 3 — real MCB (later)** | Brighten the right (stratocumulus) clouds, ideally in the standard MCB regions, over months to a year, using Paper 1's method; can detection even succeed with one Earth? | Idea; 20.4 shows it needs long runs | After Papers 1–2 |

The El Niño work is folded into Paper 2 rather than written as its own paper. A short "pitfalls" piece
(noise floor + rigged designs) remains possible as a workshop paper or a section of Paper 2.

How Papers 1 and 3 differ: Paper 1 is about the **tool** (can you predict months ahead and plan with
it; which clouds get brightened doesn't matter). Paper 3 is about **MCB itself** (point that tool at
the right clouds and ask how much they cool, where to brighten, and what side effects appear).

## 21.3 Changes to the Part 17–18 experiment plan

- **Kept:** the receding-horizon planner (Part 18 steps 13–16, built 2026-10-02) and the test world with
  a normal-climate target instead of the twin (steps 10–12, built). They are exactly Duncan's
  "gradients for control, one Earth" direction.
- **Dropped or deferred:** the copying student and the four-way training race (steps 20–22), unless
  time allows; the advanced ocean; rainfall as a control target.
- **Added:** a **sunlight-guess design** as a required opponent in Experiment 2, with the many-knob design
  as a main test (with five knobs, brute force is as cheap and as accurate); the response-**maps** check
  (does the snipped gradient get effects outside the brightened bands right?); a short registered test
  of **21- and 30-day** snips reusing Experiment 1's truth (about 1 GPU-hour); a **July** mini-replicate;
  and the **one-Earth sensing** experiment for Paper 2 (twin vs before/after vs model forecast vs
  pattern-matching ("fingerprint") sensing, crossed with fixed-map scaling vs gradient planning).
- **Costs:** every GPU estimate in Part 18 can be divided by about 4 (20, opening paragraph).

## 21.4 Paper 1 in plain words (updated after Duncan's framing)

"Paper 1 is about a method, not a prediction. In a cheap, simple climate model, we show that the
model's 'what if I brighten here?' calculation can replace lots of expensive repeat runs: one
calculation does the job of about 5–12 runs when looking 1–2 months ahead. It stays usable for about
four months if we only follow the weather two weeks at a time, though it comes out a little low. We
check it against a simple sunlight-and-clouds formula and use it to find where brightening works
best. The point is that this trick could later be used in big, realistic climate models where repeat
runs are too expensive."

## 21.5 The methods write-up (Paper 2 skeleton)

Written 2026-10-02 in `~/workspace/mcb-methods-writeup/`, outside this repository on purpose (the
GitHub fork is public). It follows the lab wiki's writing resources: one-line TL;DR comments over every
paragraph, a multi-experiment structure, ACP's abstract and conclusion structure, and AGU's template
rules.
- `main.tex` (AGU JAMES layout; falls back to `agu-standin.sty` when AGU's class file is missing),
  `references.bib` (28 entries, 22 generated from Crossref records), six figures regenerated from the
  campaign pickles by `make_writeup_figures.py`, and three tables.
- Compiles cleanly (20 pages; `main_preview.pdf`). For **OpenAI Prism**, import
  `mcb_controller_writeup.zip` or the `upload/` folder (`README_UPLOAD.md`).
- About 29 visible TODO notes: author details, training settings from the logs, and questions for
  Duncan.

## 21.6 Next steps

1. Send Duncan the write-up and the Experiment 1 figure on Slack, with the corrections in 21.1; meet
   every 1–2 weeks (next booked slot: October 20).
2. Agree the answers to the open decisions (21.7).
3. Paper 2's new experiments: one-Earth sensing, then where-to-intervene with the planner.
4. Paper 1's next steps: the maps check, the 21/30-day snip test, then Experiment 2 with the
   sunlight-guess opponent and many knobs.
5. Housekeeping: report the jax-esm Q-flux month-indexing bug upstream; archive the data on Zenodo.

## 21.7 Open decisions for Duncan

1. Should the workshop paper be the controller comparison (the write-up) or the gradient story?
2. Is detectability with one Earth part of Paper 2, or its own study?
3. Should we contact Dubey et al. (same model; their next steps overlap the planner)?
4. How should the AI use be disclosed, and what is the author list?

> **New terms:** *Testbed* = a cheap model used to try out methods, not to make trusted predictions.
> *Fingerprint sensing* = checking an observed map against the expected pattern of cooling, instead of
> a single average, to pick a signal out of noise sooner.


---

# Part 22 — The pilot: the planner's settings, fixed by measurement

*Added 2026-10-05 (Part 18 step 23). Code: `run_planner_pilot.py` (stages `offline`, `ic`, `analyze`,
with decision rules R1–R8 written into its docstring before any data existed),
`run_campaign_pilot.sh` (part 1) and `run_campaign_pilot_loop.sh` (part 2). Results:
`mcb_experiments_gpu/pilot_step23/` (`pilot_offline.json`, `ic*_pilot.*`, `pilot_decisions.json`,
`loop/`, scored by `run_pilot_loop_scores.py`). Training states only (exp3_train, branch 2 of macro states 0, 2, …, 14); no evaluation state
was touched.*

**In one sentence:** the pilot shows the planner works with the cheapest settings (one Gauss–Newton
step, one weather copy), that choosing *where* to brighten beats brightening everywhere equally by about
30%, and that a planner running from day 0 needs only about a fifth to a third of the brightening cap.

*Analogy:* before a road trip you test-drive the car on a few local roads to choose the tyre pressure and
gear settings, and write down beforehand what "good enough" means, so you can't fool yourself afterwards.

## 22.1 What was run

- **Offline (laptop, from Experiment 1's maps and the training references).**
  - *Maps check:* the 14-day snipped gradient's per-band response maps match the brute-force truth up to
    the truth's own noise (slope of truth on gradient 1.0–1.1 at 60 days, 1.06–1.24 at 120 days). It
    misses part of the far-away cooling the air carries (for example −0.034 K true vs −0.012 K at 60 days
    for the bands away from the brightening). Plain backpropagation is much worse by 60 days and blows up
    by 120.
  - *Signal to noise:* with a ramp to 6 W m⁻² by day 182, the warming stands 3.3× above the noise in
    latitude averages (zonal), but only 0.27× on single grid points. So scoring uses latitude averages,
    the warming is the 6 W m⁻² ramp (rule R5), and rainfall, evaporation and land temperature (0.13–0.43×)
    are reported descriptively only (R8).
- **Part 1 (GX10, 8 states, about 2 GPU-hours per state, 3 at a time; 13:56–20:09).** Each state runs
  uncontrolled with the ramp to day 112, then re-plans once in every candidate setting: Gauss–Newton
  after 1, 2 or 3 steps; each of 3 single copies; map vs latitude-average objective; 60 vs 120-day
  look-ahead; Dubey's penalties; Adam (15 steps); the best uniform setting; and no brightening. Each
  candidate is then held for 120 days in 8 shared weather samples to measure its true objective.
- **Part 2 (GX10, about 25 minutes).** Whole 182-day episodes from day 0 with the chosen settings
  (13 re-plans every 14 days), on 2 states, 1 weather sample each, against new ramp-6 references.

## 22.2 Decisions (8 states; paired differences, SE over states)

| Rule | Setting | Result | Decision |
|---|---|---|---|
| R1 | Optimizer | Adam vs Gauss–Newton: −0.2% (0.5 SE); Adam takes about 5× longer and moved at most 1.5 logits | **Gauss–Newton** |
| R1 | Steps | 1 step vs 3: −0.6% (1.4 SE) | **1 step** |
| R2 | Copies | single copies within −0.7% to +0.04% of 3 copies; median noise-to-signal 0.16 | **1 copy** |
| R4 | Score and objective | grid points 0.27× noise, zonal 3.3×; the map-objective planner is 12% worse on the zonal score (2.8 SE) | **latitude averages for both** |
| R3 | Look-ahead | 120 vs 60 days: 26% better over 120 days (3.3 SE) but 19% **worse** over the next 60 days (3.4 SE) | **120 days** (by the rule; the trade-off is reported) |
| R7 | Penalties | Dubey's μ = 0.01 s², λ = 0.1 s² with s = 0.25 K per unit albedo (μ = 6.3×10⁻⁴, λ = 6.3×10⁻³); they cost 0.4% of the improvement | **kept** |
| R5 | Cap headroom | late start (day 112): about 60% of the cap on average, bands at the cap in every state; from day 0: 21% over the episode, 29–32% in the scored window, no band at the cap | **passes from day 0** (fails at the late start; see 22.3) |
| R6 | Scoring window | days 98–182, 5 members | as written |

Also measured (zonal score over the next 60 days):
- *Choosing where vs one uniform setting:* 31% less error (5.8 SE). This is the first direct evidence for
  Paper 2's "where to intervene" question.
- *Planning vs no brightening:* 75% less error (9.4 SE).

## 22.3 The closed loop from day 0, and the cap problem

| | ic0002 | ic0202 |
|---|---|---|
| Ocean-mean error, days 98–182: uncontrolled → controlled | +0.154 → −0.011 K | +0.125 → −0.013 K |
| Zonal objective: uncontrolled (5-member mean) → controlled (1 member) | 0.0265 → 0.0060 | 0.0208 → 0.0044 |
| Noise of a single member alone (zonal, estimated) | about 0.0035 | about 0.0041 |
| Mean share of the cap: whole episode / scored window / last re-plan | 21% / 32% / 40% | 20% / 29% / 36% |
| Error by fortnight, controlled (uncontrolled reaches +0.19–0.24 K) | between −0.040 and +0.012 K | between −0.026 and −0.002 K |

- **The cap problem was mostly an artifact of the late start.** Starting at day 112 of an uncontrolled run
  meant the ocean was already 0.14 K too warm, and brightening can only cool slowly, so the planner pushed
  bands to the cap. Steering from day 0 keeps up with the warming at roughly a third of the cap, still
  rising to about 40% by the end, because the warming keeps growing.
- **Caveat for hidden efficacy 0.5:** about 40% at nominal strength becomes about 80% at half strength by
  the end of the episode. That fits, but only just. Longer episodes or stronger warming will need a higher
  cap or a weaker ramp.
- **The controlled run cools slightly too much in the middle** (down to −0.04 K around days 70–110 in
  ic0002). The 120-day look-ahead plans for warming still to come, which matches part 1's finding that
  the long planner is a little worse over the next 60 days.
- **Reading the full-map score.** The run's own map score (0.09 controlled vs 0.06 uncontrolled) is not
  a fair comparison. One weather sample carries about 5× the grid-point noise of a 5-member mean (single
  member alone ≈ 0.03–0.07). Scored runs must use matched members, as the fairness rule of step 11 says.
- **Cost:** a re-plan takes about 90 s with these settings, and a whole episode about 25 minutes per state
  with two side by side. That is about 3× cheaper than 3 steps × 3 copies at the same 120-day look-ahead
  (4.3 minutes per re-plan in step 17).

## 22.4 What this means for the plan

- **Revision 1's planner settings are now fixed:** Gauss–Newton, 1 step, 1 copy, 14-day snip, 120-day
  look-ahead, latitude-average objective, μ = 6.3×10⁻⁴ and λ = 6.3×10⁻³. Warming is a 6 W m⁻² ramp over
  182 days, scored on days 98–182 with 5 members. The 60-day planner stays as a comparison, because of
  the near-term trade-off.
- **Code still to do:** `jcm/mcb/planner.py` has no latitude-average objective yet (part 2 used the map
  objective), so add it before step 24. It is a small change, because the projection is linear
  (`zonal_project` in `run_planner_pilot.py`).
- **Budget:** at about 25 minutes per planned episode with one sample (two running side by side),
  Experiment 3a's 3 planners × 24 evaluation states come to roughly 15 GPU-hours. That is below step 17's
  estimate of about 25, and inside the lean budget.

> **New terms:** *Pilot* = a small, cheap rehearsal that fixes settings before the real experiment.
> *Closed loop* = the planner steers the whole run, re-planning as it goes, instead of planning once.
> *Headroom* = how much of the allowed brightening is left unused, in case spraying works less well than
> expected.

---

# Part 23 — Experiment 3a: the planner meets its opponents

*Added 2026-10-07 (Part 18 step 28). Frozen as Amendment 9 revision 1 at commit `5e11b082` and
registered on OSF before any evaluation data existed (https://osf.io/7bwe4/, 2026-10-06 11:52 PDT).
Code: `run_campaign_exp3a_train.sh`, `run_campaign_exp3a_eval.sh`, `analyze_experiment3a.py`. Results:
`mcb_experiments_gpu/exp3a/` (`train/`, `eval/` run summaries and logs, `exp3a_analysis.json`). The
evaluation used the 24 held-out states (macro states 1, 3, …, 15 × branches 2, 3, 4), each with 3
matched weather members.*

**In one sentence:** every sensible controller removes about four-fifths of the warming's latitude
pattern. The long-range planner, our main candidate, does **not** beat the simple opponents. It points
the wrong way against the 14-day planner and the classical feedback controller, though no comparison
against them clears the pre-registered bar. It clearly beats holding its own average pattern fixed.

*Analogy:* we built a chess engine that looks 120 moves ahead and entered it in a tournament. It beat
almost everyone in the hall. But a player who looks only a few moves ahead, and an old rule-of-thumb
player, each did at least as well, because the far-sighted engine kept over-correcting for threats that
had not arrived yet.

## 23.1 What was run

- **Training side (2026-10-05 20:55 to 2026-10-06 00:15, training states only).**
  - References for the 16 training states.
  - One-band response runs (+0.1 in each band) for the fixed designs and the feedback controllers.
  - 8 runs of the 120-day planner.
  - The ladder of four fixed designs, frozen into `train/ladder.json` (Part 18 step 19).
  - *Already visible before the freeze:* the planner overshot slightly on every training state (−0.034 K),
    and the linear model rated the best fixed pattern highly. Both were written into the registration.
- **Evaluation (2026-10-06 12:22 to 2026-10-07 05:27, GX10).**
  - References for the 24 states took 80 minutes.
  - Then 240 episodes (10 arms × 24 states), each 182 days of 13 fourteen-day segments with 3 members.
  - None failed. A 120-day planner episode took about 80 minutes; the whole campaign took about 17 hours.
- **Analysis (2026-10-07).** The registered script stopped on a file-format mismatch: fixed-pattern arms
  store `amplitudes`, not `schedules`. It was fixed and logged as **Deviation D1** before any output
  existed. The fix touches only two descriptive cap-use numbers, never the score or the tests.
  - A first attempt was run by hand without the campaign's memory settings. JAX grabbed 75% of the
    GX10's shared memory, and the machine ran out and rebooted. Rerun on the CPU, the analysis takes
    under a minute.

## 23.2 Every arm (24 states; scored on days 98–182)

| Arm | Zonal score `J_zonal` | vs no brightening | Ocean bias | Mean cap use |
|---|---|---|---|---|
| `plan60` (60-day look-ahead) | 0.00329 | −84% | −0.013 K | 28% |
| `plan14` (14-day look-ahead) | 0.00335 | −84% | −0.008 K | 27% |
| `pi` (GLENS-style feedback) | 0.00366 | −82% | −0.001 K | 29% |
| `linear_response` (best fixed pattern) | 0.00407 | −81% | +0.011 K | 12% |
| `adaptive` (fixed pattern, scaled by feedback) | 0.00430 | −79% | +0.001 K | 26% |
| `uniform_cancel` (one setting everywhere) | 0.00487 | −77% | +0.004 K | 12% |
| **`plan120` (PRIMARY)** | **0.00492** | **−77%** | **−0.031 K** | 28% |
| `planner_average` (plan120's mean pattern, fixed) | 0.01125 | −46% | −0.068 K | 19% |
| `uncontrolled` | 0.02096 | — | +0.131 K | 0% |
| `uniform_effort` (one setting at the planners' effort) | 0.02919 | +39% | −0.139 K | 26% |

- Every arm except the last two removes 77–84% of the zonal error (all p < 0.01 against no brightening).
- On single grid points the gains are small (G = 0.8–1.3). Three weather members leave grid-point
  noise dominant, which is why the registration scores latitude averages (Part 22, rule R4).
- **The longer the look-ahead, the more the planner overcools:** −0.008 K at 14 days, −0.013 K at 60 days,
  −0.031 K at 120 days.

## 23.3 The registered hypotheses (Holm family of four; macro states as the unit)

| | `plan120` against | Change in score | 90% interval | p (raw / Holm) | Verdict |
|---|---|---|---|---|---|
| H1 | `plan14` | **+47%** (worse) | +8% to +99% | 0.037 / 0.11 | inconclusive |
| H2 | `linear_response` | +21% (worse) | −7% to +64% | 0.21 / 0.21 | inconclusive |
| H3 | `pi` | **+35%** (worse) | +1% to +86% | 0.050 / 0.11 | inconclusive |
| H4 | `planner_average` | **−56%** (better) | −65% to −43% | 0.0013 / 0.0051 | **better** |

- The Wilcoxon test agrees with the t test everywhere except H1, where it gives p = 0.055 against the
  t test's 0.037. Neither survives Holm, so the disagreement does not change the verdict.
- *Read honestly:* H1–H3 are "inconclusive" under the registered grid, but all three point estimates go
  against the primary planner. For H1 and H3 the 90% intervals lie entirely on the "worse" side. With
  only 8 independent macro states, the data cannot call these "worse" at the registered bar, and we
  do not.

**Secondary comparisons** (unadjusted, as registered):

| Comparison | Change | p | Reading |
|---|---|---|---|
| `plan60` vs `plan120` | −33% | 0.011 | the 60-day planner is better |
| `plan60` vs `plan14` | −2% | 0.91 | indistinguishable |
| `linear_response` vs `uniform_cancel` | −16% | 0.11 | choosing *where* helps a fixed design somewhat, not significantly |
| `linear_response` vs `pi` | +11% | 0.48 | indistinguishable |
| `adaptive` vs `pi` | +17% | 0.15 | indistinguishable |
| `planner_average` vs `uniform_effort` | −61% | < 0.001 | the planner's pattern beats spreading the same effort evenly |

## 23.4 What it means (following the interpretation fixed in advance)

- **H1 (look 120 days ahead vs 14):** not better. The registered "long planning beats two-week planning"
  claim is **not supported**. The data lean the other way, and the cheaper planners (14 or 60 days) are
  the ones to recommend.
- **H2 and H4:** H4 is better, but H2 is not, so the registered claim "changing where and how much
  over time beats the best fixed pattern" is **not established**. Changing the pattern over time clearly
  beats holding the planner's own average. Part of that comes from timing: a fixed average applied
  from day 0 overcools early (−0.068 K), because the warming starts small and grows. The best fixed
  pattern, sized for the scored window, avoids this.
- **H3 (planner vs classical feedback):** not better. By the registered reading, *in a perfect model,
  feedback on three indices is enough*. The planner's case therefore moves to **Experiment 3b**
  (uncertainty about how well brightening works) and to targets richer than three indices.
- **Why the 120-day planner overshoots (post hoc; not a registered finding).** Two causes fit the data.
  1. Experiment 1 showed that the snipped gradient undercounts the true response more the further it
     looks: about 0.86 at 60 days and 0.7 at 120 days. A single Gauss–Newton step on an undercounted
     sensitivity overshoots by roughly the inverse, about 1.4× at 120 days.
  2. A long look-ahead also brightens early for warming that is still to come (Part 22.3 saw the same
     mid-run overcooling).

  Bias grows with look-ahead (−0.008, −0.013, −0.031 K), consistent with both. A damped step (scaling
  by the measured undercount) is an obvious fix. It would be a new hypothesis for a new registration,
  not a reanalysis of this one.

## 23.5 What this changes

- **The controller/methods paper (Part 21)** now has a clean, pre-registered result:
  - gradient-based receding-horizon planning works, and removes about 80% of the warming pattern;
  - in a perfect model, a short look-ahead and the field's standard feedback controller do as well or
    better;
  - the 120-day look-ahead adds overshoot, not skill.

  That is a useful, honest finding for Dubey et al.'s line of work. It also directly answers Duncan's
  question of whether gradients are needed for control.
- **The gradient paper** is unaffected: Experiment 1 measured gradient accuracy, not control value.
  Part 23 adds a cautionary example of what a 30% gradient undercount does inside a controller.
- **Next experiment:** Experiment 3b (hidden efficacy, imperfect model), where planning with a model
  might earn its keep. Freeze it in revision 2 before any of its data, as before. Optionally, register
  a damped-step planner as a new arm there.

> **New terms:** *Holm correction* = a way of raising the bar when testing several hypotheses at once,
> so luck across four tries does not look like a discovery. *Inconclusive* = the data were not strong
> enough to call it either way at the bar set in advance. *Overshoot* = correcting too far, past the
> target.

---

# Part 24 — Experiment 3b: when nobody knows how strong the spraying is (plan)

*Added 2026-10-07, before any Experiment 3b run. This is the working plan. The registered version
will be Amendment 9 revision 2, frozen and posted on OSF before any evaluation reference or run
exists, as revision 1 was.*

**In one sentence:** 3b hides how strongly brightening works — overall and band by band — from every
controller. It then asks whether a planner that *learns* the strength from what it sees, using the
model's own gradients, beats the field's standard feedback controller.

*Analogy:* a house where the radiators' labels are wrong. Some days the heating is twice as strong as
the label says, some days half, and the radiators in some rooms work better than in others. A standard
thermostat (the classical controller) just keeps nudging until each room is warm enough. A "smart"
thermostat (the learning planner) watches how each room actually responds, corrects its own notes about
each radiator, and then plans. The question is whether being smart pays, and by how much compared with
a thermostat that already knows the true strengths (the oracle).

## 24.1 Why this experiment, now

- **3a's answer:** with a perfect model, a short-sighted planner and the classical controller do
  equally well. By the interpretation fixed in revision 1, the planner's case moves to uncertainty.
- **The real world's biggest unknown is how strong the spraying is.**
  - Hirasawa et al. (GeoMIP G6-1.5K-MCB) found the sea-salt emissions needed for the same cooling
    differ about 20-fold across three climate models.
  - Regional susceptibility differs too: some cloud regions brighten much more than others.
- **Duncan's priorities fit exactly.** 3b covers:
  - gradients for control without ensembles (the learner uses one gradient run);
  - where and how much to brighten;
  - detection with one Earth (how fast can the strength be learned from one noisy realization?);
  - no machine learning for its own sake (no neural networks).
- **The field-standard opponent** is the GLENS-style PI controller, the same family as Lee et al.
  (2025, GRL), the first feedback-regulated MCB simulations in CESM2.

## 24.2 What went wrong and right before, and what 3b does about it

| Lesson (where it came from) | What 3b does |
|---|---|
| **Too little power.** 3a's planners differed by 35–47% and still failed Holm. Weather noise between branches (about ±0.0025 in the score) swamped the differences between controllers. Measured 2026-10-07: weather noise alone would give the best controllers' score, and 91% of it sits in the latitude pattern | More weather members per run (chosen by a registered power calculation from the pilot, 6–10 instead of 3); 10-member reference targets instead of 5; 16 independent ocean states instead of 8; one primary hypothesis instead of a Holm family of four |
| **The evaluation states are no longer fresh.** 3a's results and its noise analysis used the 24 evaluation states | 32 brand-new ocean states from a continuation of the same control run, interleaved: 16 for training, 16 for evaluation |
| **Pilots that judge the wrong thing.** The step-23 pilot picked 120 days from open-loop re-plans, and closed-loop results reversed it | Every pilot judgement uses whole closed-loop episodes, the real task |
| **Picking settings from a noisy score** (Stage-1, holistic review) | Defaults come from rules (the learner's noise level is measured, not tuned). A pilot setting replaces a default only if it is better by more than 2 standard errors, with the same rule for our method and the opponent |
| **Rigged designs** (Parts 7, 13, 16: a design that hands one side the answer) | Every arm gets the same hidden strengths and the same weather. The opponent may retune for robustness under the same rule. The strengths are drawn from a registered distribution justified by the literature, not chosen after seeing results. The learner can never read the true strength (a test checks this) |
| **Opponents must be strong.** | PI with feedforward and integral action (the field standard), a classical learning controller (the Tier 2 adaptive law), and the best fixed pattern |
| **The analysis broke at launch** (3a's `KeyError`) | The smoke test runs every arm and then the registered analysis on the smoke outputs, end to end |
| **The GX10 crashed** (JAX grabbed 75% of memory) and **connections dropped** | Memory limits set in every script; everything runs in tmux and can resume; checks are short |
| **What went right, kept:** pre-registration with OSF timestamps and checksums; interleaved training/evaluation states; matched weather members; macro states as the unit; hierarchical bootstrap; interpretations fixed in advance; training-side results written down before evaluation; the cheap Gauss–Newton planner (1 step, 1 copy, zonal) | Unchanged |

## 24.3 The design

- **Test world:** as in 3a. The warming ramps to 6 W m⁻² by day 182, there are 13 re-plans every 14
  days, and scoring covers days 98–182 on the latitude profile (`J_zonal`).
- **Hidden strength** (drawn per state from registered seeds; seen only by the simulated world).
  Each band's strength is `e_k = g × r_k`:
  - *Overall factor* `g` = 0.5, 1 or 2. Each evaluation ocean state has three weather branches, one at
    each value, so every ocean state sees the whole range. A factor of 2 either way is deliberately
    milder than the 20-fold spread across models, and keeps most runs inside the brightening cap.
  - *Regional factors* `r_k`: log-normal with standard deviation 0.5, re-centred to a geometric
    mean of 1. The strongest and weakest of five bands then typically differ about 3-fold.
- **Arms** (every planner is 3a's 14-day planner: Gauss–Newton, 1 step, 1 copy, zonal objective, the
  same penalties):

| Arm | What it knows |
|---|---|
| `plan_learn` (**new**) | starts at nominal strength and learns each band's strength from its own forecasts versus what happened, using the Jacobian it already computes. No extra model runs |
| `plan_naive` | assumes nominal strength throughout |
| `plan_oracle` | knows the true strengths (the ceiling) |
| `pi` | GLENS-style PI on T0–T2 with feedforward, sensitivities from training runs at nominal strength |
| `adaptive` | the Tier 2 law: learns one overall strength and scales the best fixed pattern |
| `fixed` | the best fixed pattern, designed on training states at nominal strength (no feedback) |
| `uncontrolled` | no brightening (for the gains) |

- **How the learner works.** Every planner already predicts the next two weeks and knows how its
  prediction would change with each band's strength (the Jacobian it uses to plan). After each
  fortnight it compares the prediction with what happened, and updates each band's strength by
  regularized least squares over all fortnights so far. The prior is centred on nominal strength with
  standard deviation 1. The noise level is measured from the oracle's forecast errors in the pilot,
  by a fixed rule.
- **Hypotheses:**
  - **H1 (primary):** `plan_learn` vs `pi` on `J_zonal`. Paired t test on 16 macro-state means,
    two-sided, α = 0.05.
  - **Secondary family (Holm):** H2 `plan_learn` vs `plan_naive` (does learning help?), and H3 `pi` vs
    `fixed` (is feedback essential under uncertainty?).
  - **Descriptive:**
    - each arm vs the oracle (the cost of not knowing);
    - every result by overall factor;
    - the learning curves (how many fortnights until the strength is known: detection with one Earth);
    - the bias and pattern parts separately.

## 24.4 Steps and cost

| # | Step | GPU (wall) |
|---|---|---|
| 1 | Extend the state generator (continue the control run from macro state 15 every 365 days; roles `exp3b_train`/`exp3b_eval`) and generate 32 new macro states with their weather branches | ~30 min |
| 2 | Code, while step 1 runs: hidden strength per band; the learning planner; the strength draws; saving each member's latitude profile (so the pilot can measure how noise falls with members); the 3b analysis; campaign scripts with a full smoke test | — |
| 3 | Training side: references, one-band response runs and the fixed design on the 16 training states | ~2.5 h |
| 4 | Pilot (training states only, whole episodes): (A) oracle runs, to measure the forecast noise for the learner; (B) every arm under the hidden strengths, plus two robustness variants (PI with a slower loop; the learner with 4× the noise), and a global-only learner as a diagnostic | ~4.5 h |
| 5 | Power calculation from the pilot, then the number of members (6, 8 or 10); write and freeze revision 2; post it on OSF | — |
| 6 | Evaluation: 10-member references for 48 states, then 7 arms × 48 states; the registered analysis runs automatically | ~17–27 h |
| 7 | Report Part 24 results, memory, commit and push, and a plain-language explanation | — |

**Possible outcomes, and what each would mean for the papers:**
- *Learning beats PI:* the headline of Paper 2. A differentiable model lets a controller learn where
  and how strongly brightening works, and that beats the field's standard.
- *Learning ties PI:* with 3a, a clean and well-powered message. Classical feedback handles strength
  uncertainty too, so the planner's value must come from elsewhere (many knobs, Experiment 2, or
  targets beyond three indices).
- *Either way:*
  - fixed designs fail under realistic uncertainty, which replicates Tier 2 over a far wider range;
  - the learning curves measure how fast one Earth reveals the spraying's strength (Duncan's
    detection question).
