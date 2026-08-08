# The MCB Project: A Plain-Language Report

*A beginner-friendly account of the whole effort — what we set out to do, what we built, what worked, what didn't, and what we honestly know now. No prior background assumed. Every technical term is explained the first time it appears.*

*Companion to the detailed engineering log in `MCB_IMPLEMENTATION_PLAN.md` and the independent validation report `MCB_META_AUDIT.md`. Written 2026-07-28; updated 2026-08-03 to cover the second audit and the Tier-1, Tier-2, and Tier-2b campaigns (Parts 7–11); updated 2026-08-07 with the rainfall side-effect analysis (Part 12).*

---

## The one-paragraph version

We tried to teach a small AI to fight global warming inside a computer simulation of Earth's climate. The specific idea is **Marine Cloud Brightening (MCB)**: making low ocean clouds slightly whiter so they reflect more sunlight and cool the planet a little. We built the AI as a "controller" that watches the simulated climate and decides where and how much to brighten clouds, aiming to cool the ocean by a small target amount without wrecking rainfall in sensitive places like the Amazon. Along the way we discovered — twice — that our own results couldn't be trusted: an earlier version of the work was riddled with bugs and wishful statistics, and even the careful rebuild turned out to be measuring a 5-thousandths-of-a-degree signal with an instrument that jittered by 15, in an experiment accidentally designed so the AI could never win. So we tore it down twice, fixed the measurement, and finally gave the controller a problem worth solving: uncertainty about how strongly the cloud-seeding actually works. **The honest ending, in three acts:** the corrected method *does* find a good, on-target cloud-brightening plan (now confirmed with real statistical power); under realistic uncertainty a feedback controller *demonstrably* beats the fixed plan — our strongest statistical result, and finally the project's founding claim; but the winning neural controller got every bit of its skill by *imitating a hundred-year-old control-engineering formula* — training it through the climate simulation itself, the project's founding bet, never worked, and we measured exactly why. That's a real, defensible, and genuinely interesting scientific ending — just not the one we set out to find.

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

## The four significant results

1. **Confirmatory positive (Tier 1):** the gradient-optimized static pattern hits the target on fresh climates, −0.1023 ± 0.0014 K.
2. **Significant bounded negative (Tier 1):** with nothing to correct, feedback of any kind is worth less than 4.5 mK — and the design analysis explains why (a ~6 mK ceiling).
3. **Classical feedback halves the efficacy-uncertainty error (Tier 2, replicated 3×):** 20 → 10 mK, p = 0.0006 — while directly-trained neural controllers fail to realize the same gain (a measured training failure, not an information limit).
4. **An imitation-initialized neural controller significantly beats static deployment (Tier 2b):** −11.6 mK, p = 0.00025 — with fine-tuning contributing provably nothing beyond the imitation.

## The one-sentence headline

**A neural network controller *can* demonstrably steer a climate intervention under realistic uncertainty — but in this project every bit of that ability came from imitating hundred-year-old control theory, and none of it from gradient training through the climate model itself.** The differentiable model's proven value is *design* (finding the spatial pattern); its value for *training controllers* through long chaotic rollouts is, on this evidence, bounded near zero — and we can say exactly why.

## What's still open (honestly)

- **Can learning ever exceed the classical law?** The low-efficacy "recruit more area" trend (p = 0.053) is the natural next experiment — it needs either more test climates or a design that isolates the cap-limited episodes.
- **The idealized observer.** All feedback controllers here read a noiseless measurement of the realized cooling, computed against a paired counterfactual baseline — something no real deployment could have. Adding observation noise is the external-validity test still to run.
- **Scope.** Everything holds for one season, one ocean state, a slab ocean with no currents, a 60-day horizon, an in-model forcing at or beyond published MCB feasibility, and with the model's stratocumulus-analog cloud deck unperturbed. These are statements about control and optimization in a differentiable climate model — not deployment guidance for real MCB.

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

## Where to look next in the codebase

- **`MCB_META_AUDIT.md`** — the second audit, and (in its addenda) the full Tier-1/2/2b campaign numbers behind Parts 7–10.
- **`MCB_IMPLEMENTATION_PLAN.md`** — the detailed engineering log of the original effort: the first audit (R1–R7), the rebuild, and campaigns v1/v2/v3.
- **`PREREGISTRATION.md`** — the frozen rules, now with the formal amendment log (Amendments 1–4) covering every Tier-1/2/2b design decision.
- **`jcm/mcb/gates_stats.py`** — the paired t / Wilcoxon / equivalence (TOST) statistics behind every verdict, in pure NumPy with its own test suite.
- **`run_confirmatory_eval.py`** — the micro-ensemble evaluation harness (and the PI controller) used by all three campaigns.
- **`run_noise_floor.py --cross-process`** — the corrected chaos-noise measurement (Part 7.2).
- **`run_pi_imitation.py`** — the Tier-2b distillation of the PI law, including the feature-anchoring fix.
- **`run_campaign_confirm.sh` / `run_campaign_tier2.sh` / `run_campaign_tier2b.sh`** — the gated, resumable campaign scripts exactly as run.
- **`analyze_tier2.py` / `analyze_tier2b.py`** — the primary analyses, written and frozen before the data existed.
- **`analyze_precip_sideeffects.py`** — the Part-12 rainfall side-effect analysis (Amendment 5; meta-audit Addendum 4).
- **`jcm/physics/speedy/shortwave_radiation.py`** — where MCB correctly brightens cloud albedo (the R6 fix).
