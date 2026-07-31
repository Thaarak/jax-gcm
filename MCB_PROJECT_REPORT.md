# The MCB Project: A Plain-Language Report

*A beginner-friendly account of the whole effort — what we set out to do, what we built, what worked, what didn't, and what we honestly know now. No prior background assumed. Every technical term is explained the first time it appears.*

*Companion to the detailed engineering log in `MCB_IMPLEMENTATION_PLAN.md`. Written 2026-07-28.*

---

## The one-paragraph version

We tried to teach a small AI to fight global warming inside a computer simulation of Earth's climate. The specific idea is **Marine Cloud Brightening (MCB)**: making low ocean clouds slightly whiter so they reflect more sunlight and cool the planet a little. We built the AI as a "controller" that watches the simulated climate and decides where and how much to brighten clouds, aiming to cool the ocean by a small target amount without wrecking rainfall in sensitive places like the Amazon. Along the way we discovered that an earlier version of this work was riddled with bugs and wishful statistics, so we tore it down, audited it, rebuilt it correctly, and ran a careful series of experiments. **The honest ending:** the corrected method *does* find a good, on-target cloud-brightening plan — but the "smart adaptive AI" part turned out to add nothing you can statistically prove over a simple fixed plan. That's a real, defensible scientific result, even though it's not the exciting one we hoped for.

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

- **The noise floor.** Before you can claim a result is real, you must know how big the model's *random* run-to-run wiggle is. So we ran the *same fixed* plan many times and measured the spread. This "noise floor" turned out to be about 0.014 K. Any claimed effect smaller than roughly twice that is indistinguishable from luck. Measuring this was never done before — it's what turns "it looks like it worked" into "it's bigger than noise."

- **Held-out selection.** When you pick your "best" model, you must judge it on **held-out** data — climates it was *not* trained on — not on the training data itself. Otherwise you're just picking whatever memorized the training set best. (The old code picked the best on training data, which is R2.)

- **Pre-registration and control-relative gates.** We wrote down the exact pass/fail rules **before** seeing any results (in a frozen file, `PREREGISTRATION.md`), so we couldn't unconsciously move the goalposts. And every test is **paired and control-relative**: we compare the AI head-to-head against a simple baseline on the *same* climates, and require the difference to beat a **2× standard-error** bar (a standard statistical significance threshold) to count. If the difference is within that bar, the verdict is **"underpowered"** — meaning "too small to call either way," which is honest rather than a false PASS or FAIL.

> **Two more terms:** A **standard error** measures how uncertain an average is; requiring an effect to exceed *2 standard errors* is a common bar for "probably not just chance." An **ablation** is an experiment where you deliberately remove one ingredient (e.g., take away the AI's ability to see the climate state) to test how much that ingredient actually mattered.

---

# Part 6 — The Three Campaigns

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

At the raw-loss level the fixed training genuinely improved things — it roughly *halved* the error on unseen climates (from 0.020 K to 0.011 K). But that improvement is **below the significance bar** on the pre-registered gate at our sample size of 10 test cases. It's real but too small to prove.

---

# Part 7 — The Honest Bottom Line

## What genuinely worked

- **The differentiable climate model is real and valuable.** You can compute gradients all the way through a coupled atmosphere–ocean–land simulation and use them to optimize. This is the novel engineering achievement, and it survived every round of scrutiny.
- **Differentiable optimization finds a good, physically sensible MCB plan.** Once the physics was fixed, the optimizer put the brightening exactly where real MCB belongs — the subtropical stratocumulus cloud decks — and hit the cooling target on the nose (−0.097 K vs. a −0.1 K goal), on climates it had never seen.

## What didn't

- **The headline idea — that an adaptive AI feedback controller beats a simple fixed plan — was never demonstrated.** After correcting all the bugs, the AI is *on par with* a static pattern and *indistinguishable from* a fixed open-loop schedule. The "look at the state and react" ability we were most excited about adds nothing we can statistically prove.
- **Every earlier "success" (Stages 1–5) was an artifact** of the seven bugs — broken averages, un-converged training, the wrong physics, a dead land model, and statistics run on noise.

## What's still open (honestly)

The final feedback question isn't proven *negative* — it's **underpowered**. The possible benefit (~0.005 K) is about the same size as our measurement uncertainty (~0.004–0.006 K). Resolving it would need more test climates and more random seeds, not more code. There's also one small unresolved engineering item (a numerical-precision detail called the "x64 dtype" fix) that doesn't affect any conclusion.

## The scientific lesson

This is what good science looks like when it's working: a chain of exciting-looking results was checked rigorously, most of it dissolved into noise and bugs, and what remained was a smaller but *trustworthy* truth. The corrected result — *differentiable optimization yields a good static cloud-brightening plan, but adaptive neural feedback adds nothing you can demonstrate over a fixed plan* — is a genuine, publishable, defensible finding. It's less flashy than "AI controls the climate," but unlike the original claims, it's actually true.

---

## Where to look next in the codebase

- **`MCB_IMPLEMENTATION_PLAN.md`** — the detailed engineering log: the full audit (R1–R7), the rebuild, and the campaign-by-campaign numbers (v1/v2/v3 sections at the top).
- **`PREREGISTRATION.md`** — the frozen, pre-committed rules for the final experiments.
- **`jcm/mcb/`** — the controller, the loss, the coupled training loop, and the pre-registered gates (`gates.py`).
- **`run_gradient_probe.py`** — the v3 diagnostic that showed v2's negative was a training artifact.
- **`jcm/physics/speedy/shortwave_radiation.py`** — where MCB now correctly brightens cloud albedo (the R6 fix).
