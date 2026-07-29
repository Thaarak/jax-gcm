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
