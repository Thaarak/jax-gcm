# Pre-registration: long-horizon gradients in a coupled climate model (Amendment 9, revisions 0.1-0.4)

This file is the OSF posting of Amendment 9 of this project's pre-registration (`PREREGISTRATION.md`).
Everything below the horizontal line is the registered text exactly as frozen in the public
repository; nothing in it has been edited.

## Frozen source

- Repository: https://github.com/Thaarak/jax-gcm (branch `fable-version`)
- Commit: `512c85dc072c23b30cb896549f96decb87728314`, committed 2026-10-01 16:28:44 PDT
- SHA-256 of `PREREGISTRATION.md` at that commit:
  `413bdf1e61ae2d9b66d298d7d50cddb25cfa3862fb79173ef4f345085b603e26`
- SHA-256 of the Amendment 9 section reproduced below (line 745 to the end of that file):
  `014ea91a6ff5eccbc25bafe6531ca248407dd5497c5479104eaec3fe92bdcfa5`
- The same commit freezes the code the amendment names:
  - `run_generate_macro_ics.py` (Step 0);
  - `run_gradient_fidelity.py` and `analyze_gradient_fidelity.py` (Experiment 1 and its registered
    analysis, with every threshold);
  - `jcm/mcb/gradient_truncation.py`, `jcm/mcb/gradient_fidelity.py` and `jcm/mcb/band_basis.py`;
  - `jcm/mcb/qflux.py` and `run_qflux_base_climate.py` (the base climate);
  - `run_campaign_step0_exp1.sh` (the GPU run).

## Timeline, and one disclosed deviation

- 2026-09-29: Amendment 9 frozen for Step 0 and Experiment 1; revision 0.1 added.
- 2026-09-30: revisions 0.2 and 0.3 (the Q-flux base climate) and their results. These runs only set
  up the model's base climate and produce no Step-0 or Experiment-1 data.
- 2026-10-01, about 16:30 PDT: revision 0.4 committed, and the commit pushed to the public repository.
- 2026-10-01, 16:45 PDT: the GPU campaign started: the Q-flux re-settle with its registered check,
  then the Step-0 starting states.
- 2026-10-01, 17:48 PDT: Experiment 1 started.
- 2026-10-01, 18:42 PDT: this posting.

**Deviation.** Amendment 9 says it is posted to OSF before Experiment 1 runs. This posting came after
Experiment 1 had started.

**Timestamps.** The registered text and code were public on GitHub, in the commit above, before the
campaign began.

**What had been viewed before this posting:**
- the GPU re-settle's registered check (it passed) and the Step-0 diagnostics, both reported and not
  gated;
- for the first Experiment-1 starting state only, revision 0.4's logged forward-versus-backward check,
  and the size of the registered gradient estimates.

No brute-force truth, metric, verdict or outcome-grid quantity had been computed or viewed.

---

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
