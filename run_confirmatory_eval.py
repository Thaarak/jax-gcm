#!/usr/bin/env python
"""Confirmatory micro-ensemble evaluation (2026-07-29 meta-audit, Tier 1).

Evaluates named policy arms on a FRESH IC set with k-member micro-ensembles
per (arm, IC) and reports the pre-registered gates on the registered metric
(final-10-day time-mean dSST), with formal paired t / Wilcoxon p-values and
TOST equivalence bounds.

Why this exists (see MCB_META_AUDIT.md):
  * Per-run chaos noise is ~0.014-0.017 K per 60-day dSST measurement —
    larger than the ~0.005 K effects under test — and the old noise floor
    (sigma_compile=0) could not see it. Averaging k perturbed members per
    (arm, IC) shrinks that component by sqrt(k): k=8 turns the n=10 paired
    detection floor from ~12 mK into ~4-5 mK.
  * Every member baseline is REGENERATED IN THIS PROCESS (never a cached
    pickle from another program), closing the prereg "never a cached
    baseline" violation with measured teeth.
  * The registered metric is the final-10-day time-mean dSST
    (PREREGISTRATION.md section 3), not the day-60 snapshot.
  * The old 10 held-out ICs are statistically exhausted (13 gate tests +
    ~123 selection looks); point --ic-dir at a freshly generated set
    (run_generate_ics_independent.py with a new --seed0).

Arms are given as repeatable --arm NAME=KIND=PATH with KIND one of:
  static    : Stage-1 optimized pattern pickle (bias-trick constant policy)
  fc13      : MLP checkpoint using the 13-feature config (absolute SST on)
  time-only : MLP checkpoint using the time-only feature config (open loop)
  pi        : Stage-1 pattern + hand-designed deadbeat efficacy compensator
              (the Kravitz/MacMartin-style literature baseline; no training)

Tier-2 efficacy experiment (--efficacy-mode randomized): each IC draws an
unobserved eta ~ U[range]; applied forcing = eta x command. Static/open-loop
arms structurally cannot compensate; feedback arms (fc13, pi) can read the
realized cooling from the paired-anomaly features and adjust.

Example (diya):
    python run_confirmatory_eval.py \
        --ic-dir mcb_experiments_gpu/ics_confirm \
        --arm static=static=mcb_experiments_gpu/stage1_v2/stage1_optimized_pattern.pkl \
        --arm retrain=fc13=mcb_experiments_gpu/retrain_v3/stage5_trained_policy.pkl \
        --arm openloop=time-only=mcb_experiments_gpu/ablation_openloop_v3/stage5_trained_policy.pkl \
        --members 8 --days 60 --control-interval 15 --max-perturbation 0.09 \
        --comparator static --output mcb_experiments_gpu/confirmatory_eval.pkl
"""

import argparse
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np

from jcm.mcb import (
    CoupledControllerConfig,
    CoupledFeatureConfig,
    MCBPolicyMLP,
    get_coupled_feature_dim,
)
from jcm.mcb.coupled_controller import (
    create_coupled_step_fn,
    evaluate_coupled_policy,
)
from jcm.mcb.coupled_features import compute_baseline_trajectory
from jcm.mcb.coupled_loss import region_precip_change_mm_day
from jcm.mcb.enso import (
    EnsoConfig,
    box_mean_weights,
    make_enso_ff_mean_policy_fn,
    make_enso_pi_policy_fn,
    nino_pattern,
    tail_reference_scale,
    wrap_step_fn_with_enso,
)
from jcm.mcb.coupled_train import ocean_mask_from_coupler
from jcm.mcb.gates import cooling_gate, improvement_gate
from jcm.mcb.gates_stats import paired_report
from jcm.mcb.state_features import compute_area_weights
from jcm.mcb.train import load_checkpoint

from run_coupled_training import (
    coupler_workflow,
    setup_coupled_model,
    warm_start_params,
)
from run_generate_ics_independent import perturb_sst
from run_stage5_training import START_DATE, load_ics

WORKFLOW = ["coupling", "atm", "ocn"]

FEATURE_CONFIGS = {
    "static": CoupledFeatureConfig(),  # fc11; features unused (zero kernel)
    "fc13": CoupledFeatureConfig(include_absolute_sst=True),
    "time-only": CoupledFeatureConfig(
        include_sst=False, include_sst_regions=False, include_heat_flux=False,
        include_atm_temperature=False, include_precipitation=False,
        include_time=True, include_absolute_sst=False,
    ),
    # Hand-designed ratio/deadbeat controller on the static pattern (the
    # Kravitz/MacMartin-style literature baseline for the Tier-2 efficacy
    # experiment). Uses the fc11 layout: feature 0 = realized global-mean
    # ocean dSST vs the paired baseline, feature 10 = time fraction.
    "pi": CoupledFeatureConfig(),
    # Constant uniform ocean field at a given level (PATH = the level as a
    # float, e.g. uniform=uniform=0.0442). Used by the Tier-2
    # widened-deployment efficiency probe: how much cooling per unit forcing
    # do the cells OUTSIDE the optimized pattern deliver?
    "uniform": CoupledFeatureConfig(),
    # ENSO experiment (Amendment 6): 14-feature layout with the paired
    # Nino3.4 box anomaly appended (the controller's ENSO observation).
    "fc14": CoupledFeatureConfig(include_absolute_sst=True,
                                 include_nino_box=True),
    # Classical feedforward+proportional yardstick on the rescaled pattern.
    "pi-enso": CoupledFeatureConfig(include_absolute_sst=True,
                                    include_nino_box=True),
    # Open-loop mean-feedforward schedule (fair non-feedback control):
    # compensates the MEAN hidden amplitude, reads only time.
    "ff-mean": CoupledFeatureConfig(),
}

# fc11 feature indices the PI controller reads (see coupled_features.py):
_PI_DSST_IDX = 0
_PI_TIME_IDX = 10


def make_pi_policy_fn(gain_max: float = 3.0, eta_clip=(0.25, 4.0)):
    """Deadbeat efficacy-compensating controller on a fixed pattern.

    At each control interval it estimates the realized efficacy
    eta_hat = realized_dsst / (target * time_fraction) (linear-ramp
    reference), computes the gain needed to close the REMAINING error in the
    remaining time, and commands pattern * gain / eta_hat. With exact linear
    dynamics and no cap saturation this recovers the target for any constant
    eta; cap-clipping of the command in control_step limits its authority on
    cells already at the cap (a real, reportable limitation).

    params: {"pattern": (ix, il) array, "target": float}
    """
    def pi_policy_fn(params, features):
        pattern = params["pattern"]
        target = params["target"]
        realized = features[_PI_DSST_IDX]
        tfrac = features[_PI_TIME_IDX]
        eta_hat = jnp.where(
            tfrac > 0.0,
            jnp.clip(realized / (target * jnp.maximum(tfrac, 1e-6)),
                     eta_clip[0], eta_clip[1]),
            1.0,
        )
        required = (target - realized) / ((1.0 - jnp.minimum(tfrac, 0.9))
                                          * target)
        gain = jnp.clip(required / eta_hat, 0.0, gain_max)
        return pattern * gain

    return pi_policy_fn


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ic-dir", required=True,
                   help="FRESH independent-IC directory (manifest.json). Do "
                        "not reuse ics_independent — those 10 held-out ICs "
                        "are exhausted (meta-audit).")
    p.add_argument("--arm", action="append", required=True,
                   metavar="NAME=KIND=PATH",
                   help="Policy arm; KIND in {static, fc13, time-only}. "
                        "Repeatable.")
    p.add_argument("--comparator", default="static",
                   help="Arm name every other arm is gated against.")
    p.add_argument("--split", default="heldout",
                   choices=["heldout", "train", "all"],
                   help="Which manifest split to evaluate (fresh confirmatory "
                        "sets are usually generated as all-heldout).")
    p.add_argument("--members", type=int, default=8,
                   help="Micro-ensemble members per (arm, IC). Member 0 is "
                        "the unperturbed IC.")
    p.add_argument("--member-perturb-amp", type=float, default=0.001,
                   help="SST noise amplitude (K) seeding members >= 1. Tiny: "
                        "just enough for chaos to decorrelate the weather.")
    p.add_argument("--member-seed0", type=int, default=77000)
    p.add_argument("--days", type=int, default=60)
    p.add_argument("--control-interval", type=int, default=15)
    p.add_argument("--tail-days", type=int, default=10,
                   help="Terminal time-mean window (registered metric).")
    p.add_argument("--target-cooling", type=float, default=-0.1)
    p.add_argument("--band", type=float, nargs=2, default=(-0.12, -0.08),
                   help="G2 cooling band on the registered metric.")
    p.add_argument("--max-perturbation", type=float, default=0.09)
    p.add_argument("--no-realistic-terrain", dest="realistic_terrain",
                   action="store_false", default=True)
    p.add_argument("--efficacy-mode", choices=["fixed", "randomized"],
                   default="fixed",
                   help="Tier-2: 'randomized' draws a per-IC MCB efficacy "
                        "eta ~ Uniform[range] (applied = eta x command, "
                        "unobserved by policies). Shared across arms and "
                        "members of the same IC (paired design).")
    p.add_argument("--efficacy-range", type=float, nargs=2,
                   default=(0.6, 1.4), metavar=("LO", "HI"))
    p.add_argument("--efficacy-seed", type=int, default=920,
                   help="Seed for eval-time efficacy draws (distinct from "
                        "any training-time draw stream).")
    p.add_argument("--efficacy-antithetic", action="store_true",
                   help="Draw n/2 etas and mirror them (lo+hi-eta): exact "
                        "mean (lo+hi)/2 and balanced low/high halves. Fixes "
                        "the 2026-07-31 review finding that a plain seed-920 "
                        "draw is skewed high (mean 1.07, 13/20 above 1), "
                        "flattering the PI arm and starving the NN's "
                        "low-eta direction.")
    p.add_argument("--enso-mode", choices=["off", "randomized", "fixed"],
                   default="off",
                   help="Amendment 6: impose a pacemaker El Nino on every "
                        "ARM rollout (never the baseline — the no-ENSO "
                        "baseline defines the target). 'randomized' draws a "
                        "hidden per-IC amplitude ~ U[range]; 'fixed' uses "
                        "the range midpoint for every IC.")
    p.add_argument("--enso-amp-range", type=float, nargs=2,
                   default=(0.5, 2.0), metavar=("LO", "HI"))
    p.add_argument("--enso-seed", type=int, default=940)
    p.add_argument("--enso-amp-design", choices=["random", "deterministic"],
                   default="random",
                   help="'deterministic' (Amendment 7 rev 2) replaces the RNG "
                        "with n/2 mirrored pairs at |A_j| = A_max*sqrt("
                        "(j-0.5)/(n/2)): exactly zero mean, no draw luck, and "
                        "~1.8x the design-matrix spread of a median random "
                        "draw (slope s.e. x0.74). Uses --enso-amp-range's "
                        "upper bound as A_max; --enso-seed is then unused.")
    p.add_argument("--enso-antithetic", action="store_true",
                   help="Mirror amplitude draws around the range midpoint "
                        "(exact mean, balanced halves — same rationale as "
                        "--efficacy-antithetic).")
    p.add_argument("--enso-ramp-days", type=float, default=30.0)
    p.add_argument("--enso-tau-days", type=float, default=5.0)
    p.add_argument("--enso-effect-per-k", type=float, default=0.0525,
                   help="Feedforward gain of the pi-enso/ff-mean laws: the "
                        "ENSO effect per K of Nino3.4 anomaly. Amendment 7 "
                        "requires this be measured on the REGISTERED metric "
                        "(run_calibrate_enso_plant.py); the Amendment-6 "
                        "default 0.0525 was measured on atmospheric GMST and "
                        "under-scaled the ocean-dSST response by 1.36x.")
    p.add_argument("--output", required=True)
    return p.parse_args()


def parse_arms(arm_specs):
    arms = []
    for spec in arm_specs:
        parts = spec.split("=")
        # A pi-enso arm may carry a per-arm feedforward gain as KIND@FF
        # (e.g. pi-enso@0 = the ENSO-blind outcome-feedback ablation of
        # Amendment 7 revision 1). Everything else is unchanged.
        ff_override = fb_override = None
        if len(parts) == 3 and "@" in parts[1]:
            base, _, rest = parts[1].partition("@")
            if base == "pi-enso":
                bits = rest.split("@")
                if len(bits) > 2:
                    raise SystemExit(
                        f"Bad KIND in '{spec}': pi-enso@FF[@FB], where FF and "
                        f"FB are WEIGHTS (0 = ablate, 1 = designed strength)")
                try:
                    ff_override = float(bits[0])
                    if len(bits) == 2:
                        fb_override = float(bits[1])
                except ValueError:
                    raise SystemExit(f"Bad gain in '{spec}'")
                parts[1] = base
        if len(parts) != 3 or parts[1] not in FEATURE_CONFIGS:
            raise SystemExit(
                f"Bad --arm '{spec}': expected NAME=KIND=PATH with KIND in "
                f"{sorted(FEATURE_CONFIGS)}")
        name, kind, path = parts
        if kind == "uniform":
            try:
                float(path)
            except ValueError:
                raise SystemExit(
                    f"--arm {name}: uniform kind takes a float level, "
                    f"got '{path}'")
        elif not Path(path).exists():
            raise SystemExit(f"--arm {name}: path does not exist: {path}")
        arms.append({"name": name, "kind": kind, "path": path,
                     "ff_override": ff_override,
                     "fb_override": fb_override})
    names = [a["name"] for a in arms]
    if len(set(names)) != len(names):
        raise SystemExit(f"Duplicate arm names: {names}")
    return arms


def build_arm(arm, policy, coords, target_cooling, enso_amp_mean=None,
              enso_effect_per_k=0.0525, reference_scale=1.0):
    """Build one arm; returns (policy_fn, params, feature_config)."""
    fc = FEATURE_CONFIGS[arm["kind"]]
    dim = get_coupled_feature_dim(fc)
    if arm["kind"] == "static":
        params = warm_start_params(policy, dim, arm["path"])
        return policy.apply, params, fc
    if arm["kind"] == "pi":
        with open(arm["path"], "rb") as f:
            stage1 = pickle.load(f)
        pattern = jnp.asarray(stage1["best_pattern"])
        params = {"pattern": pattern, "target": float(target_cooling)}
        return make_pi_policy_fn(), params, fc
    if arm["kind"] in ("pi-enso", "ff-mean"):
        with open(arm["path"], "rb") as f:
            stage1 = pickle.load(f)
        pattern = jnp.asarray(stage1["best_pattern"])
        params = {"pattern": pattern, "target": float(target_cooling)}
        if arm["kind"] == "pi-enso":
            # KIND@FF[@FB]: FF and FB are WEIGHTS on the feedforward and
            # feedback terms, not raw coefficients. @0 ablates a term, @1
            # leaves it at its designed strength (the measured
            # --enso-effect-per-k for feedforward). Treating FF as a raw
            # coefficient once made "@1" mean a gain 18x the measured value.
            ff = arm.get("ff_override")
            fb = arm.get("fb_override")
            return make_enso_pi_policy_fn(
                enso_effect_per_K=enso_effect_per_k,
                ff_weight=(1.0 if ff is None else ff),
                fb_weight=(1.0 if fb is None else fb),
                reference_scale=reference_scale), params, fc
        assert enso_amp_mean is not None, "ff-mean arm requires --enso-mode"
        return make_enso_ff_mean_policy_fn(
            amp_mean=enso_amp_mean,
            enso_effect_per_K=enso_effect_per_k), params, fc
    if arm["kind"] == "uniform":
        level = float(arm["path"])
        shape = coords.horizontal.nodal_shape

        def uniform_fn(params, features):
            return jnp.full(shape, params["level"])

        return uniform_fn, {"level": level}, fc
    params, meta = load_checkpoint(arm["path"])
    ckpt_dim = params["params"]["hidden_0"]["kernel"].shape[0]
    assert ckpt_dim == dim, (
        f"Arm {arm['name']}: checkpoint feature dim {ckpt_dim} != "
        f"{arm['kind']} feature dim {dim}")
    return policy.apply, params, fc


def aggregate_and_gate(cells, comparator, target, band, tail_days):
    """Aggregate member-level cells to per-IC means; compute noise + gates.

    cells: {arm_name: {metric: (n_ics, k) array}}. Pure NumPy so this is
    unit-testable without the model (run_confirmatory_eval_test.py).

    Returns (per_ic, noise, gates).
    """
    arm_names = list(cells.keys())
    some = next(iter(cells.values()))["dsst_10d"]
    n_ics, k = some.shape

    per_ic = {name: {kk: vv.mean(axis=1) for kk, vv in c.items()}
              for name, c in cells.items()}

    # Corrected noise floor: within-IC member sd = per-run chaos noise.
    noise = {}
    for name in arm_names:
        member_sd = cells[name]["dsst_10d"].std(axis=1, ddof=1) if k > 1 \
            else np.full(n_ics, np.nan)
        noise[name] = {
            "per_run_chaos_sd": float(np.nanmean(member_sd)),
            "per_ic_member_sd": member_sd.tolist(),
            "cross_ic_sd_of_means": float(per_ic[name]["dsst_10d"].std(ddof=1)),
            "se_of_arm_mean": float(per_ic[name]["dsst_10d"].std(ddof=1)
                                    / np.sqrt(n_ics)),
        }

    gates = {"n_ics": n_ics, "members": k,
             "metric": f"final_{tail_days}d_mean_dsst"}
    for name in arm_names:
        gates[f"G2_cooling[{name}]"] = cooling_gate(
            per_ic[name]["dsst_10d"], band=band)
    comp_err = np.abs(per_ic[comparator]["dsst_10d"] - target)
    for name in arm_names:
        if name == comparator:
            continue
        err = np.abs(per_ic[name]["dsst_10d"] - target)
        gates[f"G3_improvement[{name} vs {comparator}]"] = (
            improvement_gate(err, comp_err))
        gates[f"paired_report[{name} vs {comparator}]"] = (
            paired_report(err, comp_err))
    # All pairwise reports (feedback / warm-start ablations by arm naming).
    for a in range(len(arm_names)):
        for b in range(a + 1, len(arm_names)):
            na, nb = arm_names[a], arm_names[b]
            if comparator in (na, nb):
                continue
            err_a = np.abs(per_ic[na]["dsst_10d"] - target)
            err_b = np.abs(per_ic[nb]["dsst_10d"] - target)
            gates[f"paired_report[{na} vs {nb}]"] = paired_report(err_a, err_b)
    return per_ic, noise, gates


def main():
    args = parse_args()
    assert args.days % args.control_interval == 0
    band = tuple(args.band)
    arms = parse_arms(args.arm)
    if args.comparator not in {a["name"] for a in arms}:
        raise SystemExit(f"--comparator {args.comparator} is not an arm")

    print("=" * 72)
    print("CONFIRMATORY MICRO-ENSEMBLE EVALUATION (meta-audit Tier 1)")
    print("=" * 72)
    print(f"JAX devices: {jax.devices()}")
    print(f"arms: {[a['name'] + ':' + a['kind'] for a in arms]} | "
          f"comparator: {args.comparator}")
    print(f"members: {args.members} (amp {args.member_perturb_amp} K) | "
          f"metric: final-{args.tail_days}-day mean dSST | days {args.days}")

    start_datetime = jdt.to_datetime(START_DATE)
    coupler, coords, terrain, atm_model = setup_coupled_model(
        start_datetime, jdt.to_timedelta(1, "day"),
        realistic_terrain=args.realistic_terrain,
    )
    template_carry = coupler.initialize()
    ocean_mask = ocean_mask_from_coupler(coupler)
    workflow = coupler_workflow(coupler)
    step_fn = create_coupled_step_fn(coupler, workflow, jitted=True)
    area_weights = compute_area_weights(coords)

    manifest, train_ics, heldout_ics = load_ics(
        args.ic_dir, args.days, template_carry,
        require_realistic_terrain=args.realistic_terrain,
    )
    ics = {"heldout": heldout_ics, "train": train_ics,
           "all": train_ics + heldout_ics}[args.split]
    if not ics:
        raise SystemExit(f"No ICs in split '{args.split}' of {args.ic_dir}")
    print(f"ICs: {len(ics)} ({args.split}) from {args.ic_dir}")

    policy = MCBPolicyMLP(
        output_shape=coords.horizontal.nodal_shape,
        hidden_dims=(256, 256),
        max_perturbation=args.max_perturbation,
    )
    enso_amp_mean = (sum(args.enso_amp_range) / 2.0
                     if args.enso_mode != "off" else None)
    # Amendment 7: the control law's ramp reference is stretched so a perfect
    # tracker scores the target on the TAIL-AVERAGED metric rather than
    # structurally undershooting it (Addendum 5, +16.4 mK).
    reference_scale = tail_reference_scale(args.days, args.tail_days)
    print(f"pi-enso reference scale (tail-{args.tail_days} on {args.days} d): "
          f"{reference_scale:.4f} | enso_effect_per_K "
          f"{args.enso_effect_per_k:.5f}")
    arm_params = {}
    for arm in arms:
        policy_fn, params, fc = build_arm(arm, policy, coords,
                                          args.target_cooling,
                                          enso_amp_mean=enso_amp_mean,
                                          enso_effect_per_k=args.enso_effect_per_k,
                                          reference_scale=reference_scale)
        config = CoupledControllerConfig(
            control_interval_steps=args.control_interval,
            total_steps=args.days,
            target_cooling=args.target_cooling,
            feature_config=fc,
            max_perturbation=args.max_perturbation,
            use_checkpointing=True,
        )
        arm_params[arm["name"]] = (policy_fn, params, config)

    n_ics, k = len(ics), args.members

    # Tier-2 efficacy draws: one eta per IC (an episode property), shared by
    # every arm and member of that IC so comparisons stay exactly paired.
    if args.efficacy_mode == "randomized":
        lo, hi = args.efficacy_range
        eff_rng = np.random.default_rng(args.efficacy_seed)
        if args.efficacy_antithetic:
            half = eff_rng.uniform(lo, hi, (n_ics + 1) // 2)
            anti = (lo + hi) - half
            efficacies = [float(x)
                          for pair in zip(half, anti) for x in pair][:n_ics]
        else:
            efficacies = [float(x) for x in eff_rng.uniform(lo, hi, n_ics)]
        print(f"Efficacy randomization ON: eta ~ U[{lo}, {hi}] "
              f"(seed {args.efficacy_seed}"
              f"{', antithetic' if args.efficacy_antithetic else ''}): "
              f"{[round(e, 3) for e in efficacies]}")
    else:
        efficacies = [1.0] * n_ics

    # Amendment 6 ENSO draws: one hidden amplitude per IC (an episode
    # property), shared by every arm and member of that IC (paired design).
    if args.enso_mode == "randomized" and args.enso_amp_design == "deterministic":
        a_max = float(max(abs(x) for x in args.enso_amp_range))
        m = n_ics // 2
        mags = a_max * np.sqrt((np.arange(1, m + 1) - 0.5) / m)
        enso_amps = [float(x) for pair in zip(mags, -mags) for x in pair]
        enso_amps = enso_amps[:n_ics]
        print(f"ENSO DETERMINISTIC design: {m} mirrored pairs, A_max {a_max}, "
              f"mean {np.mean(enso_amps):+.3e}, "
              f"Saa {float(((np.array(enso_amps) - np.mean(enso_amps))**2).sum()):.2f}: "
              f"{[round(a, 3) for a in enso_amps]}")
    elif args.enso_mode == "randomized":
        lo, hi = args.enso_amp_range
        enso_rng = np.random.default_rng(args.enso_seed)
        if args.enso_antithetic:
            half = enso_rng.uniform(lo, hi, (n_ics + 1) // 2)
            anti = (lo + hi) - half
            enso_amps = [float(x)
                         for pair in zip(half, anti) for x in pair][:n_ics]
        else:
            enso_amps = [float(x) for x in enso_rng.uniform(lo, hi, n_ics)]
        print(f"ENSO randomization ON: A ~ U[{lo}, {hi}] "
              f"(seed {args.enso_seed}"
              f"{', antithetic' if args.enso_antithetic else ''}): "
              f"{[round(a, 3) for a in enso_amps]}")
    elif args.enso_mode == "fixed":
        enso_amps = [float(sum(args.enso_amp_range) / 2.0)] * n_ics
        print(f"ENSO fixed amplitude: {enso_amps[0]} K on every IC")
    else:
        enso_amps = [0.0] * n_ics

    enso_pattern = None
    nino_w = None
    if args.enso_mode != "off":
        enso_pattern = nino_pattern(coords.horizontal) * ocean_mask
        nino_w = box_mean_weights(coords.horizontal, enso_pattern,
                                  area_weights)

    arm_names = [a["name"] for a in arms]
    # per-arm (n_ics, k) member-level metrics
    cells = {name: {"dsst_10d": np.full((n_ics, k), np.nan),
                    "dsst_snapshot": np.full((n_ics, k), np.nan),
                    "amazon_mm_day": np.full((n_ics, k), np.nan),
                    "sahel_mm_day": np.full((n_ics, k), np.nan),
                    "mean_mcb_forcing": np.full((n_ics, k), np.nan),
                    "mean_loss": np.full((n_ics, k), np.nan),
                    "nino_realized_final": np.full((n_ics, k), np.nan)}
             for name in arm_names}

    results = {
        "config": vars(args), "arms": arms, "manifest_ic_dir": args.ic_dir,
        "ic_entries": [e for e, _, _ in ics], "cells": cells,
        "efficacies": efficacies, "enso_amps": enso_amps,
    }
    out_path = Path(args.output)
    t_campaign = time.time()

    for i, (entry, carry, _stored_baseline) in enumerate(ics):
        for m in range(k):
            if m == 0:
                member_carry = carry
            else:
                member_seed = args.member_seed0 + 97 * entry["index"] + m
                member_carry = perturb_sst(
                    carry, member_seed, args.member_perturb_amp)
            # Regenerate the paired baseline for THIS member IN THIS PROCESS
            # (cached cross-program baselines carry ~0.01 K mismatch).
            # Under --enso-mode the baseline stays UNWRAPPED (no ENSO): it
            # is the no-ENSO control that defines the target AND the
            # pacemaker's relaxation reference.
            t0 = time.time()
            member_baseline = compute_baseline_trajectory(
                member_carry, step_fn, num_steps=args.days, coords=coords)
            t_base = time.time() - t0
            step_fn_transform = None
            if args.enso_mode != "off":
                member_t0 = float(member_carry["ocn"]["state"].sim_time)
                enso_cfg = EnsoConfig(
                    amplitude=enso_amps[i],
                    ramp_days=args.enso_ramp_days,
                    relax_tau_days=args.enso_tau_days,
                )
                step_fn_transform = (
                    lambda sf, _cfg=enso_cfg, _t0=member_t0,
                    _ref=member_baseline.sst:
                    wrap_step_fn_with_enso(sf, enso_pattern, _cfg, _ref,
                                           _t0))
            for name in arm_names:
                arm_policy_fn, params, config = arm_params[name]
                t0 = time.time()
                out = evaluate_coupled_policy(
                    coupler=coupler,
                    workflow=workflow,
                    policy_fn=arm_policy_fn,
                    policy_params=params,
                    initial_carry=member_carry,
                    baseline_trajectory=member_baseline,
                    coords=coords,
                    ocean_mask=ocean_mask,
                    config=config,
                    tail_mean_days=args.tail_days,
                    efficacy=efficacies[i],
                    step_fn_transform=step_fn_transform,
                )
                mtr = out["metrics"]
                c = cells[name]
                c["dsst_10d"][i, m] = mtr["final_sst_change_10d"]
                c["dsst_snapshot"][i, m] = mtr["final_sst_change"]
                c["mean_loss"][i, m] = mtr["mean_loss"]
                c["mean_mcb_forcing"][i, m] = mtr["mean_mcb_forcing"]
                base_final = member_baseline.at_step(args.days)
                c["amazon_mm_day"][i, m] = float(region_precip_change_mm_day(
                    out["final_carry"], base_final.precipitation, coords,
                    area_weights, region="amazon"))
                c["sahel_mm_day"][i, m] = float(region_precip_change_mm_day(
                    out["final_carry"], base_final.precipitation, coords,
                    area_weights, region="sahel"))
                if args.enso_mode != "off":
                    final_sst = (out["final_carry"]["ocn"]["state"]
                                 .sea_surface_temperature)
                    c["nino_realized_final"][i, m] = float(jnp.sum(
                        (final_sst - member_baseline.sst[args.days])
                        * nino_w))
                print(f"  IC {entry['index']:02d} m{m} {name:>10}: "
                      f"dSST10d {mtr['final_sst_change_10d']:+.4f} "
                      f"(snap {mtr['final_sst_change']:+.4f}, "
                      f"eta {efficacies[i]:.3f}, "
                      f"A {enso_amps[i]:.2f}) "
                      f"[base {t_base:.0f}s, eval {time.time() - t0:.0f}s]",
                      flush=True)
                t_base = 0.0  # only report baseline time once per member
        # Crash tolerance: persist after every IC.
        with open(out_path, "wb") as f:
            pickle.dump(results, f)

    # ---- Aggregation, noise accounting, and gates ----
    per_ic, noise, gates = aggregate_and_gate(
        cells, comparator=args.comparator, target=args.target_cooling,
        band=band, tail_days=args.tail_days)
    results["per_ic"] = per_ic
    results["noise"] = noise
    results["gates"] = gates

    with open(out_path, "wb") as f:
        pickle.dump(results, f)

    print("\n" + "=" * 72)
    print(f"DONE in {(time.time() - t_campaign) / 3600:.2f} h -> {out_path}")
    print("=" * 72)
    for name in arm_names:
        g = gates[f"G2_cooling[{name}]"]
        nz = noise[name]
        print(f"{name:>10}: dSST10d {g['mean']:+.4f} +/- {g['se']:.4f} "
              f"[{g['verdict']}] | chaos sd/run {nz['per_run_chaos_sd']:.4f}")
    for key, g in gates.items():
        if key.startswith("G3_improvement"):
            print(f"{key}: improvement {g['improvement']:+.4f} +/- "
                  f"{g['se']:.4f} [{g['verdict']}]")
        if key.startswith("paired_report"):
            print(f"{key}: mean {g['mean']:+.4f} p_t={g['p_t']:.3f} "
                  f"p_w={g['p_wilcoxon']:.3f} "
                  f"|equiv|<{g['equivalence_bound_95']:.4f} @95%")


if __name__ == "__main__":
    main()
