#!/usr/bin/env python
"""Distill the PI/deadbeat law into the fc13 MLP policy (Tier-2b, Amendment 4).

Tier-2 showed the deadbeat controller halves efficacy-uncertainty error while
BPTT-trained NNs never leave the static-under-eta loss floor — a TRAINING
failure (chaotic-rollout gradients), not an information or capacity limit.
This script sidesteps the chaos entirely: supervised regression of the NN
onto the PI law over synthetic feature batches (noiseless, minutes), so
fine-tuning starts inside the ~10 mK basin instead of hunting feedback from
scratch.

Target: clip(pattern x pi_gain(f0, f10), 0, cap), where f0 = realized
global-mean ocean dSST anomaly and f10 = time fraction (the fc13 layout keeps
fc11's indices; absolute-SST features are appended at 11-12). Other features
are sampled broadly so the init is invariant to them; fine-tuning may later
learn to use them.

Acceptance (Amendment 4): mean |NN - PI| output error < 0.002 albedo on a
held-out feature grid. A FakeCoupler closed-loop equivalence test lives in
run_pi_imitation_test.py.

Usage:
    python run_pi_imitation.py \
        --stage1 mcb_experiments_gpu/stage1_v2/stage1_optimized_pattern.pkl \
        --seed 52 --output mcb_experiments_gpu/t2b_imitation_s52.pkl
"""

import argparse
import json
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jcm.mcb import CoupledFeatureConfig, MCBPolicyMLP, get_coupled_feature_dim
from jcm.mcb.enso import make_enso_pi_policy_fn
from jcm.mcb.train import save_checkpoint
from run_confirmatory_eval import _PI_DSST_IDX, _PI_TIME_IDX, make_pi_policy_fn

FC13 = CoupledFeatureConfig(include_absolute_sst=True)
FC14 = CoupledFeatureConfig(include_absolute_sst=True, include_nino_box=True)
TFRACS = (0.0, 0.25, 0.5, 0.75)  # the 4 control intervals at CI=15 / 60 d
_NINO_IDX = 13


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage1", required=True,
                   help="Stage-1 optimized pattern pickle (the PI actuator)")
    p.add_argument("--law", choices=["pi", "pi-enso"], default="pi",
                   help="'pi' = Tier-2b deadbeat on fc13 (default, "
                        "unchanged); 'pi-enso' = Amendment-6 feedforward+"
                        "proportional ENSO law on fc14.")
    p.add_argument("--target-cooling", type=float, default=-0.1)
    p.add_argument("--max-perturbation", type=float, default=0.09)
    p.add_argument("--seed", type=int, default=52)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=3e-3,
               help="3e-3/batch 512 escapes the predict-the-mean plateau the\n                    sharp clipped-gain surface creates; 1e-3/128 stalls at MSE ~5e-3.")
    p.add_argument("--f0-range", type=float, nargs=2, default=(-0.25, 0.05),
                   help="Realized-dSST sampling range for feature 0. For "
                        "--law pi-enso use a range spanning ENSO warming, "
                        "e.g. -0.40 0.30.")
    p.add_argument("--nino-range", type=float, nargs=2, default=(-0.2, 2.3),
                   help="pi-enso only: realized Nino-box anomaly sampling "
                        "range for feature 13.")
    p.add_argument("--num-intervals", type=int, default=None,
                   help="Control intervals per episode; tfracs sampled at "
                        "i/n. Default: len(TFRACS)=4 (60 d / CI 15); the "
                        "ENSO campaign uses 12 (180 d / CI 15).")
    p.add_argument("--anchors-json", type=str, default=None,
                   help="JSON from run_feature_envelope.py with measured "
                        "{f11_lo, f11_hi, f12_lo, f12_hi} nuisance-feature "
                        "envelopes (Tier-2b lesson: measure, never guess). "
                        "Default: the Tier-2b 60-day anchors.")
    p.add_argument("--acceptance", type=float, default=0.002,
                   help="Mean |NN - PI| albedo error bound (Amendment 4).")
    p.add_argument("--output", required=True)
    return p.parse_args()


# Measured fc13 operating point of the real coupled model (diya, ics_t2_train
# ICs at rollout start, 2026-08-02): the absolute-SST features are far from
# zero. The first campaign attempt sampled them N(0, 0.5), putting the real
# values at -3.7 / -8.5 sigma — the imitation net extrapolated to a frozen
# gain ~1 (behaved exactly like static; the Amendment-4 gate caught it in
# 23 min). Anchor the sampling at the measured values with generous widths.
F11_ANCHOR = -1.83   # global ocean mean SST - 288 (K)
F12_ANCHOR = -4.23   # NH - SH ocean SST difference (K)


def sample_features(rng, batch, feature_dim, f0_range, tfracs=TFRACS,
                    anchors=None, nino_range=(-0.2, 2.3)):
    """Synthetic feature batch covering the law-relevant subspace.

    The target depends only on features 0 (realized dSST), 10 (time
    fraction) and — for the ENSO law — 13 (Nino-box anomaly); every other
    feature is sampled over a generous superset of its real rollout range
    so the distilled net is INVARIANT to it across the states it will
    actually see (not just near zero).
    """
    if anchors is None:
        anchors = {"f11_lo": F11_ANCHOR - 1.5, "f11_hi": F11_ANCHOR + 1.5,
                   "f12_lo": F12_ANCHOR - 2.0, "f12_hi": F12_ANCHOR + 2.0}
    f = np.zeros((batch, feature_dim))
    f[:, 1:5] = 0.2 * rng.standard_normal((batch, 4))    # SST anomalies
    if feature_dim > 5:
        f[:, 5] = 1.0 * rng.standard_normal(batch)       # atm-temp anomaly
    if feature_dim > 6:
        f[:, 6] = 5.0 * rng.standard_normal(batch)       # heat-flux anomaly
    if feature_dim > 9:
        f[:, 7:10] = 0.5 * rng.standard_normal((batch, 3))  # precip anomalies
    f[:, _PI_DSST_IDX] = rng.uniform(f0_range[0], f0_range[1], batch)
    f[:, _PI_TIME_IDX] = rng.choice(tfracs, batch)
    if feature_dim > 11:
        f[:, 11] = rng.uniform(anchors["f11_lo"], anchors["f11_hi"], batch)
    if feature_dim > 12:
        f[:, 12] = rng.uniform(anchors["f12_lo"], anchors["f12_hi"], batch)
    if feature_dim > _NINO_IDX:
        f[:, _NINO_IDX] = rng.uniform(nino_range[0], nino_range[1], batch)
    return f


def fold_standardization(params, mu, sigma):
    """Fold z = (f - mu) / sigma into the first layer: same function on RAW f.

    W_raw[i, :] = W_z[i, :] / sigma_i;  b_raw = b_z - sum_i W_z[i, :] mu_i/sigma_i.
    Training on standardized inputs is what makes the wide nuisance-feature
    ranges optimizable (raw-scale inputs of sd ~5 drown the ~0.02-scale f0
    signal in a plateau); folding keeps the SAVED checkpoint a standard MLP
    over raw fc13 features, so fine-tuning and eval need no changes.
    """
    import flax
    params = flax.core.unfreeze(params) if isinstance(
        params, flax.core.FrozenDict) else params
    layer = params["params"]["hidden_0"]
    w = jnp.asarray(layer["kernel"])            # (feature_dim, hidden)
    b = jnp.asarray(layer["bias"])
    sigma = jnp.asarray(sigma)[:, None]
    mu = jnp.asarray(mu)[:, None]
    layer["kernel"] = w / sigma
    layer["bias"] = b - jnp.sum(w * (mu / sigma), axis=0)
    return params


def distill(pattern, args, log_every=500):
    """Train the MLP to reproduce the chosen law; returns (params, report).

    Trains on STANDARDIZED features and folds the standardization into the
    first layer afterwards (fold_standardization), so the returned params
    consume raw features. The fidelity check below runs on RAW features
    through the folded params — it validates both the fit and the fold.
    """
    law = getattr(args, "law", "pi")
    num_intervals = getattr(args, "num_intervals", None)
    anchors_json = getattr(args, "anchors_json", None)
    nino_range = tuple(getattr(args, "nino_range", (-0.2, 2.3)))
    fc = FC14 if law == "pi-enso" else FC13
    feature_dim = get_coupled_feature_dim(fc)
    if num_intervals:
        tfracs = tuple(i / num_intervals for i in range(num_intervals))
    else:
        tfracs = TFRACS
    anchors = None
    if anchors_json:
        with open(anchors_json) as f:
            anchors = json.load(f)
        print(f"  measured nuisance anchors: {anchors}")

    def sample(rng, batch):
        return sample_features(rng, batch, feature_dim, args.f0_range,
                               tfracs=tfracs, anchors=anchors,
                               nino_range=nino_range)

    policy = MCBPolicyMLP(
        output_shape=pattern.shape, hidden_dims=(256, 256),
        max_perturbation=args.max_perturbation,
    )
    pi_fn = (make_enso_pi_policy_fn() if law == "pi-enso"
             else make_pi_policy_fn())
    pi_params = {"pattern": jnp.asarray(pattern),
                 "target": float(args.target_cooling)}
    cap = args.max_perturbation

    def pi_target(feats):
        return jnp.clip(pi_fn(pi_params, feats), 0.0, cap)

    # Empirical standardization of the sampling distribution itself.
    stat_rng = np.random.default_rng(args.seed + 20_000)
    big = sample(stat_rng, 50_000)
    mu = big.mean(axis=0)
    sigma = np.maximum(big.std(axis=0), 1e-3)

    params = policy.init(jax.random.PRNGKey(args.seed),
                         jnp.zeros(feature_dim))
    opt = optax.adam(args.learning_rate)
    opt_state = opt.init(params)
    mu_j, sigma_j = jnp.asarray(mu), jnp.asarray(sigma)

    @jax.jit
    def step(params, opt_state, feats):
        def loss_fn(p):
            z = (feats - mu_j) / sigma_j
            outs = jax.vmap(lambda f: policy.apply(p, f))(z)
            targets = jax.vmap(pi_target)(feats)  # target uses RAW features
            return jnp.mean((outs - targets) ** 2)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = opt.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss

    rng = np.random.default_rng(args.seed)
    for i in range(args.steps):
        feats = jnp.asarray(sample(rng, args.batch), dtype=jnp.float32)
        params, opt_state, loss = step(params, opt_state, feats)
        if i % log_every == 0 or i == args.steps - 1:
            print(f"  step {i:5d}: imitation MSE {float(loss):.3e}",
                  flush=True)

    params = fold_standardization(params, mu, sigma)

    # Fidelity on a held-out deterministic grid (fresh RNG stream).
    grid_rng = np.random.default_rng(args.seed + 10_000)
    feats = jnp.asarray(sample(grid_rng, 2048), dtype=jnp.float32)
    outs = jax.vmap(lambda f: policy.apply(params, f))(feats)
    targets = jax.vmap(pi_target)(feats)
    err = jnp.abs(outs - targets)
    report = {
        "mean_abs_err": float(jnp.mean(err)),
        "max_abs_err": float(jnp.max(err)),
        "acceptance": args.acceptance,
        "accepted": bool(float(jnp.mean(err)) < args.acceptance),
    }
    return params, report


def main():
    args = parse_args()
    with open(args.stage1, "rb") as f:
        stage1 = pickle.load(f)
    pattern = np.asarray(stage1["best_pattern"])
    print("=" * 72)
    print(f"PI-IMITATION DISTILLATION (law {args.law}, seed {args.seed})")
    print("=" * 72)
    params, report = distill(pattern, args)
    print(f"fidelity: mean|err| {report['mean_abs_err']:.5f} albedo "
          f"(max {report['max_abs_err']:.5f}) -> "
          f"{'ACCEPTED' if report['accepted'] else 'REJECTED'} "
          f"(bound {report['acceptance']})")
    fc = FC14 if args.law == "pi-enso" else FC13
    save_checkpoint(params, args.output, metadata={
        "mode": "pi-imitation",
        "law": args.law,
        "seed": args.seed,
        "stage1": args.stage1,
        "target_cooling": args.target_cooling,
        "max_perturbation": args.max_perturbation,
        "feature_dim": get_coupled_feature_dim(fc),
        "include_absolute_sst": True,
        "include_nino_box": args.law == "pi-enso",
        "fidelity": report,
        "steps": args.steps,
    })
    print(f"saved -> {args.output}")
    if not report["accepted"]:
        raise SystemExit(
            "IMITATION FIDELITY BELOW ACCEPTANCE — do not fine-tune from "
            "this checkpoint (Amendment 4 gate).")


if __name__ == "__main__":
    main()
