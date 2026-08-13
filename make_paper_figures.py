#!/usr/bin/env python
"""Publication figures for the MCB control paper, from committed campaign data.

Every panel is drawn from a pickle in mcb_experiments_gpu/ — no new simulation
and no hand-entered numbers, so a figure can never drift from the result it
depicts. Run after any re-analysis to regenerate.

Figures:
  fig1_disturbance_rejection : the central evidence. Per-episode error against
      the hidden disturbance amplitude, per arm, for a persistent STEP and a
      growing RAMP. The slope IS the endpoint: it is exactly invariant to
      rescaling a controller, so a non-adaptive arm cannot flatten it.
  fig2_efficacy_uncertainty  : the earlier arc — a classical law halves the
      error, controllers trained through the model fail to learn it, and a
      controller taught by imitating the law succeeds.
  fig3_effects_summary       : forest plot of the headline effects with
      confidence intervals, on a common axis.

Usage:
    python make_paper_figures.py [--outdir figures]
"""

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from analyze_enso8 import _ols, hc3_se, rms_a  # noqa: E402
from analyze_enso8 import wild_bootstrap_slope  # noqa: E402

E = Path("mcb_experiments_gpu")
TARGET = -0.1

# Colour-blind-safe (Okabe-Ito). One colour per CONTROLLER CLASS, held
# consistent across every figure so the reader learns the mapping once.
C = {
    "open": "#000000",        # no control / static
    "reactive": "#0072B2",    # outcome feedback, disturbance NOT observed
    "anticip": "#D55E00",     # feedforward on the observed disturbance
    "learned": "#009E73",     # neural controller
    "classical": "#CC79A7",   # classical PI / deadbeat
    "failed": "#999999",      # trained-through-the-model (did not learn)
}


def load(name):
    with open(E / name, "rb") as f:
        return pickle.load(f)


def per_ic_err(results, arm):
    return np.asarray(results["cells"][arm]["dsst_10d"]).mean(axis=1) - TARGET


def _panel_rejection(ax, results, arms, title, subtitle):
    amps = np.asarray(results["enso_amps"], float)
    xs = np.linspace(amps.min() * 1.05, amps.max() * 1.05, 100)
    for arm, (label, colour, marker) in arms.items():
        e = per_ic_err(results, arm) * 1000
        slope, intercept, _, _ = _ols(amps, e)
        se = hc3_se(amps, e)
        ax.scatter(amps, e, s=26, color=colour, marker=marker, alpha=0.65,
                   edgecolors="none", zorder=3)
        ax.plot(xs, intercept + slope * xs, color=colour, lw=2.0, zorder=4,
                label=f"{label}  ({slope:+.1f} ± {se:.1f} mK/K)")
    ax.axhline(0, color="0.6", lw=0.8, ls=":", zorder=1)
    ax.axvline(0, color="0.6", lw=0.8, ls=":", zorder=1)
    ax.set_xlabel("hidden disturbance amplitude  (K of Niño3.4)")
    ax.set_ylabel("temperature error vs target  (mK)")
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left", pad=20)
    ax.text(0.0, 1.012, subtitle, transform=ax.transAxes, fontsize=8.5,
            color="0.35", va="bottom")
    ax.legend(fontsize=7.6, loc="upper left", framealpha=0.92,
              handlelength=1.4)
    ax.spines[["top", "right"]].set_visible(False)


def fig1(outdir):
    """Draw the central figure: slope = disturbance sensitivity."""
    step = load("enso8_eval.pkl")
    ramp = load("ramp_eval.pkl")
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), sharey=True)
    _panel_rejection(
        axes[0], step,
        {"static": ("no control", C["open"], "o"),
         "b6": ("feedback, disturbance unobserved", C["reactive"], "s"),
         "p2": ("feedback + observes disturbance", C["anticip"], "^"),
         "imitation_s72": ("neural (imitation)", C["learned"], "D")},
        "a   Persistent step disturbance",
        "ramps over 30 days, then holds — n = 24 climates, k = 4")
    _panel_rejection(
        axes[1], ramp,
        {"static": ("no control", C["open"], "o"),
         "b6": ("feedback, disturbance unobserved", C["reactive"], "s"),
         "p15": ("feedback + observes disturbance", C["anticip"], "^"),
         "imitation_s72": ("neural (imitation)", C["learned"], "D")},
        "b   Continuously growing disturbance",
        "ramps across the whole episode — n = 32 climates, k = 4")
    axes[1].set_ylabel("")
    fig.suptitle("Feedback rejects the disturbance; observing it adds nothing",
                 fontsize=12.5, fontweight="bold", x=0.5, y=1.045)
    fig.text(0.5, -0.035,
             "Slope is the pre-registered endpoint: rescaling a controller by "
             "any constant leaves it exactly unchanged,\nso a non-adaptive arm "
             "cannot flatten it. Flatter is better.",
             ha="center", fontsize=8.5, color="0.35")
    fig.tight_layout()
    p = outdir / "fig1_disturbance_rejection.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


def fig2(outdir):
    """Draw the efficacy-uncertainty arc (classical wins, training fails)."""
    t2 = load("tier2_eval.pkl")
    t2b = load("tier2b_eval.pkl")

    def group_err(res, arms):
        return np.mean([np.abs(np.asarray(res["cells"][a]["dsst_10d"]).mean(1)
                               - TARGET) for a in arms], axis=0) * 1000

    bars = [
        ("static\n(fixed plan)", group_err(t2, ["static"]), C["open"]),
        ("open-loop\nschedule", group_err(
            t2, ["openloop_s42", "openloop_s43", "openloop_s44"]), C["open"]),
        ("neural, trained\nTHROUGH the model", group_err(
            t2, ["feedback_s42", "feedback_s43", "feedback_s44"]), C["failed"]),
        ("classical\ncontroller", group_err(t2, ["pi"]), C["classical"]),
        ("neural, taught by\nIMITATING it", group_err(
            t2b, ["finetuned_s52", "finetuned_s53", "finetuned_s54"]),
         C["learned"]),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    for i, (label, vals, colour) in enumerate(bars):
        m = vals.mean()
        se = vals.std(ddof=1) / np.sqrt(vals.size)
        ax.bar(i, m, 0.62, color=colour, alpha=0.88, zorder=3)
        ax.errorbar(i, m, yerr=se, color="0.15", capsize=4, lw=1.3, zorder=4)
        ax.text(i, m + se + 0.7, f"{m:.1f}", ha="center", fontsize=9,
                fontweight="bold")
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([b[0] for b in bars], fontsize=8.6)
    ax.set_ylabel("mean miss of the cooling target  (mK)")
    ax.set_title("Under hidden seeding-efficacy uncertainty, gradients through "
                 "the model never learn\nthe control law — but imitating a "
                 "classical one transfers it",
                 fontsize=11, fontweight="bold", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.annotate("", xy=(2, bars[2][1].mean() + 3.2),
                xytext=(0, bars[0][1].mean() + 3.2),
                arrowprops=dict(arrowstyle="<->", color="0.45", lw=1.1))
    ax.text(1.0, bars[0][1].mean() + 4.4, "no better than doing nothing",
            ha="center", fontsize=8.2, color="0.35")
    fig.tight_layout()
    p = outdir / "fig2_efficacy_uncertainty.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


def fig3(outdir):
    """Draw a forest plot of the headline effects, split by unit."""
    # Two panels, because the endpoints have DIFFERENT UNITS: disturbance
    # sensitivity is mK per K of disturbance, target error is mK. Putting them
    # on one axis would invite a comparison that is not meaningful.
    slope_rows, mean_rows = [], []

    for pkl, ant, lab in (("enso8_eval.pkl", "p2", "persistent step"),
                          ("ramp_eval.pkl", "p15", "growing ramp")):
        res = load(pkl)
        amps = np.asarray(res["enso_amps"], float)
        d = (per_ic_err(res, ant) - per_ic_err(res, "b6")) * 1000
        r = wild_bootstrap_slope(amps, d, n_boot=8000)
        slope_rows.append((f"Observing the disturbance\n({lab})",
                           r["slope"], r["ci95_boot"], C["anticip"]))

    t2b = load("tier2b_eval.pkl")

    def absr(arms):
        return np.mean([np.abs(np.asarray(t2b["cells"][a]["dsst_10d"]).mean(1)
                               - TARGET) for a in arms], axis=0)

    ft = absr(("finetuned_s52", "finetuned_s53", "finetuned_s54"))
    for label, x, y, colour in (
            ("Gradient fine-tuning through the\nmodel, beyond imitation",
             ft, absr(("imitation_s52",)), C["failed"]),
            ("Neural controller vs\nthe classical law", ft, absr(("pi",)),
             C["learned"]),
            ("Neural controller vs\na fixed plan", ft, absr(("static",)),
             C["learned"])):
        diff = (x - y) * 1000
        m = diff.mean()
        se = diff.std(ddof=1) / np.sqrt(diff.size)
        mean_rows.append((label, m, (m - 1.96 * se, m + 1.96 * se), colour))

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.6),
                             gridspec_kw={"width_ratios": [1, 1.25]})
    for ax, rows, xlabel, title in (
            (axes[0], slope_rows,
             "effect on disturbance sensitivity  (mK per K)",
             "a   Does watching the disturbance help?"),
            (axes[1], mean_rows, "effect on target error  (mK)",
             "b   What produced the working controller?")):
        ys = np.arange(len(rows))[::-1]
        for y, (label, m, ci, colour) in zip(ys, rows):
            ax.plot(ci, [y, y], color=colour, lw=2.6, solid_capstyle="round",
                    zorder=3)
            ax.plot(m, y, "o", color=colour, ms=8, zorder=4)
            ax.annotate(f"{m:+.1f}", (m, y), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=8.2,
                        color="0.25")
        ax.axvline(0, color="0.25", lw=1.1, zorder=1)
        ax.set_yticks(ys)
        ax.set_yticklabels([r[0] for r in rows], fontsize=8.6)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_title(title, fontsize=10.5, fontweight="bold", loc="left",
                     pad=10)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.margins(y=0.22)
    fig.suptitle("What helped, and what did not", fontsize=12.5,
                 fontweight="bold", x=0.5, y=1.10)
    fig.text(0.5, 1.015,
             "Bars are 95% intervals; those crossing zero are effects the "
             "data cannot distinguish from nothing.",
             ha="center", fontsize=8.4, color="0.35")
    fig.tight_layout()
    p = outdir / "fig3_effects_summary.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    for fn in (fig1, fig2, fig3):
        print(f"  wrote {fn(outdir)}")

    # Numbers the caption text must match, printed so they are never retyped.
    print("\ncaption-critical values (regenerate captions if these change):")
    for name, arms in (("enso8_eval.pkl", ("static", "b6", "p2",
                                           "imitation_s72")),
                       ("ramp_eval.pkl", ("static", "b6", "p15",
                                          "imitation_s72"))):
        res = load(name)
        amps = np.asarray(res["enso_amps"], float)
        s0 = _ols(amps, per_ic_err(res, "static"))[0]
        print(f"  {name}")
        for a in arms:
            e = per_ic_err(res, a)
            s = _ols(amps, e)[0]
            print(f"    {a:>14}: slope {s * 1000:+6.1f} mK/K | "
                  f"rejected {100 * (1 - s / s0):3.0f}% | "
                  f"RMS {rms_a(e) * 1000:5.1f} mK")


if __name__ == "__main__":
    main()
