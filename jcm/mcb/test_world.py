"""The test world for Experiments 2-3 (MCB_PROJECT_REPORT.md Part 18 steps 10-12).

* **Warming only where we control** (step 10). ``wrap_step_fn_with_warming``
  adds a heat input into the slab ocean, ``F(t) = step + ramp * days``
  (W m-2, into the ocean) times a pattern. It uses the slot the Q-flux uses:
  every month of the carry's Q-flux becomes ``q_base - F(t) * pattern``, so
  the monthly interpolation gives exactly the base Q-flux minus the warming.
  It is applied only in the runs being controlled, and zero warming changes
  nothing, bit for bit.
* **Surface fields.** ``surface_fields`` reads, after every coupled day:
  - the slab-ocean SST and slab-land temperature (end of day);
  - rainfall and evaporation, which are averaged over the day by the
    atmosphere wrapper (``output_averages=True``), in mm/day.
* **Segments and episodes.** ``make_segment_fn`` jits a rollout with the band
  brightening scaled by an efficacy (the true, possibly hidden, strength of
  the spraying) and with the warming. ``run_episode`` asks a policy for band
  settings at the start of every segment, which covers fixed patterns,
  feedback controllers, planners and students alike.
* **References** (step 11). ``reference_ensemble`` averages member runs with
  no brightening. Without warming this is the "normal climate" that
  controllers steer toward and are judged against; with warming it is the
  uncontrolled warmed run that the gain compares with. Dubey et al. use
  five-member means of both.
* **The look-ahead objective** (step 12). ``make_lookahead_objective`` is the
  differentiable map objective of ``jcm.mcb.scores`` evaluated on the
  look-ahead's time-mean SST, through the warming and the atmosphere
  truncation of ``jcm.mcb.gradient_truncation``, which is what a planner
  minimizes.

Settings such as the warming strength, run lengths, windows and penalty
weights are arguments here. The pilot (step 23) chooses them, and revision 1
of Amendment 9 freezes them.
"""

from typing import Callable, Dict, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from jcm.mcb.band_basis import objective_values
from jcm.mcb.gradient_fidelity import MAP_REFERENCE_K, apply_band_control
from jcm.mcb.gradient_truncation import wrap_step_fn_with_atm_truncation
from jcm.mcb.qflux import set_qflux
from jcm.mcb.scores import (
    PATTERN_ALPHA,
    PATTERN_BETA,
    area_weights,
    segment_objective,
)

SECONDS_PER_DAY = 86400.0
# SPEEDY reports rainfall and evaporation in g m-2 s-1; 1 g m-2 = 1e-3 mm.
G_PER_M2_S_TO_MM_PER_DAY = 86.4
FIELD_NAMES = ("sst", "land_temperature", "precipitation", "evaporation")
# Largest band amplitude (cloud-albedo increase), the project's cap
# (``max_perturbation`` of the earlier controllers and policies).
BRIGHTENING_CAP = 0.15


# --- Step 10: warming only where we control --------------------------------

class Warming(NamedTuple):
    """Heat into the slab ocean: ``(step + ramp * days since t0) * pattern``.

    ``step`` in W m-2, ``ramp`` in W m-2 per day, ``pattern`` (ix, il),
    1 where the heat goes, normally the slab's ocean cells, and
    ``t0_seconds`` the episode start on the ocean clock. A NamedTuple of
    arrays, so it can be traced.
    """

    step_wm2: jnp.ndarray
    ramp_wm2_per_day: jnp.ndarray
    pattern: jnp.ndarray
    t0_seconds: jnp.ndarray


def make_warming(carry: dict, pattern, step_wm2: float = 0.0,
                 ramp_wm2_per_day: float = 0.0) -> Warming:
    """Return a ``Warming`` that starts at ``carry``'s ocean time."""
    return Warming(step_wm2=jnp.asarray(step_wm2, jnp.float32),
                   ramp_wm2_per_day=jnp.asarray(ramp_wm2_per_day,
                                                jnp.float32),
                   pattern=jnp.asarray(pattern, jnp.float32),
                   t0_seconds=jnp.asarray(carry["ocn"]["state"].sim_time))


def warming_flux(warming: Warming, days):
    """Return the warming strength (W m-2) after ``days`` days."""
    return warming.step_wm2 + warming.ramp_wm2_per_day * days


def wrap_step_fn_with_warming(step_fn: Callable, q_base,
                              warming: Warming) -> Callable:
    """Add the warming through the Q-flux slot before every coupled step.

    The carry's Q-flux (upward-positive, W m-2) becomes
    ``q_base - F * pattern`` in every month, with ``F`` evaluated at
    mid-step, where the ocean model evaluates its Q-flux. ``q_base`` must be
    the episode's original Q-flux, so the warming never compounds. Everything
    may be traced; with ``F = 0`` the step is unchanged bit for bit.
    """
    q_base = jnp.asarray(q_base)
    pattern = jnp.asarray(warming.pattern)[..., None]

    def warmed_step(carry, step_idx):
        days = ((carry["ocn"]["state"].sim_time - warming.t0_seconds)
                / SECONDS_PER_DAY + 0.5)
        flux = warming_flux(warming, days)
        return step_fn(set_qflux(carry, q_base - flux * pattern), step_idx)

    return warmed_step


# --- Surface fields ----------------------------------------------------------

def surface_fields(carry: dict) -> Dict[str, jnp.ndarray]:
    """Daily SST and land temperature (K), rainfall and evaporation (mm/day).

    Rainfall is convective plus large-scale precipitation. Evaporation is
    SPEEDY's land/sea-weighted value (``evap[..., 2]``). Both are averages
    over the coupled day.
    """
    physics = carry["atm"]["derived"]["physics"]
    precip = (physics.convection.precnv + physics.condensation.precls)
    return {
        "sst": carry["ocn"]["state"].sea_surface_temperature,
        "land_temperature": carry["lnd"]["state"].land_surface_temperature,
        "precipitation": precip * G_PER_M2_S_TO_MM_PER_DAY,
        "evaporation": (physics.surface_flux.evap[..., 2]
                        * G_PER_M2_S_TO_MM_PER_DAY),
    }


def domain_weights(latitudes_rad, ocean_mask, land_mask) -> Dict[str,
                                                                  jnp.ndarray]:
    """Return area weights for the ``ocean``, ``land`` and ``global`` domains."""
    ones = jnp.ones_like(jnp.asarray(ocean_mask, jnp.float32))
    return {"ocean": area_weights(latitudes_rad, ocean_mask),
            "land": area_weights(latitudes_rad, land_mask),
            "global": area_weights(latitudes_rad, ones)}


def time_means(fields: Dict[str, np.ndarray], start_day: int,
               end_day: int) -> Dict[str, np.ndarray]:
    """Mean of each daily field over days ``[start_day, end_day)``."""
    n = next(iter(fields.values())).shape[0]
    if not 0 <= start_day < end_day <= n:
        raise ValueError(f"need 0 <= start ({start_day}) < end ({end_day}) "
                         f"<= days ({n})")
    return {k: np.asarray(v[start_day:end_day], np.float64).mean(axis=0)
            for k, v in fields.items()}


def index_anomalies(sst, land, target_sst, target_land, weight_stack):
    """T0/T1/T2/LAND of a state minus those of the target (same day).

    The objectives are weighted sums of the maps (``jcm.mcb.band_basis``), so
    this is the projection of the anomaly maps: what a classical feedback
    controller holds on target.
    """
    return (objective_values(sst, land, weight_stack)
            - objective_values(target_sst, target_land, weight_stack))


# --- Segments and episodes -----------------------------------------------------

def make_segment_fn(step_fn: Callable, patterns, num_days: int,
                    fields_fn: Callable = surface_fields) -> Callable:
    """Jitted ``segment(carry, amplitudes, efficacy, q_base, warming)``.

    Puts ``efficacy * amplitudes`` of the band patterns in the actuator slot,
    runs ``num_days`` warmed coupled days, and returns ``(carry, fields)``
    with every field stacked as ``(num_days, ix, il)``. One compile per
    segment length; all arguments may change between calls.
    """
    if num_days < 1:
        raise ValueError("num_days must be >= 1")

    def segment(carry, amplitudes, efficacy, q_base, warming):
        stepper = wrap_step_fn_with_warming(step_fn, q_base, warming)
        controlled = apply_band_control(carry, efficacy * amplitudes,
                                        patterns)

        def body(c, step_idx):
            c, _ = stepper(c, step_idx)
            return c, fields_fn(c)

        return lax.scan(body, controlled, jnp.arange(num_days))

    return jax.jit(segment)


class EpisodeState(NamedTuple):
    """What a policy sees at the start of a segment."""

    segment: int
    day: int
    carry: dict
    previous: np.ndarray
    last_fields: Optional[Dict[str, np.ndarray]]


class Episode(NamedTuple):
    """Daily fields ``(n_days, ix, il)``, settings ``(n_segments, K)``, end."""

    fields: Dict[str, np.ndarray]
    amplitudes: np.ndarray
    final_carry: dict


def constant_policy(amplitudes) -> Callable:
    """Return a policy that holds one band setting throughout (a fixed design)."""
    a = np.asarray(amplitudes, np.float32)
    return lambda state: a


def schedule_policy(schedule) -> Callable:
    """Return a policy that plays a precomputed ``(n_segments, K)`` schedule."""
    s = np.asarray(schedule, np.float32)
    return lambda state: s[state.segment]


def run_episode(carry: dict, segment_fn: Callable, policy: Callable,
                n_segments: int, segment_days: int, k_bands: int,
                warming: Warming, efficacy: float = 1.0) -> Episode:
    """Run ``n_segments`` segments, asking ``policy`` before each one.

    ``segment_fn`` must come from ``make_segment_fn`` with ``segment_days``.
    The Q-flux at the start is the base the warming is added to, and the
    final carry gets it back, so no warming is left in a saved state.
    """
    q_base = carry["ocn"]["forcing"].q_flux
    eff = jnp.asarray(efficacy, jnp.float32)
    previous = np.zeros(k_bands, np.float32)
    last, chunks, settings = None, [], []
    for s in range(n_segments):
        state = EpisodeState(segment=s, day=s * segment_days, carry=carry,
                             previous=previous, last_fields=last)
        a = np.asarray(policy(state), np.float32)
        if a.shape != (k_bands,):
            raise ValueError(f"policy returned shape {a.shape}, expected "
                             f"({k_bands},)")
        carry, fields = segment_fn(carry, jnp.asarray(a), eff, q_base,
                                   warming)
        last = {k: np.asarray(v) for k, v in fields.items()}
        chunks.append(last)
        settings.append(a)
        previous = a
    stacked = {k: np.concatenate([c[k] for c in chunks], axis=0)
               for k in chunks[0]}
    return Episode(fields=stacked, amplitudes=np.stack(settings),
                   final_carry=set_qflux(carry, q_base))


# --- Step 11: reference ensembles ----------------------------------------------

def perturb_member(carry: dict, seed: int, amp: float) -> dict:
    """Return a copy of ``carry`` with seeded SST noise of size ``amp`` (K).

    Same draw as ``run_generate_ics_independent.perturb_sst``, so a seed gives
    the same member everywhere in the project.
    """
    state = carry["ocn"]["state"]
    sst = state.sea_surface_temperature
    noise = amp * jax.random.normal(jax.random.PRNGKey(seed), sst.shape)
    new_carry = dict(carry)
    new_carry["ocn"] = dict(carry["ocn"])
    new_carry["ocn"]["state"] = state.copy(
        {"sea_surface_temperature": sst + noise})
    return new_carry


def member_seed(seed0: int, ic_index: int, member: int) -> Optional[int]:
    """Return the seed of member ``m`` of an IC (member 0 is the IC itself)."""
    return None if member == 0 else seed0 + 97 * ic_index + member


def reference_ensemble(carry: dict, segment_fn: Callable, num_days: int,
                       k_bands: int, warming: Warming, members: int,
                       seed0: int, ic_index: int, member_amp: float = 0.001,
                       block_days: int = 5) -> Dict[str, object]:
    """Average ``members`` runs with no brightening (and the given warming).

    ``segment_fn`` must come from ``make_segment_fn`` with ``num_days``.

    Returns:
        A dict with ``mean``, the daily ensemble-mean fields
        ``(num_days, ix, il)``; ``member_blocks``, each member's
        ``block_days`` block means ``(members, num_days // block_days, ix,
        il)``, or None when ``block_days`` is 0; and ``seeds``, the member
        seeds.

    """
    if block_days and num_days % block_days:
        raise ValueError(f"block_days ({block_days}) must divide num_days "
                         f"({num_days})")
    q_base = carry["ocn"]["forcing"].q_flux
    zero = jnp.zeros((k_bands,), jnp.float32)
    one = jnp.asarray(1.0, jnp.float32)
    total, blocks, seeds = None, [], []
    for m in range(members):
        seed = member_seed(seed0, ic_index, m)
        member = carry if seed is None else perturb_member(carry, seed,
                                                           member_amp)
        _, fields = segment_fn(member, zero, one, q_base, warming)
        fields = {k: np.asarray(v, np.float64) for k, v in fields.items()}
        total = fields if total is None else {k: total[k] + fields[k]
                                              for k in total}
        if block_days:
            blocks.append({k: v.reshape((num_days // block_days, block_days)
                                        + v.shape[1:]).mean(axis=1)
                           for k, v in fields.items()})
        seeds.append(seed)
    mean = {k: (v / members).astype(np.float32) for k, v in total.items()}
    member_blocks = ({k: np.stack([b[k] for b in blocks]).astype(np.float32)
                      for k in blocks[0]} if block_days else None)
    return {"mean": mean, "member_blocks": member_blocks, "seeds": seeds}


# --- Step 12: the differentiable look-ahead objective ----------------------

def make_lookahead_sst_fn(step_fn: Callable, patterns,
                          num_days: int, return_final: bool = False) -> Callable:
    """Return the look-ahead's time-mean SST map as a function of the bands.

    ``f(amplitudes, carry, efficacy, q_base, warming, window,
    atm_decay=None)`` holds the amplitudes for ``num_days`` warmed coupled
    days and returns the time-mean SST in kelvin relative to
    ``MAP_REFERENCE_K``, shape ``(ix, il)``. The atmosphere's derivatives
    are cut every ``window`` days counted from the look-ahead's start, as in
    Experiment 1; ``NO_TRUNCATION_DAYS`` gives ordinary backpropagation.
    Forward values do not depend on ``window``. Forward mode (``jax.jacfwd``)
    gives the whole map's response to each band at once, which is what
    Gauss-Newton planning uses. With ``return_final`` it returns
    ``(mean map, final carry)``: a planner that cannot see the weather keeps
    the final carry's atmosphere as its next starting guess (Experiment 3c).
    """
    if num_days < 1:
        raise ValueError("num_days must be >= 1")

    def mean_sst(amplitudes, carry, efficacy, q_base, warming, window,
                 atm_decay=None):
        stepper = wrap_step_fn_with_warming(step_fn, q_base, warming)
        stepper = wrap_step_fn_with_atm_truncation(
            stepper, window, t0_seconds=carry["ocn"]["state"].sim_time,
            atm_decay=atm_decay)
        controlled = apply_band_control(carry, efficacy * amplitudes,
                                        patterns)

        def body(c, step_idx):
            c, _ = stepper(c, step_idx)
            return c, c["ocn"]["state"].sea_surface_temperature - \
                MAP_REFERENCE_K

        final, anomalies = lax.scan(jax.checkpoint(body), controlled,
                                    jnp.arange(num_days))
        mean = jnp.mean(anomalies, axis=0)
        return (mean, final) if return_final else mean

    return mean_sst


def make_lookahead_objective(step_fn: Callable, patterns, weights,
                             num_days: int, alpha: float = PATTERN_ALPHA,
                             beta: float = PATTERN_BETA, mu: float = 0.0,
                             lam: float = 0.0) -> Callable:
    """Return the map objective a planner minimizes over a ``num_days`` look-ahead.

    The returned function is
    ``J(amplitudes, carry, target_sst_mean, previous, efficacy, q_base,
    warming, window, atm_decay=None)``. The amplitudes are held for the whole
    look-ahead, and the error map is the look-ahead's time-mean SST
    (``make_lookahead_sst_fn``) minus the target's time-mean SST over the same
    days (the normal climate, in kelvin).
    """
    mean_sst = make_lookahead_sst_fn(step_fn, patterns, num_days)

    def objective(amplitudes, carry, target_sst_mean, previous, efficacy,
                  q_base, warming, window, atm_decay=None):
        error = (mean_sst(amplitudes, carry, efficacy, q_base, warming,
                          window, atm_decay)
                 - (target_sst_mean - MAP_REFERENCE_K))
        return segment_objective(error, weights, amplitudes, previous, alpha,
                                 beta, mu, lam)

    return objective
