"""JCM adapter to JEM"""

import numpy as np

from jcm.model import Model
from jcm.forcing import ForcingData, default_forcing

import jax
import jax.numpy as jnp
import jax_datetime as jdt


def safe_setattr(target, attribute_name, value, *, raise_exception=True):
    if hasattr(target, attribute_name):
        message = f"Attribute name `{attribute_name:s}` already exists."
        if raise_exception:
            raise Exception(message)
        else:
            print(f"Warning: {message:s}")
    
    setattr(target, attribute_name, value)

# This is a temporary solution to jcm's problem: some of the array's initiated
# by jcm is int32, but it will change to float32 after step_function. This causes
# jax.lax.scan to fail due to data type inconsistency.
def asfloat64(tree):
    return jax.tree_util.tree_map(lambda arr: jnp.array(arr).astype(jnp.float64), tree)

def make_jem_compatible(
    model: Model,
    coupling_timestep: jdt.Timedelta,
    forcing: ForcingData = None,
) -> Model:
    """Adapt the input jcm model to jem framework

    This function in-place injects `initialize`, `generate_step_function`,
    `predictions_to_xarray`, and `get_info` into jcm model object. Also, check
    if jcm's time step `dt_si` can perfectly divide `coupling_timestep`.

    Args:
        model: The jcm Model to adapt.
        coupling_timestep: Coupler timestep (must be a multiple of model.dt_si).
        forcing: Optional surface boundary conditions (ForcingData). When None
            (default), aquaplanet `default_forcing` is used (prescribed SST,
            zero land-surface fields) — the numerically stable configuration
            over flat orography. Over REALISTIC terrain the land-surface fields
            (stl_am, soilw_am, snowc_am, alb0) must be supplied via a
            `ForcingData.from_file(...)`; without them SPEEDY land physics over
            steep orography is unbalanced and the atmosphere blows up (NaN
            within ~1 day). The coupler still overrides SST every step from the
            ocean, so only the land-surface fields (and initial SST) come from
            this argument.

    """
    # Check if couopling_timestep is a multiple of jcm's native timestep
    timestep = jdt.to_timedelta(int(model.dt_si.to_timedelta().total_seconds()), "second")
    if timestep * np.floor(coupling_timestep / timestep) != coupling_timestep:
        raise Exception("Coupling timestep should be a multiple of timestep.")

    # Pre-compute these values outside closures to avoid JIT tracing issues
    # when these closures are called inside JIT-compiled functions
    _timestep_days = float((timestep / jdt.to_timedelta(1, "day")).item())
    _coupling_timestep_days = float((coupling_timestep / jdt.to_timedelta(1, "day")).item())

    # Resolve surface forcing once: realistic land-surface boundary conditions
    # if provided, else aquaplanet default (prescribed SST, zero land fields).
    _forcing = forcing if forcing is not None else default_forcing(
        model.coords.horizontal
    )

    D2_nodal_shape = model.coords.nodal_shape[1:]
    def initialize():

        state=model._prepare_initial_modal_state()
        forcing = _forcing

        # Predictions shape is still morphing in the development.
        # Use run_from_state to get the shape of predictions. This might
        # cost a few second extra but will be resilience to major code
        # update in jcm
        _, predictions = model.run_from_state(
            initial_state=state,
            save_interval=_timestep_days,
            total_time=_timestep_days,
            forcing=forcing,
            output_averages=True,
        )
        physics_no_time_dimension = jax.tree.map(lambda x: x[0], predictions.physics)

        return asfloat64(dict(
            state=state,
            derived={ # Derived
                "physics" : physics_no_time_dimension,
                "total_heat_flux" : jnp.zeros(D2_nodal_shape),   # SEA slab -> ocean
                "land_heat_flux" : jnp.zeros(D2_nodal_shape),    # LAND slab -> slab land model
                "total_freshwater_flux" : jnp.zeros(D2_nodal_shape),
                "mcb_perturbation" : jnp.zeros(D2_nodal_shape),  # For coupled MCB training
            },
            forcing=forcing,
        ))

    def generate_step_function():
        # Use pre-computed values to avoid JIT tracing issues
        def step_function(carry, step):
            state = carry["state"]
            forcing = asfloat64(carry["forcing"])

            # Dynamic MCB perturbation from coupled controller: inject it into
            # the ForcingData (a traced argument of run_from_state) so it is
            # visible to JIT and differentiable. NEVER via attribute mutation:
            # run_from_state is jitted with static self, so mutated model
            # fields are silently ignored after the first trace.
            mcb_perturbation = carry.get("derived", {}).get("mcb_perturbation", None)
            if mcb_perturbation is not None:
                forcing = forcing.copy(mcb_perturbation=mcb_perturbation)

            new_atm_modal_state, predictions = model.run_from_state(
                initial_state=state,
                save_interval=_coupling_timestep_days,
                total_time=_coupling_timestep_days,
                forcing=forcing,
                output_averages=True,
            )
            physics_no_time_dimension = jax.tree.map(lambda x: x[0], predictions.physics)
            # hfluxn/evap have a trailing slab axis: [..., 0] = LAND slab,
            # [..., 1] = SEA slab (see surface_flux.py:187 vs :260). The ocean
            # mixed layer receives the SEA-surface flux only. Summing over axis=2
            # adds the land slab (computed from a ~272 K land skin temp over ocean
            # cells) and injects ~100 W/m^2 RMS of spurious flux — silent on the
            # aquaplanet (land slab ≈ 0), but corrupts every realistic-terrain run.
            total_heat_flux = - physics_no_time_dimension.surface_flux.hfluxn[..., 1] # SEA slab, upward positive -> ocean
            land_heat_flux = - physics_no_time_dimension.surface_flux.hfluxn[..., 0]  # LAND slab, upward positive -> slab land model
            evaporation = physics_no_time_dimension.surface_flux.evap[..., 1] # upward positive, sea slab

            total_freshwater_flux = (
                evaporation
                 - physics_no_time_dimension.convection.precnv
                 - physics_no_time_dimension.condensation.precls
            ) / 1000.0 # The number 1000.0 is the convert factor of mass density flux of freshwater from g/m^2/s to kg/m^2/s

            return (
                asfloat64(dict(
                    state=new_atm_modal_state,
                    derived={
                        "physics" : physics_no_time_dimension,
                        "total_heat_flux" : total_heat_flux,
                        "land_heat_flux" : land_heat_flux,
                        "total_freshwater_flux" : total_freshwater_flux,
                        "mcb_perturbation" : mcb_perturbation,  # Preserve for JAX scan
                    },
                    forcing=forcing,
                )),
                predictions
            )

        return step_function

    def predictions_to_xarray(predictions):
        return predictions.to_xarray()

    def get_info():
        return {
            "diffusion" : str(model.diffusion),
        }

    safe_setattr(model, "initialize", initialize)
    safe_setattr(model, "predictions_to_xarray", predictions_to_xarray)
    safe_setattr(model, "generate_step_function", generate_step_function)
    safe_setattr(model, "get_info", get_info)

    return model
