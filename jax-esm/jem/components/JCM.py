"""JCM adapter to JEM"""

import numpy as np

from jcm.model import Model
from jcm.forcing import default_forcing

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
) -> Model:
    """Adapt the input jcm model to jem framework
    
    This function in-place injects `initialize`, `generate_step_function`, 
    `predictions_to_xarray`, and `get_info` into jcm model object. Also, check
    if jcm's time step `dt_si` can perfectly divide `coupling_timestep`.
    
    """    
   
    # Check if couopling_timestep is a multiple of jcm's native timestep
    timestep = jdt.to_timedelta(int(model.dt_si.to_timedelta().total_seconds()), "second")
    if timestep * np.floor(coupling_timestep / timestep) != coupling_timestep:
        raise Exception("Coupling timestep should be a multiple of timestep.")

    # Pre-compute these values outside closures to avoid JIT tracing issues
    # when these closures are called inside JIT-compiled functions
    _timestep_days = float((timestep / jdt.to_timedelta(1, "day")).item())
    _coupling_timestep_days = float((coupling_timestep / jdt.to_timedelta(1, "day")).item())

    D2_nodal_shape = model.coords.nodal_shape[1:]
    def initialize():

        state=model._prepare_initial_modal_state()
        forcing = default_forcing(model.coords.horizontal)
        
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
                "total_heat_flux" : jnp.zeros(D2_nodal_shape),
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

            # Check for dynamic MCB perturbation from coupled controller
            mcb_perturbation = carry.get("derived", {}).get("mcb_perturbation", None)
            if mcb_perturbation is not None:
                model.physics.set_mcb_perturbation(mcb_perturbation)
            else:
                model.physics.clear_mcb_perturbation()

            new_atm_modal_state, predictions = model.run_from_state(
                initial_state=state,
                save_interval=_coupling_timestep_days,
                total_time=_coupling_timestep_days,
                forcing=forcing,
                output_averages=True,
            )
            physics_no_time_dimension = jax.tree.map(lambda x: x[0], predictions.physics)
            total_heat_flux = - jnp.sum(physics_no_time_dimension.surface_flux.hfluxn, axis=2) # convert to upward positive
            evaporation = jnp.sum(physics_no_time_dimension.surface_flux.evap, axis=2) # upward positive

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
