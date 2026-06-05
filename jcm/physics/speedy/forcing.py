from jcm.terrain import TerrainData
from jcm.physics.speedy.params import Parameters
from jcm.physics.speedy.physics_data import ablco2_ref, PhysicsData
from jcm.forcing import ForcingData
from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.physics.speedy.shortwave_radiation import get_zonal_average_fields
import jax.numpy as jnp
# linear trend of co2 absorptivity (del_co2: rate of change per year)
del_co2   = 0.005

def set_forcing(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    forcing: ForcingData=None,
    terrain: TerrainData=None,
    mcb_config=None,
    mcb_perturbation=None,
) -> tuple[PhysicsTendency, PhysicsData]:
    # 2. daily-mean radiative forcing
    physics_data = get_zonal_average_fields(state, physics_data, forcing=forcing, terrain=terrain)
    tyear = physics_data.date.tyear
    model_year = physics_data.date.model_year

    # total surface albedo
    fmask = terrain.fmask

    snowc = jnp.minimum(1.0, forcing.snowc_am)
    alb_l = forcing.alb0 + snowc * (parameters.mod_radcon.albsn - forcing.alb0)
    alb_s = parameters.mod_radcon.albsea + forcing.sice_am * (parameters.mod_radcon.albice - parameters.mod_radcon.albsea)

    # Apply MCB forcing if configured
    if mcb_config is not None:
        from jcm.mcb.mcb_forcing import compute_mcb_sea_albedo
        day_of_year = physics_data.date.model_day()
        mcb_albedo = compute_mcb_sea_albedo(
            mcb_config, parameters.mod_radcon.albsea, day_of_year
        )
        # Only apply MCB over ocean (where fmask == 0)
        ocean_mask = 1.0 - fmask
        alb_s = alb_s + ocean_mask * mcb_config.active_mask * (
            mcb_albedo - parameters.mod_radcon.albsea
        )
    elif mcb_perturbation is not None:
        # Dynamic MCB perturbation from coupled controller
        # mcb_perturbation is already masked to ocean cells
        ocean_mask = 1.0 - fmask
        alb_s = alb_s + ocean_mask * mcb_perturbation

    albsfc = alb_s + fmask * (alb_l - alb_s)

    iyear_ref = parameters.forcing.co2_year_ref
    ablco2 = ablco2_ref * jnp.exp(parameters.forcing.increase_co2 * del_co2 * (model_year + tyear - iyear_ref))

    mod_radcon = physics_data.mod_radcon.copy(snowc=snowc, alb_l=alb_l, alb_s=alb_s, albsfc=albsfc, ablco2=ablco2)

    physics_data = physics_data.copy(mod_radcon=mod_radcon)
    physics_tendencies = PhysicsTendency.zeros(state.temperature.shape)

    return physics_tendencies, physics_data
