import jax
import jax.numpy as jnp
from collections import abc
from typing import Callable, Tuple
from pathlib import Path
from dinosaur.coordinate_systems import CoordinateSystem
from jcm.physics_interface import PhysicsState, PhysicsTendency, Physics
from jcm.physics.speedy.physics_data import PhysicsData
from jcm.forcing import ForcingData
from jcm.physics.speedy.params import Parameters
from jcm.physics.speedy.speedy_coords import SpeedyCoords
from jcm.terrain import TerrainData
from jcm.date import DateData
from jcm.utils import tree_index_3d

def set_physics_flags(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    forcing: ForcingData=None,
    terrain: TerrainData=None
) -> tuple[PhysicsTendency, PhysicsData]:
    from jcm.physics.speedy.physical_constants import nstrad
    '''
    Sets flags that indicate whether a tendency function should be run.
    clouds, get_shortwave_rad_fluxes are the only functions that currently depend on this. 
    This could also apply to forcing and coupling.
    '''
    model_step = physics_data.date.model_step
    compute_shortwave = (jnp.mod(model_step, nstrad) == 0)
    shortwave_data = physics_data.shortwave_rad.copy(compute_shortwave=compute_shortwave)
    physics_data = physics_data.copy(shortwave_rad=shortwave_data)

    physics_tendencies = PhysicsTendency.zeros(state.temperature.shape)
    return physics_tendencies, physics_data

class SpeedyPhysics(Physics):
    """A set of intermediate complexity atmospheric physics parameterizations from the SPEEDY model.

    Forcing data should be either simple climatological fields (assuming a 365 day year), or constant.
    Many of the parameterizations assume 8 model levels and a specific vertical coordinate system.
    """

    parameters: Parameters
    coords: CoordinateSystem
    terms: abc.Sequence[Callable[[PhysicsState], PhysicsTendency]]
    UNITS_TABLE_CSV_PATH = Path(__file__).parent / "units_table.csv"

    def __init__(self,
                 parameters: Parameters=Parameters.default(),
                 mcb_config=None,
                 checkpoint_terms=True,
                 allow_legacy_surface_albedo_mcb=False,
    ) -> None:
        """Initialize the SpeedyPhysics class with the specified parameters.

        Args:
            parameters (Parameters): Parameters for the physics model.
            mcb_config: Optional MCBConfig for Marine Cloud Brightening forcing.
                If provided, MCB albedo perturbations will be applied in set_forcing.
                DEPRECATED: this path perturbs the SEA-SURFACE albedo, which is
                anti-correlated with cloud cover — the opposite of the Twomey
                effect (2026-07-12 audit root cause R6). Requires the explicit
                opt-in flag below; use the dynamic cloud-albedo path
                (forcing.mcb_perturbation via jcm/mcb) for real MCB experiments.
            checkpoint_terms (bool): Flag to indicate if terms should be checkpointed.
            allow_legacy_surface_albedo_mcb (bool): Explicit opt-in to the
                deprecated surface-albedo mcb_config path.

        """
        if mcb_config is not None and not allow_legacy_surface_albedo_mcb:
            raise ValueError(
                "mcb_config uses the DEPRECATED surface-albedo MCB path: it "
                "brightens the sea surface, not clouds, so its effect is "
                "anti-correlated with cloud cover — the opposite of the "
                "Twomey effect (audit root cause R6). Use the dynamic "
                "cloud-albedo path (forcing.mcb_perturbation, see jcm/mcb) "
                "instead, or pass allow_legacy_surface_albedo_mcb=True if you "
                "really intend to run the legacy surface-albedo experiment."
            )
        self.parameters = parameters
        self.mcb_config = mcb_config

        from jcm.physics.speedy.humidity import spec_hum_to_rel_hum
        from jcm.physics.speedy.convection import get_convection_tendencies
        from jcm.physics.speedy.large_scale_condensation import get_large_scale_condensation_tendencies
        from jcm.physics.speedy.shortwave_radiation import get_shortwave_rad_fluxes, get_clouds
        from jcm.physics.speedy.longwave_radiation import get_downward_longwave_rad_fluxes, get_upward_longwave_rad_fluxes
        from jcm.physics.speedy.surface_flux import get_surface_fluxes
        from jcm.physics.speedy.vertical_diffusion import get_vertical_diffusion_tend
        from jcm.physics.speedy.forcing import set_forcing
        # from jcm.physics.speedy.orographic_correction import get_orographic_correction_tendencies

        # Wrapper binding the static mcb_config. Dynamic (trainable) MCB
        # perturbations flow through forcing.mcb_perturbation instead, which is
        # a traced argument and therefore visible to JIT and autodiff.
        def create_set_forcing_wrapper(physics_instance):
            def set_forcing_wrapper(state, data, params, forcing, terrain):
                return set_forcing(
                    state, data, params, forcing, terrain,
                    mcb_config=physics_instance.mcb_config,
                )
            return set_forcing_wrapper

        set_forcing_term = create_set_forcing_wrapper(self)

        physics_terms = [
            set_physics_flags,
            set_forcing_term,
            spec_hum_to_rel_hum,
            get_convection_tendencies,
            get_large_scale_condensation_tendencies,
            get_clouds,
            get_shortwave_rad_fluxes,
            get_downward_longwave_rad_fluxes,
            get_surface_fluxes,
            get_upward_longwave_rad_fluxes,
            get_vertical_diffusion_tend,
            # get_orographic_correction_tendencies # orographic corrections applied last
        ]

        static_argnums = {
            set_forcing_term: (2,),
        }

        self.terms = physics_terms if not checkpoint_terms else [jax.checkpoint(term, static_argnums=static_argnums.get(term, ()) + (4,)) for term in physics_terms]
    
    def cache_coords(self, coords: CoordinateSystem):
        """Store model coordinate system for SpeedyCoords calculation in compute_tendencies"""
        self.model_coords = coords
        self.cached_coords = SpeedyCoords.from_coordinate_system(coords)
        return 
    
    def compute_tendencies(
        self,
        state: PhysicsState,
        forcing: ForcingData,
        terrain: TerrainData,
        date: DateData,
    ) -> Tuple[PhysicsTendency, PhysicsData]:
        """Compute the physical tendencies given the current state and data structs. Loops through the Speedy physics terms, accumulating the tendencies.

        Args:
            state: Current state variables
            forcing: Forcing data
            terrain: Terrain data
            date: Date data

        Returns:
            Physical tendencies in PhysicsTendency format
            Object containing physics data (PhysicsData format)

        """
        # Initialize physics data with speedy_coords cached
        data = PhysicsData.zeros(
            self.model_coords.horizontal.nodal_shape,
            self.model_coords.nodal_shape[0],
            date=date,
            speedy_coords=self.cached_coords
        )

        # the 'physics_terms' return an instance of tendencies and data, data gets overwritten at each step
        # and implicitly passed to the next physics_term. tendencies are summed
        physics_tendency = PhysicsTendency.zeros(shape=state.u_wind.shape)

        # Slice out the relevant day of the year for time-varying forcings
        model_day_of_year = date.model_day()
        forcing_2d = tree_index_3d(forcing, model_day_of_year)

        for term in self.terms:
            tend, data = term(state, data, self.parameters, forcing_2d, terrain)
            physics_tendency += tend

        return physics_tendency, data

    def get_empty_data(self, coords: CoordinateSystem) -> PhysicsData:
        from jax.tree_util import tree_map
        # PhysicsData.zeros creates an 'initial' physics data,
        # but we need a completely zeroed one (including fields like model_year) for accumulating averages
        # Compute speedy_coords for the empty data (it's a constant cache, not zeroed)
        speedy_coords = SpeedyCoords.from_coordinate_system(coords)
        empty_data = PhysicsData.zeros(coords.horizontal.nodal_shape, coords.nodal_shape[0], speedy_coords=speedy_coords)
        # Zero out everything except speedy_coords (which should remain constant)
        return tree_map(lambda x: 0*x, empty_data).copy(speedy_coords=speedy_coords)

