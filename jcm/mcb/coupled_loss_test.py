"""Tests for ocean-masked coupled loss (Stage 5 realistic terrain).

Verifies that:
  - With an ocean_mask, the SST cooling and uniformity losses are invariant
    to arbitrary values on land cells (land is pinned to a fixed temperature
    on realistic terrain and must not corrupt the objective).
  - ocean_mask=None reproduces the current aquaplanet behavior exactly.
"""

import unittest
from typing import NamedTuple

import jax.numpy as jnp

from jcm.mcb.coupled_loss import compute_coupled_loss, CoupledLossWeights
from jcm.physics.speedy.speedy_coords import get_speedy_coords


class _SurfaceFlux(NamedTuple):
    tsfc: jnp.ndarray


class _Convection(NamedTuple):
    precnv: jnp.ndarray


class _Condensation(NamedTuple):
    precls: jnp.ndarray


class _Physics(NamedTuple):
    surface_flux: _SurfaceFlux
    convection: _Convection
    condensation: _Condensation


class _OceanState(NamedTuple):
    sea_surface_temperature: jnp.ndarray


def make_fake_carry(coords, sst_field):
    """Build a minimal coupled carry compatible with compute_coupled_loss."""
    shape = coords.horizontal.nodal_shape
    return {
        "atm": {
            "derived": {
                "physics": _Physics(
                    surface_flux=_SurfaceFlux(tsfc=jnp.full(shape, 285.0)),
                    convection=_Convection(precnv=jnp.zeros(shape)),
                    condensation=_Condensation(precls=jnp.zeros(shape)),
                ),
                "total_heat_flux": jnp.zeros(shape),
                "mcb_perturbation": jnp.zeros(shape),
            }
        },
        "ocn": {"state": _OceanState(sea_surface_temperature=sst_field)},
    }


class TestOceanMaskedLoss(unittest.TestCase):
    """Ocean masking invariance and backwards compatibility."""

    @classmethod
    def setUpClass(cls):
        cls.coords = get_speedy_coords()
        cls.shape = cls.coords.horizontal.nodal_shape
        cls.mcb = jnp.zeros(cls.shape)
        # SST-only weights so we isolate the masked cooling/uniformity terms.
        cls.weights = CoupledLossWeights(
            sst_cooling=1.0, sst_uniformity=0.1,
            amazon=0.0, sahel=0.0, tropics=0.0,
            regularization=0.0, smoothness=0.0,
        )
        # A synthetic land mask: mark ~half the longitudes as land.
        ix, il = cls.shape
        land = jnp.zeros(cls.shape).at[: ix // 2, :].set(1.0)
        cls.ocean_mask = 1.0 - land
        cls.land = land

    def _loss(self, sst, baseline_sst, ocean_mask):
        carry = make_fake_carry(self.coords, sst)
        return float(compute_coupled_loss(
            coupled_carry=carry,
            baseline_sst=baseline_sst,
            baseline_precip=jnp.zeros(self.shape),
            target_cooling=-0.1,
            mcb_forcing=self.mcb,
            coords=self.coords,
            weights=self.weights,
            ocean_mask=ocean_mask,
        ))

    def test_invariant_to_land_values(self):
        """Ocean-masked loss is unchanged by arbitrary land SST values."""
        key_ocean = 288.0 - 0.1 * self.ocean_mask  # cool the ocean uniformly
        baseline = jnp.full(self.shape, 288.0)

        # Two SST fields identical over ocean but wildly different over land.
        sst_a = key_ocean + 50.0 * self.land        # land = 338 K
        sst_b = key_ocean - 100.0 * self.land       # land = 188 K

        loss_a = self._loss(sst_a, baseline, self.ocean_mask)
        loss_b = self._loss(sst_b, baseline, self.ocean_mask)
        self.assertAlmostEqual(loss_a, loss_b, places=6)

    def test_equals_all_ocean_result(self):
        """Masked loss equals the loss on a field that is ocean everywhere."""
        ocean_sst = jnp.full(self.shape, 287.9)  # uniform 0.1 K cooling
        baseline = jnp.full(self.shape, 288.0)

        # Field with garbage on land, correct on ocean.
        mixed = jnp.where(self.ocean_mask > 0, ocean_sst, 999.0)

        masked = self._loss(mixed, baseline, self.ocean_mask)
        all_ocean = self._loss(ocean_sst, baseline, self.ocean_mask)
        self.assertAlmostEqual(masked, all_ocean, places=6)

    def test_none_reproduces_unmasked(self):
        """ocean_mask=None matches an all-ones mask (aquaplanet behavior).

        The two paths are mathematically identical but differ at float32
        precision: an all-ones mask renormalizes the area weights (which sum
        to 0.9999994 in float32, not exactly 1), and the cooling term is a
        difference of two ~288 K globals, so the tiny weight-sum discrepancy
        is amplified. places=4 covers this ~3e-5 renormalization noise.
        """
        sst = jnp.full(self.shape, 287.85)
        baseline = jnp.full(self.shape, 288.0)
        ones = jnp.ones(self.shape)

        loss_none = self._loss(sst, baseline, None)
        loss_ones = self._loss(sst, baseline, ones)
        self.assertAlmostEqual(loss_none, loss_ones, places=4)

    def test_none_default_uniform_cooling(self):
        """Sanity: uniform 0.1 K cooling hits the target -> tiny cooling loss."""
        sst = jnp.full(self.shape, 287.9)
        baseline = jnp.full(self.shape, 288.0)
        loss = self._loss(sst, baseline, None)
        # actual cooling = -0.1 == target -> cooling term ~ 0; uniformity ~ 0
        self.assertLess(loss, 1e-6)


if __name__ == "__main__":
    unittest.main()
