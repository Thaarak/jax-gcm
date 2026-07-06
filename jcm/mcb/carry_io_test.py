"""Tests for coupled carry serialization (leaves-only round trip)."""

import tempfile
import unittest
from pathlib import Path

import jax.numpy as jnp

from jcm.mcb.carry_io import load_carry, save_carry
from jcm.mcb.coupled_features import CoupledBaseline


def make_carry(offset=0.0):
    """Nested pytree with a tree_math struct, mimicking a coupled carry."""
    return {
        "atm": {
            "derived": {
                "mcb_perturbation": jnp.full((4, 3), 0.1 + offset),
                "baseline": CoupledBaseline(
                    sst=jnp.full((4, 3), 288.0 + offset),
                    surface_temperature=jnp.full((4, 3), 285.0 + offset),
                    precipitation=jnp.zeros((4, 3)) + offset,
                    heat_flux=jnp.ones((4, 3)) * offset,
                ),
            }
        },
        "ocn": {"sst": jnp.linspace(280.0, 300.0, 12).reshape(4, 3) + offset},
    }


class TestCarryIO(unittest.TestCase):
    """Round-trip and error behavior of save_carry/load_carry."""

    def test_round_trip_exactness(self):
        """Leaves survive a save/load round trip bit-exactly."""
        carry = make_carry(offset=1.5)
        template = make_carry(offset=0.0)  # same structure, different values

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "carry.pkl")
            save_carry(carry, path)
            loaded = load_carry(path, template)

        import jax
        orig_leaves = jax.tree_util.tree_leaves(carry)
        loaded_leaves = jax.tree_util.tree_leaves(loaded)
        self.assertEqual(len(orig_leaves), len(loaded_leaves))
        for a, b in zip(orig_leaves, loaded_leaves):
            self.assertTrue(jnp.array_equal(a, b))

        # Structure comes from the template
        self.assertTrue(jnp.array_equal(
            loaded["ocn"]["sst"], carry["ocn"]["sst"]
        ))
        self.assertTrue(jnp.array_equal(
            loaded["atm"]["derived"]["baseline"].sst,
            carry["atm"]["derived"]["baseline"].sst,
        ))

    def test_leaf_count_mismatch_raises(self):
        """A template with a different structure raises ValueError."""
        carry = make_carry()
        bad_template = {"only": jnp.zeros(3)}

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "carry.pkl")
            save_carry(carry, path)
            with self.assertRaises(ValueError):
                load_carry(path, bad_template)


if __name__ == "__main__":
    unittest.main()
