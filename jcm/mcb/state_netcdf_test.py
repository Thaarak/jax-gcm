"""Tests for the NetCDF state format (jcm.mcb.state_netcdf)."""

import tempfile
import unittest
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from jcm.mcb.state_netcdf import (
    flatten_state,
    load_leaves,
    load_state,
    save_state,
    states_equal,
)
from jcm.mcb.test_world_test import _carry


def _mixed_state():
    """Return a carry plus the leaf kinds a real state may hold."""
    c = _carry()
    c["extra"] = {"complex": jnp.asarray([1 + 2j, -3j], jnp.complex64),
                  "flag": np.array([True, False]),
                  "scalar": jnp.asarray(7.5, jnp.float32),
                  "count": np.int32(3),
                  "wide": np.arange(6, dtype=np.float64).reshape(2, 3)}
    return c


class RoundTripTest(unittest.TestCase):
    def test_every_leaf_comes_back_exactly(self):
        state = _mixed_state()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.nc"
            save_state(path, state, {"role": "exp2_train", "index": 1,
                                     "source_sha256": "abc"})
            back = load_state(path, state)
            _, _, attrs = load_leaves(path)
        self.assertTrue(states_equal(state, back))
        self.assertEqual(attrs["role"], "exp2_train")
        self.assertEqual(int(attrs["index"]), 1)
        paths, _ = flatten_state(state)
        self.assertIn("['extra']['complex']", paths)

    def test_a_different_structure_is_refused(self):
        state = _mixed_state()
        other = _mixed_state()
        other["extra"]["new"] = np.zeros(2)
        reshaped = _mixed_state()
        reshaped["extra"]["wide"] = np.zeros((3, 2))
        retyped = _mixed_state()
        retyped["extra"]["count"] = np.int64(0)       # a template's int
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.nc"
            save_state(path, state)
            with self.assertRaises(ValueError):
                load_state(path, other)
            with self.assertRaises(ValueError):
                load_state(path, reshaped)
            # The file's own dtypes win over a template's.
            back = load_state(path, retyped)
        self.assertTrue(states_equal(state, back))

    def test_states_equal_detects_changes(self):
        a = _mixed_state()
        b = _mixed_state()
        self.assertTrue(states_equal(a, b))
        b["extra"]["wide"] = b["extra"]["wide"] + 1e-12
        self.assertFalse(states_equal(a, b))


if __name__ == "__main__":
    unittest.main()
