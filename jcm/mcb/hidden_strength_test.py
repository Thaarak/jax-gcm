"""Tests for the hidden spraying strength of Experiment 3b."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from jcm.mcb import hidden_strength as hs


class HiddenStrengthTest(unittest.TestCase):
    def test_every_evaluation_ocean_state_sees_all_three_factors(self):
        for macro in range(17, 48, 2):
            got = sorted(hs.overall_factor(macro, b) for b in (0, 1, 2))
            self.assertEqual(got, [0.5, 1.0, 2.0])

    def test_regional_factors_have_geometric_mean_one(self):
        for idx in (1600, 1701, 4702):
            r = hs.regional_factors(idx)
            self.assertEqual(r.shape, (5,))
            self.assertAlmostEqual(float(np.exp(np.mean(np.log(r)))), 1.0)
            self.assertTrue(np.all(r > 0))

    def test_draws_are_reproducible_and_differ_between_states(self):
        np.testing.assert_array_equal(hs.hidden_strength(17, 0),
                                      hs.hidden_strength(17, 0))
        self.assertFalse(np.allclose(hs.hidden_strength(17, 0),
                                     hs.hidden_strength(17, 1)
                                     / hs.overall_factor(17, 1)
                                     * hs.overall_factor(17, 0)))

    def test_spread_matches_the_registered_sigma(self):
        logs = np.array([np.log(hs.regional_factors(i))
                         for i in range(1600, 1600 + 4000)])
        # Re-centring five draws shrinks the variance by 4/5.
        self.assertAlmostEqual(float(logs.std()), 0.5 * np.sqrt(0.8), delta=0.01)

    def test_table_from_a_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            man = Path(tmp) / "manifest.json"
            man.write_text(json.dumps({"ics": [
                {"macro_index": 17, "branch": 0, "index": 1700},
                {"macro_index": 17, "branch": 2, "index": 1702}]}))
            out = Path(tmp) / "t.json"
            hs.main(["--manifests", str(man), "--output", str(out)])
            table = json.loads(out.read_text())
        self.assertEqual(sorted(table["states"]), ["1700", "1702"])
        row = table["states"]["1702"]
        self.assertEqual(row["overall"], hs.overall_factor(17, 2))
        np.testing.assert_allclose(row["strength"], hs.hidden_strength(17, 2),
                                   atol=1e-6)


if __name__ == "__main__":
    unittest.main()
