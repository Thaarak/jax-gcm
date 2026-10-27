"""Tests for Deviation D2 of Experiment 3f's analysis (analyze_experiment3f_d2.py)."""

import importlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

import analyze_experiment3f as ex
from analyze_experiment3f_test import _args, _worlds

REGISTERED_LIMIT = 0.05            # analyze_experiment3f.MAX_OVERLAP_K as frozen
REGISTERED_CHECK = ex.check_runs   # this module imports the D2 file only in tests


def _extensions(tmp: Path, runs_3f: Path, refs, overlap: float, offset: float,
                noise: float = 0.0):
    """Give every long planner's run a target-extension file and its record.

    The extension repeats the references plus ``offset`` (a climate shift)
    and ``noise`` (weather), then runs 80 days further.
    """
    rng = np.random.default_rng(7)
    out = tmp / "ext"
    out.mkdir(exist_ok=True)
    for arm in ex.LONG:
        for js in sorted((runs_3f / arm).glob("ic*.json")):
            s = json.loads(js.read_text())
            name = js.name.replace(".json", "_references.npz")
            ref = next(Path(d) / name for d in refs if (Path(d) / name).exists())
            own = np.load(ref)["normal_sst"]
            ext = out / name
            if not ext.exists():
                longer = own + offset + noise * rng.normal(size=own.shape)
                np.savez(ext, normal_sst=np.concatenate([longer, longer[-80:]]))
            s["references"] = str(ref)
            s["target_extension"] = {"file": str(ext), "own_days": int(own.shape[0]),
                                     "extension_days": 80,
                                     "overlap_mean_abs_k": overlap}
            js.write_text(json.dumps(s))


def _apply_d2():
    ex.MAX_OVERLAP_K, ex.check_runs = REGISTERED_LIMIT, REGISTERED_CHECK
    import analyze_experiment3f_d2 as d2
    return importlib.reload(d2)


class DeviationD2Test(unittest.TestCase):
    def tearDown(self):
        ex.MAX_OVERLAP_K, ex.check_runs = REGISTERED_LIMIT, REGISTERED_CHECK

    def _run(self, overlap, offset, noise=0.0, d2=True):
        if d2:
            _apply_d2()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            runs_3f, runs_3d, refs = _worlds(tmp)
            _extensions(tmp, runs_3f, refs, overlap, offset, noise)
            with redirect_stdout(io.StringIO()):
                ex.main(_args(tmp, runs_3f, runs_3d, refs))
            return json.loads((tmp / "a.json").read_text())

    def test_the_registered_limit_refuses_weather_noise(self):
        with self.assertRaises(SystemExit):
            self._run(overlap=0.13, offset=0.0, noise=0.1, d2=False)

    def test_d2_accepts_weather_noise_and_records_its_limits(self):
        result = self._run(overlap=0.13, offset=0.0, noise=0.1)
        self.assertEqual(result["registered"]["max_overlap_k"], 0.20)
        checks = result["checks"]
        self.assertAlmostEqual(checks["target_overlap_mean_abs_k_max"], 0.13)
        self.assertEqual((checks["d2_max_overlap_k"], checks["d2_max_signed_k"]),
                         (0.20, 0.05))
        self.assertLess(checks["d2_target_signed_k_max_abs"], 0.05)
        self.assertEqual(result["hypotheses"]["H1"]["verdict"], "better")

    def test_d2_refuses_another_climate_by_its_signed_difference(self):
        # A warmed target: +0.17 K, under the 0.20 K limit on the recorded
        # mean absolute difference, but not the same climate.
        with self.assertRaises(SystemExit):
            self._run(overlap=0.18, offset=0.17)

    def test_d2_refuses_a_recorded_difference_over_its_limit(self):
        with self.assertRaises(SystemExit):
            self._run(overlap=0.3, offset=0.0)

    def test_loading_d2_twice_wraps_the_registered_check_once(self):
        _apply_d2()
        import analyze_experiment3f_d2 as d2
        importlib.reload(d2)
        self.assertIs(ex.check_runs.__wrapped__, REGISTERED_CHECK)


if __name__ == "__main__":
    unittest.main()
