"""One tiny H5OINA acquisition exercises the native-to-Python export path."""
import hashlib
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
from zebsdsolve import Solver


class OxfordExport(unittest.TestCase):
    def test_roi_export(self):
        with tempfile.TemporaryDirectory() as folder, Solver() as solver:
            source, output = (Path(folder) / name for name in ("source.h5oina", "output.h5oina"))
            with h5py.File(source, "w") as f:
                f["Manufacturer"] = "Oxford Instruments"
                f["Format Version"] = "8.0"
                header = f.create_group("1/EBSD/Header")
                data = f.create_group("1/EBSD/Data")
                for key, value, unit in [
                    ("X Cells", 3, "px"), ("Y Cells", 2, "px"),
                    ("X Step", .2, "um"), ("Y Step", .25, "um"),
                    ("Pattern Width", 40, "px"), ("Pattern Height", 32, "px"),
                    ("Hit Rate", 100., "%"),
                    ("Specimen Orientation Euler", [.2, .1, .3], "rad"),
                    ("Detector Orientation Euler", [.01, 1.5, .02], "rad"),
                    ("Scanning Rotation Angle", np.pi, "rad"),
                    ("Tilt Axis", 0., "rad"), ("Tilt Angle", -.35, "rad"),
                ]:
                    header.create_dataset(key, data=[value]).attrs["Unit"] = unit
                header.create_group("Phases")
                data.create_dataset("X", data=np.tile([0, .2, .4], 2)).attrs["Unit"] = "um"
                data.create_dataset("Y", data=np.repeat([0, .25], 3)).attrs["Unit"] = "um"
                data.create_dataset("Euler", data=np.ones((6, 3), dtype="f4")).attrs["Unit"] = "rad"
                data["Phase"] = np.full(6, 1, dtype="u1")
                data["Error"] = np.full(6, 1, dtype="u1")
                for key, val in [("Pattern Center X", .42), ("Pattern Center Y", .52), ("Detector Distance", .45)]:
                    data.create_dataset(key, data=np.full(6, val)).attrs["Unit"] = "fraction"
                data["Processed Patterns"] = np.random.default_rng(9).integers(0, 255, (6, 32, 40), dtype="u1")
                data["Band Contrast"] = np.arange(6, dtype="u1")
                f["Patterns alias"] = data["Processed Patterns"]
                f["1/Data Processing/Header/Phases"] = header["Phases"]
                f["1/EDS/Data/Spectrum"] = np.zeros((6, 10))
                f["1/EDS/Data/Live Time"] = np.arange(6)
                f["1/Data Processing/Shared EDS"] = f["1/EDS/Data"]
            digest = hashlib.sha256(source.read_bytes()).digest()
            solver.open(source)
            solver.set_crystal({"space_group": 225, "cell": [3.52] * 3 + [90] * 3,
                                "sites": [{"symbol": "Ni", "xyz": [0, 0, 0]}]})
            solver.configure(sim_halfw=24)
            solver.simulate_master()
            solver.add_simulated_phase("Ni")
            solver.configure(roi_x=1, roi_y=0, roi_w=1, roi_h=1, bandwidth=24,
                             static_background="none", dynamic_background="none")
            solver.index()
            solver.export(output)
            with h5py.File(output, "r") as f:
                data = f["1/EBSD/Data"]
                self.assertNotIn("Processed Patterns", data)
                self.assertNotIn("Patterns alias", f)
                self.assertNotIn("1/EDS/Data/Spectrum", f)
                np.testing.assert_array_equal(f["1/EDS/Data/Live Time"], np.arange(6))
                self.assertNotIn("1/Data Processing", f)
                self.assertEqual(f["1/EBSD/Header/Phases/1/Laue Group"].attrs["Symbol"], "m-3m")
                np.testing.assert_array_equal(data["Band Contrast"], np.arange(6))
                np.testing.assert_array_equal(data["Phase"], [0, 1, 0, 0, 0, 0])
                np.testing.assert_array_equal(data["Error"], data["Phase"])
                np.testing.assert_allclose(f["1/EBSD/Header/Hit Rate"], [100 / 6])
                np.testing.assert_array_equal(data["Euler"][0], [0, 0, 0])
                self.assertTrue(np.isfinite(data["Euler"][1]).all() and np.any(data["Euler"][1]))
                self.assertIn("PatternCentres", f["zebsdsolve"])
                np.testing.assert_allclose(f["1/EBSD/Header/Specimen Orientation Euler"], [[.2, .1, .3]])
            self.assertEqual(hashlib.sha256(source.read_bytes()).digest(), digest)


if __name__ == "__main__":
    unittest.main()
