"""Essential stored-quality regressions (numpy, h5py, native build)."""
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from zebsdsolve import Error, Solver


class StoredOverview(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.solver = Solver()

    @classmethod
    def tearDownClass(cls):
        cls.solver.close()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="zebsdsolve-overview-")
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "scan.h5"

    def fixture(self, fields, *, vendor="kikuchipy", patterns=True):
        names = {
            "kikuchipy": ("n_rows", "n_columns", "patterns"),
            "oxford": ("Y Cells", "X Cells", "Processed Patterns"),
            "edax": ("nRows", "nColumns", "Pattern"),
            "bruker": ("NROWS", "NCOLS", "RawPatterns"),
            "emsoft": (None, None, "EBSDPatterns"),
        }
        row, col, pattern = names[vendor]
        with h5py.File(self.path, "w") as f:
            for scan, values in fields.items():
                if vendor == "emsoft":
                    d = f.create_group("EMData/EBSD")
                else:
                    g = f.create_group(scan + "/EBSD")
                    h, d = g.create_group("Header"), g.create_group("Data")
                    h.create_dataset(row, data=[2])
                    h.create_dataset(col, data=[3])
                p = d.create_dataset(pattern, data=np.arange(480, dtype="u1").reshape(6, 8, 10),
                                     compression=None if patterns else "gzip")
                if not patterns:
                    chunk = p.id.get_chunk_info(0)
                    corrupt = (chunk.byte_offset, chunk.size)
                for name, array in values.items():
                    d.create_dataset(name, data=array)
        if not patterns:
            # Leave metadata intact and make the first compressed pattern unreadable.
            with self.path.open("r+b") as f:
                f.seek(corrupt[0])
                f.write(bytes(corrupt[1]))
        if vendor == "emsoft":
            self.solver.configure(map_rows=2, map_cols=3)
        else:
            self.solver.configure(map_rows=0, map_cols=0)
        self.solver.open(self.path)

    def test_vendor_quality_maps_preserve_pixel_order(self):
        values = np.array([5, 1, 8, 3, 6, 2], dtype="u2")
        for vendor, field, kind, shape in [
            ("kikuchipy", "band_contrast", 1, (6,)),
            ("oxford", "Band Contrast", 1, (2, 3)),
            ("edax", "IQ", 2, (6, 1)),
            ("bruker", "RadonQuality", 3, (1, 6)),
            ("emsoft", "IQ", 2, (6,)),
        ]:
            with self.subTest(vendor=vendor):
                self.fixture({"1": {field: values.reshape(shape)}}, vendor=vendor)
                result = self.solver.stored_overview()
                self.assertEqual(result["kind"], kind)
                np.testing.assert_array_equal(result["values"], values.reshape(2, 3))

    def test_missing_or_invalid_quality_stays_in_selected_scan(self):
        self.fixture({"1": {}, "2": {"IQ": np.arange(6)}})
        self.assertIsNone(self.solver.stored_overview())
        self.fixture({"1": {"Band Contrast": np.arange(6).reshape(3, 2),
                            "IQ": np.arange(6)}})
        self.assertEqual(self.solver.stored_overview()["kind"], 2)

    def test_stored_quality_does_not_read_pattern_pixels(self):
        self.fixture({"1": {"Band Contrast": np.arange(6)}}, patterns=False)
        np.testing.assert_array_equal(self.solver.stored_overview()["values"],
                                      np.arange(6).reshape(2, 3))
        with self.assertRaises(Error):
            self.solver.pattern_stats([0])



if __name__ == "__main__":
    unittest.main(verbosity=2)
