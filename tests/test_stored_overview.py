"""Portable quality-map and lazy-background checks (numpy, h5py, native build)."""
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
        self.solver.configure(static_background="average", static_patterns=6,
                              dynamic_background="none", mask=False, clahe=False)

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

    def test_pixel_order_layouts_and_vendor_fields(self):
        values = np.array([5, 1, 8, 3, 6, 2], dtype="u2")
        for vendor, field, kind in [("kikuchipy", "band_contrast", 1),
                                    ("oxford", "Band Contrast", 1), ("edax", "IQ", 2),
                                    ("bruker", "RadonQuality", 3), ("emsoft", "IQ", 2)]:
            for shape in [(6,), (2, 3), (6, 1), (1, 6)]:
                with self.subTest(vendor=vendor, shape=shape):
                    self.fixture({"1": {field: values.reshape(shape)}}, vendor=vendor)
                    r = self.solver.stored_overview()
                    self.assertEqual(r["kind"], kind)
                    self.assertEqual(r["label"], {1: "Band contrast", 2: "Image quality", 3: "Radon quality"}[kind])
                    self.assertEqual(r["values"].dtype, np.float32)
                    np.testing.assert_array_equal(r["values"], values.reshape(2, 3))

    def test_preference_and_invalid_field_fallback(self):
        self.fixture({"1": {"Band Contrast": np.arange(6), "IQ": np.full(6, 77)}})
        self.assertEqual(self.solver.stored_overview()["kind"], 1)
        for bad in [np.arange(6).reshape(3, 2), np.array([b"a"] * 6)]:
            with self.subTest(invalid=bad.shape):
                self.fixture({"1": {"Band Contrast": bad,
                                    "Image Quality": np.array([1, 2, np.nan, 4, np.inf, 1e100])}})
                r = self.solver.stored_overview()
                self.assertEqual(r["kind"], 2)
                np.testing.assert_array_equal(r["values"].ravel()[:2], [1, 2])
                self.assertTrue(np.isnan(r["values"].ravel()[[2, 4, 5]]).all())

    def test_pattern_quality_label(self):
        self.fixture({"1": {"Pattern Quality": np.arange(6)}})
        result = self.solver.stored_overview()
        self.assertEqual(result["kind"], 4)
        self.assertEqual(result["label"], "Pattern quality")

    def test_optional_and_other_scan_fields_not_used(self):
        self.fixture({"1": {}, "2": {"IQ": np.full(6, 77)}})
        self.assertIsNone(self.solver.stored_overview())
        self.solver.set_patterns(np.zeros((2, 3, 8, 10), dtype="u1"))
        self.assertIsNone(self.solver.stored_overview())

    def test_stored_map_needs_no_pattern_pixels(self):
        self.fixture({"1": {"Band Contrast": np.arange(6)}}, patterns=False)
        np.testing.assert_array_equal(self.solver.stored_overview()["values"],
                                      np.arange(6).reshape(2, 3))
        with self.assertRaises(Error):
            self.solver.pattern_stats([0])

    def test_background_stays_lazy_and_matches_average_when_requested(self):
        self.fixture({"1": {"IQ": np.arange(6)}})
        self.solver.events()
        self.solver.stored_overview()
        raw = np.stack([self.solver.pattern(c, False) for c in range(6)])
        self.assertFalse(any("static background: the average" in e.get("text", "")
                             for e in self.solver.events()))
        np.testing.assert_allclose(self.solver.pattern(0, "background"), raw.mean(axis=0), atol=1e-5)
        self.assertEqual(sum("static background: the average" in e.get("text", "")
                             for e in self.solver.events()), 1)
        self.solver.pattern(1, "background")
        self.assertFalse(any("static background: the average" in e.get("text", "")
                             for e in self.solver.events()))



if __name__ == "__main__":
    unittest.main(verbosity=2)
