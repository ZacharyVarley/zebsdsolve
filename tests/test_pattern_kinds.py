"""A file that stores processed and unprocessed patterns side by side (numpy, h5py, native build).

An Oxford acquisition can hold both: Processed Patterns (unsigned 8-bit, as the acquisition
prepared them) and Unprocessed Patterns (signed 16-bit, the detector's own), each with its
static background. The setting `stored_patterns` says which is read when the scan opens."""
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from zebsdsolve import Solver

ROWS, COLS, H, W = 2, 3, 8, 10
PROCESSED = np.arange(ROWS * COLS * H * W, dtype=np.uint8).reshape(ROWS * COLS, H, W)
UNPROCESSED = (np.arange(ROWS * COLS * H * W, dtype=np.int32).reshape(ROWS * COLS, H, W) * 97 - 20000).astype(np.int16)


class PatternKinds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.solver = Solver()

    @classmethod
    def tearDownClass(cls):
        cls.solver.close()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="zebsdsolve-kinds-")
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "scan.h5oina"

    def write(self, processed=True, unprocessed=True):
        with h5py.File(self.path, "w") as f:
            g = f.create_group("1/EBSD")
            h, d = g.create_group("Header"), g.create_group("Data")
            h.create_dataset("Y Cells", data=[ROWS])
            h.create_dataset("X Cells", data=[COLS])
            h.create_dataset("Processed Static Background", data=np.full((H, W), 7, np.uint8))
            h.create_dataset("Unprocessed Static Background", data=np.full((H, W), -300, np.int16))
            if processed:
                d.create_dataset("Processed Patterns", data=PROCESSED)
            if unprocessed:
                d.create_dataset("Unprocessed Patterns", data=UNPROCESSED)

    def open(self, which):
        self.solver.configure(stored_patterns=which)
        self.solver.open(self.path)
        return self.solver.scan

    def test_both_kinds_read_as_asked(self):
        self.write()
        scan = self.open("processed")
        self.assertEqual(scan["pattern_kinds"], ("processed", "unprocessed"))
        self.assertFalse(scan["unprocessed"])
        self.assertEqual(scan["dtype"], "uint8")
        np.testing.assert_array_equal(self.solver.pattern(4, processed=False), PROCESSED[4])
        scan = self.open("unprocessed")
        self.assertTrue(scan["unprocessed"])
        self.assertEqual(scan["dtype"], "float32")
        self.assertTrue(scan["has_background"])
        # (signed values, widened exactly)
        np.testing.assert_array_equal(self.solver.pattern(4, processed=False), UNPROCESSED[4].astype(np.float32))
        self.assertLess(self.solver.pattern(0, processed=False).min(), 0)

    def test_one_kind_read_as_it_is(self):
        for processed, unprocessed in ((True, False), (False, True)):
            with self.subTest(processed=processed):
                self.write(processed, unprocessed)
                scan = self.open("unprocessed" if processed else "processed")
                self.assertEqual(scan["pattern_kinds"], ("processed",) if processed else ("unprocessed",))
                self.assertEqual(scan["unprocessed"], unprocessed)
                want = PROCESSED[1] if processed else UNPROCESSED[1].astype(np.float32)
                np.testing.assert_array_equal(self.solver.pattern(1, processed=False), want)
        self.solver.configure(stored_patterns="processed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
