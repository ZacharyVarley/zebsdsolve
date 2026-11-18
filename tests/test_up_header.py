"""UP packed-header and incomplete-payload regressions; NumPy + native build."""
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from zebsdsolve import Error, Solver


class UpHeader(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="zebsdsolve-up-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.s = Solver()
        self.addCleanup(self.s.close)

    def fixture(self, bits=8):
        offset = 42
        header = bytearray(offset)
        struct.pack_into("<4I", header, 0, 3, 10, 8, offset)
        struct.pack_into("<BIIBdd", header, 16, 1, 3, 2, 0, .025, .05)
        values = (np.arange(480, dtype="u2") * 257).astype("u1" if bits == 8 else "<u2").reshape(6, 8, 10)
        path = self.root / f"scan.up{1 if bits == 8 else 2}"
        path.write_bytes(header + values.tobytes())
        return path, values

    def test_packed_header_keeps_geometry_and_payload(self):
        for bits in (8, 16):
            with self.subTest(bits=bits):
                path, values = self.fixture(bits)
                self.s.open(path)
                self.assertEqual((self.s.scan["rows"], self.s.scan["cols"]), (2, 3))
                self.assertAlmostEqual(self.s.geometry["step_x_um"], .025)
                self.assertAlmostEqual(self.s.geometry["step_y_um"], .05)
                for cell in (0, 5):
                    np.testing.assert_array_equal(self.s.pattern(cell, processed=False, full=True), values[cell])

    def test_incomplete_header_or_pattern_is_rejected(self):
        for bits in (8, 16):
            path, _ = self.fixture(bits=bits)
            path.write_bytes(path.read_bytes()[:-1])
            with self.subTest(bits=bits), self.assertRaisesRegex(Error, "UPIncompletePattern"):
                self.s.open(path)
        path, _ = self.fixture()
        data = bytearray(path.read_bytes())
        struct.pack_into("<I", data, 12, 41)
        path.write_bytes(data[:41] + data[42:])
        with self.assertRaisesRegex(Error, "BadScanFile"):
            self.s.open(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
