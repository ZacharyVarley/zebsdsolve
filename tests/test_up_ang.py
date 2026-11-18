"""Standard UP + ANG geometry/IQ companions (NumPy and a native build)."""
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from zebsdsolve import Error, Solver


class UpAng(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="zebsdsolve-up-ang-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.s = Solver()
        self.addCleanup(self.s.close)

    def up(self, version=1, bits=8, rows=2, cols=3, count=6, dx=.025, dy=.05):
        offset = 16 if version == 1 else 42
        header = bytearray(offset)
        struct.pack_into("<4I", header, 0, version, 10, 8, offset)
        if version >= 3:
            struct.pack_into("<BIIBdd", header, 16, 0, cols, rows, 0, dx, dy)
        patterns = np.arange(count * 80, dtype="u1" if bits == 8 else "<u2").reshape(count, 8, 10)
        path = self.root / f"scan-v{version}.up{1 if bits == 8 else 2}"
        path.write_bytes(header + patterns.tobytes())
        return path, patterns

    def ang(self, records=None, pc_header=()):
        if records is None:
            records = [[.1, .2, .3, 10 + col * .025, 20 + row * .05,
                        100 * row + col, .9, 1, 123, 0]
                       for row in range(2) for col in range(3)]
        path = self.root / "scan.ang"
        header = ["# GRID: SqrGrid", "# XSTEP: .025", "# YSTEP: .05",
                  "# NCOLS_ODD: 3", "# NCOLS_EVEN: 3", "# NROWS: 2"]
        path.write_text("\r\n".join([*pc_header, *header] + [" ".join(map(str, r)) for r in records]))
        return path, np.asarray(records)

    def test_pair_preserves_geometry_iq_and_pattern_order(self):
        for version, bits, producer_header, expected_pc in (
            (1, 8, ["# HEADER: Start", "# VERSION 7"], [.43, .73, .61]),
            (3, 16, ["# SOFTWARE: zebsdsolve 0.2.1"], [.43, .6625, .7625]),
        ):
            with self.subTest(version=version, bits=bits):
                up, patterns = self.up(version, bits)
                ang, records = self.ang(pc_header=[
                    *producer_header, "# x-star .43", "# y-star .27", "# z-star .61",
                ])
                self.s.open(up, ang)
                self.assertTrue(self.s.scan["has_pc"])
                np.testing.assert_allclose(self.s.geometry["pc"], expected_pc, atol=1e-14, rtol=0)
                self.assertEqual((self.s.scan["rows"], self.s.scan["cols"]), (2, 3))
                self.assertAlmostEqual(self.s.geometry["step_x_um"], .025)
                self.assertAlmostEqual(self.s.geometry["step_y_um"], .05)
                overview = self.s.stored_overview()
                self.assertEqual(overview["kind"], 2)
                np.testing.assert_array_equal(overview["values"], records[:, 5].reshape(2, 3))
                for cell in (0, 4, 5):
                    np.testing.assert_array_equal(self.s.pattern(cell, processed=False, full=True), patterns[cell])
                self.s.open(up)
                self.assertEqual((self.s.scan["rows"], self.s.scan["cols"]), (1, 6) if version == 1 else (2, 3))
                self.assertIsNone(self.s.stored_overview())
                self.assertFalse(self.s.scan["has_pc"])

    def test_invalid_pair_preserves_current_scan(self):
        up, patterns = self.up()
        ang, records = self.ang()
        self.s.open(up, ang)
        previous_scan = self.s.scan
        previous_geometry = self.s.geometry
        previous_iq = self.s.stored_overview()["values"].copy()
        swapped = records.copy()
        swapped[[1, 2]] = swapped[[2, 1]]
        cases = [(records[:5], {}, "ANGPatternCountMismatch"),
                 (swapped, {}, "ANGCoordinatesNotRowMajor"),
                 (records, {"version": 3, "rows": 3, "cols": 2}, "ScanShapeMismatch"),
                 (records, {"version": 3, "dx": .1}, "UPANGStepMismatch"),
                 (records, {"version": 3, "count": 5}, "UPPatternCountMismatch")]
        for data, options, error in cases:
            with self.subTest(error=error):
                bad_up, _ = self.up(**options)
                bad_ang, _ = self.ang(data)
                with self.assertRaisesRegex(Error, error):
                    self.s.open(bad_up, bad_ang)
                self.assertEqual(self.s.scan, previous_scan)
                self.assertEqual(self.s.geometry, previous_geometry)
                np.testing.assert_array_equal(self.s.stored_overview()["values"], previous_iq)
                np.testing.assert_array_equal(self.s.pattern(4, processed=False, full=True), patterns[4])


if __name__ == "__main__":
    unittest.main(verbosity=2)
