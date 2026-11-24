"""The orientations a scan file stores from its maker's own indexing, read by the scan readers and
identified against the engine's (numpy, h5py, native build).

A toy scan of nickel and GaN is written three times, laid out as Oxford (.h5oina), Bruker and
EDAX write h5ebsd, each with its orientations in a different planted convention computed here
from the convention's definition (not with the engine's code): the sample's axes, the rotation's
sense, the crystal's axes on the lattice; and each a small misorientation from the engine's, as
a maker whose geometry differs would have them. Opened, indexed and asked, the engine names each
convention and measures each misorientation, which no export applies.
"""
import os
from pathlib import Path
import sys
import shutil
import tempfile
import unittest

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from zebsdsolve import Solver  # noqa: E402

DATA = Path(os.environ.get("ZEBSDSOLVE_TESTDATA", ROOT / "testdata"))
NI = DATA / "nickel" / "Ni_master_20kV.npz"
GAN = DATA / "gan" / "GaN_master_20kV.npz"


def passive(q):
    """The matrices of v -> q v q* ((w, x, y, z) rows): sample to crystal."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


def bunge(g):
    """Passive Bunge angles of rotation matrices (g = Rz(phi2) Rx(Phi) Rz(phi1), as the
    sample-to-crystal matrix of every EBSD file), each in [0, 2pi)."""
    big = np.arccos(np.clip(g[:, 2, 2], -1, 1))
    p1 = np.arctan2(g[:, 2, 0], -g[:, 2, 1])
    p2 = np.arctan2(g[:, 0, 2], g[:, 1, 2])
    return np.stack([np.mod(p1, 2 * np.pi), big, np.mod(p2, 2 * np.pi)], -1)


def from_bunge(e):
    """The passive matrices of Bunge angles (the inverse of `bunge`)."""
    p1, big, p2 = e[:, 0], e[:, 1], e[:, 2]
    c1, s1, c, s, c2, s2 = np.cos(p1), np.sin(p1), np.cos(big), np.sin(big), np.cos(p2), np.sin(p2)
    return np.stack([
        np.stack([c1 * c2 - s1 * s2 * c, s1 * c2 + c1 * s2 * c, s2 * s], -1),
        np.stack([-c1 * s2 - s1 * c2 * c, -s1 * s2 + c1 * c2 * c, c2 * s], -1),
        np.stack([s1 * s, -c1 * s, c], -1)], -2)


def rotation(v):
    """The rotation by the rotation vector v (Rodrigues)."""
    th = np.linalg.norm(v)
    if th == 0:
        return np.eye(3)
    k = np.asarray(v) / th
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * kx + (1 - np.cos(th)) * kx @ kx


def written(q, phase, frame, misorientation=(0, 0, 0)):
    """Bunge angles of the engine's orientations q as `frame` writes them, seen from sample axes
    rotated by `misorientation` (a rotation vector about the engine's axes): the sample's axes
    S = A . M (A's rows the frame's axes in the engine's), the crystal's C (rows the frame's
    crystal axes in the master's, X || a, Z || c*), g_frame = C g S^T, transposed when active."""
    a = np.zeros((3, 3))
    for i, code in enumerate(frame["axes"]):
        a[i, abs(code) - 1] = np.sign(code)
    s = a @ rotation(misorientation)
    c30, s30 = np.cos(np.pi / 6), np.sin(np.pi / 6)
    # on a hexagonal cell X || a* lies 30 degrees from a about c (a* is perpendicular to b and c)
    hexagonal = np.array([[c30, s30, 0], [-s30, c30, 0], [0, 0, 1]]) if frame["crystal"] == "x_astar_z_c" else np.eye(3)
    g = passive(q)
    out = np.empty_like(g)
    for i in range(len(g)):
        c = hexagonal if phase[i] == 1 else np.eye(3)
        gf = c @ g[i] @ s.T
        out[i] = gf.T if frame["active"] else gf
    return bunge(out)


PLANTED = {
    "oxford": {"axes": [2, -1, 3], "active": False, "crystal": "x_astar_z_c"},
    "bruker": {"axes": [-2, 1, 3], "active": False, "crystal": "x_a_z_cstar"},
    "edax": {"axes": [1, 2, 3], "active": True, "crystal": "x_astar_z_c"},
}
# the misorientation of each maker's orientations from the engine's (engine axes, radians)
MISORIENTED = {"oxford": [0.03, 0.005, -0.02], "bruker": [-0.02, 0.01, 0.0], "edax": [0.0, 0.0, 0.012]}
# an Oxford detector orientation 2.3 degrees from the nominal (0, 90, 0), and the geometry's
# rotation it is (Oxford's detector axes x along the columns, y up the pattern, z the normal;
# the engine's the row direction, the column direction, the normal)
DETECTOR_EULER = np.array([0.012, np.pi / 2 + 0.035, 2 * np.pi - 0.02])


def detector_rot(e):
    def rz(t):
        return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])

    def rx(t):
        return np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])

    b = rx(np.pi / 2).T @ rz(e[0]) @ rx(e[1]) @ rz(e[2])
    th = np.arccos(np.clip((np.trace(b) - 1) / 2, -1, 1))
    v = th / (2 * np.sin(th)) * np.array([b[2, 1] - b[1, 2], b[0, 2] - b[2, 0], b[1, 0] - b[0, 1]])
    return np.array([-v[1], v[0], v[2]])


# Bruker's pattern centre where its patterns were clipped: PCY a fraction of the height before
PCY, UNCLIPPED = 0.3, 62


class StoredOrientations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        s = Solver()
        s.add_phase(NI)
        s.add_phase(GAN)
        s.set_geometry(pc=(0.45, 0.25, 0.52), rot=(0, 0, 0), px_um=0)
        toy = s.synthesize(toy_seed=21, toy_rows=10, toy_cols=14, toy_height=60, toy_width=60, toy_grains=12, toy_counts=0)
        cls.rows, cls.cols = 10, 14
        n = cls.rows * cls.cols
        cls.patterns = np.stack([s.pattern(c, processed=False) for c in range(n)]).astype(np.float32)
        cls.q = toy["quaternions"].reshape(-1, 4).astype(np.float64)
        cls.phase = toy["phase"].reshape(-1)
        cls.geometry = s.geometry
        s.close()
        cls.tmp = tempfile.TemporaryDirectory(prefix="zebsdsolve-stored-")
        # a few cells the maker did not index, each format's way
        cls.missing = np.arange(n) % 17 == 5

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def write(self, layout, name=None, scan_rotation=None, pcs=None):
        path = Path(self.tmp.name) / f"{name or layout}.h5"
        e = written(self.q, self.phase, PLANTED[layout], MISORIENTED[layout])
        n, rows, cols = len(e), self.rows, self.cols
        with h5py.File(path, "w") as f:
            if layout == "oxford":
                f["Manufacturer"] = "Oxford Instruments"
                f["Format Version"] = "5.0"
                h, d = f.create_group("1/EBSD/Header"), f.create_group("1/EBSD/Data")
                for k, v, unit in [("X Cells", cols, "px"), ("Y Cells", rows, "px"), ("X Step", 1.0, "um"), ("Y Step", 1.0, "um"), ("Pattern Width", 60, "px"), ("Pattern Height", 60, "px"), ("Tilt Angle", np.radians(70), "rad"), ("Hit Rate", 90.0, "%")]:
                    h[k] = [v]
                    h[k].attrs["Unit"] = unit
                h["Detector Orientation Euler"] = DETECTOR_EULER[None].astype(np.float32)
                h["Detector Orientation Euler"].attrs["Unit"] = "rad"
                if scan_rotation is not None:
                    h["Scanning Rotation Angle"] = np.array([scan_rotation], np.float32)
                    h["Scanning Rotation Angle"].attrs["Unit"] = "rad"
                if pcs is not None:
                    for k, v in pcs.items():
                        d[k] = v
                for pid, (name, sg, laue) in enumerate([("Ni", 225, 11), ("GaN", 186, 9)], 1):
                    g = h.create_group(f"Phases/{pid}")
                    g["Phase Name"] = [name]
                    g["Space Group"] = np.array([sg], np.int32)
                    g["Laue Group"] = np.array([laue], np.int32)
                d["Processed Patterns"] = self.patterns
                d["Euler"] = e.astype(np.float32)
                d["Phase"] = np.where(self.missing, 0, self.phase + 1).astype(np.uint8)
                d["Band Contrast"] = np.arange(n, dtype=np.uint8)
                d["X"] = np.tile(np.arange(cols, dtype=np.float32), rows)
                d["X"].attrs["Unit"] = "um"
                d["Y"] = np.repeat(np.arange(rows, dtype=np.float32), cols)
                d["Y"].attrs["Unit"] = "um"
                # the maker's measures of its indexing, and an earlier processing's solution
                d["Mean Angular Deviation"] = np.full(n, 0.01, np.float32)
                d["Mean Angular Deviation"].attrs["Unit"] = "rad"
                d["Bands"] = np.full(n, 8, np.uint8)
                f["1/Data Processing/1/EBSD/Data/Euler"] = e.astype(np.float32)
                # what a results file has no use for (a spectrum per cell), and what it keeps
                f["1/EDS/Data/Spectrum"] = np.zeros((n, 100), np.float32)
                f["1/Electron Image/Data/SE/Image"] = np.arange(64 * 48, dtype=np.uint8).reshape(48, 64)
            elif layout == "bruker":
                h, d = f.create_group("Scan 0/EBSD/Header"), f.create_group("Scan 0/EBSD/Data")
                for k, v in [("NROWS", rows), ("NCOLS", cols), ("PatternHeight", 60), ("PatternWidth", 60), ("XSTEP", 1.0), ("YSTEP", 1.0), ("SampleTilt", 70.0), ("CameraTilt", 0.0), ("UnClippedPatternHeight", UNCLIPPED)]:
                    h[k] = v
                d["RawPatterns"] = self.patterns
                for k, v in (("PCX", 0.45), ("PCY", PCY), ("DD", 0.52)):
                    d[k] = np.full(n, v, np.float32)
                deg = np.degrees(e)
                # (Bruker writes its angles from -180 to 180)
                deg = np.where(deg > 180, deg - 360, deg)
                for i, k in enumerate(("phi1", "PHI", "phi2")):
                    d[k] = deg[:, i].astype(np.float32)
                d["Phase"] = np.where(self.missing, 0, self.phase + 1).astype(np.int32)
            else:
                h, d = f.create_group("Scan 1/EBSD/Header"), f.create_group("Scan 1/EBSD/Data")
                for k, v in [("nRows", rows), ("nColumns", cols), ("Pattern Height", 60), ("Pattern Width", 60), ("Step X", 1.0), ("Step Y", 1.0), ("Sample Tilt", 70.0)]:
                    h[k] = v
                d["Pattern"] = self.patterns
                e = np.where(self.missing[:, None], 4 * np.pi, e)
                for i, k in enumerate(("Phi1", "Phi", "Phi2")):
                    d[k] = e[:, i].astype(np.float32)
                d["Phase"] = self.phase.astype(np.int8)
        return path

    def check(self, layout):
        s = Solver()
        try:
            s.open(self.write(layout))
            # what the readers take from the files' geometry
            if layout == "oxford":
                np.testing.assert_allclose(s.geometry["rot"], detector_rot(DETECTOR_EULER), atol=1e-6)
            if layout == "bruker":
                self.assertAlmostEqual(s.geometry["pc"][1], PCY * UNCLIPPED / 60, places=6)
            s.add_phase(NI)
            s.add_phase(GAN)
            s.set_geometry(**self.geometry)
            s.index()
            found = s.identify_frame()
            planted = PLANTED[layout]
            f = found["frame"]
            msg = f"{layout}: {f['text']}; median {found['median_deg']:.3f} deg, misorientation {np.round(found['misorientation'], 4)}, next {found['runner_up_deg']:.1f} deg, labels {found['matched']}"
            self.assertEqual(f["axes"], planted["axes"], msg)
            self.assertEqual(f["active"], planted["active"], msg)
            self.assertEqual(f["crystal"], planted["crystal"], msg)
            np.testing.assert_allclose(found["misorientation"], MISORIENTED[layout], atol=2e-3, err_msg=msg)
            np.testing.assert_allclose(found["misorientation_rows"], 0, atol=3e-3, err_msg=msg)
            np.testing.assert_allclose(found["misorientation_cols"], 0, atol=3e-3, err_msg=msg)
            self.assertTrue(found["installed"] and found["median_deg"] < 2.5 and found["runner_up_deg"] > 5, msg)
            self.assertEqual(found["cells"], int(((~self.missing) & (s.results()["phase"].reshape(-1) == self.phase)).sum()), msg)
            self.assertEqual(s.frame["axes"], planted["axes"], msg)
            # a new scan starts from the engine's frame again
            s.open(self.write(layout))
            self.assertEqual(s.frame["axes"], [1, 2, 3])
        finally:
            s.close()

    def test_oxford(self):
        self.check("oxford")

    def test_h5oina_export(self):
        """An .h5oina made from the Oxford acquisition: the acquisition copied but for its
        patterns and spectra, the indexing the engine's, its angles in the convention found."""
        src = self.write("oxford")
        out = Path(self.tmp.name) / "results.h5oina"
        s = Solver()
        try:
            s.open(src)
            s.add_phase(NI)
            s.add_phase(GAN)
            s.set_geometry(**self.geometry)
            s.index()
            with self.assertRaises(Exception):
                s.export(src)
            s.export(out)
            r = s.results()
            q = r["quaternions"].reshape(-1, 4).astype(np.float64)
            ph = r["phase"].reshape(-1)
            frame = s.frame
        finally:
            s.close()
        with h5py.File(out, "r") as f:
            d, h = f["1/EBSD/Data"], f["1/EBSD/Header"]
            # the angles as the frame found writes the engine's orientations (here computed
            # from the convention's definition), the misorientation the acquisition's own had
            # not applied
            want = written(q, ph, frame)
            got = d["Euler"][:].astype(np.float64)
            ok = ph >= 0
            self.assertLess(np.abs(from_bunge(got[ok]) - from_bunge(want[ok])).max(), 1e-4)
            np.testing.assert_array_equal(d["Phase"][:], np.where(ok, ph + 1, 0))
            np.testing.assert_array_equal(d["Error"][:], ok.astype(np.uint8))
            np.testing.assert_array_equal(d["Band Contrast"][:], np.arange(len(ph), dtype=np.uint8))
            self.assertEqual(d["X"].attrs["Unit"], "um")
            self.assertNotIn("Processed Patterns", d)
            self.assertNotIn("Spectrum", f["1/EDS/Data"])
            self.assertEqual(f["1/Electron Image/Data/SE/Image"].shape, (48, 64))
            self.assertEqual(h["X Cells"][0], self.cols)
            self.assertEqual(h["X Cells"].attrs["Unit"], "px")
            self.assertAlmostEqual(float(h["Hit Rate"][0]), 100 * ok.mean(), places=3)
            # the detector orientation the orientations were found under: the geometry's, here
            # set to the nominal one
            got_det = h["Detector Orientation Euler"][0].astype(np.float64)
            self.assertLess(np.abs(np.angle(np.exp(1j * (got_det - [0, np.pi / 2, 0])))).max(), 1e-5, got_det)
            self.assertEqual(h["Detector Orientation Euler"].attrs["Unit"], "rad")
            self.assertEqual(sorted(h["Phases"].keys()), ["1", "2"])
            self.assertEqual(h["Phases/2/Phase Name"][0], b"GaN")
            z = f["zebsdsolve"]
            self.assertEqual(z["NCC"].shape, (len(ph),))
            self.assertIn("Processed Patterns", z["Left Out"][()].decode())
            self.assertIn("Spectrum", z["Left Out"][()].decode())
            self.assertIn("X||a* Z||c", z["Convention"][()].decode())
            self.assertEqual(f["Manufacturer"][()], b"Oxford Instruments")
            # strings as the acquisition stored them, and the engine's as Oxford's files have
            # theirs: variable-length UTF-8
            def vlen(t):
                return t.get_class() == h5py.h5t.STRING and t.is_variable_str() and t.get_cset() == h5py.h5t.CSET_UTF8
            for t in (d["X"].attrs.get_id("Unit").get_type(), f["Manufacturer"].id.get_type(), h["Phases/2/Phase Name"].id.get_type(), z["Version"].id.get_type(), z["Left Out"].id.get_type()):
                self.assertTrue(vlen(t))
            # (as #13 found Oxford's files want them) the maker's other measures kept but
            # cleared; its earlier processing left out; its calibration kept beside the
            # geometry's
            self.assertTrue(np.isnan(d["Mean Angular Deviation"][:]).all())
            self.assertEqual(d["Mean Angular Deviation"].attrs["Unit"], "rad")
            np.testing.assert_array_equal(d["Bands"][:], 0)
            self.assertNotIn("Data Processing", f["1"])
            self.assertIn("Data Processing", z["Left Out"][()].decode())
            np.testing.assert_allclose(z["OriginalCalibration/Detector Orientation Euler"][0], DETECTOR_EULER, atol=1e-6)
            self.assertAlmostEqual(float(h["Tilt Angle"][0]), np.radians(self.geometry["sample_tilt_deg"]), places=5)
        # not from an acquisition that holds this writer's results (its orientations an earlier
        # run's), nor from one whose cells are off its raster
        s = Solver()
        try:
            again = Path(self.tmp.name) / "again.h5"
            shutil.copy(src, again)
            with h5py.File(again, "r+") as f:
                f["zebsdsolve/Version"] = "0.2.2"
            s.open(again)
            s.add_phase(NI)
            s.add_phase(GAN)
            s.set_geometry(**self.geometry)
            s.index()
            with self.assertRaisesRegex(Exception, "AlreadyAZebsdsolveExport"):
                s.export(Path(self.tmp.name) / "again.h5oina")
            off = Path(self.tmp.name) / "off.h5"
            shutil.copy(src, off)
            with h5py.File(off, "r+") as f:
                f["1/EBSD/Data/X"][3] += 0.5
            s.open(off)
            s.set_geometry(**self.geometry)
            s.index()
            with self.assertRaisesRegex(Exception, "CellsOffTheRaster"):
                s.export(Path(self.tmp.name) / "off.h5oina")
        finally:
            s.close()
        # only from an Oxford acquisition
        s = Solver()
        try:
            s.open(self.write("bruker"))
            s.add_phase(NI)
            s.add_phase(GAN)
            s.set_geometry(**self.geometry)
            s.index()
            with self.assertRaisesRegex(Exception, "NotAnOxfordAcquisition"):
                s.export(Path(self.tmp.name) / "no.h5oina")
        finally:
            s.close()

    def test_oxford_scan_turn(self):
        """Which way the beam went: unturned (Scanning Rotation Angle 0) the other way from a
        half turn, as #13 found Oxford's files; the file's own pattern centres, where they
        move, over what it says."""
        def exported(name, **kw):
            s = Solver()
            try:
                s.open(self.write("oxford", name, **kw))
                s.add_phase(NI)
                s.add_phase(GAN)
                s.set_geometry(**self.geometry)
                s.index()
                out = Path(self.tmp.name) / f"{name}.h5oina"
                s.export(out)
            finally:
                s.close()
            with h5py.File(out, "r") as f:
                d = f["1/EBSD/Data"]
                return {k: d[k][:] for k in ("Pattern Center X", "Pattern Center Y", "Detector Distance")}

        def drift(pcs):
            x = pcs["Pattern Center X"].reshape(self.rows, self.cols).astype(np.float64)
            return (x[:, -1] - x[:, 0]).mean()

        half = exported("half", scan_rotation=np.pi)
        none = exported("none", scan_rotation=0.0)
        self.assertGreater(abs(drift(half)), 1e-5)
        self.assertAlmostEqual(drift(none), -drift(half), delta=1e-3 * abs(drift(half)))
        # unturned, but its stored pattern centres moving as a half turn's: they decide
        stored = exported("stored", scan_rotation=0.0, pcs=half)
        self.assertAlmostEqual(drift(stored), drift(half), delta=1e-3 * abs(drift(half)))

    def test_bruker(self):
        self.check("bruker")

    def test_edax(self):
        self.check("edax")


if __name__ == "__main__":
    unittest.main()
