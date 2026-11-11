"""Fitted detector poses survive result storage, calibration and native exports.

Run locally with a rebuilt native library, GPU, NumPy, h5py and pyMTEX:
    python tests/test_fitted_pose.py

The engine's existing simulator makes a monoclinic unique-b (Y) master. Its
C2 point-group label currently reduces about Z, so the fixture exposes any
implicit symmetry folding. pyMTEX compares rotations without crystal symmetry;
its native ANG importer checks the exported orientation records. Forks exposing
export_snapshot also exercise their candidate records and EMSphInx HDF5 export.
"""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from mtex import degree, loadEBSD, rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from zebsdsolve import Solver


def angular_error(a, b):
    """Quaternion sign is immaterial; no symmetry equivalence is allowed here."""
    return np.asarray(rotation(a).angle(rotation(b))) / degree


class FittedPose(unittest.TestCase):
    def setUp(self):
        self.s = Solver()
        self.addCleanup(self.s.close)
        self.s.configure(
            master_smooth=0, static_background="none", dynamic_background="none",
            mask=False, clahe=False, bandwidth=32, refine_iters=8,
            sim_halfw=32, sim_dmin_nm=.12, sim_bin_kev=5, sim_c_weak=0,
            toy_seed=17, toy_rows=3, toy_cols=4, toy_height=40, toy_width=40,
            toy_grains=12, toy_counts=0, toy_contrast=1, batch=12, sweeps=1,
        )
        self.s.set_crystal({
            "space_group": 10, "cell": [3.5, 4.5, 5.5, 90, 105, 90],
            "sites": [
                {"symbol": "Al", "xyz": [.17, .23, .31], "b_iso": .5},
                {"symbol": "Cu", "xyz": [.29, .11, .43], "b_iso": .5},
            ],
        })
        self.assertEqual(self.s.simulate_master()["failed"], 0)
        self.s.add_simulated_phase("C2 unique-b fixture")
        self.s.set_geometry(pc=[.45, .3, .6], px_um=0)
        self.s.synthesize()

    def assert_same_pose(self, a, b, tolerance=.0001):
        self.assertLess(float(np.max(angular_error(a, b))), tolerance)

    def test_bunge_preserves_shapes_and_rejects_bad_shapes(self):
        np.testing.assert_allclose(self.s.bunge([1, 0, 0, 0]), np.zeros(3))
        batch = np.zeros((2, 3, 4), dtype=np.float32)
        batch[..., 0] = 1
        np.testing.assert_allclose(self.s.bunge(batch), np.zeros((2, 3, 3)))
        self.assertEqual(self.s.bunge(np.empty((0, 4))).shape, (0, 3))
        for invalid in (1, [], [1, 0, 0], np.zeros((4, 1)), np.zeros(8)):
            with self.subTest(shape=np.shape(invalid)):
                with self.assertRaisesRegex(ValueError, "shape"):
                    self.s.bunge(invalid)

    def test_selected_pose_angles_and_exports_preserve_raw_winner(self):
        s = self.s
        # Repeat the same one-phase search in one batch, without result storage. This
        # public API is also available upstream, where candidate snapshots are absent.
        raw = s.index_cells(np.arange(12))
        candidate = raw["quaternions"][0]
        result = s.index()
        snapshot = s.export_snapshot() if callable(getattr(s, "export_snapshot", None)) else None
        q = result["quaternions"].reshape(-1, 4)
        # One phase, one batch and one sweep ensure the same search settings and raw
        # winner. Never use a candidate from another sweep to reconstruct selected poses.
        self.assertTrue(np.all(result["phase"] == 0))
        self.assert_same_pose(q, candidate)
        np.testing.assert_allclose(np.linalg.norm(q, axis=-1), 1, atol=1e-6)
        self.assertTrue(np.all(q[:, 0] >= 0))
        np.testing.assert_allclose(result["ncc"].reshape(-1), raw["ncc"][0], atol=1e-5)
        # MTEX Bunge rotations map crystal -> sample; engine poses map sample -> crystal.
        angles = [result["euler"].reshape(-1, 3), s.bunge(q)]
        if snapshot is not None:
            self.assert_same_pose(np.asarray(snapshot["candidates"]["quaternions"]).reshape(-1, 4), candidate)
            self.assert_same_pose(np.asarray(snapshot["results"]["quaternions"]).reshape(-1, 4), candidate)
            angles.append(np.asarray(snapshot["results"]["euler"]).reshape(-1, 3))
        for euler in angles:
            self.assert_same_pose(rotation.byEuler(euler).inv().data, candidate)
        reduced, _ = s.reduce(q)
        folded = angular_error(q, reduced) > 1
        self.assertGreater(int(folded.sum()), 0, "fixture must exercise a pose outside the generic C2 zone")
        original, wrong = s.simulate(q), s.simulate(reduced)
        correlation = np.array([np.corrcoef(a.ravel(), b.ravel())[0, 1] for a, b in zip(original, wrong)])
        self.assertLess(float(correlation[folded].max()), .5)
        with tempfile.TemporaryDirectory(prefix="zebsdsolve-pose-") as tmp:
            for suffix in (("ang", "h5") if snapshot is not None else ("ang",)):
                with self.subTest(format=suffix):
                    path = Path(tmp) / f"poses.{suffix}"
                    s.export(path)
                    options = {"setting": 0} if suffix == "ang" else {"dataSet": "Scan 1", "eulerCorrection": rotation.id}
                    imported = loadEBSD(str(path), grid=False, **options)
                    # ANG writes six decimal places; permit only the corresponding rounding.
                    self.assert_same_pose(imported.rotations.inv().data, candidate, .001)

    def test_vote_returns_native_search_pose_without_folding(self):
        s = self.s
        s.configure(fit_grid=4, fit_bandwidth=32, pc_steps=1, pc_steps_z=1,
                    pc_levels=1, refine=False)
        raw = s.index_cells(np.arange(12), bandwidth=32)["quaternions"][0]
        s.fit_pc()
        cells, phases, q, _ = s.fit_patterns()
        self.assertTrue(np.all(phases == 0))
        self.assert_same_pose(q, raw[cells])
        self.assertGreater(float(angular_error(q, s.reduce(q)[0]).max()), 1)

    def test_plane_fit_returns_poses_that_reproduce_patterns(self):
        s = self.s
        s.configure(fit_bandwidth=32, plane_stride=1, plane_iters=3,
                    accept_z=0, accept_step_deg=180)
        s.fit_plane()
        cells, phases, q, _ = s.fit_patterns()
        self.assertTrue(np.all(phases == 0))
        sims = s.simulate(q, cells=cells)
        correlation = [np.corrcoef(s.pattern(int(c), processed=False).ravel(), a.ravel())[0, 1]
                       for c, a in zip(cells, sims)]
        self.assertGreater(min(correlation), .99)
        self.assertGreater(float(angular_error(q, s.reduce(q)[0]).max()), 1)

    def test_cubic_and_polar_phases_also_keep_raw_winners(self):
        s = self.s
        for folder, name in (("nickel", "Ni"), ("gan", "GaN")):
            with self.subTest(phase=name):
                s.clear_phases()
                s.add_phase(ROOT / "testdata" / folder / f"{name}_master_20kV.npz")
                s.synthesize()
                candidate = s.index_cells(np.arange(12))["quaternions"][0]
                result = s.index()
                if callable(getattr(s, "export_snapshot", None)):
                    snapshot = s.export_snapshot()
                    self.assert_same_pose(np.asarray(snapshot["candidates"]["quaternions"]).reshape(-1, 4), candidate)
                self.assert_same_pose(result["quaternions"].reshape(-1, 4), candidate)
                self.assert_same_pose(rotation.byEuler(result["euler"].reshape(-1, 3)).inv().data, candidate)


if __name__ == "__main__":
    unittest.main()
