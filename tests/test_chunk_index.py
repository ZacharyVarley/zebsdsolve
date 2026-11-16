"""h5py oracle for targeted, interleaved and complete HDF5 chunk lookup.

Generated fixtures require h5py (development only); no crystallographic changes.
"""
from pathlib import Path
import sys
import tempfile
import unittest
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'python'))
from zebsdsolve import Solver, Error


def fixture(path, chunked=True):
    n = 2048
    expected = np.zeros((n, 5, 7), dtype=np.uint16)
    with h5py.File(path, 'w', libver='earliest') as f:
        h = f.create_group('Scan 1/EBSD/Header')
        for key, value in [('nRows', 32), ('nColumns', 64), ('Pattern Height', 5), ('Pattern Width', 7)]: h.create_dataset(key,data=[value])
        options = dict(chunks=(1,3,4), compression='gzip', shuffle=True, fletcher32=True) if chunked else {}
        d = f.create_dataset('Scan 1/EBSD/Data/Pattern', shape=expected.shape, dtype='u2', **options)
        for i in range(n):
            # Missing leading/interior/trailing chunks, and edge chunks on both image axes.
            if i % 7 in (0, 3) or i == n-1: continue
            expected[i] = (i+np.arange(35).reshape(5,7)) % 65535
            d[i] = expected[i]
    return expected


class ChunkIndex(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='zebsdsolve-chunks-')
        cls.path = Path(cls.tmp.name)/'split-compressed.h5'
        cls.expected = fixture(cls.path)

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def solver(self, path=None):
        s = Solver(); self.addCleanup(s.close)
        s.configure(binning=1, static_background='none', dynamic_background='none', mask=False)
        s.open(path or self.path)
        return s

    def test_targeted_interleaved_and_complete_are_identical(self):
        s = self.solver()
        before = s.index_preparation
        for cell in [0,1,2,3,511,1024,2046,2047]:
            np.testing.assert_array_equal(s.pattern(cell,processed=False,full=True), self.expected[cell])
        targeted = s.index_preparation
        self.assertFalse(targeted['ready'])
        self.assertEqual(targeted['visited'], 0)
        self.assertLess(targeted['reads']-before['reads'], 40)
        steps = 0
        while True:
            before = s.index_preparation
            complete = s.prepare_index_step(max_nodes=2,max_bytes=16384)
            after = s.index_preparation
            self.assertLessEqual(after['reads']-before['reads'], 1)
            self.assertLessEqual(after['bytes']-before['bytes'], 16384)
            self.assertLessEqual(after['visited']-before['visited'], 2)
            steps += 1
            if complete: break
            self.assertFalse(after['ready'])
            cell = (steps*79) % 2048
            np.testing.assert_array_equal(s.pattern(cell,processed=False,full=True), self.expected[cell])
            self.assertLess(steps,1000)
        self.assertGreater(steps, 10)
        self.assertGreater(s.index_preparation['visited'], 10)
        self.assertTrue(s.index_preparation['ready'])
        self.assertEqual(s.index_preparation['pending'], 0)
        for cell in [0,1,127,500,1000,2046,2047]:
            before = s.index_preparation
            np.testing.assert_array_equal(s.pattern(cell,processed=False,full=True), self.expected[cell])
            # Complete lookup needs only the payload read (zero for missing chunks).
            self.assertLessEqual(s.index_preparation['reads']-before['reads'], 1)
        self.assertTrue(s.prepare_index_step())

    def test_small_budget_does_not_poison_previews_and_scan_switch_resets(self):
        s = self.solver()
        with self.assertRaisesRegex(Error, 'ChunkIndexBudgetTooSmall'): s.prepare_index_step(max_bytes=1)
        self.assertFalse(s.index_preparation['failed'])
        np.testing.assert_array_equal(s.pattern(21,processed=False,full=True), self.expected[21])
        while not s.prepare_index_step(max_bytes=8*1024*1024):
            pass
        self.assertTrue(s.index_preparation['ready'])
        s.open(self.path)
        self.assertFalse(s.index_preparation['ready'])
        self.assertEqual(s.index_preparation['visited'],0)
        # A cold whole-dataset read uses the bounded full walk, including holes.
        mean, std = s.pattern_stats()
        self.assertTrue(s.index_preparation['ready'])
        np.testing.assert_allclose(mean.ravel(), self.expected.mean(axis=(1,2)), rtol=1e-6)
        np.testing.assert_allclose(std.ravel(), self.expected.std(axis=(1,2)), rtol=1e-6)


if __name__ == '__main__': unittest.main(verbosity=2)
