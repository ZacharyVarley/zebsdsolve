"""The HDF5 reader finds a chunked scan's patterns by reading only the part of its chunk tree a
read needs: every pattern read that way, scattered or in batches, is what h5py reads, for one
pattern per chunk, for chunks split along both image axes (partial chunks at the edges; deflate,
shuffle and Fletcher-32), for chunks of several patterns, and for chunks never written. Each map
has thousands of chunks, so its tree has inner nodes. Needs NumPy and h5py, and the native build
(run from a checkout, or with zebsdsolve installed)."""
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from zebsdsolve import Solver

ROWS, COLS, H, W = 60, 80, 8, 10


def write(path, *, chunks=(1, H, W), written=None, **filters):
    """A kikuchipy-style h5ebsd file of ROWS × COLS patterns, each different, its pattern dataset
    chunked as asked (a version 1 B-tree of chunks), writing only the patterns `written`."""
    rng = np.random.default_rng(7)
    data = rng.integers(1, 255, (ROWS * COLS, H, W), dtype=np.uint8)
    with h5py.File(path, "w", libver="earliest") as f:
        head = f.create_group("Scan 1/EBSD/Header")
        head.create_dataset("n_rows", data=[ROWS])
        head.create_dataset("n_columns", data=[COLS])
        d = f["Scan 1/EBSD"].create_group("Data").create_dataset(
            "patterns", shape=data.shape, dtype=data.dtype, chunks=chunks, **filters)
        if written is None:
            d[...] = data
        else:
            keep = np.zeros(len(data), bool)
            keep[written] = True
            data[~keep] = 0  # chunks never written read as the fill value
            for a, b in runs(np.flatnonzero(keep)):
                d[a:b] = data[a:b]
    return data


def runs(idx):
    """[a, b) runs of consecutive indices."""
    cut = np.flatnonzero(np.diff(idx) != 1) + 1
    return [(int(r[0]), int(r[-1]) + 1) for r in np.split(idx, cut)]


class ChunkTree(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="zebsdsolve-chunks-")
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.s = Solver()
        self.addCleanup(self.s.close)

    def check(self, name, **kw):
        path = self.dir / f"{name}.h5"
        truth = write(path, **kw)
        self.s.open(path)
        n = ROWS * COLS
        rng = np.random.default_rng(1)
        # one at a time and scattered (each a new path through the tree), then neighbours of
        # ones read, the last and the first
        cells = list(rng.choice(n, 40, replace=False)) + [5, 6, n - 1, 0]
        for c in cells:
            np.testing.assert_array_equal(self.s.pattern(int(c), processed=False, full=True), truth[c], err_msg=f"{name}: cell {c}")
        # a batch across the whole map, most of it not read yet, and then every pattern
        for batch in (np.arange(1, n, 3), np.arange(n)):
            mean, _ = self.s.pattern_stats(batch)
            np.testing.assert_allclose(mean, truth[batch].reshape(len(batch), -1).mean(1), rtol=1e-5, err_msg=name)

    def test_one_pattern_per_chunk(self):
        self.check("patterns")

    def test_chunks_split_along_the_image(self):
        self.check("tiles", chunks=(1, 3, 4), compression="gzip", shuffle=True, fletcher32=True)

    def test_chunks_of_several_patterns(self):
        self.check("blocks", chunks=(7, H, W), compression="gzip")

    def test_chunks_never_written(self):
        self.check("sparse", chunks=(1, 3, 4), written=np.r_[1:50, 700:701, 1500:1900, 3000:3001, ROWS * COLS - 1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
