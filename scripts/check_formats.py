#!/usr/bin/env python3
"""Check the engine's scan readers against kikuchipy on files you have.

    python scripts/check_formats.py FILE [FILE ...]
    python scripts/check_formats.py Pattern.dat+Setting.txt     # side files joined with +

Opens each scan with zebsdsolve and with kikuchipy (which must be installed) and compares the
map shape, the pattern shape and a few patterns pixel for pixel. A development aid: the test
suite itself needs neither kikuchipy nor files beyond testdata/.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
import zebsdsolve as ze  # noqa: E402


def main():
    import kikuchipy as kp

    bad = 0
    for arg in sys.argv[1:]:
        paths = arg.split("+")
        name = Path(paths[0]).name
        try:
            with ze.Solver() as s:
                s.configure(dynamic_background="none", static_background="none", mask=False)
                s.open(*paths)
                info = s.scan
                n = info["rows"] * info["cols"]
                cells = sorted({0, min(1, n - 1), n // 2, n - 1})
                mine = np.stack([s.pattern(c, processed=False) for c in cells])
                geo = s.geometry
            ref = kp.load(paths[0], lazy=True)
            nav = ref.axes_manager.navigation_shape[::-1] or (1,)
            rows, cols = (nav if len(nav) == 2 else (1, nav[0]))
            data = ref.data.reshape(rows * cols, *ref.data.shape[-2:])
            theirs = np.stack([np.asarray(data[c], dtype=np.float32) for c in cells])
            same_shape = (info["rows"], info["cols"]) == (rows, cols) and info["pattern_shape"] == tuple(ref.data.shape[-2:])
            diff = float(np.abs(mine - theirs).max()) if mine.shape == theirs.shape else float("nan")
            ok = same_shape and diff < 1e-3
            pc = np.asarray(ref.detector.pc).reshape(-1, 3).mean(0)
            print(f"{'ok  ' if ok else 'FAIL'} {name}: {info['format']}, {info['rows']} × {info['cols']} of {info['pattern_shape']} {info['dtype']}"
                  f" (kikuchipy {rows} × {cols} of {tuple(ref.data.shape[-2:])}), max pixel difference {diff:g};"
                  f" pc {np.round(geo['pc'], 4)} (kikuchipy, Bruker convention {np.round(pc, 4)})")
            bad += not ok
        except Exception as e:  # report and go on to the next file
            print(f"FAIL {name}: {type(e).__name__}: {e}")
            bad += 1
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
