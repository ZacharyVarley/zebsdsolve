"""The plane of source points on kikuchipy's ni_gain 1: from its nine full-resolution calibration
patterns, and from the patterns of the scan itself. A development aid, like check_formats.py:
it prints the two planes side by side (docs/how-it-works.md, "The plane of source points").

    python scripts/plane_ni_gain.py [path to the ni_gain/1 directory]

The directory is NORDIF's: Pattern.dat, Setting.txt, the `Calibration (x,y).bmp` patterns and
`Background calibration pattern.bmp`. kikuchipy downloads it with `kp.data.ni_gain(1)` and
`kp.data.ni_gain_calibration(1)`; by default it is looked for in kikuchipy's cache.
"""
import glob
import os
import re
import sys

import numpy as np
from PIL import Image

import zebsdsolve as ze

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NI = os.path.join(ROOT, "testdata", "nickel", "Ni_master_20kV.npz")


def find():
    if len(sys.argv) > 1:
        return sys.argv[1]
    hits = glob.glob(os.path.join(os.path.expanduser("~"), "AppData", "Local", "kikuchipy", "kikuchipy", "Cache", "*", "data", "ni_gain", "1"))
    hits += glob.glob(os.path.join(os.path.expanduser("~"), ".cache", "kikuchipy", "*", "data", "ni_gain", "1"))
    if not hits:
        raise SystemExit("ni_gain/1 not found: pass its directory")
    return sorted(hits)[-1]


def main():
    d = find()
    txt = open(os.path.join(d, "Setting.txt"), encoding="latin-1").read()
    cal = np.array([(int(a), int(b)) for a, b in re.findall(r"Calibration \((\d+),(\d+)\)\t", txt)], float)
    # the region of interest in the electron image: its corner (pixels) and the pixels to a map cell
    top = float(re.search(r"Top\t[\d.]+ \((\d+)\)", txt).group(1))
    left = float(re.search(r"Left\t[\d.]+ \((\d+)\)", txt).group(1))
    width = float(re.search(r"Width\t[\d.]+ \((\d+)\)", txt).group(1))
    rows, cols = (int(v) for v in re.search(r"Number of samples\t(\d+)x(\d+)", txt).groups())
    per_cell = width / cols

    # each calibration pattern by itself, its source point and its orientation fitted: the nine as
    # one 3 × 3 scan whose steps are nothing, so that every one starts from the vote's pattern
    # centre, and each one's own source point read from the plane fit's points
    bg = np.asarray(Image.open(os.path.join(d, "Background calibration pattern.bmp")).convert("F"))
    s = ze.Solver()
    s.add_phase(NI)
    pats = [np.asarray(Image.open(os.path.join(d, f"Calibration ({x},{y}).bmp")).convert("F")) / np.maximum(bg, 1) for x, y in cal.astype(int)]
    s.set_patterns(np.stack(pats).reshape(3, 3, *pats[0].shape).astype(np.float32))
    s.configure(static_background="none", fit_grid=3, plane_stride=1, plane_master_auto=False, plane_outlier=0, accept_z=-1e9, accept_step_deg=1e9)
    s.set_geometry(pc=(0.42, 0.21, 0.50), rot=(0, 0, 0), sample_tilt_deg=70, detector_tilt_deg=0, px_um=70, step_x_um=0, step_y_um=0)
    s.fit_pc()
    s.fit_plane()
    pts = s.plane_points()
    pcs = pts["pc"][np.argsort(pts["cells"])]
    s.close()
    print(f"nine calibration patterns: source points' mean {np.round(pcs.mean(0), 5)}, spread {np.round(pcs.std(0), 5)}")
    col, row = (cal[:, 0] - left) / per_cell, (cal[:, 1] - top) / per_cell
    A = np.column_stack([col, row, np.ones(len(col))])

    def slopes(idx):
        c = np.linalg.lstsq(A[idx], pcs[idx], rcond=None)[0]
        return np.concatenate([c[0], c[1]])
    full = slopes(np.arange(len(col)))
    jk = np.array([slopes(np.array([i for i in range(len(col)) if i != k])) for k in range(len(col))])
    se = np.sqrt((len(col) - 1) / len(col) * ((jk - jk.mean(0)) ** 2).sum(0))

    # the scan's own patterns
    s = ze.Solver()
    s.open(os.path.join(d, "Pattern.dat"))
    s.add_phase(NI)
    s.configure(clahe=True, plane_stride=1, plane_patterns_max=0)
    s.set_geometry(pc=(0.42, 0.21, 0.50), px_um=560)
    s.fit_pc()
    r = s.fit_plane()
    P = r["plane"]
    w = r["said"]
    g = s.geometry
    f0 = g["step_x_um"] / (g["px_um"] * 60)
    al = np.radians(90 - g["sample_tilt_deg"])
    nominal = [-f0, 0, 0, 0, -f0 * np.cos(al), f0 * np.sin(al)]
    names = ["PCx per column", "PCy per column", "PCz per column", "PCx per row", "PCy per row", "PCz per row"]
    mine = np.concatenate([P[:3, 0], P[:3, 1]])
    print(f"\nthe source's movement per map cell, 1e-5 of the pattern ({r['n_used']} of {r['n_patterns']} map patterns used)")
    print(f"{'':16s} {'nine calibration':>22s} {'map patterns':>14s} {'as given':>10s}")
    for k, n in enumerate(names):
        print(f"{n:16s} {1e5 * full[k]:+10.3f} ± {1e5 * se[k]:.3f} {1e5 * mine[k]:+14.3f} {1e5 * nominal[k]:+10.3f}")
    print(f"\nthe map's plane says: tilt {np.degrees(w['rot'][1]):+.2f} ± {np.degrees(w['rot_se'][1]):.2f} deg from nominal, the scan turned {np.degrees(w['rot'][2]):+.2f} ± {np.degrees(w['rot_se'][2]):.2f} deg in its plane, "
          f"steps {w['step'][0]:.3f} ± {w['step_se'][0]:.3f} (row) and {w['step'][1]:.3f} ± {w['step_se'][1]:.3f} (column) of the given ones")


if __name__ == "__main__":
    main()
