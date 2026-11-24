#!/usr/bin/env python3
"""A fixed set of synthetic scans, each with its answer, to decide a change on.

    python scripts/bench.py                         # every case; a table, and dist/bench.json
    python scripts/bench.py --out before.json --cases 60-noisy,faint
    python scripts/bench.py --compare before.json after.json
    python scripts/bench.py --root ../other-checkout --out other.json

Every scan is the engine's toy scan (`Solver.synthesize`) from fixed seeds, so the same
checkout gives the same scans and, but for the order of sums on the device, the same results.
The cases span the pattern's size (60 × 60 to 240 × 320, one not square), the counting noise
and the share of the intensity that is pattern rather than background, a detector rotated off
its nominal attitude, a map wide enough for the beam's travel to move the pattern centre a
lot, a hexagonal phase without a centre of symmetry, two phases with blank cells, patterns
faint enough that much of the map cannot be indexed, faint large patterns searched binned, a
start further off than the others, and maps mostly of cells with no crystal (as a real scan's
pores, another phase or a sample's edge make them), one of them faint as well.

Each case starts from the true geometry with the pattern centre moved off (START_OFF) and
runs what a user would: the pattern-centre vote, the plane fit from the vote's geometry, the
plane fit straight from the start (no vote: how a fit far off fares), Auto from the start, and
indexing after Auto. Each is measured against the answer: the pattern centre's error (the
largest of x, y, z, at the map's middle), the plane fit's rounds and the patterns it kept, how
many indexed orientations are right and how far off, how many cells were accepted, and the
time each took. A stage that fails records its error.

The library is the checkout's own (python/ on the path, zig-out beside it), or the one
ZEBSDSOLVE_LIB names; ZEBSDSOLVE_WGPU names wgpu-native when .deps/ has none.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
# (--root: another checkout's package and library, measured on these cases)
_root = sys.argv[sys.argv.index("--root") + 1] if "--root" in sys.argv else None
ROOT = Path(_root).resolve() if _root else HERE
sys.path.insert(0, str(ROOT / "python"))
import zebsdsolve as ze  # noqa: E402

NI = HERE / "testdata" / "nickel" / "Ni_master_20kV.npz"
GAN = HERE / "testdata" / "gan" / "GaN_master_20kV.npz"
PC_TRUE = (0.46, 0.27, 0.55)
START_OFF = (0.012, -0.010, 0.012)

# name: masters, the toy's settings, and what else differs (the detector's rotation, the
# steps, settings, the start's offset)
BASE = dict(toy_seed=3, toy_rows=24, toy_cols=32, toy_grains=40, toy_blank=0, toy_spread_deg=0)
CASES = {
    "60-clean": ([NI], dict(toy_height=60, toy_width=60, toy_counts=200, toy_contrast=1, toy_background_sigma=0), {}),
    "60-noisy": ([NI], dict(toy_height=60, toy_width=60, toy_counts=8, toy_contrast=0.5, toy_background_sigma=0.4), {}),
    "120-faint-band": ([NI], dict(toy_height=120, toy_width=120, toy_counts=40, toy_contrast=0.25, toy_background_sigma=0.4), {}),
    "96x128-rotated": ([NI], dict(toy_height=96, toy_width=128, toy_counts=60, toy_contrast=0.4, toy_background_sigma=0.3), {"rot_deg": (0.5, -1.0, 1.0)}),
    "240x320": ([NI], dict(toy_height=240, toy_width=320, toy_rows=16, toy_cols=20, toy_grains=24, toy_counts=20, toy_contrast=0.3, toy_background_sigma=0.4), {}),
    "wide-map": ([NI], dict(toy_height=60, toy_width=72, toy_rows=18, toy_cols=24, toy_counts=200, toy_contrast=0.3, toy_background_sigma=0.4), {"step_um": 150}),
    "gan-80": ([GAN], dict(toy_height=80, toy_width=80, toy_counts=50, toy_contrast=0.5, toy_background_sigma=0.3), {}),
    "two-phase-blank": ([NI, GAN], dict(toy_height=60, toy_width=60, toy_counts=40, toy_contrast=0.5, toy_background_sigma=0.3, toy_blank=0.2, toy_spread_deg=0.5), {}),
    "faint": ([NI], dict(toy_height=60, toy_width=60, toy_counts=3, toy_contrast=0.2, toy_background_sigma=0.4), {}),
    "faint-160-binned": ([NI], dict(toy_height=160, toy_width=160, toy_rows=20, toy_cols=24, toy_counts=4, toy_contrast=0.25, toy_background_sigma=0.4), {"settings": dict(binning=4, full_resolution="refine")}),
    "far-start": ([NI], dict(toy_height=60, toy_width=60, toy_counts=8, toy_contrast=0.5, toy_background_sigma=0.4), {"start_off": (0.045, -0.04, 0.035)}),
    "mostly-blank": ([NI], dict(toy_height=60, toy_width=60, toy_counts=8, toy_contrast=0.5, toy_background_sigma=0.4, toy_blank=0.8), {}),
    "faint-mostly-blank": ([NI], dict(toy_height=60, toy_width=60, toy_counts=3, toy_contrast=0.2, toy_background_sigma=0.4, toy_blank=0.6), {}),
}


def pc_err(pc, truth):
    return float(np.abs(np.asarray(pc, np.float64) - np.asarray(truth, np.float64)).max())


def timed(f):
    t = time.perf_counter()
    out = f()
    return out, time.perf_counter() - t


def run_case(name, masters, toy, extra):
    s = ze.Solver()
    out = {}
    try:
        for m in masters:
            s.add_phase(str(m))
        rot = tuple(np.radians(extra.get("rot_deg", (0, 0, 0))))
        step = extra.get("step_um", 2)
        s.configure(**extra.get("settings", {}))
        s.set_geometry(pc=PC_TRUE, rot=rot, sample_tilt_deg=70, detector_tilt_deg=0, px_um=0, step_x_um=step, step_y_um=step)
        truth, out["synth_s"] = timed(lambda: s.synthesize(**{**BASE, **toy}))
        q_true, ph_true = truth["quaternions"].reshape(-1, 4), truth["phase"].ravel()
        start = dict(pc=tuple(p + d for p, d in zip(PC_TRUE, extra.get("start_off", START_OFF))), rot=rot, plane_kind="none")

        def stage(key, f):
            try:
                r, out[f"{key}_s"] = timed(f)
                return r
            except ze.Error as e:
                out[f"{key}_error"] = str(e).split(": ")[-1]
                return None

        s.set_geometry(**start)
        v = stage("vote", s.fit_pc)
        if v is not None:
            out["vote_pc_err"] = pc_err(v["pc"], PC_TRUE)
            p = stage("plane", s.fit_plane)
            if p is not None:
                out.update(plane_pc_err=pc_err(p["pc"], PC_TRUE), plane_rounds=int(p["iterations"]), plane_used=f"{p['n_used']}/{p['n_patterns']}", plane_ncc=float(p["ncc"]))
        # the plane fit straight from the start, no vote before it: a geometry this far off
        s.set_geometry(**start)
        c = stage("cold", s.fit_plane)
        if c is not None:
            out.update(cold_pc_err=pc_err(c["pc"], PC_TRUE), cold_rounds=int(c["iterations"]), cold_used=f"{c['n_used']}/{c['n_patterns']}")
        s.set_geometry(**start)
        a = stage("auto", s.fit_geometry)
        if a is not None:
            out.update(auto_model=a["model"], auto_pc_err=pc_err(a["pc"], PC_TRUE), auto_rounds=int(a["iterations"]))
        r = stage("index", s.index)
        if r is not None:
            ph = r["phase"].ravel()
            crystal = ph_true >= 0
            right = crystal & (ph == ph_true)
            d = np.full(ph.size, np.inf)
            for k in range(len(masters)):
                mk = right & (ph_true == k)
                if mk.any():
                    d[mk] = np.degrees(s.disorientation(r["quaternions"].reshape(-1, 4)[mk], q_true[mk], phase=k))
            acc = r["sweep"].ravel() >= 0
            n = max(int(crystal.sum()), 1)
            out.update(index_median_deg=float(np.median(d[crystal])), index_within_1deg=float((d[crystal] < 1).mean()),
                       accepted=float(acc[crystal].sum() / n), accepted_right=float((acc & (d < 1))[crystal].sum() / n))
            if (~crystal).any():
                out["accepted_blank"] = float(acc[~crystal].mean())
    finally:
        s.close()
    return out


SHOW = [("vote_pc_err", "{:.5f}"), ("vote_s", "{:.2f}"), ("plane_pc_err", "{:.5f}"), ("plane_rounds", "{}"), ("plane_used", "{}"), ("plane_s", "{:.2f}"),
        ("cold_pc_err", "{:.5f}"), ("cold_rounds", "{}"), ("cold_used", "{}"), ("cold_s", "{:.2f}"),
        ("auto_model", "{}"), ("auto_pc_err", "{:.5f}"), ("auto_s", "{:.2f}"), ("index_median_deg", "{:.3f}"), ("index_within_1deg", "{:.3f}"),
        ("accepted", "{:.3f}"), ("accepted_right", "{:.3f}"), ("accepted_blank", "{:.3f}"), ("index_s", "{:.2f}")]


def fmt(case, key, f):
    if key in case:
        return f.format(case[key])
    stage = key.split("_")[0]
    return "ERR " + case[f"{stage}_error"] if f"{stage}_error" in case else "–"


def table(runs, labels):
    names = list(dict.fromkeys(n for r in runs for n in r["cases"]))
    width = max([14] + [len(fmt(r["cases"].get(n, {}), k, f)) for r in runs for n in names for k, f in SHOW])
    for name in names:
        print(f"\n{name}")
        for key, f in SHOW:
            vals = [fmt(r["cases"].get(name, {}), key, f) for r in runs]
            if all(v == "–" for v in vals):
                continue
            mark = "  *" if len(set(vals)) > 1 and not key.endswith("_s") else ""
            print(f"  {key:18s}" + "".join(f"  {v:>{width}s}" for v in vals) + mark)
    if len(runs) > 1:
        print("\n(columns: " + ", ".join(labels) + "; * where they differ, times aside)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(HERE / "dist" / "bench.json"))
    ap.add_argument("--cases", help="comma-separated names (default: all): " + ", ".join(CASES))
    ap.add_argument("--compare", nargs="+", metavar="JSON", help="print saved runs side by side instead")
    ap.add_argument("--root", help="another checkout to measure (its python/ and zig-out/)")
    a = ap.parse_args()
    if a.compare:
        runs = [json.loads(Path(p).read_text()) for p in a.compare]
        table(runs, [f"{Path(p).stem} ({r['commit']})" for p, r in zip(a.compare, runs)])
        return
    wanted = a.cases.split(",") if a.cases else list(CASES)
    try:
        commit = subprocess.run(["git", "describe", "--always", "--dirty"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = "?"
    run = {"commit": commit, "library": os.environ.get("ZEBSDSOLVE_LIB", str(ROOT / "zig-out")), "cases": {}}
    # (the kernels are compiled once, before anything is timed)
    run_case("warm-up", [NI], dict(toy_rows=4, toy_cols=4, toy_height=60, toy_width=60, toy_grains=2), {})
    for name in wanted:
        masters, toy, extra = CASES[name]
        t = time.perf_counter()
        run["cases"][name] = run_case(name, masters, toy, extra)
        print(f"{name}: {time.perf_counter() - t:.1f} s", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(run, indent=1))
    table([run], [commit])


if __name__ == "__main__":
    main()
