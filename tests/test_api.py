#!/usr/bin/env python3
"""The Python API on the GPU, against ground truth.

    python tests/test_api.py            (installed wheel, or a checkout after `zig build`)

The scans and master patterns are the repository's testdata/ ($ZEBSDSOLVE_TESTDATA when the test
runs from elsewhere, as CI does against an installed wheel).

Synthetic: patterns simulated from the nickel master at known orientations under a known
geometry, with noise. Indexing must recover the orientations, at the stored size and reduced;
the pattern-centre vote and the pose fit must recover the geometry from a wrong start; two
candidate phases must be told apart; patterns of pure noise must not be accepted, in one pass
or in sweeps; averaging with neighbours (NLPAR) must be the method as written out in NumPy
here, quiet a noisy map, and leave a grain one cell wide its own patterns.
Real: the nickel scan (kikuchipy's `nickel_ebsd_large`), against the orientations and the
pattern centre kikuchipy refined for it; and its filters against their definitions (CLAHE as
the kernels do it, written out in NumPy here).

Scores come from f32 arithmetic on the GPU: every comparison here has a tolerance.
"""
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
try:
    import zebsdsolve as ze
except ImportError:
    sys.path.insert(0, str(ROOT / "python"))
    import zebsdsolve as ze

DATA = Path(os.environ.get("ZEBSDSOLVE_TESTDATA", ROOT / "testdata"))
NI = DATA / "nickel" / "Ni_master_20kV.npz"
GAN = DATA / "gan" / "GaN_master_20kV.npz"
SCAN = DATA / "nickel" / "patterns_v2.h5"

failures = []


def check(name, ok, info=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name}{'  ' + info if info else ''}", flush=True)
    if not ok:
        failures.append(name)


def random_quaternions(rng, n):
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q[q[:, 0] < 0] *= -1
    return q.astype(np.float32)


def synthetic(s, rng, rows, cols, pc, phases, noise):
    """A scan of rows × cols simulated 60 × 60 patterns: (patterns, true quaternions, true phase)."""
    n = rows * cols
    s.set_patterns(np.zeros((rows, cols, 60, 60), np.float32))
    s.set_geometry(pc=pc, sample_tilt_deg=70, detector_tilt_deg=0, px_um=70 * 8, step_x_um=20, step_y_um=20)
    q = random_quaternions(rng, n)
    truth = rng.integers(0, phases, n)
    sims = np.empty((n, 60, 60), np.float32)
    for p in range(phases):
        m = truth == p
        if m.any():
            sims[m] = s.simulate(q[m], cells=np.arange(n)[m], phase=p)
    sims += noise * sims.std() * rng.normal(size=sims.shape).astype(np.float32)
    return sims.reshape(rows, cols, 60, 60), q, truth


def unit(a, mask=None):
    """Zero mean and unit norm (under a mask), as the engine leaves filtered patterns."""
    m = np.ones(a.shape, bool) if mask is None else mask
    out = np.where(m, a - a[m].mean(), 0).astype(np.float64)
    return out / np.sqrt((out ** 2).sum())


def clahe_reference(img, grid, bins, clip):
    """CLAHE as the kernels define it (shaders/prep.wgsl), in
    float32 where the kernels' rounding decides a histogram level."""
    f = np.float32
    h, w = img.shape
    lo, hi = f(img.min()), f(img.max())
    v = np.clip((img.astype(f) - lo) * (f(1) / max(hi - lo, f(1e-20))), 0, 1).astype(f)
    pad_h, pad_w = (grid - h % grid) % grid, (grid - w % grid) % grid
    th, tw = max(1, (h + pad_h) // grid), max(1, (w + pad_w) // grid)

    def reflect(i, n, pad):
        x = i - pad
        x = np.where(x < 0, -x, x)
        x = np.where(x >= n, 2 * n - 2 - x, x)
        return np.clip(x, 0, n - 1)

    py, px = np.arange(h + pad_h), np.arange(w + pad_w)
    padded = v[reflect(py, h, pad_h)][:, reflect(px, w, pad_w)]
    level = np.clip(np.floor(padded * f(bins - 1)), 0, bins - 1).astype(int)
    hist = np.zeros((grid, grid, bins))
    np.add.at(hist, (np.broadcast_to((py // th)[:, None], level.shape), np.broadcast_to((px // tw)[None, :], level.shape), level), 1)
    voxels = th * tw
    limit = max(clip * voxels / bins, 1.0)
    c = np.minimum(hist, limit)
    clipped = voxels - c.sum(-1)
    residual = (clipped - bins * np.floor(clipped / bins)).astype(int)
    c = c + ((clipped - residual) / bins)[..., None] + (np.arange(bins) < residual[..., None])
    cdf = np.clip(np.cumsum(c, -1) / voxels, 0, 1)
    gy, gx = (np.arange(h) + 0.5) / h * (grid - 1), (np.arange(w) + 0.5) / w * (grid - 1)
    y0, x0 = np.floor(gy).astype(int), np.floor(gx).astype(int)
    y1, x1 = np.minimum(y0 + 1, grid - 1), np.minimum(x0 + 1, grid - 1)
    fy, fx = (gy - y0)[:, None], (gx - x0)[None, :]
    bf = v * f(bins - 1)
    b0 = np.clip(np.floor(bf), 0, bins - 2).astype(int)
    fb = bf - b0

    def look(yy, xx):
        t = cdf[yy[:, None], xx[None, :]]
        return np.take_along_axis(t, b0[..., None], -1)[..., 0] * (1 - fb) + np.take_along_axis(t, b0[..., None] + 1, -1)[..., 0] * fb

    return (look(y0, x0) * (1 - fx) + look(y0, x1) * fx) * (1 - fy) + (look(y1, x0) * (1 - fx) + look(y1, x1) * fx) * fy


def gauss_reference(img, sigma):
    """A Gaussian blur carried as far as the image goes, renormalized at the edges."""
    out = img.astype(np.float64)
    for axis, n in enumerate(img.shape):
        x = np.arange(n)
        g = np.exp(-(x[:, None] - x[None, :]) ** 2 / (2 * sigma ** 2))
        g /= g.sum(1, keepdims=True)
        out = g @ out if axis == 0 else out @ g.T
    return out


def nlpar_reference(pats, radius, passes, lam):
    """Non-local pattern averaging as shaders/nlpar.wgsl states it, over a whole map
    (rows, cols, h, w) at once: every pass on the patterns the last one left, each average
    carrying its shares of the stored patterns so that the noise in a difference is known.
    Returns the averages and those shares (rows, cols, side, side)."""
    R, C = pats.shape[:2]
    P = pats.reshape(R, C, -1).astype(np.float64)
    n = P.shape[-1]
    reach = radius * passes
    W = 2 * reach + 1
    window = [(dy, dx) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1) if (dy, dx) != (0, 0)]
    nearest = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]

    def shifted(a, dy, dx, fill, axes=(0, 1)):
        """out[i] = a[i + (dy, dx)] along the two axes, `fill` where that leaves the array."""
        out = np.full(a.shape, fill, a.dtype)
        src, dst = [slice(None)] * a.ndim, [slice(None)] * a.ndim
        for ax, d in zip(axes, (dy, dx)):
            m = a.shape[ax]
            src[ax], dst[ax] = slice(max(d, 0), m + min(d, 0)), slice(max(-d, 0), m + min(-d, 0))
        out[tuple(dst)] = a[tuple(src)]
        return out

    inside = np.ones((R, C), bool)
    least = np.min([np.where(shifted(inside, *o, False), ((P - shifted(P, *o, 0.0)) ** 2).mean(-1), np.inf) for o in nearest], axis=0)
    sigma2 = 0.5 * np.where(np.isfinite(least), least, 0.0)
    # the noise of the stored pattern at each offset from each cell, and each pattern's shares
    noise_at = np.stack([np.stack([shifted(sigma2, uy, ux, 0.0) for ux in range(-reach, reach + 1)], -1) for uy in range(-reach, reach + 1)], -2)
    shares = np.zeros((R, C, W, W))
    shares[:, :, reach, reach] = 1
    for _ in range(passes):
        acc, total, new = P.copy(), np.ones((R, C)), shares.copy()
        for o in window:
            theirs = shifted(shifted(shares, *o, 0.0), -o[0], -o[1], 0.0, axes=(2, 3))
            e = ((shares - theirs) ** 2 * noise_at).sum((2, 3))
            d = ((P - shifted(P, *o, 0.0)) ** 2).mean(-1)
            with np.errstate(divide="ignore", invalid="ignore"):
                w = np.where(d <= e, 1.0, np.where(e > 0, np.exp(-np.sqrt(n / 2) * (d / e - 1) / lam ** 2), 0.0))
            w = np.where(shifted(inside, *o, False), w, 0.0)
            acc += w[..., None] * shifted(P, *o, 0.0)
            new += w[..., None, None] * theirs
            total += w
        P, shares = acc / total[..., None], new / total[..., None, None]
    return P.reshape(pats.shape), shares


def main():
    rng = np.random.default_rng(7)
    t_all = time.perf_counter()
    s = ze.Solver()
    print("adapter:", s.adapter, " version:", ze.__version__, flush=True)

    # ── settings ──
    try:
        s.configure(no_such_option=1)
        check("an unknown setting is refused", False)
    except ze.Error:
        check("an unknown setting is refused", True)
    try:
        s.configure(bandwidth=3)
        check("a setting out of range is refused", False)
    except ze.Error:
        check("a setting out of range is refused", s.settings["bandwidth"] == 64)

    # ── synthetic, one phase ──
    s.configure(static_background="none", dynamic_background="none")
    s.add_phase(NI)
    ph = s.phases[0]
    check("the nickel master loads", ph["point_group"] == 32 and ph["laue"] == "m-3m" and ph["side"] == 501, str(ph))
    m = s.master(0)
    check("the master is normalized over the sphere", abs(float(m.mean())) < 1e-3 and abs(float(m.std()) - 1) < 1e-3)

    pc_true = (0.44, 0.24, 0.53)
    pats, q_true, _ = synthetic(s, rng, 8, 12, pc_true, 1, noise=1.0)
    s.set_patterns(pats)
    t0 = time.perf_counter()
    r = s.index()
    dis = np.degrees(s.disorientation(r["quaternions"].reshape(-1, 4), q_true))
    check("spherical indexing recovers the orientations", np.median(dis) < 0.15 and dis.max() < 1.0,
          f"median {np.median(dis):.3f}°, max {dis.max():.3f}°, mean NCC {r['ncc'].mean():.3f}, {time.perf_counter() - t0:.2f} s")
    check("every cell is indexed, in the fundamental zone", (r["phase"] == 0).all() and (r["quaternions"][..., 0] >= np.cos(np.radians(62.81) / 2) - 1e-4).all())

    s.configure(refine=False)
    r0 = s.index()
    dis0 = np.degrees(s.disorientation(r0["quaternions"].reshape(-1, 4), q_true))
    check("the spherical peak alone is within the grid's reach", np.median(dis0) < 1.5 and np.percentile(dis0, 95) < 3, f"median {np.median(dis0):.3f}°")
    s.configure(refine=True)

    s.configure(method="dictionary", dict_grid=50)
    rd = s.index()
    disd = np.degrees(s.disorientation(rd["quaternions"].reshape(-1, 4), q_true))
    check("dictionary indexing recovers the orientations", np.median(disd) < 0.2 and (disd < 1).mean() > 0.9, f"median {np.median(disd):.3f}°, within 1°: {100 * (disd < 1).mean():.0f} %")
    s.configure(method="pca", dict_grid=50)
    rp = s.index()
    disp = np.degrees(s.disorientation(rp["quaternions"].reshape(-1, 4), q_true))
    learned = [e["text"] for e in s.events() if e["kind"] == "log" and "components" in e["text"]]
    check("a dictionary matched in its leading components recovers them too", np.median(disp) < 0.2 and (disp < 1).mean() > 0.9 and len(learned) == 1,
          f"median {np.median(disp):.3f}°, within 1°: {100 * (disp < 1).mean():.0f} %; {learned[0] if learned else ''}")
    rp2 = s.index()
    check("the components are learned once", not any("components" in e["text"] for e in s.events() if e["kind"] == "log") and np.allclose(rp2["quaternions"], rp["quaternions"], atol=1e-5))
    s.configure(method="spherical")

    # the static background as the patterns' own average: random orientations average flat
    s.configure(static_background="average")
    ra = s.index()
    disa = np.degrees(s.disorientation(ra["quaternions"].reshape(-1, 4), q_true))
    check("subtracting the average pattern leaves the orientations", np.median(disa) < 0.15 and disa.max() < 1.0, f"median {np.median(disa):.3f}°")
    s.configure(static_background="none")

    # ── reduced patterns ──
    s.configure(binning=2)
    check("binning halves the pattern the search sees", s.pattern_shape == (30, 30) and s.shape(full=True) == (60, 60) and s.pattern(5).shape == (30, 30) and s.pattern(5, full=True).shape == (60, 60))
    raw2, raw1 = s.pattern(5, processed=False), s.pattern(5, processed=False, full=True)
    check("a binned pattern is the mean of its blocks", np.allclose(raw2, raw1.reshape(30, 2, 30, 2).mean((1, 3)), atol=1e-4))
    rb = {}
    for mode in ("none", "scores", "refine"):
        s.configure(full_resolution=mode)
        rb[mode] = s.index()
        rb[mode]["dis"] = np.degrees(s.disorientation(rb[mode]["quaternions"].reshape(-1, 4), q_true))
    check("a search on binned patterns, refined at the stored size, loses nothing",
          np.median(rb["refine"]["dis"]) < 0.15 and rb["refine"]["dis"].max() < 1.0 and np.median(rb["none"]["dis"]) < 0.4 and np.allclose(rb["scores"]["dis"], rb["none"]["dis"], atol=0.05),
          f"median {np.median(rb['none']['dis']):.3f}° all binned, {np.median(rb['refine']['dis']):.3f}° refined at the stored size (it was {np.median(dis):.3f}° without binning)")
    check("scores taken at the stored size are the stored size's", abs(float(rb["scores"]["ncc"].mean() - rb["none"]["ncc"].mean())) > 1e-3 and rb["refine"]["ncc"].mean() > rb["scores"]["ncc"].mean() - 1e-3,
          f"mean NCC {rb['none']['ncc'].mean():.3f} binned, {rb['scores']['ncc'].mean():.3f} scored at the stored size, {rb['refine']['ncc'].mean():.3f} refined there")
    s.configure(binning=1.5, full_resolution="none")
    rr = s.index()
    disr = np.degrees(s.disorientation(rr["quaternions"].reshape(-1, 4), q_true))
    check("a factor that is not whole resamples by area", s.pattern_shape == (40, 40) and np.median(disr) < 0.3, f"60 → 40 pixels, median {np.median(disr):.3f}°")
    s.configure(binning=1)

    # ── acceptance, and sweeps ──
    # a third of the cells hold nothing but noise: no orientation fits them better than chance
    noise_cells = rng.choice(pats.shape[0] * pats.shape[1], 32, replace=False)
    mixed = pats.copy().reshape(-1, 60, 60)
    mixed[noise_cells] = rng.normal(size=(32, 60, 60)).astype(np.float32)
    s.set_patterns(mixed.reshape(pats.shape))
    is_noise = np.zeros(mixed.shape[0], bool)
    is_noise[noise_cells] = True
    r1 = s.index()
    z = r1["z"].ravel()
    cells = np.arange(len(z))
    one = s.index_cells(cells)
    check("matches stand far above chance, noise does not", z[~is_noise].min() > 15 and z[is_noise].max() < 8 and (r1["sweep"].ravel() >= 0).tolist() == (~is_noise).tolist()
          and np.degrees(one["step"][0][~is_noise]).max() < 0.2,
          f"z of the patterns {z[~is_noise].min():.1f} … {np.median(z[~is_noise]):.1f} (median), of the noise at most {z[is_noise].max():.1f}")
    s.configure(sweeps=3, sweep_bandwidth=16)
    check("sweeps climb to the bandwidth", [p[0] for p in s.sweep_plan] == [16, 32, 64], f"bandwidths {[p[0] for p in s.sweep_plan]}")
    visits = []
    r3 = s.index(lambda st: visits.append((st["sweep"], st["done"], st["total"])))
    sw = r3["sweep"].ravel()
    dis3 = np.degrees(s.disorientation(r3["quaternions"].reshape(-1, 4)[~is_noise], q_true[~is_noise]))
    last = [t for k, _, t in visits if k == 2]
    check("sweeps index what one pass does, visiting less at the full bandwidth", (sw[~is_noise] >= 0).all() and (sw[is_noise] < 0).all() and np.median(dis3) < 0.15 and dis3.max() < 1.0 and bool(last) and last[-1] < len(sw) // 2,
          f"accepted in sweeps 1, 2, 3: {[int((sw == k).sum()) for k in range(3)]}, never: {int((sw < 0).sum())} (the noise); median {np.median(dis3):.3f}°; the last sweep saw {last[-1] if last else '?'} of {len(sw)}")
    # a cheap first sweep of another kind: the dictionary's components, then the correlation
    s.configure(sweeps=2, sweep_method="pca", sweep_dict_grid=40)
    check("sweeps mix methods", [p[2] for p in s.sweep_plan] == ["pca", "spherical"], str(s.sweep_plan))
    rm = s.index()
    swm = rm["sweep"].ravel()
    dism = np.degrees(s.disorientation(rm["quaternions"].reshape(-1, 4)[~is_noise], q_true[~is_noise]))
    check("dictionary components first, the correlation for the rest", (swm[~is_noise] >= 0).all() and (swm[is_noise] < 0).all() and (swm == 0).sum() > 40 and np.median(dism) < 0.15 and dism.max() < 1.0,
          f"accepted by the components: {int((swm == 0).sum())}, by the correlation: {int((swm == 1).sum())}, never: {int((swm < 0).sum())} (the noise); median {np.median(dism):.3f}°")
    s.configure(sweeps=1, sweep_method="same")
    s.set_patterns(pats)

    # the fits, from a wrong start
    s.set_geometry(pc=(0.50, 0.32, 0.58))
    f = s.fit_pc()
    e = np.abs(np.array(f["pc"]) - pc_true)
    check("the pattern-centre vote finds the centre", e.max() < 0.012 and f["score"] > f["start_score"],
          f"pc {np.round(f['pc'], 4)}, error {np.round(e, 4)}, mean NCC {f['start_score']:.3f} → {f['score']:.3f}, {f['n_candidates']} candidates")
    scores, x, y, z = s.pc_scores()
    check("the votes are returned with their axes", scores.shape == (len(z), len(y), len(x)) and np.isfinite(scores).all())
    levels = s.pc_levels()
    last = levels[-1]
    # (the grid's choice is the ballots': the densest place of the patterns' best candidates)
    iz, iy, ix = (int(round(v)) for v in last["peak"][::-1])
    cast = sum(int((p >= 0.85 * p.max()).sum() >= 12) for p in last["pattern_scores"])
    check("each pattern casts ballots and the densest place wins", last["density"].shape == last["scores"].shape and last["ballots"] == 12 * cast and 0 < last["inliers"] <= last["ballots"]
          and last["pattern_scores"].shape == (25, *last["scores"].shape) and np.allclose(last["pattern_scores"].mean(0), last["scores"], atol=1e-5)
          and np.abs(np.array(np.unravel_index(int(np.argmax(last["density"])), last["density"].shape)) - (iz, iy, ix)).max() <= 2,
          f"{last['ballots']} ballots, {last['inliers']} of them by the winner")
    check("every grid of the search is kept", len(levels) >= 3 and np.array_equal(last["scores"], scores) and last["winner"] == (ix, iy, iz)
          and last["pattern_best"].shape == (25, 3) and np.isfinite(last["pattern_ncc"]).all()
          and np.ptp(levels[0]["x"]) > np.ptp(last["x"]),
          f"{len(levels)} grids, x spans {np.ptp(levels[0]['x']):.3f} → {np.ptp(last['x']):.4f}; {int((last['pattern_best'] == last['winner']).all(1).sum())} of 25 patterns score highest at the last winner")
    # the mean of the patterns' scores, the rule the ballots replace, still to be had
    s.set_geometry(pc=(0.50, 0.32, 0.58))
    s.configure(pc_vote="mean")
    fm = s.fit_pc()
    lm = s.pc_levels()[-1]
    check("the mean vote is still there", lm["density"] is None and lm["winner"] == tuple(int(v) for v in np.unravel_index(int(np.argmax(lm["scores"])), lm["scores"].shape)[::-1])
          and np.abs(np.array(fm["pc"]) - pc_true).max() < 0.02, f"error {np.round(np.abs(np.array(fm['pc']) - pc_true), 4)}")
    s.configure(pc_vote="ballots")
    s.set_geometry(pc=f["pc"])
    p = s.fit_pose()
    e = np.abs(np.array(p["pc"]) - pc_true)
    check("the pose fit refines the centre", e.max() < 0.002 and p["ncc"] >= p["start_ncc"],
          f"pc {np.round(p['pc'], 5)}, error {np.round(e, 5)}, mean NCC {p['start_ncc']:.4f} → {p['ncc']:.4f}, {p['iterations']} iterations")
    cells, phs, quats, ncc = s.fit_patterns()
    d = np.degrees(s.disorientation(quats, q_true[cells.astype(int)]))
    # (the pose fit's own lattice, pose_grid a side: every cell of this 8 × 12 map)
    check("the fit patterns' orientations are the true ones", len(cells) == 96 and np.median(d) < 0.15, f"median {np.median(d):.3f}° over {len(cells)} patterns")
    # each of the pose's degrees of freedom can be held: here the detector distance, from a
    # start off in all three coordinates
    kept = s.geometry
    s.set_geometry(pc=(pc_true[0] + 0.01, pc_true[1] - 0.01, pc_true[2] + 0.01))
    s.configure(pose_z=False)
    h = s.fit_pose()
    s.configure(pose_z=True)
    check("a held degree of freedom stays put", abs(h["pc"][2] - pc_true[2] - 0.01) < 1e-9 and abs(h["pc"][0] - pc_true[0]) < 0.008,
          f"z held at {h['pc'][2]:.5f}; x, y moved to {h['pc'][0]:.5f}, {h['pc'][1]:.5f}")
    s.set_geometry(pc=kept["pc"], rot=kept["rot"])

    # the whole pose: a detector turned in a way the fit is not told, on a map wide enough for the
    # beam's travel to show it (a tenth of the pattern), found from zero; and with the scale
    # freed as well (Sim(3)), from a pixel size 30 % off
    rng_pose = np.random.default_rng(11)
    rot_true = np.radians((1.0, -2.0, 1.5))
    s.set_patterns(np.zeros((20, 20, 60, 60), np.float32))
    s.set_geometry(pc=pc_true, rot=tuple(rot_true), sample_tilt_deg=70, detector_tilt_deg=0, px_um=560, step_x_um=200, step_y_um=200)
    qs = rng_pose.normal(size=(400, 4))
    qs /= np.linalg.norm(qs, axis=1, keepdims=True)
    sims = s.simulate(qs, cells=np.arange(400))
    s.set_patterns((sims + 0.5 * sims.std() * rng_pose.normal(size=sims.shape)).reshape(20, 20, 60, 60).astype(np.float32))
    s.configure(fit_grid=9)
    s.set_geometry(pc=(0.45, 0.23, 0.54), rot=(0, 0, 0))
    w = s.fit_pose()
    e_rot = np.abs(np.degrees(w["rot"]) - np.degrees(rot_true)).max()
    check("the pose fit finds the detector's rotation", e_rot < 0.3 and np.abs(np.array(w["pc"]) - pc_true).max() < 0.001,
          f"swing, tilt, twist {np.round(np.degrees(w['rot']), 3)} for {np.round(np.degrees(rot_true), 1)}; pc within {np.abs(np.array(w['pc']) - pc_true).max():.5f}")
    s.configure(pose_scale=True)
    s.set_geometry(pc=(0.45, 0.23, 0.54), rot=(0, 0, 0), px_um=400)
    w = s.fit_pose()
    e_rot = np.abs(np.degrees(w["rot"]) - np.degrees(rot_true)).max()
    check("and the pixel size with it", abs(w["px_um"] - 560) < 6 and w["start_px_um"] == 400 and abs(s.geometry["px_um"] - w["px_um"]) < 1e-9 and e_rot < 0.3,
          f"pixel {w['px_um']:.1f} µm for 560 from 400; rotation within {e_rot:.3f}°")
    s.configure(pose_scale=False, fit_grid=5)
    s.set_geometry(rot=(0, 0, 0))

    # ── averaging with neighbours (NLPAR) ──
    # two grains, and a third one cell wide running through each of them, under noise six times
    # the patterns' own contrast
    rows, cols = 14, 20
    s.set_patterns(np.zeros((rows, cols, 60, 60), np.float32))
    s.set_geometry(pc=pc_true, sample_tilt_deg=70, detector_tilt_deg=0, px_um=70 * 8, step_x_um=2, step_y_um=2)
    grain = np.zeros((rows, cols), int)
    grain[:, 10:] = 1
    grain[:, 4] = 2
    grain[7, 12:18] = 2
    q_grain = random_quaternions(rng, 3)[grain.ravel()]
    clean = s.simulate(q_grain, cells=np.arange(rows * cols)).reshape(rows, cols, 60, 60)
    noisy = (128 + 50 * clean / clean.std() + 300 * rng.normal(size=clean.shape)).astype(np.float32)
    s.set_patterns(noisy)
    s.set_geometry(pc=pc_true)
    thin = grain.ravel() == 2
    s.configure(mask=False)
    d = ze.Solver().settings
    check("the averaging's defaults are the published method", (d["nlpar"], d["nlpar_radius"], d["nlpar_iterations"], d["nlpar_lambda"]) == (False, 3, 1, 0.7), "off; when on, one pass over 7 × 7 at lambda 0.7")
    worst = worst_raw = worst_shares = 0.0
    views = []
    for radius, passes, lam in ((1, 3, 0.7), (2, 2, 0.9), (3, 1, 0.7)):
        s.configure(nlpar=True, nlpar_radius=radius, nlpar_iterations=passes, nlpar_lambda=lam)
        ref, shares = nlpar_reference(noisy, radius, passes, lam)
        for cell in (0, cols - 1, 4 * cols + 4, 7 * cols + 14, 6 * cols + 10, rows * cols - 1):
            worst = max(worst, float(np.abs(s.pattern(cell) - unit(ref[cell // cols, cell % cols])).max()))
            # the average before the filters, and what the engine says went into it
            worst_raw = max(worst_raw, float(np.abs(s.pattern(cell, "averaged") - ref[cell // cols, cell % cols]).max()) / float(noisy.std()))
            v = s.nlpar_view()
            mine = shares[cell // cols, cell % cols]
            worst_shares = max(worst_shares, float(np.abs(np.nan_to_num(v["shares"]) - mine).max()))
            views.append(v["shares"].shape == mine.shape and abs(v["own"] - mine[mine.shape[0] // 2, mine.shape[1] // 2]) < 1e-5 and 0 < v["left"] < 1 and v["noise"] > 0
                         # (no share where the window leaves the map)
                         and np.isnan(v["shares"]).any() == (min(cell // cols, cell % cols, rows - 1 - cell // cols, cols - 1 - cell % cols) < radius * passes))
    check("neighbour averaging is the method as written", worst < 2e-5, f"largest difference from the reference {worst:.1e} (pixels are about {1 / 60:.3f}), for 3 passes of 3 × 3, 2 of 5 × 5 and 1 of 7 × 7")
    check("a pattern can be looked at averaged and not yet filtered, with what went into it", worst_raw < 1e-5 and worst_shares < 1e-5 and all(views),
          f"the unfiltered averages differ from the reference by at most {worst_raw:.1e} of the patterns' spread, each one's shares of its neighbours by {worst_shares:.1e}")
    s.configure(mask=True, nlpar=False)
    s.pattern(0)
    check("a pattern not averaged has no such account", s.nlpar_view() is None and np.array_equal(s.pattern(5, "averaged"), s.pattern(5, False)))

    def averaged(**kw):
        s.configure(**kw)
        r = s.index()
        d = np.degrees(s.disorientation(r["quaternions"].reshape(-1, 4), q_grain))
        said = [e["text"] for e in s.events() if e["kind"] == "log" and "averaging" in e["text"]]
        left = float(said[0].split(" % of it left")[0].split()[-1]) if said else 100.0
        return r, d, left

    r0, d0, _ = averaged()
    ra, da, left = averaged(nlpar=True, nlpar_radius=1, nlpar_iterations=3, nlpar_lambda=0.7)
    check("it quiets the patterns and keeps a grain one cell wide", left < 60 and ra["ncc"].mean() > 1.5 * r0["ncc"].mean() and (da[thin] < 1).sum() >= max((d0[thin] < 1).sum(), 0.9 * thin.sum()) and (da < 1).mean() >= (d0 < 1).mean(),
          f"{left:.0f} % of the noise left; mean NCC {r0['ncc'].mean():.3f} → {ra['ncc'].mean():.3f}; within 1°: {100 * (d0 < 1).mean():.1f} % → {100 * (da < 1).mean():.1f} %, of the thin grain's {int(thin.sum())} cells {int((d0[thin] < 1).sum())} → {int((da[thin] < 1).sum())}")
    rw, dw, left_w = averaged(nlpar_radius=3, nlpar_iterations=1)
    check("one wide pass at the same lambda averages the thin grain away", (dw[thin] < 1).mean() < (da[thin] < 1).mean(),
          f"one pass of 7 × 7: {left_w:.0f} % of the noise left, of the thin grain's cells {int((dw[thin] < 1).sum())} are still its own")
    rb, db, _ = averaged(nlpar_radius=1, nlpar_iterations=3, batch=37)
    apart = np.degrees(s.disorientation(ra["quaternions"].reshape(-1, 4), rb["quaternions"].reshape(-1, 4)))
    check("the averages do not depend on the batch", apart.max() < 0.02 and np.abs(ra["ncc"] - rb["ncc"]).max() < 1e-3, f"batches of 256 and of 37: at most {apart.max():.4f}° and {np.abs(ra['ncc'] - rb['ncc']).max():.1e} in NCC apart")
    s.configure(nlpar=False, batch=256)

    # ── synthetic, two phases ──
    s.add_phase(GAN)
    gan = s.phases[1]
    check("the GaN master loads (hexagonal, polar)", gan["point_group"] == 25 and gan["laue"] == "6/mmm" and gan["side"] == 301, str(gan))
    pats, q_true, ph_true = synthetic(s, rng, 6, 10, pc_true, 2, noise=1.5)
    s.set_patterns(pats)
    for by in ("nmi", "ncc"):
        s.configure(phase_by=by)
        r = s.index()
        right = r["phase"].ravel() == ph_true
        dis = np.empty(len(ph_true))
        for k in (0, 1):
            mk = ph_true == k
            dis[mk] = np.degrees(s.disorientation(r["quaternions"].reshape(-1, 4)[mk], q_true[mk], phase=k))
        check(f"two phases are told apart by {by}", right.mean() > 0.97 and np.median(dis[right]) < 0.3, f"{100 * right.mean():.0f} % right, median {np.median(dis[right]):.3f}°")
    ni, gan = s.phases
    # an orientation and its stored form are the same orientation: they simulate the same pattern.
    # For GaN that takes the rotations of 6mm itself (six about c), not its Laue class's twelve,
    # half of which would turn the polar axis over
    qs = rng.normal(size=(200, 4)).astype(np.float32)
    qs /= np.linalg.norm(qs, axis=1, keepdims=True)

    def same(ph):
        a, b = s.simulate(qs, phase=ph), s.simulate(s.reduce(qs, phase=ph)[0], phase=ph)
        a = a - a.mean((1, 2), keepdims=True)
        b = b - b.mean((1, 2), keepdims=True)
        return float(((a * b).sum((1, 2)) / np.sqrt((a * a).sum((1, 2)) * (b * b).sum((1, 2)))).min())
    flipped = qs.astype(np.float64) @ np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]])  # q · (a half turn about the crystal's x) … as (w, x, y, z) rows
    d_flip = np.degrees(s.disorientation(flipped, qs, phase=1))
    check("a polar crystal's orientations are reduced by its own rotations", same(0) > 0.995 and same(1) > 0.995 and d_flip.min() > 1,
          f"patterns of an orientation and of its stored form correlate at least {same(0):.4f} (nickel), {same(1):.4f} (GaN); GaN turned over about x is {d_flip.min():.0f}° to {d_flip.max():.0f}° away, not 0")
    # a toy scan from the two masters: a seed is a scan, and its answer comes with it
    s.set_geometry(pc=(0.45, 0.25, 0.52), rot=(0, 0, 0), px_um=0)
    toy = s.synthesize(toy_seed=7, toy_rows=20, toy_cols=28, toy_height=60, toy_width=60, toy_grains=12, toy_counts=20)
    first = s.pattern(5, processed=False).copy()
    again = s.synthesize()
    same_scan = np.array_equal(toy["quaternions"], again["quaternions"]) and np.array_equal(toy["phase"], again["phase"]) and np.abs(s.pattern(5, processed=False) - first).max() < 1e-4 * np.abs(first).max()
    other = s.synthesize(toy_seed=8)
    s.synthesize(toy_seed=7)
    rt = s.index()
    found = rt["phase"].ravel()
    ok_ph = found == toy["phase"].ravel()
    dt = np.full(found.size, np.nan)
    for k in (0, 1):
        mk = ok_ph & (found == k)
        dt[mk] = np.degrees(s.disorientation(rt["quaternions"].reshape(-1, 4)[mk], toy["quaternions"].reshape(-1, 4)[mk], phase=k))
    grains = len(np.unique(toy["quaternions"].reshape(-1, 4), axis=0))
    # counted electrons: whole numbers, 20 a pixel on average, as noisy as counts are
    raw = np.stack([s.pattern(c, processed=False) for c in range(0, 560, 7)])
    counted = np.array_equal(raw, np.round(raw)) and raw.min() >= 0 and abs(raw.mean() - 20) < 1.5
    s.synthesize(toy_counts=0)
    clean = np.stack([s.pattern(c, processed=False) for c in range(0, 560, 7)])
    lam = (clean - clean.min()) * 20 / (clean.mean() - clean.min())
    shot = float(((raw - lam) ** 2).mean() / lam.mean())
    s.synthesize(toy_counts=20)
    check("a toy scan is made from a seed, with its answer", s.scan["rows"] == 20 and s.scan["cols"] == 28 and s.scan["pattern_shape"] == (60, 60) and same_scan and not np.array_equal(other["quaternions"], toy["quaternions"])
          and 2 <= grains <= 12 and set(np.unique(toy["phase"])) <= {0, 1} and ok_ph.mean() > 0.97 and np.nanmedian(dt) < 0.3 and counted and 0.85 < shot < 1.15,
          f"{grains} grains of {len(np.unique(toy['phase']))} phases, {raw.mean():.1f} electrons a pixel, noise variance {shot:.2f} × the mean; indexed: phase right for {100 * ok_ph.mean():.1f} %, median {np.nanmedian(dt):.3f}° from the answer")
    # the same as an experiment on fitting: a detector turned and a wide map (the true geometry),
    # an orientation per cell, a tenth of the cells blank, patterns that are mostly background,
    # not square, counted twice over with different noise; then the geometry moved off, for the
    # pose fit to bring back
    true_rot = np.radians((0.5, -1.0, 1.0))
    s.set_geometry(pc=(0.45, 0.25, 0.52), rot=tuple(true_rot), px_um=560, step_x_um=150, step_y_um=150)
    exp = dict(toy_seed=11, toy_rows=18, toy_cols=24, toy_height=60, toy_width=72, toy_grains=18 * 24, toy_blank=0.1, toy_contrast=0.3, toy_background_sigma=0.4, toy_counts=200)
    t5 = s.synthesize(toy_noise_seed=5, **exp)
    p5 = s.pattern(3, processed=False).copy()
    t6 = s.synthesize(toy_noise_seed=6)
    recount = np.array_equal(t5["quaternions"], t6["quaternions"]) and np.array_equal(t5["phase"], t6["phase"]) and not np.array_equal(s.pattern(3, processed=False), p5)
    blank = t6["phase"].ravel() < 0
    s.set_geometry(pc=(0.46, 0.24, 0.53), rot=(0, 0, 0))
    pf = s.fit_pose()
    e_pc = np.abs(np.array(pf["pc"]) - t6["geometry"]["pc"]).max()
    e_rot = np.degrees(np.abs(np.array(pf["rot"]) - t6["geometry"]["rot"]).max())
    rx = s.index()
    acc = rx["sweep"].ravel() >= 0
    okx = rx["phase"].ravel() == t6["phase"].ravel()
    dx = np.full(blank.size, np.nan)
    for k in (0, 1):
        mk = okx & (t6["phase"].ravel() == k)
        dx[mk] = np.degrees(s.disorientation(rx["quaternions"].reshape(-1, 4)[mk], t6["quaternions"].reshape(-1, 4)[mk], phase=k))
    check("a toy scan can be an experiment on fitting", s.scan["pattern_shape"] == (60, 72) and 0.05 < blank.mean() < 0.16 and recount and len(np.unique(t6["grain"])) > 300
          and np.allclose(t6["geometry"]["rot"], true_rot) and e_pc < 0.002 and e_rot < 0.5 and not acc[blank].any() and acc[~blank].mean() > 0.9 and np.nanmedian(dx) < 0.5,
          f"{100 * blank.mean():.0f} % blank cells, none accepted ({100 * acc[~blank].mean():.0f} % of the crystals are); the pose fit from 0.01 off and no rotation: pattern centre within {e_pc:.5f}, rotation within {e_rot:.3f}°; orientations a median {np.nanmedian(dx):.3f}° from the answer")
    s.configure(toy_blank=0, toy_contrast=1, toy_background_sigma=0, toy_noise_seed=0, toy_counts=20, toy_height=60, toy_width=60, toy_grains=24)
    check("the masters' symmetries shape the search", ni["n_fold"] == 4 and ni["inversion"] and gan["n_fold"] in (2, 6) and not gan["inversion"], f"nickel: order {ni['n_fold']} about z, inversion; GaN: order {gan['n_fold']} at least, none")
    s.close()

    # ── the real scan ──
    s = ze.Solver()
    s.open(SCAN)
    info = s.scan
    check("the nickel scan opens", info["rows"] == 55 and info["cols"] == 75 and info["pattern_shape"] == (60, 60) and info["format"] == "kikuchipy h5ebsd" and info["has_background"] and info["has_pc"], str(info))
    pc_file = np.array(s.geometry["pc"])
    check("its geometry comes from the file", np.allclose(pc_file, (0.42326, 0.21363, 0.50207), atol=1e-4) and s.geometry["sample_tilt_deg"] == 70)
    raw = s.pattern(0, processed=False)
    proc = s.pattern(0, processed=True)
    check("patterns come as stored and filtered", raw.shape == (60, 60) and 0 <= raw.min() and raw.max() <= 255 and abs(float(proc.sum())) < 1e-2 and abs(float((proc ** 2).sum()) - 1) < 1e-3)

    # the filters against their definitions, one at a time on stored pixels
    cells = s.lattice(1024)
    avg = np.mean([s.pattern(int(c), processed=False) for c in cells], axis=0)
    s.configure(static_background="average", dynamic_background="none", mask=False)
    check("the static background is the average pattern", len(cells) > 900 and np.abs(s.pattern(7) - unit(s.pattern(7, processed=False) - avg)).max() < 1e-4, f"over {len(cells)} patterns of a lattice")
    s.configure(static_background="none", dynamic_background="subtract")
    # the dynamic background is a Gaussian blur of the pattern: a narrow one tap by tap out to
    # three sigma, a wide one from sums over boxes of pixels (prep.wgsl)
    stored = s.pattern(7, processed=False).astype(np.float64)
    off = {}
    for frac in (0.15, 0.3, 0.5):
        s.configure(dynamic_sigma=frac)
        off[frac] = float(np.sqrt(((s.pattern(7) - unit(stored - gauss_reference(stored, frac * 60))) ** 2).sum()))
    check("the dynamic background is a Gaussian blur", off[0.15] < 0.02 and off[0.3] < 0.01 and off[0.5] < 0.01,
          "rms difference from the blur carried to the edges, as a share of the filtered pattern: " + ", ".join(f"{100 * v:.2f} % at sigma {k * 60:g} px" for k, v in off.items()))
    s.configure(dynamic_sigma=0.15)
    before = s.pattern(7)
    worst = 0.0
    for grid, bins, clip in ((4, 64, 40.0), (7, 32, 2.0), (8, 128, 40.0)):
        s.configure(clahe=True, clahe_grid=grid, clahe_bins=bins, clahe_clip=clip)
        worst = max(worst, float(np.abs(s.pattern(7) - unit(clahe_reference(before, grid, bins, clip))).max()))
    check("CLAHE is the reference's", worst < 2e-4, f"largest difference from the reference {worst:.1e} (pixels are about {1 / 60:.3f})")
    s.configure(mask=True, clahe_grid=4, clahe_bins=64, clahe_clip=40)
    inside = (np.arange(60)[:, None] - 29.5) ** 2 + (np.arange(60)[None, :] - 29.5) ** 2 <= 30 ** 2
    check("the mask comes last", np.abs(s.pattern(7) - unit(clahe_reference(before, 4, 64, 40.0), inside)).max() < 2e-4 and (s.pattern(7)[~inside] == 0).all())
    s.configure(static_background="average", dynamic_background="subtract", clahe=False)
    mean, std = s.pattern_stats()
    check("pattern statistics give an overview", mean.shape == (55, 75) and abs(float(mean[0, 0]) - raw.mean()) < 1e-2 and (std > 0).all())
    s.add_phase(NI)
    # the vote on the real scan, from a wrong start: the ballots land where the scan's own
    # centre is, the mean of the scores a ridge away from it
    off = {}
    for rule in ("ballots", "mean"):
        s.configure(pc_vote=rule)
        s.set_geometry(pc=(0.36, 0.16, 0.46))
        off[rule] = float(np.abs(np.array(s.fit_pc()["pc"]) - pc_file).max())
    s.configure(pc_vote="ballots")
    s.set_geometry(pc=pc_file)
    check("ballots find the nickel scan's centre where the mean vote does not", off["ballots"] < 0.006 and off["mean"] > 2 * off["ballots"], f"off by {off['ballots']:.4f} against {off['mean']:.4f}")
    q_ref = np.fromfile(DATA / "nickel" / "reference_quaternions.f32", "<f4").reshape(-1, 4)
    seen = []
    t0 = time.perf_counter()
    r = s.index(lambda st: seen.append((st["done"], st["total"])))
    dt = time.perf_counter() - t0
    dis = np.degrees(s.disorientation(r["quaternions"].reshape(-1, 4), q_ref))
    check("the nickel scan indexes as kikuchipy refined it", np.median(dis) < 0.6 and (dis < 3).mean() > 0.97 and r["ncc"].mean() > 0.45,
          f"median {np.median(dis):.3f}° apart, within 3°: {100 * (dis < 3).mean():.1f} %, mean NCC {r['ncc'].mean():.3f}, {4125 / dt:.0f} patterns/s")
    check("progress is reported per batch", len(seen) == int(np.ceil(4125 / 256)) and seen[-1] == (4125, 4125))
    check("nearly every match is accepted", (r["sweep"] >= 0).mean() > 0.98 and np.median(r["z"]) > 30, f"{100 * (r['sweep'] >= 0).mean():.1f} % at z ≥ {s.settings['accept_z']:g}, median z {np.median(r['z']):.1f}")
    # acceptance: one threshold for the scan. Chance is measured once, on a sample; z is the
    # NMI on that scale whatever the batch; and an absolute NMI can be the threshold instead
    med, spread = s.chance()
    zs = (r["nmi"] - med) / spread
    s.configure(batch=97)
    rb = s.index()
    s.configure(batch=256, accept_nmi=0.05)
    ra = s.index()
    s.configure(accept_nmi=0)
    check("acceptance is one threshold for the scan, not a batch's", 0.005 < med < 0.03 and 0 < spread < 0.005 and np.abs(zs - r["z"]).max() < 1e-3 * np.abs(r["z"]).max()
          and np.array_equal(rb["z"], r["z"]) and np.array_equal(rb["sweep"], r["sweep"]),
          f"chance NMI {med:.4f} ± {spread:.4f}: z {s.settings['accept_z']:g} is NMI {med + s.settings['accept_z'] * spread:.4f}; batches of 97 and 256 accept the same cells")
    acc = ra["sweep"] >= 0
    check("or an absolute mutual information", acc.mean() > 0.95 and (ra["nmi"][acc] >= 0.05).all() and not acc[ra["nmi"] < 0.05].any(), f"NMI ≥ 0.05 accepts {100 * acc.mean():.1f} %")
    rgba = s.map("ipf_z")
    check("maps are images", rgba.shape == (55, 75, 4) and (rgba[..., 3] == 255).all() and s.map("ncc").shape == (55, 75, 4) and s.map("z").shape == (55, 75, 4) and s.map("sweep").shape == (55, 75, 4))
    key, box = s.ipf_key(0, 128)
    k = 127 / (box[2] - box[0])
    at = lambda x, y: key[127 - int(round(y * k)), int(round(x * k))]  # noqa: E731
    check("the inverse pole figure colours come with their key", key.shape == (128, 128, 4) and np.allclose(box, (0, 0, np.sqrt(2) - 1, (np.sqrt(3) - 1) / 2), atol=1e-6)
          and tuple(at(0.004, 0.001)[:3] > 128) == (True, False, False) and tuple(at(0.405, 0.004)[:3] > 128) == (False, True, False) and tuple(at(0.36, 0.352)[:3] > 128) == (False, False, True)
          and key[0, 0, 3] == 0 and 0.3 < (key[..., 3] > 0).mean() < 0.5,
          "the cubic triangle: red at [001], green at [101], blue at [111], nothing outside it")

    # the other colouring: white at the triangle's centre, the same corners
    s.configure(ipf_colors="hue")
    hue, _ = s.ipf_key(0, 128)
    s.configure(ipf_colors="classic")
    at = lambda x, y: hue[127 - int(round(y * k)), int(round(x * k))]  # noqa: E731
    check("or white at the sector's centre", (at(0.264, 0.118)[:3] > 225).all() and tuple(at(0.004, 0.001)[:3] > 128) == (True, False, False) and tuple(at(0.405, 0.004)[:3] > 128) == (False, True, False)
          and tuple(at(0.36, 0.352)[:3] > 128) == (False, False, True) and not np.array_equal(hue, key) and np.array_equal(hue[..., 3], key[..., 3]),
          f"centre {tuple(int(v) for v in at(0.264, 0.118)[:3])}")

    # the keys of Nolze and Hielscher: a light gray centre and the same corners; and by the
    # rotations alone, the triangle beside its mirror image, one centre light and the other dark
    def at_box(img, box, x, y):
        k2 = 127 / max(box[2] - box[0], box[3] - box[1])
        return img[127 - int(round((y - box[1]) * k2)), int(round((x - box[0]) * k2))]
    s.configure(ipf_colors="perfect")
    pk, pbox = s.ipf_key(0, 128)
    s.configure(ipf_colors="perfect_rotation")
    rk, rbox = s.ipf_key(0, 128)
    lk, lbox = s.ipf_key(0, 128, lower=True)
    dark_map = s.map("ipf_z")
    s.configure(ipf_colors="classic")
    mid = at_box(pk, pbox, 0.2585, 0.1162)[:3].astype(int)
    check("the paper's key by direction: a gray centre, the same corners", np.abs(mid - 223).max() < 8 and np.allclose(pbox, box, atol=1e-6)
          and tuple(at_box(pk, pbox, 0.004, 0.001)[:3] > 128) == (True, False, False) and tuple(at_box(pk, pbox, 0.405, 0.004)[:3] > 128) == (False, True, False)
          and tuple(at_box(pk, pbox, 0.36, 0.352)[:3] > 128) == (False, False, True), f"centre {tuple(int(v) for v in mid)}")
    light = at_box(rk, rbox, 0.2585, 0.1162)[:3].astype(int)
    dark = at_box(rk, rbox, 0.1162, 0.2585)[:3].astype(int)
    check("and by orientation: the triangle and its mirror image, light and dark", np.allclose(rbox, (0, 0, np.sqrt(2) - 1, np.sqrt(2) - 1), atol=1e-6) and light.min() > 200 and dark.max() < 60
          and lbox == (0, 0, 0, 0) and lk[..., 3].max() == 0 and (dark_map[..., :3].max(-1) < 140).mean() > 0.05 and not np.array_equal(dark_map, rgba),
          f"centres {tuple(int(v) for v in light)} and {tuple(int(v) for v in dark)}; {100 * (dark_map[..., :3].max(-1) < 140).mean():.0f} % of the map's cells on the dark side")

    # an orientation as the results store one: in the fundamental zone, with its Bunge angles
    qz, eu = s.reduce(q_ref[:50])
    again, _ = s.reduce(qz)
    stored = r["quaternions"].reshape(-1, 4)[:50]
    check("orientations reduce to the fundamental zone", np.degrees(s.disorientation(qz, q_ref[:50])).max() < 1e-3 and np.abs(again - qz).max() < 1e-6 and np.isfinite(eu).all()
          and np.abs(s.reduce(stored)[0] - stored).max() < 1e-6 and np.abs(r["euler"].reshape(-1, 3)[:50] - s.reduce(stored)[1]).max() < 1e-5,
          "as the results are stored, and their Bunge angles the results' own")

    # any cells can be indexed without storing, with the settings' search or one given
    some = np.arange(0, 4125, 207)
    a = s.index_cells(some)
    b = s.index_cells(some, bandwidth=32)
    s.configure(bandwidth=32)
    c = s.index_cells(some)
    s.configure(bandwidth=64)
    near = np.degrees(s.disorientation(a["quaternions"][0], r["quaternions"].reshape(-1, 4)[some]))
    check("cells can be indexed on the spot, with a search of their own", np.median(near) < 0.05 and np.array_equal(b["quaternions"], c["quaternions"]) and not np.array_equal(a["quaternions"], b["quaternions"])
          and np.array_equal(s.results()["quaternions"], r["quaternions"]),
          f"{len(some)} cells: median {np.median(near):.4f}° from what indexing stored; at bandwidth 32 as the setting would give")

    # tuning: the fastest batch size and cutting-up of the work for this machine, by a few
    # timed trials on a sample; the stored results stay as they are and the orientations do
    # not depend on it
    kept = s.results()["quaternions"].copy()
    calls = []
    t0 = time.perf_counter()
    tuned = s.tune(lambda done: calls.append(done))
    took = time.perf_counter() - t0
    said = tuned["summary"].split()
    rt = s.index()
    apart = np.degrees(s.disorientation(rt["quaternions"].reshape(-1, 4), r["quaternions"].reshape(-1, 4)))
    check("tuning picks its settings by the clock and leaves the results alone", np.array_equal(kept, r["quaternions"]) and tuned["summary"].startswith("tuned on")
          and 5 <= int(said[5]) <= 16 and all(s.settings[k] == v for k, v in tuned["after"].items()) and np.percentile(apart, 99) < 0.05,
          f"{took:.1f} s; {tuned['summary']}; indexed again, 99 % of the orientations within {np.percentile(apart, 99):.4f}°")
    s.configure(**tuned["before"])
    # the sample is sized by what the search costs: a quarter of it at a bandwidth whose
    # orientation grid is twice as fine
    s.configure(bandwidth=96)
    t0 = time.perf_counter()
    high = s.tune()
    took96 = time.perf_counter() - t0
    check("the higher the bandwidth, the fewer patterns a trial indexes", int(said[2]) == 54 * 75 and int(high["summary"].split()[2]) == 13 * 75,
          f"{said[2]} patterns at bandwidth 64, {high['summary'].split()[2]} at 96 ({took96:.1f} s, its tables and masters built on the way)")
    s.configure(bandwidth=64, **high["before"])

    # two sweeps: bandwidth 32 for what it can index, 64 for the rest
    s.configure(sweeps=2, sweep_bandwidth=32)
    t0 = time.perf_counter()
    r2 = s.index()
    dt2 = time.perf_counter() - t0
    dis2 = np.degrees(s.disorientation(r2["quaternions"].reshape(-1, 4), q_ref))
    first = float((r2["sweep"] == 0).mean())
    check("two sweeps index the nickel scan as one pass does", np.median(dis2) < 0.6 and (dis2 < 3).mean() > 0.965 and first > 0.8 and (r2["sweep"] >= 0).mean() > 0.98,
          f"median {np.median(dis2):.3f}°, within 3°: {100 * (dis2 < 3).mean():.1f} %, {100 * first:.1f} % accepted at bandwidth 32, {4125 / dt2:.0f} patterns/s")
    s.configure(sweeps=1)

    # the master filtered as the patterns are: CLAHE and the band pass keep the scan indexable
    s.configure(clahe=True, bandpass=True)
    rf = s.index()
    disf = np.degrees(s.disorientation(rf["quaternions"].reshape(-1, 4), q_ref))
    check("CLAHE and the band pass act on the master too", (disf < 3).mean() > 0.97 and rf["ncc"].mean() > r["ncc"].mean(), f"within 3°: {100 * (disf < 3).mean():.1f} %, mean NCC {rf['ncc'].mean():.3f} (it was {r['ncc'].mean():.3f} without them)")
    s.configure(clahe=False, bandpass=False)
    r = s.index()
    out = Path(tempfile.mkdtemp(prefix="zebsdsolve-"))
    s.export(out / "nickel.ang")
    s.export(out / "nickel.ctf")
    ang = [l for l in (out / "nickel.ang").read_text().splitlines() if not l.startswith("#")]
    ctf = (out / "nickel.ctf").read_text().splitlines()
    check("the results export as .ang and .ctf", len(ang) == 4125 and len(ang[0].split()) == 10 and ctf[0] == "Channel Text File" and len(ctf) == 4125 + 15, f"{len(ang)} and {len(ctf)} lines")

    # a region of interest, then the fits from a wrong pattern centre
    s.configure(roi_x=10, roi_y=5, roi_w=20, roi_h=8)
    s.index()
    check("a region of interest indexes only itself", int((s.results()["phase"] >= 0).sum()) == 4125)  # earlier results stay
    s.configure(roi_x=0, roi_y=0, roi_w=0, roi_h=0)
    s.set_geometry(pc=(0.50, 0.30, 0.55))
    f = s.fit_pc()
    p = s.fit_pose()
    e = np.abs(np.array(p["pc"]) - pc_file)
    check("the fits recover the nickel scan's pattern centre", e.max() < 0.004,
          f"vote {np.round(f['pc'], 4)}, pose {np.round(p['pc'], 5)}, {np.round(e, 5)} from kikuchipy's, mean NCC {p['ncc']:.4f}")
    ev = s.events()
    check("events are queued for the caller", any(x["kind"] == "pc" for x in ev) and any(x["kind"] == "pose" for x in ev) and any(x["kind"] == "progress" for x in ev))
    s.close()

    # ── the plain functions ──
    res = ze.index(SCAN, [NI], roi_x=0, roi_y=0, roi_w=30, roi_h=10, batch=128)
    check("ze.index returns a Result", res.quaternions.shape == (55, 75, 4) and int((res.phase >= 0).sum()) == 300 and res.z.shape == (55, 75) and res.phases[0]["name"] == "Ni_master_20kV")
    pats3 = ze.load_patterns(SCAN, cells=[0, 1, 2])
    check("ze.load_patterns reads patterns", pats3.shape == (3, 60, 60))

    print(f"api {'OK' if not failures else 'FAILED: ' + ', '.join(failures)}  ({time.perf_counter() - t_all:.1f} s)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
