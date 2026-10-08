"""zebsdsolve: EBSD pattern-centre fitting and indexing on the GPU (Zig + WGSL over WebGPU).

The plain functions in :mod:`zebsdsolve.api` (re-exported here) cover the common workflow::

    import zebsdsolve as ze

    result = ze.index("scan.h5", ["Ni.npz"], fit="pose", bandwidth=64)
    result.quaternions, result.ncc, result.phase       # per map cell
    result.geometry["pc"], result.geometry["rot"]      # the fitted pattern centre and detector rotation

`Solver` is the engine itself, for when the steps are driven one at a time (as the web app does):
open a scan, add phases, look at patterns, fit the pattern centre, fit the pose, index.

Conventions: quaternions are (w, x, y, z) and map sample directions to crystal directions;
Bunge angles are radians. The geometry is one pose of the detector and, for each pattern, the
point of the sample the beam stood on: the pattern centre (Bruker convention: x from the left
and y from the top in fractions of the pattern's width and height, z in fractions of its
height) is that pose's position, written for the map's centre. Map cells are numbered row-major.
"""
import ctypes
import json
import os

import numpy as np

from . import _native as _n

__version__ = "0.1.0"

_lib = None
_wgpu = None

LAUE = {1: "-1", 2: "2/m", 3: "mmm", 4: "4/m", 5: "4/mmm", 6: "-3", 7: "-3m", 8: "6/m", 9: "6/mmm", 10: "m-3", 11: "m-3m"}
METHODS = ("spherical", "dictionary", "pca")
MAPS = {"ipf_x": 0, "ipf_y": 1, "ipf_z": 2, "ncc": 3, "nmi": 4, "phase": 5, "z": 6, "sweep": 7}


class Error(RuntimeError):
    """An engine call failed; the message says which and why."""


def _load():
    global _lib, _wgpu
    if _lib is None:
        _lib, _wgpu = _n.load()
        if _lib.ze_version().decode() != __version__:
            raise Error(f"library version {_lib.ze_version().decode()} does not match the package's {__version__}")
    return _lib


def _f32(a):
    return np.ascontiguousarray(a, dtype=np.float32)


def _ptr(a, t):
    return a.ctypes.data_as(t)


class Solver:
    """One engine on one GPU device. Close it (or use it as a context manager) to free the device."""

    def __init__(self):
        lib = _load()
        self._lib = lib
        self._e = lib.ze_create(_wgpu.encode())
        if not self._e:
            buf = ctypes.create_string_buffer(512)
            n = lib.ze_create_error(buf, 512)
            raise Error(buf.raw[:n].decode() or "engine creation failed")

    def close(self):
        if getattr(self, "_e", None):
            self._lib.ze_destroy(self._e)
            self._e = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ── plumbing ──

    def _error(self):
        buf = ctypes.create_string_buffer(1024)
        n = self._lib.ze_error(self._e, buf, 1024)
        return buf.raw[:n].decode(errors="replace")

    def _check(self, rc, what):
        err = self._error()
        if rc < 0 or err:
            raise Error(f"{what}: {err or rc}")
        return rc

    @property
    def adapter(self):
        """The GPU adapter's description."""
        buf = ctypes.create_string_buffer(256)
        n = self._lib.ze_adapter(self._e, buf, 256)
        return buf.raw[:n].decode(errors="replace")

    def events(self):
        """Progress events queued since the last call, as a list of dicts."""
        out = []
        buf = ctypes.create_string_buffer(1 << 16)
        while True:
            n = self._lib.ze_events(self._e, buf, len(buf))
            if n == 0:
                break
            out += [json.loads(line) for line in buf.raw[:n].decode(errors="replace").splitlines() if line]
        return out

    def cancel(self):
        """Ask a running fit or indexing to stop at its next step."""
        self._lib.ze_cancel(self._e)

    # ── settings and geometry ──

    def configure(self, **settings):
        """Set options by name (src/settings.zig lists them all, with what they mean)."""
        if settings:
            j = json.dumps(settings).encode()
            self._check(self._lib.ze_configure(self._e, j, len(j)), "configure")
        return self

    @property
    def settings(self):
        """Every option's current value."""
        cap = 8192
        buf = ctypes.create_string_buffer(cap)
        n = self._lib.ze_settings(self._e, buf, cap)
        if n > cap:
            buf = ctypes.create_string_buffer(n)
            n = self._lib.ze_settings(self._e, buf, n)
        return json.loads(buf.raw[:n].decode())

    @property
    def geometry(self):
        """The detector geometry: a dict with pc (3), sample_tilt_deg, detector_tilt_deg, rot (3,
        rotation vector, radians, in the detector's own axes: swing about its row direction,
        tilt about its column direction, twist about its normal), px_um, step_x_um, step_y_um."""
        g = _n.Geometry()
        self._lib.ze_get_geometry(self._e, ctypes.byref(g))
        return {
            "pc": list(g.pc), "sample_tilt_deg": g.sample_tilt_deg, "detector_tilt_deg": g.detector_tilt_deg,
            "rot": list(g.rot), "px_um": g.px_um, "step_x_um": g.step_x_um, "step_y_um": g.step_y_um,
        }

    def set_geometry(self, **fields):
        """Set any of the geometry's fields, e.g. ``set_geometry(pc=(0.5, 0.2, 0.5), px_um=70)``."""
        g = _n.Geometry()
        self._lib.ze_get_geometry(self._e, ctypes.byref(g))
        for k, v in fields.items():
            if k in ("pc", "rot"):
                setattr(g, k, (ctypes.c_double * 3)(*[float(x) for x in v]))
            elif hasattr(g, k):
                setattr(g, k, float(v))
            else:
                raise Error(f"geometry has no field {k!r}")
        self._lib.ze_set_geometry(self._e, ctypes.byref(g))
        return self

    # ── scan ──

    def open(self, *paths):
        """Open a scan file, and any side files it needs after it. NORDIF's Pattern.dat needs its
        Setting.txt, which is picked up from beside it when not given."""
        paths = [str(p) for p in paths]
        if len(paths) == 1 and paths[0].lower().endswith(".dat"):
            side = os.path.join(os.path.dirname(paths[0]), "Setting.txt")
            if os.path.isfile(side):
                paths.append(side)
        joined = "\n".join(paths).encode()
        self._check(self._lib.ze_open_scan(self._e, joined, len(joined)), "open")
        return self

    def synthesize(self, **settings):
        """Make a toy scan from the phases loaded, in place of any scan open, and return its
        answer: a dict with quaternions (rows, cols, 4), phase (rows, cols; −1: a blank cell),
        grain (rows, cols) and geometry (the geometry it was simulated under, which is its true
        one: set it before this call, and move it afterwards to give a fit something to find).

        Keywords are settings. The microstructure, from `toy_seed`: `toy_rows` × `toy_cols`
        cells cut into `toy_grains` grains (1: a single crystal), each with a random orientation
        and phase; `toy_spread_deg` turns every cell a little within its grain; `toy_blank` is
        the share of cells holding no crystal. The detector: patterns of `toy_height` ×
        `toy_width` pixels, `toy_contrast` of whose intensity is the simulation and the rest a
        background (flat, or a hump of `toy_background_sigma` × the width). The counting: a
        Poisson draw per pixel at a mean of `toy_counts` electrons (0: no noise), from
        `toy_noise_seed` (0: from `toy_seed`), so one microstructure can be counted many times."""
        if settings:
            self.configure(**settings)
        st = self.settings
        rows, cols = st["toy_rows"], st["toy_cols"]
        n = rows * cols
        q = np.empty(4 * n, np.float32)
        ph = np.empty(n, np.int32)
        gr = np.empty(n, np.int32)
        geometry = self.geometry
        i32p = ctypes.POINTER(ctypes.c_int32)
        self._check(self._lib.ze_synthesize(self._e, _ptr(q, _n.f32p), ph.ctypes.data_as(i32p), gr.ctypes.data_as(i32p), n), "synthesize")
        return {"quaternions": q.reshape(rows, cols, 4), "phase": ph.reshape(rows, cols), "grain": gr.reshape(rows, cols), "geometry": geometry}

    def set_patterns(self, patterns):
        """Take patterns from an array: (rows, cols, h, w) or (n, h, w), uint8, uint16 or float32."""
        a = np.asarray(patterns)
        if a.ndim == 3:
            a = a[None]
        if a.ndim != 4:
            raise Error("patterns must be (rows, cols, h, w) or (n, h, w)")
        if a.dtype == np.uint8:
            pixel = 0
        elif a.dtype == np.uint16:
            pixel = 1
        else:
            a = a.astype(np.float32)
            pixel = 2
        a = np.ascontiguousarray(a)
        rows, cols, h, w = a.shape
        self._check(self._lib.ze_set_patterns(self._e, a.ctypes.data, a.nbytes, pixel, rows, cols, h, w), "set_patterns")
        return self

    @property
    def scan(self):
        """What the scan is: map and pattern shape, pixel type, format, what metadata it carries."""
        s = _n.ScanInfo()
        self._lib.ze_scan_info(self._e, ctypes.byref(s))
        buf = ctypes.create_string_buffer(64)
        n = self._lib.ze_scan_format(self._e, buf, 64)
        fmt = buf.raw[:n].decode()
        return {
            "rows": s.rows, "cols": s.cols, "pattern_shape": (s.pat_h, s.pat_w), "dtype": ["uint8", "uint16", "float32"][s.pixel],
            "format": fmt, "has_background": bool(s.has_background), "has_pc": bool(s.has_pc),
            "voltage_kv": s.voltage_kv or None, "bytes": int(s.bytes),
        }

    def shape(self, full=False):
        """(height, width) of a pattern: reduced by the `binning` setting, or (`full`) as stored."""
        out = (ctypes.c_uint32 * 2)()
        self._check(self._lib.ze_pattern_shape(self._e, int(full), out), "pattern_shape")
        return int(out[0]), int(out[1])

    @property
    def pattern_shape(self):
        """(height, width) of a pattern as the search sees it: reduced by the `binning` setting."""
        return self.shape()

    def pattern(self, cell, processed=True, full=False):
        """One pattern as float32 (h, w), at the reduced size (the `binning` setting) or (`full`)
        the stored one. `processed`: True, as it is indexed (averaged with its neighbours when
        `nlpar` is on, then filtered); False, as it is stored; "averaged", averaged with its
        neighbours and not filtered."""
        h, w = self.shape(full)
        out = np.empty(h * w, np.float32)
        stage = 2 if processed == "averaged" else int(bool(processed))
        self._check(self._lib.ze_pattern(self._e, int(cell), stage, int(full), _ptr(out, _n.f32p), out.size), "pattern")
        return out.reshape(h, w)

    def nlpar_view(self):
        """What averaging with its neighbours (`nlpar`) did to the pattern `pattern` last
        returned: a dict with its `noise` as stored (standard deviation a pixel), the share of
        that noise the average keeps (`left`), the share of the average that is the pattern
        itself (`own`), and `shares`, the average's shares of the stored patterns around it
        (side × side, the pattern itself in the middle, NaN outside the map). None when the
        pattern was not averaged."""
        out = np.empty(3, np.float32)
        side = self._lib.ze_nlpar_view(self._e, _ptr(out, _n.f32p), None, 0)
        if side == 0:
            return None
        shares = np.empty(side * side, np.float32)
        self._lib.ze_nlpar_view(self._e, _ptr(out, _n.f32p), _ptr(shares, _n.f32p), shares.size)
        return {"noise": float(out[0]), "left": float(out[1]), "own": float(out[2]), "shares": shares.reshape(side, side)}

    def lattice(self, n):
        """About `n` map cells on a lattice over the whole map (every cell when it has no more):
        the cells the static background is averaged over, and a quick preview's."""
        count = self._check(self._lib.ze_lattice(self._e, int(n), None, 0), "lattice")
        out = np.empty(count, np.uint64)
        self._check(self._lib.ze_lattice(self._e, int(n), _ptr(out, _n.u64p), count), "lattice")
        return out

    @property
    def pixel_angle(self):
        """Angular width of one stored detector pixel as the master filters take it, degrees:
        the `pixel_angle_deg` setting, or the pixel at the pattern centre when that is 0."""
        return float(self._lib.ze_pixel_angle(self._e))

    def pattern_stats(self, cells=None):
        """(mean, standard deviation) of the stored patterns of the given map cells (all of them
        by default, shaped (rows, cols)): maps of either show the microstructure before indexing."""
        s = self.scan
        whole = cells is None
        c = np.arange(s["rows"] * s["cols"], dtype=np.uint64) if whole else np.ascontiguousarray(cells, dtype=np.uint64).reshape(-1)
        out = np.empty(2 * len(c), np.float32)
        step = 4096
        for i in range(0, len(c), step):
            part = np.ascontiguousarray(c[i:i + step])
            o = out[2 * i:2 * (i + len(part))]
            self._check(self._lib.ze_pattern_stats(self._e, _ptr(part, _n.u64p), len(part), _ptr(o, _n.f32p), o.size), "pattern_stats")
        mean, std = out[0::2], out[1::2]
        return (mean.reshape(s["rows"], s["cols"]), std.reshape(s["rows"], s["cols"])) if whole else (mean, std)

    # ── phases ──

    def add_phase(self, path):
        """Add a candidate phase from a master pattern (ebsdsim .npz or EMsoft .h5); returns its index."""
        p = str(path).encode()
        return self._check(self._lib.ze_add_phase(self._e, p, len(p)), "add_phase")

    def clear_phases(self):
        self._lib.ze_clear_phases(self._e)

    @property
    def phases(self):
        """The candidate phases, each a dict."""
        out = []
        for i in range(self._lib.ze_phase_count(self._e)):
            p = _n.PhaseInfo()
            self._check(self._lib.ze_phase_info(self._e, i, ctypes.byref(p)), "phase_info")
            out.append({
                "name": p.name.decode(errors="replace"), "side": p.side, "point_group": p.point_group, "space_group": p.space_group,
                "laue": LAUE.get(p.laue, "?"), "voltage_kv": p.voltage_kv,
                "n_fold": p.n_fold, "mirror": bool(p.mirror), "inversion": bool(p.inversion),
            })
        return out

    def master(self, phase=0):
        """The phase's master pattern as the search compares patterns with it (smoothed, and
        filtered as the patterns are): (2, side, side), north then south."""
        side = self.phases[phase]["side"]
        out = np.empty(2 * side * side, np.float32)
        self._check(self._lib.ze_master_image(self._e, phase, _ptr(out, _n.f32p), out.size), "master")
        return out.reshape(2, side, side)

    def simulate(self, quaternions, cells=None, phase=0, full=False):
        """Simulated patterns (n, h, w) of a phase at the given orientations, each as the map cell
        `cells[k]` would record it (the map's reference cell, its centre, by default), at the
        reduced size or (`full`) the stored one."""
        q = _f32(quaternions).reshape(-1, 4)
        n = len(q)
        if cells is None:
            s = self.scan
            cells = np.full(n, (s["rows"] // 2) * s["cols"] + s["cols"] // 2 if s["rows"] else 0)
        c = np.ascontiguousarray(cells, dtype=np.uint64).reshape(-1)
        h, w = self.shape(full)
        out = np.empty(n * h * w, np.float32)
        self._check(self._lib.ze_simulate(self._e, phase, _ptr(c, _n.u64p), _ptr(q, _n.f32p), n, int(full), _ptr(out, _n.f32p), out.size), "simulate")
        return out.reshape(n, h, w)

    # ── fits ──

    def fit_pc(self):
        """Fit the pattern centre by the voting search (settings fit_grid, pc_*). The geometry takes
        the result; returns a dict with pc, start, score, start_score, n_patterns, n_candidates, at_border."""
        r = _n.PcResult()
        self._check(self._lib.ze_fit_pc(self._e, ctypes.byref(r)), "fit_pc")
        return {
            "pc": list(r.pc), "start": list(r.start), "score": r.score, "start_score": r.start_score,
            "n_patterns": r.n_patterns, "n_candidates": r.n_candidates, "levels": r.levels, "at_border": bool(r.at_border),
        }

    def pc_scores(self):
        """The last search level's votes: (scores[nz, ny, nx], pcx values, pcy values, pcz values)."""
        shape = (ctypes.c_uint32 * 3)()
        n = self._lib.ze_pc_scores(self._e, shape, None, 0, None, 0)
        nz, ny, nx = (int(v) for v in shape)
        axes = np.empty(nx + ny + nz, np.float64)
        scores = np.empty(max(n, 0), np.float32)
        self._lib.ze_pc_scores(self._e, shape, _ptr(axes, _n.f64p), axes.size, _ptr(scores, _n.f32p), scores.size)
        return scores.reshape(nz, ny, nx), axes[:nx], axes[nx:nx + ny], axes[nx + ny:]

    def pc_levels(self):
        """Every grid the last pattern-centre search scored, in order: a list of dicts with
        scores[nz, ny, nx] (the fit patterns' mean score per candidate), the candidates' x, y, z,
        the winner (ix, iy, iz: the candidate nearest the grid's choice), walked (the grid then
        moved onto a winner at its border), per fit pattern the candidate it scored highest
        under (pattern_best, rows of (ix, iy, iz)), its score there (pattern_ncc) and its score
        under every candidate (pattern_scores[n, nz, ny, nx]); and, when ballots decided
        (`pc_vote`), their density[nz, ny, nx], the choice in candidate steps (peak: x, y, z),
        and how many ballots were cast and how many lay near the choice (ballots, inliers)."""
        out = []
        for i in range(self._lib.ze_pc_level_count(self._e)):
            shape = (ctypes.c_uint32 * 3)()
            info = (ctypes.c_uint32 * 2)()
            n = self._lib.ze_pc_level(self._e, i, shape, info, None, 0, None, 0, None, None, 0)
            nz, ny, nx = (int(v) for v in shape)
            axes = np.empty(nx + ny + nz, np.float64)
            scores = np.empty(n, np.float32)
            best = np.empty(1024, np.uint32)
            ncc = np.empty(1024, np.float32)
            self._lib.ze_pc_level(self._e, i, shape, info, _ptr(axes, _n.f64p), axes.size, _ptr(scores, _n.f32p), n, _ptr(best, _n.u32p), _ptr(ncc, _n.f32p), best.size)
            n_pat = self._lib.ze_fit_patterns(self._e, None, None, None, None, 0) or len(best)

            def split(c):
                return np.stack([c % nx, (c // nx) % ny, c // (nx * ny)], -1)

            per = np.empty(self._lib.ze_pc_level_scores(self._e, i, None, 0), np.float32)
            self._lib.ze_pc_level_scores(self._e, i, _ptr(per, _n.f32p), per.size)
            out.append({"scores": scores.reshape(nz, ny, nx), "x": axes[:nx], "y": axes[nx:nx + ny], "z": axes[nx + ny:],
                        "pattern_scores": per.reshape(-1, nz, ny, nx),
                        **self._pc_ballots(i, (nz, ny, nx)),
                        "winner": tuple(int(v) for v in split(np.array(int(info[0])))), "walked": bool(info[1]),
                        "pattern_best": split(best[:n_pat].astype(np.int64)), "pattern_ncc": ncc[:n_pat].copy()})
        return out

    def _pc_ballots(self, i, shape):
        info = np.empty(5, np.float64)
        n = self._lib.ze_pc_level_ballots(self._e, i, None, 0, _ptr(info, _n.f64p))
        den = np.empty(n, np.float32)
        self._lib.ze_pc_level_ballots(self._e, i, _ptr(den, _n.f32p), n, _ptr(info, _n.f64p))
        return {"density": den.reshape(shape) if n else None, "peak": tuple(float(v) for v in info[:3]), "ballots": int(info[3]), "inliers": int(info[4])}

    def fit_pose(self):
        """Fit the detector's pose and the fit patterns' orientations together (settings pose_grid,
        pose_iters): the whole pose, six degrees of freedom. Any of them can be held with the
        settings pose_x, pose_y, pose_z (the pattern centre's coordinates) and pose_tilt,
        pose_swing, pose_twist (the detector's turns about the source). `pose_scale` adds a
        seventh, the scale between the beam's travel and the detector (Sim(3)): the detector's
        pixel size. The geometry takes the result; returns a dict, its `rot` the detector's
        swing, tilt and twist (radians) and `px_um` the pixel size."""
        r = _n.PoseResult()
        self._check(self._lib.ze_fit_pose(self._e, ctypes.byref(r)), "fit_pose")
        return {
            "pc": list(r.pc), "rot": list(r.rot), "start_pc": list(r.start_pc), "start_rot": list(r.start_rot),
            "ncc": r.ncc, "start_ncc": r.start_ncc, "mean_turn": r.mean_turn, "px_um": r.px_um, "start_px_um": r.start_px_um,
            "iterations": r.iterations, "accepted": r.accepted, "n_patterns": r.n_patterns, "converged": bool(r.converged),
        }

    def fit_patterns(self):
        """The patterns of the last fit: (cells, phases, quaternions (n, 4), ncc)."""
        n = self._lib.ze_fit_patterns(self._e, None, None, None, None, 0)
        cells = np.empty(n, np.uint64)
        phases = np.empty(n, np.int32)
        quats = np.empty(4 * n, np.float32)
        ncc = np.empty(n, np.float32)
        self._lib.ze_fit_patterns(self._e, _ptr(cells, _n.u64p), _ptr(phases, _n.i32p), _ptr(quats, _n.f32p), _ptr(ncc, _n.f32p), n)
        return cells, phases, quats.reshape(n, 4), ncc

    # ── indexing ──

    def index_cells(self, cells, method=None, bandwidth=None, dict_grid=None):
        """Index the given map cells without storing anything: a dict of arrays shaped
        (phases, n[, 4]): quaternions, ncc, nmi, step (the rotation of the last refinement step
        that improved the match, radians: small once the refinement has settled) and z (how far
        the NMI stands above chance, in chance's standard deviations). The search is the
        settings' (`method`, `bandwidth`, `dict_grid`) unless given here."""
        c = np.ascontiguousarray(cells, dtype=np.uint64).reshape(-1)
        n_ph = self._lib.ze_phase_count(self._e)
        out = np.empty(n_ph * len(c) * 8, np.float32)
        if method is None and bandwidth is None and dict_grid is None:
            self._check(self._lib.ze_index_cells(self._e, _ptr(c, _n.u64p), len(c), _ptr(out, _n.f32p), out.size), "index_cells")
        else:
            st = self.settings
            m = METHODS.index(method if method is not None else st["method"])
            self._check(self._lib.ze_index_cells_by(self._e, _ptr(c, _n.u64p), len(c), m, int(bandwidth or st["bandwidth"]), int(dict_grid or st["dict_grid"]),
                                                    _ptr(out, _n.f32p), out.size), "index_cells")
        r = out.reshape(n_ph, len(c), 8)
        return {"quaternions": r[..., :4].copy(), "ncc": r[..., 4].copy(), "nmi": r[..., 5].copy(), "step": r[..., 6].copy(), "z": r[..., 7].copy()}

    def tune(self, progress=None):
        """Fit how the work is cut up to this machine and this scan. A sample of the region is
        indexed and timed, and `batch`, `refine_pixels`, `wg_refine` and `wg_fft` each in turn
        climb from the value they have: doubled while that is faster, else halved while that
        is. The sample is sized by the bandwidth (8192 patterns up to 32, 4096 up to 64, 1024 up
        to 128, 128 beyond), so a trial stays a fraction of a second: about ten trials, seconds
        in all. Nothing stored is touched, and neither the orientations found nor which matches
        are accepted depend on these settings. Returns the settings before and after and the
        engine's summary. `progress(calls_done)` is called after each timed call; return False
        from it to stop (the best so far stays)."""
        import time

        keys = ("batch", "refine_pixels", "wg_refine", "wg_fft")
        before = {k: self.settings[k] for k in keys}
        self._check(self._lib.ze_tune_begin(self._e), "tune")
        done = 0
        while True:
            t = time.perf_counter()
            rc = self._lib.ze_tune_trial(self._e)
            seconds = time.perf_counter() - t
            if rc != 0:
                # a value this machine cannot run (out of memory, say) just loses
                message = self._error()
                if "Cancelled" in message:
                    raise Error("tune: " + message)
                seconds = -1.0
            done += 1
            more = self._lib.ze_tune_report(self._e, seconds)
            if not more or (progress is not None and progress(done) is False):
                break
        said = [e["text"] for e in self.events() if e["kind"] == "log" and e["text"].startswith("tuned on")]
        return {"before": before, "after": {k: self.settings[k] for k in keys}, "summary": said[-1] if said else ""}

    def chance(self, phase=0):
        """What normalized mutual information chance gives on this scan for a phase: (median,
        spread), measured once on a sample of the region's patterns against simulations at
        random orientations, the same for every pattern. z in the results is the NMI on that
        scale; with `accept_nmi` 0 a match is accepted at median + accept_z × spread, and
        `accept_nmi` above 0 is an absolute threshold in its place."""
        out = np.empty(2, np.float32)
        self._check(self._lib.ze_chance(self._e, int(phase), _ptr(out, _n.f32p)), "chance")
        return float(out[0]), float(out[1])

    def index_begin(self):
        """Start indexing the region of interest; `index_step` then does a batch at a time."""
        self._check(self._lib.ze_index_begin(self._e), "index")

    def index_step(self):
        """Index the next batch; returns the number of cells left to visit (0: done)."""
        return self._check(self._lib.ze_index_step(self._e), "index")

    @property
    def index_status(self):
        """Where indexing stands: the sweep in progress (of `sweeps`) and what it searches with,
        the patterns done and in it, the cells of the region and how many are accepted."""
        st = _n.IndexStatus()
        self._lib.ze_index_status(self._e, ctypes.byref(st))
        out = {k: int(getattr(st, k)) for k in ("sweep", "sweeps", "bandwidth", "dict_grid", "done", "total", "region", "accepted")}
        out["method"] = METHODS[st.method]
        return out

    @property
    def sweep_plan(self):
        """What each sweep searches with, in order: a list of (bandwidth, dictionary grid, method)."""
        out = (ctypes.c_uint32 * 3)()
        plan = []
        for i in range(self.settings["sweeps"]):
            self._lib.ze_sweep(self._e, i, out)
            plan.append((int(out[0]), int(out[1]), METHODS[out[2]]))
        return plan

    def index(self, progress=None):
        """Index the region of interest (settings roi_*, the whole map by default), a batch at a
        time, in settings.sweeps sweeps. `progress(status)` is called after each batch with
        `index_status`; return False from it to stop."""
        self.index_begin()
        while True:
            left = self.index_step()
            if progress is not None and progress(self.index_status) is False:
                break
            if left == 0:
                break
        return self.results()

    def results(self):
        """The stored results per map cell: a dict of arrays shaped (rows, cols[, k]):
        quaternions (4), euler (3, Bunge, radians), ncc, nmi, phase (−1: not indexed), z (how far
        the NMI stands above chance, standard deviations) and sweep (the sweep that found an
        accepted match, one at or above the threshold (`accept_nmi`, or `accept_z`) whose refinement had settled; −1: none did, and
        the cell holds the best there was)."""
        s = self.scan
        rows, cols = s["rows"], s["cols"]
        n = rows * cols

        def get(kind, dtype, k):
            a = np.empty(n * k, dtype)
            self._check(self._lib.ze_results(self._e, kind, a.ctypes.data, a.nbytes), "results")
            return a.reshape((rows, cols, k) if k > 1 else (rows, cols))

        return {
            "quaternions": get(0, np.float32, 4), "euler": get(4, np.float32, 3),
            "ncc": get(1, np.float32, 1), "nmi": get(2, np.float32, 1), "phase": get(3, np.int32, 1),
            "z": get(5, np.float32, 1), "sweep": get(6, np.int32, 1),
        }

    def map(self, kind="ipf_z"):
        """A result map as RGBA uint8 (rows, cols, 4): ipf_x, ipf_y, ipf_z, ncc, nmi, phase, z or sweep."""
        s = self.scan
        out = np.empty(s["rows"] * s["cols"] * 4, np.uint8)
        self._check(self._lib.ze_map(self._e, MAPS[kind], _ptr(out, _n.u8p), out.size), "map")
        return out.reshape(s["rows"], s["cols"], 4)

    def ipf_key(self, phase=0, size=256, lower=False):
        """The key to a phase's inverse pole figure colours, as the settings have them
        (`ipf_colors`: "classic", "hue", "perfect" or "perfect_rotation"; `ipf_lightness` and
        `ipf_gray` for the last two): (rgba, box). `rgba` is uint8 (size, size, 4): the crystal
        directions that stand for all the others, in stereographic projection with the crystal's
        x to the right ([001] at the origin), transparent outside them. That is the triangle
        between [001], [101] and [111] for the cubic classes, a wedge from the c axis for any
        other, and twice either under a key that gives a class a light and a dark half. `box` is
        that region's bounding box in the projection (x0, y0, x1, y1), drawn from the image's
        lower left corner. `lower` asks for the region's part in the lower hemisphere instead,
        which only "perfect_rotation" has, and only for the classes of a single rotation axis
        (2/m, 4/m, 6/m, -3, -1): for the others the image is empty and the box all zero."""
        out = np.empty(size * size * 4, np.uint8)
        box = np.empty(6, np.float32)
        self._check(self._lib.ze_ipf_key(self._e, int(phase), int(size), int(bool(lower)), _ptr(out, _n.u8p), out.size, _ptr(box, _n.f32p)), "ipf_key")
        return out.reshape(size, size, 4), tuple(float(v) for v in box[:4])

    def export(self, path):
        """Write the results as an EDAX .ang or an HKL .ctf file, by the path's extension."""
        kind = {"ang": 0, "ctf": 1}.get(str(path).rsplit(".", 1)[-1].lower())
        if kind is None:
            raise Error("export writes .ang or .ctf")
        n = self._check(self._lib.ze_export(self._e, kind, None, 0), "export")
        buf = ctypes.create_string_buffer(n)
        self._check(self._lib.ze_export(self._e, kind, buf, n), "export")
        with open(path, "wb") as f:
            f.write(buf.raw[:n])

    def reduce(self, quaternions, phase=0):
        """Orientations as the results store them: (quaternions, euler), each quaternion's
        representative in the phase's fundamental zone and its Bunge angles (radians). The zone
        is that of the proper rotations of the phase's point group (for GaN, 6mm, the six about
        c: not its Laue class's twelve, half of which turn the polar axis over)."""
        q = _f32(quaternions).reshape(-1, 4).copy()
        euler = np.empty((len(q), 3), np.float64)
        for i in range(len(q)):
            self._lib.ze_reduce(self._e, phase, _ptr(q[i:i + 1], _n.f32p), _ptr(euler[i:i + 1], _n.f64p))
        return q, euler

    def disorientation(self, a, b, phase=0):
        """Smallest rotation between orientations a and b (arrays of quaternions) under the
        phase's symmetry, radians."""
        qa = _f32(a).reshape(-1, 4)
        qb = _f32(b).reshape(-1, 4)
        out = np.empty(len(qa), np.float64)
        for i in range(len(qa)):
            out[i] = self._lib.ze_disorientation(self._e, phase, _ptr(qa[i:i + 1], _n.f32p), _ptr(qb[i:i + 1], _n.f32p))
        return out

    # ── tests and profiling ──

    def _compile_wgsl(self, code, entry):
        """Build one kernel; None when it builds, else the compiler's message."""
        msg = ctypes.create_string_buffer(4096)
        c = code.encode()
        en = entry.encode()
        n = self._lib.ze_compile_wgsl(self._e, c, len(c), en, len(en), msg, 4096)
        return None if n == 0 else msg.raw[:n].decode(errors="replace")

    def gpu_counts(self):
        """GPU traffic since the last call: dispatches, submits, reads, waits."""
        out = (ctypes.c_uint32 * 4)()
        self._lib.ze_gpu_counts(self._e, out)
        return dict(zip(("dispatches", "submits", "reads", "waits"), (int(v) for v in out)))

    def profile(self, on=True):
        """Time every dispatch on the GPU from now on; `profile_report()` prints the totals."""
        return bool(self._lib.ze_profile(self._e, int(on)))

    def profile_report(self):
        buf = ctypes.create_string_buffer(1 << 16)
        n = self._lib.ze_profile_report(self._e, buf, len(buf))
        return buf.raw[:n].decode()


def shader_modules():
    """The engine's WGSL modules as {name: source}, the text it compiles."""
    lib = _load()
    out = {}
    i = 0
    while True:
        name = ctypes.create_string_buffer(64)
        n = lib.ze_shader(i, name, 64, None, 0)
        if n == 0:
            return out
        buf = ctypes.create_string_buffer(n)
        lib.ze_shader(i, name, 64, buf, n)
        out[name.value.decode()] = buf.raw[:n].decode()
        i += 1


from .api import index, fit_pattern_center, fit_pose, load_patterns, Result  # noqa: E402,F401
