"""The plain-function API: one call per job, each opening its own engine.

    import zebsdsolve as ze

    res = ze.index("scan.h5", ["Ni.npz"])                 # index with the file's geometry
    res = ze.index("scan.h5", ["Ni.npz"], fit="pose")     # fit the pattern centre and pose first
    fit = ze.fit_pattern_center("scan.h5", ["Ni.npz"], pc=(0.5, 0.3, 0.5), pc_span=0.2)

`scan` is a path (or a sequence of paths, the pattern file first), or an array of patterns
shaped (rows, cols, h, w). `phases` are master pattern files. Keyword arguments that name a
geometry field (pc, rot, sample_tilt_deg, detector_tilt_deg, px_um, step_x_um, step_y_um) set
the geometry; every other keyword is a setting (src/settings.zig).
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

GEOMETRY_FIELDS = ("pc", "rot", "sample_tilt_deg", "detector_tilt_deg", "px_um", "step_x_um", "step_y_um")


@dataclass
class Result:
    """Indexing results per map cell, with the geometry they were indexed under."""

    quaternions: np.ndarray
    """(rows, cols, 4): w, x, y, z; sample → crystal, in the phase's fundamental zone"""
    euler: np.ndarray
    """(rows, cols, 3): the quaternions' Bunge angles, radians"""
    ncc: np.ndarray
    """(rows, cols): normalized cross-correlation of each pattern with its simulation"""
    nmi: np.ndarray
    """(rows, cols): normalized mutual information of the same pair"""
    phase: np.ndarray
    """(rows, cols): index into `phases`; −1 where a cell was not indexed"""
    z: np.ndarray
    """(rows, cols): the match's NMI on the scan's own scale: how far above what chance gives on
    this scan (measured once, on a sample of it), in chance's standard deviations"""
    sweep: np.ndarray
    """(rows, cols): the sweep that found an accepted match (at or above the threshold,
    `accept_nmi` or `accept_z`, its refinement settled); −1 where none did and the cell holds
    the best match there was"""
    geometry: dict
    phases: list
    settings: dict
    fit: dict = field(default_factory=dict)
    """what the fits reported, by name ("pc", "pose"), when they ran"""


def _open(solver, scan, phases, kwargs):
    geo = {k: kwargs.pop(k) for k in list(kwargs) if k in GEOMETRY_FIELDS}
    # options that decide how a file is read must be set before it is opened
    solver.configure(**kwargs)
    if isinstance(scan, (str, Path)):
        solver.open(scan)
    elif isinstance(scan, (list, tuple)) and scan and isinstance(scan[0], (str, Path)):
        solver.open(*scan)
    else:
        solver.set_patterns(scan)
    if geo:
        solver.set_geometry(**geo)
    for p in [phases] if isinstance(phases, (str, Path)) else phases:
        solver.add_phase(p)


def load_patterns(scan, cells=None, processed=False, **settings):
    """Patterns of a scan file as float32 (n, h, w): the given map cells (row-major indices; all
    of them by default), as stored or after the filters, reduced by the `binning` setting."""
    from . import Solver

    with Solver() as s:
        s.configure(**settings)
        s.open(*([scan] if isinstance(scan, (str, Path)) else scan))
        info = s.scan
        cells = range(info["rows"] * info["cols"]) if cells is None else cells
        return np.stack([s.pattern(c, processed) for c in cells])


def fit_pattern_center(scan, phases, **kwargs):
    """Fit the pattern centre by the voting search. Returns the fit's report (a dict: pc, score,
    start, …) with the votes of the last search level under "scores" and its axes under "axes"."""
    from . import Solver

    with Solver() as s:
        _open(s, scan, phases, kwargs)
        r = s.fit_pc()
        scores, x, y, z = s.pc_scores()
        r.update(scores=scores, axes=(x, y, z), geometry=s.geometry)
        return r


def fit_pose(scan, phases, **kwargs):
    """Fit the detector's whole pose (and, with `pose_scale`, its pixel size) together with the
    orientations of a lattice of patterns, from the current geometry. Returns the fit's report
    (a dict) with the final "geometry" and the lattice's cells, phases, orientations and scores."""
    from . import Solver

    with Solver() as s:
        _open(s, scan, phases, kwargs)
        r = s.fit_pose()
        cells, ph, quats, ncc = s.fit_patterns()
        r.update(geometry=s.geometry, cells=cells, phases=ph, quaternions=quats, pattern_ncc=ncc)
        return r


def index(scan, phases, fit=None, progress=None, **kwargs):
    """Index a scan against candidate phases.

    fit: None indexes with the geometry as given (the file's, or the keywords'); "pc" first fits
    the pattern centre by voting; "pose" fits the pattern centre, then the detector pose.
    progress(status) is called after each batch with where indexing stands (Solver.index_status:
    sweep, sweeps, done, total, region, accepted); returning False stops early.
    """
    from . import Solver

    if fit not in (None, "pc", "pose"):
        raise ValueError('fit is None, "pc" or "pose"')
    with Solver() as s:
        _open(s, scan, phases, kwargs)
        report = {}
        if fit in ("pc", "pose"):
            report["pc"] = s.fit_pc()
        if fit == "pose":
            report["pose"] = s.fit_pose()
        r = s.index(progress)
        return Result(geometry=s.geometry, phases=s.phases, settings=s.settings, fit=report, **r)
