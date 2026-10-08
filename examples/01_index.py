# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # From a scan file to an orientation map
#
# The nickel scan that ships with kikuchipy (55 × 75 patterns of 60 × 60 pixels), from the file
# to an indexed map: looking at patterns and what the filters do to them, fitting the pattern
# centre from a wrong start by a vote, refining the detector pose, indexing, and comparing the
# result with the orientations and the pattern centre kikuchipy refined for the same scan.

# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import zebsdsolve as ze

# the scan and its master pattern: the repository's testdata/ (or $ZEBSDSOLVE_TESTDATA)
DATA = Path(os.environ.get("ZEBSDSOLVE_TESTDATA", ""))
if not (DATA / "nickel").is_dir():
    DATA = next(p / "testdata" for p in [Path.cwd(), *Path.cwd().parents] if (p / "testdata" / "nickel").is_dir())
SCAN = DATA / "nickel" / "patterns_v2.h5"
MASTER = DATA / "nickel" / "Ni_master_20kV.npz"

s = ze.Solver()
print(s.adapter)

# %% [markdown]
# ## The scan
#
# `open` reads only the file's metadata. Patterns are read when something needs them, in
# ranges, so the size of the file does not matter. The geometry takes what the file records.

# %%
s.open(SCAN)
s.add_phase(MASTER)
print(s.scan)
print(s.phases[0])
pc_file = np.array(s.geometry["pc"])
print("pattern centre in the file (Bruker convention):", pc_file)

# %% [markdown]
# Before anything is indexed, the contrast of each stored pattern already shows the
# microstructure.

# %%
mean, std = s.pattern_stats()
fig, ax = plt.subplots(figsize=(6, 4.4))
ax.imshow(std, cmap="gray")
ax.set_title("pattern contrast (standard deviation)")
ax.set_axis_off()

# %% [markdown]
# ## Patterns and filters
#
# A pattern as stored, and after the default filters: the average pattern of the scan (its
# static background) and a blurred copy of the pattern itself are subtracted, and the result is
# made zero-mean and unit-norm inside a circular mask. Every filter is a setting; CLAHE and the
# band pass, when on, are applied to the master pattern as well.

# %%
cell = 27 * 75 + 37
fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
axes[0].imshow(s.pattern(cell, processed=False), cmap="gray")
axes[0].set_title("as stored")
axes[1].imshow(s.pattern(cell), cmap="gray")
axes[1].set_title("default filters")
s.configure(clahe=True)
axes[2].imshow(s.pattern(cell), cmap="gray")
axes[2].set_title("+ CLAHE")
s.configure(clahe=False)
for a in axes:
    a.set_axis_off()

# %% [markdown]
# ## The pattern centre, by a vote
#
# Start from a pattern centre that is wrong by about a tenth of the pattern in every coordinate.
# 25 patterns on a 5 × 5 lattice over the map are indexed by spherical cross-correlation under
# each candidate pattern centre of a grid; each votes for a candidate with the height of its best
# correlation over all orientations, and the grid's winner is where the patterns' best candidates
# lie densest. Three levels, each narrower and centred on the last winner.

# %%
s.set_geometry(pc=(0.50, 0.30, 0.55))
vote = s.fit_pc()
print({k: (np.round(v, 5).tolist() if isinstance(v, list) else v) for k, v in vote.items()})

# %% [markdown]
# The last level's votes, at the candidate distance nearest the fitted one. The peak is sharp
# because a wrong centre distorts the pattern on the sphere so that no rotation of the master
# fits it.

# %%
scores, xs, ys, zs = s.pc_scores()
iz = int(np.argmin(np.abs(zs - vote["pc"][2])))
fig, ax = plt.subplots(figsize=(4.6, 4))
im = ax.imshow(scores[iz], extent=(xs[0], xs[-1], ys[-1], ys[0]), cmap="viridis", aspect="auto")
ax.plot(*vote["pc"][:2], "w+", ms=12)
ax.plot(*pc_file[:2], "rx", ms=8)
ax.set_xlabel("pattern centre x")
ax.set_ylabel("pattern centre y")
ax.set_title("mean best NCC of the fit patterns")
fig.colorbar(im)

# %% [markdown]
# ## The pose fit
#
# From the vote, the detector's position and the 25 patterns' orientations are refined together
# by Levenberg–Marquardt on the Jacobian of the normalized simulations.

# %%
pose = s.fit_pose()
print("pattern centre:", np.round(pose["pc"], 5), " kikuchipy's:", np.round(pc_file, 5))
print(f"mean NCC {pose['start_ncc']:.4f} → {pose['ncc']:.4f} in {pose['iterations']} iterations")

# %% [markdown]
# ## Indexing
#
# Every pattern: projected back onto the sphere from where the beam stood, correlated with the master over
# all orientations, and refined on the detector.

# %%
res = s.index()
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
axes[0].imshow(s.map("ipf_z"))
axes[0].set_title("inverse pole figure, sample normal")
im = axes[1].imshow(res["ncc"], cmap="gray")
axes[1].set_title("NCC")
fig.colorbar(im, ax=axes[1])
for a in axes:
    a.set_axis_off()

# %% [markdown]
# A pattern beside its simulation at the orientation found for it.

# %%
q = res["quaternions"].reshape(-1, 4)[cell]
sim = s.simulate(q[None], cells=[cell])[0]
fig, axes = plt.subplots(1, 2, figsize=(6, 3.2))
axes[0].imshow(s.pattern(cell), cmap="gray")
axes[0].set_title("experimental")
axes[1].imshow(sim, cmap="gray")
axes[1].set_title(f"simulated, NCC {res['ncc'].ravel()[cell]:.3f}")
for a in axes:
    a.set_axis_off()

# %% [markdown]
# ## Against kikuchipy
#
# The scan file carries the orientations kikuchipy refined (as quaternions in
# `reference_quaternions.f32`). The disorientation between the two results, under the cubic
# symmetry:

# %%
q_ref = np.fromfile(DATA / "nickel" / "reference_quaternions.f32", "<f4").reshape(-1, 4)
dis = np.degrees(s.disorientation(res["quaternions"].reshape(-1, 4), q_ref))
print(f"median {np.median(dis):.3f}°, within 3°: {100 * (dis < 3).mean():.1f} %, mean NCC {res['ncc'].mean():.3f}")
fig, ax = plt.subplots(figsize=(5, 3))
ax.hist(np.clip(dis, 0, 3), bins=60)
ax.set_xlabel("disorientation from kikuchipy's orientation (°), clipped at 3")

# %% [markdown]
# ## Sweeps
#
# Most patterns do not need the full bandwidth. With `sweeps=2` the scan is first indexed at
# `sweep_bandwidth`, where the correlation costs an eighth. A match is accepted when its mutual
# information stands `accept_z` standard deviations above chance (above the same patterns
# against the master at random orientations, which every batch measures) and its refinement
# had settled; only what is left goes on to the full bandwidth.

# %%
import time

t0 = time.perf_counter()
s.index()
t_one = time.perf_counter() - t0
s.configure(sweeps=2, sweep_bandwidth=32)
print("the sweeps' (bandwidth, dictionary grid):", s.sweep_plan)
t0 = time.perf_counter()
two = s.index()
t_two = time.perf_counter() - t0
s.configure(sweeps=1)
dis2 = np.degrees(s.disorientation(two["quaternions"].reshape(-1, 4), q_ref))
first = (two["sweep"] == 0).mean()
print(f"one pass {t_one:.2f} s, two sweeps {t_two:.2f} s; {100 * first:.1f} % accepted at bandwidth 32")
print(f"median {np.median(dis2):.3f}° from kikuchipy's, within 3°: {100 * (dis2 < 3).mean():.1f} %")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
axes[0].imshow(s.map("sweep"))
axes[0].set_title("accepted in sweep 1 (blue), 2 (orange), never (dark)")
axes[0].set_axis_off()
axes[1].hist(two["z"].ravel(), bins=80)
axes[1].axvline(s.settings["accept_z"], color="k")
axes[1].set_xlabel("z: the match's NMI above chance, in chance's standard deviations")

# %% [markdown]
# ## Smaller patterns for the search
#
# `binning` reduces every pattern on the CPU as it is read (a whole factor bins; any other
# resamples by area). `full_resolution` says what still uses the stored size: here the search
# runs on 30 × 30 patterns and the refinement and the scores on the stored 60 × 60.

# %%
s.configure(binning=2, full_resolution="refine")
print("the search sees", s.pattern_shape, "of the stored", s.shape(full=True))
small = s.index()
s.configure(binning=1, full_resolution="none")
dis3 = np.degrees(s.disorientation(small["quaternions"].reshape(-1, 4), q_ref))
print(f"median {np.median(dis3):.3f}° from kikuchipy's, within 3°: {100 * (dis3 < 3).mean():.1f} %")

# %% [markdown]
# ## Averaging with neighbours
#
# `nlpar=True` replaces every pattern, before the filters, by a weighted mean of itself and
# its neighbours in the map: a neighbour weighs as much as it looks like the same pattern
# under other noise (NLPAR). By default it is the published method, one pass over 7 × 7 cells;
# `nlpar_iterations` above 1 runs several passes over a smaller window instead, each on the
# patterns the last one left, so that a small grain is averaged only with its own kind. The log
# says how much of the noise is left and how much of an average is the pattern itself.

# %%
plain = s.pattern(cell)
s.configure(nlpar=True)
fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2))
axes[0].imshow(plain, cmap="gray")
axes[0].set_title("default filters")
axes[1].imshow(s.pattern(cell), cmap="gray")
axes[1].set_title("averaged with neighbours first")
for a in axes:
    a.set_axis_off()
quiet = s.index()
print("\n".join(e["text"] for e in s.events() if e["kind"] == "log" and "averaging" in e["text"]))
s.configure(nlpar=False)
dis4 = np.degrees(s.disorientation(quiet["quaternions"].reshape(-1, 4), q_ref))
print(f"mean NCC {res['ncc'].mean():.3f} → {quiet['ncc'].mean():.3f}, median z {np.median(res['z']):.0f} → {np.median(quiet['z']):.0f}; median {np.median(dis4):.3f}° from kikuchipy's")

# %%
# the checks that make this tutorial a test
assert quiet["ncc"].mean() > res["ncc"].mean() + 0.03 and np.median(dis4) < 0.6 and (dis4 < 3).mean() > 0.97, (quiet["ncc"].mean(), np.median(dis4))
assert np.abs(np.array(vote["pc"]) - pc_file).max() < 0.015, vote["pc"]
assert np.abs(np.array(pose["pc"]) - pc_file).max() < 0.004, pose["pc"]
assert np.median(dis) < 0.6 and (dis < 3).mean() > 0.97, (np.median(dis), (dis < 3).mean())
assert res["ncc"].mean() > 0.45
assert np.median(dis2) < 0.6 and (dis2 < 3).mean() > 0.965 and first > 0.8, (np.median(dis2), (dis2 < 3).mean(), first)
assert np.median(dis3) < 0.6 and (dis3 < 3).mean() > 0.965, (np.median(dis3), (dis3 < 3).mean())
s.close()
