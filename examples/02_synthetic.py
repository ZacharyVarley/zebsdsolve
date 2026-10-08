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
# # A scan with a known answer
#
# The engine can simulate the patterns it indexes, so a scan can be made whose geometry,
# orientations and phases are known exactly. This one is wide enough for the beam's travel to
# show on the detector, holds two phases (cubic nickel and hexagonal, polar GaN), and is
# as noisy as its signal. The fits and the indexing are then checked against the truth.

# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import zebsdsolve as ze

DATA = Path(os.environ.get("ZEBSDSOLVE_TESTDATA", ""))
if not (DATA / "nickel").is_dir():
    DATA = next(p / "testdata" for p in [Path.cwd(), *Path.cwd().parents] if (p / "testdata" / "nickel").is_dir())

rng = np.random.default_rng(3)
s = ze.Solver()
s.add_phase(DATA / "nickel" / "Ni_master_20kV.npz")
s.add_phase(DATA / "gan" / "GaN_master_20kV.npz")
for p in s.phases:
    print(p["name"], "· point group number", p["point_group"], "· Laue class", p["laue"])

# %% [markdown]
# ## Making the scan
#
# 16 × 24 cells with 60 µm steps, seen by a detector of 60 × 60 pixels of 560 µm: the beam
# travels 1.4 mm across the map, 4 % of the detector's width. `simulate` renders each cell as
# that cell would record it, from where the beam stood.

# %%
rows, cols = 16, 24
n = rows * cols
pc_true = np.array([0.46, 0.23, 0.52])
s.set_patterns(np.zeros((rows, cols, 60, 60), np.float32))
s.set_geometry(pc=pc_true, sample_tilt_deg=70, px_um=560, step_x_um=60, step_y_um=60)

q_true = rng.normal(size=(n, 4)).astype(np.float32)
q_true /= np.linalg.norm(q_true, axis=1, keepdims=True)
q_true[q_true[:, 0] < 0] *= -1
phase_true = (rng.random(n) < 0.4).astype(int)  # 40 % GaN
cells = np.arange(n)
sims = np.empty((n, 60, 60), np.float32)
for k in (0, 1):
    m = phase_true == k
    sims[m] = s.simulate(q_true[m], cells=cells[m], phase=k)
noisy = sims + sims.std() * rng.normal(size=sims.shape).astype(np.float32)
s.set_patterns(noisy.reshape(rows, cols, 60, 60))
s.configure(static_background="none", dynamic_background="none")

fig, axes = plt.subplots(1, 4, figsize=(11, 3))
for a, k, title in zip(axes, (0, 0, cols - 1, n - 1), ("clean, corner", "noisy, corner", "noisy, far column", "noisy, far corner")):
    a.imshow(sims[k] if title.startswith("clean") else noisy[k], cmap="gray")
    a.set_title(title)
    a.set_axis_off()

# %% [markdown]
# ## The fits, from a wrong geometry
#
# The vote finds the pattern centre; the pose fit then refines the detector's whole pose, its
# position and its attitude, together with the orientations of a lattice of patterns.
#
# The attitude is the weak part. Turning the detector about the source and turning every
# crystal the other way change any one pattern alike; only the beam's travel between patterns
# tells them apart, and here that is 4 % of the detector under noise as strong as the signal.
# This detector was simulated at its nominal attitude, so the swing, tilt and twist the fit
# reports are its error, and every orientation indexed afterwards carries that same turn. On a
# real detector the nominal attitude is itself a guess, and a wider map, or more patterns in
# the fit (`pose_grid`), is what pins it.

# %%
s.set_geometry(pc=(0.52, 0.31, 0.58))
vote = s.fit_pc()
pose = s.fit_pose()
turn = np.degrees(pose["rot"])
print("true          ", pc_true)
print("vote          ", np.round(vote["pc"], 4), f" mean NCC {vote['score']:.3f}")
print("pose fit      ", np.round(pose["pc"], 5), f" mean NCC {pose['ncc']:.4f}  over {pose['n_patterns']} patterns")
print("swing, tilt, twist (°), true 0:", np.round(turn, 3))

# %% [markdown]
# ## Indexing, and the phases
#
# Each phase proposes its own best orientation for every pattern; the pattern takes the phase
# whose simulation shares the most information with it.

# %%
res = s.index()
found = res["phase"].ravel()
right = found == phase_true
dis = np.full(n, np.nan)
for k in (0, 1):
    m = right & (phase_true == k)
    dis[m] = np.degrees(s.disorientation(res["quaternions"].reshape(-1, 4)[m], q_true[m], phase=k))
print(f"phase right for {100 * right.mean():.1f} % of the patterns")
print(f"orientation error: median {np.nanmedian(dis):.3f}°, largest {np.nanmax(dis):.3f}° (the detector's fitted turn is {np.linalg.norm(turn):.3f}°)")

fig, axes = plt.subplots(1, 3, figsize=(11, 3))
axes[0].imshow(phase_true.reshape(rows, cols), cmap="coolwarm")
axes[0].set_title("true phase")
axes[1].imshow(s.map("phase"))
axes[1].set_title("found")
im = axes[2].imshow(dis.reshape(rows, cols), cmap="magma")
axes[2].set_title("orientation error (°)")
fig.colorbar(im, ax=axes[2])
for a in axes:
    a.set_axis_off()

# %% [markdown]
# ## What the beam's travel is worth
#
# The same scan indexed as if every pattern came from one point of the sample (the detector's
# pixel size forgotten): the correlation drops toward the map's edges, the farthest from that
# point.

# %%
ncc_with = res["ncc"].copy()
s.set_geometry(px_um=0)
flat = s.index()
s.set_geometry(px_um=560)
fig, axes = plt.subplots(1, 2, figsize=(9, 3))
for a, v, title in zip(axes, (ncc_with, flat["ncc"]), ("with the beam's travel", "as if from one point")):
    im = a.imshow(v, vmin=min(ncc_with.min(), flat["ncc"].min()), vmax=ncc_with.max(), cmap="gray")
    a.set_title(f"{title}\nmean NCC {v.mean():.3f}")
    a.set_axis_off()
fig.colorbar(im, ax=axes)

# %%
# the checks that make this tutorial a test
assert np.abs(np.array(vote["pc"]) - pc_true).max() < 0.015, vote["pc"]
assert np.abs(np.array(pose["pc"]) - pc_true).max() < 0.002, pose["pc"]
assert right.mean() > 0.97, right.mean()
assert np.linalg.norm(turn) < 0.6, turn
assert np.nanmedian(dis) < 0.6, np.nanmedian(dis)
assert ncc_with.mean() > flat["ncc"].mean()
s.close()
