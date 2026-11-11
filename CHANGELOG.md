# Changelog

What changed in each release, newest first. Work lands under an **Unreleased** heading; a release renames that
heading to the version and date, and the release workflow publishes the section as the release's
notes.

## Unreleased

### Fixed

- Orientations were reduced to the fundamental zone by the point group's rotations as tables
  write them (the unique axis along z, a two-fold along x), whatever axes the master pattern's
  crystal was set up with. For a master in another setting (a monoclinic crystal with the
  unique axis b; trigonal P-31m against P-3m1; EMsoft's overlap masters of twins) a stored
  orientation, its Euler angles in the exports, and the fits' poses could be another
  orientation, simulating another pattern, though the match itself was right. Each master's
  symmetry is now measured, when it is loaded, from the operators its file gives or its space
  group and cell imply, kept only where its intensities obey them; orientations are reduced
  about the master's own axes, and the dictionary, `reduce`, disorientations and the inverse
  pole figure colours follow the same rotations. Nothing changes for masters in the usual
  setting (the bundled nickel and GaN); results indexed with others before should be indexed
  again. Found by [Tijmenvermeij](https://github.com/Tijmenvermeij) (#3).
- `zig build test` now runs the symmetry module's own tests, which it had left out.

## 0.2.0 — 2026-11-10

### Geometry

- The detector's geometry is fitted as a vote on the pattern centre and then a plane of source
  points: how the beam's source moves over the scan. The SE(3) pose fit is removed.
- The bundled nickel scan's `detector_pixel_size` is set to 70 µm (the file had 1, kikuchipy's
  value for unknown); see `testdata/README.md`.

### Simulator

- Master patterns are simulated natively from a CIF file (the app's Simulate tab and the Python
  API's `simulate`): a CIF reader, the Joy Monte Carlo energy and depth distribution from a small
  surrogate network, and a dynamical solve per energy bin.
- Bethe weak beams enter both the strong-beam matrix and the intensity. The weak-beam cutoff c₂ is
  exposed and defaults to 40, and the correction is on by default.
- The number of beams is limited by the device's workgroup memory (512, 1024 or 1536 beams for 16,
  32 or 48 KiB), not fixed at 512; each energy bin is compiled at the smallest size that fits, and
  strong beams past capacity are followed as weak beams.
- Per-site Debye–Waller factors and the Bethe cutoffs are editable; `edit_crystal` in the Python
  API.
- A simulated master can be added as a phase or saved as `.npz`.

### App

- Master view: globe, Lambert, stereographic and joint (single-square) Lambert projections, with
  zoom and pan; one hemisphere for centrosymmetric crystals and an N − S′ difference for the
  others; a stereogram mode (distance and eye separation, default 0.5 m and 60 mm); several masters
  can be switched between and toggled on or off; Ctrl + scroll turns about the north pole.
- Crystal pane: the unit cell drawn by the engine, with an optional 3 × 3 × 3 supercell, a, b and c
  views over the picture, and a lock to the master's rotation.
- Energy and depth charts run from the beam's energy down to 0, with arrow-key navigation and an
  integrated view.
- The region of interest is drawn as a box on the map; the map takes arrow keys.
- The pattern stage (stored, filtered, background) and difference or split are chosen in place;
  the static background can be viewed.
- The Patterns tab is laid out as one card per processing step; the Simulate panel is reorganised
  with the run and its result at the top.
- What the file reader found and the simulator's summaries go to the log.

### Fixed

- Spherical indexing could give a pattern a wrong orientation and a low NCC depending on the GPU
  work before it: a dispatch's two uniform blocks could be split by the arena filling up, the
  first then overwritten before it ran (#2, by [Tijmenvermeij](https://github.com/Tijmenvermeij)). The master view's dispatch is
  guarded the same way.
- ANG and CTF exports write each phase's lattice parameters, read from ebsdsim and EMsoft masters
  and taken from the crystal for simulated ones, in place of `1 1 1 90 90 90`; a phase whose
  master carries none is refused (#1, by [Tijmenvermeij](https://github.com/Tijmenvermeij)).

## 0.1.0 — 2026-10-08

First release: indexing EBSD scans by dictionary matching against master patterns on the GPU
(Zig and WebGPU), as a browser app and a Python package.
