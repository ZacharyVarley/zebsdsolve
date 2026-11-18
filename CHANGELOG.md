# Changelog

What changed in each release, newest first. Work lands under an **Unreleased** heading; a release renames that
heading to the version and date, and the release workflow publishes the section as the release's
notes.

## Unreleased

## 0.2.2 — 2026-11-18

### Added

- **Auto fit geometry** takes the simplest model of how the source moves over the scan that the
  map supports: one pattern centre, a tilt line (PCx the same everywhere, PCy and PCz along the
  rows) or the plane, compared by their error on 6 × 6 blocks of the map left out of the fit,
  the simplest within one standard error of the best. On the nickel example the whole map takes
  the plane and a strip six columns wide the tilt line. `fit_geometry()` reports the model and
  each one's error (`model`, `models`).
- The plane can be fitted to all the fit patterns at once, each pattern's orientation
  following, instead of through each pattern's own source point (`plane_method` `joint`; the
  per-pattern fit stays the default: on simulated scans the two are equally accurate, and the
  joint fit's points lean the comparison of models toward the full plane).
- An EDAX `.up1` / `.up2` can be opened with its scan's `.ang`, which gives the map's shape, the
  steps, the image quality map and the pattern centre: select or drop the two together, add the
  `.ang` from the scan facts after the `.up1` / `.up2`, or `Solver.open("scan.up1", "scan.ang")`.
  The `.ang` must match the patterns cell for cell (#8, by
  [Tijmenvermeij](https://github.com/Tijmenvermeij)).
- The map shows the quality the scan file stores for each pattern (Oxford's band contrast,
  EDAX's image quality, Bruker's Radon quality) the moment the file opens, read without reading
  a pattern; a file that stores none shows the patterns' contrast as before, and
  **calculate contrast** gives it for any file. `Solver.stored_overview()` in Python
  (#5, by [Tijmenvermeij](https://github.com/Tijmenvermeij)).
- Indexing says how long it has left (#9, by [Tijmenvermeij](https://github.com/Tijmenvermeij)).
- A cell can be indexed and simulated under another geometry than the one in use, which stays
  as it is: `Solver.cell_under(cell, pc=…)` in Python, `cellUnder` in JavaScript.

### App

- The settings are short names beside their values, two to a row, under plain section names,
  with what is rarely changed folded under **Advanced**; a fit's or indexing's results are a
  small table of names and values.
- Geometry: the pattern centre the file gave, the one in use (typed in to set it) and their
  difference in one table; the search's range, slices and tiers first. The dock shows the
  selected pattern beside its simulations under the two centres in turn, each with its own
  refined orientation, and the search's tiers as a volume and as every slice along one axis.
  The plane's view shows each coordinate's trend along the rows and the columns, or the three
  as one colour over the map (x and y as hue, z as lightness) for the fit, the patterns' own
  values and their difference.
- The map's IPF colours are chosen beside it once a map is indexed (the axis, and the key), with
  a key to show or hide; the key is Nolze and Hielscher's by default.
- The dock follows the step: the search and the plane on Geometry, the master on Data and
  Patterns, the matches above chance on Index.

### Changed

- A chunked HDF5 scan's first pattern no longer waits for the whole chunk index: the reader
  reads only the paths through the chunk tree to the chunks a read needs (keeping the inner
  nodes), where it read every leaf first. On a 501 × 601 EDAX scan of one chunk per pattern the
  first pattern reads 40 KB in 9 reads instead of 16 MB in 5,384, which over a network share or
  from a server is the difference between at once and a wait. Found by
  [Tijmenvermeij](https://github.com/Tijmenvermeij) (#6).
- `ipf_colors` defaults to `perfect` (Nolze and Hielscher's key) instead of `classic`.

### Fixed

- EDAX's pattern centre, read from an EDAX h5ebsd file and written to `.ang`, scaled y and z by
  the pattern's width; EDAX measures y in fractions of its height and z of its smaller side.
  Square patterns were unaffected; on rectangular ones the pattern centre was wrong. An `.ang`
  written by 0.2.1 or earlier is read as it was written (found through #8).
- A region of interest drawn on one scan stayed in force on the next, which could quietly limit
  its fits and indexing; a scan now opens on the whole map, in Python as in the app, and a
  project opened with its scan keeps its own region (#9, by
  [Tijmenvermeij](https://github.com/Tijmenvermeij)).
- EDAX `.up1` / `.up2` files of version 3 and later had their map size and steps read one byte
  early, so a rectangular scan could open as one row with wrong steps; a file that ends partway
  through a pattern, or whose header is cut short, is refused (#7, by
  [Tijmenvermeij](https://github.com/Tijmenvermeij)).
- `closeScan()` in the JavaScript module closed the scan at once, even under a call still waiting
  on the file's bytes; it now waits for the calls before it, as every other call does (#6).

## 0.2.1 — 2026-11-11

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
- In the app, switching a master pattern off or on, or adding one, left patterns indexed on the
  spot naming phases by their old places in the list, and a pattern being drawn while the list
  was rebuilt could look up a phase no longer there. The pattern view could then draw the wrong
  phase's simulation, or fail with "Cannot read properties of undefined". Both are dropped now
  whenever the list changes.

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
