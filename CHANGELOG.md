# Changelog

What changed in each release, newest first. Work lands under an **Unreleased** heading; a release renames that
heading to the version and date, and the release workflow publishes the section as the release's
notes.

## Unreleased

### Added

- **`scripts/bench.py`**: a fixed set of synthetic scans with their answers (the engine's toy
  scans from fixed seeds, over pattern sizes from 60 × 60 to 240 × 320, noise, faint bands,
  a rotated detector, a wide map, a hexagonal phase, two phases with blank cells, a start far
  off), on which the vote, the plane fit, Auto and indexing are measured against the answer.
  `--root` runs another checkout and `--compare` sets runs side by side: changes to the fits
  and to indexing are decided on it.

- **Reference frames found from the data.** A convention (the sample's axes as signed axes of
  the frame the engine computes in, kikuchipy's; passive or active angles; the crystal's axes
  on the lattice, X ∥ a or X ∥ a\* and the like) is described as data, never by a maker's name,
  and `identify_frame()` finds the one a scan file's own stored orientations are in. What is
  left is a misorientation (`misorientation`, and how it changes over the rows and the
  columns), reported and never applied: it is where the two geometries or indexings differ.
  On an Oxford, a Bruker and an EDAX acquisition the makers' orientations lie a median 0.20°,
  1.16° and 0.76° from the engine's. Oxford's sample and crystal axes come out as #13 (by
  [Tijmenvermeij](https://github.com/Tijmenvermeij)) had them from Oxford's documentation and
  MTEX. `.ang` and `.ctf` write in the convention found (or set: `Solver.frame`, the selector
  under Export, a project file) and name it in their headers; `stored_orientations()` gives a
  file's own as stored. The readers keep them for Oxford, Bruker and EDAX h5ebsd and the `.ang`
  beside a UP file. See [docs/conventions.md](docs/conventions.md).
- **The detector orientation an Oxford file stores** (`Detector Orientation Euler`) is read
  into the geometry, as #13 read it. Left out, the engine's orientations on an AZtec map lay
  3.4° from AZtec's; with it, 0.10°.
- **A menu** in the app (the three bars left of the name): opening and saving, the theme
  (dark, light, or as the system is set), and **View conventions**, which draws each reference
  frame and pattern-centre convention (the lab with the sample's axes over the engine's, and
  the pattern as stored with where its pattern centre is measured from), the scan's own once
  identified among them.
- **`.h5oina` export** of a scan opened from an Oxford acquisition (`solver.export("x.h5oina")`,
  and among the other formats under Export): the acquisition's groups, datasets and attributes copied (not its
  patterns, nor spectra), its indexing replaced by these results in the convention found from
  its own orientations, each cell's pattern centre in Oxford's convention, and the engine's
  scores under `/zebsdsolve`. On an Oxford zirconium map its orientations lie a median 0.20°
  from AZtec's, and its header's detector orientation is the geometry's. The file is written by the engine's own HDF5 writer, in both the browser and
  Python, so neither needs an HDF5 library; the HDF5 reader now reads attributes. After #13 (by
  [Tijmenvermeij](https://github.com/Tijmenvermeij)), which wrote it from the front ends.

### Changed

- **Full size for: everything** is gone (`full_resolution` `"all"`): it ran the search, the
  refinement and the scores on the patterns as stored while the reduced size was only looked
  at, which is what `binning` 1 does without reading every pattern twice. A project or a
  script that sets it is refused; set `binning` 1 instead (#19, by
  [Tijmenvermeij](https://github.com/Tijmenvermeij)).
- The inverse pole figure keys' gradients are no longer settings: `ipf_lightness` and
  `ipf_gray` are gone, and the Nolze–Hielscher keys use the λ_L and λ_S of a quarter each that
  the paper recommends, so a map's key is always one others draw. The key itself is still a
  choice (`ipf_colors`).
- The Index step shows the method's own parameters under it, and changes them as the method
  changes (the bandwidth for spherical indexing, the grid for a dictionary, and the grid,
  components and Oja passes for its components); the sweeps are under Advanced, with the early
  sweeps' parameters likewise.
- Rotations are named as rotations: the detector's angles away from its nominal attitude are
  azimuthal, tilt and twist (kikuchipy's names), and the plane fit reports the scan rotation.

### Fixed

- In the app: the log's times no longer wrap onto a second line; a lattice parameter tied to
  another reads as the one it follows (3.524, not 3.5240 beside it); the crystal's facts wrap
  instead of being cut off, as do wide readouts (a sweep plan, the plane fit's misfits); the
  pattern centre in use reads to the four places the file's does; the z histogram says how many
  of the cells above the threshold were accepted (acceptance also asks the refinement to have
  settled, so the two counts can differ); the pattern-centre search's legend shows its marks in
  the light theme; the plane fit's panel titles no longer run into their numbers in a narrower
  window; the beam's step directions are under Geometry, Advanced (every scan's geometry uses
  them), not under Raw stream; the conventions drawing's labels keep off each other and its
  lines; every choice draws the same caret, and a setting's value reads as text beside its
  muted label; the map's region and key tools and the fit-pattern toggle wait for a scan; the
  plane fit's distance map keeps its edge squares inside the map; the page has an icon.
- `.ctf` and `.ang` wrote a hexagonal phase with the masters' crystal axes (X ∥ a), 30° about c
  from Oxford's (X ∥ a\*), and in a sample frame rotated 90° from Oxford's and Bruker's: on an
  Oxford map of zirconium the `.ctf` lay a median 84° from AZtec's own orientations. A scan
  that stores its maker's orientations now exports in the convention found from them (0.20°
  there, with the file's detector orientation read); the header says which.
- A Bruker h5ebsd whose patterns were clipped (`UnClippedPatternHeight` above `PatternHeight`)
  has its PCY read as a fraction of the height before clipping, from the top: Esprit's own
  per-pattern pattern centres agree with its tilts and step only so, and the patterns put
  their pattern centre 2 px from it (9 px from the reading before). On a ferrite map the
  misorientation to Esprit's orientations fell from 1.75° to 1.02°.

- **Auto fit geometry**, asked for a projective plane, said projective when it took a tilt line
  and installed it as the affine plane it is (#10, by [Tijmenvermeij](https://github.com/Tijmenvermeij)).
- The joint fit of a projective plane stopped with `TooFewFitPatterns` when it started from one
  pattern centre for every cell (no steps, no plane) (#11, by [Tijmenvermeij](https://github.com/Tijmenvermeij)).
- `cell_under(c, pc=…)` after a plane fit kept the plane and so ignored the centre given; it now
  takes the centre as `set_geometry` would (#12, by [Tijmenvermeij](https://github.com/Tijmenvermeij)).
- In the browser, a GPU device lost during indexing failed a pending readback with a `mapAsync`
  error and left the page working on a dead device. The app now says the device is lost, stops
  GPU work and keeps the orientations already indexed for `.ang` and `.ctf` export; the log
  records the adapter, bandwidth, batch and tile size (#14, by [Tijmenvermeij](https://github.com/Tijmenvermeij)).
- A projective plane through source points that hardly move (a small region of interest) took
  its two slopes from their noise, the plane's other numbers making up for them where the points
  were and nowhere else: on a 3 × 3 region of the nickel example the source at the map's middle
  came out at (130857, −40056, 83901). The slopes are now held near zero unless the points tell
  them (a prior of a tenth on how far they move the denominator from the map's middle to its
  edge, against the points' own scatter), and a plane whose denominator falls below a tenth of
  its middle's anywhere on the map is refused. Small regions now give the affine plane; the
  nickel example's whole map barely moves, and what its plane says of the detector's rotation
  varies across blocks of the map by 15° at most instead of 199°.
- The joint fit (`plane_method="joint"`) of a projective plane had no such hold: on a 12 × 12
  region of the nickel example (CLAHE on) its denominator went through zero inside the map. It
  now holds the slopes the same way, the prior weighed against its pseudo-points' scatter, and
  takes no step past the same bound; the denominator moves under 2 % there, and 15 % rather
  than 17 % over the whole map.

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
  can be switched between and toggled on or off; Ctrl + scroll rotates about the north pole.
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
