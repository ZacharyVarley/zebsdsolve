# zebsdsolve: how it works, and what was measured

The long form of the [README](../README.md): the app pane by pane, each method with the numbers behind its defaults, what is not verified, and the references.

EBSD pattern-centre fitting and indexing on the GPU, in the browser or from Python. Point it at a scan of Kikuchi patterns, of any size, and at a master pattern for each phase that might be in it (or simulate one from the phase's CIF); look at the patterns, fit the detector geometry, and index the map.

One engine does the work for both front ends: the math is WGSL, the schedule is Zig, and it runs natively (Vulkan, Metal or Direct3D 12 through wgpu-native) and as WebAssembly on WebGPU.

## Run the app

```bash
pip install zebsdsolve
python -m zebsdsolve
```

This serves the app on <http://127.0.0.1:4190/> and opens it in your browser. Everything runs on your GPU inside the page. A scan is read in place, in pieces, by the page itself, so a 100 GB file is neither uploaded anywhere nor loaded into memory. You need Chrome or Edge 137+ (WebGPU and JavaScript Promise Integration). `--port` picks another port, and `--no-browser` only prints the URL.

To install from a clone instead, run `pip install .` in it. The first build downloads Zig and wgpu-native into `.deps/` and compiles for your machine, which takes a minute or two.

The steps are tabs down the left, each with its own settings, under a header that names the scan and the GPU; at the foot of the column, a line says what the app is doing (a job's progress drawn along its top edge, with a Stop button) and opens into the log when clicked (errors turn it red and are counted on it until the log is opened). The panes beside the column take the window's whole height.

| Step | What it does |
|---|---|
| **Simulate** | A master pattern from a crystal structure, for a phase that has none or to look at one: open a CIF (or drop it on the window, or pick a preset; how it was read goes to the log), change what needs changing (the lattice parameters its crystal family leaves free, the others following; each site's occupancy and Debye–Waller B), set the beam and sample, the master's size, the Bethe cutoffs and the rank (the slider trades speed for accuracy), simulate (below), and add the result as a candidate phase or save it as an `.npz`. It needs no scan. On this step the map and the pattern give way: the master takes the window's height (the dock shows nothing else here), and beside it stands the crystal, its unit cell drawn by the engine (the cell's edges a, b, c and the angles between them named; pointing at a lattice parameter or a site in the rail picks it out on the cell) over the energy model's account of the backscattered electrons: their share in each energy bin from the beam's energy down to none, and their spread over energy and depth together (depth downward), with the depth profile of them all. After a simulation, a solved bin's bar (or the arrow keys, from the beam's energy down; **integrated** for all of them again), or a site's name, shows that bin's or that site's master alone. **Lock to master**, over the cell, makes the cell and the master one rotation. |
| **Data** | Open a pattern file (or drop it on the window) and add one master pattern per candidate phase. The map appears as each pattern's contrast, coarse at once and sharpening, so the microstructure shows before anything is indexed. A project `.json` saved under Index, dropped with the scan or after it, brings its settings and geometry back. With no scan at hand, **Make toy scan** makes one from the master patterns: a map cut into grains with random orientations and phases, its patterns simulated under the geometry set, laid on a background and counted as electrons with their shot noise, all from a seed, and its answer kept to check the indexing against. |
| **Patterns** | How every pattern is prepared, in order: the size it is reduced to as it is read (and what still uses the stored size), averaging with its neighbours (NLPAR), then the filters: static background (the scan's average pattern), dynamic background, CLAHE, band pass, circular mask. CLAHE and the band pass act on the master patterns as well. |
| **Geometry** | The pattern centre, tilts, detector rotation and pixel size, from the file where it records them (a pixel size the file does not record is estimated from the pattern's width, and can be typed in). **Fit centre** is the vote for one pattern centre for the map; **Fit plane** the plane of source points (below): every fit pattern's own source point, and the affine or projective transformation of the scan's rows and columns through them, which asks nothing of the steps, the pixel size or the tilt; **Auto fit geometry** is the one and then the other, from the geometry as it was given. Any of them can be undone, and the geometry put back as the scan was opened. |
| **Index** | Spherical indexing (or a dictionary) of the map or a region of it, each orientation refined on the detector, a batch at a time, in one pass or in sweeps of rising bandwidth; the map fills in as it goes. **Tune speed** fits the batch and workgroup sizes to the machine first. Export as `.ang`, `.ctf`, and the geometry and settings as JSON. |

Beside the steps are three panes, with bars between them to drag (with no scan open, the pattern's pane gives its room to the others):

- **The map** zooms with the scroll wheel about the pointer (never out past the whole map), a drag moves a zoomed map, and the arrow keys step the selected pattern (wherever the focus is but in a field). The region of interest, which the fits and indexing keep to, is a box drawn on the map: press **region** on its bar and drag (or shift-drag at any time); the bar then gives its size, and a cross clears it. Its corner says what its colours mean: the inverse pole figure key of the phase's Laue class in the colours chosen under Index, a scale, or the phases.
- **The pattern** selected on the map, at any stage of its preparation (as stored, the static background subtracted from every pattern, averaged with its neighbours, filtered: chosen under the image itself), beside its simulation, and the two set against each other: their difference, or both in one frame either side of a bar to drag across. A pattern nothing has indexed yet is indexed on the spot, as the fits index theirs, so the simulation and the scores are there for any cell and follow the filters and the geometry as you change them; so is one the map holds an orientation for, once the geometry is no longer the one the map was indexed under (that orientation is another geometry's match, and would show as bands out of place). With the averaging on, a small map in the pattern's corner shows how much of each neighbour went into it.
- **The dock**: the pattern-centre search, the plane fit, a master pattern, and the histogram of z. On the Geometry step it takes most of the window. The master is any candidate phase's (as it was loaded, or filtered as patterns are matched with it) or the last simulation's (all its energy bins and sites together, or one alone), rendered by the engine on the GPU in one of four projections: a globe; the two hemispheres as equal-area (Lambert) squares; as stereographic discs; or the whole sphere in one square, the upper Lambert square in the middle as a diamond and the lower one's four triangles folded out into the corners (the joint Rosca–Lambert layout), so that the pattern runs on unbroken across the equator. Drag to turn it (the point under the pointer stays under it, in every projection; the flat projections are locked by default, their poles where they are), scroll to zoom, control-scroll to spin it about the view's north pole, double-click for the first view; buttons look down the crystal's x, y and z. The globe can be a **stereogram**: two views, one for each eye, side by side for parallel viewing, as far apart on the screen as the eyes are (60 mm, on a screen taken to be 96 dpi) and turned from each other by the eyes' angle on the sphere at the distance to the screen (500 mm: 6.9°); both are sliders at the foot of the picture. The crystal's axes and its equator are drawn over it. The search is a volume of votes over (x, y, z) for each grid it scored: it shows whole, as a cloud to turn about with the mouse (each candidate a glow as bright as its vote, the next grid's extent and the way the search went drawn in it), beside three cuts through a cross-hair, or every slice along one axis side by side. A click in the cloud or on a cut moves the cross-hair; the cuts go through the pattern centre in use, and again whenever it changes, in the finest grid that holds it. The plane fit shows every fit pattern's own source point against the plane through them, PCx, PCy and PCz two at a time, and how far each lies from the plane across the map.

Scan formats: h5ebsd as written by kikuchipy, Bruker, EDAX and Oxford (`.h5oina`); EMsoft's simulated pattern files; EDAX `.up1` / `.up2`; Oxford `.ebsp` (uncompressed); NORDIF `Pattern.dat` with its `Setting.txt` (open the two together; Python finds the second beside the first). Master patterns: [ebsdsim](https://github.com/ZacharyVarley/ebsdsim) `.npz` and EMsoft / EMsoftOO `.h5`, or simulated here from a CIF. The HDF5 files are read by a reader written for this project (no HDF5 library), in Zig, which is what lets the browser read them in ranges.

## How it works

### Geometry: one detector, and where the beam stood

The geometry is one rigid pose of the detector in the sample's frame and, for every pattern, the point of the sample the beam stood on when it was recorded. The pattern centre (Bruker convention) is how the pose's position is written: the foot of the perpendicular onto the detector from the source point at the map's centre, with the detector distance. A pattern from anywhere else on the map sees that same detector from a source point displaced by the beam's travel. Until a plane of source points is fitted (below), that travel is the scan's steps in micrometres, turned into pixels by the detector's pixel size, so no pattern has a pattern centre of its own. A file that records no pixel size is given one from the detector's width over the pattern's pixels across, until one is typed in. Once a plane is fitted, the plane says where the beam stood for every cell, and the steps and the pixel size are not used.

### Indexing: cross-correlation over all orientations

A pattern is a window onto the Kikuchi sphere. With the detector geometry known, each pattern is projected back onto the sphere (in the sample frame, from the point the beam stood on) and expanded in spherical harmonics. Its normalized cross-correlation with a master pattern at every orientation is then one three-dimensional FFT away, following Gutman et al. [[1]](#references) and Lenthe et al. (EMSphInx) [[2]](#references):

- The correlation over the rotation S = Rz(a) Ry(b) Rz(c) is a Fourier series in (a, b, c) whose coefficients are sums over the degree l of products of the two functions' coefficients and Wigner d-matrices at π/2. The engine assembles that spectrum on the GPU, using the master's symmetry (a rotation axis along z thins its orders, an inversion centre removes odd degrees) and two symmetries of the series itself, each worth half the work: the correlation is real, and b and −b hold the same orientations. The b axis is therefore transformed first, straight from the sums, and only its half kept; the a axis last, as a real transform.
- The normalization is the window's: the denominator of a true normalized cross-correlation under the detector's footprint is itself two correlations, of the window with the master and with the master's square. They are computed once per geometry.
- The peak is interpolated, and then refined on the detector: Gauss–Newton steps on the normalized cross-correlation between the pattern and the master projected onto the detector from the pattern's source point, a few iterations entirely on the GPU. The Jacobian is analytic and lives on the sphere: turning the crystal by a small rotation φ changes the intensity seen along direction c by φ · (c × ∇M(c)), so the master's gradient field ∇M is computed once per master and each iteration only looks it up where its rays land. (This is the saving of inverse-compositional alignment, the costly derivatives taken once on the template, without its bookkeeping: the rotation group acts on the whole sphere, so there is no warp to compose and no image border to fall off.) The Gauss–Newton matrix is the correlation's curvature only for a pattern that is its simulation; for a real one the curvature is smaller, by the correlation itself and again where the pattern lacks the simulation's fine detail, so each step's gain is compared with what the model promised and the matrix rescaled by it.

On the nickel scan below this indexes about 23 000 patterns per second on a desktop GPU (15 000 in the browser, where each batch's round trip counts for more), and 67 000 in two sweeps.

### Bandwidth

The bandwidth L of the correlation decides how fine a detail of the pattern takes part, and the cost: the orientation grid has 64 points per turn up to L = 32, 128 up to 64 and 256 up to 128, and the work follows its cube. The pattern-centre vote and indexing each have their own (`fit_bandwidth`, `bandwidth`), because they want different things from it. On the nickel scan the vote lands within 0.003 of the pattern centre at 64 and within 0.002 at 96, seven times slower; at 32 it is off by 0.02 to 0.04. Indexing at 32 finds 96 % of what 64 finds, in a quarter to a half of the time.

### Dictionaries, and their components

`method: "dictionary"` compares each pattern with the simulated patterns of a grid of orientations pixel by pixel, and simulates them again for every batch. `method: "pca"` simulates the dictionary once. As it streams by, Oja's rule learns an orthonormal basis of the directions that hold most of its variance (`pca_components` of them; each step is followed by a Cholesky factorization of the basis's Gram matrix to keep it orthonormal), and every entry is kept as its coefficients in that basis. A batch of patterns is then projected onto the basis and matched by coefficients. On the nickel scan a grid of 40 (21 343 orientations) kept as 256 components indexes 36 000 patterns per second where the plain dictionary indexes 645, and finds the same 97 % within 1°; 128 components find 94 %. A dictionary of either kind is simulated for one source point, the map's centre; refinement works from each pattern's own.

### Sweeps

Indexing at bandwidth 32 finds 96 % of what 64 finds in a quarter to a half of the time, and that is what sweeps use. With `sweeps` above 1, the region is first indexed at `sweep_bandwidth` (or `sweep_dict_grid` for a dictionary), and each later sweep takes only the patterns no earlier one indexed acceptably, at a higher bandwidth, the last at `bandwidth`. A match is accepted when two things hold:

- Its normalized mutual information with its simulation reaches the threshold, which is one number for the whole scan. `accept_nmi` sets it outright. Left at 0, it comes from a calibration made once per scan: about 256 of the region's patterns are scored against the master at random orientations, and the threshold is that chance level plus `accept_z` (15) of its spreads; z in the results is the NMI on that scale. Nothing here is per pattern or per batch. The calibration exists because the chance level is not universal: it moves with the patterns' size and with the histogram. On the nickel scan below, where a correct match scores about 0.106, chance is 0.015 ± 0.002 as stored (60 × 60 pixels), 0.051 ± 0.006 binned to 30 × 30, and 0.037 with 32 histogram levels in place of 16. An absolute threshold of 0.04 to 0.05 suits that scan as stored (0.05 accepts 98.3 % of it; z ≥ 15, which is NMI ≥ 0.039 there, 99.2 %).
- Its refinement had settled: the last step that improved it turned the crystal by no more than `accept_step_deg`. A match whose iterations ran out on the way is not wrong, but it is not finished, and the next sweep starts it closer.

Every cell keeps the best match any sweep gave it, with its z and the sweep that accepted it (or none); the app draws both as maps and shows the z histogram with the threshold on it.

The sweeps before the last may search by another method than the last (`sweep_method`): the dictionary's components first and the spherical correlation for what they leave, say. Which first sweep is cheapest depends on the scan. Its cost is a search plus what every sweep pays: reading the patterns, filtering them and refining the matches. Where refinement is most of a cheap sweep's time, as it is for larger patterns, a faster search has little left to buy.

### Averaging with neighbours (NLPAR)

`nlpar` replaces every pattern, before any filter, by a weighted mean of itself and its neighbours in the map (Brewick, Wright and Rowenhorst [[3]](#references)). A neighbour weighs as much as it looks like the same pattern under other noise: the mean squared difference of the two patterns is compared with what their noise alone would give, and the weight is exp(−d / λ²) for a difference d standard deviations above that. Each pattern's noise is half its smallest squared difference from one of its eight nearest neighbours.

The default is the published method: one pass over 7 × 7 cells (`nlpar_radius: 3`, `nlpar_iterations: 1`) at `nlpar_lambda: 0.7`. Each of the three is a setting, in the app as in Python.

`nlpar_iterations` above 1 is a variant of mine: several passes, each on the patterns the pass before left, so three passes of 3 × 3 in place of one of 7 × 7. A pattern then reaches as far, but only through neighbours it resembles at every step, and the later passes decide their weights on patterns already quieter. Averaged patterns are not independent of each other, so their noise cannot be measured again the way the stored patterns' was. It does not need to be: every average is a known mix of stored patterns, the engine keeps those shares beside it, and the noise in the difference of two averages follows exactly. The test statistic so keeps its meaning in every pass, and `nlpar_lambda` with it.

What the passes buy is safety for small features. On a simulated map with a grain one cell wide under noise six times the patterns' contrast (`tests/test_api.py`), three passes of 3 × 3 at λ = 0.7 leave 46 % of the noise and index 19 of the thin grain's 20 cells (18 without any averaging), while one pass of 7 × 7 at the same λ leaves 29 % of the noise and 16 of the 20. That is one trade seen from two ends, not a free gain: at equal noise left, the two keep the thin grain about equally well in my tests (the wide pass somewhat better when averaging hard, the passes better under the heaviest noise).

What to set λ by is in view. The pattern view shows the selected pattern averaged (before the filters or after them) and redraws as the three settings change; beside it are how much of the pattern's noise its average keeps, how much of the average is the pattern itself, and a map of each neighbour's share. The log says the same of the first batch when indexing starts. `Solver.pattern(cell, "averaged")` and `Solver.nlpar_view()` give both in Python.

The averaging runs on the device as the patterns are read: a batch's cells with a margin of neighbours (as wide as the passes reach, plus one) are read and averaged, batches are square blocks of the map so that the margin is small beside them, and a cache of stored patterns keeps what neighbouring blocks share. On the nickel scan one pass of 7 × 7 leaves 50 % of the noise, raises the mean NCC from 0.53 to 0.62 and the patterns accepted at bandwidth 32 from 94.1 % to 97.4 %, at 66 000 patterns per second instead of 83 000 (three passes of 3 × 3: 64 % of the noise, 0.59, 96.9 %, at the same speed). With larger patterns the cost grows: putting each block's patterns and their margin together on the host is then what a batch waits for. A later sweep reads and averages its patterns' neighbourhoods again; where the patterns a first sweep left are spread over the whole map, that is the cost of averaging the map a second time.

### Tuning to a machine

Four settings decide only how the work is cut up: `batch` (patterns read and indexed per step), `refine_pixels` (pixels one GPU invocation adds up in refinement), and `wg_refine` and `wg_fft` (invocations per workgroup of refinement's sums and of the correlation's FFT kernels). Their defaults are meant to be reasonable on any GPU. Before a large scan, **Tune speed** in the app (`Solver.tune()` in Python) fits them to the machine and the scan in a few seconds.

It indexes a sample of the region, times it, and tries four moves in turn, each climbing from the values held: doubled while that is faster by more than 3 %, else halved while that is. The moves are `refine_pixels`; `batch` and `refine_pixels` together; `wg_refine`; `wg_fft`. The second is there because the first two are not independent: their fastest values rise together (256 with 128, 512 with 256, 1024 with 512), a ridge that a search of one setting at a time stops short of. All but the last are timed on the first sweep, where most patterns are indexed; `wg_fft` shapes the correlation alone and is timed on the last sweep, at the full bandwidth.

The sample is sized by what the search costs, so that a trial stays a fraction of a second at any bandwidth: 8192 patterns up to bandwidth 32, 4096 up to 64, 1024 up to 128 and 128 beyond (each step is where the orientation grid doubles and the correlation's cost grows eightfold). A trial is one batch untimed, in which a workgroup size not used before is compiled, and two timed runs of the sample.

That is 7 to 11 trials and 2 to 7 seconds, and on the nickel scan it is worth:

| | tuning takes | patterns per second, before → after |
|---|---|---|
| Nickel scan, bandwidth 64 | 3.3 s | 22 100 → 25 600 (+16 %) |
| Nickel scan, two sweeps | 1.8 s | 63 500 → 75 900 (+19 %) |

It lands on a batch of 512 or 1024 with `refine_pixels` 128 to 512, and on 128 or 256 invocations for the correlation's workgroups; refinement's workgroup stayed at 64 in all but one run. `tile_mb` (how many patterns of a batch are correlated at once) never mattered here and is left alone, and how many lines of a transform share a workgroup moved things by 4 % either way, so that stays fixed.

The orientations do not depend on these settings beyond rounding, and neither does acceptance: its threshold is one number for the scan.

### Reduced patterns

`binning` reduces every pattern on the CPU as it is read: each pixel of the reduced pattern is the average of the detector area it covers. A whole factor that divides the pattern is binning; any other factor resamples by area. `full_resolution` says what still works on the patterns as stored: nothing, the final scores (and with them the phase and the acceptance), the refinement too, or everything (the reduced size is then only for looking). The file is read once per batch either way.

### Filters

The patterns' filters run in a fixed order: the static background (the average of about a thousand patterns spread over the map, or the image the file stores) is subtracted, then the dynamic background, then CLAHE, then the band pass, then the mask. A test checks CLAHE against the same algorithm written out in NumPy. The dynamic background is a Gaussian blur of the pattern itself, a sixth of the pattern wide by default. A Gaussian that wide is computed from sums over boxes of pixels (a sixth of its sigma across), blurred on their own coarse grid and spread back, at a seventh of the cost of visiting every tap; a test holds it to the Gaussian carried all the way to the edges, which it matches to 0.03 % where the tap-by-tap kernel, cut off at three sigma, matches to 0.3 %.

A master pattern has to be filtered as the patterns are, or the two are no longer images of the same thing. So CLAHE and the band pass are applied to the master whenever they are applied to the patterns, on the sphere, at the sizes the pattern filters have there once a detector pixel is given its angular width: by default the pixel at the pattern centre, 1 / (pcz · pattern height) radians, or `pixel_angle_deg`.

### Pattern centre: a vote

The app keeps every grid the search scored and shows each as slices through any candidate: the density of the ballots (what chose the winner) or the patterns' mean correlation, with the next grid's extent, the pattern centres worth comparing (the start, the file's, the current one) and, per candidate, how many fit patterns score highest under it on their own. A click moves the cross-hair, and the candidate under it can be made the pattern centre to fit again from there.

A wrong pattern centre distorts a pattern on the sphere so that no rotation of the master fits it. So: take a few patterns on a coarse lattice over the map (5 × 5 by default, `fit_grid`) and index them by spherical correlation under each candidate pattern centre of a grid. Each pattern's best correlation over all orientations is then a surface over the candidates, high near the truth. Each candidate gets its own window normalization.

What matters is how the patterns' surfaces become one choice (`pc_vote`). Averaging them and taking the highest candidate (`mean`) is the obvious rule and a poor one: every surface also has broad ridges that are not the truth (a shift of the pattern centre traded against a turn of the crystal), the ridges of many patterns overlap, and their sum pulls the maximum along them. The default, `ballots`, asks where the patterns agree instead. Each pattern casts a ballot at each of its `pc_top_k` best candidates (12; only those scoring at least `pc_gate` × its own best), the ballots are spread by a Gaussian of a fixed width on the detector (`pc_kde_sigma`, 0.015 of the pattern), and the densest place wins. It is then moved once to the weighted centre of the ballots within `pc_inlier_radius` of it and found again among those alone. A pattern that is wrong is outvoted, not averaged in. This is the consensus I arrived at with sliding correlations in the camera plane (gated top-K ballots and a density mode); it carries over to the spherical correlation's volumes unchanged.

The search repeats around the choice on a narrower grid; if the choice is on the border of its grid, the grid moves onto it before it narrows. Measured on the same three grids (9 × 9 × 5 candidates) with 5 × 5 fit patterns unless the row says otherwise (2.1 s for the mean and 1.2 s for the ballots, which do not chase a border), as the largest error of the three coordinates:

| | mean | ballots |
|---|---|---|
| Nickel scan, four starts up to 0.1 away | 0.007, 0.018, 0.019, 0.018 | 0.003, 0.003, 0.003, 0.003 |
| The same at bandwidth 96 | 0.004 | 0.002 |
| The same at bandwidth 32 | 0.019 | 0.036 |
| Its 25 fit patterns with noise added, 0.3 of their spread (two draws) | 0.007, 0.029 | 0.002, 0.017 |
| … 0.45 of their spread | lost (0.6, 0.6) | 0.020, 0.019 |
| 81 fit patterns at that noise | lost (0.7, 0.7) | 0.005, 0.011 |
| 81 fit patterns, noise 0.6 of their spread | lost | lost |

The mean's error on the clean scans is nearly all in x, and the same from every start: the ridge. Two things did not help. Non-maximum suppression (`pc_nms`: a ballot only at a local maximum of a pattern's surface) lost every noisy case, as wider suppression did in the camera-plane tests. And a spread counted in candidate steps instead of on the detector got worse as the grids narrowed, because the patterns' own peaks scatter by about a hundredth of the pattern whatever the grid. At bandwidth 32 the surfaces are too blunt for either rule; use 64.

### The plane of source points

The vote gives one pattern centre for the map, and from it every other pattern's source is the beam's travel away: a square grid of the steps the file gives, in pixels of the size it gives, its rows straight down the tilted surface. A microscope's scan is none of those to better than a few percent or a few degrees. Winkelmann, Nolze, Cios, Tokarski and Bała [[10]](#references) calibrate the other way round, and kikuchipy [[4]](#references) after them (`refine_orientation_projection_center`, then `EBSDDetector.fit_pc`): each of a few patterns is fitted by itself, its orientation and where its own source stood against the detector, and a transformation of the scan's indices is fitted through those source points by least squares. Affine, nine numbers: the scan is a parallelogram on a plane. Projective, eleven: the beam is deflected from a pivot onto the tilted sample, and the scan's parallels need not stay parallel. Nothing is asked of the step, the pixel size or the tilt. They are what the plane says afterwards: its normal is the sample's, and the lengths and directions of its two edges are the scan's steps as the detector sees them.

A rigid pose of the detector over a known scan is three of the plane's six slopes. The other three are the lengths of the scan's two steps and the angle between them, and a fit that cannot move them puts their error where it can, into the tilt.

*Fit plane* (`fit_plane` in Python) does this with the scan's own patterns, and *Auto fit geometry* (`fit_geometry`) is the vote and then this. The fit patterns are every `plane_stride`-th cell each way (2; the stride is widened until there are no more than `plane_patterns_max`, 4096), indexed at the geometry as it stands, and those whose match indexing itself would not accept are left out: a pattern matched to the wrong orientation puts its source wherever suits its mistake. Each of the others is refined by itself over six unknowns, a shift of its source and a turn of its crystal, by Levenberg–Marquardt on its own normalized cross-correlation with its simulation. The GPU reduces each pattern to 43 sums, the correlation's gradient and Gauss–Newton matrix over those six directions; all the patterns take a step together, one readback a step, and a pattern's step is kept where its own correlation rose. A source may move `plane_trust` of the pattern's height, and one that ends at that edge is left out. The plane (`plane_kind`) is then fitted to the points, each weighed by how sharply its own pattern tells it (`plane_weights` `information`: the curvature of that pattern's score over its source's place, its orientation following), or all alike (`equal`, as kikuchipy fits a few good patterns). Points further than `plane_outlier` robust standard deviations from the plane go, and it is fitted again. The geometry takes the plane as kikuchipy's matrix P, with P · (column, row, 1) = w · (PCx, PCy, PCz, 1), and from then on every pattern's source is where the plane has it; the steps and the pixel size are not used. A pattern centre set by hand and the vote go back to one pattern centre and the scan's steps. What the plane says of the tilt, the scan's turn in its own plane, its two steps and the angle between them comes with a standard error each, from fitting the plane again without the points of one block of the map at a time (6 × 6 blocks).

The sample frame the orientations are in is a separate choice (`plane_frame`): `nominal` leaves it where the tilts and the detector's turn have it, and the plane only says where the beam stood; `follow` takes it from the plane, its normal the plane's and its x along the scan's rows, as kikuchipy sets its sample tilt from the plane. On a simulated map with the detector turned by 1°, −2° and 1.5° that the fit is not told of, orientations indexed afterwards lie a median 2.69° from the true ones with the frame nominal (the turn itself) and 0.06° with it following.

kikuchipy fits its plane to calibration patterns: a few, at the detector's full resolution, taken across the region before the scan. ni_gain 1 comes with nine, of 480 × 480 pixels, and its map has 29 800 patterns of 60 × 60. Fitted with this engine (as one 3 × 3 scan with no steps, each pattern's own source point read from the plane fit's points), the nine give source points with a mean of (0.42078, 0.21413, 0.50177) and a spread of (0.0029, 0.0019, 0.0006); kikuchipy's tutorial has (0.42068, 0.21330, 0.50038) and (0.0026, 0.0018, 0.0007) for the same patterns. The plane through them, and the plane from the map's own patterns (every cell, 29 020 used, 5 s), as the source's movement per map cell in units of 10⁻⁵ of the pattern:

| | Nine calibration patterns (480 × 480), each left out in turn for the error | The map's patterns (60 × 60) | The steps and pixel size as given, 70° |
|---|---|---|---|
| PCx per column | −5.25 ± 0.40 | −5.06 | −4.46 |
| PCy per column | −0.35 ± 0.53 | −0.37 | 0 |
| PCz per column | −0.20 ± 0.16 | −0.16 | 0 |
| PCx per row | +0.10 ± 0.57 | +0.05 | 0 |
| PCy per row | −4.47 ± 0.82 | −3.89 | −4.20 |
| PCz per row | +1.19 ± 0.22 | +1.82 | +1.53 |

The map's small patterns give the calibration patterns' plane within the nine's own errors on five of the six slopes, and tighter (`scripts/plane_ni_gain.py` prints this table). On the sixth, PCz per row, the map's is 2.9 of the nine's errors away; an error from nine points left out in turn is itself rough (another fit of the same nine patterns gave +1.16 ± 0.39). Read as steps, the beam goes 1.136 ± 0.047 of the given step for a column and 0.961 ± 0.036 for a row (the nine: 1.18 and 1.04), the scan is turned 1.9° ± 3.3° in its plane, and the tilt is 5.1° ± 3.0° from nominal. The column step is what a rigid pose over the given scan cannot express. Taking every fifth cell instead of every one changes the answer by less than its error, and does not shrink the error: the patterns of a grain err together, so the blocks of the map are what count, not the patterns.

On the superalloy scan (3687 patterns, 1.7 s): steps 1.030 ± 0.016 and 1.062 ± 0.012, the scan turned −1.85° ± 0.62°, tilt −1.1° ± 0.9°, the scan's directions −1.1° ± 0.9° off square; affine and projective agree to the second decimal.

On the 55 × 75 nickel example the plane's source at the map's middle is right (0.0006 from kikuchipy's pattern centre, 0.0017 with CLAHE on) and its slopes are not. With the default filters they come with errors that say so (a tilt of 4.8° ± 11.8°, steps 1.08 ± 0.21 and 1.40 ± 0.35); with CLAHE on, a scan 32° ± 6° off square and turned 22° ± 8°, where the same microscope's larger scan with its calibration patterns says 2° ± 4°. That map is 0.3 % of the detector wide and a few dozen grains; its blocks are not independent, and the errors quoted are too small for it. Indexed under the plane its orientations lie a median 0.373° from kikuchipy's, against 0.362° with the plane dropped (CLAHE on; 0.326° and 0.323° without): on a map this small the plane costs a little.

How many patterns the plane needs is a matter of the map's blocks as well. On a toy scan of 18 × 24 cells (the beam travelling a tenth of the detector, the patterns mostly background and not corrected for it, the detector turned by 0.5°, −1° and 1°), the plane from every second cell (89 patterns) put the turn within 0.45°, inside its own standard error of 0.58°, and the orientations indexed afterwards a median 0.52° from the truth; from every cell (355 patterns), within 0.15° and 0.16°.

**How the master is sampled** (`plane_interp`, `plane_master_auto`, `plane_master_px`). The per-pattern fit samples the master by bicubic (Catmull–Rom) interpolation with that surface's own derivative (`bicubic`, the default): bilinear interpolation has a kink at every master pixel's edge, and its gradient is a finite difference a master pixel wide, not the slope of the surface the score is taken from (`keys6` is Keys' six-point kernel, an order more accurate). The fit also resamples its own copy of the master so that a master pixel is as wide as suits the patterns, each new pixel a tent-weighted average of the old about it (an average over the sphere, since the sheets are equal-area): of 0 (the master as it is), 0.5, 1, 1.5 and 2 detector pixels, the one under which the fit patterns correlate best when the fit starts, or `plane_master_px` by hand.

**The pixel size as given.** Until a plane is fitted, the pixel size turns the scan's steps into the beam's travel in pixels: the file's, one typed in, or, when neither says, the detector's width (`detector_width_mm`, 24: 480 pixels of 50 µm) over the pattern's stored pixels across, so that a 60-pixel pattern is taken as binned eight times, 400 µm a pixel. The plane does not use it. The steps the plane reports are against it, so a pixel size that is off shows there and nowhere else: on a simulated map with pixels of 560 µm given as 400, the plane's steps are 0.714 of what the steps and that pixel size say, and its source and the detector's turn are where they were.

### Several phases

Each phase gives every pattern its own best orientation and correlation. The phase is then chosen by the normalized mutual information between the pattern and each phase's simulation at that orientation (`phase_by: "nmi"`, the default) or by the correlation itself (`"ncc"`). The same choice is made wherever a pattern is scored: in the pattern-centre vote, in the plane fit and in indexing.

### Dictionary indexing

`method: "dictionary"` replaces the spherical correlation by inner products with simulated patterns of a grid of orientations (a cubochoric grid [[5]](#references) restricted to the phase's fundamental zone), as dictionary indexing does [[6]](#references). A dictionary can only be simulated for one source point, the map's centre; the refinement that follows works from each pattern's own, as always.

### Maps: inverse pole figure colours

An inverse pole figure map colours each cell by the crystal direction that lies along a sample axis. `ipf_colors` chooses the key, which the map shows in its corner:

- `classic`: red, green and blue at [001], [101] and [111] of the cubic triangle and mixed between them; for the other classes, the polar angle from the c axis and the azimuth within the class's wedge.
- `hue`: white at the centre of the class's sector, and the hue once round it at an even rate.
- `perfect`: the key of Nolze and Hielscher [[7]](#references) for the Laue class. The hue's rate is balanced so that red, yellow, green, cyan, blue and magenta take even shares of the sector, and the light centre is widened and turned toward gray (`ipf_lightness` and `ipf_gray`, their λ_L and λ_S, a quarter each as they recommend). A class whose sector is not bounded by mirror planes (m-3, -3m, 6/m, 4/m, 2/m), where colouring the one sector alone leaves a jump across an edge, gets their extended key: half its sector with a light centre and half with a dark one, which meet without a jump. For -1 and -3 the paper proves no key can be free of jumps and give every direction its own colour; theirs jumps at the equator.
- `perfect_rotation`: the same construction for the proper rotations of the phase's point group alone (432 for m-3m, 6 for GaN's 6mm), which is how the paper colours an orientation rather than a direction. A direction and its opposite then differ, every class has a light and a dark half, and none has a jump. Orientations that the Laue key colours alike because they turn the one direction into its opposite, as a twin does when the sample axis lies in its twin plane, come apart: on the nickel scan 17 % of the cells fall in the dark half.

The last two are written from the paper's text and equations. I have not compared them with MTEX's output, and the paper leaves a detail open that MTEX may settle differently: its equation 10 is taken here as a blend toward mid-gray, the one reading under which a λ_S of 1 makes the light and the dark centre the same gray, as the paper says it does.

### Master patterns simulated from a crystal

The Simulate step (`Solver.set_crystal`, `simulate_master` in Python) makes the master pattern by the method of ebsdsim [[11]](#references), on this engine: the physics and the CIF reading are Zig, the solve is WGSL, and the front ends only pass the file in and the pattern out.

**The crystal.** The CIF reader (`src/sim/cif.zig`) takes the symmetry, in this order of trust, from the operators the file lists (or its Hall symbol [[12]](#references)); else from the group it names (IT number or Hermann–Mauguin symbol), in whichever tabulated setting leaves its sites in place with the most conventional cell; else, when it declares nothing or P1, from the sites alone, as the largest tabulated group that leaves them in place. It then takes the crystal into the International Tables' standard setting (origin choice 2; R groups on hexagonal axes) by a change of basis of determinant +1, which never turns a crystal into its enantiomorph. The change is looked up among every setting of every group whose cell keeps its system's metric (667 of them, built from the Hall symbols the first time a CIF is read), and searched for otherwise. A file that disagrees with itself is refused, and the app and Python say why. This is ebsdsim's reader in what it does, written again: on its 21 test CIFs and 3 presets, and on 72 variants of them made to take its other paths (without their operators, with the H-M symbol alone, written out in P1), the two agree on all 96, the space group, the change of basis, the cell to 1e-9 and every site, and on the files they refuse.

Two things it does as ebsdsim's does, which a user may not expect. Operators listed on a sheared basis are made standard, but the cell is not reduced: a C2 crystal listed on the basis (a + c, b, c) comes back as a valid C2 cell with β = 156° rather than the conventional 104°, so the master's frame, and the orientations reported against it, are that cell's. And the descent from P1 links groups by the divisibility of their orders, so from the hexagonal family's largest groups (the R groups' 36 operators on hexagonal axes) it never reaches P6/mmm and its kin (24); with the sites at 1e-5 (fractional) it also misses a group whose positions the file gives to five decimals (1/3 as 0.33333).

**The method.** For every direction of the point group's fundamental sector on the Lambert square, the reflections down to `sim_dmin_nm` give the beams: those whose excitation error, over their coupling to the others, is least (Bethe's criterion: `sim_c_strong` is c₁, as many beams for every direction as the one with the most strong ones has, at most 512; none beyond `sim_c_cutoff`, c₃). Those are solved for exactly. The reflections after them within `sim_c_weak` (c₂) are Bethe's weak beams, followed by perturbation: each one's amplitude is taken from the strong beams', ψ_w = −Σ_s A_ws ψ_s / 2πi s_w, which folds them into the strong beams' equations (A_eff = A_ss + A_sw W) and counts them in the intensity (X over strong and weak beams is T X Tᴴ, T = [I; W], formed from the low-rank factors). Counted in the intensity is what makes them pay. Against the full solve (every reflection to d_min, the matrix eigendecomposed and the depth integral in closed form; nickel at d_min 0.05, one 20 kV bin, 32 directions), every Bethe variant is about 0.88–0.90 of it, nearly one scale for all directions; once that scale is taken out, c₁ = 20 solved exactly errs 1.46 % median (r 0.9937), with weak beams to c₂ = 40 in the equations alone (EMsoftOO's Bethe potentials, which it sums over the strong beams only) 2.2 % (r 0.9889), and with them also in the intensity 1.13 % (r 0.9974); the engine at rank 3 with its correction follows each within a few tenths of a percent (GaN, at half-width 8, ranks them the same). The default c₂ is 40 (c₁ 20; ebsdsim's default solve has no weak beams in its intensity). A direction takes as many beams as the device's workgroup memory allows (512 in WebGPU's default 16 KiB, 1024 in 32 KiB as Chrome gives on this machine, 1536 in 48 KiB natively; each bin's kernels compiled for the least that holds it); one with more strong beams than that solves its nearest and follows the rest as weak beams, whatever c₂ is, and the log says so. The potential's coefficients come from the electron scattering factors of Weickenmeier and Kohl [[13]](#references), with their Debye–Waller factors and their absorptive parts. The depth-integrated intensity under an exponential depth profile is `f = tr(S X)`, where X solves the Lyapunov equation `Ã X + X Ãᴴ + e₀ e₀ᴴ = 0` and S holds a site's structure factors. X is taken in a rational Krylov subspace of `sim_rank` columns with one repeated pole (Druskin and Simoncini's RKSM [[14]](#references)), by Galerkin projection, as ebsdsim does. Then, with `sim_correct` (on by default), the projection's error in f is corrected by its dual-weighted residual [[15]](#references): the error is exactly `tr(Y R)`, where `R = Ã X + X Ãᴴ + e₀ e₀ᴴ` is the residual of the projected X (of rank 2·`sim_rank` + 1, so cheap to apply) and Y solves the dual equation `Ãᴴ Y + Y Ã + S = 0`. An approximate Y leaves an error of second order. Here Y is taken diagonal, `Y_ij = −S_ij / (conj d_i + d_j)` (each beam with only its own excitation error, Bethe's picture), except on the 30 beams where the residual is largest against `|d|`, where it is exact: by the same duality, that part is `tr(S_c X̃)` with `Ã_c X̃ + X̃ Ã_cᴴ + R_c = 0` on those beams, solved by a Cayley transform and Smith's doubling [[16]](#references). The correction rides on the pass that adds up the intensities. The backscattered electrons are spread over energy-loss bins of `sim_bin_kev` by a small network trained on Monte Carlo simulations (ebsdsim's surrogate; its weights are `src/sim/surrogate.bin`); each bin is solved at its own voltage, from the least loss up, until one changes the pattern by less than `sim_stop`. The result keeps each site and each bin, as ebsdsim's files do.

On the GPU (`shaders/simulate.wgsl`) a workgroup takes a direction through five passes that share its scratch. It selects the direction's beams by an exact radix select. It then forms the diagonal and the pole, and the preconditioner: the 30 beams nearest the Bragg condition keep their mutual coupling exactly, by an inverse of that block. It builds the Krylov columns by BiCGSTAB with ebsdsim's left Jacobi scaling and that block on the right (a third fewer iterations than Jacobi alone), orthogonalizes them twice over, and forms and solves the small projected equation (squared Smith). It solves the correction's core. And it adds up the corrected intensity over half the beam pairs: the structure factors and the corrected X are Hermitian. The coupling U(g_i − g_j) is read from the potential's table, which stays in cache for a direction. Everything stays within WebGPU's default limits (256 invocations, 16 KiB of workgroup memory, at most 6 storage buffers a kernel); the directions go in tiles of at most 256 MiB of scratch, one submission each; nothing depends on the adapter.

What was measured. Against the exact solution of each direction's Lyapunov equation (Bartels–Stewart in double precision, on the engine's own beams, 64 random directions, the 20 kV bin, error of the summed sites' intensity):

| Median / p90 / largest error | Ti (n 300) | GaN (n 146) | Alexandrite (n 418) |
|---|---|---|---|
| rank 3, corrected (the default) | 0.57 / 2.1 / 5.6 % | 0.35 / 1.3 / 6.9 % | 0.22 / 0.65 / 2.6 % |
| rank 4, corrected | 0.54 / 1.8 / 3.1 % | 0.32 / 0.83 / 6.5 % | 0.16 / 0.48 / 1.4 % |
| rank 8, not corrected (ebsdsim's default) | 1.2 / 2.8 / 6.1 % | 0.99 / 2.4 / 9.0 % | 1.1 / 1.8 / 8.6 % |
| rank 3, not corrected | 3.4 / 10 / 25 % | 4.3 / 8.5 / 27 % | 3.0 / 5.5 / 12 % |

Ti is α-titanium at d_min 0.03 nm, GaN and alexandrite (Cr-doped chrysoberyl, Pnma, 6 sites) at 0.05 nm. One bin at 241 × 241 a hemisphere takes 0.18 s for Ti, 0.10 s for GaN and 0.96 s for alexandrite here at the defaults, against 1.07, 0.30 and 4.87 s for ebsdsim 0.2.4's solve at its defaults (rank 8, no correction), on the same desktop GPU. Without the correction, rank 3 takes 0.16, 0.07 and 0.92 s: the correction costs 5–40 % of the time and takes a factor of 6–14 off the median error.

Whole masters, against ebsdsim 0.2.4 run on the same crystals at its defaults (its times are its own WebGPU path):

| | Correlation with ebsdsim's master | Time here | ebsdsim |
|---|---|---|---|
| Nickel, 201 × 201 a hemisphere | 0.99917 | 0.08 s | 1.8 s |
| GaN, 201 × 201 (no centre of symmetry: both hemispheres) | 0.99793 | 0.73 s | 2.7 s |
| GaN, 501 × 501 | 0.99788 | 3.6 s | 13.9 s |
| Nickel, 501 × 501, against the master bundled with the tests (ebsdsim's, at rank 20) | 0.99950 | 0.37 s | |
| Nickel, 1001 × 1001 | | 1.1 s | |

The correlations stop short of 1 by ebsdsim's own error, not this one's: the same rank-8 solve without the correction agrees with ebsdsim to 0.99992 (GaN, 201 × 201, every bin), and ebsdsim's rank 8 and rank 16 differ by as much (0.99895), the higher rank moving toward the corrected master here. No direction failed to solve.

The nickel scan indexes as well with a simulated master as with ebsdsim's (CLAHE on, against kikuchipy's orientations): median 0.326° and 99.1 % within 3° with either, mean NCC 0.544 with ebsdsim's and 0.545 with this one. Both are for ebsdsim's nickel cell, 3.4751 Å (a computed value, 1.4 % short of the measured 3.524 Å).

Things that differ from ebsdsim on purpose, besides the correction and the preconditioner. ebsdsim orthogonalizes each Krylov column once and drops it when less than 1e-6 of the first column's norm is left; in single precision, as the columns converge on the slowest mode, that lets nearly dependent columns in, and at rank 12 the largest error was 30 %. Here each column is orthogonalized twice and dropped when less than 1e-4 of its own norm is left. ebsdsim keeps a direction's coupling in a packed table in workgroup memory (as much as the adapter offers), sized by an LLL-reduced box; here the coupling is read from the potential's table, within the default limits, which measured no slower. And ebsdsim gives space groups 38 to 41 the centring C where their standard setting has A, which changes which reflections are allowed; here the centring is read from the group's own operators.

### What I measured

On [kikuchipy's nickel scan](../testdata/README.md) [[4]](#references) (55 × 75 patterns of 60 × 60 pixels), against the orientations and the pattern centre kikuchipy refined for it, with an ebsdsim master pattern:

| | |
|---|---|
| Orientations | median 0.32° from kikuchipy's, 99.0 % within 3° |
| Mean NCC | 0.53 (kikuchipy's own refined scores average 0.49) |
| Pattern centre, from a start 0.1 away in each coordinate | vote: within 0.004; the plane's at the map's middle: within 0.0007 (0.0017 with CLAHE on) |
| The same from (0.6, 0.5, 0.7), 0.3 away, the vote searching ±0.3 and ±0.25 on 13 × 13 × 9 candidates | vote: within 0.003 in 4.8 s; the plane's: within 0.0007 |
| Speed (one desktop GPU) | 22 700 patterns/s natively, 15 000 in Chrome; two sweeps (32, then 64): 67 000 natively, 34 000 in Chrome; pattern-centre vote 1.4 s on its 5 × 5 patterns (`fit_grid`; 5 s on 10 × 10, which lands no nearer on this scan); plane fit 0.3 s on its 1055 patterns |
| Two sweeps against one pass | the same median, 98.7 % within 3°; 94 % of the patterns accepted at bandwidth 32 |
| Two candidate phases (nickel, GaN) | 99.9 % of the patterns assigned nickel, by mutual information and by correlation alike |
| Averaged with neighbours (one pass of 7 × 7, λ 0.7) | 50 % of the noise left; mean NCC 0.62; the same orientations (median 0.32° from kikuchipy's) |

On the silicon wafer scan [[8]](#references) (one crystal; 50 × 50 patterns of 480 × 480 over 2 mm × 2 mm, NORDIF's `Pattern.dat`, reduced by 4 as read), with an ebsdsim silicon master pattern. Every pattern has the one orientation, so the angle between each cell's indexed orientation and the map centre's measures the geometry it was indexed under:

| Geometry | Mean NCC | Angle from the centre's orientation: median, 95th percentile | Cells more than 1° off |
|---|---|---|---|
| Every pattern taken from one point (the scan's steps set to nothing), the turns off | 0.377 | 2.78°, 4.69° | 93.7 % |
| The beam's travel, at the 50 µm a pixel the detector's width gives (the file records no pixel size), the turns off | 0.457 | 0.20°, 0.50° | 2.0 % |

Each has the pattern centre fitted from near the right one (0.5185, 0.1564, 0.4894; the file records none, and the vote does not reach it from the middle of the pattern on a single crystal whose map is this wide), on every third cell of the map, by an earlier version of the engine; the plane has not been fitted to this scan. Indexing was at bandwidth 96, which silicon wants: at 64, 5 to 9 % of the patterns take a pseudo-symmetric match 35° or 60° away, whatever the geometry. A single crystal also wants the static background off (the average of its patterns is the pattern) and a narrower dynamic one (`dynamic_sigma` 0.05); the detector's shadow was masked (`mask_radius` 0.85).

On synthetic scans (known geometry and orientations, noise as strong as the signal) the orientations come back to 0.05°, the pattern centre to 0.0002 and the phases without error; cells holding nothing but noise score z below 2 where the patterns score above 80, and no sweep accepts them. [tests/test_api.py](../tests/test_api.py) asserts these.

A few choices came from such measurements rather than from first principles:

- **Master smoothing.** Sampling a sharp master pattern at detector pixel centres aliases. The master is smoothed on the sphere by a Gaussian whose sigma is 0.3 detector pixels (`master_smooth`), about the spread of a square pixel. On the nickel scan the mean NCC is 0.521 without it, 0.545 at 0.2 to 0.3, and 0.413 at 1.0.
- **Filters on the sphere.** Master filters act on directions, not on the stored square images, whose projection is not conformal: smoothing the images directly breaks the crystal's symmetry, and symmetry-equivalent orientations then score up to 5 % apart. CLAHE, which has no spherical form, is followed by an average over the master's own point group, which the engine measures from the master itself.
- **The static background.** The scan's own average pattern does nearly what the file's background image does on the nickel scan (mean NCC 0.531 against 0.542, the same orientations), and every scan has one. Without either, 79 % of the patterns index.
- **The refinement's Jacobian.** Looking the master's gradient up from a field computed once, and leaving out the sums only the plane fit needs, halved the refinement's cost (eight iterations on a 120 × 120 pattern: 35 µs to 18 µs) with the same orientations. Laying the master's value beside its gradient, so that one lookup reads both, and keeping the sums in registers halved it again (10 µs).
- **Where the time went.** The speeds above are three times those of the version before, from the same arithmetic done in a better order, measured kernel by kernel on 16 000 patterns of 120 × 120 pixels. The correlation at bandwidth 64 fell from 1460 ms to 530 ms: it uses both symmetries of its series (above) and so transforms half as much, keeps a column of the spectrum in a workgroup's memory from the sums through its first transform instead of storing and reading it back, and holds its tables with the index that neighbouring invocations differ in innermost, so that a step of the sums reads a few neighbouring words where it read thirty-two scattered ones (that last change alone took the sums from 855 ms to 245 ms, and the same change took the Legendre sums from 69 ms to 9 ms). The dynamic background fell from 112 ms to 17 ms, the refinement from 341 ms to 159 ms. And the host reads (and, for a reduced level, reduces) the next batch while the device works on the present one, straight into the buffer that batch will be uploaded from, which took the time outside the GPU from 200 ms to 60 ms; the reducer's loops were given strides the compiler knows, and run in a third of the time. Nothing here is tuned to one GPU: workgroups are 64 or 256 invocations and no size depends on the adapter.
- **The refinement's curvature.** Plain Gauss–Newton steps converged slowly on real patterns: on patterns of 120 × 120 pixels eight iterations left a median 0.095° from the converged orientation and twenty-four were needed. Rescaling the matrix from the gains makes eight iterations converge (median 0.000°), and from a bandwidth-32 start they leave 0.017° where they left 0.59°. Sweeps depend on this.
- **The acceptance threshold.** At `accept_z` 15 on the nickel scan, bandwidth 32 accepts 94.9 % of the patterns and 0.6 % of those differ from the bandwidth-64 result by more than 3°; at 10 it is 95.8 % and 1.0 %, at 20, 93.9 % and 0.4 %. z grows with the number of pixels compared (a correct 30 × 30 match scores about 17, a 120 × 120 one about 115), so judge it from the app's z histogram on your own scan.
- **Expanding an ebsdsim master.** Its file holds the fundamental sector only; every pixel of both hemispheres takes the value of its symmetry-equivalent direction deepest inside the sector, which means trying each of the point group's operators. Where an operator of the crystal is also one of the Lambert grid's own (x ↦ −x, y ↦ −y, x ↔ y, and z ↦ −z across the two sheets, sixteen in all, every one of them in m-3m), the pixels it takes to others are copied, not looked up. Nickel at 1001 × 1001 a hemisphere loads in 85 ms natively where it took 515, and in 0.21 s in Chrome where it took 0.89; the sheets are the same.
- **Mutual information for the phase.** It is the default, but on the two-phase cases I could test (nickel against GaN, real and synthetic) the correlation chose exactly as well. It needs a harder pair of phases to show what it adds.

What sweeps do not protect against is a match that is good and wrong. In a cubic crystal the twin of the right orientation (60° about ⟨111⟩, which shares most bands) can be what a low bandwidth finds; it settles, and it scores far above chance. Raising `accept_z` to where the z histogram's main peak begins removes most such matches.

Reducing patterns saves little time for small patterns: the correlation over SO(3) costs the same whatever the pattern size, and refining at the stored size is where the pixels are counted. It saves memory and upload, and it matters more the larger the patterns are.

### Not verified

Every format listed above is checked pixel for pixel against kikuchipy (`scripts/check_formats.py`), Bruker's on one file. Compressed `.ebsp` files and HDF5 files written with `libver="latest"` that use dense groups or the newer chunk indexes are refused with a message that says so. The direction the beam moves on the sample as the map index grows is a convention of the acquisition; the default (`scan_x_sign`, `scan_y_sign` = −1) is kikuchipy's, and it only matters for maps wide enough for the beam's travel to show on the detector. A dictionary coarse enough to be cheap puts few patterns within the refinement's reach (a grid of 16 indexes 44 % of the nickel scan, 30 indexes 96 %), and its size grows as the crystal's symmetry falls, which the spherical correlation's cost does not; I measured dictionaries on cubic crystals only. Every real scan here is cubic (nickel, silicon): a hexagonal phase (GaN) has been indexed only in simulated scans, where its orientations come back as nickel's do. The two inverse pole figure keys after Nolze and Hielscher were not compared with MTEX. The simulated master patterns were compared with ebsdsim's only, not with EMsoft's or with experiment beyond the nickel scan; their energy spread comes from ebsdsim's trained surrogate, not from a Monte Carlo run here. Everything ran on one desktop GPU, natively and in Chrome.

## Python

The same engine as plain functions:

```python
import zebsdsolve as ze

res = ze.index("scan.h5", ["Ni.npz"], fit="geometry") # fit the geometry, then index
res.quaternions, res.ncc, res.phase                   # per map cell
res.geometry["pc"], res.fit["geometry"]               # the vote's pattern centre, the plane and what it says

# a large scan: search on binned patterns in two sweeps, refine at the stored size
res = ze.index("scan.h5", ["Ni.npz"], binning=2, full_resolution="refine", sweeps=2)
res.z, res.sweep                                      # how far above chance, and which sweep accepted

# a very large scan: time a sample first (seconds) and keep the batch and workgroup sizes that run fastest here
s = ze.Solver(); s.open("scan.h5"); s.add_phase("Ni.npz"); s.tune(); s.index()

# a noisy scan: average each pattern with the neighbours that resemble it first (NLPAR)
res = ze.index("scan.h5", ["Ni.npz"], nlpar=True, nlpar_lambda=0.9)

fit = ze.fit_pattern_center("scan.h5", ["Ni.npz"], pc=(0.5, 0.3, 0.5), pc_span=0.2)
# no scan at hand: a toy one from the masters, with its answer (a seed is a scan)
s = ze.Solver(); s.add_phase("Ni.npz"); truth = s.synthesize(toy_seed=7, toy_grains=30, toy_counts=20); res = s.index()

# no master pattern for a phase: simulate one from its CIF
s = ze.Solver(); s.set_crystal("GaN.cif")                 # (or a dict already in the standard setting)
s.configure(sim_voltage_kv=20, sim_halfw=250); s.simulate_master()
s.add_simulated_phase("GaN"); s.save_simulation("GaN.npz")
```

### Toy scans as experiments

`Solver.synthesize` (the app's **Make toy scan**) is meant for comparing ways of fitting the geometry, so what a fit could depend on is a setting, and everything it should recover comes back as the answer:

| | Settings |
|---|---|
| The true geometry | Whatever the geometry is when the scan is made: pattern centre, tilts, detector rotation, pixel size, steps (and so the beam's travel). Set it, synthesize, then move it and let a fit bring it back. |
| The microstructure | `toy_seed`; `toy_rows`, `toy_cols`; `toy_grains` (1: a single crystal; one per cell: no two patterns alike); `toy_spread_deg`, a random turn of every cell within its grain; `toy_blank`, the share of cells with no crystal |
| The detector | `toy_height`, `toy_width`; `toy_contrast`, the share of the intensity that is the pattern and not background; `toy_background_sigma`, the background's hump |
| The counting | `toy_counts`, mean electrons a pixel (Poisson; 0: no noise); `toy_noise_seed`, to count one microstructure again with other noise |

```python
s.set_geometry(pc=(0.45, 0.25, 0.52), rot=np.radians((0.5, -1, 1)), px_um=560, step_x_um=150, step_y_um=150)
truth = s.synthesize(toy_seed=11, toy_grains=400, toy_blank=0.1, toy_contrast=0.3, toy_background_sigma=0.4, toy_counts=200)
s.set_geometry(pc=(0.46, 0.24, 0.53), rot=(0, 0, 0))     # a wrong start
s.configure(plane_frame="follow"); fit = s.fit_plane()   # … against truth["geometry"]
res = s.index()                                           # … against truth["quaternions"], truth["phase"] (−1: blank), truth["grain"]
```

In the app the fits' readouts and the indexing's say how far from the scan's own answer they land.

`scan` is a path or an array of patterns shaped (rows, cols, h, w); every option of the engine ([src/settings.zig](../src/settings.zig)) is a keyword. `ze.Solver` is the engine itself, for driving the steps one at a time: open, add phases, look at patterns and their simulations, fit, index a batch at a time with progress, draw maps, export.

Quaternions are (w, x, y, z) and map sample directions to crystal directions. Indexing and calibration retain the fitted pose, normalized with a non-negative real part, without crystal symmetry reduction. Result Euler angles, simulations and exports use that same pose. `bunge` converts a pose to Bunge angles without reducing it; `reduce` explicitly selects its equivalent of smallest rotation under the proper rotations of the phase's point group. Those are the Laue class's rotations only for a crystal with a centre of symmetry: GaN (6mm) has the six about its c axis, not the twelve of 6/mmm, whose other six would turn the polar axis over, which is another orientation and another pattern. The dictionary's orientations and `disorientation` use the same rotations. Explicit reduction requires the symmetry operators to be expressed in the master's crystal frame; retaining a fit does not assume that a point-group label establishes those axes.

In the same conventions, the sample frame has x down the tilted surface toward the detector and z along the surface normal (kikuchipy's and EDAX's); the pattern centre is in the Bruker convention.

### Tutorials

Two scripts in [examples/](../examples/) run as notebooks ([scripts/run_tutorials.py](../scripts/run_tutorials.py)) and end with checks, so they fail like tests:

| Notebook | What it shows |
|---|---|
| [01_index](../examples/01_index.py) | The nickel scan from file to orientation map: looking at patterns and filters, the pattern-centre vote from a wrong start and its vote surface, the plane of source points, indexing, and the result against kikuchipy's; then the same in two sweeps, with the search on binned patterns, and with the patterns averaged with their neighbours |
| [02_synthetic](../examples/02_synthetic.py) | A simulated scan with a known answer: a map wide enough for the beam's travel to show on the detector, two phases (nickel and GaN) and noise; the fits and the indexing against the truth, and what accounting for that travel is worth |

## Building and contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the file map, the build and test commands and the conventions.

## License

MIT ([LICENSE](../LICENSE)). The wheels bundle [wgpu-native](https://github.com/gfx-rs/wgpu-native) (MIT / Apache-2.0). The example scan is kikuchipy's `nickel_ebsd_large`, CC BY 4.0 ([sources](../testdata/README.md)).

## References

1. B. Gutman, Y. Wang, T. Chan, P. M. Thompson, A. W. Toga. Shape registration with spherical cross correlation. *2nd MICCAI Workshop on Mathematical Foundations of Computational Anatomy* (2008) 56–67.
2. W. C. Lenthe, S. Singh, M. De Graef. A spherical harmonic transform approach to the indexing of electron back-scattered diffraction patterns. *Ultramicroscopy* 207 (2019) 112841. [doi:10.1016/j.ultramic.2019.112841](https://doi.org/10.1016/j.ultramic.2019.112841)
3. P. T. Brewick, S. I. Wright, D. J. Rowenhorst. NLPAR: non-local smoothing for enhanced EBSD pattern indexing. *Ultramicroscopy* 200 (2019) 50–61. [doi:10.1016/j.ultramic.2019.02.013](https://doi.org/10.1016/j.ultramic.2019.02.013)
4. H. W. Ånes, L. A. H. Lervik, O. Natlandsmyr, T. Bergh, Z. Xu, E. Prestat, M. Nord. kikuchipy. [doi:10.5281/zenodo.3597646](https://doi.org/10.5281/zenodo.3597646)
5. D. Roşca, A. Morawiec, M. De Graef. A new method of constructing a grid in the space of 3D rotations and its applications to texture analysis. *Modelling and Simulation in Materials Science and Engineering* 22 (2014) 075013. [doi:10.1088/0965-0393/22/7/075013](https://doi.org/10.1088/0965-0393/22/7/075013)
6. Y. H. Chen, S. U. Park, D. Wei, G. Newstadt, M. A. Jackson, J. P. Simmons, M. De Graef, A. O. Hero. A dictionary approach to electron backscatter diffraction indexing. *Microscopy and Microanalysis* 21 (2015) 739–752. [doi:10.1017/S1431927615000756](https://doi.org/10.1017/S1431927615000756)
7. G. Nolze, R. Hielscher. Orientations – perfectly colored. *Journal of Applied Crystallography* 49 (2016) 1786–1802. [doi:10.1107/S1600576716012942](https://doi.org/10.1107/S1600576716012942)
8. H. W. Ånes, J. Hjelen, A. T. J. van Helvoort, K. Marthinsen. Electron backscatter diffraction patterns from a single crystal silicon wafer. Zenodo (2022). [doi:10.5281/zenodo.7491388](https://doi.org/10.5281/zenodo.7491388)
9. M. Sugiyama. Machine learning with squared-loss mutual information. *Entropy* 15(1) (2013) 80–112. [doi:10.3390/e15010080](https://doi.org/10.3390/e15010080)
10. A. Winkelmann, G. Nolze, G. Cios, T. Tokarski, P. Bała. Refined calibration model for improving the orientation precision of electron backscatter diffraction maps. *Materials* 13 (2020) 2816. [doi:10.3390/ma13122816](https://doi.org/10.3390/ma13122816)
11. Z. Varley. ebsdsim: EBSD master pattern simulation. [github.com/ZacharyVarley/ebsdsim](https://github.com/ZacharyVarley/ebsdsim)
12. S. R. Hall. Space-group notation with an explicit origin. *Acta Crystallographica* A37 (1981) 517–525. [doi:10.1107/S0567739481001228](https://doi.org/10.1107/S0567739481001228)
13. A. Weickenmeier, H. Kohl. Computation of absorptive form factors for high-energy electron diffraction. *Acta Crystallographica* A47 (1991) 590–597. [doi:10.1107/S0108767391004804](https://doi.org/10.1107/S0108767391004804)
14. V. Druskin, V. Simoncini. Adaptive rational Krylov subspaces for large-scale dynamical systems. *Systems & Control Letters* 60 (2011) 546–560. [doi:10.1016/j.sysconle.2011.04.013](https://doi.org/10.1016/j.sysconle.2011.04.013)
15. R. Becker, R. Rannacher. An optimal control approach to a posteriori error estimation in finite element methods. *Acta Numerica* 10 (2001) 1–102. [doi:10.1017/S0962492901000010](https://doi.org/10.1017/S0962492901000010)
16. R. A. Smith. Matrix equation XA + BX = C. *SIAM Journal on Applied Mathematics* 16 (1968) 198–201. [doi:10.1137/0116017](https://doi.org/10.1137/0116017)
