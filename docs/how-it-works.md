# zebsdsolve: how it works, and what was measured

The long form of the [README](../README.md): the app pane by pane, each method with the numbers behind its defaults, what is not verified, and the references.

EBSD pattern-centre fitting and indexing on the GPU, in the browser or from Python. Point it at a scan of Kikuchi patterns, of any size, and at a master pattern for each phase that might be in it; look at the patterns, fit the detector geometry, and index the map.

One engine does the work for both front ends: the math is WGSL, the schedule is Zig, and it runs natively (Vulkan, Metal or Direct3D 12 through wgpu-native) and as WebAssembly on WebGPU.

## Run the app

```bash
pip install zebsdsolve
python -m zebsdsolve
```

This serves the app on <http://127.0.0.1:4190/> and opens it in your browser. Everything runs on your GPU inside the page. A scan is read in place, in pieces, by the page itself, so a 100 GB file is neither uploaded anywhere nor loaded into memory. You need Chrome or Edge 137+ (WebGPU and JavaScript Promise Integration). `--port` picks another port, and `--no-browser` only prints the URL.

To install from a clone instead, run `pip install .` in it. The first build downloads Zig and wgpu-native into `.deps/` and compiles for your machine, which takes a minute or two.

The steps are tabs, each with its own settings:

| Step | What it does |
|---|---|
| **Data** | Open a pattern file (or drop it on the window) and add one master pattern per candidate phase. The map appears as each pattern's contrast, coarse at once and sharpening, so the microstructure shows before anything is indexed. A project `.json` saved under Index, dropped with the scan or after it, brings its settings and geometry back. With no scan at hand, **Make toy scan** makes one from the master patterns: a map cut into grains with random orientations and phases, its patterns simulated under the geometry set, laid on a background and counted as electrons with their shot noise, all from a seed, and its answer kept to check the indexing against. |
| **Patterns** | How every pattern is prepared, in order: the size it is reduced to as it is read (and what still uses the stored size), averaging with its neighbours (NLPAR), then the filters: static background (the scan's average pattern), dynamic background, CLAHE, band pass, circular mask. CLAHE and the band pass act on the master patterns as well. |
| **Geometry** | The pattern centre, tilts, detector rotation and pixel size, from the file where it records them. **Fit pattern centre** finds the centre by voting; **Fit pose** refines the detector pose and the fit patterns' orientations together. Either can be undone, and the geometry put back as the scan was opened. |
| **Index** | Spherical indexing (or a dictionary) of the map or a region of it, each orientation refined on the detector, a batch at a time, in one pass or in sweeps of rising bandwidth; the map fills in as it goes. **Tune speed** fits the batch and workgroup sizes to the machine first. Export as `.ang`, `.ctf`, and the geometry and settings as JSON. |

Beside the steps are three panes, with bars between them to drag:

- **The map** zooms with the scroll wheel about the pointer (never out past the whole map), a drag moves a zoomed map, the arrow keys step the selected pattern, and shift-drag sets the region of interest. Its corner says what its colours mean: the inverse pole figure key of the phase's Laue class in the colours chosen under Index, a scale, or the phases.
- **The pattern** selected on the map, at any stage of its preparation (as stored, averaged with its neighbours, filtered), beside its simulation and their difference. A pattern nothing has indexed yet is indexed on the spot, as the fits index theirs, so the simulation and the scores are there for any cell and follow the filters and the geometry as you change them. With the averaging on, a small map in the pattern's corner shows how much of each neighbour went into it.
- **The dock**: the pattern-centre search, the master pattern as patterns are compared with it, the histogram of z, and the log. The search is a volume of votes over (x, y, z) for each grid it scored. It shows as three slices through a cross-hair, or as every slice along one axis side by side; a slider under each slice (or the scroll wheel over it) moves through the volume, and the vote's profile along each axis is drawn beside. The slices cut through the pattern centre in use, and cut through it again whenever it changes, by a fit or by a typed value, in the finest grid that holds it.

Scan formats: h5ebsd as written by kikuchipy, Bruker, EDAX and Oxford (`.h5oina`); EMsoft's simulated pattern files; EDAX `.up1` / `.up2`; Oxford `.ebsp` (uncompressed); NORDIF `Pattern.dat` with its `Setting.txt` (open the two together; Python finds the second beside the first). Master patterns: [ebsdsim](https://github.com/ZacharyVarley/ebsdsim) `.npz` and EMsoft / EMsoftOO `.h5`. The HDF5 files are read by a reader written for this project (no HDF5 library), in Zig, which is what lets the browser read them in ranges.

## How it works

### Geometry: one detector, and where the beam stood

The geometry is one rigid pose of the detector in the sample's frame and, for every pattern, the point of the sample the beam stood on when it was recorded. The pattern centre (Bruker convention) is how the pose's position is written: the foot of the perpendicular onto the detector from the source point at the map's centre, with the detector distance. A pattern from anywhere else on the map sees that same detector from a source point displaced by the beam's travel, which the scan's step sizes give in micrometres and the detector's pixel size turns into pixels. So there is one pose to fit, with six degrees of freedom (seven with the pixel size), and no pattern has a pattern centre of its own. A file that records no pixel size leaves the beam's travel unknown, and every pattern is then taken to come from the one point.

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

A wrong pattern centre distorts a pattern on the sphere so that no rotation of the master fits it. So: take a few patterns on a coarse lattice over the map (5 × 5 by default) and index them by spherical correlation under each candidate pattern centre of a grid. Each pattern's best correlation over all orientations is then a surface over the candidates, high near the truth. Each candidate gets its own window normalization.

What matters is how the patterns' surfaces become one choice (`pc_vote`). Averaging them and taking the highest candidate (`mean`) is the obvious rule and a poor one: every surface also has broad ridges that are not the truth (a shift of the pattern centre traded against a turn of the crystal), the ridges of many patterns overlap, and their sum pulls the maximum along them. The default, `ballots`, asks where the patterns agree instead. Each pattern casts a ballot at each of its `pc_top_k` best candidates (12; only those scoring at least `pc_gate` × its own best), the ballots are spread by a Gaussian of a fixed width on the detector (`pc_kde_sigma`, 0.015 of the pattern), and the densest place wins. It is then moved once to the weighted centre of the ballots within `pc_inlier_radius` of it and found again among those alone. A pattern that is wrong is outvoted, not averaged in. This is the consensus I arrived at with sliding correlations in the camera plane (gated top-K ballots and a density mode); it carries over to the spherical correlation's volumes unchanged.

The search repeats around the choice on a narrower grid; if the choice is on the border of its grid, the grid moves onto it before it narrows. Measured on the same three grids (9 × 9 × 5 candidates, 2.1 s for the mean and 1.2 s for the ballots, which do not chase a border), as the largest error of the three coordinates:

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

### Pose: the Jacobian of everything at once

Starting from the vote, the detector's pose and the orientations of a lattice of patterns are refined together, in the tangent space: the mean normalized cross-correlation is maximized over a left perturbation of the pose in se(3) and an update Exp(φ) of every fit pattern's orientation, 6 + 3N unknowns, by Levenberg–Marquardt on the Jacobian of the normalized simulations. Every iteration moves the pose and all the orientations in one step, and is kept only if the correlation rose. The GPU reduces each pattern to 43 sums; the normal equations have an arrowhead shape and are solved on the host by a Schur complement, one readback per iteration. The lattice is `pose_grid` a side (15: a pattern costs this fit little, and the fit of 225 takes 0.1 to 0.2 s), and the patterns on it whose starting match indexing itself would not accept are left out, since a pattern matched to the wrong orientation pulls the pose toward whatever suits its mistake.

The fit moves all six of the pose's degrees of freedom, and each can be held, named in the detector's own axes: `pose_x`, `pose_y`, `pose_z` are the pattern centre's coordinates, and `pose_tilt`, `pose_swing`, `pose_twist` turn the detector about the source point around its column direction, its row direction and its normal (which leaves the pattern centre where it is). `pose_scale` adds a seventh and makes the pose a similarity (Sim(3)): the scale between the beam's travel and the detector, which is the detector's pixel size, and which the fit then writes back to the geometry.

Turning the detector about the source and turning every crystal the other way change the patterns alike. What tells the two apart is the beam's travel across the map, so the rotation is as good as the map is wide against the detector:

| Simulated map, 225 fit patterns | Beam travel, of the detector | Swing, tilt, twist found (°) | Orientations indexed afterwards, median error |
|---|---|---|---|
| Detector turned by 1, −2, 1.5; no noise | 0.14 | 1.00, −1.99, 1.50 | 0.01° |
| The same under noise as strong as the patterns (as are all below) | 0.14 | 1.01, −2.06, 1.29 | 0.22° |
| The same, the rotation held at zero | 0.14 | | 2.68° |
| A wider map | 0.34 | 1.01, −2.03, 1.57 | 0.09° |
| A narrow map | 0.04 | 0.49, −1.35, 0.75 | 1.12° (held: 2.69°) |
| A narrow map, the detector at its nominal attitude | 0.04 | 0.11, −0.03, 0.34 | 0.37° (held: 0.05°) |
| Pixel size given as 400 µm for 560, and held | 0.14 | 1.8, −15.2, 0.9 | 13.3° |
| The same, the pixel size freed (found: 560.3 µm) | 0.14 | 1.02, −2.03, 1.30 | 0.20° |

Every orientation indexed afterwards carries the detector's fitted rotation, so the rotation's error is theirs. Fitting it trades one error for another: the error of assuming the nominal attitude, which is the detector's real misalignment whatever it is, for an error of estimation that falls with the beam's travel and with the number of fit patterns (on the narrow map above, 25 patterns leave 1.3° and 225 about 0.3°). Where the beam travels a few percent of the detector and the detector is known to sit square, holding the three turns is the better choice, and that is what their switches are for. A wrong pixel size shows up as a false tilt of many degrees, which freeing the scale removes.

On real scans:

| Scan | Beam travel, of the detector | Found |
|---|---|---|
| Silicon wafer [[8]](#references): one crystal, 50 × 50 patterns over 2 mm; no pixel size in the file | 0.09 | Pixel size 48.0 to 48.2 µm from starts of 40, 80 and 160 µm; swing −1.0°, tilt 0.5°, twist 4.1° from each (within 0.15°) |
| Nickel: 75 × 55 cells of 1.5 µm; no pixel size in the file | none | The rotation is not fitted |

The wafer is the test of it, because one crystal has one orientation: how far the indexed orientations spread across the map measures the geometry they were indexed under ([below](#what-i-measured)). With the scan directions set the wrong way round, the fit turns the detector far round to make up for it, or, with the scale free, lets the pixel size run away; the log says what that means. With no pixel size at all, a turn of the detector and the opposite turn of every crystal are exactly the same thing, and the rotation is left as set.

### Several phases

Each phase gives every pattern its own best orientation and correlation. The phase is then chosen by the normalized mutual information between the pattern and each phase's simulation at that orientation (`phase_by: "nmi"`, the default) or by the correlation itself (`"ncc"`). The same choice is made wherever a pattern is scored: in the pattern-centre vote, in the pose fit and in indexing.

### Dictionary indexing

`method: "dictionary"` replaces the spherical correlation by inner products with simulated patterns of a grid of orientations (a cubochoric grid [[5]](#references) restricted to the phase's fundamental zone), as dictionary indexing does [[6]](#references). A dictionary can only be simulated for one source point, the map's centre; the refinement that follows works from each pattern's own, as always.

### Maps: inverse pole figure colours

An inverse pole figure map colours each cell by the crystal direction that lies along a sample axis. `ipf_colors` chooses the key, which the map shows in its corner:

- `classic`: red, green and blue at [001], [101] and [111] of the cubic triangle and mixed between them; for the other classes, the polar angle from the c axis and the azimuth within the class's wedge.
- `hue`: white at the centre of the class's sector, and the hue once round it at an even rate.
- `perfect`: the key of Nolze and Hielscher [[7]](#references) for the Laue class. The hue's rate is balanced so that red, yellow, green, cyan, blue and magenta take even shares of the sector, and the light centre is widened and turned toward gray (`ipf_lightness` and `ipf_gray`, their λ_L and λ_S, a quarter each as they recommend). A class whose sector is not bounded by mirror planes (m-3, -3m, 6/m, 4/m, 2/m), where colouring the one sector alone leaves a jump across an edge, gets their extended key: half its sector with a light centre and half with a dark one, which meet without a jump. For -1 and -3 the paper proves no key can be free of jumps and give every direction its own colour; theirs jumps at the equator.
- `perfect_rotation`: the same construction for the proper rotations of the phase's point group alone (432 for m-3m, 6 for GaN's 6mm), which is how the paper colours an orientation rather than a direction. A direction and its opposite then differ, every class has a light and a dark half, and none has a jump. Orientations that the Laue key colours alike because they turn the one direction into its opposite, as a twin does when the sample axis lies in its twin plane, come apart: on the nickel scan 17 % of the cells fall in the dark half.

The last two are written from the paper's text and equations. I have not compared them with MTEX's output, and the paper leaves a detail open that MTEX may settle differently: its equation 10 is taken here as a blend toward mid-gray, the one reading under which a λ_S of 1 makes the light and the dark centre the same gray, as the paper says it does.

### What I measured

On [kikuchipy's nickel scan](../testdata/README.md) [[4]](#references) (55 × 75 patterns of 60 × 60 pixels), against the orientations and the pattern centre kikuchipy refined for it, with an ebsdsim master pattern:

| | |
|---|---|
| Orientations | median 0.32° from kikuchipy's, 99.0 % within 3° |
| Mean NCC | 0.53 (kikuchipy's own refined scores average 0.49) |
| Pattern centre, from a start 0.1 away in each coordinate | vote: within 0.003; pose fit: within 0.0008 |
| The same from (0.6, 0.5, 0.7), 0.3 away, the vote searching ±0.3 and ±0.25 on 13 × 13 × 9 candidates | vote: within 0.003 in 4.5 s; pose fit: within 0.0008 |
| Speed (one desktop GPU) | 22 700 patterns/s natively, 15 000 in Chrome; two sweeps (32, then 64): 67 000 natively, 34 000 in Chrome; pattern-centre vote 1.3 s; pose fit 0.13 s |
| Two sweeps against one pass | the same median, 98.7 % within 3°; 94 % of the patterns accepted at bandwidth 32 |
| Two candidate phases (nickel, GaN) | 99.9 % of the patterns assigned nickel, by mutual information and by correlation alike |
| Averaged with neighbours (one pass of 7 × 7, λ 0.7) | 50 % of the noise left; mean NCC 0.62; the same orientations (median 0.32° from kikuchipy's) |

On the silicon wafer scan [[8]](#references) (one crystal; 50 × 50 patterns of 480 × 480 over 2 mm × 2 mm, NORDIF's `Pattern.dat`, reduced by 4 as read), with an ebsdsim silicon master pattern. Every pattern has the one orientation, so the angle between each cell's indexed orientation and the map centre's measures the geometry it was indexed under:

| Geometry | Mean NCC | Angle from the centre's orientation: median, 95th percentile | Cells more than 1° off |
|---|---|---|---|
| The pattern centre alone: every pattern taken from one point (the file records no pixel size) | 0.378 | 2.71°, 4.72° | 92.8 % |
| The beam's travel, the pixel size fitted (48.0 µm) with the position | 0.458 | 0.18°, 0.43° | 2.1 % |
| The whole pose and the pixel size, Sim(3) (48.1 µm; swing −1.0°, tilt 0.5°, twist 4.1°) | 0.458 | 0.11°, 0.27° | 1.5 % |

The pattern centre came from the vote and the pose fit on a 4 × 4 block at the map's centre, where the beam hardly travels; the pose fits then used 15 × 15 patterns of the whole map. Indexing was at bandwidth 96, which silicon wants: at 64, 5 to 9 % of the patterns take a pseudo-symmetric match 35° or 60° away, whatever the geometry. A single crystal also wants the static background off (the average of its patterns is the pattern) and a narrower dynamic one (`dynamic_sigma` 0.05); the detector's shadow was masked (`mask_radius` 0.85).

On synthetic scans (known geometry and orientations, noise as strong as the signal) the orientations come back to 0.05°, the pattern centre to 0.0002 and the phases without error; cells holding nothing but noise score z below 2 where the patterns score above 80, and no sweep accepts them. [tests/test_api.py](../tests/test_api.py) asserts these.

A few choices came from such measurements rather than from first principles:

- **Master smoothing.** Sampling a sharp master pattern at detector pixel centres aliases. The master is smoothed on the sphere by a Gaussian whose sigma is 0.3 detector pixels (`master_smooth`), about the spread of a square pixel. On the nickel scan the mean NCC is 0.521 without it, 0.545 at 0.2 to 0.3, and 0.413 at 1.0.
- **Filters on the sphere.** Master filters act on directions, not on the stored square images, whose projection is not conformal: smoothing the images directly breaks the crystal's symmetry, and symmetry-equivalent orientations then score up to 5 % apart. CLAHE, which has no spherical form, is followed by an average over the master's own point group, which the engine measures from the master itself.
- **The static background.** The scan's own average pattern does nearly what the file's background image does on the nickel scan (mean NCC 0.531 against 0.542, the same orientations), and every scan has one. Without either, 79 % of the patterns index.
- **The refinement's Jacobian.** Looking the master's gradient up from a field computed once, and leaving out the sums only the pose fit needs, halved the refinement's cost (eight iterations on a 120 × 120 pattern: 35 µs to 18 µs) with the same orientations. Laying the master's value beside its gradient, so that one lookup reads both, and keeping the sums in registers halved it again (10 µs).
- **Where the time went.** The speeds above are three times those of the version before, from the same arithmetic done in a better order, measured kernel by kernel on 16 000 patterns of 120 × 120 pixels. The correlation at bandwidth 64 fell from 1460 ms to 530 ms: it uses both symmetries of its series (above) and so transforms half as much, keeps a column of the spectrum in a workgroup's memory from the sums through its first transform instead of storing and reading it back, and holds its tables with the index that neighbouring invocations differ in innermost, so that a step of the sums reads a few neighbouring words where it read thirty-two scattered ones (that last change alone took the sums from 855 ms to 245 ms, and the same change took the Legendre sums from 69 ms to 9 ms). The dynamic background fell from 112 ms to 17 ms, the refinement from 341 ms to 159 ms. And the host reads (and, for a reduced level, reduces) the next batch while the device works on the present one, straight into the buffer that batch will be uploaded from, which took the time outside the GPU from 200 ms to 60 ms; the reducer's loops were given strides the compiler knows, and run in a third of the time. Nothing here is tuned to one GPU: workgroups are 64 or 256 invocations and no size depends on the adapter.
- **The refinement's curvature.** Plain Gauss–Newton steps converged slowly on real patterns: on patterns of 120 × 120 pixels eight iterations left a median 0.095° from the converged orientation and twenty-four were needed. Rescaling the matrix from the gains makes eight iterations converge (median 0.000°), and from a bandwidth-32 start they leave 0.017° where they left 0.59°. Sweeps depend on this.
- **The acceptance threshold.** At `accept_z` 15 on the nickel scan, bandwidth 32 accepts 94.9 % of the patterns and 0.6 % of those differ from the bandwidth-64 result by more than 3°; at 10 it is 95.8 % and 1.0 %, at 20, 93.9 % and 0.4 %. z grows with the number of pixels compared (a correct 30 × 30 match scores about 17, a 120 × 120 one about 115), so judge it from the app's z histogram on your own scan.
- **Mutual information for the phase.** It is the default, but on the two-phase cases I could test (nickel against GaN, real and synthetic) the correlation chose exactly as well. It needs a harder pair of phases to show what it adds.

What sweeps do not protect against is a match that is good and wrong. In a cubic crystal the twin of the right orientation (60° about ⟨111⟩, which shares most bands) can be what a low bandwidth finds; it settles, and it scores far above chance. Raising `accept_z` to where the z histogram's main peak begins removes most such matches.

Reducing patterns saves little time for small patterns: the correlation over SO(3) costs the same whatever the pattern size, and refining at the stored size is where the pixels are counted. It saves memory and upload, and it matters more the larger the patterns are.

### Not verified

Every format listed above is checked pixel for pixel against kikuchipy (`scripts/check_formats.py`), Bruker's on one file. Compressed `.ebsp` files and HDF5 files written with `libver="latest"` that use dense groups or the newer chunk indexes are refused with a message that says so. The direction the beam moves on the sample as the map index grows is a convention of the acquisition; the default (`scan_x_sign`, `scan_y_sign` = −1) is kikuchipy's, and it only matters for maps wide enough for the beam's travel to show on the detector. A dictionary coarse enough to be cheap puts few patterns within the refinement's reach (a grid of 16 indexes 44 % of the nickel scan, 30 indexes 96 %), and its size grows as the crystal's symmetry falls, which the spherical correlation's cost does not; I measured dictionaries on cubic crystals only. Every real scan here is cubic (nickel, silicon): a hexagonal phase (GaN) has been indexed only in simulated scans, where its orientations come back as nickel's do. The two inverse pole figure keys after Nolze and Hielscher were not compared with MTEX. Everything ran on one desktop GPU, natively and in Chrome.

## Python

The same engine as plain functions:

```python
import zebsdsolve as ze

res = ze.index("scan.h5", ["Ni.npz"], fit="pose")     # vote, pose fit, then index
res.quaternions, res.ncc, res.phase                   # per map cell
res.geometry["pc"], res.fit["pose"]                   # what the fits found

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
fit = s.fit_pose()                                        # … against truth["geometry"]
res = s.index()                                           # … against truth["quaternions"], truth["phase"] (−1: blank), truth["grain"]
```

In the app the fits' readouts and the indexing's say how far from the scan's own answer they land.

`scan` is a path or an array of patterns shaped (rows, cols, h, w); every option of the engine ([src/settings.zig](../src/settings.zig)) is a keyword. `ze.Solver` is the engine itself, for driving the steps one at a time: open, add phases, look at patterns and their simulations, fit, index a batch at a time with progress, draw maps, export.

Quaternions are (w, x, y, z) and map sample directions to crystal directions, and each is stored as its equivalent of smallest rotation under the proper rotations of the phase's point group. Those are the Laue class's rotations only for a crystal with a centre of symmetry: GaN (6mm) has the six about its c axis, not the twelve of 6/mmm, whose other six would turn the polar axis over, which is another orientation and another pattern. The dictionary's orientations and `disorientation` use the same rotations.

In the same conventions, the sample frame has x down the tilted surface toward the detector and z along the surface normal (kikuchipy's and EDAX's); the pattern centre is in the Bruker convention.

### Tutorials

Two scripts in [examples/](../examples/) run as notebooks ([scripts/run_tutorials.py](../scripts/run_tutorials.py)) and end with checks, so they fail like tests:

| Notebook | What it shows |
|---|---|
| [01_index](../examples/01_index.py) | The nickel scan from file to orientation map: looking at patterns and filters, the pattern-centre vote from a wrong start and its vote surface, the pose fit, indexing, and the result against kikuchipy's; then the same in two sweeps, with the search on binned patterns, and with the patterns averaged with their neighbours |
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
