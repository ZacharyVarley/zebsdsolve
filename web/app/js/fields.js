// The app's controls, one or more per engine setting (src/settings.zig): the key is the setting's
// name, the control shows the engine's value and writes the engine's value. Nothing here has a
// default of its own, and no control stands for a value the engine does not have.

// the method the sweeps before the last use, and which methods any sweep uses
const firstMethod = (s) => (s.sweep_method === "same" ? s.method : s.sweep_method);
const uses = (s, m) => s.method === m || (s.sweeps > 1 && firstMethod(s) === m);
const usesSpherical = (s) => uses(s, "spherical");
const usesDictionary = (s) => uses(s, "dictionary") || uses(s, "pca");
const usesPca = (s) => uses(s, "pca");

export const FIELDS = {
  scan: [
    { key: "map_rows", label: "rows", type: "int", min: 0, hint: "Rows of the map, for files that do not record it. 0: as the file says." },
    { key: "map_cols", label: "columns", type: "int", min: 0, hint: "Columns of the map, for files that do not record it. 0: as the file says." },
    { key: "pattern_height", label: "pattern h", type: "int", min: 0, hint: "Pattern height in pixels, for raw streams without a header." },
    { key: "pattern_width", label: "pattern w", type: "int", min: 0, hint: "Pattern width in pixels, for raw streams without a header." },
    { key: "scan_x_sign", label: "column step", type: "enum", options: [[-1, "−y"], [1, "+y"]], hint: "Which way the beam moves on the sample for a step along the map's columns, along the sample frame's y (the detector's column direction)." },
    { key: "scan_y_sign", label: "row step", type: "enum", options: [[-1, "−x"], [1, "+x"]], hint: "Which way the beam moves for a step down the map's rows, along the sample frame's x (down the tilted surface, toward the detector)." },
  ],
  toy: [
    { key: "toy_seed", label: "seed", type: "int", min: 0, max: 4294967295, hint: "The microstructure comes from this number: the grains' places, their orientations and phases, the turns within them and the blank cells." },
    { key: "toy_grains", label: "grains", type: "int", min: 1, hint: "Points scattered over the map; each cell belongs to the nearest, which makes the grains. 1: a single crystal. As many as there are cells: an orientation per cell." },
    { key: "toy_rows", label: "rows", type: "int", min: 1, max: 4096 },
    { key: "toy_cols", label: "columns", type: "int", min: 1, max: 4096 },
    { key: "toy_height", label: "pattern h", type: "int", min: 8, max: 1024 },
    { key: "toy_width", label: "pattern w", type: "int", min: 8, max: 1024 },
    { key: "toy_spread_deg", label: "spread °", type: "num", step: 0.1, min: 0, hint: "Every cell is turned from its grain's orientation by a small random rotation of this root-mean-square angle. 0: a grain has one orientation." },
    { key: "toy_blank", label: "blank", type: "num", step: 0.01, min: 0, max: 1, hint: "The share of cells that hold no crystal: their patterns are the background alone, and nothing should index them." },
    { key: "toy_contrast", label: "contrast", type: "num", step: 0.05, min: 0.01, max: 1, hint: "The share of a pixel's intensity that is the simulated pattern; the rest is background. 1: the simulation alone. A real detector's patterns are mostly background." },
    { key: "toy_background_sigma", label: "bg σ", type: "num", step: 0.05, min: 0, when: (s) => s.toy_contrast < 1, hint: "The background's shape: flat at 0, else a Gaussian hump over the detector's centre of this sigma, as a fraction of the pattern's width." },
    { key: "toy_counts", label: "e⁻ / px", type: "num", step: 5, min: 0, max: 1000000, hint: "Shot noise. The intensities are scaled to a mean of this many electrons a pixel, and every pixel is a Poisson draw at its own: its variance is its mean, so fewer electrons make a noisier scan. 0: no noise." },
    { key: "toy_noise_seed", label: "noise seed", type: "int", min: 0, max: 4294967295, when: (s) => s.toy_counts > 0, hint: "Where the counting's random draws come from. 0: from the seed above. Another number counts the same microstructure again, with other noise." },
  ],
  resolution: [
    { key: "binning", label: "reduce by", type: "num", step: 0.5, min: 1, max: 64, hint: "Patterns are reduced by this factor as they are read, on the CPU: each pixel of the reduced pattern is the average of the detector area it covers. A whole factor is binning; any other factor is area resampling." },
    { key: "full_resolution", label: "full size for", type: "enum", when: (s) => s.binning !== 1,
      options: [["none", "nothing"], ["scores", "final scores"], ["refine", "refine + scores"], ["all", "everything"]],
      hint: "What still works on the patterns as stored. The search (and the pattern-centre vote) is the costly part and needs the least resolution; refinement and the final scores gain the most from all of it." },
  ],
  nlpar: [
    { key: "nlpar", label: "NLPAR", type: "bool", hint: "Before every filter: replace each pattern by a weighted mean of itself and its neighbours in the map. A neighbour weighs as much as it looks like the same pattern under other noise (Brewick, Wright and Rowenhorst 2019). As the settings below stand, this is the published method: one averaging over 7 × 7 cells at λ 0.7. The pattern view shows the result, and in its corner which neighbours went into it." },
    { key: "nlpar_lambda", label: "λ", type: "num", step: 0.05, min: 0.05, hint: "How fast a neighbour's weight falls once it differs by more than noise explains: exp(−d / λ²), d in standard deviations of what noise alone gives. Smaller: only near-identical neighbours count. Below: how much noise the selected pattern's average keeps." },
    { key: "nlpar_radius", label: "radius", type: "int", min: 1, max: 8, hint: "Neighbours out to this many map cells each way take part in a pass: 3 is a window of 7 × 7, the published method's." },
    { key: "nlpar_iterations", label: "passes", type: "int", min: 1, max: 8, hint: "1: the published method, a single averaging. More: each pass works on the patterns the one before left, with its weights decided anew from them, so a pattern reaches cells × passes away (at most 8) but only through neighbours it resembles at every step. Three passes of 1 cell in place of one pass of 3 keep a grain a cell or two wide from being averaged into the grains around it. The pattern view shows any number of passes." },
  ],
  bgstatic: [
    { key: "static_background", label: "static", type: "enum", options: [["none", "none"], ["average", "average"], ["file", "file"]], hint: "Subtract a static background: the average of patterns spread over the whole map, or the image the file stores." },
    { key: "static_patterns", label: "patterns", type: "int", min: 1, when: (s) => s.static_background === "average", hint: "How many patterns the average is taken over, on a lattice over the map (all of them when the map has fewer)." },
  ],
  bgdynamic: [
    { key: "dynamic_background", label: "dynamic", type: "enum", options: ["none", "subtract", "divide"], hint: "Then use a Gaussian-blurred copy of each pattern as its own background." },
    { key: "dynamic_sigma", label: "σ × width", type: "num", step: 0.01, min: 0.01, when: (s) => s.dynamic_background !== "none", hint: "The blur's sigma as a fraction of the pattern's width." },
  ],
  clahe: [
    { key: "clahe", label: "CLAHE", type: "bool", hint: "Then contrast-limited adaptive histogram equalization. The master patterns get it too." },
    { key: "clahe_grid", label: "tiles", type: "int", min: 1, max: 16, when: (s) => s.clahe, hint: "Tiles per side; each is equalized on its own and pixels blend the nearest four." },
    { key: "clahe_bins", label: "bins", type: "int", min: 2, max: 256, when: (s) => s.clahe, hint: "Histogram levels per tile." },
    { key: "clahe_clip", label: "clip", type: "num", step: 1, min: 0.5, when: (s) => s.clahe, hint: "A level's count is clipped at this many times the mean count, and what is cut is spread over all levels. Large values leave the histogram as it is." },
  ],
  bandpass: [
    { key: "bandpass", label: "band pass", type: "bool", hint: "Then the difference of two Gaussians. The master patterns get it too." },
    { key: "bp_fine", label: "fine σ px", type: "num", step: 0.25, min: 0, when: (s) => s.bandpass, hint: "Sigma of the band pass's fine Gaussian, in pixels of the pattern as stored." },
    { key: "bp_coarse", label: "coarse σ px", type: "num", step: 0.5, min: 0.5, when: (s) => s.bandpass, hint: "Sigma of the band pass's coarse Gaussian, in pixels of the pattern as stored." },
  ],
  mask: [
    { key: "mask", label: "circular mask", type: "bool", hint: "Last: score only the pixels inside a circle." },
    { key: "mask_radius", label: "radius", type: "num", step: 0.05, min: 0.1, when: (s) => s.mask, hint: "The mask's radius as a fraction of half the pattern's smaller side." },
  ],
  master: [
    { key: "pixel_angle_deg", label: "pixel °", type: "num", step: 0.01, min: 0, hint: "Angular width of one stored detector pixel: what turns the pattern filters' pixel sizes into sizes on the sphere. 0: from the geometry, the pixel at the pattern centre." },
    { key: "master_smooth", label: "smooth px", type: "num", step: 0.05, min: 0, hint: "Gaussian smoothing on the sphere, as a sigma in pixels of the pattern being compared. A detector pixel averages the sphere over its footprint; 0.3 is about a square pixel's spread." },
  ],
  // the pattern-centre search: the range it starts from, its slices and its tiers
  pcrange: [
    { key: "pc_span", label: "range xy ±", type: "num", step: 0.01, min: 0, hint: "How far from the starting pattern centre the first tier looks, in x and in y, either way, in fractions of the pattern (the pattern centre's own units)." },
    { key: "pc_span_z", label: "range z ±", type: "num", step: 0.01, min: 0, hint: "How far from the starting pattern centre the first tier looks in z, either way, in pattern heights." },
    { key: "pc_steps", label: "slices xy", type: "int", min: 1, max: 65, hint: "Candidates along x and along y in each tier (an odd count includes the tier's centre). Every fit pattern is correlated under every candidate: the cost grows with slices x, y squared times slices z." },
    { key: "pc_steps_z", label: "slices z", type: "int", min: 1, max: 65, hint: "Candidates along z in each tier." },
    { key: "pc_levels", label: "tiers", type: "int", min: 1, max: 8, hint: "Searches one after the other, each centred on the last one's winner and narrower by the factor beside." },
    { key: "pc_shrink", label: "shrink", type: "num", step: 0.05, min: 0.05, max: 1, hint: "Each tier's ranges are the last one's times this." },
  ],
  // the plane of source points
  plane: [
    { key: "plane_kind", label: "model", type: "enum", options: [["affine", "affine"], ["projective", "projective"]], hint: "What the fit patterns' source points are fitted as, against the scan's rows and columns. Affine: nine numbers, the scan a parallelogram on a plane (steps, their directions and the plane's tilt all free). Projective: eleven, the beam deflected from a pivot onto the tilted sample, so that the scan's parallels need not stay parallel; it matters on scans a tenth of the detector wide or more (Winkelmann et al. 2020), and its two slopes stay near zero unless the points tell them. Fit plane fits this; Auto fit geometry checks it against one centre and a tilt line (PCx the same everywhere, PCy and PCz along the rows) and takes the simplest the map supports." },
    { key: "plane_frame", label: "frame", type: "enum", options: [["nominal", "nominal"], ["follow", "plane"]], hint: "Which sample frame orientations are in once a plane is fitted. As the tilts say: where the sample tilt, the detector tilt and the detector's turn above have it; the plane only says where the beam stood. Follows the plane: its normal is the plane's own and its x runs along the scan's rows, as kikuchipy sets its sample tilt from the plane; the detector's turn above then becomes the plane's." },
  ],
  planemore: [
    { key: "plane_method", label: "method", type: "enum", options: [["points", "per pattern"], ["joint", "joint"]], hint: "Through each pattern's own source (the default): Winkelmann et al.'s and kikuchipy's way, each pattern fitted by itself over its source and its turn (six unknowns), then the model through those points. To all the patterns at once: the model's numbers (3 for one centre, 5 a tilt line, 9 or 11 the plane) fitted to every fit pattern together, by Gauss–Newton on their mean NCC with each pattern's orientation following. Equally accurate on synthetic scans and a little faster, but Auto's comparison of the models leans toward the full plane with it: its points are one step from the plane fitted to every pattern, the held-out ones too." },
    { key: "plane_weights", label: "weights", type: "enum", options: [["information", "information"], ["equal", "equal"]], when: (s) => s.plane_method === "points", hint: "How the patterns' own source points count in the plane's fit. By how sharply each is told: by the curvature of that pattern's own score over its source's place, its orientation following. A small pattern tells its source's distance far better than where it stands across the detector, and some orientations tell it better than others; weighed so, thousands of small map patterns do the work of a few large calibration patterns. Alike: every point the same, as kikuchipy fits a few good patterns." },
    { key: "plane_stride", label: "stride", type: "int", min: 1, max: 4096, hint: "The plane's fit patterns: every n-th cell of the region of interest along its rows and along its columns." },
    { key: "plane_patterns_max", label: "max patterns", type: "int", min: 0, max: 10000000, hint: "No more than this many: the stride is widened until they are (0: no such limit)." },
    { key: "plane_trust", label: "trust", type: "num", step: 0.01, min: 0.001, max: 1, when: (s) => s.plane_method === "points", hint: "A pattern's own source may move this far from where the geometry has it, in pattern heights; a pattern that ends at that edge is left out of the plane." },
    { key: "plane_outlier", label: "outlier σ", type: "num", step: 0.5, min: 0, hint: "A pattern's own point further from the fitted plane than this many robust standard deviations of them all is left out, and the plane fitted again. 0: none is." },
    { key: "plane_iters", label: "max iters", type: "int", min: 1, max: 500, hint: "Most rounds of the patterns' own refinement." },
    { key: "plane_interp", label: "interp.", type: "enum", options: [["bilinear", "bilinear"], ["bicubic", "bicubic"], ["keys6", "Keys 6-pt"]], hint: "How the plane fit samples the master pattern. Bilinear: four texels, and a gradient that is a finite difference a master pixel wide. Bicubic (Catmull–Rom, sixteen texels) and the six-point kernel (thirty-six) come with their own derivative: the score is smooth across the master's pixels and the gradient is exactly its slope. Indexing and refinement are bilinear either way." },
    { key: "plane_master_auto", label: "auto master px", type: "bool", hint: "The plane fit resamples its own copy of the master to the pixel width that suits these patterns: of 0 (as it is), 0.5, 1, 1.5 and 2 detector pixels, the one under which the fit patterns score highest when the fit starts. Patterns that are sharp to the pixel want the master as it is; blurrier ones a coarser one. The log says which was taken." },
    { key: "plane_master_px", label: "master px", type: "num", step: 0.25, min: 0, max: 8, when: (s) => !s.plane_master_auto, hint: "The plane fit resamples its own copy of the master so that one of its pixels is this many detector pixels wide at the pattern centre, each new pixel an average of the master's about it: the master then holds no detail finer than the patterns can show. 0: the master as it is." },
  ],
  pcsearch: [
    { key: "fit_grid", label: "grid", type: "int", min: 1, max: 31, hint: "Fit pattern centre uses a lattice of this many × this many patterns over the region of interest. Every one of them is correlated under every candidate, so the vote's time grows with their number." },
    { key: "fit_bandwidth", label: "bandwidth", type: "int", min: 8, max: 256, hint: "Harmonic bandwidth of the correlation in the pattern-centre vote, of the one that gives the plane fit's patterns their first orientations, and of the one that indexes the selected pattern on the spot (indexing has its own, under Index). The vote's peak sharpens with it and its cost follows the cube of the orientation grid: 64 points per turn up to L = 32, 128 up to 64, 256 up to 128." },
    { key: "pc_z_min", label: "min z", type: "num", step: 0.05, min: 0, hint: "A candidate that puts the detector nearer the source than this (z, in pattern heights) gets no score. Toward z = 0 the detector fills the whole hemisphere and the scores stop being correlations; a search let loose there runs away." },
    { key: "pc_vote", label: "vote", type: "enum", options: [["ballots", "ballots"], ["mean", "mean"]],
      hint: "How each tier's winner is chosen from each fit pattern's own best correlation under every candidate. Ballots: every pattern casts a ballot at each of its few best candidates, the ballots are spread a little, and the densest place wins; patterns that disagree are simply outvoted. Mean: the candidate whose scores, averaged over the patterns, are highest; one broad wrong ridge shared by many patterns then pulls the winner along it." },
    { key: "pc_top_k", label: "top K", type: "int", min: 1, max: 256, when: (s) => s.pc_vote === "ballots", hint: "Each fit pattern casts a ballot at this many of its best candidates (top K)." },
    { key: "pc_gate", label: "gate", type: "num", step: 0.01, min: 0, max: 1, when: (s) => s.pc_vote === "ballots", hint: "… but only at candidates scoring at least this share of the pattern's own best." },
    { key: "pc_kde_sigma", label: "KDE σ", type: "num", step: 0.005, min: 0.001, max: 1, when: (s) => s.pc_vote === "ballots", hint: "The Gaussian each ballot is spread by (a kernel density estimate), as a sigma in fractions of the pattern like the pattern centre itself: the same width in every tier, and never under 0.75 of a candidate step. The densest place of the spread ballots wins." },
    { key: "pc_inlier_radius", label: "inlier r", type: "num", step: 0.01, min: 0.005, when: (s) => s.pc_vote === "ballots", hint: "The winner is refined once to the weighted centre of the ballots within this distance of it (a fraction of the pattern), and found again among those alone." },
    { key: "pc_nms", label: "NMS", type: "int", min: 0, max: 9, when: (s) => s.pc_vote === "ballots", hint: "Non-maximum suppression: a ballot only where the pattern's score is the highest within a cube of this many candidates a side. 0: any of its best candidates, which measured better here (and in the camera-plane tests before): wider suppression pushes ballots onto symmetry-alias peaks." },
  ],
  pixel: [
    { key: "detector_width_mm", label: "detector mm", type: "num", step: 1, min: 1, hint: "What a scan is given a pixel size from when its file records none and none is typed in: this width over the pattern's stored pixels across. 24 mm is 480 pixels of 50 µm. It is used when a scan is opened, and when the pixel size is cleared." },
  ],
  index: [
    { key: "method", label: "method", type: "enum", options: [["spherical", "spherical"], ["dictionary", "dictionary"], ["pca", "dictionary PCA"]], hint: "How each orientation is first found (the last sweep's method, when there are several). Spherical: harmonic cross-correlation over all orientations, each pattern from the point of the sample the beam stood on. Dictionary: inner products, pixel by pixel, with simulated patterns of a grid of orientations, simulated for the map's centre. Dictionary components: the same dictionary simulated once and kept as its leading components (learned by Oja's rule), matched in those." },
    { key: "bandwidth", label: "bandwidth", type: "int", min: 8, max: 256, when: usesSpherical, hint: "Harmonic bandwidth of the spherical correlation when indexing (the pattern-centre vote has its own, under Geometry). The orientation grid has the next power of two ≥ 2L − 1 points per turn (64 up to L = 32, 128 up to 64, 256 up to 128), and the cost follows its cube." },
    { key: "dict_grid", label: "grid", type: "int", min: 2, max: 200, when: usesDictionary, hint: "Cubochoric grid points per half edge; the orientations in the phase's fundamental zone are kept. The mean spacing is about 45° / grid for cubic crystals." },
    { key: "refine", label: "refine", type: "bool", hint: "Refine each orientation on the detector (Levenberg–Marquardt on the NCC), the simulation projected from the pattern's own source point." },
    { key: "refine_iters", label: "iterations", type: "int", min: 0, max: 100, when: (s) => s.refine },
    { key: "refine_step", label: "max step °", type: "num", step: 0.25, min: 0.05, when: (s) => s.refine },
  ],
  accept: [
    { key: "accept_nmi", label: "NMI ≥", type: "num", step: 0.005, min: 0, max: 1, hint: "A match is accepted when its normalized mutual information with its simulation is at least this: an absolute threshold, the same for every pattern. 0: the threshold comes from the calibration beside it instead. The readout below says what NMI chance gives on this scan." },
    { key: "accept_z", label: "z ≥", type: "num", step: 1, when: (s) => !(s.accept_nmi > 0), hint: "With no absolute threshold: a calibration made once for the scan, the same for every pattern. A sample of the region's patterns is scored against simulations at random orientations; a match is accepted when its mutual information stands this many of that chance level's standard deviations above it. The readout below says what NMI that is." },
    { key: "accept_step_deg", label: "settled °", type: "num", step: 0.05, min: 0.01, hint: "… and when its refinement had settled: when the last step that improved it turned the crystal by no more than this. A match whose iterations ran out on the way is left for the next sweep." },
  ],
  sweeps: [
    { key: "sweeps", label: "sweeps", type: "int", min: 1, max: 6, hint: "Passes over the region. With more than one, the first is cheap and each later one takes only the patterns no earlier one indexed acceptably; the last runs at the bandwidth (or dictionary grid) above." },
    { key: "sweep_method", label: "early by", type: "enum", when: (s) => s.sweeps > 1,
      options: [["same", "same"], ["spherical", "spherical"], ["dictionary", "dictionary"], ["pca", "dictionary PCA"]],
      hint: "What the sweeps before the last search with. A cheap search first, and the spherical correlation at full bandwidth for what it leaves, is the usual mix." },
    { key: "sweep_bandwidth", label: "first L", type: "int", min: 8, max: 256, when: (s) => s.sweeps > 1 && firstMethod(s) === "spherical", hint: "The first sweep's bandwidth. The sweeps between it and the last are spaced by equal ratios." },
    { key: "sweep_dict_grid", label: "first grid", type: "int", min: 2, max: 200, when: (s) => s.sweeps > 1 && firstMethod(s) !== "spherical", hint: "The first sweep's dictionary grid. The sweeps between it and the last are spaced by equal ratios." },
  ],
  indexmore: [
    { key: "phase_by", label: "phase by", type: "enum", options: [["nmi", "NMI"], ["ncc", "NCC"]], hint: "With several phases: each pattern takes the phase whose best orientation scores highest by this." },
    { key: "nmi_bins", label: "NMI bins", type: "int", min: 2, max: 32, hint: "Histogram levels per axis of the mutual information. Small patterns have too few pixels for many." },
    { key: "pca_components", label: "components", type: "int", min: 4, max: 1024, when: usesPca, hint: "Components kept of a dictionary. More find more patterns and cost more per comparison: on the nickel scan 128 of a grid of 40 find 94 %, 256 find 97 %." },
    { key: "pca_epochs", label: "Oja passes", type: "int", min: 1, max: 16, when: usesPca, hint: "Passes of Oja's rule over the dictionary to learn its components. One is nearly enough." },
    { key: "batch", label: "batch", type: "int", min: 1, max: 4096, hint: "Patterns read and indexed per step (and per readback). Speed only: the results do not depend on it. Tune speed sets this, the pixels per invocation and the two workgroups to what runs fastest here." },
    { key: "refine_pixels", label: "px / thread", type: "int", min: 8, max: 65536, hint: "Pixels of a pattern one GPU invocation adds up in refinement. Speed only. The fastest value grows with the batch, so Tune speed moves the two together." },
    { key: "wg_refine", label: "WG refine", type: "enum", options: [[16, "16"], [32, "32"], [64, "64"], [128, "128"], [256, "256"]], hint: "GPU invocations per workgroup of refinement's sums. Speed only; each value is compiled once." },
    { key: "wg_fft", label: "WG FFT", type: "enum", options: [[16, "16"], [32, "32"], [64, "64"], [128, "128"], [256, "256"]], hint: "GPU invocations per workgroup of the correlation's FFT kernels. Speed only; each value is compiled once." },
    { key: "tile_mb", label: "FFT MiB", type: "int", min: 8, max: 8192, hint: "Memory the correlation's spectra may take per dispatch: how many patterns of a batch are correlated at once. Speed only, and it made no difference where it was measured, so Tune speed leaves it." },
  ],
  maps: [
    { key: "ipf_lightness", label: "λL", type: "num", step: 0.05, min: 0, max: 1, when: (s) => s.ipf_colors.startsWith("perfect"), hint: "The paper's λL: 1 makes the lightness linear in the distance from a sector's centre, which leaves a small white spot; less widens the light (and dark) centre. The paper recommends a quarter." },
    { key: "ipf_gray", label: "λS", type: "num", step: 0.05, min: 0, max: 1, when: (s) => s.ipf_colors.startsWith("perfect"), hint: "The paper's λS: 0 leaves the centres white and black; more turns them toward gray, which keeps white and black free for marks on the map. The paper recommends a quarter." },
  ],
  // the Simulate step
  simbeam: [
    { key: "sim_voltage_kv", label: "kV", type: "num", step: 1, min: 1, hint: "The beam's energy. The energy model spreads the backscattered electrons from it down over energy loss, and each energy bin is solved at its own voltage." },
    { key: "sim_tilt_deg", label: "tilt °", type: "num", step: 1, min: 0, max: 90, hint: "The sample's tilt (the beam's angle from the surface normal): with the beam's energy and the crystal's density and composition, what the energy model takes to spread the backscattered electrons over energy and depth." },
    { key: "sim_bin_kev", label: "bin keV", type: "num", step: 0.5, min: 0.1, hint: "The width of the energy-loss bins: each is solved at its own voltage and weighed by the share of electrons the energy model puts in it." },
    { key: "sim_stop", label: "stop ≤", type: "num", step: 0.005, min: 0, hint: "Energy bins are solved from the least loss up until one changes the pattern (each normalized to its range) by less than this, relatively. 0: every bin. The chart under the cell shows which were." },
  ],
  simmaster: [
    { key: "sim_halfw", label: "half-width", type: "int", min: 8, max: 2000, hint: "Each hemisphere of the master is a Lambert square of 2 × this + 1 pixels a side. Only the directions in the point group's fundamental sector are solved; the time grows with this squared." },
    { key: "sim_dmin_nm", label: "d min nm", type: "num", step: 0.005, min: 0.005, hint: "The reflections in the potential: every lattice plane whose spacing is above this. Smaller: more reflections, more beams to a direction, finer detail and a longer solve." },
  ],
  simbethe: [
    { key: "sim_c_strong", label: "c₁ strong", type: "num", step: 1, min: 0.1, hint: "Bethe's criterion: a reflection is a strong beam when its excitation error over its coupling to the others, |s_g| / |U_g|, is within c₁. Every direction takes as many beams as the direction with the most strong ones has (at most 512), its own nearest, and solves for them exactly." },
    { key: "sim_c_weak", label: "c₂ weak", type: "num", step: 5, min: 0, hint: "Bethe's weak beams: the reflections after the strong ones whose |s_g| / |U_g| is within c₂ are followed by perturbation, not solved for: each one's amplitude comes from the strong beams' (ψ_w = −Σ A_ws ψ_s / 2πi s_w), it is folded into their equations, and it counts in the intensity. At or below c₁: none. A direction with more strong beams than the device lets it solve (512 in 16 KiB of workgroup memory, 1024 in 32, 1536 in 48) solves its nearest and follows the rest so, whatever this is. Against the full solve (every reflection, by eigendecomposition; nickel, d_min 0.05, one 20 kV bin), once one scale is taken out: c₁ 20 alone errs 1.46 % median (r 0.9937), with weak beams to c₂ 40 1.13 % (r 0.9974); weak beams in the equations alone, as EMsoft has them, 2.2 % (r 0.9889)." },
    { key: "sim_c_cutoff", label: "c₃ cutoff", type: "num", step: 10, min: 0.1, hint: "Bethe's criterion: a reflection whose |s_g| / |U_g| is beyond c₃ is never a beam." },
    { key: "sim_dbdiff_cutoff", label: "dd |s|", type: "num", step: 0.1, min: 0, hint: "A reflection the structure forbids (reached only by double diffraction, through others) is a beam only when its excitation error is within this." },
  ],
  simsolver: [
    { key: "sim_rank", label: "rank", type: "int", min: 1, max: 8, slider: true, wide: true, hint: "The size of the subspace each direction's depth integral is solved in: the time grows about in proportion. Measured against the exact solution with the correction on (Ti, GaN and alexandrite, one bin): rank 2 about 0.2–0.7 % median error, 3 about 0.2–0.6 % (p90 within 2 %), 4 about 0.2–0.5 % (p90 within 1.8 %); without it, rank 8 errs about 1 % (p90 2–3 %)." },
    { key: "sim_correct", label: "DWR correction", type: "bool", wide: true, hint: "Correct the subspace's estimate of each direction's intensity by its residual, weighted by the dual problem: taken diagonal over most beam pairs (each beam's own excitation error) and solved exactly on the 30 beams where the residual is largest. The error left is second order; it costs about one more pass over the beam pairs. Off: the plain Galerkin estimate, several times less accurate at the same rank." },
  ],
  roi: [
    { key: "roi_x", label: "left", type: "int", min: 0 },
    { key: "roi_y", label: "top", type: "int", min: 0 },
    { key: "roi_w", label: "width", type: "int", min: 0 },
    { key: "roi_h", label: "height", type: "int", min: 0 },
  ],
};

/** The group a setting's (first) control is in. */
export function groupOf(key) {
  return Object.entries(FIELDS).find(([, list]) => list.some((f) => f.key === key))?.[0];
}

// The geometry is the engine's state, not a setting: [field, label, hint, step, shown per unit
// of the engine's value (the rotation is kept in radians and shown in degrees), width in its row].
export const GEOMETRY = [
  ["pc", "PC", "Where the detector stands, for the beam at the map's centre. Bruker convention: x from the left and y from the top in fractions of the pattern's width and height; z, the detector distance, in fractions of its height.", 0.001, 1],
  ["sampleTiltDeg", "sample tilt °", "", 0.5, 1],
  ["detectorTiltDeg", "detector tilt °", "", 0.5, 1],
  ["rot", "rotation °", "The detector's attitude away from the one the two tilts give it, as a rotation vector in its own axes: swing about its row direction (down the pattern), tilt about its column direction (in the detector tilt's sense), twist about its normal. A turn is about the source point: the pattern centre stays as it is written, and the pattern at the map's centre does not change (its crystal's orientation turns with the detector). A plane fit whose sample frame follows the plane sets it.", 0.05, 180 / Math.PI],
  ["stepXUm", "step x µm", "The scan's step along the map's columns.", 0.1, 1],
  ["stepYUm", "step y µm", "The scan's step down the map's rows.", 0.1, 1],
  ["pxUm", "pixel µm", "The size of one stored pattern pixel on the detector (a binned pixel, when the patterns were binned): with the scan's steps, how far the beam moved between patterns, in pixels, until a plane of source points is fitted: the plane then says where the beam stood, and this only how its steps compare with the scan's. It is the file's when the file records one, else the detector's width over the pattern's pixels across, until a value is typed here. Cleared (or 0), it goes back to that. No fit changes it.", 10, 1],
];

/** Build the controls of every `.fields[data-group]`; `apply(key, value)` sets the engine's value. */
export function buildFields(root, apply) {
  const inputs = [];
  for (const box of root.querySelectorAll(".fields[data-group]")) {
    for (const f of FIELDS[box.dataset.group] || []) {
      const el = document.createElement("label");
      // a label and its control side by side; a choice whose words do not fit beside its label takes the row
      const words = f.type === "enum" ? Math.max(...f.options.map((o) => String(Array.isArray(o) ? o[1] : o).length)) : 0;
      const wide = f.wide || f.slider || (f.type === "enum" && f.label.length + words > 16);
      el.className = "field" + (f.type === "bool" ? " toggle" : "") + (wide ? " wide" : "");
      el.dataset.key = f.key;
      if (f.hint) el.title = f.hint;
      let input;
      if (f.type === "enum") {
        input = document.createElement("select");
        for (const o of f.options) {
          const [v, text] = Array.isArray(o) ? o : [o, o];
          const opt = document.createElement("option");
          opt.value = String(v);
          opt.textContent = text;
          input.append(opt);
        }
      } else {
        input = document.createElement("input");
        input.type = f.type === "bool" ? "checkbox" : f.slider ? "range" : "number";
        if (f.type !== "bool") {
          input.step = f.type === "int" ? "1" : String(f.step ?? "any");
          if (f.min !== undefined) input.min = String(f.min);
          if (f.max !== undefined) input.max = String(f.max);
        }
      }
      const span = document.createElement("span");
      span.textContent = f.label;
      // a slider shows its value beside it, following the thumb as it moves
      let readout = null;
      if (f.slider) {
        readout = document.createElement("output");
        readout.className = "readout";
        input.addEventListener("input", () => { readout.textContent = input.value; });
      }
      if (f.type === "bool") el.append(input, span);
      else if (readout) {
        const row = document.createElement("div");
        row.className = "slider-row";
        row.append(input, readout);
        el.append(span, row);
      } else el.append(span, input);
      box.append(el);
      inputs.push({ f, input, el, readout });
      input.addEventListener("change", async () => {
        let v;
        if (f.type === "bool") v = input.checked;
        else if (f.type === "enum") v = typeof (Array.isArray(f.options[0]) ? f.options[0][0] : f.options[0]) === "number" ? +input.value : input.value;
        else v = f.type === "int" ? Math.round(+input.value) : +input.value;
        if (f.type !== "bool" && f.type !== "enum" && !Number.isFinite(v)) return;
        await apply(f.key, v);
      });
    }
  }
  return {
    /** Show the engine's values. */
    show(settings) {
      for (const { f, input, el, readout } of inputs) {
        if (!(f.key in settings)) continue;
        if (f.type === "bool") input.checked = !!settings[f.key]; else input.value = String(settings[f.key]);
        if (readout) readout.textContent = String(settings[f.key]);
        el.dataset.off = f.when && !f.when(settings) ? "1" : "0";
      }
    },
  };
}
