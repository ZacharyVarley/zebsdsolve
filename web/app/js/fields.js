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
    { key: "map_rows", label: "map rows", type: "int", min: 0, hint: "Rows of the map, for files that do not record it. 0: as the file says." },
    { key: "map_cols", label: "map columns", type: "int", min: 0, hint: "Columns of the map, for files that do not record it. 0: as the file says." },
    { key: "pattern_height", label: "pattern height", type: "int", min: 0, hint: "Pattern height in pixels, for raw streams without a header." },
    { key: "pattern_width", label: "pattern width", type: "int", min: 0, hint: "Pattern width in pixels, for raw streams without a header." },
    { key: "scan_x_sign", label: "column step along y", type: "enum", options: [[-1, "−y (kikuchipy)"], [1, "+y"]], hint: "Which way the beam moves on the sample for a step along the map's columns, along the sample frame's y (the detector's column direction)." },
    { key: "scan_y_sign", label: "row step along x", type: "enum", options: [[-1, "−x (kikuchipy)"], [1, "+x"]], hint: "Which way the beam moves for a step down the map's rows, along the sample frame's x (down the tilted surface, toward the detector)." },
  ],
  toy: [
    { key: "toy_seed", label: "seed", type: "int", min: 0, max: 4294967295, hint: "The microstructure comes from this number: the grains' places, their orientations and phases, the turns within them and the blank cells." },
    { key: "toy_grains", label: "grains", type: "int", min: 1, hint: "Points scattered over the map; each cell belongs to the nearest, which makes the grains. 1: a single crystal. As many as there are cells: an orientation per cell." },
    { key: "toy_rows", label: "map rows", type: "int", min: 1, max: 4096 },
    { key: "toy_cols", label: "map columns", type: "int", min: 1, max: 4096 },
    { key: "toy_height", label: "pattern height (px)", type: "int", min: 8, max: 1024 },
    { key: "toy_width", label: "pattern width (px)", type: "int", min: 8, max: 1024 },
    { key: "toy_spread_deg", label: "spread in a grain (°)", type: "num", step: 0.1, min: 0, hint: "Every cell is turned from its grain's orientation by a small random rotation of this root-mean-square angle. 0: a grain has one orientation." },
    { key: "toy_blank", label: "blank cells (share)", type: "num", step: 0.01, min: 0, max: 1, hint: "The share of cells that hold no crystal: their patterns are the background alone, and nothing should index them." },
    { key: "toy_contrast", label: "contrast", type: "num", step: 0.05, min: 0.01, max: 1, hint: "The share of a pixel's intensity that is the simulated pattern; the rest is background. 1: the simulation alone. A real detector's patterns are mostly background." },
    { key: "toy_background_sigma", label: "background sigma × width", type: "num", step: 0.05, min: 0, when: (s) => s.toy_contrast < 1, hint: "The background's shape: flat at 0, else a Gaussian hump over the detector's centre of this sigma, as a fraction of the pattern's width." },
    { key: "toy_counts", label: "electrons per pixel", type: "num", step: 5, min: 0, max: 1000000, hint: "Shot noise. The intensities are scaled to a mean of this many electrons a pixel, and every pixel is a Poisson draw at its own: its variance is its mean, so fewer electrons make a noisier scan. 0: no noise." },
    { key: "toy_noise_seed", label: "noise seed", type: "int", min: 0, max: 4294967295, when: (s) => s.toy_counts > 0, hint: "Where the counting's random draws come from. 0: from the seed above. Another number counts the same microstructure again, with other noise." },
  ],
  resolution: [
    { key: "binning", label: "reduce by", type: "num", step: 0.5, min: 1, max: 64, hint: "Patterns are reduced by this factor as they are read, on the CPU: each pixel of the reduced pattern is the average of the detector area it covers. A whole factor is binning; any other factor is area resampling." },
    { key: "full_resolution", label: "at stored size", type: "enum", wide: true, when: (s) => s.binning !== 1,
      options: [["none", "nothing: everything at the reduced size"], ["scores", "the final scores and the phase"], ["refine", "refinement, final scores and the phase"], ["all", "everything: the reduced size is for viewing only"]],
      hint: "What still works on the patterns as stored. The search (and the pattern-centre vote) is the costly part and needs the least resolution; refinement and the final scores gain the most from all of it." },
  ],
  nlpar: [
    { key: "nlpar", label: "average with neighbours (NLPAR)", type: "bool", wide: true, hint: "Before every filter: replace each pattern by a weighted mean of itself and its neighbours in the map. A neighbour weighs as much as it looks like the same pattern under other noise (Brewick, Wright and Rowenhorst 2019). As the settings below stand, this is the published method: one averaging over 7 × 7 cells at λ 0.7. The pattern view shows the result, and in its corner which neighbours went into it." },
    { key: "nlpar_lambda", label: "lambda", type: "num", step: 0.05, min: 0.05, hint: "How fast a neighbour's weight falls once it differs by more than noise explains: exp(−d / λ²), d in standard deviations of what noise alone gives. Smaller: only near-identical neighbours count. Below: how much noise the selected pattern's average keeps." },
    { key: "nlpar_radius", label: "cells each way", type: "int", min: 1, max: 8, hint: "Neighbours out to this many map cells each way take part in a pass: 3 is a window of 7 × 7, the published method's." },
    { key: "nlpar_iterations", label: "passes", type: "int", min: 1, max: 8, wide: true, hint: "1: the published method, a single averaging. More: each pass works on the patterns the one before left, with its weights decided anew from them, so a pattern reaches cells × passes away (at most 8) but only through neighbours it resembles at every step. Three passes of 1 cell in place of one pass of 3 keep a grain a cell or two wide from being averaged into the grains around it. The pattern view shows any number of passes." },
  ],
  filters: [
    { key: "static_background", label: "static background", type: "enum", options: [["none", "none"], ["average", "average pattern"], ["file", "the file's"]], hint: "Subtract a static background: the average of patterns spread over the whole map, or the image the file stores." },
    { key: "static_patterns", label: "averaged over", type: "int", min: 1, when: (s) => s.static_background === "average", hint: "How many patterns the average is taken over, on a lattice over the map (all of them when the map has fewer)." },
    { key: "dynamic_background", label: "dynamic background", type: "enum", options: ["none", "subtract", "divide"], hint: "Then use a Gaussian-blurred copy of each pattern as its own background." },
    { key: "dynamic_sigma", label: "its sigma × width", type: "num", step: 0.01, min: 0.01, when: (s) => s.dynamic_background !== "none", hint: "The blur's sigma as a fraction of the pattern's width." },
    { key: "clahe", label: "CLAHE", type: "bool", hint: "Then contrast-limited adaptive histogram equalization. The master patterns get it too." },
    { key: "clahe_grid", label: "tiles per side", type: "int", min: 1, max: 16, when: (s) => s.clahe, hint: "Tiles per side; each is equalized on its own and pixels blend the nearest four." },
    { key: "clahe_bins", label: "levels", type: "int", min: 2, max: 256, when: (s) => s.clahe, hint: "Histogram levels per tile." },
    { key: "clahe_clip", label: "clip", type: "num", step: 1, min: 0.5, when: (s) => s.clahe, hint: "A level's count is clipped at this many times the mean count, and what is cut is spread over all levels. Large values leave the histogram as it is." },
    { key: "bandpass", label: "band pass", type: "bool", wide: true, hint: "Then the difference of two Gaussians. The master patterns get it too." },
    { key: "bp_fine", label: "fine sigma (px)", type: "num", step: 0.25, min: 0, when: (s) => s.bandpass, hint: "Sigma of the band pass's fine Gaussian, in pixels of the pattern as stored." },
    { key: "bp_coarse", label: "coarse sigma (px)", type: "num", step: 0.5, min: 0.5, when: (s) => s.bandpass, hint: "Sigma of the band pass's coarse Gaussian, in pixels of the pattern as stored." },
    { key: "mask", label: "circular mask", type: "bool", hint: "Last: score only the pixels inside a circle." },
    { key: "mask_radius", label: "mask radius", type: "num", step: 0.05, min: 0.1, when: (s) => s.mask, hint: "The mask's radius as a fraction of half the pattern's smaller side." },
  ],
  master: [
    { key: "pixel_angle_deg", label: "pixel width (°)", type: "num", step: 0.01, min: 0, hint: "Angular width of one stored detector pixel: what turns the pattern filters' pixel sizes into sizes on the sphere. 0: from the geometry, the pixel at the pattern centre." },
    { key: "master_smooth", label: "smoothing (px)", type: "num", step: 0.05, min: 0, hint: "Gaussian smoothing on the sphere, as a sigma in pixels of the pattern being compared. A detector pixel averages the sphere over its footprint; 0.3 is about a square pixel's spread." },
  ],
  fit: [
    { key: "fit_bandwidth", label: "bandwidth L", type: "int", min: 8, max: 256, hint: "Harmonic bandwidth of the correlation in the pattern-centre vote, and of the one that indexes the selected pattern on the spot (indexing has its own, under Index). The vote's peak sharpens with it and its cost follows the cube of the orientation grid: 64 points per turn up to L = 32, 128 up to 64, 256 up to 128." },
    { key: "fit_grid", label: "vote: patterns per side", type: "int", min: 1, max: 31, hint: "Fit pattern centre uses a lattice of this many × this many patterns over the region of interest. Every one of them is correlated under every candidate, so the vote's time grows with their number." },
    { key: "pose_grid", label: "pose: patterns per side", type: "int", min: 1, max: 63, hint: "Fit pose uses a lattice of this many × this many patterns over the region of interest. A pattern costs it little, and more of them are what pins the detector's rotation: turning the detector and turning every crystal the other way look alike in any one pattern, and only the beam's travel between patterns tells them apart." },
  ],
  fitmore: [
    { key: "pose_iters", label: "pose iterations", type: "int", min: 1, max: 500, hint: "Most Levenberg–Marquardt iterations of the pose fit." },
    { key: "pc_vote", label: "the grid's winner by", type: "enum", wide: true, options: [["ballots", "ballots: where the most patterns agree"], ["mean", "mean: the highest average score"]],
      hint: "How a grid's winner is chosen from each fit pattern's own best correlation under every candidate. Ballots: every pattern casts a ballot at each of its few best candidates, the ballots are spread a little, and the densest place wins; patterns that disagree are simply outvoted. Mean: the candidate whose scores, averaged over the patterns, are highest; one broad wrong ridge shared by many patterns then pulls the winner along it." },
    { key: "pc_top_k", label: "ballots per pattern", type: "int", min: 1, max: 256, when: (s) => s.pc_vote === "ballots", hint: "Each fit pattern casts a ballot at this many of its best candidates (top K)." },
    { key: "pc_gate", label: "gate", type: "num", step: 0.01, min: 0, max: 1, when: (s) => s.pc_vote === "ballots", hint: "… but only at candidates scoring at least this share of the pattern's own best." },
    { key: "pc_kde_sigma", label: "spread (of the pattern)", type: "num", step: 0.005, min: 0.001, max: 1, when: (s) => s.pc_vote === "ballots", hint: "The Gaussian each ballot is spread by (a kernel density estimate), as a sigma in fractions of the pattern like the pattern centre itself: the same width on every grid, and never under 0.75 of a candidate step. The densest place of the spread ballots wins." },
    { key: "pc_inlier_radius", label: "inlier radius (of the pattern)", type: "num", step: 0.01, min: 0.005, when: (s) => s.pc_vote === "ballots", hint: "The winner is refined once to the weighted centre of the ballots within this distance of it (a fraction of the pattern), and found again among those alone." },
    { key: "pc_nms", label: "local maxima within", type: "int", min: 0, max: 9, wide: true, when: (s) => s.pc_vote === "ballots", hint: "Non-maximum suppression: a ballot only where the pattern's score is the highest within a cube of this many candidates a side. 0: any of its best candidates, which measured better here (and in the camera-plane tests before): wider suppression pushes ballots onto symmetry-alias peaks." },
    { key: "pc_steps", label: "candidates x, y", type: "int", min: 1, max: 65, hint: "Pattern-centre candidates per axis in x and in y (odd counts include the starting value)." },
    { key: "pc_span", label: "half-width x, y", type: "num", step: 0.01, min: 0, hint: "Half-width of the search in x and y, in fractions of the pattern." },
    { key: "pc_steps_z", label: "candidates z", type: "int", min: 1, max: 65 },
    { key: "pc_span_z", label: "half-width z", type: "num", step: 0.01, min: 0 },
    { key: "pc_z_min", label: "nearest z", type: "num", step: 0.05, min: 0, hint: "A candidate that puts the detector nearer the source than this (z, in pattern heights) gets no score. Toward z = 0 the detector fills the whole hemisphere and the scores stop being correlations; a search let loose there runs away." },
    { key: "pc_levels", label: "levels", type: "int", min: 1, max: 8, hint: "Searches, each centred on the last one's best and narrower." },
    { key: "pc_shrink", label: "shrink per level", type: "num", step: 0.05, min: 0.05, max: 1, hint: "Each level's half-widths are the last one's times this." },
    { key: "pose_x", label: "pose: fit centre x", type: "bool", hint: "The pose fit moves the detector along its columns: the pattern centre's x. Off: x keeps its value." },
    { key: "pose_y", label: "pose: fit centre y", type: "bool", hint: "The pose fit moves the detector along its rows: the pattern centre's y." },
    { key: "pose_z", label: "pose: fit centre z", type: "bool", hint: "The pose fit moves the detector along its normal: the detector distance." },
    { key: "pose_tilt", label: "pose: fit tilt", type: "bool", hint: "The pose fit turns the detector about its column direction (across the pattern), in the detector tilt's sense. Like swing and twist it is a turn about the source point, which leaves the pattern centre where it is." },
    { key: "pose_swing", label: "pose: fit swing", type: "bool", hint: "The pose fit turns the detector about its row direction (down the pattern)." },
    { key: "pose_twist", label: "pose: fit twist", type: "bool", hint: "The pose fit turns the detector about its own normal: a rotation of the camera in its plane." },
    { key: "pose_scale", label: "pose: fit pixel size", type: "bool", hint: "A seventh degree of freedom, making the pose a similarity (Sim(3)): the scale between the beam's travel over the sample and the detector, which is the detector's pixel size. It needs a pixel size to start from, and a map wide enough for the pattern centre to move visibly across it." },
  ],
  index: [
    { key: "method", label: "method", type: "enum", options: [["spherical", "spherical"], ["dictionary", "dictionary"], ["pca", "dictionary components"]], hint: "How each orientation is first found (the last sweep's method, when there are several). Spherical: harmonic cross-correlation over all orientations, each pattern from the point of the sample the beam stood on. Dictionary: inner products, pixel by pixel, with simulated patterns of a grid of orientations, simulated for the map's centre. Dictionary components: the same dictionary simulated once and kept as its leading components (learned by Oja's rule), matched in those." },
    { key: "bandwidth", label: "bandwidth L", type: "int", min: 8, max: 256, when: usesSpherical, hint: "Harmonic bandwidth of the spherical correlation when indexing (the pattern-centre vote has its own, under Geometry). The orientation grid has the next power of two ≥ 2L − 1 points per turn (64 up to L = 32, 128 up to 64, 256 up to 128), and the cost follows its cube." },
    { key: "dict_grid", label: "dictionary grid", type: "int", min: 2, max: 200, when: usesDictionary, hint: "Cubochoric grid points per half edge; the orientations in the phase's fundamental zone are kept. The mean spacing is about 45° / grid for cubic crystals." },
    { key: "refine", label: "refine", type: "bool", wide: true, hint: "Refine each orientation on the detector (Levenberg–Marquardt on the NCC), the simulation projected from the pattern's own source point." },
    { key: "refine_iters", label: "iterations", type: "int", min: 0, max: 100, when: (s) => s.refine },
    { key: "refine_step", label: "largest step (°)", type: "num", step: 0.25, min: 0.05, when: (s) => s.refine },
  ],
  sweeps: [
    { key: "sweeps", label: "sweeps", type: "int", min: 1, max: 6, hint: "Passes over the region. With more than one, the first is cheap and each later one takes only the patterns no earlier one indexed acceptably; the last runs at the bandwidth (or dictionary grid) above." },
    { key: "accept_nmi", label: "accept at NMI ≥", type: "num", step: 0.005, min: 0, max: 1, hint: "A match is accepted when its normalized mutual information with its simulation is at least this: an absolute threshold, the same for every pattern. 0: the threshold comes from the calibration beside it instead. The readout below says what NMI chance gives on this scan." },
    { key: "accept_z", label: "or at z ≥", type: "num", step: 1, when: (s) => !(s.accept_nmi > 0), hint: "With no absolute threshold: a calibration made once for the scan, the same for every pattern. A sample of the region's patterns is scored against simulations at random orientations; a match is accepted when its mutual information stands this many of that chance level's standard deviations above it. The readout below says what NMI that is." },
    { key: "accept_step_deg", label: "and settled within (°)", type: "num", step: 0.05, min: 0.01, wide: true, hint: "… and when its refinement had settled: when the last step that improved it turned the crystal by no more than this. A match whose iterations ran out on the way is left for the next sweep." },
    { key: "sweep_method", label: "earlier sweeps by", type: "enum", wide: true, when: (s) => s.sweeps > 1,
      options: [["same", "the same method"], ["spherical", "spherical"], ["dictionary", "dictionary"], ["pca", "dictionary components"]],
      hint: "What the sweeps before the last search with. A cheap search first, and the spherical correlation at full bandwidth for what it leaves, is the usual mix." },
    { key: "sweep_bandwidth", label: "first bandwidth", type: "int", min: 8, max: 256, when: (s) => s.sweeps > 1 && firstMethod(s) === "spherical", hint: "The first sweep's bandwidth. The sweeps between it and the last are spaced by equal ratios." },
    { key: "sweep_dict_grid", label: "first dictionary grid", type: "int", min: 2, max: 200, when: (s) => s.sweeps > 1 && firstMethod(s) !== "spherical", hint: "The first sweep's dictionary grid. The sweeps between it and the last are spaced by equal ratios." },
  ],
  indexmore: [
    { key: "phase_by", label: "choose phase by", type: "enum", options: [["nmi", "mutual information"], ["ncc", "cross-correlation"]], hint: "With several phases: each pattern takes the phase whose best orientation scores highest by this." },
    { key: "nmi_bins", label: "NMI levels", type: "int", min: 2, max: 32, hint: "Histogram levels per axis of the mutual information. Small patterns have too few pixels for many." },
    { key: "pca_components", label: "components", type: "int", min: 4, max: 1024, when: usesPca, hint: "Components kept of a dictionary. More find more patterns and cost more per comparison: on the nickel scan 128 of a grid of 40 find 94 %, 256 find 97 %." },
    { key: "pca_epochs", label: "Oja passes", type: "int", min: 1, max: 16, when: usesPca, hint: "Passes of Oja's rule over the dictionary to learn its components. One is nearly enough." },
    { key: "batch", label: "patterns per step", type: "int", min: 1, max: 4096, hint: "Patterns read and indexed per step (and per readback). Speed only: the results do not depend on it. Tune speed sets this, the pixels per invocation and the two workgroups to what runs fastest here." },
    { key: "refine_pixels", label: "pixels per invocation", type: "int", min: 8, max: 65536, hint: "Pixels of a pattern one GPU invocation adds up in refinement. Speed only. The fastest value grows with the batch, so Tune speed moves the two together." },
    { key: "wg_refine", label: "workgroup, refinement", type: "enum", options: [[16, "16"], [32, "32"], [64, "64"], [128, "128"], [256, "256"]], hint: "GPU invocations per workgroup of refinement's sums. Speed only; each value is compiled once." },
    { key: "wg_fft", label: "workgroup, correlation", type: "enum", options: [[16, "16"], [32, "32"], [64, "64"], [128, "128"], [256, "256"]], hint: "GPU invocations per workgroup of the correlation's FFT kernels. Speed only; each value is compiled once." },
    { key: "tile_mb", label: "correlation memory (MiB)", type: "int", min: 8, max: 8192, wide: true, hint: "Memory the correlation's spectra may take per dispatch: how many patterns of a batch are correlated at once. Speed only, and it made no difference where it was measured, so Tune speed leaves it." },
  ],
  maps: [
    { key: "ipf_colors", label: "inverse pole figure colours", type: "enum", wide: true, options: [["classic", "classic: red, green, blue corners"], ["hue", "white centre, the hue round it"], ["perfect", "Nolze & Hielscher, by direction"], ["perfect_rotation", "Nolze & Hielscher, by orientation"]],
      hint: "Classic: the cubic triangle has red, green and blue at [001], [101] and [111] and mixes them between. White centre: white at the middle of the sector, the hue round it at an even rate. The two after Nolze and Hielscher (2016, 'Orientations – perfectly colored') balance the hue so that red, yellow, green, cyan, blue and magenta take even shares, and widen and gray the centre. By direction: the key of the Laue class; a class whose sector is not bounded by mirrors (m-3, -3m, 6/m, 4/m, 2/m) gets a light and a dark half that meet without a jump. By orientation: the key of the class's proper rotations alone, so that a direction and its opposite differ and every class has a light and a dark half; this is the one that tells apart orientations the Laue key colours alike. Both are written from the paper; neither was compared with MTEX's own output." },
    { key: "ipf_lightness", label: "centre's width (λL)", type: "num", step: 0.05, min: 0, max: 1, when: (s) => s.ipf_colors.startsWith("perfect"), hint: "The paper's λL: 1 makes the lightness linear in the distance from a sector's centre, which leaves a small white spot; less widens the light (and dark) centre. The paper recommends a quarter." },
    { key: "ipf_gray", label: "centre's gray (λS)", type: "num", step: 0.05, min: 0, max: 1, when: (s) => s.ipf_colors.startsWith("perfect"), hint: "The paper's λS: 0 leaves the centres white and black; more turns them toward gray, which keeps white and black free for marks on the map. The paper recommends a quarter." },
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
// of the engine's value (the rotation is kept in radians and shown in degrees)].
export const GEOMETRY = [
  ["pc", "pattern centre x, y, z", "Where the detector stands, for the beam at the map's centre. Bruker convention: x from the left and y from the top in fractions of the pattern's width and height; z, the detector distance, in fractions of its height.", 0.001, 1],
  ["sampleTiltDeg", "sample tilt (°)", "", 0.5, 1],
  ["detectorTiltDeg", "detector tilt (°)", "", 0.5, 1],
  ["rot", "swing, tilt, twist (°)", "A turn of the detector about the source point, away from the attitude the two tilts give it, as a rotation vector in the detector's own axes: swing about its row direction (down the pattern), tilt about its column direction (in the detector tilt's sense), twist about its normal. The pattern centre stays where it is under it. Fit pose finds them, unless its options hold them.", 0.05, 180 / Math.PI],
  ["pxUm", "detector pixel (µm)", "Size of one stored pattern pixel on the detector: what turns the beam's travel across the map into pixels. 0: unknown, and every pattern is then taken to come from the one point of the sample. Fit pose can find it (fit options: fit pixel size) from a rough value, on a map wide enough.", 1, 1],
  ["stepXUm", "step x (µm)", "", 0.1, 1],
  ["stepYUm", "step y (µm)", "", 0.1, 1],
];

/** Build the controls of every `.fields[data-group]`; `apply(key, value)` sets the engine's value. */
export function buildFields(root, apply) {
  const inputs = [];
  for (const box of root.querySelectorAll(".fields[data-group]")) {
    for (const f of FIELDS[box.dataset.group] || []) {
      const el = document.createElement("label");
      el.className = "field" + (f.type === "bool" ? " toggle" : "") + (f.wide ? " wide" : "");
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
        input.type = f.type === "bool" ? "checkbox" : "number";
        if (f.type !== "bool") {
          input.step = f.type === "int" ? "1" : String(f.step ?? "any");
          if (f.min !== undefined) input.min = String(f.min);
          if (f.max !== undefined) input.max = String(f.max);
        }
      }
      const span = document.createElement("span");
      span.textContent = f.label;
      if (f.type === "bool") el.append(input, span); else el.append(span, input);
      box.append(el);
      inputs.push({ f, input, el });
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
      for (const { f, input, el } of inputs) {
        if (!(f.key in settings)) continue;
        if (f.type === "bool") input.checked = !!settings[f.key]; else input.value = String(settings[f.key]);
        el.dataset.off = f.when && !f.when(settings) ? "1" : "0";
      }
    },
  };
}
