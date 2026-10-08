//! Every option of the pipeline, with the app's defaults. Front ends set them with a JSON
//! object (ze_configure): keys are these field names, enums are their names as strings; absent
//! keys keep their current value. A new option goes here first; both front ends pick it up by
//! name. The detector geometry is not an option: it is the engine's state (geometry.zig), set
//! from the scan file, by hand, or by the fits.
const std = @import("std");

pub const Static = enum { none, average, file };
pub const Background = enum { none, subtract, divide };
pub const Method = enum { spherical, dictionary, pca };
/// What the sweeps before the last search with: the last one's method, or one of their own.
pub const SweepMethod = enum { same, spherical, dictionary, pca };
pub const PhaseBy = enum { nmi, ncc };
/// How the fit patterns' scores under the candidate pattern centres become one choice.
pub const PcVote = enum { ballots, mean };
/// How inverse pole figure colours are assigned (sym.Colors).
pub const IpfColors = @import("sym.zig").Colors;
/// Which stages work on patterns at their stored size rather than the reduced one.
pub const FullResolution = enum { none, scores, refine, all };

pub const Settings = struct {
    // ── scan ──
    /// Map shape and pattern size for files that do not record them (raw streams); 0: as the
    /// file says.
    map_rows: u32 = 0,
    map_cols: u32 = 0,
    pattern_height: u32 = 0,
    pattern_width: u32 = 0,
    /// Which way the beam moves on the sample as the map index grows, in the sample frame the
    /// detector geometry is given in (x down the tilted surface toward the detector, y along
    /// the detector's columns): +1 or −1 along y for a column step, along x for a row step.
    /// −1, −1 is kikuchipy's convention (rows run up the slope, away from the detector).
    scan_x_sign: i32 = -1,
    scan_y_sign: i32 = -1,
    /// Region of the map to index, in map cells (0 width or height: the whole map).
    roi_x: u32 = 0,
    roi_y: u32 = 0,
    roi_w: u32 = 0,
    roi_h: u32 = 0,

    // ── a toy scan (Engine.synthesize) ──
    /// A scan made from the phases loaded, with a known answer, for trying things on and for
    /// comparing ways of fitting the geometry. It is simulated under the geometry as it stands
    /// when it is made (pattern centre, tilts, detector rotation, pixel size and steps: those
    /// are its true geometry, and the beam travels across it as they say).
    ///
    /// The microstructure, drawn from toy_seed: a map of toy_rows × toy_cols cells cut into
    /// toy_grains grains (each cell belongs to the nearest of that many points scattered over
    /// the map; 1 is a single crystal, as many as there are cells is an orientation per cell),
    /// every grain with an orientation and a phase drawn at random.
    toy_seed: u32 = 1,
    toy_rows: u32 = 48,
    toy_cols: u32 = 64,
    toy_grains: u32 = 24,
    /// Within a grain, every cell is turned from the grain's orientation by a small random
    /// rotation of this root-mean-square angle, degrees (0: a grain has one orientation).
    toy_spread_deg: f64 = 0,
    /// The share of cells that hold no crystal at all (a pore, a scratch): their patterns are
    /// the background alone, and their true phase is −1.
    toy_blank: f64 = 0,
    /// The patterns' size, pixels.
    toy_height: u32 = 60,
    toy_width: u32 = 60,
    /// The detector: every pixel's intensity is (1 − toy_contrast) × a smooth background +
    /// toy_contrast × the simulated pattern, both scaled to a mean of 1 (the pattern with the
    /// scan's darkest pixel at 0). 1 is the simulation alone; a real detector's patterns are
    /// mostly background. The background is flat, or with toy_background_sigma above 0 a
    /// Gaussian hump over the detector's centre of that sigma (a fraction of its width).
    toy_contrast: f64 = 1,
    toy_background_sigma: f64 = 0,
    /// Shot noise: the intensities are scaled to a mean of toy_counts electrons a pixel and
    /// every pixel is a Poisson draw at its own (its variance is its mean, so fewer counts are
    /// a noisier scan). 0: no noise, the intensities themselves. The draws come from
    /// toy_noise_seed, or from toy_seed when that is 0, so the one microstructure can be
    /// counted again and again.
    toy_counts: f64 = 20,
    toy_noise_seed: u32 = 0,

    // ── resolution ──
    /// Patterns are reduced by this factor as they are read, on the CPU: each pixel of the
    /// reduced pattern is the average of the detector area it covers (reduce.zig). A whole
    /// factor is binning; any other is area resampling. The reduced pattern has
    /// floor(size / binning) pixels each way.
    binning: f64 = 1,
    /// What still works on the patterns as stored once they are reduced:
    ///   none    nothing: the fits, the search, the refinement and the scores use the reduced size
    ///   scores  the final scores (NCC, NMI) and with them the choice of phase and acceptance
    ///   refine  the refinement too (and the pose fit, which is one)
    ///   all     everything: the reduced size is for looking at patterns only
    full_resolution: FullResolution = .none,

    // ── averaging with neighbours (before every filter) ──
    /// Non-local pattern averaging (NLPAR; Brewick, Wright and Rowenhorst 2019): each pattern
    /// is replaced by a weighted mean of itself and its neighbours in the map, a neighbour
    /// weighing as much as it looks like the same pattern under other noise.
    nlpar: bool = false,
    /// Neighbours out to this many map cells each way take part in a pass. One pass over 3
    /// cells each way (7 × 7) is the published method, and the default.
    nlpar_radius: u32 = 3,
    /// Passes. 1: the published method, a single averaging. More: each pass works on the
    /// patterns the one before left, with the weights decided anew from them (and from exactly
    /// how much noise is left in each: nlpar.wgsl). A pattern then reaches nlpar_radius ×
    /// nlpar_iterations cells (at most 8), but only through neighbours it resembles at every
    /// step, so a grain a cell or two wide is not averaged into the grains around it, as one
    /// pass over a wide window can do while the weights are still noisy: several passes of
    /// radius 1 in place of one of radius 3.
    nlpar_iterations: u32 = 1,
    /// How fast a neighbour's weight falls as it differs by more than noise explains:
    /// exp(−d / λ²), d counting standard deviations of what noise alone would give. Smaller:
    /// only near-identical neighbours count.
    nlpar_lambda: f64 = 0.7,

    // ── pattern filters (applied in this order) ──
    /// The static background, subtracted: the average of patterns spread over the whole map
    /// (`average`), or the one the file stores (`file`).
    static_background: Static = .average,
    /// How many patterns the average is taken over, on a lattice over the map (all of them
    /// when the map has fewer).
    static_patterns: u32 = 1024,
    /// A Gaussian-blurred copy of each pattern as its own background; the blur's sigma is
    /// dynamic_sigma × the pattern's width.
    dynamic_background: Background = .subtract,
    dynamic_sigma: f64 = 0.15,
    /// Contrast-limited adaptive histogram equalization: clahe_grid × clahe_grid tiles, each
    /// equalized over clahe_bins levels with its histogram clipped at clahe_clip times the
    /// mean bin count.
    clahe: bool = false,
    clahe_grid: u32 = 4,
    clahe_bins: u32 = 64,
    clahe_clip: f64 = 40,
    /// Band pass: the difference of Gaussians of these sigmas, in pixels of the pattern as
    /// stored (so the band is the same on the detector at every binning).
    bandpass: bool = false,
    bp_fine: f64 = 1,
    bp_coarse: f64 = 8,
    /// Circular mask; its radius is mask_radius × half the pattern's smaller side.
    mask: bool = true,
    mask_radius: f64 = 1,

    // ── master patterns ──
    // The master is filtered as the patterns are, on the sphere: CLAHE and the band pass are
    // applied to it whenever they are applied to the patterns, at the sizes the patterns'
    // filters have when a detector pixel is given its angular width.
    /// Angular width of one stored detector pixel, degrees. 0: from the geometry, the pixel at
    /// the pattern centre: 1 / (pcz · pattern height) radians.
    pixel_angle_deg: f64 = 0,
    /// Gaussian smoothing of the master before anything is projected from it, as a sigma in
    /// pixels of the pattern being compared. A detector pixel averages the sphere over its own
    /// footprint; sampling a sharper master at pixel centres would alias; 0.3 is about the
    /// spread of a square pixel (1 / sqrt(12)). 0: none.
    master_smooth: f64 = 0.3,

    // ── spherical indexing ──
    /// Harmonic bandwidth L of the correlation when indexing: how fine a detail of the pattern
    /// on the sphere takes part. The orientation grid has the next power of two ≥ 2L − 1
    /// points per turn (64 for L ≤ 32, 128 for L ≤ 64, 256 for L ≤ 128), and the cost follows
    /// its cube.
    bandwidth: u32 = 64,

    // ── pattern-centre fit ──
    /// Bandwidth of the correlation in the pattern-centre vote (and for the pose fit's starting
    /// orientations). The patterns' surfaces sharpen with it: on the nickel scan the vote lands
    /// within 0.003 of the pattern centre at 64, within 0.002 at 96 at seven times the cost,
    /// and 0.02 to 0.04 off at 32.
    fit_bandwidth: u32 = 64,
    /// Patterns on a fit_grid × fit_grid lattice over the region of interest.
    fit_grid: u32 = 5,
    /// Candidates per axis and the search's half-widths (fractions of the pattern, Bruker
    /// convention), around the current pattern centre.
    pc_steps: u32 = 9,
    pc_span: f64 = 0.1,
    pc_steps_z: u32 = 5,
    pc_span_z: f64 = 0.06,
    /// How a grid's winner is chosen from each fit pattern's own best correlation under every
    /// candidate:
    ///   ballots  every pattern casts a ballot at each of its `pc_top_k` best candidates (those
    ///            scoring at least `pc_gate` × its own best), the ballots are spread by a
    ///            Gaussian of sigma `pc_kde_sigma` (a fraction of the pattern, like the pattern
    ///            centre: the same width on every grid, and never under 0.75 of a candidate
    ///            step), and the densest place wins: where the most patterns agree, whatever
    ///            the patterns that disagree score elsewhere
    ///   mean     the candidate whose scores, averaged over the patterns, are highest
    pc_vote: PcVote = .ballots,
    pc_top_k: u32 = 12,
    pc_gate: f64 = 0.85,
    pc_kde_sigma: f64 = 0.015,
    /// The winner is then refined once to the weighted centre of the ballots within this
    /// distance of it (a fraction of the pattern), and found again among those alone.
    pc_inlier_radius: f64 = 0.04,
    /// A ballot only where a pattern's score is the highest within a cube of this many
    /// candidates a side (non-maximum suppression); 0 or 1: any of its best candidates.
    pc_nms: u32 = 0,
    /// The nearest a candidate may put the detector to the source (z, in pattern heights). Under
    /// it a candidate gets no score: as z falls toward 0 the detector fills the whole
    /// hemisphere, the window's normalization loses its footing, and a search let loose there
    /// runs away after scores that are not correlations.
    pc_z_min: f64 = 0.1,
    /// Searches, each centred on the last one's best with spans × pc_shrink.
    pc_levels: u32 = 3,
    pc_shrink: f64 = 0.35,

    // ── pose fit (detector pose and the fit patterns' orientations together) ──
    /// Patterns on a pose_grid × pose_grid lattice over the region of interest: more than the
    /// vote's, because a pattern costs the pose fit little, and what tells the detector's
    /// rotation from a common turn of the crystals is weak in any one pattern. On a simulated
    /// map whose beam travels 4 % of the detector, under noise as strong as the patterns, the
    /// fitted rotation (and with it every orientation) is off by 1.3° from 25 patterns and by
    /// 0.25° from 225.
    pose_grid: u32 = 15,
    pose_iters: u32 = 30,
    /// Which of the detector's six degrees of freedom the pose fit moves, each named in the
    /// detector's own axes; one that is off keeps its value. The position is the
    /// pattern centre's three coordinates:
    pose_x: bool = true,
    pose_y: bool = true,
    pose_z: bool = true,
    /// The attitude is three turns about the source point, which leave the pattern centre
    /// where it is (geometry.rot): tilt about the detector's column direction (the detector
    /// tilt's sense), swing about its row direction, twist about its normal. What tells them
    /// from a common turn of the crystals is the beam's travel across the map; with no pixel
    /// size there is none to go by and they are left as set.
    pose_tilt: bool = true,
    pose_swing: bool = true,
    pose_twist: bool = true,
    /// A seventh: the scale between the beam's travel over the sample and the detector, which
    /// makes the pose a similarity (Sim(3)). It is the detector's pixel size (geometry.px_um),
    /// which the fit then changes; it needs a pixel size to start from.
    pose_scale: bool = false,

    // ── orientation refinement on the detector ──
    refine: bool = true,
    refine_iters: u32 = 8,
    /// Largest rotation one iteration may take, degrees.
    refine_step: f64 = 2,

    // ── phases ──
    /// How the phase is chosen between each phase's best orientation.
    phase_by: PhaseBy = .nmi,
    nmi_bins: u32 = 16,

    // ── maps ──
    /// Inverse pole figure colours (sym.Colors): `classic`; `hue`, white at the sector's centre
    /// and the hue at an even rate round it; `perfect`, the keys of Nolze and Hielscher (2016)
    /// for the Laue class; `perfect_rotation`, theirs for the class's proper rotations, which
    /// tell a direction from its opposite.
    ipf_colors: IpfColors = .classic,
    /// The perfect keys' two gradients (the paper's equations 9 and 10). `ipf_lightness` is
    /// λ_L: 1 makes the lightness linear in the distance from a sector's centre, less widens
    /// the light (and dark) centre. `ipf_gray` is λ_S: 0 leaves the centres white and black,
    /// more turns them toward gray, at 1 the same gray. The paper recommends a quarter for both.
    ipf_lightness: f64 = 0.25,
    ipf_gray: f64 = 0.25,

    // ── indexing ──
    /// How each pattern's orientation is first found, before refinement:
    ///   spherical   the correlation over all orientations, each pattern from its own source
    ///               point
    ///   dictionary  inner products with the simulated patterns of a grid of orientations,
    ///               pixel by pixel, simulated for the source point at the map's centre
    ///   pca         the same dictionary kept as its leading components (learned by Oja's rule
    ///               as the dictionary is simulated once) and matched in those
    method: Method = .spherical,
    /// Dictionary: orientations sampled on a cubochoric grid of this many points per half edge
    /// (the mean spacing is about 45° / dict_grid for cubic crystals' zone).
    dict_grid: u32 = 30,
    /// Patterns read and indexed per step.
    batch: u32 = 256,

    // ── how the work is cut up (speed only: the orientations found do not depend on these) ──
    // The defaults suit every GPU reasonably. `tune` measures this machine and this scan and
    // sets `batch` above, `refine_pixels` and the two workgroup sizes to what runs fastest.
    /// Memory the correlation's spectra may take per dispatch, MiB: how many patterns of a
    /// batch are correlated at once.
    tile_mb: u32 = 256,
    /// Pixels of a pattern that one GPU invocation adds up in refinement. Fewer: more
    /// invocations at once, and more partial sums to gather.
    refine_pixels: u32 = 64,
    /// Invocations per workgroup of the kernels built on the line FFT (the correlation's
    /// spectrum and transforms) and of refinement's sums: powers of two from 16 to 256. Each
    /// other value is one more compile of those kernels on this machine.
    wg_fft: u32 = 64,
    wg_refine: u32 = 64,

    // ── sweeps ──
    /// Passes over the region. With more than one, the first is cheap (a low bandwidth or a
    /// coarse dictionary) and each later one costs more and takes only the patterns no earlier
    /// one indexed acceptably; the last runs at `bandwidth` / `dict_grid`. 1: one pass.
    sweeps: u32 = 1,
    /// The first sweep's bandwidth and dictionary grid; the ones between it and the last are
    /// spaced by equal ratios.
    sweep_bandwidth: u32 = 32,
    sweep_dict_grid: u32 = 20,
    /// What the sweeps before the last search with (`same`: as the last does, `method`). A
    /// cheap search first and the spherical correlation for what it leaves is the usual mix.
    sweep_method: SweepMethod = .same,
    /// Components kept of a dictionary for `pca`, and passes of Oja's rule over the dictionary
    /// to learn them.
    pca_components: u32 = 256,
    pca_epochs: u32 = 2,
    /// A match is accepted, and its pattern left out of later sweeps, when its normalized mutual
    /// information with its simulation is at least `accept_nmi`: an absolute threshold, the
    /// same for every pattern. 0: the threshold comes from a calibration instead, made once for
    /// the scan and also the same for every pattern: a sample of the region's patterns is
    /// scored against simulations at random orientations, and the threshold is that chance
    /// level plus `accept_z` times its spread (the log says what NMI that is). z in the results
    /// is the NMI on that scale.
    accept_nmi: f64 = 0,
    accept_z: f64 = 15,
    /// … and when its refinement had settled: when the last step that improved it turned the
    /// crystal by no more than this, degrees. A match whose iterations ran out on the way is
    /// left for the next sweep, which starts it closer.
    accept_step_deg: f64 = 0.2,

    /// Apply a JSON object of overrides. Unknown keys are an error (so typos are caught).
    pub fn apply(self: *Settings, gpa: std.mem.Allocator, json: []const u8) !void {
        const parsed = try std.json.parseFromSlice(std.json.Value, gpa, json, .{});
        defer parsed.deinit();
        const obj = switch (parsed.value) {
            .object => |o| o,
            else => return error.ExpectedObject,
        };
        var next = self.*;
        var it = obj.iterator();
        outer: while (it.next()) |kv| {
            inline for (@typeInfo(Settings).@"struct".fields) |f| {
                if (std.mem.eql(u8, kv.key_ptr.*, f.name)) {
                    @field(next, f.name) = try std.json.parseFromValueLeaky(f.type, gpa, kv.value_ptr.*, .{});
                    continue :outer;
                }
            }
            return error.UnknownSetting;
        }
        try next.check();
        self.* = next;
    }

    /// Reject values the pipeline cannot run with (rather than clamping them silently).
    pub fn check(self: *const Settings) !void {
        if (!(self.binning >= 1 and self.binning <= 64)) return error.BadSetting;
        if (self.bandwidth < 8 or self.bandwidth > 256 or self.fit_bandwidth < 8 or self.fit_bandwidth > 256) return error.BadSetting;
        if (self.clahe_grid < 1 or self.clahe_grid > 16 or self.clahe_bins < 2 or self.clahe_bins > 256 or !(self.clahe_clip > 0)) return error.BadSetting;
        if (self.fit_grid < 1 or self.fit_grid > 31 or self.pose_grid < 1 or self.pose_grid > 63) return error.BadSetting;
        if (self.pc_steps < 1 or self.pc_steps > 65 or self.pc_steps_z < 1 or self.pc_steps_z > 65) return error.BadSetting;
        if (self.pc_levels < 1 or self.pc_levels > 8) return error.BadSetting;
        if (!(self.ipf_lightness >= 0 and self.ipf_lightness <= 1) or !(self.ipf_gray >= 0 and self.ipf_gray <= 1)) return error.BadSetting;
        if (!(self.pc_z_min >= 0)) return error.BadSetting;
        if (self.toy_rows < 1 or self.toy_rows > 4096 or self.toy_cols < 1 or self.toy_cols > 4096 or self.toy_grains < 1) return error.BadSetting;
        if (self.toy_height < 8 or self.toy_height > 1024 or self.toy_width < 8 or self.toy_width > 1024) return error.BadSetting;
        if (!(self.toy_spread_deg >= 0) or !(self.toy_blank >= 0 and self.toy_blank <= 1) or !(self.toy_contrast > 0 and self.toy_contrast <= 1) or !(self.toy_background_sigma >= 0)) return error.BadSetting;
        if (!(self.toy_counts >= 0 and self.toy_counts <= 1e6)) return error.BadSetting;
        if (self.pc_top_k < 1 or self.pc_top_k > 256 or !(self.pc_gate >= 0 and self.pc_gate <= 1) or !(self.pc_kde_sigma > 0 and self.pc_kde_sigma <= 1) or !(self.pc_inlier_radius > 0) or self.pc_nms > 9) return error.BadSetting;
        if (self.nmi_bins < 2 or self.nmi_bins > 32) return error.BadSetting;
        if (self.batch < 1 or self.batch > 4096) return error.BadSetting;
        for ([_]u32{ self.wg_fft, self.wg_refine }) |wg| {
            if (wg < 16 or wg > 256 or !std.math.isPowerOfTwo(wg)) return error.BadSetting;
        }
        if (self.tile_mb < 8 or self.tile_mb > 8192 or self.refine_pixels < 8 or self.refine_pixels > 65536) return error.BadSetting;
        if (self.dict_grid < 2 or self.dict_grid > 200) return error.BadSetting;
        if (self.sweeps < 1 or self.sweeps > 6) return error.BadSetting;
        if (self.pca_components < 4 or self.pca_components > 1024 or self.pca_epochs < 1 or self.pca_epochs > 16) return error.BadSetting;
        if (self.sweep_bandwidth < 8 or self.sweep_bandwidth > 256 or self.sweep_dict_grid < 2 or self.sweep_dict_grid > 200) return error.BadSetting;
        if (self.static_patterns < 1) return error.BadSetting;
        if (self.nlpar_radius < 1 or self.nlpar_iterations < 1 or self.nlpar_radius * self.nlpar_iterations > 8 or !(self.nlpar_lambda > 0)) return error.BadSetting;
        if (@abs(self.scan_x_sign) != 1 or @abs(self.scan_y_sign) != 1) return error.BadSetting;
        if (!(self.mask_radius > 0) or !(self.dynamic_sigma > 0) or !(self.bp_fine >= 0) or !(self.bp_coarse > self.bp_fine)) return error.BadSetting;
        if (!(self.pixel_angle_deg >= 0) or !(self.master_smooth >= 0) or !(self.accept_step_deg > 0) or !(self.accept_nmi >= 0 and self.accept_nmi <= 1)) return error.BadSetting;
    }

    /// Bandwidth and dictionary grid of sweep `i` of `sweeps`: the first sweep's values, the
    /// last's (`bandwidth`, `dict_grid`), and equal ratios between.
    pub fn sweep(self: *const Settings, i: u32) struct { bandwidth: u32, dict_grid: u32, method: Method } {
        if (self.sweeps <= 1 or i + 1 >= self.sweeps) return .{ .bandwidth = self.bandwidth, .dict_grid = self.dict_grid, .method = self.method };
        const method: Method = switch (self.sweep_method) {
            .same => self.method,
            .spherical => .spherical,
            .dictionary => .dictionary,
            .pca => .pca,
        };
        const f = @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(self.sweeps - 1));
        const between = struct {
            fn of(a: u32, b: u32, t: f64) u32 {
                const x: f64 = @floatFromInt(a);
                const y: f64 = @floatFromInt(b);
                return @intFromFloat(@round(x * std.math.pow(f64, y / x, t)));
            }
        }.of;
        return .{ .bandwidth = between(self.sweep_bandwidth, self.bandwidth, f), .dict_grid = between(self.sweep_dict_grid, self.dict_grid, f), .method = method };
    }

    /// The settings as JSON (caller frees).
    pub fn toJson(self: *const Settings, gpa: std.mem.Allocator) ![]u8 {
        var out: std.Io.Writer.Allocating = .init(gpa);
        errdefer out.deinit();
        try std.json.Stringify.value(self.*, .{}, &out.writer);
        return out.toOwnedSlice();
    }
};

test "apply" {
    var s: Settings = .{};
    try s.apply(std.testing.allocator, "{\"bandwidth\":88,\"clahe\":true,\"method\":\"dictionary\",\"static_background\":\"file\",\"binning\":2.5}");
    try std.testing.expectEqual(@as(u32, 88), s.bandwidth);
    try std.testing.expect(s.clahe);
    try std.testing.expectEqual(Method.dictionary, s.method);
    try std.testing.expectEqual(Static.file, s.static_background);
    try std.testing.expectEqual(@as(f64, 2.5), s.binning);
    try std.testing.expectError(error.UnknownSetting, s.apply(std.testing.allocator, "{\"nope\":1}"));
    // a rejected value leaves the settings as they were
    try std.testing.expectError(error.BadSetting, s.apply(std.testing.allocator, "{\"bandwidth\":4,\"binning\":2}"));
    try std.testing.expectEqual(@as(f64, 2.5), s.binning);
    const j = try s.toJson(std.testing.allocator);
    defer std.testing.allocator.free(j);
    try std.testing.expect(std.mem.indexOf(u8, j, "\"method\":\"dictionary\"") != null);
}

test "sweeps climb to the final bandwidth and dictionary" {
    var s: Settings = .{ .sweeps = 3, .sweep_bandwidth = 32, .bandwidth = 128, .sweep_dict_grid = 10, .dict_grid = 40 };
    try std.testing.expectEqual(@as(u32, 32), s.sweep(0).bandwidth);
    try std.testing.expectEqual(@as(u32, 64), s.sweep(1).bandwidth);
    try std.testing.expectEqual(@as(u32, 128), s.sweep(2).bandwidth);
    try std.testing.expectEqual(@as(u32, 20), s.sweep(1).dict_grid);
    s.sweeps = 1;
    try std.testing.expectEqual(@as(u32, 128), s.sweep(0).bandwidth);
}
