//! The engine's C ABI, shared by the native library (Python, ctypes) and the browser module
//! (web/zebsdsolve.js). Root-specific functions (creation, allocation) live in native_api.zig
//! and web_api.zig. Every call returns 0 (or a count or size) on success and a negative code on
//! failure; ze_error says why. Calls that wait for the GPU or read the scan block natively and
//! suspend in the browser.
//!
//! Result structs are mirrored in web/zebsdsolve.js (byte offsets) and python/zebsdsolve/
//! _native.py (ctypes.Structure): append fields at the end and update all three together.
const std = @import("std");
const engine = @import("engine.zig");
const gpu_mod = @import("gpu/gpu.zig");
const scan_mod = @import("scan.zig");
const sym = @import("sym.zig");
const rot = @import("rot.zig");
const pcfit = @import("pcfit.zig");
const posefit = @import("posefit.zig");
const output = @import("output.zig");

const Engine = engine.Engine;

fn failed(e: *Engine, what: []const u8, err: anyerror) i32 {
    // the HDF5 reader says which feature it met and does not cover
    if (err == error.UnsupportedHdf5 and scan_mod.last_why.len != 0) {
        e.fail("{s}: this HDF5 file uses a feature the reader does not cover: {s}", .{ what, scan_mod.last_why });
    } else e.fail("{s}: {s}", .{ what, @errorName(err) });
    return -1;
}

export fn ze_adapter(e: *Engine, buf: [*]u8, cap: usize) usize {
    const s = e.dev.info();
    const n = @min(s.len, cap);
    @memcpy(buf[0..n], s[0..n]);
    return n;
}

/// The last error of this engine (or the GPU) into buf; returns its length.
export fn ze_error(e: *Engine, buf: [*]u8, cap: usize) usize {
    return e.takeError(buf[0..cap]);
}

/// Apply settings overrides (a JSON object; see settings.zig).
export fn ze_configure(e: *Engine, json: [*]const u8, len: usize) i32 {
    e.configure(json[0..len]) catch |err| return failed(e, "configure", err);
    return 0;
}

/// The current settings as JSON into buf; returns the full length (retry with a larger buf if
/// it exceeds cap).
export fn ze_settings(e: *Engine, buf: [*]u8, cap: usize) usize {
    const j = e.set.toJson(e.gpa) catch return 0;
    defer e.gpa.free(j);
    @memcpy(buf[0..@min(cap, j.len)], j[0..@min(cap, j.len)]);
    return j.len;
}

/// Queued progress events (native), newline-separated JSON; returns the bytes copied.
export fn ze_events(e: *Engine, buf: [*]u8, cap: usize) usize {
    return e.events.take(buf[0..cap]);
}

/// Ask the running fit or indexing to stop at its next step (it keeps what it has).
export fn ze_cancel(e: *Engine) void {
    e.cancel = true;
}

// ── scan ──

/// Open a scan: `paths` holds one or more names separated by '\n', the pattern file first.
export fn ze_open_scan(e: *Engine, paths: [*]const u8, len: usize) i32 {
    var list: [8][]const u8 = undefined;
    var n: usize = 0;
    var it = std.mem.tokenizeScalar(u8, paths[0..len], '\n');
    while (it.next()) |p| {
        if (n == list.len) break;
        list[n] = p;
        n += 1;
    }
    e.openScan(list[0..n]) catch |err| return failed(e, "open_scan", err);
    return 0;
}

/// Take rows · cols patterns of h × w pixels from memory (pixel: 0 u8, 1 u16, 2 f32), row-major.
export fn ze_set_patterns(e: *Engine, data: [*]const u8, len: usize, pixel: u32, rows: u32, cols: u32, h: u32, w: u32) i32 {
    if (pixel > 2) return failed(e, "set_patterns", error.UnsupportedPixelType);
    e.setPatterns(data[0..len], @enumFromInt(pixel), rows, cols, h, w) catch |err| return failed(e, "set_patterns", err);
    return 0;
}

/// Make a toy scan from the phases loaded (settings toy_*), in place of any scan open: grains
/// with random orientations and phases, simulated under the present geometry (which is so its
/// true geometry), laid on a background and counted as electrons. The answer goes to `quats`
/// (4 f32 per cell), `phase` (i32 per cell; −1: a blank cell) and `grain` (i32 per cell); n is
/// toy_rows · toy_cols.
export fn ze_synthesize(e: *Engine, quats: [*]f32, phase: [*]i32, grain: [*]i32, n: usize) i32 {
    e.cancel = false;
    e.synthesize(quats[0 .. 4 * n], phase[0..n], grain[0..n]) catch |err| return failed(e, "synthesize", err);
    return 0;
}

export fn ze_close_scan(e: *Engine) void {
    e.closeScan();
}

export fn ze_scan_info(e: *Engine, out: *engine.ScanInfo) void {
    out.* = e.scanInfo();
}

/// The scan's format name into buf; returns its length.
export fn ze_scan_format(e: *Engine, buf: [*]u8, cap: usize) usize {
    const s: []const u8 = switch (e.source) {
        .file => |f| f.formatName(),
        .memory => "memory",
        .none => "",
    };
    const n = @min(s.len, cap);
    @memcpy(buf[0..n], s[0..n]);
    return n;
}

export fn ze_get_geometry(e: *Engine, out: *engine.Geometry) void {
    out.* = e.geo;
}

export fn ze_set_geometry(e: *Engine, in: *const engine.Geometry) void {
    e.geo = in.*;
}

// ── phases ──

/// Add a candidate phase from a master pattern file; returns its index.
export fn ze_add_phase(e: *Engine, path: [*]const u8, len: usize) i32 {
    const i = e.addPhase(path[0..len]) catch |err| return failed(e, "add_phase", err);
    return @intCast(i);
}

export fn ze_clear_phases(e: *Engine) void {
    e.clearPhases();
}

export fn ze_phase_count(e: *Engine) u32 {
    return @intCast(e.phases.items.len);
}

export fn ze_phase_info(e: *Engine, index: u32, out: *engine.PhaseInfo) i32 {
    out.* = e.phaseInfo(index) catch |err| return failed(e, "phase_info", err);
    return 0;
}

/// The phase's master after the master filters: [north | south], side × side f32 each.
export fn ze_master_image(e: *Engine, phase: u32, out: [*]f32, len: usize) i32 {
    e.masterImage(phase, out[0..len]) catch |err| return failed(e, "master_image", err);
    return 0;
}

// ── patterns ──

/// (height, width) of a pattern: reduced by settings.binning, or (`full`) as stored.
export fn ze_pattern_shape(e: *Engine, full: u32, out: *[2]u32) i32 {
    out.* = e.patternShape(e.viewLevel(full != 0)) catch |err| return failed(e, "pattern_shape", err);
    return 0;
}

/// One pattern (map cell index, row-major) as f32, reduced or (`full`) at stored size, at a
/// stage of the pipeline: `processed` 0 as it is stored, 1 as it is indexed (averaged with its
/// neighbours when settings.nlpar, then filtered), 2 averaged with its neighbours and not
/// filtered.
export fn ze_pattern(e: *Engine, cell: u64, processed: u32, full: u32, out: [*]f32, len: usize) i32 {
    if (processed > 2) return failed(e, "pattern", error.UnknownKind);
    e.pattern(cell, @enumFromInt(processed), e.viewLevel(full != 0), out[0..len]) catch |err| return failed(e, "pattern", err);
    return 0;
}

/// What averaging with its neighbours did to the pattern ze_pattern last returned: its noise
/// as stored (standard deviation a pixel), the share of that noise the average keeps and the
/// share of the average that is the pattern itself into `out`; the average's shares of the
/// stored patterns around it into `shares`, side × side, row-major, the pattern itself in the
/// middle (NaN outside the map). Returns side (written when `cap` suffices); 0 when the
/// pattern was not averaged.
export fn ze_nlpar_view(e: *Engine, out: *[3]f32, shares: ?[*]f32, cap: usize) u32 {
    const side = e.nl_view_side;
    if (side == 0) return 0;
    out.* = .{ @floatCast(e.nl_view.noise), @floatCast(e.nl_view.left), @floatCast(e.nl_view.own) };
    if (shares) |s| if (cap >= e.nl_view_shares.len) @memcpy(s[0..e.nl_view_shares.len], e.nl_view_shares);
    return side;
}

/// About `want` cells on a lattice over the whole map (the lattice the static background is
/// averaged over, and a preview's): returns how many, written when `cap` suffices.
export fn ze_lattice(e: *Engine, want: u32, out: [*]u64, cap: usize) i64 {
    const cells = e.lattice(want) catch |err| return failed(e, "lattice", err);
    defer e.gpa.free(cells);
    if (cap >= cells.len) @memcpy(out[0..cells.len], cells);
    return @intCast(cells.len);
}

/// Angular width of one stored detector pixel as the master filters take it, degrees
/// (settings.pixel_angle_deg, or from the geometry when that is 0); 0 without a scan.
export fn ze_pixel_angle(e: *Engine) f64 {
    return std.math.radiansToDegrees(e.pixelAngle(.full));
}

/// Mean and standard deviation of the stored patterns of n cells (2 f32 per cell).
export fn ze_pattern_stats(e: *Engine, cells: [*]const u64, n: usize, out: [*]f32, len: usize) i32 {
    e.patternStats(cells[0..n], out[0..len]) catch |err| return failed(e, "pattern_stats", err);
    return 0;
}

/// Simulated patterns of a phase: n cells with n orientations (4 f32 each: w, x, y, z), at the
/// reduced size or (`full`) the stored one.
export fn ze_simulate(e: *Engine, phase: u32, cells: [*]const u64, quats: [*]const f32, n: usize, full: u32, out: [*]f32, len: usize) i32 {
    e.simulate(phase, cells[0..n], quats[0 .. 4 * n], e.viewLevel(full != 0), out[0..len]) catch |err| return failed(e, "simulate", err);
    return 0;
}

// ── fits ──

/// Fit the pattern centre by the search of settings pc_*; the geometry takes the result.
export fn ze_fit_pc(e: *Engine, out: *pcfit.Result) i32 {
    out.* = pcfit.run(e) catch |err| return failed(e, "fit_pc", err);
    return 0;
}

/// The last search level's score volume: counts along (z, y, x) into `shape`, the candidates'
/// pattern-centre values into `axes` (x values, then y, then z), the scores (z-major) into
/// `scores`. Returns the number of scores, or what it would be when the buffers are too small.
export fn ze_pc_scores(e: *Engine, shape: *[3]u32, axes: [*]f64, axes_cap: usize, scores: [*]f32, cap: usize) i32 {
    const nx = e.pc_axes[0].len;
    const ny = e.pc_axes[1].len;
    const nz = e.pc_axes[2].len;
    shape.* = .{ @intCast(nz), @intCast(ny), @intCast(nx) };
    if (axes_cap >= nx + ny + nz and cap >= e.pc_scores.len) {
        @memcpy(axes[0..nx], e.pc_axes[0]);
        @memcpy(axes[nx..][0..ny], e.pc_axes[1]);
        @memcpy(axes[nx + ny ..][0..nz], e.pc_axes[2]);
        @memcpy(scores[0..e.pc_scores.len], e.pc_scores);
    }
    return @intCast(e.pc_scores.len);
}

/// How many grids the last pattern-centre search scored (its levels, and the moves of a grid
/// whose winner was on its border).
export fn ze_pc_level_count(e: *Engine) u32 {
    return @intCast(e.pc_history.items.len);
}

/// Grid `i` of the last search: counts along (z, y, x) into `shape`; its candidates' values
/// (x, then y, then z) into `axes`; the mean vote per candidate (z-major) into `scores`; per
/// fit pattern, the candidate it scored highest under into `pat_best` and its score there into
/// `pat_ncc`; (winner's index, 1 if the grid then moved onto it) into `info`. Returns the number
/// of scores (written when every capacity suffices), or −1 past the last grid.
export fn ze_pc_level(e: *Engine, i: u32, shape: *[3]u32, info: *[2]u32, axes: ?[*]f64, axes_cap: usize, scores: ?[*]f32, cap: usize, pat_best: ?[*]u32, pat_ncc: ?[*]f32, pat_cap: usize) i32 {
    if (i >= e.pc_history.items.len) return -1;
    const l = &e.pc_history.items[i];
    const nx = l.axes[0].len;
    const ny = l.axes[1].len;
    const nz = l.axes[2].len;
    shape.* = .{ @intCast(nz), @intCast(ny), @intCast(nx) };
    info.* = .{ l.best, l.walked };
    if (axes_cap >= nx + ny + nz and cap >= l.scores.len and pat_cap >= l.pat_best.len) {
        @memcpy(axes.?[0..nx], l.axes[0]);
        @memcpy(axes.?[nx..][0..ny], l.axes[1]);
        @memcpy(axes.?[nx + ny ..][0..nz], l.axes[2]);
        @memcpy(scores.?[0..l.scores.len], l.scores);
        @memcpy(pat_best.?[0..l.pat_best.len], l.pat_best);
        @memcpy(pat_ncc.?[0..l.pat_ncc.len], l.pat_ncc);
    }
    return @intCast(l.scores.len);
}

/// Grid `i` of the last search, pattern by pattern: every fit pattern's own best correlation
/// under every candidate (pattern-major, each pattern's candidates z-major as the mean votes
/// are). Returns the number of values (written when `cap` suffices), or −1 past the last grid.
export fn ze_pc_level_scores(e: *Engine, i: u32, out: ?[*]f32, cap: usize) i32 {
    if (i >= e.pc_history.items.len) return -1;
    const s = e.pc_history.items[i].pat_scores;
    if (out) |o| if (cap >= s.len) @memcpy(o[0..s.len], s);
    return @intCast(s.len);
}

/// Grid `i` of the last search as the ballots decided it (settings.pc_vote): the density of
/// the patterns' ballots per candidate (z-major) into `out`, and into `info` where the grid's
/// choice lies in candidate steps along x, y, z, the ballots cast and those within the inlier
/// radius of the choice. Returns the number of densities (0 when the mean vote decided; `info`
/// is filled either way), or −1 past the last grid.
export fn ze_pc_level_ballots(e: *Engine, i: u32, out: ?[*]f32, cap: usize, info: *[5]f64) i32 {
    if (i >= e.pc_history.items.len) return -1;
    const l = &e.pc_history.items[i];
    info.* = .{ l.peak[0], l.peak[1], l.peak[2], @floatFromInt(l.ballots), @floatFromInt(l.inliers) };
    if (out) |o| if (cap >= l.density.len) @memcpy(o[0..l.density.len], l.density);
    return @intCast(l.density.len);
}

/// Fit the detector's pose and the fit patterns' orientations together; the geometry takes the
/// result.
export fn ze_fit_pose(e: *Engine, out: *posefit.Result) i32 {
    out.* = posefit.run(e) catch |err| return failed(e, "fit_pose", err);
    return 0;
}

/// The fit patterns of the last fit: their cells, phases, orientations (4 f32) and scores.
/// Returns how many there are (copied when every capacity suffices).
export fn ze_fit_patterns(e: *Engine, cells: [*]u64, phases: [*]i32, quats: [*]f32, ncc: [*]f32, cap: usize) i32 {
    const n = e.fit_cells.len;
    if (cap >= n and e.fit_quats.len == 4 * n) {
        @memcpy(cells[0..n], e.fit_cells);
        @memcpy(phases[0..n], e.fit_phase);
        @memcpy(quats[0 .. 4 * n], e.fit_quats);
        @memcpy(ncc[0..n], e.fit_ncc);
    }
    return @intCast(n);
}

// ── indexing ──

/// Start indexing the region of interest (settings roi_*; the whole map by default).
export fn ze_index_begin(e: *Engine) i32 {
    e.indexBegin() catch |err| return failed(e, "index_begin", err);
    return 0;
}

/// Index the next batch of the sweep in progress. Returns the number of cells left to visit
/// (0: done; a sweep that ends hands the cells it did not accept to the next), or a negative
/// code. ze_index_status says where it stands.
export fn ze_index_step(e: *Engine) i64 {
    const left = e.indexStep() catch |err| return failed(e, "index_step", err);
    return @intCast(left);
}

/// Tune how the work is cut up (settings batch, refine_pixels, wg_refine, wg_fft) to this
/// machine and scan. The engine has no clock, so the caller times the trials: ze_tune_begin,
/// then ze_tune_trial (time it) and ze_tune_report(seconds, or a negative number if the call
/// failed) until the report returns 0; the settings then hold the fastest values. The map's
/// results are not touched.
export fn ze_tune_begin(e: *Engine) i32 {
    e.tuneBegin() catch |err| return failed(e, "tune", err);
    return 0;
}

export fn ze_tune_trial(e: *Engine) i32 {
    e.tuneTrial() catch |err| return failed(e, "tune", err);
    return 0;
}

export fn ze_tune_report(e: *Engine, seconds: f64) i32 {
    return @intFromBool(e.tuneReport(seconds));
}

/// What mutual information chance gives on this scan for a phase (measured now if what the
/// scores depend on has changed): its median and spread into `out`. Every match's z is its
/// NMI on that scale, and with settings.accept_nmi 0 a match is accepted at
/// out[0] + accept_z · out[1].
export fn ze_chance(e: *Engine, phase: u32, out: *[2]f32) i32 {
    e.calibrate() catch |err| return failed(e, "chance", err);
    if (phase >= e.chance_med.len) return failed(e, "chance", error.NoSuchPhase);
    out.* = .{ e.chance_med[phase], e.chance_spread[phase] };
    return 0;
}

export fn ze_index_status(e: *Engine, out: *engine.IndexStatus) void {
    out.* = e.indexStatus();
}

/// What sweep `i` (0-based) of settings.sweeps searches with: (bandwidth, dictionary grid,
/// method in settings.Method's order).
export fn ze_sweep(e: *Engine, i: u32, out: *[3]u32) void {
    const s = e.set.sweep(i);
    out.* = .{ s.bandwidth, s.dict_grid, @intFromEnum(s.method) };
}

/// Index given cells without touching the stored results: one record per phase per cell
/// (8 f32 each: orientation w x y z, NCC, NMI, the size of the last refinement step that
/// improved the match in radians, and the NMI's distance above chance in chance's standard
/// deviations), phase-major.
export fn ze_index_cells(e: *Engine, cells: [*]const u64, n: usize, out: [*]f32, len: usize) i32 {
    const n_ph = e.phases.items.len;
    if (len != n_ph * n * 8) return failed(e, "index_cells", error.BadLength);
    const recs: []engine.Record = @as([*]engine.Record, @ptrCast(@alignCast(out)))[0 .. n_ph * n];
    e.indexCells(cells[0..n], e.searchOf(), recs) catch |err| return failed(e, "index_cells", err);
    return 0;
}

/// ze_index_cells with the search given instead of the settings': `method` in
/// settings.Method's order, the spherical correlation's bandwidth, the dictionary's grid.
export fn ze_index_cells_by(e: *Engine, cells: [*]const u64, n: usize, method: u32, bandwidth: u32, dict_grid: u32, out: [*]f32, len: usize) i32 {
    const n_ph = e.phases.items.len;
    if (len != n_ph * n * 8) return failed(e, "index_cells", error.BadLength);
    if (method > 2 or bandwidth < 8 or bandwidth > 256 or dict_grid < 2 or dict_grid > 200) return failed(e, "index_cells", error.BadSetting);
    const recs: []engine.Record = @as([*]engine.Record, @ptrCast(@alignCast(out)))[0 .. n_ph * n];
    e.indexCells(cells[0..n], .{ .method = @enumFromInt(method), .bandwidth = bandwidth, .dict_grid = dict_grid }, recs) catch |err| return failed(e, "index_cells", err);
    return 0;
}

/// Results per map cell (row-major), by kind: 0 orientations (4 f32 per cell: w, x, y, z),
/// 1 NCC, 2 NMI (f32), 3 phase (i32, −1: not indexed), 4 Bunge angles (3 f32, radians: those of
/// the stored quaternions, each in its phase's fundamental zone),
/// 5 the NMI's distance above chance (f32, standard deviations), 6 the sweep that found an
/// accepted match (i32, −1: none). Returns the number of values written.
export fn ze_results(e: *Engine, kind: u32, out: [*]u8, len: usize) i64 {
    const n = e.res.ncc.len;
    const src: []const u8 = switch (kind) {
        0 => std.mem.sliceAsBytes(e.res.quat),
        1 => std.mem.sliceAsBytes(e.res.ncc),
        2 => std.mem.sliceAsBytes(e.res.nmi),
        3 => std.mem.sliceAsBytes(e.res.phase),
        5 => std.mem.sliceAsBytes(e.res.z),
        6 => std.mem.sliceAsBytes(e.res.sweep),
        4 => {
            if (len < n * 12) return failed(e, "results", error.BufferTooSmall);
            const o: []f32 = @as([*]f32, @ptrCast(@alignCast(out)))[0 .. 3 * n];
            for (0..n) |i| {
                const q = e.res.quat[4 * i ..][0..4];
                const ph = e.res.phase[i];
                if (ph < 0) {
                    for (0..3) |k| o[3 * i + k] = 0;
                    continue;
                }
                const b = rot.toBunge(.{ q[0], q[1], q[2], q[3] });
                for (0..3) |k| o[3 * i + k] = @floatCast(b[k]);
            }
            return @intCast(3 * n);
        },
        else => return failed(e, "results", error.UnknownKind),
    };
    if (len < src.len) return failed(e, "results", error.BufferTooSmall);
    @memcpy(out[0..src.len], src);
    return @intCast(src.len / 4);
}

/// A map as RGBA8, rows × cols: kind 0–2 inverse pole figure of sample x, y, z; 3 NCC;
/// 4 NMI; 5 phase; 6 the NMI's distance above chance; 7 the sweep that accepted each cell.
export fn ze_map(e: *Engine, kind: u32, out: [*]u8, len: usize) i32 {
    output.map(e, kind, out[0..len]) catch |err| return failed(e, "map", err);
    return 0;
}

/// The key to a phase's inverse pole figure colours, as the settings have them (ipf_colors,
/// ipf_lightness, ipf_gray), as RGBA8, size × size: the directions that stand for all the
/// others, in stereographic projection with the crystal's x to the right, transparent outside
/// them. `lower` = 0 gives their part in the upper hemisphere ([001] at the origin), 1 their
/// part in the lower one, which only some keys have (output.ipfKey). `box` receives the part's
/// bounding box in the projection (x0, y0, x1, y1; all zero where there is none), drawn from
/// the image's lower left corner at (size − 1) / its longer side pixels per unit, and then the
/// two azimuths, in degrees, of a wedge about the c axis (NaN for the cubic classes' polygons).
export fn ze_ipf_key(e: *Engine, phase: u32, size: u32, lower: u32, out: [*]u8, len: usize, box: *[6]f32) i32 {
    if (phase >= e.phases.items.len) return failed(e, "ipf_key", error.NoSuchPhase);
    const key = e.ipfKeyOf(phase);
    output.ipfKey(&key, lower != 0, size, out[0..len], box) catch |err| return failed(e, "ipf_key", err);
    return 0;
}

/// The results as a text file: kind 0 EDAX .ang, 1 HKL .ctf. Returns the size; the bytes are
/// written when cap suffices.
export fn ze_export(e: *Engine, kind: u32, buf: ?[*]u8, cap: usize) i64 {
    const text = output.exportText(e, kind) catch |err| return failed(e, "export", err);
    defer e.gpa.free(text);
    if (buf) |b| if (cap >= text.len) @memcpy(b[0..text.len], text);
    return @intCast(text.len);
}

/// An orientation as the results store one: `q` is replaced by its representative in the
/// phase's fundamental zone (the equivalent of the smallest rotation, among those the proper
/// rotations of the phase's point group give), and `euler` receives that quaternion's Bunge
/// angles, radians.
export fn ze_reduce(e: *Engine, phase: u32, q: *[4]f32, euler: *[3]f64) void {
    const laue = if (phase < e.phases.items.len) e.phases.items[phase].master.rotation else sym.Laue.c1;
    const r = sym.toFz(.{ q[0], q[1], q[2], q[3] }, laue);
    for (0..4) |k| q[k] = @floatCast(r[k]);
    euler.* = rot.toBunge(r);
}

/// Smallest rotation between two orientations under a phase's symmetry (the proper rotations
/// of its point group), radians.
export fn ze_disorientation(e: *Engine, phase: u32, a: *const [4]f32, b: *const [4]f32) f64 {
    const laue = if (phase < e.phases.items.len) e.phases.items[phase].master.rotation else sym.Laue.c1;
    return sym.disorientation(.{ a[0], a[1], a[2], a[3] }, .{ b[0], b[1], b[2], b[3] }, laue);
}

// ── tests and profiling ──

/// Build a compute pipeline from WGSL source (tests/check_shaders.py): 0 when it builds, else
/// the length of the compiler's message written to msg (at least 1). Native only (−1 on the web).
export fn ze_compile_wgsl(e: *Engine, code: [*]const u8, code_len: usize, entry: [*]const u8, entry_len: usize, msg: [*]u8, cap: usize) i32 {
    if (comptime gpu_mod.is_web) {
        return -1;
    } else {
        _ = e.g.dev.createPipeline(code[0..code_len], entry[0..entry_len]) catch {
            const m = e.g.dev.failMessage();
            const src = if (m.len > 0) m else "failed";
            const n = @min(src.len, cap);
            @memcpy(msg[0..n], src[0..n]);
            _ = e.g.dev.takeError();
            return @intCast(@max(n, 1));
        };
        return 0;
    }
}

/// The WGSL of a shader module by index (the same text the engine compiles); returns its
/// length, 0 past the last module. `name` receives the module's name.
export fn ze_shader(index: u32, name: [*]u8, name_cap: usize, buf: ?[*]u8, cap: usize) usize {
    const shaders = @import("shaders");
    inline for (shaders.modules, 0..) |m, i| {
        if (i == index) {
            const nm: []const u8 = m[0];
            const code: []const u8 = m[1];
            const k = @min(nm.len, name_cap - 1);
            @memcpy(name[0..k], nm[0..k]);
            name[k] = 0;
            if (buf) |b| if (cap >= code.len) @memcpy(b[0..code.len], code);
            return code.len;
        }
    }
    return 0;
}

/// GPU traffic since the last call: dispatches, submits, reads, waits (round trips).
export fn ze_gpu_counts(e: *Engine, out: *[4]u32) void {
    const c = e.g.countsTake();
    out.* = .{ c.dispatches, c.submits, c.reads, c.waits };
}

/// Time every dispatch on the GPU from now on (native; 0 when unavailable).
export fn ze_profile(e: *Engine, on: u32) i32 {
    return @intFromBool(e.g.setProfile(on != 0));
}

/// GPU time per pipeline since the last call, slowest first, as text; returns its length.
export fn ze_profile_report(e: *Engine, buf: [*]u8, cap: usize) usize {
    return e.g.profileReport(buf[0..cap]);
}
