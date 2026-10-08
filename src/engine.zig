//! The indexing engine both front ends drive. Operations are written as ordinary blocking
//! code: record the GPU schedule, queue the reads of its results, `wait`, compute the result.
//! Natively `wait` polls the device and a file read is a system call; in the browser the module
//! suspends at both (JS Promise Integration) while the adapter awaits the GPU or a file slice.
//! Same code, same schedule.
//!
//! State: a scan (a file read in ranges, or patterns handed over in memory), candidate phases
//! (master patterns), the detector geometry, the settings, and per map cell the result.
//!
//! Patterns exist at two resolutions (`Level`): reduced by settings.binning on the host as
//! they are read, and as stored. Each stage of indexing (the search, the refinement, the final
//! scores) works at the level settings.full_resolution gives it; everything a level needs (the
//! filtered patterns, the static background, each phase's filtered master) is kept per level.
//!
//! Indexing keeps the file and the device busy at once: between submitting a batch and waiting
//! for it, the host reads the next batch's patterns (into a cache by map cell) and reduces them
//! if a level is reduced. With settings.nlpar every pattern is first averaged with its
//! neighbours in the map, on the device, in `uploadCells`: a batch is then read with a margin
//! of cells around it, batches are square blocks of the map, and the cache holds what
//! neighbouring blocks share.
const std = @import("std");
const gpu_mod = @import("gpu/gpu.zig");
const rot = @import("rot.zig");
const sym = @import("sym.zig");
const sh = @import("sh.zig");
const scan_mod = @import("scan.zig");
const master_mod = @import("master.zig");
const geometry_mod = @import("geometry.zig");
const settings_mod = @import("settings.zig");
const events_mod = @import("events.zig");
const prep_mod = @import("prep.zig");
const reduce = @import("reduce.zig");
const spherical = @import("spherical.zig");
const detector = @import("detector.zig");
const pca_mod = @import("pca.zig");
const nlpar_mod = @import("nlpar.zig");

const Gpu = gpu_mod.Gpu;
const Buf = gpu_mod.Buf;
const Allocator = std.mem.Allocator;
pub const Settings = settings_mod.Settings;
pub const Geometry = geometry_mod.Geometry;

pub const version = "0.1.0";

/// A pattern resolution: reduced by settings.binning, or as stored.
pub const Level = enum(u1) { reduced = 0, full = 1 };
pub const Stage = enum { search, refine, score };

/// What the scan is, for the front ends (`ze_scan_info`).
pub const ScanInfo = extern struct {
    rows: u32 = 0,
    cols: u32 = 0,
    pat_h: u32 = 0,
    pat_w: u32 = 0,
    /// 0: u8, 1: u16, 2: f32
    pixel: u32 = 0,
    has_background: u32 = 0,
    /// 1 when the file gave a pattern centre
    has_pc: u32 = 0,
    /// index into scan.zig's Format, or 100 for patterns in memory
    format: u32 = 0,
    voltage_kv: f64 = 0,
    bytes: f64 = 0,
};

pub const PhaseInfo = extern struct {
    side: u32 = 0,
    point_group: u32 = 0,
    space_group: u32 = 0,
    laue: u32 = 0,
    /// harmonic selection rules at the bandwidth last used: rotation order about z, mirror, inversion
    n_fold: u32 = 0,
    mirror: u32 = 0,
    inversion: u32 = 0,
    pad: u32 = 0,
    voltage_kv: f64 = 0,
    name: [64]u8 = @splat(0),
};

/// Where indexing stands (`ze_index_status`).
pub const IndexStatus = extern struct {
    /// the sweep in progress (0-based) of `sweeps`, and what it searches with
    sweep: u32 = 0,
    sweeps: u32 = 1,
    bandwidth: u32 = 0,
    dict_grid: u32 = 0,
    /// patterns of this sweep done, and in it
    done: u64 = 0,
    total: u64 = 0,
    /// cells of the region, and how many have an accepted match so far
    region: u64 = 0,
    accepted: u64 = 0,
    /// the sweep's method (settings.Method's order)
    method: u32 = 0,
    pad: u32 = 0,
};

/// What a phase's dictionary components were learned for.
const PcaKey = struct {
    grid: u32,
    k: u32,
    epochs: u32,
    level: Level,
    serial: u32,
    h: u32,
    w: u32,
    mask: [3]f32,
    t: [3]f32,
    c: [9]f32,
};

/// What a phase's master was filtered for at one level.
const BakeKey = struct {
    /// a pattern pixel of that level on the master's sheets, in sheet pixels, as a step of a
    /// 5 % ladder (so a pattern centre that moves a little does not rebuild everything);
    /// i32 minimum: unknown (no scan), the master stays as loaded
    px_q: i32,
    /// in pattern pixels of that level
    smooth: f64,
    bandpass: bool,
    fine: f64,
    coarse: f64,
    clahe: bool,
    tiles: u32,
    bins: u32,
    clip: f64,
};

const RdenKey = struct {
    c: [9]f32,
    t: [3]f32,
    h: u32,
    w: u32,
    mask: [3]f32,
};

const BgKey = struct {
    kind: settings_mod.Static,
    n: u32,
    bin: f64,
};

/// A dictionary of one grid, with what was learned of it.
const Dictionary = struct {
    dict: []f32 = &.{},
    dict_gpu: Buf = .{},
    dict_grid: u32 = 0,
    pca_w: Buf = .{},
    pca_coef: Buf = .{},
    pca_key: ?PcaKey = null,

    fn release(self: *Dictionary, g: *Gpu) void {
        g.gpa.free(self.dict);
        g.release(&self.dict_gpu);
        g.release(&self.pca_w);
        g.release(&self.pca_coef);
        self.* = .{};
    }
};

pub const Phase = struct {
    master: master_mod.Master,
    /// the filtered master per level: on the device, on the host, what it was filtered for,
    /// and a count of its rebuilds (what was derived from it knows by that when it is stale)
    det: [2]detector.MasterGpu = .{ .{}, .{} },
    work: [2][]f32 = .{ &.{}, &.{} },
    baked: [2]?BakeKey = .{ null, null },
    serial: [2]u32 = .{ 0, 0 },
    /// dictionary orientations (4 f32 each) in the phase's fundamental zone, and on the device
    dict: []f32 = &.{},
    dict_gpu: Buf = .{},
    dict_grid: u32 = 0,
    /// the dictionary's leading components (k rows of pixels) and every entry's coefficients
    pca_w: Buf = .{},
    pca_coef: Buf = .{},
    pca_key: ?PcaKey = null,
    /// dictionaries of other grids (sweeps alternate between a few), set aside whole
    spare: [3]Dictionary = .{ .{}, .{}, .{} },
    spare_next: usize = 0,

    /// Make the dictionary of grid N the current one if it was set aside; the current one is
    /// set aside in its place (or over the oldest). False: none was kept, and the current
    /// fields are empty for a new one.
    fn takeDictionary(self: *Phase, g: *Gpu, N: u32) bool {
        var cur: Dictionary = .{ .dict = self.dict, .dict_gpu = self.dict_gpu, .dict_grid = self.dict_grid, .pca_w = self.pca_w, .pca_coef = self.pca_coef, .pca_key = self.pca_key };
        var found: ?Dictionary = null;
        var slot: usize = self.spare_next;
        for (&self.spare, 0..) |*d, i| {
            if (d.dict.len != 0 and d.dict_grid == N) {
                found = d.*;
                slot = i;
            }
        }
        if (found == null) {
            for (&self.spare, 0..) |*d, i| {
                if (d.dict.len == 0) slot = i;
            }
            self.spare[slot].release(g);
            self.spare_next = (slot + 1) % self.spare.len;
        }
        self.spare[slot] = if (cur.dict.len != 0) cur else .{};
        if (cur.dict.len == 0) cur.release(g);
        const now: Dictionary = found orelse .{};
        self.dict = now.dict;
        self.dict_gpu = now.dict_gpu;
        self.dict_grid = now.dict_grid;
        self.pca_w = now.pca_w;
        self.pca_coef = now.pca_coef;
        self.pca_key = now.pca_key;
        return found != null;
    }

    fn deinit(self: *Phase, g: *Gpu) void {
        for (&self.spare) |*d| d.release(g);
        g.release(&self.pca_w);
        g.release(&self.pca_coef);
        const gpa = g.gpa;
        for (0..2) |k| {
            g.release(&self.det[k].buf);
            g.release(&self.det[k].grad);
            gpa.free(self.work[k]);
        }
        gpa.free(self.dict);
        g.release(&self.dict_gpu);
        self.master.deinit();
    }
};

/// A phase's harmonic side at one bandwidth, from its master as filtered for `level`.
const BandPhase = struct {
    hm: master_mod.Harmonics,
    gpu: spherical.PhaseGpu,
    rden_key: ?RdenKey = null,
    level: Level,
    serial: u32,
};

/// What spherical indexing needs at one bandwidth: its tables and each phase's harmonic side.
/// Sweeps and the fits alternate between bandwidths, so a few are kept.
pub const Band = struct {
    L: u32,
    tables: spherical.Tables,
    ph: std.ArrayList(?BandPhase) = .empty,
    /// last use
    stamp: u64 = 0,

    fn dropPhase(self: *Band, g: *Gpu, pi: usize) void {
        if (self.ph.items[pi]) |*x| {
            x.hm.deinit(g.gpa);
            x.gpu.deinit(g);
        }
        self.ph.items[pi] = null;
    }

    fn destroy(self: *Band, g: *Gpu) void {
        for (0..self.ph.items.len) |pi| self.dropPhase(g, pi);
        self.ph.deinit(g.gpa);
        self.tables.deinit(g);
        g.gpa.destroy(self);
    }
};

const MAX_BANDS = 6;

const Memory = struct {
    data: []u8,
    info: scan_mod.Info,
};

/// Tuning how the work is cut up (settings.batch, refine_pixels, wg_refine, wg_fft) to this
/// machine and this scan. A sample of the region is indexed as indexing would index it, the
/// front end times that (the engine has no clock of its own), and a few moves are tried in
/// turn, each climbing from the values held: doubled while that is faster, else halved while
/// that is. About ten trials, not a search of every value. Nothing of the map's results is
/// touched, and neither the orientations indexing finds nor which matches are accepted
/// depend on these settings.
pub const Tune = struct {
    /// The settings tuned.
    pub const count = 4;
    pub const names = [count][]const u8{ "batch", "refine_pixels", "wg_refine", "wg_fft" };
    /// The moves, in the order they are tried: the settings each scales together.
    ///   refine_pixels          how many invocations a batch's refinement makes
    ///   batch, refine_pixels   the batch at that many invocations: the two are not
    ///                          independent, and the fastest pairs lie along this line
    ///   wg_refine              the refinement's workgroup
    ///   wg_fft                 the correlation's workgroup
    /// All but the last are timed on the first sweep, where most patterns go and reading,
    /// filtering and refinement weigh most; wg_fft shapes the correlation alone and is timed
    /// on the last sweep, at the full bandwidth.
    pub const moves = [_][count]bool{
        .{ false, true, false, false },
        .{ true, true, false, false },
        .{ false, false, true, false },
        .{ false, false, false, true },
    };
    pub const on_last = 3;
    /// How far each setting may climb.
    pub const lo = [count]u32{ 64, 16, 16, 16 };
    pub const hi = [count]u32{ 4096, 2048, 256, 256 };
    /// Timed runs of a trial (the faster counts). Before them one batch runs untimed: a
    /// workgroup size not used before is compiled in it, and a larger batch gets its buffers.
    pub const runs = 2;
    /// A move is kept only when it is faster by more than this share.
    pub const worth = 0.03;

    /// Patterns a trial indexes, by what the search costs, so that a trial stays a fraction of
    /// a second: the spherical correlation's cost follows the cube of its orientation grid
    /// (64 points a turn up to bandwidth 32, 128 up to 64, 256 up to 128, 512 beyond).
    pub fn sampleOf(method: settings_mod.Method, bandwidth: u32, dict_grid: u32) usize {
        return switch (method) {
            .spherical => if (bandwidth <= 32) 8192 else if (bandwidth <= 64) 4096 else if (bandwidth <= 128) 1024 else 128,
            .dictionary => if (dict_grid <= 30) 1024 else 256,
            .pca => 4096,
        };
    }

    /// the samples timed on the first sweep and on the last: rectangles of the region from
    /// its first cell
    cells: [2][]u64 = .{ &.{}, &.{} },
    /// the values held, their seconds on the sweep being timed, and what tuning began with
    best: [count]u32 = @splat(0),
    best_time: f64 = std.math.inf(f64),
    first: [count]u32 = @splat(0),
    /// seconds before and after, on the first sweep and on the last (0: not timed)
    was: [2]f64 = .{ 0, 0 },
    now: [2]f64 = .{ 0, 0 },
    /// the trial in hand: the move (`moves.len`: none is left), whether it times the values
    /// held (what a sweep's candidates must beat) or a candidate, that candidate, which way
    /// the move is climbing and whether a step that way was kept
    move: u32 = 0,
    base: bool = true,
    cand: [count]u32 = @splat(0),
    up: bool = true,
    kept: bool = false,
    /// whether the last sweep is the one timed; the run of the trial and its fastest so far
    last: bool = false,
    run: u32 = 0,
    time: f64 = std.math.inf(f64),
    trials: u32 = 0,
    active: bool = false,
};

/// A cell a read must fetch, and where in the read it was asked for.
const Miss = struct {
    cell: u64,
    at: usize,

    fn before(_: void, a: Miss, b: Miss) bool {
        return a.cell < b.cell;
    }
};

/// Stored patterns kept on the host by map cell, the oldest dropped first: the neighbours that
/// consecutive batches share when patterns are averaged with their neighbours (NLPAR). Only
/// then: a pattern that passes through here is copied twice more than one read straight into
/// its batch, and on a fast device those copies are what a batch waits for.
const PatternCache = struct {
    /// Bytes of patterns kept at most: enough for the rows of the map between a block and the
    /// block below it on a scan of ordinary size.
    const budget: usize = if (gpu_mod.is_web) 96 << 20 else 384 << 20;

    /// bytes of one pattern
    pb: usize = 0,
    /// the cell each slot holds; slots fill in order and then turn over from `next`
    cells: std.ArrayList(u64) = .empty,
    data: std.ArrayList(u8) = .empty,
    slot_of: std.AutoHashMapUnmanaged(u64, u32) = .empty,
    next: u32 = 0,

    fn deinit(self: *PatternCache, gpa: Allocator) void {
        self.cells.deinit(gpa);
        self.data.deinit(gpa);
        self.slot_of.deinit(gpa);
        self.* = .{};
    }

    /// Empty the cache for patterns of `pb` bytes.
    fn reset(self: *PatternCache, gpa: Allocator, pb: usize) void {
        self.deinit(gpa);
        self.pb = pb;
    }

    fn slots(self: *const PatternCache) usize {
        return if (self.pb == 0) 0 else budget / self.pb;
    }

    fn get(self: *const PatternCache, cell: u64) ?[]const u8 {
        const slot = self.slot_of.get(cell) orelse return null;
        return self.data.items[@as(usize, slot) * self.pb ..][0..self.pb];
    }

    /// Keep a pattern (a failed allocation only means it is not kept).
    fn put(self: *PatternCache, gpa: Allocator, cell: u64, bytes: []const u8) void {
        const max = self.slots();
        if (max == 0 or self.slot_of.contains(cell)) return;
        var slot: u32 = undefined;
        if (self.cells.items.len < max) {
            self.slot_of.ensureUnusedCapacity(gpa, 1) catch return;
            self.cells.append(gpa, cell) catch return;
            self.data.appendSlice(gpa, bytes) catch {
                self.cells.items.len -= 1;
                return;
            };
            slot = @intCast(self.cells.items.len - 1);
        } else {
            slot = self.next;
            self.next = @intCast((@as(usize, slot) + 1) % max);
            _ = self.slot_of.remove(self.cells.items[slot]);
            self.cells.items[slot] = cell;
            @memcpy(self.data.items[@as(usize, slot) * self.pb ..][0..self.pb], bytes);
        }
        self.slot_of.putAssumeCapacity(cell, slot);
    }
};

const Source = union(enum) {
    none,
    file: *scan_mod.Scan,
    memory: Memory,
};

/// Per map cell (row-major): the chosen phase's orientation and scores; phase −1: not indexed.
pub const Results = struct {
    quat: []f32 = &.{},
    ncc: []f32 = &.{},
    nmi: []f32 = &.{},
    phase: []i32 = &.{},
    /// how far the match's NMI stands above chance, in chance's standard deviations
    z: []f32 = &.{},
    /// the sweep that found an accepted match (−1: none did)
    sweep: []i32 = &.{},
    /// whether the stored match's refinement had settled
    settled: []bool = &.{},

    fn resize(self: *Results, gpa: Allocator, n: usize) !void {
        self.free(gpa);
        self.quat = try gpa.alloc(f32, 4 * n);
        self.ncc = try gpa.alloc(f32, n);
        self.nmi = try gpa.alloc(f32, n);
        self.phase = try gpa.alloc(i32, n);
        self.z = try gpa.alloc(f32, n);
        self.sweep = try gpa.alloc(i32, n);
        self.settled = try gpa.alloc(bool, n);
        self.clear();
    }

    fn clear(self: *Results) void {
        @memset(self.quat, 0);
        @memset(self.ncc, 0);
        @memset(self.nmi, 0);
        @memset(self.phase, -1);
        @memset(self.z, 0);
        @memset(self.sweep, -1);
        @memset(self.settled, false);
    }

    fn free(self: *Results, gpa: Allocator) void {
        gpa.free(self.quat);
        gpa.free(self.ncc);
        gpa.free(self.nmi);
        gpa.free(self.phase);
        gpa.free(self.z);
        gpa.free(self.sweep);
        gpa.free(self.settled);
        self.* = .{};
    }
};

/// One pattern's outcome for one phase: what the device records, and (z) what the host adds.
pub const Record = extern struct {
    q: [4]f32,
    ncc: f32,
    nmi: f32,
    /// the rotation of the last refinement step that improved the match, radians: small when
    /// the refinement had settled, large when its iterations ran out on the way (0 without
    /// refinement)
    step: f32,
    /// (NMI − chance) / chance's spread: the NMI on a scale measured once for the scan (a
    /// sample of the region's patterns against the phase at random orientations), the same
    /// for every pattern
    z: f32,
};

/// What a search runs with: its method, and the bandwidth of the spherical correlation or the
/// dictionary's grid.
pub const Search = struct {
    method: settings_mod.Method = .spherical,
    bandwidth: u32,
    dict_grid: u32,
};

pub const Engine = struct {
    gpa: Allocator,
    dev: *gpu_mod.Backend.Device,
    g: Gpu,
    set: Settings = .{},
    geo: Geometry = .{},
    events: events_mod.Sink,
    source: Source = .none,
    phases: std.ArrayList(Phase) = .empty,
    bands: std.ArrayList(*Band) = .empty,
    band_clock: u64 = 0,
    /// the spherical work buffers, pointed at the band in use
    sp: spherical.Spherical,
    prep: prep_mod.Prep,
    det: detector.Detector,
    pca: pca_mod.Pca,
    /// per level: the batch as uploaded, the batch filtered (n · h · w f32), its items
    /// ((t, pattern index) each), the static background and what it was made for
    raw: [2]Buf = .{ .{}, .{} },
    pats: [2]Buf = .{ .{}, .{} },
    items_gpu: [2]Buf = .{ .{}, .{} },
    bg_gpu: [2]Buf = .{ .{}, .{} },
    bg_key: [2]?BgKey = .{ null, null },
    raw_host: std.ArrayList(u8) = .empty,
    red_host: std.ArrayList(f32) = .empty,
    /// stored patterns by cell (see PatternCache), and scratch for what a read finds missing
    cache: PatternCache = .{},
    miss_host: std.ArrayList(u8) = .empty,
    miss_cells: std.ArrayList(u64) = .empty,
    miss: std.ArrayList(Miss) = .empty,
    /// averaging with neighbours: the kernels' buffers; the window in hand (its cells, the
    /// wanted ones first and then ring after ring around them, where each lies in that list,
    /// how many lie within each ring, and every cell's neighbours); what the first batch of an
    /// indexing run reports
    nl: nlpar_mod.Nlpar,
    nl_cells: std.ArrayList(u64) = .empty,
    nl_index: std.AutoHashMapUnmanaged(u64, u32) = .empty,
    nl_counts: std.ArrayList(u32) = .empty,
    nl_nbr: std.ArrayList(u32) = .empty,
    nl_report: enum { off, wanted, queued } = .off,
    /// for the report: every listed cell's noise, the wanted cells' shares of the stored
    /// patterns and where those lie in the list
    nl_noise: []f32 = &.{},
    nl_shares: []f32 = &.{},
    nl_where: []u32 = &.{},
    nl_stats_n: usize = 0,
    /// what the averaging did to the pattern last looked at (`pattern`): its summary and its
    /// shares of the stored patterns around it (`nl_view_side`² of them; 0: it was not averaged)
    nl_view: NlparSummary = .{ .noise = 0, .left = 1, .own = 1 },
    nl_view_shares: []f32 = &.{},
    nl_view_side: u32 = 0,
    /// the cells of the batch after the one being indexed, and what the host has ready of them
    /// by the time the device is done with this one: the cells read (`ahead_cells`), their
    /// stored bytes when `ahead_bytes` (a buffer the next read takes over whole), and their
    /// reduction by `ahead_bin` when `ahead_reduced`
    ahead: []const u64 = &.{},
    ahead_cells: std.ArrayList(u64) = .empty,
    ahead_host: std.ArrayList(u8) = .empty,
    ahead_bytes: bool = false,
    ahead_red: std.ArrayList(f32) = .empty,
    ahead_reduced: bool = false,
    ahead_bin: f64 = 0,
    /// the average pattern at stored size, and how many patterns it was asked over
    avg: []f32 = &.{},
    avg_n: u32 = 0,
    told_no_background: bool = false,
    /// each phase's spherical peaks; the refinement records; chance's mutual information
    peaks: std.ArrayList(Buf) = .empty,
    records: Buf = .{},
    chance: Buf = .{},
    /// what mutual information chance gives on this scan, per phase: measured once on a sample
    /// of the region (`calibrate`), the same for every pattern, and what it was measured for
    chance_med: []f32 = &.{},
    chance_spread: []f32 = &.{},
    chance_key: u64 = 0,
    res: Results = .{},
    /// indexing in progress: the cells of the sweep in progress and how many are done
    index_cells: []u64 = &.{},
    index_done: usize = 0,
    index_region: usize = 0,
    index_sweep: u32 = 0,
    index_accepted: usize = 0,
    /// the last pattern-centre search: its score volume (z, y, x) and axes
    pc_scores: []f32 = &.{},
    /// every grid that search scored, in order (pcfit.zig)
    pc_history: std.ArrayList(@import("pcfit.zig").Level) = .empty,
    pc_axes: [3][]f64 = .{ &.{}, &.{}, &.{} },
    /// the fit patterns' cells and, per cell, phase and orientation from the last fit
    fit_cells: []u64 = &.{},
    fit_quats: []f32 = &.{},
    fit_phase: []i32 = &.{},
    fit_ncc: []f32 = &.{},
    tune: Tune = .{},
    cancel: bool = false,
    err_buf: [1024]u8 = undefined,
    err_len: usize = 0,

    pub fn create(gpa: Allocator, dev: *gpu_mod.Backend.Device) !*Engine {
        const self = try gpa.create(Engine);
        errdefer gpa.destroy(self);
        self.* = .{ .gpa = gpa, .dev = dev, .g = undefined, .events = .{ .gpa = gpa }, .sp = undefined, .prep = undefined, .det = undefined, .pca = undefined, .nl = undefined };
        self.g = try Gpu.init(gpa, dev);
        self.sp = spherical.Spherical.init(&self.g);
        self.prep = prep_mod.Prep.init(&self.g);
        self.det = detector.Detector.init(&self.g);
        self.pca = pca_mod.Pca.init(&self.g);
        self.nl = nlpar_mod.Nlpar.init(&self.g);
        return self;
    }

    pub fn destroy(self: *Engine) void {
        const gpa = self.gpa;
        self.closeScan();
        for (self.bands.items) |b| b.destroy(&self.g);
        self.bands.deinit(gpa);
        self.sp.deinit();
        for (self.phases.items) |*p| p.deinit(&self.g);
        self.phases.deinit(gpa);
        self.prep.deinit();
        self.det.deinit();
        self.pca.deinit();
        self.nl.deinit();
        self.nl_cells.deinit(gpa);
        self.nl_index.deinit(gpa);
        self.nl_counts.deinit(gpa);
        self.nl_nbr.deinit(gpa);
        gpa.free(self.nl_noise);
        gpa.free(self.nl_shares);
        gpa.free(self.nl_where);
        gpa.free(self.nl_view_shares);
        for (self.peaks.items) |*b| self.g.release(b);
        self.peaks.deinit(gpa);
        for (0..2) |k| inline for (.{ "raw", "pats", "items_gpu", "bg_gpu" }) |f| self.g.release(&@field(self, f)[k]);
        self.g.release(&self.records);
        self.g.release(&self.chance);
        gpa.free(self.chance_med);
        gpa.free(self.chance_spread);
        self.raw_host.deinit(gpa);
        self.red_host.deinit(gpa);
        self.cache.deinit(gpa);
        self.miss_host.deinit(gpa);
        self.miss_cells.deinit(gpa);
        self.miss.deinit(gpa);
        self.ahead_cells.deinit(gpa);
        self.ahead_host.deinit(gpa);
        self.ahead_red.deinit(gpa);
        self.res.free(gpa);
        gpa.free(self.pc_scores);
        for (self.pc_axes) |a| gpa.free(a);
        @import("pcfit.zig").clearHistory(self);
        self.pc_history.deinit(gpa);
        self.events.deinit();
        self.g.deinit();
        gpa.destroy(self);
    }

    // ── errors ──

    pub fn fail(self: *Engine, comptime fmt: []const u8, args: anytype) void {
        const m = std.fmt.bufPrint(&self.err_buf, fmt, args) catch self.err_buf[0..0];
        self.err_len = m.len;
    }

    /// The first engine or GPU error since the last call (copied into `buf`), or 0.
    pub fn takeError(self: *Engine, buf: []u8) usize {
        if (self.g.oversize) |o| {
            const tail = std.fmt.bufPrint(self.err_buf[self.err_len..], " ({s}: {d}×{d}×{d} workgroups, limit {d} per dimension)", .{
                self.g.pipeName(o.pipe), o.n[0], o.n[1], o.n[2], gpu_mod.MAX_GROUPS,
            }) catch "";
            self.err_len += tail.len;
            self.g.oversize = null;
        }
        var m: []const u8 = self.err_buf[0..self.err_len];
        if (m.len == 0) m = self.dev.takeError() orelse "";
        const n = @min(m.len, buf.len);
        @memcpy(buf[0..n], m[0..n]);
        self.err_len = 0;
        return n;
    }

    // ── settings ──

    pub fn configure(self: *Engine, json: []const u8) !void {
        try self.set.apply(self.gpa, json);
    }

    // ── scan ──

    pub fn closeScan(self: *Engine) void {
        switch (self.source) {
            .file => |s| s.close(),
            .memory => |m| self.gpa.free(m.data),
            .none => {},
        }
        self.source = .none;
        self.tuneFree();
        self.cache.deinit(self.gpa);
        self.ahead = &.{};
        self.ahead_cells.clearRetainingCapacity();
        self.ahead_bytes = false;
        self.ahead_reduced = false;
        self.res.free(self.gpa);
        self.gpa.free(self.index_cells);
        self.index_cells = &.{};
        self.index_done = 0;
        self.index_region = 0;
        self.index_sweep = 0;
        self.index_accepted = 0;
        self.clearFit();
        self.gpa.free(self.avg);
        self.avg = &.{};
        self.avg_n = 0;
        self.bg_key = .{ null, null };
        self.told_no_background = false;
    }

    fn clearFit(self: *Engine) void {
        const gpa = self.gpa;
        gpa.free(self.fit_cells);
        gpa.free(self.fit_quats);
        gpa.free(self.fit_phase);
        gpa.free(self.fit_ncc);
        self.fit_cells = &.{};
        self.fit_quats = &.{};
        self.fit_phase = &.{};
        self.fit_ncc = &.{};
    }

    /// Open a scan from files (the pattern file first). The geometry takes whatever the file
    /// records; the rest keeps its current value.
    pub fn openScan(self: *Engine, paths: []const []const u8) !void {
        const s = scan_mod.Scan.open(self.gpa, paths, .{
            .rows = self.set.map_rows,
            .cols = self.set.map_cols,
            .pat_h = self.set.pattern_height,
            .pat_w = self.set.pattern_width,
        }) catch |e| {
            return e;
        };
        self.closeScan();
        self.source = .{ .file = s };
        try self.adoptInfo(s.info);
        self.events.log("opened {s}: {d} × {d} patterns of {d} × {d}", .{ s.formatName(), s.info.rows, s.info.cols, s.info.pat_h, s.info.pat_w });
    }

    /// Take patterns from memory: rows · cols patterns of h × w pixels, row-major, copied.
    pub fn setPatterns(self: *Engine, data: []const u8, pixel: scan_mod.Pixel, rows: u32, cols: u32, h: u32, w: u32) !void {
        const bpp: usize = switch (pixel) {
            .u8 => 1,
            .u16 => 2,
            .f32 => 4,
        };
        if (rows == 0 or cols == 0 or h == 0 or w == 0) return error.BadShape;
        if (data.len != @as(usize, rows) * cols * h * w * bpp) return error.BadLength;
        const copy = try self.gpa.dupe(u8, data);
        self.closeScan();
        self.source = .{ .memory = .{ .data = copy, .info = .{ .rows = rows, .cols = cols, .pat_h = h, .pat_w = w, .pixel = pixel } } };
        try self.res.resize(self.gpa, @as(usize, rows) * cols);
    }

    fn adoptInfo(self: *Engine, si: scan_mod.Info) !void {
        try self.res.resize(self.gpa, @as(usize, si.rows) * si.cols);
        const g = &self.geo;
        if (!std.math.isNan(si.pc[0]) and !std.math.isNan(si.pc[1]) and !std.math.isNan(si.pc[2])) {
            g.pc = si.pc;
            g.rot = .{ 0, 0, 0 };
        }
        if (!std.math.isNan(si.sample_tilt_deg)) g.sample_tilt_deg = si.sample_tilt_deg;
        if (!std.math.isNan(si.detector_tilt_deg)) g.detector_tilt_deg = si.detector_tilt_deg;
        if (!std.math.isNan(si.px_um)) g.px_um = si.px_um;
        if (!std.math.isNan(si.step_x_um)) g.step_x_um = si.step_x_um;
        if (!std.math.isNan(si.step_y_um)) g.step_y_um = si.step_y_um;
    }

    pub fn info(self: *const Engine) !scan_mod.Info {
        return switch (self.source) {
            .file => |s| s.info,
            .memory => |m| m.info,
            .none => error.NoScan,
        };
    }

    pub fn scanInfo(self: *const Engine) ScanInfo {
        const i = self.info() catch return .{};
        var out: ScanInfo = .{ .rows = i.rows, .cols = i.cols, .pat_h = i.pat_h, .pat_w = i.pat_w, .pixel = @intFromEnum(i.pixel) };
        out.has_pc = @intFromBool(!std.math.isNan(i.pc[0]));
        if (!std.math.isNan(i.voltage_kv)) out.voltage_kv = i.voltage_kv;
        switch (self.source) {
            .file => |s| {
                out.has_background = @intFromBool(s.background.len != 0);
                out.format = @intFromEnum(s.format);
                out.bytes = @floatFromInt(s.files[0].size);
            },
            .memory => |m| {
                out.format = 100;
                out.bytes = @floatFromInt(m.data.len);
            },
            .none => {},
        }
        return out;
    }

    // ── levels ──

    /// The factor a level's patterns are reduced by.
    pub fn binOf(self: *const Engine, lv: Level) f64 {
        return if (lv == .full) 1 else self.set.binning;
    }

    /// The level a stage works at (the one level there is when nothing is reduced).
    pub fn levelOf(self: *const Engine, stage: Stage) Level {
        if (self.set.binning == 1) return .reduced;
        return switch (self.set.full_resolution) {
            .none => .reduced,
            .scores => if (stage == .score) .full else .reduced,
            .refine => if (stage == .search) .reduced else .full,
            .all => .full,
        };
    }

    /// The level patterns are looked at: the reduced one, or as stored.
    pub fn viewLevel(self: *const Engine, full: bool) Level {
        return if (full and self.set.binning != 1) .full else .reduced;
    }

    /// Size of a pattern at a level.
    pub fn patternShape(self: *const Engine, lv: Level) ![2]u32 {
        const i = try self.info();
        const hw = reduce.shape(i.pat_h, i.pat_w, self.binOf(lv));
        if (hw[0] < 4 or hw[1] < 4) return error.PatternTooSmall;
        return hw;
    }

    /// The stored pattern's size in a level's pixels (rows, cols): what the pattern centre's
    /// fractions are fractions of. Not whole numbers when the reduction dropped a remainder.
    pub fn sizeOf(self: *const Engine, lv: Level) ![2]f64 {
        const i = try self.info();
        const f = self.binOf(lv);
        return .{ @as(f64, @floatFromInt(i.pat_h)) / f, @as(f64, @floatFromInt(i.pat_w)) / f };
    }

    /// Angular width of one of a level's pixels, radians: settings.pixel_angle_deg for a stored
    /// pixel when set, else the pixel at the pattern centre. 0 without a scan.
    pub fn pixelAngle(self: *const Engine, lv: Level) f64 {
        const i = self.info() catch return 0;
        const stored = if (self.set.pixel_angle_deg > 0)
            std.math.degreesToRadians(self.set.pixel_angle_deg)
        else
            self.geo.pixelAngle(@floatFromInt(i.pat_h));
        return stored * self.binOf(lv);
    }

    // ── reading ──

    fn patternBytes(self: *const Engine) !usize {
        const i = try self.info();
        return @as(usize, i.pat_h) * i.pat_w * bytesPerPixel(i.pixel);
    }

    /// The stored bytes of the given cells, back to back (padded to whole words). Ascending
    /// cells of a file are fetched together. `cached`: for indexing, which reads ahead: what
    /// was read ahead is handed over as it lies, and with neighbours shared between batches
    /// (NLPAR) the rest goes by way of the pattern cache.
    fn readCells(self: *Engine, cells: []const u64, cached: bool) ![]u8 {
        const i = try self.info();
        const pb = try self.patternBytes();
        const count = @as(u64, i.rows) * i.cols;
        for (cells) |c| if (c >= count) return error.OutOfRange;
        if (cached and self.ahead_bytes and std.mem.eql(u64, self.ahead_cells.items, cells)) {
            std.mem.swap(std.ArrayList(u8), &self.raw_host, &self.ahead_host);
            self.ahead_bytes = false;
            return self.raw_host.items;
        }
        const total = std.mem.alignForward(usize, cells.len * pb, 4);
        try self.raw_host.resize(self.gpa, total);
        const buf = self.raw_host.items;
        @memset(buf[cells.len * pb ..], 0);
        const out = buf[0 .. cells.len * pb];
        switch (self.source) {
            .file => |s| try self.fetch(s, cells, out, cached and self.set.nlpar and self.cacheHolds(pb, cells.len)),
            .memory => |m| for (cells, 0..) |c, k| {
                @memcpy(out[k * pb ..][0..pb], m.data[@intCast(c * pb)..][0..pb]);
            },
            .none => return error.NoScan,
        }
        return buf;
    }

    /// Whether the cache can serve a read of `n` patterns of `pb` bytes (it is emptied for a
    /// new pattern size).
    fn cacheHolds(self: *Engine, pb: usize, n: usize) bool {
        if (self.cache.pb != pb) self.cache.reset(self.gpa, pb);
        return n <= self.cache.slots();
    }

    /// Read cells of a file, in any order and with repeats: `out` (when given) gets them back
    /// to back as asked. `through`: by way of the cache, which serves what it holds and keeps
    /// what it does not. What must be read is read in ascending order, neighbours together.
    fn fetch(self: *Engine, s: *scan_mod.Scan, cells: []const u64, out: ?[]u8, through: bool) !void {
        const gpa = self.gpa;
        const pb = try self.patternBytes();
        if (!through) {
            // already in file order (a preview, a plain batch): straight to where it goes
            var ascending = true;
            for (1..cells.len) |k| {
                if (cells[k] <= cells[k - 1]) ascending = false;
            }
            if (ascending) return if (out) |o| s.readCells(cells, o);
        }
        self.miss.clearRetainingCapacity();
        // what is held first: keeping what is missing may push it out
        for (cells, 0..) |c, k| {
            if (through) if (self.cache.get(c)) |bytes| {
                if (out) |o| @memcpy(o[k * pb ..][0..pb], bytes);
                continue;
            };
            try self.miss.append(gpa, .{ .cell = c, .at = k });
        }
        const miss = self.miss.items;
        if (miss.len == 0) return;
        std.mem.sort(Miss, miss, {}, Miss.before);
        self.miss_cells.clearRetainingCapacity();
        for (miss, 0..) |m, j| {
            if (j == 0 or m.cell != miss[j - 1].cell) try self.miss_cells.append(gpa, m.cell);
        }
        try self.miss_host.resize(gpa, self.miss_cells.items.len * pb);
        try s.readCells(self.miss_cells.items, self.miss_host.items);
        var u: usize = 0;
        for (miss, 0..) |m, j| {
            if (j != 0 and m.cell != miss[j - 1].cell) u += 1;
            const bytes = self.miss_host.items[u * pb ..][0..pb];
            if (out) |o| @memcpy(o[m.at * pb ..][0..pb], bytes);
            if (through) self.cache.put(gpa, m.cell, bytes);
        }
    }

    /// Do on the host what indexing `cells` will need of it, while the device is busy with
    /// the batch before: read their patterns and, when a level is reduced, reduce them.
    /// Whatever goes wrong here is left for the upload proper to report.
    fn readAhead(self: *Engine, cells: []const u64) void {
        const gpa = self.gpa;
        const i = self.info() catch return;
        const pb = self.patternBytes() catch return;
        self.ahead_cells.clearRetainingCapacity();
        self.ahead_bytes = false;
        self.ahead_reduced = false;
        var list = cells;
        if (self.set.nlpar) {
            _ = self.nlparWindow(cells) catch return;
            list = self.nl_cells.items;
        } else if (self.cache.pb != 0) {
            // (no batch shares neighbours with the next any more)
            self.cache.deinit(gpa);
        }
        if (list.len == 0) return;
        const bin = self.binOf(.reduced);
        var reduced = false;
        for ([_]Stage{ .search, .refine, .score }) |st| {
            if (bin != 1 and self.levelOf(st) == .reduced) reduced = true;
        }
        // With neighbours shared, by way of the cache, which then has them for the batches
        // after as well. Otherwise straight into a buffer the batch takes over as it lies.
        const through = self.set.nlpar and self.source == .file and self.cacheHolds(pb, 2 * list.len);
        if (through and !reduced) {
            self.fetch(self.source.file, list, null, true) catch {};
            return;
        }
        // (patterns in memory need no reading, only reducing)
        if (self.source != .file and !reduced) return;
        self.ahead_host.resize(gpa, std.mem.alignForward(usize, list.len * pb, 4)) catch return;
        @memset(self.ahead_host.items[list.len * pb ..], 0);
        const bytes = self.ahead_host.items[0 .. list.len * pb];
        switch (self.source) {
            .file => |s| self.fetch(s, list, bytes, through) catch return,
            .memory => |m| for (list, 0..) |c, k| {
                @memcpy(bytes[k * pb ..][0..pb], m.data[@intCast(c * pb)..][0..pb]);
            },
            .none => return,
        }
        self.ahead_cells.appendSlice(gpa, list) catch return;
        self.ahead_bytes = true;
        if (!reduced) return;
        const hw = self.patternShape(.reduced) catch return;
        self.ahead_red.resize(gpa, list.len * @as(usize, hw[0]) * hw[1]) catch return;
        reduce.patterns(gpa, bytes, list.len, i.pat_h, i.pat_w, @intCast(bytesPerPixel(i.pixel)), bin, self.ahead_red.items) catch return;
        self.ahead_bin = bin;
        self.ahead_reduced = true;
    }

    /// The patterns of `cells` (their stored bytes in `bytes`) reduced for a level, into
    /// `red_host`: what was reduced ahead, when that was these cells.
    fn reduceLevel(self: *Engine, lv: Level, cells: []const u64, bytes: []const u8) !void {
        const i = try self.info();
        const bin = self.binOf(lv);
        if (self.ahead_reduced and self.ahead_bin == bin and std.mem.eql(u64, self.ahead_cells.items, cells)) {
            std.mem.swap(std.ArrayList(f32), &self.red_host, &self.ahead_red);
            self.ahead_reduced = false;
            return;
        }
        const hw = try self.patternShape(lv);
        const bpp = bytesPerPixel(i.pixel);
        try self.red_host.resize(self.gpa, cells.len * @as(usize, hw[0]) * hw[1]);
        try reduce.patterns(self.gpa, bytes[0 .. cells.len * i.pat_h * i.pat_w * bpp], cells.len, i.pat_h, i.pat_w, @intCast(bpp), bin, self.red_host.items);
    }

    // ── averaging with neighbours ──

    /// Bytes of one copy of a window's patterns on the device at most: a batch whose window
    /// would take more is averaged in parts.
    const nlpar_bytes: u64 = 256 << 20;

    /// The window averaging `wanted` needs: the wanted cells, then ring after ring of the
    /// map's cells around them, as far out as the passes reach, with every cell's neighbours.
    /// Left in the engine's lists; the result points into them.
    fn nlparWindow(self: *Engine, wanted: []const u64) !nlpar_mod.Window {
        const gpa = self.gpa;
        const i = try self.info();
        const rows: i64 = i.rows;
        const cols: i64 = i.cols;
        const radius = self.set.nlpar_radius;
        const passes = self.set.nlpar_iterations;
        const cells = &self.nl_cells;
        cells.clearRetainingCapacity();
        self.nl_index.clearRetainingCapacity();
        self.nl_counts.clearRetainingCapacity();
        for (wanted) |c| {
            const at = try self.nl_index.getOrPut(gpa, c);
            if (!at.found_existing) at.value_ptr.* = @intCast(cells.items.len);
            try cells.append(gpa, c);
        }
        try self.nl_counts.append(gpa, @intCast(cells.items.len));
        var first: usize = 0;
        for (0..nlpar_mod.Window.ringsOf(radius, passes)) |_| {
            const end = cells.items.len;
            for (first..end) |k| {
                const y: i64 = @intCast(cells.items[k] / i.cols);
                const x: i64 = @intCast(cells.items[k] % i.cols);
                var dy: i64 = -1;
                while (dy <= 1) : (dy += 1) {
                    var dx: i64 = -1;
                    while (dx <= 1) : (dx += 1) {
                        if (y + dy < 0 or y + dy >= rows or x + dx < 0 or x + dx >= cols) continue;
                        const at = try self.nl_index.getOrPut(gpa, @intCast((y + dy) * cols + x + dx));
                        if (at.found_existing) continue;
                        at.value_ptr.* = @intCast(cells.items.len);
                        try cells.append(gpa, at.key_ptr.*);
                    }
                }
            }
            first = end;
            try self.nl_counts.append(gpa, @intCast(cells.items.len));
        }
        const r: i64 = nlpar_mod.Window.reachOf(radius, passes);
        const side: usize = @intCast(2 * r + 1);
        try self.nl_nbr.resize(gpa, cells.items.len * side * side);
        for (cells.items, 0..) |c, k| {
            const y: i64 = @intCast(c / i.cols);
            const x: i64 = @intCast(c % i.cols);
            var dy: i64 = -r;
            while (dy <= r) : (dy += 1) {
                var dx: i64 = -r;
                while (dx <= r) : (dx += 1) {
                    const slot: usize = @intCast((dy + r) * @as(i64, @intCast(side)) + dx + r);
                    const inside = y + dy >= 0 and y + dy < rows and x + dx >= 0 and x + dx < cols;
                    self.nl_nbr.items[k * side * side + slot] = if (inside) self.nl_index.get(@intCast((y + dy) * cols + x + dx)) orelse nlpar_mod.NONE else nlpar_mod.NONE;
                }
            }
        }
        return .{ .nbr = self.nl_nbr.items, .counts = self.nl_counts.items, .radius = radius, .passes = passes };
    }

    /// Record: the window's patterns (their stored bytes in `bytes`, in the window's order)
    /// at a level, averaged, the wanted ones into the level's pattern buffer from pattern
    /// `out0` on.
    fn averageLevel(self: *Engine, lv: Level, bytes: []const u8, win: nlpar_mod.Window, out0: u32) !void {
        const g = &self.g;
        const i = try self.info();
        const k = @intFromEnum(lv);
        const hw = try self.patternShape(lv);
        const px = hw[0] * hw[1];
        const n_win: u32 = @intCast(self.nl_cells.items.len);
        g.ensure(&self.nl.a, @as(u64, n_win) * px * 4);
        if (self.binOf(lv) == 1) {
            g.ensure(&self.raw[k], bytes.len);
            g.write(self.raw[k], 0, bytes);
            self.prep.load(self.raw[k], n_win, hw[0], hw[1], @intFromEnum(i.pixel), self.nl.a);
        } else {
            try self.reduceLevel(lv, self.nl_cells.items, bytes);
            g.writeSlice(self.nl.a, 0, f32, self.red_host.items);
        }
        self.nl.average(win, px, self.set.nlpar_lambda, self.pats[k], out0);
    }

    /// What the averaging did to the cells last read back with `account`: the stored patterns'
    /// noise (standard deviation a pixel), the share of it the averages keep, and the share of
    /// an average that is the pattern itself, each a mean over those cells.
    pub const NlparSummary = struct { noise: f64, left: f64, own: f64 };

    fn nlparSummary(self: *const Engine) NlparSummary {
        const n = self.nl_stats_n;
        const f = self.nl_shares.len / @max(n, 1);
        var noise: f64 = 0;
        var kept: f64 = 0;
        var own: f64 = 0;
        var counted: f64 = 0;
        for (0..n) |c| {
            const before: f64 = self.nl_noise[c];
            // an average's noise: Σ share² · the noise of the pattern it is a share of
            var after: f64 = 0;
            for (self.nl_shares[c * f ..][0..f], self.nl_where[c * f ..][0..f]) |a, at| {
                if (at != nlpar_mod.NONE) after += @as(f64, a) * a * self.nl_noise[at];
            }
            own += self.nl_shares[c * f + f / 2];
            if (!(before > 0)) continue;
            noise += @sqrt(before);
            kept += @sqrt(after / before);
            counted += 1;
        }
        return .{ .noise = noise / @max(counted, 1), .left = kept / @max(counted, 1), .own = own / @as(f64, @floatFromInt(@max(n, 1))) };
    }

    /// Log what the averaging did to the first batch of an indexing run (read with its
    /// results).
    fn reportNlpar(self: *Engine) void {
        if (self.nl_report != .queued) return;
        self.nl_report = .off;
        const a = self.nlparSummary();
        const s = &self.set;
        self.events.log("neighbour averaging: {d} pass{s} over {d} × {d} cells (λ {d:.2}); noise about {d:.2} a pixel, {d:.0} % of it left; an average is {d:.0} % the pattern itself", .{
            s.nlpar_iterations,     if (s.nlpar_iterations == 1) "" else "es",
            2 * s.nlpar_radius + 1, 2 * s.nlpar_radius + 1,
            s.nlpar_lambda,         a.noise,
            100 * a.left,           100 * a.own,
        });
    }

    /// About `want` cells on a lattice over the whole map, each at the centre of its share
    /// (every cell when the map has no more), ascending. Caller frees.
    pub fn lattice(self: *const Engine, want: u32) ![]u64 {
        const i = try self.info();
        const rows: u64 = i.rows;
        const cols: u64 = i.cols;
        var gy = rows;
        var gx = cols;
        if (rows * cols > want) {
            const ratio = @as(f64, @floatFromInt(rows)) / @as(f64, @floatFromInt(cols));
            gy = std.math.clamp(@as(u64, @intFromFloat(@round(@sqrt(@as(f64, @floatFromInt(want)) * ratio)))), 1, rows);
            gx = std.math.clamp(want / gy, 1, cols);
        }
        const cells = try self.gpa.alloc(u64, @intCast(gy * gx));
        for (0..@intCast(gy)) |r| for (0..@intCast(gx)) |c| {
            cells[r * @as(usize, @intCast(gx)) + c] = ((2 * r + 1) * rows / (2 * gy)) * cols + (2 * c + 1) * cols / (2 * gx);
        };
        return cells;
    }

    /// The average pattern at stored size, over settings.static_patterns patterns of a lattice.
    fn ensureAverage(self: *Engine) !void {
        if (self.avg.len != 0 and self.avg_n == self.set.static_patterns) return;
        const gpa = self.gpa;
        const i = try self.info();
        const px = @as(usize, i.pat_h) * i.pat_w;
        const bpp = bytesPerPixel(i.pixel);
        const cells = try self.lattice(self.set.static_patterns);
        defer gpa.free(cells);
        const sum = try gpa.alloc(f64, px);
        defer gpa.free(sum);
        @memset(sum, 0);
        var done: usize = 0;
        while (done < cells.len) {
            const n = @min(cells.len - done, 256);
            const bytes = try self.readCells(cells[done..][0..n], false);
            for (0..n) |k| {
                const pat = bytes[k * px * bpp ..][0 .. px * bpp];
                switch (i.pixel) {
                    .u8 => for (sum, pat) |*a, v| {
                        a.* += @floatFromInt(v);
                    },
                    .u16 => for (sum, 0..) |*a, j| {
                        a.* += @floatFromInt(std.mem.readInt(u16, pat[2 * j ..][0..2], .little));
                    },
                    .f32 => for (sum, 0..) |*a, j| {
                        a.* += @as(f32, @bitCast(std.mem.readInt(u32, pat[4 * j ..][0..4], .little)));
                    },
                }
            }
            done += n;
            if (cells.len > 256) self.events.emit(.{ .kind = "progress", .stage = "background", .done = done, .total = cells.len });
        }
        if (self.avg.len != px) {
            gpa.free(self.avg);
            self.avg = &.{};
            self.avg = try gpa.alloc(f32, px);
        }
        const k = 1.0 / @as(f64, @floatFromInt(cells.len));
        for (self.avg, sum) |*a, v| a.* = @floatCast(v * k);
        self.avg_n = self.set.static_patterns;
        self.bg_key = .{ null, null };
        self.events.log("static background: the average of {d} patterns over the map", .{cells.len});
    }

    /// The static background of a level on the device (null: none is subtracted).
    fn staticBackground(self: *Engine, lv: Level) !?Buf {
        const kind = self.set.static_background;
        var src: []const f32 = &.{};
        switch (kind) {
            .none => return null,
            .average => {
                try self.ensureAverage();
                src = self.avg;
            },
            .file => {
                src = switch (self.source) {
                    .file => |s| s.background,
                    else => &.{},
                };
                if (src.len == 0) {
                    if (!self.told_no_background) self.events.log("static background: this scan stores none, so none is subtracted (static_background: \"average\" takes the patterns' own)", .{});
                    self.told_no_background = true;
                    return null;
                }
            },
        }
        const i = try self.info();
        const k = @intFromEnum(lv);
        const key: BgKey = .{ .kind = kind, .n = if (kind == .average) self.avg_n else 0, .bin = self.binOf(lv) };
        if (self.bg_key[k] == null or !std.meta.eql(self.bg_key[k].?, key)) {
            const hw = try self.patternShape(lv);
            const out = try self.gpa.alloc(f32, @as(usize, hw[0]) * hw[1]);
            defer self.gpa.free(out);
            try reduce.patterns(self.gpa, std.mem.sliceAsBytes(src), 1, i.pat_h, i.pat_w, 4, key.bin, out);
            self.g.ensure(&self.bg_gpu[k], out.len * 4);
            self.g.writeSlice(self.bg_gpu[k], 0, f32, out);
            self.bg_key[k] = key;
        }
        return self.bg_gpu[k];
    }

    /// The pattern filters at a level; the band pass keeps its size on the detector.
    pub fn prepOptions(self: *const Engine, lv: Level, hw: [2]u32, static: ?Buf) prep_mod.Options {
        const s = &self.set;
        const f = self.binOf(lv);
        return .{
            .static = static,
            .dynamic = s.dynamic_background,
            .dynamic_sigma = s.dynamic_sigma * @as(f64, @floatFromInt(hw[1])),
            .clahe = s.clahe,
            .clahe_grid = s.clahe_grid,
            .clahe_bins = s.clahe_bins,
            .clahe_clip = s.clahe_clip,
            .bandpass = s.bandpass,
            .bp_fine = s.bp_fine / f,
            .bp_coarse = s.bp_coarse / f,
            .mask = self.maskOf(hw[0], hw[1]),
        };
    }

    fn maskOf(self: *const Engine, h: u32, w: u32) [3]f32 {
        if (!self.set.mask) return .{ 0, 0, 0 };
        const r = self.set.mask_radius * 0.5 * @as(f64, @floatFromInt(@min(h, w)));
        return .{ 0.5 * @as(f32, @floatFromInt(h - 1)), 0.5 * @as(f32, @floatFromInt(w - 1)), @floatCast(r * r) };
    }

    /// Record: unpack `n` patterns from the host bytes last read (as stored, or reduced here
    /// for a reduced level) into the level's pattern buffer, filtered when `opts` is given.
    fn uploadLevel(self: *Engine, lv: Level, cells: []const u64, bytes: []const u8, opts: ?prep_mod.Options) !void {
        const g = &self.g;
        const i = try self.info();
        const k = @intFromEnum(lv);
        const hw = try self.patternShape(lv);
        const px = @as(usize, hw[0]) * hw[1];
        const n: u32 = @intCast(cells.len);
        const o = opts orelse prep_mod.Options{ .normalize = false };
        g.ensure(&self.pats[k], @as(u64, n) * px * 4);
        if (self.binOf(lv) == 1) {
            g.ensure(&self.raw[k], bytes.len);
            g.write(self.raw[k], 0, bytes);
            self.prep.run(self.raw[k], n, hw[0], hw[1], @intFromEnum(i.pixel), o, self.pats[k]);
        } else {
            try self.reduceLevel(lv, cells, bytes);
            g.ensure(&self.raw[k], self.red_host.items.len * 4);
            g.writeSlice(self.raw[k], 0, f32, self.red_host.items);
            self.prep.run(self.raw[k], n, hw[0], hw[1], 2, o, self.pats[k]);
        }
    }

    /// Record: read the cells once and leave them, filtered, in the pattern buffer of each of
    /// the given levels.
    pub fn uploadCells(self: *Engine, cells: []const u64, levels: []const Level) !void {
        return self.uploadCellsAs(cells, levels, .{});
    }

    const Upload = struct {
        /// false: averaged with their neighbours (when that is on) and nothing more
        filter: bool = true,
        /// also read back what the averaging did (`nlparSummary` after the wait)
        account: bool = false,
    };

    fn uploadCellsAs(self: *Engine, cells: []const u64, levels: []const Level, how: Upload) !void {
        var need = [2]bool{ false, false };
        for (levels) |lv| need[@intFromEnum(lv)] = true;
        // the backgrounds first: averaging reads patterns through the same host buffer
        var bg: [2]?Buf = .{ null, null };
        for (0..2) |k| {
            if (need[k] and how.filter) bg[k] = try self.staticBackground(@enumFromInt(k));
        }
        if (!self.set.nlpar) {
            const bytes = try self.readCells(cells, true);
            for (0..2) |k| {
                if (!need[k]) continue;
                const lv: Level = @enumFromInt(k);
                try self.uploadLevel(lv, cells, bytes, if (how.filter) self.prepOptions(lv, try self.patternShape(lv), bg[k]) else null);
            }
            return;
        }
        // averaged with their neighbours first: the cells' windows are read, averaged on the
        // device, and the filters then run on the averages
        const n: u32 = @intCast(cells.len);
        var px_max: u64 = 1;
        for (0..2) |k| {
            if (!need[k]) continue;
            const hw = try self.patternShape(@enumFromInt(k));
            const px = @as(u64, hw[0]) * hw[1];
            self.g.ensure(&self.pats[k], n * px * 4);
            px_max = @max(px_max, px);
        }
        const max_win: usize = @intCast(@max(1, nlpar_bytes / (px_max * 4)));
        var t0: u32 = 0;
        while (t0 < n) {
            // as many of the cells at once as their window allows
            var m: u32 = n - t0;
            var win = try self.nlparWindow(cells[t0..][0..m]);
            while (m > 1 and self.nl_cells.items.len > max_win) {
                m = (m + 1) / 2;
                win = try self.nlparWindow(cells[t0..][0..m]);
            }
            const bytes = try self.readCells(self.nl_cells.items, true);
            var reported = false;
            for (0..2) |k| {
                if (!need[k]) continue;
                try self.averageLevel(@enumFromInt(k), bytes, win, t0);
                if (how.account and t0 == 0 and !reported) {
                    reported = true;
                    const gpa = self.gpa;
                    const f = @as(usize, win.side()) * win.side();
                    gpa.free(self.nl_noise);
                    self.nl_noise = &.{};
                    self.nl_noise = try gpa.alloc(f32, self.nl_cells.items.len);
                    gpa.free(self.nl_shares);
                    self.nl_shares = &.{};
                    self.nl_shares = try gpa.alloc(f32, m * f);
                    gpa.free(self.nl_where);
                    self.nl_where = &.{};
                    self.nl_where = try gpa.dupe(u32, win.nbr[0 .. m * f]);
                    self.nl_stats_n = m;
                    self.g.read(self.nl.sig, 0, std.mem.sliceAsBytes(self.nl_noise));
                    self.g.read(self.nl.final, 0, std.mem.sliceAsBytes(self.nl_shares));
                }
            }
            t0 += m;
        }
        if (!how.filter) return;
        for (0..2) |k| {
            if (!need[k]) continue;
            const lv: Level = @enumFromInt(k);
            const hw = try self.patternShape(lv);
            self.prep.filter(self.pats[k], n, hw[0], hw[1], self.prepOptions(lv, hw, bg[k]));
        }
    }

    // ── geometry on the device ──

    pub fn frame(self: *const Engine, lv: Level) !spherical.Frame {
        const hw = try self.patternShape(lv);
        const R = self.geo.rotation();
        return .{
            .c0 = .{ @floatCast(R[0]), @floatCast(R[3]), @floatCast(R[6]) },
            .c1 = .{ @floatCast(R[1]), @floatCast(R[4]), @floatCast(R[7]) },
            .c2 = .{ @floatCast(R[2]), @floatCast(R[5]), @floatCast(R[8]) },
            .pat_h = hw[0],
            .pat_w = hw[1],
            .mask = self.maskOf(hw[0], hw[1]),
        };
    }

    /// The detector's position under `geo` in a level's pixels (the reference cell's).
    pub fn translationOf(self: *const Engine, geo: *const Geometry, lv: Level) !rot.Vec3 {
        const sz = try self.sizeOf(lv);
        return geo.translation(sz[0], sz[1]);
    }

    /// Beam offset of a cell from the map's reference (its centre), in a level's pixels.
    pub fn offsetOf(self: *const Engine, geo: *const Geometry, cell: u64, lv: Level) rot.Vec3 {
        const i = self.info() catch return .{ 0, 0, 0 };
        const row: f64 = @floatFromInt(cell / i.cols);
        const col: f64 = @floatFromInt(cell % i.cols);
        return geo.scanOffset(row, col, 0.5 * @as(f64, @floatFromInt(i.rows - 1)), 0.5 * @as(f64, @floatFromInt(i.cols - 1)), self.set.scan_x_sign, self.set.scan_y_sign, self.binOf(lv));
    }

    /// The detector position for a cell under `geo`, in a level's pixels.
    pub fn positionOf(self: *const Engine, geo: *const Geometry, cell: u64, lv: Level) ![3]f32 {
        const t = rot.sub(try self.translationOf(geo, lv), self.offsetOf(geo, cell, lv));
        return .{ @floatCast(t[0]), @floatCast(t[1]), @floatCast(t[2]) };
    }

    // ── phases ──

    pub fn addPhase(self: *Engine, path: []const u8) !u32 {
        var m = try master_mod.load(self.gpa, path);
        errdefer m.deinit();
        try self.peaks.ensureUnusedCapacity(self.gpa, 1);
        for (self.bands.items) |b| try b.ph.ensureUnusedCapacity(self.gpa, 1);
        try self.phases.append(self.gpa, .{ .master = m });
        self.peaks.appendAssumeCapacity(.{});
        for (self.bands.items) |b| b.ph.appendAssumeCapacity(null);
        self.events.log("phase {d}: {s} ({s}, {d} px, point group {s})", .{ self.phases.items.len - 1, m.name, m.source, m.side, if (m.point_group >= 1 and m.point_group <= 32) sym.point_group_names[m.point_group - 1] else "?" });
        return @intCast(self.phases.items.len - 1);
    }

    pub fn clearPhases(self: *Engine) void {
        for (self.bands.items) |b| {
            for (0..b.ph.items.len) |pi| b.dropPhase(&self.g, pi);
            b.ph.clearRetainingCapacity();
        }
        for (self.phases.items) |*p| p.deinit(&self.g);
        self.phases.clearRetainingCapacity();
        for (self.peaks.items) |*b| self.g.release(b);
        self.peaks.clearRetainingCapacity();
        self.clearFit();
    }

    pub fn phaseInfo(self: *Engine, index: u32) !PhaseInfo {
        if (index >= self.phases.items.len) return error.NoSuchPhase;
        const p = &self.phases.items[index];
        var out: PhaseInfo = .{ .side = p.master.side, .point_group = p.master.point_group, .space_group = p.master.space_group, .laue = @intFromEnum(p.master.laue), .voltage_kv = p.master.voltage_kv };
        // the selection rules as the band last used found them
        var newest: u64 = 0;
        for (self.bands.items) |b| {
            if (b.stamp < newest) continue;
            if (b.ph.items[index]) |x| {
                newest = b.stamp;
                out.n_fold = x.hm.rules.n_fold;
                out.mirror = @intFromBool(x.hm.rules.mirror);
                out.inversion = @intFromBool(x.hm.rules.inversion);
            }
        }
        const n = @min(p.master.name.len, out.name.len - 1);
        @memcpy(out.name[0..n], p.master.name[0..n]);
        return out;
    }

    /// What a phase's master must be filtered for at a level: the patterns' own filters, at
    /// the sizes they have on the sphere when a pattern pixel has its angular width.
    fn bakeKeyOf(self: *const Engine, p: *const Phase, lv: Level) BakeKey {
        const s = &self.set;
        var key: BakeKey = .{ .px_q = std.math.minInt(i32), .smooth = s.master_smooth, .bandpass = s.bandpass, .fine = 0, .coarse = 0, .clahe = s.clahe, .tiles = 0, .bins = 0, .clip = 0 };
        const angle = self.pixelAngle(lv);
        const hw = self.patternShape(lv) catch return key;
        if (!(angle > 0)) return key;
        // the sheet spreads 2π steradians over side² pixels
        const px = angle * @as(f64, @floatFromInt(p.master.side)) / @sqrt(2.0 * std.math.pi);
        key.px_q = @intFromFloat(@round(@log(px) / @log(1.05)));
        const f = self.binOf(lv);
        if (s.bandpass) {
            key.fine = s.bp_fine / f;
            key.coarse = s.bp_coarse / f;
        }
        if (s.clahe) {
            // tiles as wide on the sphere as the patterns' are: pattern height / grid pixels
            const tile = @as(f64, @floatFromInt(hw[0])) / @as(f64, @floatFromInt(s.clahe_grid)) * px;
            key.tiles = @intFromFloat(std.math.clamp(@round(@as(f64, @floatFromInt(p.master.side)) / tile), 1, 64));
            key.bins = s.clahe_bins;
            key.clip = s.clahe_clip;
        }
        return key;
    }

    /// Bring every phase's filtered master up to date with the settings and the geometry, at
    /// each of the given levels.
    pub fn prepareMasters(self: *Engine, levels: []const Level) !void {
        if (self.phases.items.len == 0) return error.NoPhases;
        for (self.phases.items) |*p| for (levels) |lv| {
            const k = @intFromEnum(lv);
            const key = self.bakeKeyOf(p, lv);
            if (p.baked[k] != null and std.meta.eql(p.baked[k].?, key)) continue;
            try self.bakeMaster(p, lv, key);
            self.prep.gradient(p.det[k].buf, p.master.side, &p.det[k].grad);
            p.baked[k] = key;
            p.serial[k] +%= 1;
        };
    }

    /// Filter the master's sheets for a level (on the device), normalize them together and
    /// keep them on both sides. The order is the patterns': what a pixel averages (smoothing),
    /// CLAHE, band pass.
    fn bakeMaster(self: *Engine, p: *Phase, lv: Level, key: BakeKey) !void {
        const g = &self.g;
        const gpa = self.gpa;
        const k = @intFromEnum(lv);
        const m = &p.master;
        const side = m.side;
        const n = m.raw.len;
        if (p.work[k].len != n) {
            gpa.free(p.work[k]);
            p.work[k] = &.{};
            p.work[k] = try gpa.alloc(f32, n);
        }
        const work = p.work[k];
        @memcpy(work, m.raw);
        g.ensure(&p.det[k].buf, n * 4);
        p.det[k].side = side;
        const buf = p.det[k].buf;
        g.writeSlice(buf, 0, f32, m.raw);
        if (key.px_q == std.math.minInt(i32)) return;
        const px = std.math.pow(f64, 1.05, @floatFromInt(key.px_q));
        const smooth = key.smooth * px;
        if (!(smooth > 0.05 or key.bandpass or key.clahe)) return;
        var fine = key.fine * px;
        if (key.clahe) {
            self.prep.sphereBlur(buf, side, smooth);
            self.prep.filter(buf, 2, side, side, .{ .clahe = true, .clahe_grid = key.tiles, .clahe_bins = key.bins, .clahe_clip = key.clip, .normalize = false });
            // CLAHE works on the sheets as images, which the crystal's symmetry does not
            // respect: average the result over the master's own point group
            if (m.ops.len > 1) {
                const rows = try gpa.alloc([4]f32, m.ops.len * 3);
                defer gpa.free(rows);
                for (m.ops, 0..) |op, j| for (0..3) |r| {
                    rows[j * 3 + r] = .{ @floatCast(op[r * 3]), @floatCast(op[r * 3 + 1]), @floatCast(op[r * 3 + 2]), 0 };
                };
                var ops = g.storage(rows.len * 16);
                defer g.release(&ops);
                g.writeSlice(ops, 0, [4]f32, rows);
                self.prep.symmetrize(buf, side, ops, @intCast(m.ops.len));
            }
        } else {
            // what the pixel averages and the band pass's fine Gaussian compose into one
            fine = @sqrt(smooth * smooth + fine * fine);
            if (!key.bandpass) self.prep.sphereBlur(buf, side, smooth);
        }
        if (key.bandpass) {
            // the difference of two smoothings on the sphere
            var low = g.storage(n * 4);
            defer g.release(&low);
            g.copy(buf, 0, low, 0, n * 4);
            self.prep.sphereBlur(buf, side, fine);
            self.prep.sphereBlur(low, side, key.coarse * px);
            const lo_h = try gpa.alloc(f32, n);
            defer gpa.free(lo_h);
            g.read(buf, 0, std.mem.sliceAsBytes(work));
            g.read(low, 0, std.mem.sliceAsBytes(lo_h));
            try g.wait();
            for (work, lo_h) |*v, l| v.* -= l;
        } else {
            g.read(buf, 0, std.mem.sliceAsBytes(work));
            try g.wait();
        }
        // one scale for both sheets again
        var sum: f64 = 0;
        for (work) |v| sum += v;
        const mean = sum / @as(f64, @floatFromInt(n));
        var ss: f64 = 0;
        for (work) |v| ss += (v - mean) * (v - mean);
        const scale = 1.0 / @max(@sqrt(ss / @as(f64, @floatFromInt(n))), 1e-30);
        for (work) |*v| v.* = @floatCast((v.* - mean) * scale);
        g.writeSlice(buf, 0, f32, work);
    }

    /// Spherical indexing at bandwidth L against every phase's master as filtered for `lv`
    /// (built on first use, kept for the next). The work buffers (`self.sp`) are left pointing
    /// at this band's tables.
    pub fn band(self: *Engine, L: u32, lv: Level) !*Band {
        const g = &self.g;
        const gpa = self.gpa;
        try self.prepareMasters(&.{lv});
        var found: ?*Band = null;
        for (self.bands.items) |b| {
            if (b.L == L) found = b;
        }
        if (found == null) {
            if (self.bands.items.len >= MAX_BANDS) {
                var oldest: usize = 0;
                for (self.bands.items, 0..) |b, i| {
                    if (b.stamp < self.bands.items[oldest].stamp) oldest = i;
                }
                self.bands.swapRemove(oldest).destroy(g);
            }
            try self.bands.ensureUnusedCapacity(gpa, 1);
            const b = try gpa.create(Band);
            errdefer gpa.destroy(b);
            b.* = .{ .L = L, .tables = try spherical.Tables.init(g, L) };
            self.bands.appendAssumeCapacity(b);
            found = b;
        }
        const b = found.?;
        self.band_clock += 1;
        b.stamp = self.band_clock;
        self.sp.t = &b.tables;
        self.sp.wg = self.set.wg_fft;
        while (b.ph.items.len < self.phases.items.len) try b.ph.append(gpa, null);
        const k = @intFromEnum(lv);
        for (self.phases.items, 0..) |*p, pi| {
            if (b.ph.items[pi]) |x| {
                if (x.level == lv and x.serial == p.serial[k]) continue;
                b.dropPhase(g, pi);
            }
            var hm = try master_mod.Harmonics.init(gpa, &p.master, p.work[k], L);
            errdefer hm.deinit(gpa);
            const gp = try spherical.PhaseGpu.init(g, &b.tables, &hm);
            b.ph.items[pi] = .{ .hm = hm, .gpu = gp, .level = lv, .serial = p.serial[k] };
            const r = hm.rules;
            self.events.log("phase {d}: bandwidth {d}, rotation order {d} about z{s}{s}", .{ pi, L, r.n_fold, if (r.mirror) ", mirror" else "", if (r.inversion) ", inversion" else "" });
        }
        return b;
    }

    /// Record (when stale): each phase's denominator volume for the reference geometry.
    fn prepareDenominators(self: *Engine, b: *Band, lv: Level, fr: spherical.Frame) !void {
        const t64 = try self.translationOf(&self.geo, lv);
        const t: [3]f32 = .{ @floatCast(t64[0]), @floatCast(t64[1]), @floatCast(t64[2]) };
        const key: RdenKey = .{ .c = .{ fr.c0[0], fr.c0[1], fr.c0[2], fr.c1[0], fr.c1[1], fr.c1[2], fr.c2[0], fr.c2[1], fr.c2[2] }, .t = t, .h = fr.pat_h, .w = fr.pat_w, .mask = fr.mask };
        for (b.ph.items) |*slot| {
            const x = &slot.*.?;
            if (x.rden_key != null and std.meta.eql(x.rden_key.?, key)) continue;
            self.sp.denominators(&x.gpu, &.{.{ t[0], t[1], t[2], 0 }}, fr, &x.gpu.rden);
            x.rden_key = key;
        }
    }

    /// The phase's dictionary: points of a cubochoric grid of N per half edge whose
    /// orientation is its own fundamental-zone representative.
    fn prepareDictionary(self: *Engine, p: *Phase, N: u32) !void {
        const gpa = self.gpa;
        if (p.dict_grid == N and p.dict.len != 0) return;
        if (p.takeDictionary(&self.g, N)) return;
        var list: std.ArrayList(f32) = .empty;
        errdefer list.deinit(gpa);
        const ops = sym.operators(p.master.rotation);
        const a = rot.cubochoric_half_edge;
        const nf: f64 = @floatFromInt(N);
        var i: i64 = -@as(i64, N);
        while (i <= N) : (i += 1) {
            var j: i64 = -@as(i64, N);
            while (j <= N) : (j += 1) {
                var k: i64 = -@as(i64, N);
                while (k <= N) : (k += 1) {
                    const q = rot.qstd(rot.fromCubochoric(.{ a * @as(f64, @floatFromInt(i)) / nf, a * @as(f64, @floatFromInt(j)) / nf, a * @as(f64, @floatFromInt(k)) / nf }));
                    var in_fz = true;
                    for (ops[1..]) |g| {
                        if (@abs(rot.qmul(g, q)[0]) > q[0] + 1e-9) {
                            in_fz = false;
                            break;
                        }
                    }
                    if (!in_fz) continue;
                    for (q) |c| try list.append(gpa, @floatCast(c));
                }
            }
        }
        p.dict = try list.toOwnedSlice(gpa);
        p.dict_grid = N;
        // in a shuffled order (the same every time): a run of grid neighbours is no sample of
        // the dictionary, and Oja's rule learns from one run at a time
        var prng = std.Random.DefaultPrng.init(0x6469_6374);
        const r = prng.random();
        var left = p.dict.len / 4;
        while (left > 1) : (left -= 1) {
            const j = r.uintLessThan(usize, left);
            const hi = p.dict[4 * (left - 1) ..][0..4];
            const lo = p.dict[4 * j ..][0..4];
            const tmp = hi.*;
            hi.* = lo.*;
            lo.* = tmp;
        }
        self.g.ensure(&p.dict_gpu, p.dict.len * 4);
        self.g.writeSlice(p.dict_gpu, 0, f32, p.dict);
        self.events.log("dictionary: {d} orientations for the rotations {s} (grid {d})", .{ p.dict.len / 4, p.master.rotation.rotationName(), N });
    }

    /// The phase's dictionary as its leading components: learned by Oja's rule while the
    /// dictionary is simulated (at the reference detector position `t`, the search's level),
    /// then every entry's coefficients. Kept until the dictionary, the geometry, the filters
    /// or the settings change.
    fn preparePca(self: *Engine, p: *Phase, N: u32, lv: Level, fr: spherical.Frame, t: [3]f32) !void {
        const g = &self.g;
        const s = &self.set;
        try self.prepareDictionary(p, N);
        const total: u32 = @intCast(p.dict.len / 4);
        const px: u32 = fr.pat_h * fr.pat_w;
        const k: u32 = @min(s.pca_components, @min(px, total));
        const key: PcaKey = .{ .grid = N, .k = k, .epochs = s.pca_epochs, .level = lv, .serial = p.serial[@intFromEnum(lv)], .h = fr.pat_h, .w = fr.pat_w, .mask = fr.mask, .t = t, .c = .{ fr.c0[0], fr.c0[1], fr.c0[2], fr.c1[0], fr.c1[1], fr.c1[2], fr.c2[0], fr.c2[1], fr.c2[2] } };
        if (p.pca_key != null and std.meta.eql(p.pca_key.?, key)) return;
        p.pca_key = null;
        if (@as(u64, total) * k * 4 > @min(g.lim.max_storage_binding, g.lim.max_buffer)) return error.DictionaryTooLargeForPca;
        const m = p.det[@intFromEnum(lv)];
        // dictionary entries per step: about 64 MiB of simulated patterns
        const chunk: u32 = @min(total, @as(u32, @intCast(std.math.clamp((64 << 20) / (@as(u64, px) * 4), 256, 4096))));
        try self.pca.begin(&p.pca_w, k, px);
        var steps: u32 = 0;
        const all_steps = s.pca_epochs * ((total + chunk - 1) / chunk);
        for (0..s.pca_epochs) |_| {
            var d0: u32 = 0;
            while (d0 < total) : (d0 += chunk) {
                if (self.cancel) return error.Cancelled;
                const nb = @min(chunk, total - d0);
                steps += 1;
                // a short tail is no sample: it would pull the basis toward its few patterns
                if (nb < chunk / 2 and d0 != 0) continue;
                self.det.dictSims(m, fr, t, p.dict_gpu, nb, d0);
                try self.pca.learn(p.pca_w, self.det.sims, nb, k, px);
                self.events.emit(.{ .kind = "progress", .stage = "pca", .done = steps, .total = all_steps });
            }
        }
        g.ensure(&p.pca_coef, @as(u64, total) * k * 4);
        var d0: u32 = 0;
        while (d0 < total) : (d0 += chunk) {
            const nb = @min(chunk, total - d0);
            self.det.dictSims(m, fr, t, p.dict_gpu, nb, d0);
            self.pca.project(p.pca_w, self.det.sims, nb, k, px, p.pca_coef, d0);
        }
        // how much of the dictionary the components hold: a unit pattern's coefficients have
        // squared norm equal to the share of it they keep
        const ns: usize = @min(total, 2048);
        const head = try self.gpa.alloc(f32, ns * k);
        defer self.gpa.free(head);
        g.read(p.pca_coef, 0, std.mem.sliceAsBytes(head));
        try g.wait();
        var kept: f64 = 0;
        for (head) |v| kept += @as(f64, v) * v;
        p.pca_key = key;
        self.events.log("dictionary components: {d} of {d} pixels hold {d:.1} % of {d} patterns' variance ({d} steps of Oja's rule)", .{ k, px, 100 * kept / @as(f64, @floatFromInt(ns)), total, self.pca.steps });
    }

    // ── the batch ──

    /// What a search runs with outside a sweep: the settings' method, bandwidth and dictionary.
    pub fn searchOf(self: *const Engine) Search {
        return .{ .method = self.set.method, .bandwidth = self.set.bandwidth, .dict_grid = self.set.dict_grid };
    }

    /// Index the given cells: per phase, find each pattern's orientation (spherical correlation
    /// or the dictionary) at the search's level, refine it on the detector at the refinement's,
    /// score it (NCC, NMI) at the scores', and measure there what NMI chance gives. `out` gets
    /// one record per phase per cell ([phase][cell]). One readback.
    pub fn indexCells(self: *Engine, cells: []const u64, prm: Search, out: []Record) !void {
        const g = &self.g;
        const gpa = self.gpa;
        const s = &self.set;
        const n_ph = self.phases.items.len;
        const n: u32 = @intCast(cells.len);
        if (n_ph == 0) return error.NoPhases;
        if (out.len != n_ph * cells.len) return error.BadLength;
        if (n == 0) return;
        const ls = self.levelOf(.search);
        const lr = self.levelOf(.refine);
        const lc = self.levelOf(.score);
        self.det.chunk_px = s.refine_pixels;
        self.det.wg = s.wg_refine;
        const spherical_method = prm.method == .spherical;
        var bd: ?*Band = null;
        if (spherical_method) bd = try self.band(prm.bandwidth, ls);
        try self.prepareMasters(&.{ ls, lr, lc });
        // a dictionary is simulated at one detector position: the map's reference
        const t64 = try self.translationOf(&self.geo, ls);
        const t_avg: [3]f32 = .{ @floatCast(t64[0]), @floatCast(t64[1]), @floatCast(t64[2]) };
        // (its components are learned before the batch is recorded: learning waits on the device)
        if (prm.method == .pca) {
            for (self.phases.items) |*p| try self.preparePca(p, prm.dict_grid, ls, try self.frame(ls), t_avg);
        }
        // (what chance gives is measured once for the scan, before this batch's patterns go up:
        // it uses the same buffers)
        try self.calibrate();
        const account = self.set.nlpar and self.nl_report == .wanted;
        try self.uploadCellsAs(cells, &.{ ls, lr, lc }, .{ .account = account });
        if (account) self.nl_report = .queued;

        // the items at each level in use: the detector's position for each pattern, in pixels
        const items = try gpa.alloc(spherical.Item, n);
        defer gpa.free(items);
        var have = [2]bool{ false, false };
        for ([_]Level{ lr, lc, ls }) |lv| {
            const k = @intFromEnum(lv);
            if (have[k] and lv != ls) continue;
            for (cells, 0..) |c, b| {
                const t = try self.positionOf(&self.geo, c, lv);
                items[b] = .{ t[0], t[1], t[2], @floatFromInt(b) };
            }
            if (!have[k]) {
                g.ensure(&self.items_gpu[k], @as(u64, n) * 16);
                g.writeSlice(self.items_gpu[k], 0, spherical.Item, items);
            }
            have[k] = true;
        }
        // (the search's items were built last and stay on the host for its tiles)
        const fr_s = try self.frame(ls);
        const fr_r = try self.frame(lr);
        const fr_c = try self.frame(lc);
        g.ensure(&self.records, @as(u64, n_ph) * n * @sizeOf(Record));
        const refine: detector.RefineOptions = .{
            .iters = if (s.refine) s.refine_iters else 0,
            .max_step = @floatCast(std.math.degreesToRadians(s.refine_step)),
            .nmi_bins = s.nmi_bins,
        };

        if (spherical_method) {
            const b = bd.?;
            const sp = &self.sp;
            try self.prepareDenominators(b, ls, fr_s);
            var tile: u32 = n;
            for (b.ph.items) |*x| tile = @min(tile, sp.tileItems(&x.*.?.gpu, @as(u64, s.tile_mb) << 20));
            for (self.peaks.items[0..n_ph]) |*pk| g.ensure(pk, @as(u64, n) * 32);
            var t0: u32 = 0;
            while (t0 < n) : (t0 += tile) {
                const nt = @min(tile, n - t0);
                sp.coefficients(self.pats[@intFromEnum(ls)], items[t0..][0..nt], fr_s, spherical.FLAG_CENTER | spherical.FLAG_NORM, &sp.alm, &sp.stats);
                for (b.ph.items, 0..) |*x, pi| {
                    sp.correlate(&x.*.?.gpu, nt, x.*.?.gpu.rden.at(8), nt);
                    g.copy(sp.peaks, 0, self.peaks.items[pi], @as(u64, t0) * 32, @as(u64, nt) * 32);
                }
            }
        }
        for (self.phases.items, 0..) |*p, pi| {
            const first: u32 = @intCast(pi * n);
            if (spherical_method) {
                self.det.seed(fr_r, self.items_gpu[@intFromEnum(lr)], self.peaks.items[pi], 0, n);
            } else if (prm.method == .pca) {
                const k = p.pca_key.?.k;
                const total: u32 = @intCast(p.dict.len / 4);
                g.ensure(&self.pca.y, @as(u64, n) * k * 4);
                self.pca.project(p.pca_w, self.pats[@intFromEnum(ls)], n, k, fr_s.pat_h * fr_s.pat_w, self.pca.y, 0);
                try self.det.dictBegin(n);
                self.pca.match(p.pca_coef.at(1), self.pca.y, n, k, total, 0, self.det.dict_best);
                self.det.seedDict(fr_r, self.items_gpu[@intFromEnum(lr)], p.dict_gpu, n);
            } else {
                // entries per chunk: about 64 MiB of simulated patterns
                const chunk: u32 = @intCast(std.math.clamp((64 << 20) / (@as(u64, fr_s.pat_h) * fr_s.pat_w * 4), 64, 16384));
                try self.prepareDictionary(p, prm.dict_grid);
                const total: u32 = @intCast(p.dict.len / 4);
                try self.det.dictBegin(n);
                var d0: u32 = 0;
                while (d0 < total) : (d0 += chunk) {
                    self.det.dictChunk(p.det[@intFromEnum(ls)], fr_s, t_avg, self.pats[@intFromEnum(ls)], p.dict_gpu, @min(chunk, total - d0), d0, n);
                }
                self.det.seedDict(fr_r, self.items_gpu[@intFromEnum(lr)], p.dict_gpu, n);
            }
            self.det.refine(p.det[@intFromEnum(lr)], fr_r, self.pats[@intFromEnum(lr)], n, refine, self.records, first);
            if (lc != lr) {
                // the refined orientations scored again where the scores are taken
                self.det.seed(fr_c, self.items_gpu[@intFromEnum(lc)], self.records, first, n);
                var score = refine;
                score.iters = 0;
                self.det.refine(p.det[@intFromEnum(lc)], fr_c, self.pats[@intFromEnum(lc)], n, score, self.records, first);
            }
        }
        g.read(self.records, 0, std.mem.sliceAsBytes(out));
        // the device starts on this batch; the host reads the next one meanwhile
        if (self.ahead.len != 0) {
            g.submit();
            self.readAhead(self.ahead);
        }
        try g.wait();
        self.reportNlpar();
        // each match's NMI on the scan's own scale
        for (0..n_ph) |pi| {
            for (out[pi * n ..][0..n]) |*r| r.z = (r.nmi - self.chance_med[pi]) / self.chance_spread[pi];
        }
    }

    /// Measure, once for the scan as it is prepared now, what mutual information chance gives:
    /// a sample of the region's patterns (about 256 on a lattice) against each phase at random
    /// orientations, at the level the scores are taken. Its median and spread (1.4826 × the
    /// median deviation) per phase are the scale every match's NMI is then read on; they are
    /// measured again only when what the scores depend on changes (the patterns' preparation,
    /// the geometry, the phases, the region).
    /// The key to a phase's inverse pole figure colours, as the settings have them.
    pub fn ipfKeyOf(self: *const Engine, phase: usize) sym.Key {
        // a direction's colour follows the Laue class; an orientation's, the point group's own rotations
        const m = &self.phases.items[phase].master;
        return sym.Key.init(if (self.set.ipf_colors == .perfect_rotation) m.rotation else m.laue, self.set.ipf_colors, self.set.ipf_lightness, self.set.ipf_gray);
    }

    pub fn calibrate(self: *Engine) !void {
        const gpa = self.gpa;
        const g = &self.g;
        const s = &self.set;
        const n_ph = self.phases.items.len;
        if (n_ph == 0) return error.NoPhases;
        // the key: the settings with those that do not touch a score put aside
        var k = s.*;
        inline for (.{ "batch", "tile_mb", "refine_pixels", "wg_fft", "wg_refine", "sweeps", "sweep_bandwidth", "sweep_dict_grid", "bandwidth", "dict_grid", "fit_bandwidth", "fit_grid", "pose_grid", "refine_iters", "pc_steps", "pc_steps_z", "pc_levels", "pc_top_k", "pca_components", "pca_epochs", "pose_iters", "pc_nms", "toy_seed", "toy_rows", "toy_cols", "toy_height", "toy_width", "toy_grains", "toy_noise_seed" }) |f| @field(k, f) = 0;
        inline for (.{ "toy_counts", "toy_spread_deg", "toy_blank", "toy_background_sigma" }) |f| @field(k, f) = 0;
        k.toy_contrast = 1;
        inline for (.{ "accept_z", "accept_nmi", "accept_step_deg", "refine_step", "pc_span", "pc_span_z", "pc_shrink", "pc_gate", "pc_kde_sigma", "pc_inlier_radius", "pc_z_min" }) |f| @field(k, f) = 0;
        k.method = .spherical;
        k.sweep_method = .same;
        k.pc_vote = .ballots;
        k.refine = false;
        inline for (.{ "pose_x", "pose_y", "pose_z", "pose_tilt", "pose_swing", "pose_twist", "pose_scale" }) |f| @field(k, f) = false;
        k.ipf_colors = .classic;
        k.ipf_lightness = 0;
        k.ipf_gray = 0;
        const json = try k.toJson(gpa);
        defer gpa.free(json);
        var h = std.hash.Wyhash.init(n_ph);
        h.update(json);
        h.update(std.mem.asBytes(&self.geo));
        for (self.phases.items) |*p| h.update(std.mem.asBytes(&p.serial));
        const key = h.final() | 1;
        if (key == self.chance_key and self.chance_med.len == n_ph) return;
        self.chance_key = 0;

        const region = try self.regionCells();
        defer gpa.free(region);
        const n: u32 = @intCast(@min(region.len, 256));
        const cells = try gpa.alloc(u64, n);
        defer gpa.free(cells);
        for (cells, 0..) |*c, j| c.* = region[(2 * j + 1) * region.len / (2 * @as(usize, n))];
        const draws: u32 = 8;
        const lc = self.levelOf(.score);
        try self.prepareMasters(&.{lc});
        // (not a batch of indexing: nothing read ahead is its, and none of its account)
        const ahead = self.ahead;
        self.ahead = &.{};
        defer self.ahead = ahead;
        try self.uploadCellsAs(cells, &.{lc}, .{});
        const fr = try self.frame(lc);
        // each sample pattern where the detector stood for it; chance supplies the orientations
        const poses = try gpa.alloc(detector.Pose, n);
        defer gpa.free(poses);
        for (cells, 0..) |c, b| poses[b] = .{ .t = try self.positionOf(&self.geo, c, lc), .pat = @floatFromInt(b), .q = .{ 1, 0, 0, 0 } };
        self.det.setPoses(poses);
        const per = @as(usize, n) * draws;
        g.ensure(&self.chance, @as(u64, n_ph) * per * 4);
        for (self.phases.items, 0..) |*p, pi| {
            self.det.chanceOf(p.det[@intFromEnum(lc)], fr, self.pats[@intFromEnum(lc)], n, draws, s.nmi_bins);
            g.copy(self.det.chance, 0, self.chance, @as(u64, pi) * per * 4, per * 4);
        }
        const host = try gpa.alloc(f32, n_ph * per);
        defer gpa.free(host);
        g.read(self.chance, 0, std.mem.sliceAsBytes(host));
        try g.wait();
        if (self.chance_med.len != n_ph) {
            gpa.free(self.chance_med);
            gpa.free(self.chance_spread);
            self.chance_med = &.{};
            self.chance_spread = &.{};
            self.chance_med = try gpa.alloc(f32, n_ph);
            self.chance_spread = try gpa.alloc(f32, n_ph);
        }
        for (0..n_ph) |pi| {
            const c = host[pi * per ..][0..per];
            std.mem.sort(f32, c, {}, std.sort.asc(f32));
            const med = c[per / 2];
            for (c) |*v| v.* = @abs(v.* - med);
            std.mem.sort(f32, c, {}, std.sort.asc(f32));
            self.chance_med[pi] = med;
            self.chance_spread[pi] = @max(1.4826 * c[per / 2], 1e-6);
            self.events.log("chance, phase {d}: NMI {d:.4} ± {d:.4} over {d} patterns at random orientations; z {d:.0} is NMI {d:.4}", .{ pi, med, self.chance_spread[pi], n, s.accept_z, med + s.accept_z * self.chance_spread[pi] });
        }
        self.chance_key = key;
    }

    /// The phase a pattern is assigned among its per-phase records.
    pub fn choosePhase(self: *const Engine, recs: []const Record, n: usize, b: usize) usize {
        var best: usize = 0;
        var best_v: f32 = -std.math.inf(f32);
        for (0..recs.len / n) |p| {
            const r = recs[p * n + b];
            const v = if (self.set.phase_by == .nmi) r.nmi else r.ncc;
            if (v > best_v) {
                best_v = v;
                best = p;
            }
        }
        return best;
    }

    // ── indexing a region, a batch per step, in sweeps ──

    /// The cells of the region of interest (the whole map when it is empty), row-major.
    pub fn regionCells(self: *Engine) ![]u64 {
        const i = try self.info();
        const s = &self.set;
        var x0: u32 = 0;
        var y0: u32 = 0;
        var w: u32 = i.cols;
        var h: u32 = i.rows;
        if (s.roi_w != 0 and s.roi_h != 0) {
            x0 = @min(s.roi_x, i.cols - 1);
            y0 = @min(s.roi_y, i.rows - 1);
            w = @min(s.roi_w, i.cols - x0);
            h = @min(s.roi_h, i.rows - y0);
        }
        const cells = try self.gpa.alloc(u64, @as(usize, w) * h);
        for (0..h) |r| for (0..w) |c| {
            cells[r * w + c] = @as(u64, y0 + r) * i.cols + x0 + c;
        };
        return cells;
    }

    pub fn indexBegin(self: *Engine) !void {
        if (self.phases.items.len == 0) return error.NoPhases;
        const cells = try self.regionCells();
        self.gpa.free(self.index_cells);
        self.index_cells = cells;
        self.nl_report = if (self.set.nlpar) .wanted else .off;
        if (self.set.nlpar) try self.orderInBlocks(cells);
        self.index_done = 0;
        self.index_region = cells.len;
        self.index_sweep = 0;
        self.index_accepted = 0;
        self.cancel = false;
        for (cells) |c| {
            self.res.phase[@intCast(c)] = -1;
            self.res.sweep[@intCast(c)] = -1;
            self.res.z[@intCast(c)] = 0;
        }
    }

    /// Reorder the region's cells (a rectangle, row by row) into square blocks of about a
    /// batch each: a batch averaged with its neighbours reads a margin of cells around itself,
    /// which is small beside a block and many times a strip of rows.
    fn orderInBlocks(self: *Engine, cells: []u64) !void {
        const i = try self.info();
        if (cells.len == 0) return;
        const w: usize = @intCast(cells[cells.len - 1] % i.cols - cells[0] % i.cols + 1);
        const h = cells.len / w;
        const side: usize = @intFromFloat(@ceil(@sqrt(@as(f64, @floatFromInt(self.set.batch)))));
        const rowwise = try self.gpa.dupe(u64, cells);
        defer self.gpa.free(rowwise);
        var k: usize = 0;
        var by: usize = 0;
        while (by < h) : (by += side) {
            var bx: usize = 0;
            while (bx < w) : (bx += side) {
                for (by..@min(by + side, h)) |y| for (bx..@min(bx + side, w)) |x| {
                    cells[k] = rowwise[y * w + x];
                    k += 1;
                };
            }
        }
    }

    pub fn indexStatus(self: *const Engine) IndexStatus {
        const sw = self.set.sweep(self.index_sweep);
        return .{
            .sweep = self.index_sweep,
            .sweeps = self.set.sweeps,
            .bandwidth = sw.bandwidth,
            .dict_grid = sw.dict_grid,
            .done = self.index_done,
            .total = self.index_cells.len,
            .region = self.index_region,
            .accepted = self.index_accepted,
            .method = @intFromEnum(sw.method),
        };
    }

    /// Index the next batch of the sweep in progress; when the sweep ends, the cells without an
    /// accepted match become the next one's. Returns the cells left to visit (0: done). A match
    /// is accepted when it stands far enough above chance (settings.accept_z) and its
    /// refinement had settled (settings.accept_step_deg). Every cell keeps the best match any
    /// sweep gave it, accepted or not.
    pub fn indexStep(self: *Engine) !usize {
        const s = &self.set;
        const left = self.index_cells.len - self.index_done;
        if (left == 0 or self.cancel) return 0;
        const n = @min(left, s.batch);
        const cells = self.index_cells[self.index_done..][0..n];
        const n_ph = self.phases.items.len;
        const recs = try self.gpa.alloc(Record, n_ph * n);
        defer self.gpa.free(recs);
        const sw = s.sweep(self.index_sweep);
        self.ahead = self.index_cells[self.index_done + n ..][0..@min(left - n, s.batch)];
        defer self.ahead = &.{};
        try self.indexCells(cells, .{ .method = sw.method, .bandwidth = sw.bandwidth, .dict_grid = sw.dict_grid }, recs);
        const settle: f32 = @floatCast(std.math.degreesToRadians(s.accept_step_deg));
        for (cells, 0..) |c, b| {
            const p = self.choosePhase(recs, n, b);
            const r = recs[p * n + b];
            const at: usize = @intCast(c);
            // an acceptable match replaces whatever was kept; otherwise the better one stays
            const good = if (s.accept_nmi > 0) r.nmi >= s.accept_nmi else r.z >= s.accept_z;
            if (self.res.phase[at] < 0 or (good and r.step <= settle) or r.z > self.res.z[at]) {
                const q = sym.toFz(.{ r.q[0], r.q[1], r.q[2], r.q[3] }, self.phases.items[p].master.rotation);
                for (0..4) |k| self.res.quat[4 * at + k] = @floatCast(q[k]);
                self.res.ncc[at] = r.ncc;
                self.res.nmi[at] = r.nmi;
                self.res.z[at] = r.z;
                self.res.phase[at] = @intCast(p);
                self.res.settled[at] = r.step <= settle;
            }
            const kept_good = if (s.accept_nmi > 0) self.res.nmi[at] >= s.accept_nmi else self.res.z[at] >= s.accept_z;
            if (self.res.sweep[at] < 0 and kept_good and self.res.settled[at]) {
                self.res.sweep[at] = @intCast(self.index_sweep);
                self.index_accepted += 1;
            }
        }
        self.index_done += n;
        self.events.emit(.{ .kind = "progress", .stage = "index", .done = self.index_done, .total = self.index_cells.len, .sweep = self.index_sweep, .sweeps = s.sweeps });
        if (self.index_done == self.index_cells.len and self.index_sweep + 1 < s.sweeps) {
            // the next sweep takes what this one left
            var keep: usize = 0;
            for (self.index_cells) |c| {
                if (self.res.sweep[@intCast(c)] >= 0) continue;
                self.index_cells[keep] = c;
                keep += 1;
            }
            self.events.log("sweep {d} of {d} ({s} {d}): {d} of {d} patterns accepted, {d} left", .{
                self.index_sweep + 1,                                             s.sweeps,
                switch (sw.method) {
                    .spherical => "spherical, bandwidth",
                    .dictionary => "dictionary, grid",
                    .pca => "dictionary components, grid",
                },
                if (sw.method == .spherical) sw.bandwidth else sw.dict_grid,
                self.index_cells.len - keep,                                      self.index_cells.len,
                keep,
            });
            if (keep != 0) {
                self.index_cells = try self.gpa.realloc(self.index_cells, keep);
                self.index_done = 0;
                self.index_sweep += 1;
            }
        }
        return self.index_cells.len - self.index_done;
    }

    // ── tuning ──

    fn tuneFree(self: *Engine) void {
        for (self.tune.cells) |c| self.gpa.free(c);
        self.tune = .{};
    }

    fn tuneValues(self: *const Engine) [Tune.count]u32 {
        return .{ self.set.batch, self.set.refine_pixels, self.set.wg_refine, self.set.wg_fft };
    }

    fn tuneApply(self: *Engine, v: [Tune.count]u32) void {
        self.set.batch = v[0];
        self.set.refine_pixels = v[1];
        self.set.wg_refine = v[2];
        self.set.wg_fft = v[3];
    }

    /// Whether the last sweep searches otherwise than the first (and so is timed on its own).
    fn tuneTwoSweeps(self: *const Engine) bool {
        const a = self.set.sweep(0);
        const b = self.set.sweep(self.set.sweeps - 1);
        return a.method != b.method or a.bandwidth != b.bandwidth or a.dict_grid != b.dict_grid;
    }

    /// The first `want` or so cells of the region as a rectangle from its first cell: whole
    /// rows, as indexing reads them; or, when patterns are averaged with their neighbours, as
    /// square as the region lets it be, as indexing's blocks are. Caller frees.
    fn tuneCells(self: *Engine, want: usize) ![]u64 {
        const gpa = self.gpa;
        const region = try self.regionCells();
        defer gpa.free(region);
        const i = try self.info();
        const w: usize = @intCast(region[region.len - 1] % i.cols - region[0] % i.cols + 1);
        const h = region.len / w;
        const side: usize = @intFromFloat(@ceil(@sqrt(@as(f64, @floatFromInt(want)))));
        const across = @min(w, if (self.set.nlpar) side else want);
        const down = std.math.clamp(want / across, 1, h);
        const cells = try gpa.alloc(u64, across * down);
        for (0..down) |y| @memcpy(cells[y * across ..][0..across], region[y * w ..][0..across]);
        return cells;
    }

    /// Whether a set of values can be tried on this scan: each in its range, and a batch other
    /// than the one held only if the sample holds it twice over (a single batch never waits on
    /// the next one's reading) and one buffer holds its patterns.
    fn tuneFits(self: *const Engine, v: [Tune.count]u32) bool {
        for (v, Tune.lo, Tune.hi) |x, a, b| if (x < a or x > b) return false;
        if (v[0] == self.tune.best[0]) return true;
        const i = self.info() catch return false;
        const bytes = @as(u64, v[0]) * i.pat_h * i.pat_w * 4;
        return 2 * @as(usize, v[0]) <= self.tune.cells[0].len and bytes <= @min(self.g.lim.max_storage_binding, self.g.lim.max_buffer) / 2;
    }

    /// Start tuning: take the samples. The first trial times the settings as they are.
    pub fn tuneBegin(self: *Engine) !void {
        if (self.phases.items.len == 0) return error.NoPhases;
        self.tuneFree();
        const t = &self.tune;
        const a = self.set.sweep(0);
        const b = self.set.sweep(self.set.sweeps - 1);
        t.cells[0] = try self.tuneCells(Tune.sampleOf(a.method, a.bandwidth, a.dict_grid));
        t.cells[1] = try self.tuneCells(Tune.sampleOf(b.method, b.bandwidth, b.dict_grid));
        t.best = self.tuneValues();
        t.first = t.best;
        t.active = true;
        self.cancel = false;
    }

    /// Run the trial in hand: index its sample with its settings as indexing would a sweep of
    /// it (batches, reading ahead), keeping nothing. The front end times this call. A trial's
    /// first call indexes one batch only, to settle what a new value needs built.
    pub fn tuneTrial(self: *Engine) !void {
        const t = &self.tune;
        if (!t.active) return error.NotTuning;
        self.tuneApply(if (t.base) t.best else t.cand);
        defer self.tuneApply(t.best);
        const gpa = self.gpa;
        const batch: usize = self.set.batch;
        const sample = t.cells[@intFromBool(t.last)];
        const cells = try gpa.dupe(u64, if (t.run == 0) sample[0..@min(batch, sample.len)] else sample);
        defer gpa.free(cells);
        if (self.set.nlpar and t.run != 0) try self.orderInBlocks(cells);
        const n_ph = self.phases.items.len;
        const recs = try gpa.alloc(Record, n_ph * @min(batch, cells.len));
        defer gpa.free(recs);
        const sw = self.set.sweep(if (t.last) self.set.sweeps - 1 else 0);
        defer self.ahead = &.{};
        var at: usize = 0;
        while (at < cells.len) {
            if (self.cancel) return error.Cancelled;
            const n = @min(batch, cells.len - at);
            self.ahead = cells[at + n ..][0..@min(batch, cells.len - at - n)];
            try self.indexCells(cells[at..][0..n], .{ .method = sw.method, .bandwidth = sw.bandwidth, .dict_grid = sw.dict_grid }, recs[0 .. n_ph * n]);
            at += n;
        }
    }

    /// The next candidate worth a trial, into `cand`: the values held with the move in hand
    /// applied (its settings doubled, or halved once doubling did not pay), on to the next
    /// move when neither can be. False: no move is left.
    fn tuneNext(self: *Engine) bool {
        const t = &self.tune;
        while (t.move < Tune.moves.len) {
            if (t.move >= Tune.on_last and !t.last and self.tuneTwoSweeps()) {
                // from here on the last sweep is what is timed: first the values held, on it
                t.last = true;
                t.base = true;
                return true;
            }
            var c = t.best;
            for (&c, Tune.moves[t.move]) |*x, takes_part| {
                if (takes_part) x.* = if (t.up) x.* * 2 else x.* / 2;
            }
            if (self.tuneFits(c)) {
                t.cand = c;
                return true;
            }
            self.tuneTurn();
        }
        return false;
    }

    /// The candidate in this direction was not kept: try the other way if this was the move's
    /// first step up, else leave the move.
    fn tuneTurn(self: *Engine) void {
        const t = &self.tune;
        if (t.up and !t.kept) {
            t.up = false;
            return;
        }
        t.move += 1;
        t.up = true;
        t.kept = false;
    }

    /// Take the seconds the last call of `tuneTrial` took (negative: it failed) and move on.
    /// Returns whether a call is left; when none is, the settings hold the fastest values found.
    pub fn tuneReport(self: *Engine, seconds: f64) bool {
        const t = &self.tune;
        if (!t.active) return false;
        const failed = !(seconds >= 0);
        if (!failed) {
            // (the first call of a trial is one batch, untimed)
            if (t.run > 0) t.time = @min(t.time, seconds);
            t.run += 1;
            if (t.run <= Tune.runs) return true;
        }
        const which: usize = @intFromBool(t.last);
        t.trials += 1;
        if (t.base) {
            if (failed) {
                // the settings as they are do not run: nothing to tune from
                t.active = false;
                return false;
            }
            t.best_time = t.time;
            t.was[which] = t.time;
            t.base = false;
        } else if (!failed and t.time < t.best_time * (1 - Tune.worth)) {
            t.best_time = t.time;
            t.best = t.cand;
            t.kept = true;
        } else {
            self.tuneTurn();
        }
        t.now[which] = t.best_time;
        t.run = 0;
        t.time = std.math.inf(f64);
        const more = self.tuneNext();
        self.tuneApply(t.best);
        self.events.emit(.{ .kind = "progress", .stage = "tune", .done = @min(t.move, Tune.moves.len), .total = Tune.moves.len });
        if (more) return true;
        t.active = false;
        var buf: [320]u8 = undefined;
        var w: std.Io.Writer = .fixed(&buf);
        for (Tune.names, t.first, t.best) |name, a, b| {
            if (a == b) w.print(", {s} {d}", .{ name, a }) catch {} else w.print(", {s} {d} → {d}", .{ name, a, b }) catch {};
        }
        const n0: f64 = @floatFromInt(t.cells[0].len);
        const n1: f64 = @floatFromInt(t.cells[1].len);
        if (t.was[1] > 0) w.print("; first sweep {d:.0} → {d:.0} patterns per second, last (on {d}) {d:.0} → {d:.0}", .{ n0 / t.was[0], n0 / t.now[0], t.cells[1].len, n1 / t.was[1], n1 / t.now[1] }) catch {} else w.print("; {d:.0} → {d:.0} patterns per second", .{ n0 / t.was[0], n0 / t.now[0] }) catch {};
        self.events.log("tuned on {d} patterns in {d} trials{s}", .{ t.cells[0].len, t.trials, w.buffered() });
        return false;
    }

    // ── looking at things ──

    /// How far along the pipeline a pattern is looked at.
    pub const View = enum(u32) {
        /// as stored (reduced, at a reduced level)
        stored = 0,
        /// averaged with its neighbours (when that is on) and filtered: what is indexed
        filtered = 1,
        /// averaged with its neighbours (when that is on), before any filter
        averaged = 2,
    };

    /// One pattern as f32 at a level, at a stage of the pipeline. When it was averaged with
    /// its neighbours, `nl_view` and `nl_view_shares` then say what that did to it.
    pub fn pattern(self: *Engine, cell: u64, view: View, lv: Level, out: []f32) !void {
        const hw = try self.patternShape(lv);
        if (out.len != @as(usize, hw[0]) * hw[1]) return error.BadLength;
        const g = &self.g;
        const averaged = self.set.nlpar and view != .stored;
        self.nl_view_side = 0;
        if (view == .stored) {
            const bytes = try self.readCells(&.{cell}, false);
            try self.uploadLevel(lv, &.{cell}, bytes, null);
        } else {
            try self.uploadCellsAs(&.{cell}, &.{lv}, .{ .filter = view == .filtered, .account = averaged });
        }
        g.read(self.pats[@intFromEnum(lv)], 0, std.mem.sliceAsBytes(out));
        try g.wait();
        if (!averaged) return;
        self.nl_view = self.nlparSummary();
        const side = 2 * nlpar_mod.Window.reachOf(self.set.nlpar_radius, self.set.nlpar_iterations) + 1;
        const f = @as(usize, side) * side;
        if (self.nl_view_shares.len != f) {
            self.gpa.free(self.nl_view_shares);
            self.nl_view_shares = &.{};
            self.nl_view_shares = try self.gpa.alloc(f32, f);
        }
        // (a cell outside the map has no share: NaN, to tell it from a neighbour given none)
        for (self.nl_view_shares, self.nl_shares[0..f], self.nl_where[0..f]) |*o, a, at| o.* = if (at == nlpar_mod.NONE) std.math.nan(f32) else a;
        self.nl_view_side = side;
    }

    /// Mean and standard deviation of each given cell's stored pattern (2 f32 per cell), on the
    /// host: a map of either shows the microstructure before anything is indexed.
    pub fn patternStats(self: *Engine, cells: []const u64, out: []f32) !void {
        if (out.len != 2 * cells.len) return error.BadLength;
        const i = try self.info();
        const total = @as(u64, i.rows) * i.cols;
        for (cells) |c| if (c >= total) return error.OutOfRange;
        const bytes = try self.readCells(cells, false);
        const px = @as(usize, i.pat_h) * i.pat_w;
        for (0..cells.len) |k| {
            var sum: f64 = 0;
            var sq: f64 = 0;
            switch (i.pixel) {
                .u8 => for (bytes[k * px ..][0..px]) |v| {
                    const f: f64 = @floatFromInt(v);
                    sum += f;
                    sq += f * f;
                },
                .u16 => for (0..px) |j| {
                    const f: f64 = @floatFromInt(std.mem.readInt(u16, bytes[(k * px + j) * 2 ..][0..2], .little));
                    sum += f;
                    sq += f * f;
                },
                .f32 => for (0..px) |j| {
                    const f: f64 = @as(f32, @bitCast(std.mem.readInt(u32, bytes[(k * px + j) * 4 ..][0..4], .little)));
                    sum += f;
                    sq += f * f;
                },
            }
            const n: f64 = @floatFromInt(px);
            const mean = sum / n;
            out[2 * k] = @floatCast(mean);
            out[2 * k + 1] = @floatCast(@sqrt(@max(sq / n - mean * mean, 0)));
        }
    }

    /// Simulated patterns of a phase at the given orientations, each as cell `cells[k]` would
    /// record it at a level, from the master as filtered for that level.
    pub fn simulate(self: *Engine, phase: u32, cells: []const u64, quats: []const f32, lv: Level, out: []f32) !void {
        if (phase >= self.phases.items.len) return error.NoSuchPhase;
        const hw = try self.patternShape(lv);
        const n = cells.len;
        if (quats.len != 4 * n or out.len != n * hw[0] * hw[1]) return error.BadLength;
        try self.prepareMasters(&.{lv});
        const fr = try self.frame(lv);
        const poses = try self.gpa.alloc(detector.Pose, n);
        defer self.gpa.free(poses);
        for (cells, 0..) |c, b| {
            poses[b] = .{ .t = try self.positionOf(&self.geo, c, lv), .pat = 0, .q = quats[4 * b ..][0..4].* };
        }
        self.det.setPoses(poses);
        self.det.project(self.phases.items[phase].det[@intFromEnum(lv)], fr, @intCast(n));
        self.g.read(self.det.sims, 0, std.mem.sliceAsBytes(out));
        try self.g.wait();
    }

    /// A toy scan from the phases loaded (settings toy_*), in place of whatever scan was open:
    /// grains by nearest of toy_grains random points, each with a random orientation and
    /// phase, every cell simulated as it would be recorded under the present geometry, laid on
    /// a background and counted as electrons. The microstructure comes from toy_seed and the
    /// counting from toy_noise_seed; the patterns themselves are the device's arithmetic.
    /// The answer goes to `quats` (4 per cell: w, x, y, z), `phase` (−1: a blank cell) and
    /// `grain` (the cell's grain).
    pub fn synthesize(self: *Engine, quats: []f32, phase: []i32, grain: []i32) !void {
        const gpa = self.gpa;
        const s = self.set;
        const n_ph = self.phases.items.len;
        if (n_ph == 0) return error.NoPhases;
        const n: usize = @as(usize, s.toy_rows) * s.toy_cols;
        const px: usize = @as(usize, s.toy_height) * s.toy_width;
        if (quats.len != 4 * n or phase.len != n or grain.len != n) return error.BadLength;
        const data = try gpa.alloc(f32, n * px);
        defer gpa.free(data);
        @memset(data, 0);
        // an empty scan of the shape first: a cell's place in it is part of its geometry
        try self.setPatterns(std.mem.sliceAsBytes(data), .f32, s.toy_rows, s.toy_cols, s.toy_height, s.toy_width);

        // the microstructure
        var structure = std.Random.DefaultPrng.init(s.toy_seed);
        const r = structure.random();
        const Grain = struct { x: f64, y: f64, q: rot.Quat, phase: i32 };
        const grains = try gpa.alloc(Grain, @min(@as(usize, s.toy_grains), n));
        defer gpa.free(grains);
        for (grains) |*gr| {
            gr.x = r.float(f64) * @as(f64, @floatFromInt(s.toy_cols));
            gr.y = r.float(f64) * @as(f64, @floatFromInt(s.toy_rows));
            // four normal numbers, normalized: uniform over the rotations
            gr.q = rot.qstd(.{ r.floatNorm(f64), r.floatNorm(f64), r.floatNorm(f64), r.floatNorm(f64) });
            gr.phase = @intCast(r.uintLessThan(usize, n_ph));
        }
        // (a rotation vector of three equal normal components has their root-sum-square for its angle)
        const spread = std.math.degreesToRadians(s.toy_spread_deg) / @sqrt(3.0);
        for (0..s.toy_rows) |row| for (0..s.toy_cols) |col| {
            const x = @as(f64, @floatFromInt(col)) + 0.5;
            const y = @as(f64, @floatFromInt(row)) + 0.5;
            var best: usize = 0;
            var best_d: f64 = std.math.inf(f64);
            for (grains, 0..) |gr, i| {
                const d = (gr.x - x) * (gr.x - x) + (gr.y - y) * (gr.y - y);
                if (d < best_d) {
                    best_d = d;
                    best = i;
                }
            }
            const at = row * s.toy_cols + col;
            var q = grains[best].q;
            if (spread > 0) q = rot.qstd(rot.qmul(rot.so3Exp(.{ spread * r.floatNorm(f64), spread * r.floatNorm(f64), spread * r.floatNorm(f64) }), q));
            quats[4 * at ..][0..4].* = .{ @floatCast(q[0]), @floatCast(q[1]), @floatCast(q[2]), @floatCast(q[3]) };
            grain[at] = @intCast(best);
            phase[at] = if (s.toy_blank > 0 and r.float(f64) < s.toy_blank) -1 else grains[best].phase;
        };

        // each phase's cells, a readback at a time
        const lv = self.viewLevel(true);
        const chunk: usize = @max(1, (4 << 20) / px);
        const cells = try gpa.alloc(u64, chunk);
        defer gpa.free(cells);
        const qs = try gpa.alloc(f32, 4 * chunk);
        defer gpa.free(qs);
        const sims = try gpa.alloc(f32, chunk * px);
        defer gpa.free(sims);
        var made: usize = 0;
        for (0..n_ph) |p| {
            var at: usize = 0;
            while (at < n) {
                var k: usize = 0;
                while (at < n and k < chunk) : (at += 1) {
                    if (phase[at] != @as(i32, @intCast(p))) continue;
                    cells[k] = at;
                    qs[4 * k ..][0..4].* = quats[4 * at ..][0..4].*;
                    k += 1;
                }
                if (k == 0) continue;
                if (self.cancel) return error.Cancelled;
                try self.simulate(@intCast(p), cells[0..k], qs[0 .. 4 * k], lv, sims[0 .. k * px]);
                for (cells[0..k], 0..) |c, i| @memcpy(data[@intCast(c * px)..][0..px], sims[i * px ..][0..px]);
                made += k;
                self.events.emit(.{ .kind = "progress", .stage = "synthesize", .done = made, .total = n });
            }
        }

        // the detector: the simulation (its darkest pixel 0, its mean 1) on a background (mean 1)
        var mean: f64 = 0;
        var low: f64 = std.math.inf(f64);
        var crystals: usize = 0;
        for (0..n) |c| {
            if (phase[c] < 0) continue;
            crystals += 1;
            for (data[c * px ..][0..px]) |v| {
                mean += v;
                low = @min(low, v);
            }
        }
        if (crystals > 0) mean /= @floatFromInt(crystals * px);
        const gain = if (crystals > 0 and mean > low) 1.0 / (mean - low) else 0;
        const back = try gpa.alloc(f64, px);
        defer gpa.free(back);
        @memset(back, 1);
        if (s.toy_background_sigma > 0) {
            const sg = s.toy_background_sigma * @as(f64, @floatFromInt(s.toy_width));
            var sum: f64 = 0;
            for (0..s.toy_height) |yy| for (0..s.toy_width) |xx| {
                const dx = @as(f64, @floatFromInt(xx)) + 0.5 - 0.5 * @as(f64, @floatFromInt(s.toy_width));
                const dy = @as(f64, @floatFromInt(yy)) + 0.5 - 0.5 * @as(f64, @floatFromInt(s.toy_height));
                const v = @exp(-(dx * dx + dy * dy) / (2 * sg * sg));
                back[yy * s.toy_width + xx] = v;
                sum += v;
            };
            for (back) |*v| v.* *= @as(f64, @floatFromInt(px)) / sum;
        }
        // counted: the mean pixel toy_counts electrons, every pixel a Poisson draw at its own intensity
        var counting = std.Random.DefaultPrng.init(if (s.toy_noise_seed != 0) s.toy_noise_seed else @as(u64, s.toy_seed) ^ 0x9E3779B97F4A7C15);
        const rn = counting.random();
        for (0..n) |c| {
            const blank = phase[c] < 0 or gain == 0;
            for (data[c * px ..][0..px], back) |*v, bg| {
                const signal = if (blank) 1.0 else (v.* - low) * gain;
                const intensity = (1 - s.toy_contrast) * bg + s.toy_contrast * signal;
                v.* = if (s.toy_counts > 0) @floatFromInt(poisson(rn, s.toy_counts * intensity)) else @floatCast(intensity);
            }
        }
        try self.setPatterns(std.mem.sliceAsBytes(data), .f32, s.toy_rows, s.toy_cols, s.toy_height, s.toy_width);
        self.events.log("toy scan: {d} × {d} patterns of {d} × {d}, {d} grains of {d} phase(s), {d} blank cells, contrast {d:.2}, {d:.4} electrons a pixel, seed {d}", .{ s.toy_rows, s.toy_cols, s.toy_height, s.toy_width, grains.len, n_ph, n - crystals, s.toy_contrast, s.toy_counts, s.toy_seed });
    }

    /// A phase's master as the search compares patterns with it (filtered for the search's
    /// level): [north | south], side × side each.
    pub fn masterImage(self: *Engine, phase: u32, out: []f32) !void {
        if (phase >= self.phases.items.len) return error.NoSuchPhase;
        const lv = self.levelOf(.search);
        try self.prepareMasters(&.{lv});
        const work = self.phases.items[phase].work[@intFromEnum(lv)];
        if (out.len != work.len) return error.BadLength;
        @memcpy(out, work);
    }
};

/// A Poisson draw of mean `lam`: by multiplying uniform numbers for small means (Knuth), by
/// transformed rejection for larger ones (Hörmann 1993, PTRS).
fn poisson(r: std.Random, lam: f64) u32 {
    if (!(lam > 0)) return 0;
    if (lam < 10) {
        const limit = @exp(-lam);
        var k: u32 = 0;
        var p: f64 = r.float(f64);
        while (p > limit) : (k += 1) p *= r.float(f64);
        return k;
    }
    const slam = @sqrt(lam);
    const b = 0.931 + 2.53 * slam;
    const a = -0.059 + 0.02483 * b;
    const inv_alpha = 1.1239 + 1.1328 / (b - 3.4);
    const vr = 0.9277 - 3.6224 / (b - 2);
    while (true) {
        const u = r.float(f64) - 0.5;
        const v = r.float(f64);
        const us = 0.5 - @abs(u);
        const kf = @floor((2 * a / us + b) * u + lam + 0.43);
        if (us >= 0.07 and v <= vr) return @intFromFloat(kf);
        if (kf < 0 or (us < 0.013 and v > us)) continue;
        if (@log(v) + @log(inv_alpha) - @log(a / (us * us) + b) <= -lam + kf * @log(lam) - std.math.lgamma(f64, kf + 1)) return @intFromFloat(kf);
    }
}

fn bytesPerPixel(pixel: scan_mod.Pixel) usize {
    return switch (pixel) {
        .u8 => 1,
        .u16 => 2,
        .f32 => 4,
    };
}
