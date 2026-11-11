//! The plane of source points: the geometry's fit after the pattern-centre vote.
//!
//! Winkelmann, Nolze, Cios, Tokarski and Bała (Materials 13 (2020) 2816), and kikuchipy after
//! them (refine_orientation_projection_center, then EBSDDetector.fit_pc): each of a set of
//! patterns is fitted by itself, its orientation and where its own source stood against the
//! detector, and a transformation of the scan's indices is fitted through those source points
//! (plane.zig). Nothing is asked of the scan's steps, the detector's pixel size or the
//! sample's tilt: they are what the plane says.
//!
//! The fit patterns are a lattice over the region (pcfit.fitCellsEvery: every
//! settings.plane_stride-th cell each way), indexed at the geometry as it stands; those whose
//! match indexing itself would not accept (settings.accept_nmi or accept_z, and
//! accept_step_deg) are left out: a pattern matched to the wrong orientation puts its source
//! wherever suits its mistake. Each of the others is refined by itself over six unknowns, a
//! shift of its rays' origin and a turn of its crystal, by Levenberg–Marquardt on its own
//! normalized cross-correlation with its simulation. The device supplies, per pattern, that
//! correlation's gradient and Gauss–Newton matrix over those six directions (detector.wgsl
//! `sums`); all the patterns take a step together, one readback a step.
//!
//! `auto` is the automatic fit: the vote for the pattern centre (pcfit), then the plane.
const std = @import("std");
const engine = @import("engine.zig");
const rot = @import("rot.zig");
const detector = @import("detector.zig");
const pcfit = @import("pcfit.zig");
const Geometry = @import("geometry.zig").Geometry;
const plane_mod = @import("plane.zig");

const Engine = engine.Engine;
const NS = detector.NS;

/// A source point the plane fit measured on one fit pattern.
pub const PlanePoint = extern struct {
    cell: u64 = 0,
    /// where that pattern's own fit put its source, in pattern-centre coordinates
    pc: [3]f64 = .{ 0, 0, 0 },
    /// where the fitted plane has it
    fit: [3]f64 = .{ 0, 0, 0 },
    /// the pattern's score at its own best
    ncc: f64 = 0,
    /// 1: it took part in the plane's fit; 0: left out (its match not accepted, at the edge
    /// of its trust region, or too far from the plane)
    used: u32 = 0,
    _pad: u32 = 0,
};

pub const PlaneResult = extern struct {
    /// P, 4 × 3 row-major: P · (column, row, 1) = w · (PCx, PCy, PCz, 1)
    plane: [12]f64 = @splat(0),
    /// the source at the map's middle by the plane (the geometry's pattern centre now), and
    /// the pattern centre the fit began from (the vote's, in the automatic fit)
    pc: [3]f64 = .{ 0, 0, 0 },
    start_pc: [3]f64 = .{ 0, 0, 0 },
    /// the geometry's turn now (the plane's when the frame follows it, else as it was)
    rot: [3]f64 = .{ 0, 0, 0 },
    /// What the plane says (plane.Said) and each one's standard error, by leaving out the
    /// points of one block of the map at a time (a jackknife over `blocks` blocks): the
    /// sample's frame against the nominal one (swing, tilt, twist, radians), how far the beam
    /// goes for a row and for a column against the scan's steps and the pixel size, the angle
    /// between the scan's directions off a right angle, and the source at the map's middle.
    said_rot: [3]f64 = .{ 0, 0, 0 },
    said_rot_se: [3]f64 = .{ 0, 0, 0 },
    step: [2]f64 = .{ 0, 0 },
    step_se: [2]f64 = .{ 0, 0 },
    skew: f64 = 0,
    skew_se: f64 = 0,
    pc_se: [3]f64 = .{ 0, 0, 0 },
    /// root mean square of the used points about the plane (PCx, PCy, PCz)
    rmsd: [3]f64 = .{ 0, 0, 0 },
    /// mean NCC of the fit patterns, each at its own best, and where they started
    ncc: f64 = 0,
    start_ncc: f64 = 0,
    /// plane.Kind; 0: the frame nominal, 1: following the plane
    kind: u32 = 0,
    frame: u32 = 0,
    /// fit patterns measured, and those the plane was fitted to
    n_patterns: u32 = 0,
    n_used: u32 = 0,
    /// rounds of the patterns' own refinement
    iterations: u32 = 0,
    blocks: u32 = 0,
};

/// A pattern's normalized cross-correlation with its simulation, with its gradient and
/// Gauss–Newton matrix (the score falls by ½ δᵀ H δ about its top) over the device's six
/// directions: a shift of the pattern's rays' origin (pixels, in the sample's axes), then a
/// turn of its crystal (radians).
const Local = struct {
    ok: bool = false,
    ncc: f64 = 0,
    g: [6]f64 = @splat(0),
    H: [36]f64 = @splat(0),
};

/// From the NS sums of detector.wgsl `sums`.
fn localOf(sums: []const f32) Local {
    var out: Local = .{};
    const n: f64 = @max(sums[0], 1);
    const mean = sums[1] / n;
    const var_n = sums[2] - n * mean * mean;
    if (!(var_n > 1e-12)) return out;
    const inv = 1.0 / @sqrt(var_n);
    out.ok = true;
    out.ncc = sums[3] * inv;
    var sd: [6]f64 = undefined;
    var au: [6]f64 = undefined;
    for (0..6) |i| {
        sd[i] = sums[4 + i];
        au[i] = (sums[10 + i] - mean * sd[i]) * inv;
        out.g[i] = (sums[16 + i] - out.ncc * au[i]) * inv;
    }
    var o: usize = 22;
    for (0..6) |i| for (i..6) |j| {
        const v = (sums[o] - sd[i] * sd[j] / n - au[i] * au[j]) * inv * inv;
        out.H[i * 6 + j] = v;
        out.H[j * 6 + i] = v;
        o += 1;
    };
    return out;
}

/// Solve A x = b for a small dense system by elimination with partial pivoting.
fn solve(comptime N: usize, A0: [N * N]f64, b0: [N]f64) ?[N]f64 {
    var A = A0;
    var b = b0;
    for (0..N) |c| {
        var piv = c;
        for (c + 1..N) |r| if (@abs(A[r * N + c]) > @abs(A[piv * N + c])) {
            piv = r;
        };
        if (@abs(A[piv * N + c]) < 1e-300) return null;
        if (piv != c) {
            for (0..N) |k| std.mem.swap(f64, &A[c * N + k], &A[piv * N + k]);
            std.mem.swap(f64, &b[c], &b[piv]);
        }
        for (c + 1..N) |r| {
            const f = A[r * N + c] / A[c * N + c];
            if (f == 0) continue;
            for (c..N) |k| A[r * N + k] -= f * A[c * N + k];
            b[r] -= f * b[c];
        }
    }
    var x: [N]f64 = undefined;
    var r: usize = N;
    while (r > 0) {
        r -= 1;
        var acc = b[r];
        for (r + 1..N) |k| acc -= A[r * N + k] * x[k];
        x[r] = acc / A[r * N + r];
    }
    return x;
}

/// A pattern's system over `k` of its unknowns from the last (3: the turn alone; 6: the
/// source's shift too), damped by Marquardt's `lambda` (0: as it is).
fn damped(comptime k: usize, l: *const Local, lambda: f64) struct { A: [k * k]f64, g: [k]f64 } {
    const o = 6 - k;
    var A: [k * k]f64 = undefined;
    var g: [k]f64 = undefined;
    var tr: f64 = 0;
    for (0..k) |i| {
        g[i] = l.g[o + i];
        for (0..k) |j| A[i * k + j] = l.H[(o + i) * 6 + o + j];
        tr += A[i * k + i];
    }
    for (0..k) |i| A[i * k + i] = A[i * k + i] * (1 + lambda) + 1e-9 * tr + 1e-30;
    return .{ .A = A, .g = g };
}

/// The fit patterns, indexed at the geometry as it stands and on the device, each with its
/// orientation and its source's shift away from where the geometry puts it.
const Fit = struct {
    e: *Engine,
    arena: std.heap.ArenaAllocator,
    lv: engine.Level,
    sz: [2]f64,
    n: usize,
    /// the lattice's cells and each one's phase (the engine's once committed)
    cells: []u64,
    phase: []i32,
    /// which take part: those whose match indexing would accept
    use: []bool,
    n_used: usize,
    seed_ncc: []f32,
    /// each pattern's detector position as the geometry has it (pixels, the sample's axes)
    base: []rot.Vec3,
    /// the orientations (crystal to sample) and the sources' shifts, and a step's trial of them
    q: []rot.Quat,
    shift: []rot.Vec3,
    q_t: []rot.Quat,
    shift_t: []rot.Vec3,
    /// each pattern's system where it stands, and where `eval` was last
    keep: []Local,
    locals: []Local,
    /// each pattern's own damping
    lam: []f64,
    sums: []f32,
    poses: []detector.Pose,
    /// the fit's own copies of the phases' masters, resampled (settings.plane_master_px), and
    /// which of them are its own buffers to give back; none: the engine's
    masters: []detector.MasterGpu = &.{},
    own: []bool,
    /// the master's pixel the fit works with, detector pixels (0: the master as it is)
    master_px: f64 = 0,
    ncc: f64 = 0,
    committed: bool = false,
    /// the geometry when the fit began: what an error or a cancel puts back
    geo0: Geometry,

    /// The fit patterns: every `every`-th cell each way, and no more than `most`.
    fn open(e: *Engine, every: u32, most: u32) !*Fit {
        const gpa = e.gpa;
        const s = &e.set;
        e.cancel = false;
        // the fit is a refinement: it works where refinement does
        const lv = e.levelOf(.refine);
        const sz = try e.sizeOf(lv);
        const n_ph = e.phases.items.len;
        if (n_ph == 0) return error.NoPhases;
        // the stride widened until the patterns are no more than `most` and fit in one buffer here
        const room = @min(e.g.lim.max_storage_binding, e.g.lim.max_buffer) / 2;
        const bytes: u64 = @as(u64, @intFromFloat(@ceil(sz[0] * sz[1]))) * 4;
        var stride: u64 = every;
        var cells = try pcfit.fitCellsEvery(e, stride);
        while (cells.len > 1 and ((most > 0 and cells.len > most) or cells.len * bytes > room)) {
            gpa.free(cells);
            stride += 1;
            cells = try pcfit.fitCellsEvery(e, stride);
        }
        errdefer gpa.free(cells);
        if (stride != every) e.events.log("fit patterns: a stride of {d} cells each way, {d} patterns (asked every {d}; no more than {d}, nor than fit in one buffer here)", .{ stride, cells.len, every, most });
        const n = cells.len;
        const phase = try gpa.alloc(i32, n);
        errdefer gpa.free(phase);
        const self = try gpa.create(Fit);
        errdefer gpa.destroy(self);
        self.arena = std.heap.ArenaAllocator.init(gpa);
        errdefer self.arena.deinit();
        const a = self.arena.allocator();
        // seeds: the fit patterns indexed at the current geometry (this also leaves them,
        // filtered, on the device)
        const recs = try a.alloc(engine.Record, n_ph * n);
        try e.indexCells(cells, .{ .bandwidth = s.fit_bandwidth, .dict_grid = s.dict_grid }, recs);
        const use = try a.alloc(bool, n);
        const seed_ncc = try a.alloc(f32, n);
        const base = try a.alloc(rot.Vec3, n);
        const q = try a.alloc(rot.Quat, n);
        const shift = try a.alloc(rot.Vec3, n);
        const q_t = try a.alloc(rot.Quat, n);
        const shift_t = try a.alloc(rot.Vec3, n);
        const keep = try a.alloc(Local, n);
        const locals = try a.alloc(Local, n);
        const lam = try a.alloc(f64, n);
        const sums = try a.alloc(f32, n_ph * n * NS);
        const poses = try a.alloc(detector.Pose, n);
        const own = try a.alloc(bool, n_ph);
        // (the arena's state as the allocations above left it)
        const arena = self.arena;
        self.* = .{
            .e = e,
            .arena = arena,
            .lv = lv,
            .sz = sz,
            .n = n,
            .cells = cells,
            .phase = phase,
            .use = use,
            .n_used = 0,
            .seed_ncc = seed_ncc,
            .base = base,
            .q = q,
            .shift = shift,
            .q_t = q_t,
            .shift_t = shift_t,
            .keep = keep,
            .locals = locals,
            .lam = lam,
            .sums = sums,
            .poses = poses,
            .own = own,
            .geo0 = e.geo,
        };
        @memset(self.own, false);
        @memset(self.shift, .{ 0, 0, 0 });
        // which patterns take part: those whose match indexing would accept
        const still: f32 = @floatCast(std.math.degreesToRadians(s.accept_step_deg));
        for (0..n) |b| {
            const p = e.choosePhase(recs, n, b);
            phase[b] = @intCast(p);
            const r = recs[p * n + b];
            self.seed_ncc[b] = r.ncc;
            self.use[b] = (if (s.accept_nmi > 0) r.nmi >= s.accept_nmi else r.z >= s.accept_z) and (!s.refine or r.step <= still);
            if (self.use[b]) self.n_used += 1;
            self.q[b] = rot.qstd(.{ r.q[0], r.q[1], r.q[2], r.q[3] });
            self.base[b] = try e.placeOf(&e.geo, cells[b], lv);
        }
        if (self.n_used < 3) {
            // nothing to choose among: a start this far off, or a threshold this high, leaves the fit all of them
            e.events.log("plane fit: {d} of the {d} fit patterns have a match that would be accepted here; fitting to all of them", .{ self.n_used, n });
            @memset(self.use, true);
            self.n_used = n;
        } else if (self.n_used < n) {
            e.events.log("plane fit: {d} of the {d} fit patterns left out, their matches not accepted", .{ n - self.n_used, n });
        }
        if (s.plane_master_auto) {
            // The master's pixel that suits these patterns: of a few widths, the one under
            // which the fit patterns correlate best, each with its orientations settled there.
            const widths = [_]f64{ 0, 0.5, 1, 1.5, 2 };
            const seeds = try a.dupe(rot.Quat, self.q);
            var best: f64 = 0;
            var top: f64 = -std.math.inf(f64);
            var said: [160]u8 = undefined;
            var w: std.Io.Writer = .fixed(&said);
            for (widths) |width| {
                try self.resample(width);
                @memcpy(self.q, seeds);
                _ = try self.settle(3, 0, 60);
                w.print("{s}{d}: {d:.4}", .{ if (w.end > 0) ", " else "", width, self.ncc }) catch {};
                if (self.ncc > top) {
                    top = self.ncc;
                    best = width;
                }
            }
            @memcpy(self.q, seeds);
            try self.resample(best);
            self.master_px = best;
            e.events.log("plane fit: master pixel {d} detector pixels (0: as it is), the best of [{s}] by the fit patterns' mean NCC", .{ best, w.buffered() });
        } else {
            try self.resample(s.plane_master_px);
            self.master_px = s.plane_master_px;
        }
        self.ncc = try self.eval(self.q, self.shift);
        @memcpy(self.keep, self.locals);
        return self;
    }

    fn masterOf(self: *const Fit, p: *const engine.Phase, pi: usize) detector.MasterGpu {
        return if (self.masters.len > 0) self.masters[pi] else p.det[@intFromEnum(self.lv)];
    }

    /// The fit's own masters (settings.plane_master_px): each phase's filtered sheets resampled
    /// so that a master pixel is that many detector pixels wide at the pattern centre. A new
    /// pixel is the tent-weighted average of the old ones within one new pixel of it, along
    /// each way of the sheet: the sheets are equal-area, so that is an average over the sphere.
    fn resample(self: *Fit, want: f64) !void {
        const e = self.e;
        const g = &e.g;
        const a = self.arena.allocator();
        for (self.masters, 0..) |*m, pi| {
            if (!self.own[pi]) continue;
            g.release(&m.buf);
            g.release(&m.grad);
            self.own[pi] = false;
        }
        self.masters = &.{};
        if (!(want > 0)) return;
        const lv = self.lv;
        const pixel = e.pixelAngle(lv);
        if (!(pixel > 0)) return;
        const n_ph = e.phases.items.len;
        const masters = try a.alloc(detector.MasterGpu, n_ph);
        for (masters) |*m| m.* = .{};
        for (e.phases.items, 0..) |*p, pi| {
            const side: usize = p.det[@intFromEnum(lv)].side;
            const work = p.work[@intFromEnum(lv)];
            // a hemisphere's sheet is sqrt(2π) across
            const across = @sqrt(2.0 * std.math.pi) / (want * pixel);
            const new: usize = @intFromFloat(std.math.clamp(@round(across) + 1, 16, @as(f64, @floatFromInt(side))));
            if (new >= side or work.len != 2 * side * side) {
                masters[pi] = p.det[@intFromEnum(lv)];
                continue;
            }
            const ratio = @as(f64, @floatFromInt(side - 1)) / @as(f64, @floatFromInt(new - 1));
            // the weights of one new pixel along one way: old pixels within `ratio` of it
            const reach: usize = @intFromFloat(@ceil(ratio));
            const span = 2 * reach + 1;
            const first = try a.alloc(usize, new);
            const wts = try a.alloc(f32, new * span);
            @memset(wts, 0);
            for (0..new) |i| {
                const x = @as(f64, @floatFromInt(i)) * ratio;
                const lo: usize = @intFromFloat(@max(@floor(x) - @as(f64, @floatFromInt(reach)) + 1, 0));
                first[i] = lo;
                var sum: f64 = 0;
                for (0..span) |k| {
                    if (lo + k >= side) break;
                    const w = @max(1.0 - @abs(@as(f64, @floatFromInt(lo + k)) - x) / ratio, 0);
                    wts[i * span + k] = @floatCast(w);
                    sum += w;
                }
                for (0..span) |k| wts[i * span + k] = @floatCast(wts[i * span + k] / sum);
            }
            const out = try a.alloc(f32, 2 * new * new);
            const rows = try a.alloc(f32, new * side);
            for (0..2) |sheet| {
                const src = work[sheet * side * side ..][0 .. side * side];
                // along the columns, then along the rows
                for (0..side) |r| for (0..new) |i| {
                    var acc: f32 = 0;
                    for (0..span) |k| {
                        if (first[i] + k >= side) break;
                        acc += wts[i * span + k] * src[r * side + first[i] + k];
                    }
                    rows[r * new + i] = acc;
                };
                for (0..new) |j| for (0..new) |i| {
                    var acc: f32 = 0;
                    for (0..span) |k| {
                        if (first[j] + k >= side) break;
                        acc += wts[j * span + k] * rows[(first[j] + k) * new + i];
                    }
                    out[sheet * new * new + j * new + i] = acc;
                };
            }
            masters[pi] = .{ .side = @intCast(new) };
            masters[pi].buf = g.storage(@as(u64, out.len) * 4);
            g.writeSlice(masters[pi].buf, 0, f32, out);
            e.prep.gradient(masters[pi].buf, @intCast(new), &masters[pi].grad);
            self.own[pi] = true;
        }
        self.masters = masters;
    }

    /// Mean NCC with orientations `q` and shifts `shift`, leaving every pattern's system in
    /// `locals`.
    fn eval(self: *Fit, q: []const rot.Quat, shift: []const rot.Vec3) !f64 {
        const e = self.e;
        const g = &e.g;
        const n = self.n;
        const fr = try e.frame(self.lv);
        for (0..n) |b| {
            const t = rot.add(self.base[b], shift[b]);
            self.poses[b] = .{
                .t = .{ @floatCast(t[0]), @floatCast(t[1]), @floatCast(t[2]) },
                .pat = @floatFromInt(b),
                .q = .{ @floatCast(q[b][0]), @floatCast(q[b][1]), @floatCast(q[b][2]), @floatCast(q[b][3]) },
            };
        }
        e.det.setPoses(self.poses);
        e.det.interp = switch (e.set.plane_interp) {
            .bilinear => 0,
            .bicubic => 4,
            .keys6 => 8,
        };
        defer e.det.interp = 0;
        for (e.phases.items, 0..) |*p, pi| {
            e.det.sumsOf(self.masterOf(p, pi), fr, e.pats[@intFromEnum(self.lv)], @intCast(n), false);
            g.read(e.det.sums, 0, std.mem.sliceAsBytes(self.sums[pi * n * NS ..][0 .. n * NS]));
        }
        try g.wait();
        const n_ph = e.phases.items.len;
        var mean: f64 = 0;
        for (0..n) |b| {
            self.locals[b] = .{};
            const p: usize = @intCast(@max(self.phase[b], 0));
            if (p >= n_ph or !self.use[b]) continue;
            self.locals[b] = localOf(self.sums[(p * n + b) * NS ..][0..NS]);
            mean += self.locals[b].ncc;
        }
        return mean / @as(f64, @floatFromInt(self.n_used));
    }

    /// Every fit pattern taken to its own best by itself, by Levenberg–Marquardt on its own
    /// score: over its orientation alone (`k` 3), or with the place of its own source (`k` 6:
    /// what Winkelmann et al. and kikuchipy's refine_orientation_projection_center do a
    /// pattern at a time). All the patterns take a step together, a pattern's step kept where
    /// its own score rose, until a round gains the mean less than 1e-7 (about what single
    /// precision on the device leaves of it) or after `most` rounds. A source may move
    /// `trust` pixels each way from where it started. Returns the rounds taken; leaves the
    /// mean score in `ncc` and the patterns' systems in `keep`.
    fn settle(self: *Fit, comptime k: usize, trust: f64, most: u32) !u32 {
        const n = self.n;
        const count: f64 = @floatFromInt(self.n_used);
        self.ncc = try self.eval(self.q, self.shift);
        @memcpy(self.keep, self.locals);
        @memset(self.lam, 1e-3);
        var it: u32 = 0;
        while (it < most) : (it += 1) {
            if (self.e.cancel) return error.Cancelled;
            for (0..n) |b| {
                self.q_t[b] = self.q[b];
                self.shift_t[b] = self.shift[b];
                if (!self.use[b]) continue;
                const sys = damped(k, &self.keep[b], self.lam[b]);
                const d = solve(k, sys.A, sys.g) orelse continue;
                if (k == 6) {
                    var sh = rot.add(self.shift[b], .{ d[0], d[1], d[2] });
                    for (&sh) |*v| v.* = std.math.clamp(v.*, -trust, trust);
                    self.shift_t[b] = sh;
                }
                self.q_t[b] = rot.qstd(rot.qmul(rot.so3Exp(.{ d[k - 3], d[k - 2], d[k - 1] }), self.q[b]));
            }
            _ = try self.eval(self.q_t, self.shift_t);
            var gain: f64 = 0;
            for (0..n) |b| {
                if (!self.use[b]) continue;
                const now = self.locals[b].ncc;
                if (now > self.keep[b].ncc) {
                    gain += now - self.keep[b].ncc;
                    self.q[b] = self.q_t[b];
                    self.shift[b] = self.shift_t[b];
                    self.keep[b] = self.locals[b];
                    self.lam[b] = @max(self.lam[b] / 3, 1e-4);
                } else self.lam[b] = @min(self.lam[b] * 5, 1e8);
            }
            if (k == 6) self.e.events.emit(.{ .kind = "progress", .stage = "plane", .done = it + 1, .total = most });
            if (gain / count < 1e-7) break;
        }
        var mean: f64 = 0;
        for (0..n) |b| {
            if (self.use[b]) mean += self.keep[b].ncc;
        }
        self.ncc = mean / count;
        return it;
    }

    fn close(self: *Fit) void {
        const gpa = self.e.gpa;
        for (self.masters, 0..) |*m, pi| {
            if (!self.own[pi]) continue;
            self.e.g.release(&m.buf);
            self.e.g.release(&m.grad);
        }
        if (!self.committed) {
            self.e.geo = self.geo0;
            gpa.free(self.cells);
            gpa.free(self.phase);
        }
        self.arena.deinit();
        gpa.destroy(self);
    }
};

/// The plane of source points (plane.zig): every fit pattern's own source point and
/// orientation by that pattern's own fit (Fit.settle), then the transformation of the
/// scan's indices that passes through them (settings.plane_kind), by weighted least squares
/// with the points far from it left out (plane_weights, plane_outlier). The geometry takes
/// the plane: the beam's position for every map cell is then the plane's, and the scan's steps
/// and the detector's pixel size say nothing of it.
pub fn plane(e: *Engine) !PlaneResult {
    const s = &e.set;
    const before = e.geo;
    errdefer e.geo = before;
    const info = try e.info();
    const fit = try Fit.open(e, s.plane_stride, s.plane_patterns_max);
    defer fit.close();
    const gpa = e.gpa;
    const a = fit.arena.allocator();
    const n = fit.n;
    const sz = fit.sz;
    const aspect = sz[1] / sz[0];
    var res: PlaneResult = .{ .start_pc = before.pc, .start_ncc = fit.ncc, .n_patterns = @intCast(fit.n_used) };
    const trust = s.plane_trust * sz[0];
    res.iterations = try fit.settle(6, trust, s.plane_iters);
    res.ncc = fit.ncc;

    // the measured points, and how well each is told
    const pts = try a.alloc(plane_mod.Point, n);
    const R = before.rotation();
    // a source's place, t, against (PCx · aspect, PCy, PCz): t = H R D y + const (geometry.translationFor)
    const M: rot.Mat3 = .{ -R[1], -R[0], R[2], -R[4], -R[3], R[5], -R[7], -R[6], R[8] };
    for (0..n) |b| {
        const pc = Geometry.pcOf(R, rot.add(fit.base[b], fit.shift[b]), sz[0], sz[1]);
        var at_edge = false;
        for (fit.shift[b]) |v| at_edge = at_edge or @abs(v) >= trust * (1 - 1e-9);
        pts[b] = .{
            .col = @floatFromInt(fit.cells[b] % info.cols),
            .row = @floatFromInt(fit.cells[b] / info.cols),
            .y = .{ pc[0] * aspect, pc[1], pc[2] },
            .used = fit.use[b] and !at_edge,
        };
        if (s.plane_weights != .information or !fit.use[b]) continue;
        // the curvature of the pattern's score over its source's place, its orientation
        // following: S = H_tt − H_tq H_qq⁻¹ H_qt
        const l = &fit.keep[b];
        const B = damped(3, l, 0).A;
        var S: [9]f64 = undefined;
        var ok = true;
        for (0..3) |c| {
            const col = solve(3, B, .{ l.H[c * 6 + 3], l.H[c * 6 + 4], l.H[c * 6 + 5] }) orelse {
                ok = false;
                break;
            };
            for (0..3) |i| {
                var acc: f64 = l.H[i * 6 + c];
                for (0..3) |k| acc -= l.H[i * 6 + 3 + k] * col[k];
                S[i * 3 + c] = acc;
            }
        }
        if (!ok) {
            pts[b].used = false;
            continue;
        }
        var W = rot.matMul(rot.transpose(M), rot.matMul(S, M));
        const tr = W[0] + W[4] + W[8];
        if (!(tr > 0)) {
            pts[b].used = false;
            continue;
        }
        for (0..3) |i| {
            for (i + 1..3) |j| {
                const v = 0.5 * (W[i * 3 + j] + W[j * 3 + i]);
                W[i * 3 + j] = v;
                W[j * 3 + i] = v;
            }
            W[i * 3 + i] += 1e-9 * tr;
        }
        pts[b].w = W;
    }

    const kind: plane_mod.Kind = if (s.plane_kind == .projective) .projective else .affine;
    const fr = plane_mod.Frame.of(info.rows, info.cols, aspect);
    var P = plane_mod.fit(kind, pts, fr) orelse return error.TooFewFitPatterns;
    // the points far from the plane go, and the plane is fitted again, until none more goes
    const dist = try a.alloc(f64, n);
    const sorted = try a.alloc(f64, n);
    if (s.plane_outlier > 0) {
        var round: u32 = 0;
        while (round < 8) : (round += 1) {
            var m: usize = 0;
            for (pts, 0..) |pt, b| {
                dist[b] = 0;
                if (!pt.used) continue;
                const pc = plane_mod.at(P, pt.row, pt.col);
                const r: [3]f64 = .{ pt.y[0] - pc[0] * aspect, pt.y[1] - pc[1], pt.y[2] - pc[2] };
                var q: f64 = 0;
                for (0..3) |i| for (0..3) |j| {
                    q += r[i] * pt.w[i * 3 + j] * r[j];
                };
                dist[b] = @sqrt(@max(q, 0));
                sorted[m] = dist[b];
                m += 1;
            }
            if (m < 8) break;
            std.mem.sort(f64, sorted[0..m], {}, std.sort.asc(f64));
            // (the median of the length of a three-component Gaussian error is 1.538 of its standard deviation)
            const sigma = sorted[m / 2] / 1.538;
            var dropped: usize = 0;
            for (pts, 0..) |*pt, b| {
                if (pt.used and dist[b] > s.plane_outlier * sigma) {
                    pt.used = false;
                    dropped += 1;
                }
            }
            if (dropped == 0) break;
            P = plane_mod.fit(kind, pts, fr) orelse return error.TooFewFitPatterns;
        }
    }
    const h_px: f64 = @floatFromInt(info.pat_h);
    const said = plane_mod.said(P, &before, info.rows, info.cols, aspect, h_px, s.scan_x_sign, s.scan_y_sign);

    // each one's standard error: the plane fitted again without the points of one block of the map, block by block
    {
        const BL = 6;
        var val: [BL * BL][9]f64 = undefined;
        var mean: [9]f64 = @splat(0);
        var got: usize = 0;
        const was = try a.alloc(bool, n);
        for (pts, 0..) |pt, b| was[b] = pt.used;
        for (0..BL * BL) |g| {
            var any = false;
            for (pts, 0..) |*pt, b| {
                const at: usize = @intCast((fit.cells[b] / info.cols * BL / info.rows) * BL + fit.cells[b] % info.cols * BL / info.cols);
                pt.used = was[b] and at != g;
                any = any or (was[b] and at == g);
            }
            if (!any) continue;
            const Pg = plane_mod.fit(kind, pts, fr) orelse continue;
            const w = plane_mod.said(Pg, &before, info.rows, info.cols, aspect, h_px, s.scan_x_sign, s.scan_y_sign);
            val[got] = .{ w.rot[0], w.rot[1], w.rot[2], w.step[0], w.step[1], w.skew, w.pc[0], w.pc[1], w.pc[2] };
            for (0..9) |i| mean[i] += val[got][i];
            got += 1;
        }
        for (pts, 0..) |*pt, b| pt.used = was[b];
        res.blocks = @intCast(got);
        if (got >= 2) {
            const G: f64 = @floatFromInt(got);
            var se: [9]f64 = @splat(0);
            for (0..9) |i| mean[i] /= G;
            for (val[0..got]) |v| {
                for (0..9) |i| se[i] += (v[i] - mean[i]) * (v[i] - mean[i]);
            }
            for (&se) |*v| v.* = @sqrt(v.* * (G - 1) / G);
            res.said_rot_se = .{ se[0], se[1], se[2] };
            res.step_se = .{ se[3], se[4] };
            res.skew_se = se[5];
            res.pc_se = .{ se[6], se[7], se[8] };
        }
    }

    // the geometry takes the plane
    res.kind = @intFromEnum(kind);
    res.frame = @intFromBool(s.plane_frame == .follow);
    res.plane = P;
    res.pc = said.pc;
    res.said_rot = said.rot;
    res.step = said.step;
    res.skew = said.skew;
    e.geo = before;
    e.geo.plane_kind = @intFromEnum(kind);
    e.geo.plane = P;
    e.geo.pc = said.pc;
    if (s.plane_frame == .follow) e.geo.rot = said.rot;
    res.rot = e.geo.rot;

    e.plane_points.clearRetainingCapacity();
    var sq: [3]f64 = .{ 0, 0, 0 };
    var used: f64 = 0;
    for (pts, 0..) |pt, b| {
        const pc = plane_mod.at(P, pt.row, pt.col);
        const own: [3]f64 = .{ pt.y[0] / aspect, pt.y[1], pt.y[2] };
        try e.plane_points.append(gpa, .{ .cell = fit.cells[b], .pc = own, .fit = pc, .ncc = if (fit.use[b]) fit.keep[b].ncc else fit.seed_ncc[b], .used = @intFromBool(pt.used) });
        if (!pt.used) continue;
        used += 1;
        for (0..3) |i| sq[i] += (own[i] - pc[i]) * (own[i] - pc[i]);
    }
    res.n_used = @intFromFloat(used);
    for (0..3) |i| res.rmsd[i] = @sqrt(sq[i] / @max(used, 1));

    // The fit patterns, with their orientations in the sample's frame as the geometry now has
    // it. A turn of the geometry from E to E' turns that frame: q' = q h(E) h(E')⁻¹, with h
    // the detector's turn in the sample's axes, Exp(R0 log E).
    const quats = try gpa.alloc(f32, 4 * n);
    errdefer gpa.free(quats);
    const nccs = try gpa.alloc(f32, n);
    const R0 = before.nominal();
    const turn = rot.qmul(rot.so3Exp(rot.matVec(R0, before.rot)), rot.qconj(rot.so3Exp(rot.matVec(R0, e.geo.rot))));
    for (0..n) |k| {
        const q = rot.qstd(rot.qmul(fit.q[k], turn));
        for (0..4) |c| quats[4 * k + c] = @floatCast(q[c]);
        nccs[k] = if (fit.use[k]) @floatCast(fit.keep[k].ncc) else fit.seed_ncc[k];
    }
    gpa.free(e.fit_cells);
    gpa.free(e.fit_quats);
    gpa.free(e.fit_phase);
    gpa.free(e.fit_ncc);
    e.fit_cells = fit.cells;
    e.fit_phase = fit.phase;
    e.fit_quats = quats;
    e.fit_ncc = nccs;
    fit.committed = true;

    const deg = 180.0 / std.math.pi;
    e.events.log("plane of source points ({s}): {d} of {d} fit patterns used; about it they scatter by {d:.5} {d:.5} {d:.5} (PCx, PCy, PCz); mean NCC {d:.4} with each pattern at its own best (was {d:.4})", .{ @tagName(kind), res.n_used, res.n_patterns, res.rmsd[0], res.rmsd[1], res.rmsd[2], res.ncc, res.start_ncc });
    e.events.log("plane: source at the map's middle {d:.5} {d:.5} {d:.5}; the sample's frame against the nominal one: swing {d:.2}° ± {d:.2}°, tilt {d:.2}° ± {d:.2}°, the scan turned in its plane {d:.2}° ± {d:.2}°; the beam's travel for a row {d:.3} ± {d:.3} and for a column {d:.3} ± {d:.3} of what the steps and the pixel size say; the scan's directions {d:.2}° ± {d:.2}° off square", .{ res.pc[0], res.pc[1], res.pc[2], said.rot[0] * deg, res.said_rot_se[0] * deg, said.rot[1] * deg, res.said_rot_se[1] * deg, said.rot[2] * deg, res.said_rot_se[2] * deg, said.step[0], res.step_se[0], said.step[1], res.step_se[1], said.skew * deg, res.skew_se * deg });
    return res;
}

/// The automatic fit: the vote for the pattern centre (pcfit), then the plane of source
/// points from there. It starts from the geometry as it was given (engine.geo_given: by the
/// scan, or by hand), not from what an earlier fit left: the same scan and settings give the
/// same fit however often.
pub fn auto(e: *Engine) !PlaneResult {
    const before = e.geo;
    errdefer e.geo = before;
    e.geo = e.geo_given;
    _ = try pcfit.run(e);
    if (e.cancel) return error.Cancelled;
    return plane(e);
}

test "small dense solve" {
    const x = solve(3, .{ 2, 1, 0, 1, 3, 1, 0, 1, 4 }, .{ 3, 5, 5 }).?;
    for (x) |v| try std.testing.expectApproxEqAbs(@as(f64, 1), v, 1e-12);
    try std.testing.expect(solve(2, .{ 1, 2, 2, 4 }, .{ 1, 1 }) == null);
}
