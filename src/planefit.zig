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
const sym = @import("sym.zig");
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
    /// plane.Model the geometry took: the full plane by itself (Fit plane); in the automatic
    /// fit, the simplest model the map supports
    model: u32 = 0,
    /// 1: the models were compared (the automatic fit)
    compared: u32 = 0,
    /// each model's error in predicting a block of the map from the others (mean rᵀ W r a point,
    /// over the blocks: 0 where it was not tried), and its standard error against the best
    /// model's, block by block
    cv: [4]f64 = @splat(0),
    cv_se: [4]f64 = @splat(0),
    /// settings.PlaneMethod: 0 joint, 1 through the patterns' own points
    method: u32 = 0,
    _pad: u32 = 0,
};

/// The models compared, block by block: each fitted without one block of the map and asked
/// where that block's points stand, as mean rᵀ W r. The simplest model whose error is not
/// above the best one's by more than the standard error of their difference (the "one
/// standard error" rule, over paired blocks) is the one the map supports.
fn choose(a: std.mem.Allocator, models: []const plane_mod.Model, pts: []plane_mod.Point, cells: []const u64, rows: u32, cols: u32, fr: plane_mod.Frame, aspect: f64, res: *PlaneResult) !plane_mod.Model {
    const BL = 6;
    const K = 4;
    var err: [BL * BL][K]f64 = undefined;
    var got: usize = 0;
    const was = try a.alloc(bool, pts.len);
    for (pts, 0..) |pt, b| was[b] = pt.used;
    for (0..BL * BL) |g| {
        var in_block: usize = 0;
        for (pts, 0..) |*pt, b| {
            const at: usize = @intCast((cells[b] / cols * BL / rows) * BL + cells[b] % cols * BL / cols);
            pt.used = was[b] and at != g;
            in_block += @intFromBool(was[b] and at == g);
        }
        if (in_block == 0) continue;
        var row: [K]f64 = @splat(0);
        var ok = true;
        for (models) |m| {
            const P = plane_mod.fitModel(m, pts, fr) orelse {
                ok = false;
                break;
            };
            var sum: f64 = 0;
            for (pts, 0..) |pt, b| {
                const at: usize = @intCast((cells[b] / cols * BL / rows) * BL + cells[b] % cols * BL / cols);
                if (was[b] and at == g) sum += plane_mod.misfit(P, pt, aspect);
            }
            row[@intFromEnum(m)] = sum / @as(f64, @floatFromInt(in_block));
        }
        if (!ok) continue;
        err[got] = row;
        got += 1;
    }
    for (pts, 0..) |*pt, b| pt.used = was[b];
    res.compared = 1;
    if (got < 3) return models[models.len - 1];
    const G: f64 = @floatFromInt(got);
    var best = models[0];
    for (models) |m| {
        const i = @intFromEnum(m);
        var s: f64 = 0;
        for (err[0..got]) |e| s += e[i];
        res.cv[i] = s / G;
        if (res.cv[i] < res.cv[@intFromEnum(best)]) best = m;
    }
    const bi = @intFromEnum(best);
    for (models) |m| {
        const i = @intFromEnum(m);
        var mean: f64 = 0;
        for (err[0..got]) |e| mean += e[i] - e[bi];
        mean /= G;
        var v: f64 = 0;
        for (err[0..got]) |e| v += (e[i] - e[bi] - mean) * (e[i] - e[bi] - mean);
        res.cv_se[i] = @sqrt(v / (G - 1) / G);
    }
    for (models) |m| {
        const i = @intFromEnum(m);
        if (res.cv[i] - res.cv[bi] <= res.cv_se[i]) return m;
    }
    return best;
}

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

// ── the joint fit ──

/// What the joint fit works in: the fit patterns, the model's frame, and how a source's place
/// in the fit's coordinates y = (PCx · aspect, PCy, PCz) becomes the detector's position the
/// device takes (pixels, the sample's axes): t = R (½ − h y₁, ½ − h y₀, h y₂), so ∂t/∂y = h M
/// (geometry.translationFor).
const Joint = struct {
    fit: *Fit,
    fr: plane_mod.Frame,
    R: rot.Mat3,
    M: rot.Mat3,
    h: f64,
    cols: u32,

    fn cellOf(self: *const Joint, b: usize) [2]f64 {
        const c = self.fit.cells[b];
        return .{ @floatFromInt(c / self.cols), @floatFromInt(c % self.cols) };
    }

    /// Each pattern's source where the model puts it, as a shift from where the fit began.
    fn sourcesOf(self: *const Joint, theta: [plane_mod.N]f64, out: []rot.Vec3) void {
        for (0..self.fit.n) |b| {
            const rc = self.cellOf(b);
            const y = plane_mod.modelAt(theta, self.fr, rc[0], rc[1]).y;
            const t = rot.matVec(self.R, .{ 0.5 - y[1] * self.h, 0.5 - y[0] * self.h, y[2] * self.h });
            out[b] = rot.sub(t, self.fit.base[b]);
        }
    }

    /// d(t)/d(y) for a column of y's Jacobian: h M J.
    fn jt(self: *const Joint, J: [3 * plane_mod.N]f64) [3 * plane_mod.N]f64 {
        const N = plane_mod.N;
        var out: [3 * N]f64 = undefined;
        for (0..3) |k| for (0..N) |i| {
            var acc: f64 = 0;
            for (0..3) |l| acc += self.M[k * 3 + l] * J[l * N + i];
            out[k * N + i] = self.h * acc;
        };
        return out;
    }
};

/// A pattern's information about its source with its turn profiled out (pixels, the sample's
/// axes): the curvature S = H_tt − H_tq H_qq⁻¹ H_qt and the pull g̃ = g_t − H_tq H_qq⁻¹ g_q, what
/// its score says of its source once its orientation follows. Null where its turn is not told.
fn profiled(l: *const Local) ?struct { S: [9]f64, g: [3]f64 } {
    const B = damped(3, l, 0).A;
    var S: [9]f64 = undefined;
    for (0..3) |c| {
        const col = solve(3, B, .{ l.H[c * 6 + 3], l.H[c * 6 + 4], l.H[c * 6 + 5] }) orelse return null;
        for (0..3) |i| {
            var acc: f64 = l.H[i * 6 + c];
            for (0..3) |k| acc -= l.H[i * 6 + 3 + k] * col[k];
            S[i * 3 + c] = acc;
        }
    }
    const xg = solve(3, B, .{ l.g[3], l.g[4], l.g[5] }) orelse return null;
    var g: [3]f64 = undefined;
    for (0..3) |i| {
        var acc = l.g[i];
        for (0..3) |k| acc -= l.H[i * 6 + 3 + k] * xg[k];
        g[i] = acc;
    }
    for (0..3) |i| for (i + 1..3) |j| {
        const v = 0.5 * (S[i * 3 + j] + S[j * 3 + i]);
        S[i * 3 + j] = v;
        S[j * 3 + i] = v;
    };
    return .{ .S = S, .g = g };
}

/// The model's numbers fitted to all the used fit patterns at once: Levenberg–Marquardt on
/// their mean NCC, the unknowns the model's (plane.Model.free), each pattern's orientation
/// profiled out of the step and then taken along with it (δq = H_qq⁻¹ (g_q − H_qt δt)). No
/// pattern has a source of its own: each is where the model puts it. Returns the rounds taken;
/// leaves the patterns' orientations, their systems (`keep`) and the mean NCC in the fit.
fn jointFit(j: *const Joint, which: plane_mod.Model, theta: *[plane_mod.N]f64, most: u32) !u32 {
    const N = plane_mod.N;
    const fit = j.fit;
    const n = fit.n;
    const F = which.free();
    j.sourcesOf(theta.*, fit.shift);
    fit.ncc = try fit.eval(fit.q, fit.shift);
    @memcpy(fit.keep, fit.locals);
    var lam: f64 = 1e-3;
    var it: u32 = 0;
    while (it < most) : (it += 1) {
        if (fit.e.cancel) return error.Cancelled;
        var A: [N * N]f64 = @splat(0);
        var v: [N]f64 = @splat(0);
        for (0..n) |b| {
            if (!fit.use[b] or !fit.keep[b].ok) continue;
            const pr = profiled(&fit.keep[b]) orelse continue;
            const rc = j.cellOf(b);
            const Jt = j.jt(plane_mod.modelAt(theta.*, j.fr, rc[0], rc[1]).J);
            for (F) |fi| {
                var sj: [3]f64 = .{ 0, 0, 0 };
                for (0..3) |k| for (0..3) |l| {
                    sj[k] += pr.S[k * 3 + l] * Jt[l * N + fi];
                };
                for (0..3) |k| v[fi] += Jt[k * N + fi] * pr.g[k];
                for (F) |fj| {
                    for (0..3) |k| A[fi * N + fj] += Jt[k * N + fj] * sj[k];
                }
            }
        }
        // the damped system over the model's own unknowns
        var Am: [N * N]f64 = @splat(0);
        var vm: [N]f64 = @splat(0);
        var tr: f64 = 0;
        for (F, 0..) |fi, a| tr += A[fi * N + fi] / @as(f64, @floatFromInt(F.len)) + 0 * @as(f64, @floatFromInt(a));
        for (F, 0..) |fi, a| {
            vm[a] = v[fi];
            for (F, 0..) |fj, c| Am[a * N + c] = A[fi * N + fj];
            Am[a * N + a] = Am[a * N + a] * (1 + lam) + 1e-12 * tr + 1e-300;
        }
        const d = plane_mod.solve(F.len, Am, vm) orelse break;
        var trial = theta.*;
        for (F, 0..) |fi, a| trial[fi] += d[a];
        j.sourcesOf(trial, fit.shift_t);
        for (0..n) |b| {
            fit.q_t[b] = fit.q[b];
            if (!fit.use[b] or !fit.keep[b].ok) continue;
            const l = &fit.keep[b];
            const dt = rot.sub(fit.shift_t[b], fit.shift[b]);
            var r: [3]f64 = undefined;
            for (0..3) |i| {
                var acc = l.g[3 + i];
                for (0..3) |k| acc -= l.H[(3 + i) * 6 + k] * dt[k];
                r[i] = acc;
            }
            const sys = damped(3, l, lam);
            const dq = solve(3, sys.A, r) orelse continue;
            fit.q_t[b] = rot.qstd(rot.qmul(rot.so3Exp(.{ dq[0], dq[1], dq[2] }), fit.q[b]));
        }
        const now = try fit.eval(fit.q_t, fit.shift_t);
        fit.e.events.emit(.{ .kind = "progress", .stage = "plane", .done = it + 1, .total = most });
        if (now > fit.ncc) {
            const gain = now - fit.ncc;
            theta.* = trial;
            @memcpy(fit.q, fit.q_t);
            @memcpy(fit.shift, fit.shift_t);
            @memcpy(fit.keep, fit.locals);
            fit.ncc = now;
            lam = @max(lam / 3, 1e-6);
            if (gain < 1e-8) {
                it += 1;
                break;
            }
        } else {
            lam *= 5;
            if (lam > 1e8) break;
        }
    }
    // (the systems where the patterns now stand, for what is read of them next)
    _ = try fit.eval(fit.q, fit.shift);
    @memcpy(fit.keep, fit.locals);
    return it;
}

/// What each used pattern says of its own source, one Newton step from where the model puts
/// it, with its information as the weight: ŷ = y + W⁻¹ p, W = h² Mᵀ S M, p = h Mᵀ g̃ (the
/// inverse a little regularized, for directions a pattern does not tell). With these points
/// and weights the model's weighted least-squares fit is the joint fit's own Gauss–Newton
/// step, so the comparison of models and the standard errors read them as they read measured
/// points; they are also the points the plane's view draws.
fn pseudoPoints(j: *const Joint, theta: [plane_mod.N]f64, pts: []plane_mod.Point) void {
    const fit = j.fit;
    for (0..fit.n) |b| {
        const rc = j.cellOf(b);
        const y = plane_mod.modelAt(theta, j.fr, rc[0], rc[1]).y;
        pts[b] = .{ .row = rc[0], .col = rc[1], .y = y, .used = false };
        if (!fit.use[b] or !fit.keep[b].ok) continue;
        const pr = profiled(&fit.keep[b]) orelse continue;
        var W = rot.matMul(rot.transpose(j.M), rot.matMul(pr.S, j.M));
        for (&W) |*w| w.* *= j.h * j.h;
        const tr = W[0] + W[4] + W[8];
        if (!(tr > 0)) continue;
        const p = rot.scale(rot.matVec(rot.transpose(j.M), pr.g), j.h);
        var Wr: [9]f64 = W;
        for (0..3) |i| Wr[i * 3 + i] += 1e-6 * tr;
        const dy = solve(3, Wr, p) orelse continue;
        for (0..3) |i| W[i * 3 + i] += 1e-9 * tr;
        pts[b].y = .{ y[0] + dy[0], y[1] + dy[1], y[2] + dy[2] };
        pts[b].w = W;
        pts[b].used = true;
    }
}

/// The joint fit of the full model from the geometry as it stands, with the patterns whose
/// pull is far beyond the others' (settings.plane_outlier, as the points' fit has it) left out
/// and the fit run again. Leaves the pseudo-points in `pts`; returns the rounds taken.
fn jointPlane(j: *const Joint, which: plane_mod.Model, theta: *[plane_mod.N]f64, pts: []plane_mod.Point, outlier: f64, most: u32, a: std.mem.Allocator) !u32 {
    const fit = j.fit;
    // the model through the sources where the geometry has them: where the joint fit starts
    for (0..fit.n) |b| {
        const rc = j.cellOf(b);
        const pc = Geometry.pcOf(j.R, fit.base[b], j.h, j.h * j.fr.aspect);
        pts[b] = .{ .row = rc[0], .col = rc[1], .y = .{ pc[0] * j.fr.aspect, pc[1], pc[2] } };
    }
    // (from one centre for every cell the projective row is not told: the fit through the
    // points keeps it at zero, and the patterns say it)
    theta.* = plane_mod.fitTheta(which, pts, j.fr) orelse return error.TooFewFitPatterns;
    var rounds = try jointFit(j, which, theta, most);
    pseudoPoints(j, theta.*, pts);
    if (!(outlier > 0)) return rounds;
    const dist = try a.alloc(f64, fit.n);
    const sorted = try a.alloc(f64, fit.n);
    var round: u32 = 0;
    while (round < 4) : (round += 1) {
        var m: usize = 0;
        for (pts, 0..) |pt, b| {
            dist[b] = 0;
            if (!pt.used) continue;
            const y = plane_mod.modelAt(theta.*, j.fr, pt.row, pt.col).y;
            const r: [3]f64 = .{ pt.y[0] - y[0], pt.y[1] - y[1], pt.y[2] - y[2] };
            var q: f64 = 0;
            for (0..3) |i| for (0..3) |k| {
                q += r[i] * pt.w[i * 3 + k] * r[k];
            };
            dist[b] = @sqrt(@max(q, 0));
            sorted[m] = dist[b];
            m += 1;
        }
        if (m < 8) break;
        std.mem.sort(f64, sorted[0..m], {}, std.sort.asc(f64));
        const sigma = sorted[m / 2] / 1.538;
        var dropped: usize = 0;
        for (pts, 0..) |pt, b| {
            if (pt.used and dist[b] > outlier * sigma) {
                fit.use[b] = false;
                fit.n_used -= 1;
                dropped += 1;
            }
        }
        if (dropped == 0) break;
        rounds += try jointFit(j, which, theta, most);
        pseudoPoints(j, theta.*, pts);
    }
    return rounds;
}

/// The plane of source points (plane.zig): every fit pattern's own source point and
/// orientation by that pattern's own fit (Fit.settle), then the transformation of the
/// scan's indices that passes through them (settings.plane_kind), by weighted least squares
/// with the points far from it left out (plane_weights, plane_outlier). The geometry takes
/// the plane: the beam's position for every map cell is then the plane's, and the scan's steps
/// and the detector's pixel size say nothing of it.
pub fn plane(e: *Engine) !PlaneResult {
    return fitPlane(e, false);
}

/// `plane`, or (`compare`) with the simpler models checked as well: the geometry takes the
/// simplest the map supports (choose). One centre leaves the vote's geometry as it is.
fn fitPlane(e: *Engine, compare: bool) !PlaneResult {
    const s = &e.set;
    const before = e.geo;
    errdefer e.geo = before;
    const info = try e.info();
    const fit = try Fit.open(e, s.plane_stride, s.plane_patterns_max);
    defer fit.close();
    const a = fit.arena.allocator();
    const n = fit.n;
    const sz = fit.sz;
    const aspect = sz[1] / sz[0];
    var res: PlaneResult = .{ .start_pc = before.pc, .start_ncc = fit.ncc, .n_patterns = @intCast(fit.n_used) };
    const trust = s.plane_trust * sz[0];
    const pts = try a.alloc(plane_mod.Point, n);
    const R = e.calibration.rotation(&before);
    // a source's place, t, against (PCx · aspect, PCy, PCz): t = H R D y + const (geometry.translationFor)
    const M: rot.Mat3 = .{ -R[1], -R[0], R[2], -R[4], -R[3], R[5], -R[7], -R[6], R[8] };
    const kind: plane_mod.Kind = if (s.plane_kind == .projective) .projective else .affine;
    const fr = plane_mod.Frame.of(info.rows, info.cols, aspect);
    const full = plane_mod.Model.of(kind);
    const joint: Joint = .{ .fit = fit, .fr = fr, .R = R, .M = M, .h = sz[0], .cols = info.cols };
    var theta: [plane_mod.N]f64 = @splat(0);
    var P: [12]f64 = undefined;
    res.method = @intFromEnum(s.plane_method);
    if (s.plane_method == .joint) {
        // the orientations settled where the geometry puts the sources, then the model's numbers
        // fitted to all the patterns at once
        _ = try fit.settle(3, 0, s.plane_iters);
        res.start_ncc = fit.ncc;
        res.iterations = try jointPlane(&joint, full, &theta, pts, s.plane_outlier, s.plane_iters, a);
        P = plane_mod.toPlane(full, theta, fr) orelse return error.TooFewFitPatterns;
    } else {
        res.iterations = try fit.settle(6, trust, s.plane_iters);
        P = try pointsPlane(fit, pts, M, kind, fr, aspect, trust, info.cols, a);
    }
    res.ncc = fit.ncc;
    const h_px: f64 = @floatFromInt(info.pat_h);
    // the model: the full plane by itself, or in the automatic fit the simplest the map supports
    const model = if (compare) try choose(a, &[_]plane_mod.Model{ .centre, .line, full }, pts, fit.cells, info.rows, info.cols, fr, aspect, &res) else full;
    res.model = @intFromEnum(model);
    if (model != full) {
        if (s.plane_method == .joint) {
            // the chosen model fitted to the patterns themselves, from its fit to the points
            theta = plane_mod.fitTheta(model, pts, fr) orelse return error.TooFewFitPatterns;
            res.iterations += try jointFit(&joint, model, &theta, s.plane_iters);
            res.ncc = fit.ncc;
            P = plane_mod.toPlane(model, theta, fr) orelse return error.TooFewFitPatterns;
        } else P = plane_mod.fitModel(model, pts, fr) orelse return error.TooFewFitPatterns;
    }
    const signs = e.calibration.scanSigns(s.scan_x_sign, s.scan_y_sign);
    const said = plane_mod.saidWithNominal(P, &before, e.calibration.nominal(&before), info.rows, info.cols, aspect, h_px, signs[0], signs[1]);
    return finishPlane(e, fit, &res, &before, P, model, full, kind, pts, fr, aspect, said, info);
}

/// The way of Winkelmann et al. and kikuchipy: each pattern's own source point (after `settle`
/// over six unknowns), weighed by how sharply its pattern tells it, and the plane through
/// them, with the points far from it left out.
fn pointsPlane(fit: *Fit, pts: []plane_mod.Point, M: rot.Mat3, kind: plane_mod.Kind, fr: plane_mod.Frame, aspect: f64, trust: f64, cols: u32, a: std.mem.Allocator) ![12]f64 {
    const e = fit.e;
    const s = &e.set;
    const n = fit.n;
    const sz = fit.sz;
    const R = e.calibration.rotation(&fit.geo0);
    const info_cols = cols;
    for (0..n) |b| {
        const pc = Geometry.pcOf(R, rot.add(fit.base[b], fit.shift[b]), sz[0], sz[1]);
        var at_edge = false;
        for (fit.shift[b]) |v| at_edge = at_edge or @abs(v) >= trust * (1 - 1e-9);
        pts[b] = .{
            .col = @floatFromInt(fit.cells[b] % info_cols),
            .row = @floatFromInt(fit.cells[b] / info_cols),
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
    return P;
}

/// What follows either way of fitting: each number's standard error by blocks, the geometry
/// taking the model, the points kept for the view, and the fit patterns' orientations.
fn finishPlane(e: *Engine, fit: *Fit, res_out: *PlaneResult, before_in: *const Geometry, P: [12]f64, model: plane_mod.Model, full: plane_mod.Model, kind: plane_mod.Kind, pts: []plane_mod.Point, fr: plane_mod.Frame, aspect: f64, said: plane_mod.Said, info: anytype) !PlaneResult {
    const s = &e.set;
    const gpa = e.gpa;
    const a = fit.arena.allocator();
    const n = fit.n;
    const before = before_in.*;
    var res = res_out.*;
    const compare = res.compared != 0;
    const h_px: f64 = @floatFromInt(info.pat_h);
    const signs = e.calibration.scanSigns(s.scan_x_sign, s.scan_y_sign);

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
            const Pg = plane_mod.fitModel(model, pts, fr) orelse continue;
            const w = plane_mod.saidWithNominal(Pg, &before, e.calibration.nominal(&before), info.rows, info.cols, aspect, h_px, signs[0], signs[1]);
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

    // the geometry takes the plane (a tilt line is an affine plane; one centre leaves the
    // vote's geometry, its pattern centre and the scan's steps, as it is)
    res.kind = switch (model) {
        .centre => 0,
        .line, .affine => @intFromEnum(plane_mod.Kind.affine),
        .projective => @intFromEnum(plane_mod.Kind.projective),
    };
    res.frame = @intFromBool(s.plane_frame == .follow and model != .centre);
    res.plane = P;
    res.pc = if (model == .centre) before.pc else said.pc;
    res.said_rot = said.rot;
    res.step = said.step;
    res.skew = said.skew;
    e.geo = before;
    if (model != .centre) {
        e.geo.plane_kind = res.kind;
        e.geo.plane = P;
        e.geo.pc = said.pc;
        if (s.plane_frame == .follow) e.geo.rot = said.rot;
    }
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
    const R0 = e.calibration.nominal(&before);
    const turn = rot.qmul(rot.so3Exp(rot.matVec(R0, before.rot)), rot.qconj(rot.so3Exp(rot.matVec(R0, e.geo.rot))));
    for (0..n) |k| {
        const p: usize = @intCast(fit.phase[k]);
        const q = sym.toFz(rot.qmul(fit.q[k], turn), e.phases.items[p].master.rotations());
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
    if (compare) e.events.log("the source over the scan, each model asked where a block of the map stands from the others (mean misfit a point): one centre {d:.3}, a tilt line {d:.3}, the {s} plane {d:.3}; the map supports {s}", .{ res.cv[0], res.cv[1], @tagName(kind), res.cv[@intFromEnum(full)], switch (model) {
        .centre => "one centre: the vote's geometry stays",
        .line => "a tilt line: PCx the same everywhere, PCy and PCz along the rows",
        else => "the plane",
    } });
    e.events.log("plane of source points ({s}): {d} of {d} fit patterns used; about it they scatter by {d:.5} {d:.5} {d:.5} (PCx, PCy, PCz); mean NCC {d:.4} with each pattern at its own best (was {d:.4})", .{ @tagName(model), res.n_used, res.n_patterns, res.rmsd[0], res.rmsd[1], res.rmsd[2], res.ncc, res.start_ncc });
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
    return fitPlane(e, true);
}

test "small dense solve" {
    const x = solve(3, .{ 2, 1, 0, 1, 3, 1, 0, 1, 4 }, .{ 3, 5, 5 }).?;
    for (x) |v| try std.testing.expectApproxEqAbs(@as(f64, 1), v, 1e-12);
    try std.testing.expect(solve(2, .{ 1, 2, 2, 4 }, .{ 1, 1 }) == null);
}
