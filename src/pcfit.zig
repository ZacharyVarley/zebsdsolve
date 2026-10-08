//! Pattern-centre fit by voting.
//!
//! A few patterns on a coarse lattice over the map (settings.fit_grid², e.g. 5 × 5) are indexed
//! by spherical correlation under every candidate pattern centre of a grid around the current
//! one: a wrong pattern centre distorts the pattern on the sphere so that no rotation of the
//! master fits it, so each pattern's best normalized correlation over all orientations (and,
//! with several phases, of the phase its mutual information picks) is a surface over the
//! candidates that is high near the truth.
//!
//! How those surfaces become one choice is settings.pc_vote. `mean` averages them and takes
//! the highest candidate. That is not robust: every surface also has broad ridges that are
//! not the truth (a shift of the pattern centre traded against a turn of the crystal), the
//! ridges of many patterns overlap, and their sum pulls the maximum along them; under noise
//! the sum's maximum is anywhere. `ballots`, the default, asks instead where the patterns
//! agree: each pattern casts a ballot at each of its few best candidates (settings.pc_top_k,
//! gated at pc_gate × its own best), the ballots are spread by a Gaussian of a fixed width on
//! the detector (pc_kde_sigma), and the densest place wins; it is then moved once to the
//! weighted centre of the ballots near it (pc_inlier_radius) and found again among those
//! alone. A pattern that is wrong is outvoted rather than averaged in. This is the gated
//! top-K, density-mode consensus of my camera-plane experiments, on
//! the spherical correlation's volumes in place of sliding correlations on the detector.
//!
//! The search then repeats around the choice with a narrower grid (settings.pc_levels,
//! pc_shrink). A choice on the border of its grid means the true centre may lie outside: the
//! grid is then moved onto it at the same width (up to MAX_WALK times per level) before it
//! narrows.
//!
//! Each candidate has its own window on the sphere, so its own denominator volume of the
//! normalized correlation; the beam's position on the sample shifts the detector for each
//! pattern, as everywhere. The vote is a search: it runs at the search's resolution, at
//! settings.fit_bandwidth.
const std = @import("std");
const engine = @import("engine.zig");
const gpu_mod = @import("gpu/gpu.zig");
const spherical = @import("spherical.zig");
const detector = @import("detector.zig");
const sym = @import("sym.zig");

const Engine = engine.Engine;
const Buf = gpu_mod.Buf;

/// Times one level's grid may be re-centred, at its width, on a winner at its border.
const MAX_WALK = 4;

pub const Result = extern struct {
    /// the fitted pattern centre (Bruker convention) and where the search started
    pc: [3]f64 = .{ 0, 0, 0 },
    start: [3]f64 = .{ 0, 0, 0 },
    /// mean of the fit patterns' best normalized correlations at the fitted pattern centre,
    /// and at the start
    score: f64 = 0,
    start_score: f64 = 0,
    n_patterns: u32 = 0,
    /// candidates scored over all levels
    n_candidates: u32 = 0,
    levels: u32 = 0,
    /// 1 when the best candidate of the last level lay on the search's border
    at_border: u32 = 0,
};

/// The fit patterns: a lattice of grid × grid cells over the region of interest, each at the
/// centre of its share (the whole region when it is smaller than the lattice). The vote's
/// lattice is settings.fit_grid, the pose fit's settings.pose_grid.
pub fn fitCells(e: *Engine, grid: u32) ![]u64 {
    const gpa = e.gpa;
    const region = try e.regionCells();
    defer gpa.free(region);
    const i = try e.info();
    // the region is a rectangle: recover its shape from its corner cells
    const c_first = region[0];
    const c_last = region[region.len - 1];
    const x0: u64 = c_first % i.cols;
    const y0: u64 = c_first / i.cols;
    const w: u64 = c_last % i.cols - x0 + 1;
    const h: u64 = c_last / i.cols - y0 + 1;
    const gy: u64 = @min(grid, h);
    const gx: u64 = @min(grid, w);
    const cells = try gpa.alloc(u64, @intCast(gy * gx));
    for (0..@intCast(gy)) |r| for (0..@intCast(gx)) |c| {
        const row = y0 + (2 * r + 1) * h / (2 * gy);
        const col = x0 + (2 * c + 1) * w / (2 * gx);
        cells[r * @as(usize, @intCast(gx)) + c] = row * i.cols + col;
    };
    return cells;
}

fn axis(gpa: std.mem.Allocator, centre: f64, span: f64, steps: u32) ![]f64 {
    const a = try gpa.alloc(f64, steps);
    for (0..steps) |k| {
        a[k] = if (steps == 1) centre else centre - span + 2 * span * @as(f64, @floatFromInt(k)) / @as(f64, @floatFromInt(steps - 1));
    }
    return a;
}

/// Offset of a parabola's maximum through three equally spaced values (in steps).
fn vertex(lo: f64, mid: f64, hi: f64) f64 {
    const den = lo - 2 * mid + hi;
    if (den >= -1e-15) return 0;
    return std.math.clamp(0.5 * (lo - hi) / den, -0.5, 0.5);
}

/// One grid of the search as it was scored: kept so the front ends can show how the vote went.
pub const Level = struct {
    /// candidate values along x, y, z
    axes: [3][]f64,
    /// mean vote per candidate (z-major: index = x + nx · (y + ny · z))
    scores: []f32,
    /// per fit pattern: the candidate it scored highest under, and its score there
    pat_best: []u32,
    pat_ncc: []f32,
    /// every fit pattern's own score under every candidate (pattern-major: pattern · n + candidate)
    pat_scores: []f32,
    /// the density of the patterns' ballots per candidate (empty when the mean vote decided)
    density: []f32 = &.{},
    /// where the grid's choice lies, in candidate steps along x, y, z (between candidates)
    peak: [3]f64 = .{ 0, 0, 0 },
    /// ballots cast, and those within the inlier radius of the choice
    ballots: u32 = 0,
    inliers: u32 = 0,
    /// the winner
    best: u32,
    /// 1 when the winner was on the border and the grid moved onto it at the same width
    walked: u32,

    pub fn deinit(self: *Level, gpa: std.mem.Allocator) void {
        for (self.axes) |a| gpa.free(a);
        gpa.free(self.scores);
        gpa.free(self.pat_best);
        gpa.free(self.pat_ncc);
        gpa.free(self.pat_scores);
        gpa.free(self.density);
    }
};

const Ballot = struct { at: [3]f64, w: f64 };

/// The maximum of a volume (x innermost), each coordinate moved between candidates by the
/// parabola through its neighbours.
fn peakOf(m: []const f32, n: [3]u32) [3]f64 {
    var best: usize = 0;
    for (m, 0..) |v, i| {
        if (v > m[best]) best = i;
    }
    const at = [3]u32{ @intCast(best % n[0]), @intCast((best / n[0]) % n[1]), @intCast(best / (n[0] * n[1])) };
    const strides = [3]usize{ 1, n[0], n[0] * n[1] };
    var out: [3]f64 = undefined;
    for (0..3) |a| {
        out[a] = @floatFromInt(at[a]);
        if (n[a] < 3 or at[a] == 0 or at[a] == n[a] - 1) continue;
        out[a] += vertex(m[best - strides[a]], m[best], m[best + strides[a]]);
    }
    return out;
}

/// Spread ballots over the candidates by a Gaussian of `sigma[a]` steps along axis a (out to
/// three sigma; what would fall outside the grid is lost).
fn splat(ballots: []const Ballot, n: [3]u32, sigma: [3]f64, out: []f32) void {
    @memset(out, 0);
    var r: [3]i64 = undefined;
    var inv: [3]f64 = undefined;
    for (0..3) |a| {
        r[a] = if (n[a] > 1) @intFromFloat(@ceil(3 * sigma[a])) else 0;
        inv[a] = 1.0 / (2 * sigma[a] * sigma[a]);
    }
    for (ballots) |b| {
        const c = [3]i64{ @intFromFloat(@round(b.at[0])), @intFromFloat(@round(b.at[1])), @intFromFloat(@round(b.at[2])) };
        var z = @max(c[2] - r[2], 0);
        while (z <= @min(c[2] + r[2], @as(i64, n[2]) - 1)) : (z += 1) {
            var y = @max(c[1] - r[1], 0);
            while (y <= @min(c[1] + r[1], @as(i64, n[1]) - 1)) : (y += 1) {
                var x = @max(c[0] - r[0], 0);
                while (x <= @min(c[0] + r[0], @as(i64, n[0]) - 1)) : (x += 1) {
                    const d = [3]f64{ @floatFromInt(x - c[0]), @floatFromInt(y - c[1]), @floatFromInt(z - c[2]) };
                    out[@intCast(x + n[0] * (y + n[1] * z))] += @floatCast(b.w * @exp(-(d[0] * d[0] * inv[0] + d[1] * d[1] * inv[1] + d[2] * d[2] * inv[2])));
                }
            }
        }
    }
}

/// The choice of a grid by ballots (settings.pc_vote): each pattern's best candidates, gated
/// against its own best and weighted by their scores, spread into a density whose mode wins;
/// the mode is moved once to the weighted centre of the ballots near it and found again among
/// those alone. `step`: the candidates' spacing along x, y, z (the spread and the inlier
/// radius are widths on the detector, the same on every grid). Returns where the choice lies
/// in candidate steps; `density` gets the first density.
fn consensus(gpa: std.mem.Allocator, set: *const engine.Settings, pat_scores: []const f32, n_pat: u32, n: [3]u32, step: [3]f64, density: []f32, counts: *[2]u32) ![3]f64 {
    const n_c: usize = n[0] * n[1] * n[2];
    const k: usize = @min(set.pc_top_k, n_c);
    // (never narrower than the candidates can show)
    var sigma: [3]f64 = undefined;
    for (0..3) |a| sigma[a] = if (step[a] > 0) @max(set.pc_kde_sigma / step[a], 0.75) else 1;
    var ballots: std.ArrayList(Ballot) = .empty;
    defer ballots.deinit(gpa);
    const order = try gpa.alloc(u32, n_c);
    defer gpa.free(order);
    const half: i64 = if (set.pc_nms > 1) @intCast(set.pc_nms / 2) else 0;
    for (0..n_pat) |b| {
        const sc = pat_scores[b * n_c ..][0..n_c];
        var top: f32 = -std.math.inf(f32);
        for (sc) |v| top = @max(top, v);
        // its candidates from the best down, as far as the K-th ballot needs
        for (order, 0..) |*o, i| o.* = @intCast(i);
        std.mem.sort(u32, order, sc, struct {
            fn before(v: []const f32, a: u32, c: u32) bool {
                return v[a] > v[c];
            }
        }.before);
        var cast: usize = 0;
        for (order) |c| {
            if (cast == k or !(sc[c] >= set.pc_gate * top) or !(sc[c] > 0)) break;
            const at = [3]i64{ c % n[0], (c / n[0]) % n[1], c / (n[0] * n[1]) };
            if (half > 0) {
                // only where the pattern's score is the highest of its neighbourhood
                var is_max = true;
                var z = @max(at[2] - half, 0);
                while (z <= @min(at[2] + half, @as(i64, n[2]) - 1) and is_max) : (z += 1) {
                    var y = @max(at[1] - half, 0);
                    while (y <= @min(at[1] + half, @as(i64, n[1]) - 1) and is_max) : (y += 1) {
                        var x = @max(at[0] - half, 0);
                        while (x <= @min(at[0] + half, @as(i64, n[0]) - 1)) : (x += 1) {
                            if (sc[@intCast(x + n[0] * (y + n[1] * z))] > sc[c]) is_max = false;
                        }
                    }
                }
                if (!is_max) continue;
            }
            try ballots.append(gpa, .{ .at = .{ @floatFromInt(at[0]), @floatFromInt(at[1]), @floatFromInt(at[2]) }, .w = sc[c] });
            cast += 1;
        }
    }
    counts.* = .{ @intCast(ballots.items.len), 0 };
    splat(ballots.items, n, sigma, density);
    var pk = peakOf(density, n);
    if (ballots.items.len == 0) return pk;
    const r2 = set.pc_inlier_radius * set.pc_inlier_radius;
    const near = struct {
        fn of(b: Ballot, p: [3]f64, st: [3]f64, lim: f64) bool {
            const d = [3]f64{ (b.at[0] - p[0]) * st[0], (b.at[1] - p[1]) * st[1], (b.at[2] - p[2]) * st[2] };
            return d[0] * d[0] + d[1] * d[1] + d[2] * d[2] <= lim;
        }
    }.of;
    // once to the weighted centre of the ballots around the mode …
    var sum = [4]f64{ 0, 0, 0, 0 };
    for (ballots.items) |b| if (near(b, pk, step, r2)) {
        for (0..3) |a| sum[a] += b.w * b.at[a];
        sum[3] += b.w;
    };
    if (sum[3] > 0) pk = .{ sum[0] / sum[3], sum[1] / sum[3], sum[2] / sum[3] };
    // … then the mode of those ballots alone
    var kept: usize = 0;
    for (ballots.items) |b| if (near(b, pk, step, r2)) {
        ballots.items[kept] = b;
        kept += 1;
    };
    counts[1] = @intCast(kept);
    if (kept == 0) return pk;
    const inl = try gpa.alloc(f32, n_c);
    defer gpa.free(inl);
    splat(ballots.items[0..kept], n, sigma, inl);
    return peakOf(inl, n);
}

pub fn clearHistory(e: *Engine) void {
    for (e.pc_history.items) |*l| l.deinit(e.gpa);
    e.pc_history.clearRetainingCapacity();
}

pub fn run(e: *Engine) !Result {
    const gpa = e.gpa;
    clearHistory(e);
    const g = &e.g;
    const s = &e.set;
    e.cancel = false;
    const lv = e.levelOf(.search);
    const k_lv = @intFromEnum(lv);
    const bd = try e.band(s.fit_bandwidth, lv);
    const sp = &e.sp;
    const n_ph = e.phases.items.len;
    const sz = try e.sizeOf(lv);
    const fr = try e.frame(lv);

    const cells = try fitCells(e, e.set.fit_grid);
    errdefer gpa.free(cells);
    const n_pat: u32 = @intCast(cells.len);
    try e.uploadCells(cells, &.{lv});
    const pats = e.pats[k_lv];

    var tile: u32 = 4096;
    for (bd.ph.items) |*x| tile = @min(tile, sp.tileItems(&x.*.?.gpu, @as(u64, s.tile_mb) << 20));
    // whole candidates per tile when a candidate's patterns fit one, else one candidate in parts
    const cand_per_tile: u32 = @max(1, tile / n_pat);
    const chunk: u32 = @min(tile, n_pat);

    var res: Result = .{ .start = e.geo.pc, .n_patterns = n_pat, .levels = s.pc_levels };
    var pc = e.geo.pc;
    var span = [3]f64{ s.pc_span, s.pc_span, s.pc_span_z };
    const steps = [3]u32{ s.pc_steps, s.pc_steps, s.pc_steps_z };
    const n_c: u32 = steps[0] * steps[1] * steps[2];
    if (@as(u64, n_c) * n_pat > 4 << 20) return error.SearchTooLarge;

    // per phase: every candidate's peaks of this level, and (several phases) its records
    const lvl_peaks = try gpa.alloc(Buf, n_ph);
    defer {
        for (lvl_peaks) |*b| g.release(b);
        gpa.free(lvl_peaks);
    }
    for (lvl_peaks) |*b| b.* = g.storage(@as(u64, n_c) * n_pat * 32);
    var lvl_recs: Buf = if (n_ph > 1) g.storage(@as(u64, n_ph) * n_c * n_pat * 32) else .{};
    defer g.release(&lvl_recs);
    const rdens = try gpa.alloc(Buf, n_ph);
    defer {
        for (rdens) |*b| g.release(b);
        gpa.free(rdens);
    }
    @memset(rdens, .{});
    var items_buf: Buf = .{};
    defer g.release(&items_buf);

    const peaks_h = try gpa.alloc(f32, @as(usize, n_ph) * n_c * n_pat * 8);
    defer gpa.free(peaks_h);
    const recs_h = try gpa.alloc(engine.Record, if (n_ph > 1) @as(usize, n_ph) * n_c * n_pat else 0);
    defer gpa.free(recs_h);
    const scores = try gpa.alloc(f32, n_c);
    errdefer gpa.free(scores);
    var axes: [3][]f64 = .{ &.{}, &.{}, &.{} };
    errdefer for (axes) |a| gpa.free(a);
    const windows = try gpa.alloc(spherical.Item, cand_per_tile);
    defer gpa.free(windows);
    const items = try gpa.alloc(spherical.Item, @as(usize, cand_per_tile) * n_pat);
    defer gpa.free(items);
    // each pattern's beam offset (pixels of the search's level), the same under every candidate
    const offs = try gpa.alloc([3]f64, n_pat);
    defer gpa.free(offs);
    for (cells, 0..) |c, b| offs[b] = e.offsetOf(&e.geo, c, lv);

    var best_c: u32 = 0;
    var level: u32 = 0;
    var walked: u32 = 0;
    while (level < s.pc_levels) {
        for (0..3) |a| {
            gpa.free(axes[a]);
            axes[a] = &.{};
            axes[a] = try axis(gpa, pc[a], span[a], steps[a]);
        }
        var c0: u32 = 0;
        var since_wait: u32 = 0;
        while (c0 < n_c) : (c0 += cand_per_tile) {
            if (e.cancel) return error.Cancelled;
            const k = @min(cand_per_tile, n_c - c0);
            for (0..k) |j| {
                const c = c0 + @as(u32, @intCast(j));
                var geo = e.geo;
                geo.pc = .{ axes[0][c % steps[0]], axes[1][(c / steps[0]) % steps[1]], axes[2][c / (steps[0] * steps[1])] };
                const t = geo.translation(sz[0], sz[1]);
                windows[j] = .{ @floatCast(t[0]), @floatCast(t[1]), @floatCast(t[2]), 0 };
                for (0..n_pat) |b| {
                    items[j * n_pat + b] = .{ @floatCast(t[0] - offs[b][0]), @floatCast(t[1] - offs[b][1]), @floatCast(t[2] - offs[b][2]), @floatFromInt(b) };
                }
            }
            for (bd.ph.items, 0..) |*x, pi| sp.denominators(&x.*.?.gpu, windows[0..k], fr, &rdens[pi]);
            // the tile's items in chunks: all of them at once when whole candidates fit
            const n_items = k * n_pat;
            var first: u32 = 0;
            while (first < n_items) {
                const ni = if (cand_per_tile > 1 or n_pat <= tile) n_items else @min(chunk, n_items - first);
                sp.coefficients(pats, items[first..][0..ni], fr, spherical.FLAG_CENTER | spherical.FLAG_NORM, &sp.alm, &sp.stats);
                for (e.phases.items, 0..) |*p, pi| {
                    // whole candidates: item b reads volume b / n_pat; a part of one candidate: volume 0
                    sp.correlate(&bd.ph.items[pi].?.gpu, ni, rdens[pi].at(8), if (ni == n_items) n_pat else ni);
                    g.copy(sp.peaks, 0, lvl_peaks[pi], (@as(u64, c0) * n_pat + first) * 32, @as(u64, ni) * 32);
                    if (n_ph > 1) {
                        // the phases compete on mutual information at their own best orientations
                        g.ensure(&items_buf, @as(u64, ni) * 16);
                        g.writeSlice(items_buf, 0, spherical.Item, items[first..][0..ni]);
                        e.det.seed(fr, items_buf, sp.peaks, 0, ni);
                        e.det.refine(p.det[k_lv], fr, pats, ni, .{ .iters = 0, .nmi_bins = s.nmi_bins }, lvl_recs, @intCast((pi * n_c + c0) * n_pat + first));
                    }
                }
                first += ni;
                since_wait += ni;
            }
            if (since_wait >= 2048 and c0 + cand_per_tile < n_c) {
                try g.wait();
                since_wait = 0;
                e.events.emit(.{ .kind = "progress", .stage = "fit_pc", .done = @min(level, s.pc_levels - 1) * n_c + c0 + k, .total = s.pc_levels * n_c });
            }
        }
        const per_phase = @as(usize, n_c) * n_pat * 8;
        for (lvl_peaks, 0..) |b, pi| g.read(b, 0, std.mem.sliceAsBytes(peaks_h[pi * per_phase ..][0..per_phase]));
        if (n_ph > 1) g.read(lvl_recs, 0, std.mem.sliceAsBytes(recs_h));
        try g.wait();

        // a candidate nearer the source than pc_z_min has no score, and neither has one whose
        // score is not a number
        for (0..n_c) |c| {
            const near = axes[2][c / (steps[0] * steps[1])] < s.pc_z_min;
            for (0..n_ph) |p| for (0..n_pat) |b| {
                const v = &peaks_h[p * per_phase + (c * n_pat + b) * 8 + 4];
                if (near or !std.math.isFinite(v.*)) v.* = 0;
            };
        }

        // the votes
        best_c = 0;
        for (0..n_c) |c| {
            var sum: f64 = 0;
            for (0..n_pat) |b| {
                const at = c * n_pat + b;
                var p: usize = 0;
                if (n_ph > 1) {
                    var bv: f32 = -std.math.inf(f32);
                    for (0..n_ph) |q| {
                        const r = recs_h[q * n_c * n_pat + at];
                        const v = if (s.phase_by == .nmi) r.nmi else r.ncc;
                        if (v > bv) {
                            bv = v;
                            p = q;
                        }
                    }
                }
                sum += peaks_h[p * per_phase + at * 8 + 4];
            }
            scores[c] = @floatCast(sum / @as(f64, @floatFromInt(n_pat)));
            if (scores[c] > scores[best_c]) best_c = @intCast(c);
        }
        res.n_candidates += n_c;
        {
            // this grid, for the record
            var lvl: Level = .{ .axes = .{ &.{}, &.{}, &.{} }, .scores = &.{}, .pat_best = &.{}, .pat_ncc = &.{}, .pat_scores = &.{}, .best = best_c, .walked = 0 };
            errdefer lvl.deinit(gpa);
            for (0..3) |a| lvl.axes[a] = try gpa.dupe(f64, axes[a]);
            lvl.scores = try gpa.dupe(f32, scores);
            lvl.pat_best = try gpa.alloc(u32, n_pat);
            lvl.pat_ncc = try gpa.alloc(f32, n_pat);
            lvl.pat_scores = try gpa.alloc(f32, @as(usize, n_pat) * n_c);
            for (0..n_pat) |b| {
                var top: f32 = -std.math.inf(f32);
                var at_c: u32 = 0;
                for (0..n_c) |c| {
                    const at = c * n_pat + b;
                    var p: usize = 0;
                    if (n_ph > 1) {
                        var bv: f32 = -std.math.inf(f32);
                        for (0..n_ph) |q| {
                            const r = recs_h[q * n_c * n_pat + at];
                            const v = if (s.phase_by == .nmi) r.nmi else r.ncc;
                            if (v > bv) {
                                bv = v;
                                p = q;
                            }
                        }
                    }
                    const v = peaks_h[p * per_phase + at * 8 + 4];
                    lvl.pat_scores[b * n_c + c] = v;
                    if (v > top) {
                        top = v;
                        at_c = @intCast(c);
                    }
                }
                lvl.pat_best[b] = at_c;
                lvl.pat_ncc[b] = top;
            }
            try e.pc_history.append(gpa, lvl);
        }
        // the grid's choice, in candidate steps: the mean vote's best candidate (moved between
        // candidates on the last grid by the parabola through its neighbours), or the ballots'
        var bi = [3]u32{ best_c % steps[0], (best_c / steps[0]) % steps[1], best_c / (steps[0] * steps[1]) };
        var peak = [3]f64{ @floatFromInt(bi[0]), @floatFromInt(bi[1]), @floatFromInt(bi[2]) };
        const lvl = &e.pc_history.items[e.pc_history.items.len - 1];
        if (s.pc_vote == .ballots) {
            lvl.density = try gpa.alloc(f32, n_c);
            var counts: [2]u32 = .{ 0, 0 };
            var step: [3]f64 = undefined;
            for (0..3) |a| step[a] = if (steps[a] > 1) axes[a][1] - axes[a][0] else 0;
            peak = try consensus(gpa, s, lvl.pat_scores, n_pat, steps, step, lvl.density, &counts);
            lvl.ballots = counts[0];
            lvl.inliers = counts[1];
            for (0..3) |a| bi[a] = @intFromFloat(std.math.clamp(@round(peak[a]), 0, @as(f64, @floatFromInt(steps[a] - 1))));
            best_c = bi[0] + steps[0] * (bi[1] + steps[1] * bi[2]);
            lvl.best = best_c;
        } else if (level + 1 == s.pc_levels) {
            const strides = [3]u32{ 1, steps[0], steps[0] * steps[1] };
            for (0..3) |a| {
                if (steps[a] < 3 or bi[a] == 0 or bi[a] == steps[a] - 1) continue;
                peak[a] += vertex(scores[best_c - strides[a]], scores[best_c], scores[best_c + strides[a]]);
            }
        }
        lvl.peak = peak;
        if (level == 0 and walked == 0) {
            // the start's own score: the candidate at the grid's centre (odd step counts)
            const mid = (steps[0] / 2) + steps[0] * ((steps[1] / 2) + steps[1] * (steps[2] / 2));
            res.start_score = scores[mid];
        }
        for (0..3) |a| pc[a] = if (steps[a] > 1) axes[a][0] + peak[a] * (axes[a][1] - axes[a][0]) else axes[a][0];
        res.score = scores[best_c];
        res.at_border = 0;
        for (0..3) |a| {
            if (steps[a] > 1 and (bi[a] == 0 or bi[a] == steps[a] - 1)) res.at_border = 1;
        }
        e.events.emit(.{ .kind = "pc", .level = level, .pc = pc, .score = res.score, .span = span, .at_border = res.at_border != 0 });
        if (res.at_border != 0 and walked < MAX_WALK) {
            // the peak may lie outside this grid: look again around the winner, as wide
            lvl.walked = 1;
            walked += 1;
            continue;
        }
        walked = 0;
        level += 1;
        if (level < s.pc_levels) {
            for (&span) |*v| v.* *= s.pc_shrink;
        }
    }

    // keep the outcome: the geometry, the last level's votes, the fit patterns' orientations
    e.geo.pc = pc;
    res.pc = pc;
    gpa.free(e.pc_scores);
    e.pc_scores = scores;
    for (0..3) |a| {
        gpa.free(e.pc_axes[a]);
        e.pc_axes[a] = axes[a];
    }
    gpa.free(e.fit_cells);
    gpa.free(e.fit_quats);
    gpa.free(e.fit_phase);
    gpa.free(e.fit_ncc);
    e.fit_cells = cells;
    e.fit_quats = &.{};
    e.fit_phase = &.{};
    e.fit_ncc = &.{};
    e.fit_quats = try gpa.alloc(f32, 4 * @as(usize, n_pat));
    e.fit_phase = try gpa.alloc(i32, n_pat);
    e.fit_ncc = try gpa.alloc(f32, n_pat);
    const per_phase = @as(usize, n_c) * n_pat * 8;
    for (0..n_pat) |b| {
        const at = @as(usize, best_c) * n_pat + b;
        var p: usize = 0;
        if (n_ph > 1) {
            var bv: f32 = -std.math.inf(f32);
            for (0..n_ph) |q| {
                const r = recs_h[q * n_c * n_pat + at];
                const v = if (s.phase_by == .nmi) r.nmi else r.ncc;
                if (v > bv) {
                    bv = v;
                    p = q;
                }
            }
        }
        const pk = peaks_h[p * per_phase + at * 8 ..][0..8];
        const q = sym.toFz(.{ pk[0], pk[1], pk[2], pk[3] }, e.phases.items[p].master.rotation);
        for (0..4) |k| e.fit_quats[4 * b + k] = @floatCast(q[k]);
        e.fit_phase[b] = @intCast(p);
        e.fit_ncc[b] = pk[4];
    }
    // the reference geometry changed: its denominators are stale (keyed, so nothing to do here)
    e.events.log("pattern centre {d:.5} {d:.5} {d:.5}  mean NCC {d:.4} (was {d:.4})", .{ pc[0], pc[1], pc[2], res.score, res.start_score });
    return res;
}
