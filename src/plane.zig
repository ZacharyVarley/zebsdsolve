//! The plane of source points: where the beam stood for every map cell, as the detector sees it.
//!
//! A pattern by itself tells where its source was against the detector (three numbers, with
//! its orientation three more). Over a scan those points lie on the sample's surface, and
//! Winkelmann, Nolze, Cios, Tokarski and Bała (Materials 13 (2020) 2816) take them as a
//! transformation of the scan's indices, fitted by least squares to the points measured on a
//! few patterns: affine (the scan a parallelogram on a plane: nine numbers) or projective (the
//! beam deflected from a pivot onto a tilted plane, so that the scan's parallels need not stay
//! parallel: eleven). kikuchipy's `EBSDDetector.fit_pc` is the same. Nothing is asked of the
//! scan's step, the detector's pixel size or the sample's tilt: they are what the plane says,
//! not what it is given.
//!
//! The matrix is theirs, P (4 × 3, row-major): P · (column, row, 1) = w · (PCx, PCy, PCz, 1),
//! the source's place for that map cell in pattern-centre coordinates (Bruker's), affine when
//! its last row is (0, 0, 1). This file is the arithmetic: the fit to measured points, with a
//! weight matrix a point (posefit.zig measures them), and what a plane says of the geometry.
const std = @import("std");
const rot = @import("rot.zig");
const Geometry = @import("geometry.zig").Geometry;

pub const Kind = enum(u32) { none = 0, affine = 1, projective = 2 };

/// The models of how the source moves over the scan, simplest first, each inside the next:
/// one place for every cell; a tilt line, PCx the same everywhere and PCy, PCz moving along the
/// scan's rows (down the tilted surface: where the sample's tilt shows first) and not across
/// them; the affine plane; the projective one. The first two are affine planes with some of
/// their numbers held at zero.
pub const Model = enum(u32) {
    centre = 0,
    line = 1,
    affine = 2,
    projective = 3,

    /// The model's unknowns among the fit's (θ: x, y, z rows of (column, row, 1), then the
    /// projective row).
    pub fn free(m: Model) []const usize {
        return switch (m) {
            .centre => &[_]usize{ 2, 5, 8 },
            .line => &[_]usize{ 2, 4, 5, 7, 8 },
            .affine => &[_]usize{ 0, 1, 2, 3, 4, 5, 6, 7, 8 },
            .projective => &[_]usize{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
        };
    }

    pub fn of(kind: Kind) Model {
        return if (kind == .projective) .projective else .affine;
    }
};

/// A source point measured on one pattern.
pub const Point = struct {
    col: f64,
    row: f64,
    /// (PCx · aspect, PCy, PCz): all three in pattern heights
    y: [3]f64,
    /// how well it is told (3 × 3, the inverse of its covariance up to a factor); the identity
    /// for points taken alike
    w: [9]f64 = .{ 1, 0, 0, 0, 1, 0, 0, 0, 1 },
    used: bool = true,
};

/// Unknowns: the three rows of the affine part (9), then the two of the projective row.
pub const N = 11;

/// Solve A x = b by elimination with partial pivoting (n ≤ N; the rest of the arrays is not read).
pub fn solve(n: usize, A0: [N * N]f64, b0: [N]f64) ?[N]f64 {
    var A = A0;
    var b = b0;
    for (0..n) |c| {
        var p = c;
        for (c + 1..n) |r| {
            if (@abs(A[r * N + c]) > @abs(A[p * N + c])) p = r;
        }
        if (!(@abs(A[p * N + c]) > 1e-300)) return null;
        if (p != c) {
            for (0..n) |k| std.mem.swap(f64, &A[c * N + k], &A[p * N + k]);
            std.mem.swap(f64, &b[c], &b[p]);
        }
        for (c + 1..n) |r| {
            const f = A[r * N + c] / A[c * N + c];
            if (f == 0) continue;
            for (c..n) |k| A[r * N + k] -= f * A[c * N + k];
            b[r] -= f * b[c];
        }
    }
    var x: [N]f64 = @splat(0);
    var r: usize = n;
    while (r > 0) {
        r -= 1;
        var acc = b[r];
        for (r + 1..n) |k| acc -= A[r * N + k] * x[k];
        x[r] = acc / A[r * N + r];
    }
    return x;
}

/// The map's indices brought to about ±1 for the fit, and back.
pub const Frame = struct {
    c0: f64,
    r0: f64,
    sc: f64,
    sr: f64,
    aspect: f64,

    pub fn of(rows: u32, cols: u32, aspect: f64) Frame {
        const c0 = 0.5 * @as(f64, @floatFromInt(cols - 1));
        const r0 = 0.5 * @as(f64, @floatFromInt(rows - 1));
        return .{ .c0 = c0, .r0 = r0, .sc = @max(c0, 1), .sr = @max(r0, 1), .aspect = aspect };
    }
};

/// A model's place for a map cell and its Jacobian against the unknowns θ (3 × N, row-major), in
/// the fit's own coordinates (PCx · aspect, PCy, PCz).
pub fn modelAt(theta: [N]f64, fr: Frame, row: f64, col: f64) struct { y: [3]f64, J: [3 * N]f64 } {
    const c = (col - fr.c0) / fr.sc;
    const r = (row - fr.r0) / fr.sr;
    const f = model(theta, c, r);
    const b: [3]f64 = .{ c, r, 1 };
    var J: [3 * N]f64 = @splat(0);
    for (0..3) |k| {
        for (0..3) |j| J[k * N + 3 * k + j] = b[j] / f.w;
        J[k * N + 9] = -f.y[k] * c / f.w;
        J[k * N + 10] = -f.y[k] * r / f.w;
    }
    return .{ .y = f.y, .J = J };
}

fn model(theta: [N]f64, c: f64, r: f64) struct { y: [3]f64, w: f64 } {
    const w = theta[9] * c + theta[10] * r + 1;
    var y: [3]f64 = undefined;
    for (0..3) |k| y[k] = (theta[3 * k] * c + theta[3 * k + 1] * r + theta[3 * k + 2]) / w;
    return .{ .y = y, .w = w };
}

/// How far the projective row moves the plane's denominator, from the map's middle to its edge,
/// before the points have to say so: the prior on its two slopes (θ's last two, in the frame's
/// units). A beam swung from a pivot a working distance above a tilted sample lands where the
/// denominator 1 + y tan(tilt) / (working distance) puts it: from the middle to the edge of a
/// field 100 µm high at 15 mm, under a percent; of a millimetre, about nine.
pub const slope_prior = 0.1;

/// The plane through the points that are `used`, by weighted least squares: Σ rᵀ W r least,
/// r a point's measured place less the plane's. Affine is one linear solve; projective starts
/// there and takes Gauss–Newton steps (as Winkelmann: "starting from an affine transformation
/// and minimizing the root mean square deviation"), its two slopes held to `slope_prior` against
/// the points' scatter: points that hardly move do not tell them (any slopes fit, the plane's
/// other numbers making up for them), and would otherwise take whatever their noise says. Null
/// with too few points, or none apart, or for a projective plane whose denominator falls below a
/// tenth of its value at the map's middle somewhere on the map (where it reaches zero the
/// source is at infinity).
pub fn fit(kind: Kind, pts: []const Point, fr: Frame) ?[12]f64 {
    return fitModel(Model.of(kind), pts, fr);
}

/// `fit` for any of the models: the unknowns the model leaves free, the others held at zero.
pub fn fitModel(which: Model, pts: []const Point, fr: Frame) ?[12]f64 {
    const theta = fitTheta(which, pts, fr) orelse return null;
    return toPlane(which, theta, fr);
}

/// The model's unknowns themselves (θ, in the frame `fr`), fitted to the points.
pub fn fitTheta(which: Model, pts: []const Point, fr: Frame) ?[N]f64 {
    const all = which.free();
    var theta: [N]f64 = @splat(0);
    var count: usize = 0;
    for (pts) |p| count += @intFromBool(p.used);
    if (count * 3 < all.len + 1) return null;
    var it: u32 = 0;
    const most: u32 = if (which == .projective) 40 else 1;
    while (it < most) : (it += 1) {
        // (the first pass of the projective fit is the affine one, the projective row held at zero)
        const F = if (which == .projective and it == 0) Model.affine.free() else all;
        const m = F.len;
        var A: [N * N]f64 = @splat(0);
        var g: [N]f64 = @splat(0);
        // (Σ rᵀ W r where the plane stands, and Σ yᵀ W y to weigh it against)
        var chi2: f64 = 0;
        var yy: f64 = 0;
        for (pts) |p| {
            if (!p.used) continue;
            const c = (p.col - fr.c0) / fr.sc;
            const r = (p.row - fr.r0) / fr.sr;
            const f = model(theta, c, r);
            const b: [3]f64 = .{ c, r, 1 };
            // J (3 × m): the plane's place against each unknown
            var J: [3 * N]f64 = @splat(0);
            for (0..3) |k| {
                for (0..3) |j| J[k * N + 3 * k + j] = b[j] / f.w;
                J[k * N + 9] = -f.y[k] * c / f.w;
                J[k * N + 10] = -f.y[k] * r / f.w;
            }
            var res: [3]f64 = undefined;
            for (0..3) |k| res[k] = p.y[k] - f.y[k];
            for (0..3) |k| for (0..3) |l| {
                chi2 += res[k] * p.w[k * 3 + l] * res[l];
                yy += p.y[k] * p.w[k * 3 + l] * p.y[l];
            };
            for (F, 0..) |fi, i| {
                var wj: [3]f64 = .{ 0, 0, 0 };
                for (0..3) |k| for (0..3) |l| {
                    wj[k] += p.w[k * 3 + l] * J[l * N + fi];
                };
                for (0..3) |k| g[i] += wj[k] * res[k];
                for (F, 0..) |fj, j| {
                    for (0..3) |k| A[i * N + j] += wj[k] * J[k * N + fj];
                }
            }
        }
        if (which == .projective and it > 0) {
            // (points on a plane: the projective row they tell is zero, or, when they do not move,
            // none; either way the affine plane is the fit)
            if (chi2 <= 1e-20 * yy) break;
            // the slopes' prior, against the points' scatter about the plane (W is their inverse
            // covariance up to a factor, which Σ rᵀ W r over the degrees of freedom says)
            const scatter = chi2 / @as(f64, @floatFromInt(count * 3 - all.len));
            const lam = scatter / (slope_prior * slope_prior);
            for (F, 0..) |fi, i| {
                if (fi < 9) continue;
                A[i * N + i] += lam;
                g[i] -= lam * theta[fi];
            }
        }
        const d = solve(m, A, g) orelse return null;
        var moved: f64 = 0;
        for (F, 0..) |fi, i| {
            theta[fi] += d[i];
            moved = @max(moved, @abs(d[i]));
        }
        if (it > 0 and moved < 1e-13) break;
    }
    // (the denominator over the map, the frame's c and r running from −1 to 1, is at least
    // 1 − |θ₉| − |θ₁₀|)
    if (which == .projective and !(@abs(theta[9]) + @abs(theta[10]) <= 0.9)) return null;
    return theta;
}

/// The model's unknowns as the plane P of the scan's own indices (Bruker's PCx): P = Θ T, T
/// taking (col, row, 1) to (c, r, 1). Null where the plane would be undefined.
pub fn toPlane(which: Model, theta: [N]f64, fr: Frame) ?[12]f64 {
    var P: [12]f64 = undefined;
    for (0..4) |k| {
        const a = if (k < 3) theta[3 * k] else theta[9];
        const b = if (k < 3) theta[3 * k + 1] else theta[10];
        const c = if (k < 3) theta[3 * k + 2] else 1;
        const s: f64 = if (k == 0) 1 / fr.aspect else 1;
        P[3 * k] = s * a / fr.sc;
        P[3 * k + 1] = s * b / fr.sr;
        P[3 * k + 2] = s * (c - a * fr.c0 / fr.sc - b * fr.r0 / fr.sr);
    }
    if (!(@abs(P[11]) > 1e-12)) return null;
    const scale = 1 / P[11];
    for (&P) |*v| v.* *= scale;
    if (which != .projective) {
        P[9] = 0;
        P[10] = 0;
    }
    return P;
}

/// A point's squared distance from a plane, by its weight: rᵀ W r.
pub fn misfit(P: [12]f64, p: Point, aspect: f64) f64 {
    const pc = at(P, p.row, p.col);
    const r: [3]f64 = .{ p.y[0] - pc[0] * aspect, p.y[1] - pc[1], p.y[2] - pc[2] };
    var q: f64 = 0;
    for (0..3) |i| for (0..3) |j| {
        q += r[i] * p.w[i * 3 + j] * r[j];
    };
    return q;
}

/// The plane's source point for a map cell, in pattern-centre coordinates.
pub fn at(P: [12]f64, row: f64, col: f64) [3]f64 {
    const w = P[9] * col + P[10] * row + P[11];
    return .{
        (P[0] * col + P[1] * row + P[2]) / w,
        (P[3] * col + P[4] * row + P[5]) / w,
        (P[6] * col + P[7] * row + P[8]) / w,
    };
}

/// What a plane says of the geometry, at the map's middle.
pub const Said = struct {
    /// the source's place there
    pc: [3]f64,
    /// The sample's frame as the plane has it, against the nominal one (rotation vector in the
    /// detector's axes, as geometry.rot: swing, tilt, twist): its normal is the plane's, and
    /// its x runs along the scan's rows. The twist is then the scan's own turn in the sample's
    /// plane against the detector, as much as the detector's.
    rot: [3]f64,
    /// how far the beam goes for a row and for a column, against what the scan's steps and
    /// the detector's pixel size say (1: as given; 0: no pixel size to say it)
    step: [2]f64,
    /// the angle between the scan's two directions, off a right angle (radians)
    skew: f64,
};

/// `rows`, `cols`: the map's; `h_px`: the stored pattern's height in pixels; `sx`, `sy`: the
/// scan's signs (settings.scan_x_sign, scan_y_sign).
pub fn said(P: [12]f64, geo: *const Geometry, rows: u32, cols: u32, aspect: f64, h_px: f64, sx: i32, sy: i32) Said {
    return saidWithNominal(P, geo, geo.nominal(), rows, cols, aspect, h_px, sx, sy);
}

/// As `said`, measured against an acquisition's full calibrated nominal attitude.
pub fn saidWithNominal(P: [12]f64, geo: *const Geometry, nominal: rot.Mat3, rows: u32, cols: u32, aspect: f64, h_px: f64, sx: i32, sy: i32) Said {
    const c0 = 0.5 * @as(f64, @floatFromInt(cols - 1));
    const r0 = 0.5 * @as(f64, @floatFromInt(rows - 1));
    // the source in the detector's axes (down its rows, across its columns, along its normal),
    // in pattern heights, and how it moves for a row and for a column
    const place = struct {
        fn of(pc: [3]f64, asp: f64) rot.Vec3 {
            return .{ 0.5 - pc[1], (0.5 - pc[0]) * asp, pc[2] };
        }
    };
    const h = 0.5;
    const a = rot.scale(rot.sub(place.of(at(P, r0 + h, c0), aspect), place.of(at(P, r0 - h, c0), aspect)), 1 / (2 * h));
    const b = rot.scale(rot.sub(place.of(at(P, r0, c0 + h), aspect), place.of(at(P, r0, c0 - h), aspect)), 1 / (2 * h));
    // u(s) = u0 − Rᵀ s, s = (sy · row step, sx · column step, 0): the sample's x and y as the detector sees them
    const xs = rot.scale(a, -@as(f64, @floatFromInt(sy)));
    const ys = rot.scale(b, -@as(f64, @floatFromInt(sx)));
    const lx = @sqrt(rot.dot(xs, xs));
    const ly = @sqrt(rot.dot(ys, ys));
    var out: Said = .{ .pc = at(P, r0, c0), .rot = .{ 0, 0, 0 }, .step = .{ 0, 0 }, .skew = 0 };
    if (geo.px_um > 0 and h_px > 0) {
        out.step = .{ lx * h_px * geo.px_um / geo.step_y_um, ly * h_px * geo.px_um / geo.step_x_um };
    }
    if (!(lx > 0)) return out;
    const x = rot.scale(xs, 1 / lx);
    // (a tilt line says nothing across the rows: the sample's y is then the nominal one, made
    // square to the x the line gives, and the scan is square)
    const yn = if (ly > 0) rot.scale(ys, 1 / ly) else blk: {
        const y0: rot.Vec3 = .{ nominal[3], nominal[4], nominal[5] };
        const v = rot.sub(y0, rot.scale(x, rot.dot(y0, x)));
        const lv = @sqrt(rot.dot(v, v));
        if (!(lv > 1e-9)) return out;
        break :blk rot.scale(v, 1 / lv);
    };
    var z = rot.cross(x, yn);
    const lz = @sqrt(rot.dot(z, z));
    if (!(lz > 1e-9)) return out;
    z = rot.scale(z, 1 / lz);
    const y = rot.cross(z, x);
    // R takes the detector's coordinates to the sample's: its rows are the sample's axes there
    const R: rot.Mat3 = .{ x[0], x[1], x[2], y[0], y[1], y[2], z[0], z[1], z[2] };
    out.rot = rot.so3Log(rot.fromMat(rot.matMul(rot.transpose(nominal), R)));
    out.skew = std.math.asin(std.math.clamp(rot.dot(x, yn), -1, 1));
    return out;
}

test "each model is found again from its own points, and holds its zeros" {
    const fr = Frame.of(40, 60, 1.25);
    // a tilt line: PCx the same everywhere, PCy and PCz moving with the row alone
    const L: [12]f64 = .{ 0, 0, 0.43, 0, -3.6e-4, 0.22, 0, 1.3e-4, 0.5, 0, 0, 1 };
    var pts: [49]Point = undefined;
    for (0..7) |i| for (0..7) |j| {
        const row: f64 = @floatFromInt(i * 6 + 1);
        const col: f64 = @floatFromInt(j * 9 + 2);
        const pc = at(L, row, col);
        pts[i * 7 + j] = .{ .col = col, .row = row, .y = .{ pc[0] * fr.aspect, pc[1], pc[2] } };
    };
    for ([_]Model{ .line, .affine }) |m| {
        const P = fitModel(m, &pts, fr).?;
        for (P, L) |got, want| try std.testing.expectApproxEqAbs(want, got, 1e-11);
    }
    // one centre: the points' mean, every slope zero
    const C = fitModel(.centre, &pts, fr).?;
    var mean: [3]f64 = .{ 0, 0, 0 };
    for (pts) |q| {
        const pc = at(L, q.row, q.col);
        for (0..3) |k| mean[k] += pc[k] / pts.len;
    }
    for (0..3) |k| try std.testing.expectApproxEqAbs(mean[k], C[3 * k + 2], 1e-11);
    for ([_]usize{ 0, 1, 3, 4, 6, 7 }) |i| try std.testing.expectEqual(@as(f64, 0), C[i]);
}

test "a plane is found again from its own points, affine and projective" {
    const fr = Frame.of(40, 60, 1.25);
    const A: [12]f64 = .{ -4.0e-4, 1.0e-5, 0.43, 2.0e-5, -3.6e-4, 0.22, -1.0e-5, 1.3e-4, 0.5, 0, 0, 1 };
    var Pp = A;
    Pp[9] = 2.0e-5;
    Pp[10] = -3.0e-5;
    for ([_]Kind{ .affine, .projective }) |kind| {
        const truth = if (kind == .affine) A else Pp;
        var pts: [49]Point = undefined;
        for (0..7) |i| for (0..7) |j| {
            const row: f64 = @floatFromInt(i * 6 + 1);
            const col: f64 = @floatFromInt(j * 9 + 2);
            const pc = at(truth, row, col);
            pts[i * 7 + j] = .{ .col = col, .row = row, .y = .{ pc[0] * fr.aspect, pc[1], pc[2] } };
        };
        const P = fit(kind, &pts, fr).?;
        for (P, truth) |got, want| try std.testing.expectApproxEqAbs(want, got, 1e-11);
    }
}

test "a projective plane through points that hardly move keeps its slopes near zero" {
    // nine points of a 55 × 75 map, all at one source, then with a little noise: the slopes are
    // not told, and the plane at the map's middle (θ's constants, the denominator 1 there) stays
    // near the points
    const fr = Frame.of(55, 75, 1);
    const y: [3]f64 = .{ 0.4233, 0.2136, 0.5021 };
    var pts: [9]Point = undefined;
    var seed: u64 = 7;
    for ([_]f64{ 0, 1e-4 }) |noise| {
        for (0..3) |i| for (0..3) |j| {
            var p: Point = .{ .row = @floatFromInt(20 + i), .col = @floatFromInt(32 + j), .y = y };
            for (&p.y) |*v| {
                seed = seed *% 6364136223846793005 +% 1442695040888963407;
                v.* += noise * (@as(f64, @floatFromInt(seed >> 11)) / 9007199254740992.0 - 0.5);
            }
            pts[i * 3 + j] = p;
        };
        const theta = fitTheta(.projective, &pts, fr).?;
        if (noise == 0) {
            try std.testing.expectEqual(@as(f64, 0), theta[9]);
            try std.testing.expectEqual(@as(f64, 0), theta[10]);
        }
        try std.testing.expect(@abs(theta[9]) + @abs(theta[10]) < 0.01);
        for (0..3) |k| try std.testing.expectApproxEqAbs(y[k], theta[3 * k + 2], 0.003);
    }
}

test "the plane of the model's own scan says the model's geometry" {
    // a detector turned a little, a scan of known steps: the source points the pose gives, and
    // back from their plane
    var g: Geometry = .{ .pc = .{ 0.42, 0.21, 0.5 }, .rot = .{ 0.02, -0.03, 0.04 }, .px_um = 560, .step_x_um = 30, .step_y_um = 40 };
    const rows: u32 = 30;
    const cols: u32 = 40;
    const R = g.rotation();
    const t = g.translation(60, 60);
    var pts: [30]Point = undefined;
    const fr = Frame.of(rows, cols, 1);
    for (0..30) |k| {
        const row: f64 = @floatFromInt((k * 7) % rows);
        const col: f64 = @floatFromInt((k * 11) % cols);
        const s = g.scanOffset(row, col, fr.r0, fr.c0, -1, -1, 1);
        const pc = Geometry.pcOf(R, rot.sub(t, s), 60, 60);
        pts[k] = .{ .col = col, .row = row, .y = pc };
    }
    const P = fit(.affine, &pts, fr).?;
    const w = said(P, &g, rows, cols, 1, 60, -1, -1);
    for (0..3) |i| {
        try std.testing.expectApproxEqAbs(g.pc[i], w.pc[i], 1e-10);
        try std.testing.expectApproxEqAbs(g.rot[i], w.rot[i], 1e-9);
    }
    try std.testing.expectApproxEqAbs(@as(f64, 1), w.step[0], 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 1), w.step[1], 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 0), w.skew, 1e-9);
}
