//! Reference-frame conventions as data, and how to find the one a file is in.
//!
//! An orientation is a rotation between a sample's frame and a crystal's, and whoever writes one
//! chooses both frames and how the rotation is written. This engine computes in one choice,
//! kikuchipy's (the engine's frame); every other is described in the same few terms, never by
//! naming who uses it:
//!   * the sample axes: which of the engine's axes, with which sign, each of X, Y and Z is (the
//!     24 right-handed ways);
//!   * passive (Bunge angles taking the sample's axes to the crystal's, as every EBSD file reads
//!     them) or active (the inverse);
//!   * the crystal axes on the lattice: X ∥ a, Z ∥ c* (the masters' own frame, EMsoft's and
//!     ebsdsim's), X ∥ a*, Z ∥ c, Y ∥ b, Z ∥ c*, or Y ∥ b*, Z ∥ c. On an orthogonal cell the four
//!     agree; on a hexagonal one the second is the first rotated 30° about c.
//! The engine's sample frame, by what it is tied to: x on the surface toward the detector, y
//! along the tilt axis, the way the pattern's columns run, z out of the surface (geometry.zig).
//! Its crystal frame is each master's.
//!
//! A file that stores its maker's orientations says which convention they are in, whatever its
//! documentation claims or its version did: `identify` compares them with this engine's, cell
//! by cell under each phase's own symmetry, over every convention. Packages disagree, and some
//! disagree with themselves; the data does not.
//!
//! What is left between the two after that is a misorientation, and it is not part of any
//! convention: it is where the two geometries differ (a detector orientation one of them did
//! not use, a pattern centre measured differently or held fixed as the beam moved) or where
//! either indexed less well. `identify` measures it (its mean, and how it changes over the
//! map's rows and columns) and reports it; nothing applies it.
const std = @import("std");
const rot = @import("rot.zig");

const Quat = rot.Quat;
const Mat3 = rot.Mat3;

/// How a frame's crystal axes sit on the lattice (two of them fixed; the third completes a
/// right-handed set).
pub const CrystalAxes = enum(u8) {
    x_a_z_cstar,
    x_astar_z_c,
    y_b_z_cstar,
    y_bstar_z_c,

    pub fn label(self: CrystalAxes) []const u8 {
        return switch (self) {
            .x_a_z_cstar => "X||a Z||c*",
            .x_astar_z_c => "X||a* Z||c",
            .y_b_z_cstar => "Y||b Z||c*",
            .y_bstar_z_c => "Y||b* Z||c",
        };
    }
};

/// A reference frame convention: where orientations written in it put the sample's and the
/// crystal's axes, and which way they rotate. The default is the engine's own.
pub const Frame = struct {
    /// the frame's sample X, Y, Z as the engine's axes: 1 x, 2 y, 3 z, negative: reversed
    axes: [3]i8 = .{ 1, 2, 3 },
    /// angles that take the crystal's axes to the sample's (the inverse of the usual reading)
    active: bool = false,
    crystal: CrystalAxes = .x_a_z_cstar,

    pub fn valid(self: Frame) bool {
        var seen: [3]bool = @splat(false);
        for (self.axes) |a| {
            const k = @abs(a);
            if (k < 1 or k > 3 or seen[k - 1]) return false;
            seen[k - 1] = true;
        }
        return rot.det(axesMatrix(self.axes)) > 0;
    }

    pub fn isEngine(self: Frame) bool {
        return std.meta.eql(self, Frame{});
    }

    /// The frame's sample axes in the engine's coordinates, as the rotation that takes the
    /// engine's coordinates to the frame's.
    pub fn sample(self: Frame) Quat {
        return rot.qstd(rot.fromMat(axesMatrix(self.axes)));
    }

    /// The frame's crystal axes in a master's Cartesian coordinates (X ∥ a, Z ∥ c*), as the
    /// rotation that takes them to the frame's, for a cell (a, b, c in Å; α, β, γ in degrees).
    /// Null where the frame's crystal axes need a cell and none is known.
    pub fn crystalOf(self: Frame, lattice: ?[6]f64) ?Quat {
        return crystalRotation(self.crystal, lattice);
    }

    /// An orientation in the engine's frame (q v q* takes sample directions to the master's
    /// crystal directions) as this frame writes it; `c` is `crystalOf` for its phase.
    pub fn toFrame(self: Frame, q: Quat, c: Quat) Quat {
        const g = rot.qmul(rot.qmul(c, q), rot.qconj(self.sample()));
        return rot.qstd(if (self.active) rot.qconj(g) else g);
    }

    /// The inverse of `toFrame`: an orientation written in this frame, in the engine's.
    pub fn fromFrame(self: Frame, g0: Quat, c: Quat) Quat {
        const g = if (self.active) rot.qconj(g0) else g0;
        return rot.qstd(rot.qmul(rot.qmul(rot.qconj(c), g), self.sample()));
    }

    /// One line of ASCII, as the exports' headers carry it: "X +y Y -x Z +z, passive, X||a* Z||c".
    pub fn describe(self: Frame, w: *std.Io.Writer) !void {
        const names = "xyz";
        for (self.axes, 0..) |a, i| {
            if (i != 0) try w.writeByte(' ');
            try w.print("{c} {c}{c}", .{ "XYZ"[i], @as(u8, if (a < 0) '-' else '+'), names[@abs(a) - 1] });
        }
        try w.print(", {s}, {s}", .{ if (self.active) "active" else "passive", self.crystal.label() });
    }
};

fn axesMatrix(axes: [3]i8) Mat3 {
    var m: Mat3 = @splat(0);
    for (axes, 0..) |a, i| {
        const k = @abs(a);
        if (k < 1 or k > 3) return m;
        m[i * 3 + k - 1] = if (a < 0) -1 else 1;
    }
    return m;
}

fn crystalRotation(axes: CrystalAxes, lattice: ?[6]f64) ?Quat {
    if (axes == .x_a_z_cstar) return rot.quat_identity;
    const l = lattice orelse return null;
    for (l) |v| if (!(v > 0) or !std.math.isFinite(v)) return null;
    const d = std.math.degreesToRadians;
    const ca = @cos(d(l[3]));
    const cb = @cos(d(l[4]));
    const cg = @cos(d(l[5]));
    const sg = @sin(d(l[5]));
    const vol = 1 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg;
    if (!(vol > 0) or !(sg > 0)) return null;
    // the lattice's vectors in the masters' Cartesian frame (a along x, c* along z)
    const a: rot.Vec3 = .{ l[0], 0, 0 };
    const b: rot.Vec3 = .{ l[1] * cg, l[1] * sg, 0 };
    const c: rot.Vec3 = .{ l[2] * cb, l[2] * (ca - cb * cg) / sg, l[2] * @sqrt(vol) / sg };
    var x: rot.Vec3 = undefined;
    var y: rot.Vec3 = undefined;
    var z: rot.Vec3 = undefined;
    switch (axes) {
        .x_a_z_cstar => unreachable,
        .x_astar_z_c => {
            x = rot.normalize(rot.cross(b, c));
            z = rot.normalize(c);
            y = rot.cross(z, x);
        },
        .y_b_z_cstar => {
            y = rot.normalize(b);
            z = rot.normalize(rot.cross(a, b));
            x = rot.cross(y, z);
        },
        .y_bstar_z_c => {
            y = rot.normalize(rot.cross(c, a));
            z = rot.normalize(c);
            x = rot.cross(y, z);
        },
    }
    // rows: the frame's axes in master coordinates, the matrix taking those to the frame's
    return rot.fromMat(.{ x[0], x[1], x[2], y[0], y[1], y[2], z[0], z[1], z[2] });
}

// ── identification ──

/// Orientations from elsewhere, one per map cell: Bunge angles (radians), and a label for the
/// phase (any number; negative: not indexed). Labels are matched to this engine's phases by
/// where they agree on the map, not by name.
pub const Reference = struct {
    euler: []const f64,
    phase: []const i32,
};

/// What `identify` needs of a phase: the rotations its master obeys and its cell.
pub const PhaseView = struct {
    rotations: []const Quat,
    lattice: ?[6]f64,
};

pub const max_phases = 16;

pub const Found = struct {
    frame: Frame = .{},
    /// cells compared: indexed in both, with matching phases
    cells: u32 = 0,
    /// of those, the fraction within 2° in the convention, and the median (degrees)
    within: f64 = 0,
    median_deg: f64 = 180,
    /// each phase's median under the best other convention, averaged (degrees): how clearly
    /// this one was told
    runner_up_deg: f64 = 180,
    /// for each phase, the reference's label it was compared with (−1: none)
    matched: [max_phases]i32 = @splat(-1),
    /// The misorientation left between the reference and this engine's orientations, as the
    /// rotation of the engine's sample axes that takes its orientations closest to the
    /// reference's (rotation vector, radians, about the engine's x, y, z): its value at the
    /// map's middle, and how much it changes from the first row to the last and from the first
    /// column to the last (a least-squares plane over the cells within a few degrees).
    misorientation: [3]f64 = @splat(0),
    misorientation_rows: [3]f64 = @splat(0),
    misorientation_cols: [3]f64 = @splat(0),
    /// the cells it was measured on
    misorientation_cells: u32 = 0,
};

/// The smallest angle (radians) between orientation a and the rotations g · b, g a phase's.
fn apart(rotations: []const Quat, a: Quat, b: Quat) f64 {
    const d = rot.qmul(a, rot.qconj(b));
    var best: f64 = 0;
    for (rotations) |g| {
        // the scalar part of g · d
        const w = @abs(g[0] * d[0] - g[1] * d[1] - g[2] * d[2] - g[3] * d[3]);
        best = @max(best, w);
    }
    return 2 * std.math.acos(@min(best, 1));
}

const Pair = struct { q: Quat, g: Quat, phase: u8, cell: u32 };

fn median(xs: []f64) f64 {
    if (xs.len == 0) return std.math.pi;
    std.mem.sort(f64, xs, {}, std.sort.asc(f64));
    return xs[xs.len / 2];
}

/// Each of this engine's phases' label in a reference: the one the reference gives most often
/// where this engine found that phase (−1: none; ties to the smaller label).
pub fn matchLabels(gpa: std.mem.Allocator, phase: []const i32, ref_phase: []const i32, n_phases: usize) ![max_phases]i32 {
    var matched: [max_phases]i32 = @splat(-1);
    var labels: std.AutoHashMapUnmanaged(u64, u32) = .empty;
    defer labels.deinit(gpa);
    for (phase, ref_phase) |p, r| {
        if (p < 0 or p >= n_phases or p >= max_phases or r < 0) continue;
        const key = (@as(u64, @intCast(p)) << 32) | @as(u32, @intCast(r));
        const got = try labels.getOrPut(gpa, key);
        if (!got.found_existing) got.value_ptr.* = 0;
        got.value_ptr.* += 1;
    }
    var most: [max_phases]u32 = @splat(0);
    var it = labels.iterator();
    while (it.next()) |kv| {
        const p: usize = @intCast(kv.key_ptr.* >> 32);
        const l: i32 = @intCast(kv.key_ptr.* & 0xffff_ffff);
        if (kv.value_ptr.* > most[p] or (kv.value_ptr.* == most[p] and l < matched[p])) {
            most[p] = kv.value_ptr.*;
            matched[p] = l;
        }
    }
    return matched;
}

/// The convention of `ref`'s frame: the one under which its orientations lie closest to this
/// engine's (`q`, quaternions per cell; `phase`, −1 where not indexed; `cols`, the map's width,
/// for where each cell is), by each phase's median over the cells both index, averaged over
/// the phases; then the misorientation that remains, measured on the cells within a few degrees.
pub fn identify(gpa: std.mem.Allocator, q: []const f32, phase: []const i32, cols: usize, ref: Reference, phases: []const PhaseView) !Found {
    const n = phase.len;
    if (q.len != 4 * n or ref.euler.len != 3 * n or ref.phase.len != n) return error.BadLength;
    if (phases.len == 0 or phases.len > max_phases) return error.NoPhases;
    var out: Found = .{};
    out.matched = try matchLabels(gpa, phase, ref.phase, phases.len);
    var all: std.ArrayList(Pair) = .empty;
    defer all.deinit(gpa);
    for (0..n) |i| {
        const p = phase[i];
        if (p < 0 or p >= phases.len or ref.phase[i] < 0 or ref.phase[i] != out.matched[@intCast(p)]) continue;
        const e = ref.euler[3 * i ..][0..3].*;
        if (!std.math.isFinite(e[0]) or !std.math.isFinite(e[1]) or !std.math.isFinite(e[2])) continue;
        const qq: Quat = .{ q[4 * i], q[4 * i + 1], q[4 * i + 2], q[4 * i + 3] };
        try all.append(gpa, .{ .q = rot.qstd(qq), .g = rot.fromBunge(e), .phase = @intCast(p), .cell = @intCast(i) });
    }
    out.cells = @intCast(all.items.len);
    if (all.items.len < 8) return error.TooFewCommonCells;
    // (grouped by phase, each scored by itself: a cubic phase cannot tell crystal axes apart,
    // and its cells must not outvote a hexagonal phase that can)
    std.mem.sort(Pair, all.items, {}, struct {
        fn lt(_: void, a: Pair, b: Pair) bool {
            return a.phase < b.phase;
        }
    }.lt);
    // the cells the conventions are told apart on
    const fit = try thin(gpa, all.items, 2000);
    defer gpa.free(fit);
    const probe = try thin(gpa, fit, 300);
    defer gpa.free(probe);
    const angles = try gpa.alloc(f64, fit.len);
    defer gpa.free(angles);

    // the crystal axes that differ for some phase in play (the rest give the same orientations)
    var crystals: [4]CrystalAxes = undefined;
    var n_crystal: usize = 0;
    for (std.enums.values(CrystalAxes)) |cand| {
        var distinct = n_crystal == 0;
        for (crystals[0..n_crystal]) |seen| {
            var differs = false;
            for (phases) |ph| {
                const a = crystalRotation(cand, ph.lattice) orelse continue;
                const b = crystalRotation(seen, ph.lattice) orelse continue;
                if (apart(ph.rotations, a, b) > 1e-6) differs = true;
            }
            distinct = differs;
            if (!distinct) break;
        }
        if (distinct) {
            crystals[n_crystal] = cand;
            n_crystal += 1;
        }
    }
    const crystal_rots = try gpa.alloc(Quat, phases.len * n_crystal);
    defer gpa.free(crystal_rots);
    for (phases, 0..) |ph, p| for (crystals[0..n_crystal], 0..) |c, k| {
        crystal_rots[p * n_crystal + k] = crystalRotation(c, ph.lattice) orelse rot.quat_identity;
    };

    // every convention, the engine's first (it wins ties)
    var best: Frame = .{};
    var best_med: f64 = std.math.inf(f64);
    var second: f64 = std.math.inf(f64);
    const perms = [_][3]i8{ .{ 1, 2, 3 }, .{ 1, 3, 2 }, .{ 2, 1, 3 }, .{ 2, 3, 1 }, .{ 3, 1, 2 }, .{ 3, 2, 1 } };
    const signs = [_][3]i8{ .{ 1, 1, 1 }, .{ 1, -1, -1 }, .{ -1, 1, -1 }, .{ -1, -1, 1 }, .{ -1, -1, -1 }, .{ -1, 1, 1 }, .{ 1, -1, 1 }, .{ 1, 1, -1 } };
    for ([_]bool{ false, true }) |active| for (0..n_crystal) |k| for (perms) |pm| for (signs) |sg| {
        const cand: Frame = .{ .axes = .{ pm[0] * sg[0], pm[1] * sg[1], pm[2] * sg[2] }, .active = active, .crystal = crystals[k] };
        if (!cand.valid()) continue;
        const med = score(cand, probe, phases, crystal_rots, n_crystal, k, angles);
        if (med < best_med - 1e-9) {
            if (best_med < second and !sameAxes(best, cand)) second = best_med;
            best_med = med;
            best = cand;
        } else if (med < second) second = med;
    };
    out.frame = best;
    const kb = std.mem.indexOfScalar(CrystalAxes, crystals[0..n_crystal], best.crystal).?;
    out.runner_up_deg = std.math.radiansToDegrees(second);

    // the convention's own numbers, over every cell compared
    const every = try gpa.alloc(f64, all.items.len);
    defer gpa.free(every);
    var close: usize = 0;
    for (all.items, every) |pr, *v| {
        v.* = apart(phases[pr.phase].rotations, pr.q, best.fromFrame(pr.g, crystal_rots[pr.phase * n_crystal + kb]));
        close += @intFromBool(v.* < std.math.degreesToRadians(2.0));
    }
    out.within = @as(f64, @floatFromInt(close)) / @as(f64, @floatFromInt(every.len));
    out.median_deg = std.math.radiansToDegrees(median(every));

    // the misorientation left: on each cell within a few degrees, the rotation m of the
    // engine's sample axes with the reference = c · gk · q · (a · m)*, gk the symmetric
    // equivalent closest to it; then the plane m = m0 + u · m_rows + w · m_cols through them,
    // u and w the cell's row and column from −½ (first) to ½ (last)
    const near = std.math.degreesToRadians(@max(5.0, 3 * out.median_deg));
    const a = rot.fromMat(axesMatrix(best.axes));
    const rows: usize = if (cols > 0) (n + cols - 1) / cols else 0;
    var ata: [9]f64 = @splat(0);
    var atb: [3][3]f64 = @splat(@splat(0));
    var used: u32 = 0;
    for (all.items) |pr| {
        const ph = phases[pr.phase];
        const c = crystal_rots[pr.phase * n_crystal + kb];
        const r = best.fromFrame(pr.g, c);
        var gk: Quat = rot.quat_identity;
        var top: f64 = -1;
        const d = rot.qmul(pr.q, rot.qconj(r));
        for (ph.rotations) |g| {
            const w = @abs(g[0] * d[0] - g[1] * d[1] - g[2] * d[2] - g[3] * d[3]);
            if (w > top) {
                top = w;
                gk = g;
            }
        }
        if (2 * std.math.acos(@min(top, 1)) > near) continue;
        // in the convention's axes, t = reference* · c · gk · q · a*; in the engine's, a* t a
        const passive = if (best.active) rot.qconj(pr.g) else pr.g;
        const t = rot.qmul(rot.qmul(rot.qmul(rot.qconj(passive), c), rot.qmul(gk, pr.q)), rot.qconj(a));
        const m = rot.so3Log(rot.qstd(rot.qmul(rot.qmul(rot.qconj(a), t), a)));
        const u: f64 = if (rows > 1) @as(f64, @floatFromInt(pr.cell / cols)) / @as(f64, @floatFromInt(rows - 1)) - 0.5 else 0;
        const w: f64 = if (cols > 1) @as(f64, @floatFromInt(pr.cell % cols)) / @as(f64, @floatFromInt(cols - 1)) - 0.5 else 0;
        const x = [3]f64{ 1, u, w };
        for (0..3) |i| {
            for (0..3) |j| ata[3 * i + j] += x[i] * x[j];
            for (0..3) |k| atb[k][i] += x[i] * m[k];
        }
        used += 1;
    }
    out.misorientation_cells = used;
    if (used >= 8) {
        // cells all in one row or one column say nothing of a change that way: hold it at zero
        for (1..3) |i| {
            const spread = ata[3 * i + i] - ata[i] * ata[i] / ata[0];
            if (spread < 1e-6 * ata[0]) {
                for (0..3) |j| {
                    ata[3 * i + j] = 0;
                    ata[3 * j + i] = 0;
                }
                ata[3 * i + i] = 1;
                for (0..3) |k| atb[k][i] = 0;
            }
        }
        const inv = rot.inverse(ata);
        for (0..3) |k| {
            const s = rot.matVec(inv, atb[k]);
            out.misorientation[k] = s[0];
            out.misorientation_rows[k] = s[1];
            out.misorientation_cols[k] = s[2];
        }
    }
    return out;
}

fn sameAxes(a: Frame, b: Frame) bool {
    return std.meta.eql(a.axes, b.axes) and a.active == b.active and a.crystal == b.crystal;
}

/// Up to `most` of each phase's pairs, evenly spread (`xs` grouped by phase, and so is the result).
fn thin(gpa: std.mem.Allocator, xs: []const Pair, most: usize) ![]Pair {
    var out: std.ArrayList(Pair) = .empty;
    errdefer out.deinit(gpa);
    var i: usize = 0;
    while (i < xs.len) {
        var j = i;
        while (j < xs.len and xs[j].phase == xs[i].phase) j += 1;
        const step = @max(1, (j - i + most - 1) / most);
        var k = i;
        while (k < j) : (k += step) try out.append(gpa, xs[k]);
        i = j;
    }
    return out.toOwnedSlice(gpa);
}

/// How far the reference read in `cand` (crystal description number `k`) lies from this
/// engine's orientations: each phase's median angle (radians), averaged over the phases with a
/// few cells or more (`pairs` grouped by phase).
fn score(cand: Frame, pairs: []const Pair, phases: []const PhaseView, crystal_rots: []const Quat, n_crystal: usize, k: usize, buf: []f64) f64 {
    var sum: f64 = 0;
    var count: usize = 0;
    var i: usize = 0;
    while (i < pairs.len) {
        var j = i;
        while (j < pairs.len and pairs[j].phase == pairs[i].phase) j += 1;
        if (j - i >= 5) {
            const v = buf[0 .. j - i];
            for (pairs[i..j], v) |pr, *x| x.* = apart(phases[pr.phase].rotations, pr.q, cand.fromFrame(pr.g, crystal_rots[pr.phase * n_crystal + k]));
            sum += median(v);
            count += 1;
        }
        i = j;
    }
    return if (count == 0) std.math.pi else sum / @as(f64, @floatFromInt(count));
}

// ── tests ──

const testing = std.testing;

fn hexRotations() [12]Quat {
    var out: [12]Quat = undefined;
    for (0..6) |k| {
        const z = rot.so3Exp(.{ 0, 0, @as(f64, @floatFromInt(k)) * std.math.pi / 3 });
        out[k] = z;
        out[6 + k] = rot.qmul(z, rot.so3Exp(.{ std.math.pi, 0, 0 }));
    }
    return out;
}

test "the crystal axes on a hexagonal cell: X || a* is X || a rotated 30° about c" {
    const hex: [6]f64 = .{ 3.232, 3.232, 5.148, 90, 90, 120 };
    const t = crystalRotation(.x_astar_z_c, hex).?;
    const v = rot.qapply(rot.qconj(t), .{ 1, 0, 0 });
    try testing.expectApproxEqAbs(@sqrt(3.0) / 2.0, v[0], 1e-12);
    try testing.expectApproxEqAbs(0.5, v[1], 1e-12);
    // Y || b on a hexagonal cell is X || a* (b ⊥ a*, and c* || c)
    const y = crystalRotation(.y_b_z_cstar, hex).?;
    try testing.expect(apart(&.{rot.quat_identity}, t, y) < 1e-12);
    // on a cube every description is the masters' own
    for (std.enums.values(CrystalAxes)) |c| try testing.expect(apart(&.{rot.quat_identity}, crystalRotation(c, .{ 3.52, 3.52, 3.52, 90, 90, 90 }).?, rot.quat_identity) < 1e-12);
    // and on a monoclinic cell X || a* is X || a rotated by β − 90° about b
    const mono = crystalRotation(.x_astar_z_c, .{ 5.1, 5.2, 5.3, 90, 99.2, 90 }).?;
    try testing.expectApproxEqAbs(std.math.degreesToRadians(9.2), rot.angleBetween(mono, rot.quat_identity), 1e-9);
}

test "a frame's two directions undo each other" {
    const f: Frame = .{ .axes = .{ 2, -1, 3 }, .active = true, .crystal = .x_astar_z_c };
    try testing.expect(f.valid());
    try testing.expect(!(Frame{ .axes = .{ 2, 1, 3 } }).valid());
    const c = f.crystalOf(.{ 3.2, 3.2, 5.1, 90, 90, 120 }).?;
    const q = rot.so3Exp(.{ 0.3, -1.1, 2.0 });
    // (acos near 1: a dot product one rounding below it reads as 1e-8)
    try testing.expect(rot.angleBetween(f.fromFrame(f.toFrame(q, c), c), q) < 1e-7);
}

test "identify finds a planted convention, and measures the misorientation left and how it changes over the map" {
    const gpa = testing.allocator;
    const hex: [6]f64 = .{ 3.232, 3.232, 5.148, 90, 90, 120 };
    const sym = hexRotations();
    const phases = [_]PhaseView{.{ .rotations = &sym, .lattice = hex }};
    var prng = std.Random.DefaultPrng.init(7);
    const r = prng.random();
    for ([_]bool{ false, true }) |active| {
        const planted: Frame = .{ .axes = .{ 2, -1, 3 }, .active = active, .crystal = .x_astar_z_c };
        const c = planted.crystalOf(hex).?;
        // the reference's sample axes rotated from the engine's by m0 at the map's middle,
        // changing by dm from the first row to the last (engine axes), as when a geometry differs
        const m0 = [3]f64{ 0.02, -0.01, 0.005 };
        const dm = [3]f64{ 0, 0.03, 0 };
        const cols = 30;
        const n = 600;
        const q = try gpa.alloc(f32, 4 * n);
        defer gpa.free(q);
        const ph = try gpa.alloc(i32, n);
        defer gpa.free(ph);
        const eu = try gpa.alloc(f64, 3 * n);
        defer gpa.free(eu);
        const lab = try gpa.alloc(i32, n);
        defer gpa.free(lab);
        for (0..n) |i| {
            const qi = rot.so3Exp(.{ 3 * (r.float(f64) - 0.5), 3 * (r.float(f64) - 0.5), 3 * (r.float(f64) - 0.5) });
            for (0..4) |k| q[4 * i + k] = @floatCast(qi[k]);
            ph[i] = 0;
            // the reference's own symmetric equivalent, a little noise, and a tenth of it wrong
            const g = sym[r.intRangeLessThan(usize, 0, sym.len)];
            const noise = rot.so3Exp(.{ 0.003 * r.floatNorm(f64), 0.003 * r.floatNorm(f64), 0.003 * r.floatNorm(f64) });
            const u = @as(f64, @floatFromInt(i / cols)) / @as(f64, @floatFromInt(n / cols - 1)) - 0.5;
            const m = rot.so3Exp(.{ m0[0] + u * dm[0], m0[1] + u * dm[1], m0[2] + u * dm[2] });
            // the engine's orientation as sample axes rotated by m see it: q · m*
            const seen = rot.qmul(rot.qstd(.{ q[4 * i], q[4 * i + 1], q[4 * i + 2], q[4 * i + 3] }), rot.qconj(m));
            var w = planted.toFrame(rot.qmul(noise, rot.qmul(g, seen)), c);
            if (i % 10 == 0) w = rot.so3Exp(.{ r.float(f64), r.float(f64), r.float(f64) });
            eu[3 * i ..][0..3].* = rot.toBunge(w);
            lab[i] = 3;
        }
        const found = try identify(gpa, q, ph, cols, .{ .euler = eu, .phase = lab }, &phases);
        try testing.expectEqual(planted.axes, found.frame.axes);
        try testing.expectEqual(active, found.frame.active);
        try testing.expectEqual(CrystalAxes.x_astar_z_c, found.frame.crystal);
        for (m0, found.misorientation) |a, b| try testing.expectApproxEqAbs(a, b, 1e-3);
        for (dm, found.misorientation_rows) |a, b| try testing.expectApproxEqAbs(a, b, 2e-3);
        for (found.misorientation_cols) |b| try testing.expectApproxEqAbs(0, b, 2e-3);
        try testing.expectEqual(@as(i32, 3), found.matched[0]);
        try testing.expect(found.median_deg < 3 and found.runner_up_deg > 10);
    }
}
