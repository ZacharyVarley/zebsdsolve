//! The crystal a master pattern is simulated for, as the simulation needs it.
//!
//! The input is a crystal in its International Tables standard setting (as the CIF reader,
//! cif.zig, gives it): the space group's number, the conventional cell (Å, degrees) and the
//! sites of the asymmetric unit. From that: the space group's operators, from its standard
//! Hall symbol (origin choice 2 for the groups with two; R groups on hexagonal axes); every
//! site's orbit; the cell's metrics in nanometres; and the point group the master pattern has,
//! as Cartesian matrices, with the fundamental sector it is folded to.
const std = @import("std");
const rot = @import("../rot.zig");
const sym = @import("../sym.zig");
const tables = @import("tables.zig");
const group = @import("group.zig");

const Op = group.Op;
const DEN = group.DEN;
/// a site's B factor when it gives none, Å² (a room-temperature value)
pub const default_b_iso = 0.5;

/// A site of the asymmetric unit, as the reader gives it.
pub const Site = struct {
    /// atomic number
    z: u32,
    /// fractional coordinates in the standard setting
    xyz: [3]f64,
    occupancy: f64 = 1,
    /// isotropic B, Å² (≤ 0 or not finite: default_b_iso)
    b_iso: f64 = 0,
};

/// The lattice parameters (a, b, c, α, β, γ) a space group's crystal family leaves free, in
/// the standard setting (b unique for the monoclinic groups, hexagonal axes for the trigonal).
pub fn freeParameters(space_group: u32) [6]bool {
    return switch (space_group) {
        1...2 => .{ true, true, true, true, true, true },
        3...15 => .{ true, true, true, false, true, false },
        16...74 => .{ true, true, true, false, false, false },
        75...194 => .{ true, false, true, false, false, false },
        else => .{ true, false, false, false, false, false },
    };
}

/// `cell` (Å, degrees) with what its family ties put back from the free parameters: b = a
/// (tetragonal, trigonal, hexagonal and cubic), c = a (cubic), the right angles, and γ = 120°
/// (trigonal and hexagonal).
pub fn constrained(space_group: u32, cell: [6]f64) [6]f64 {
    var c = cell;
    switch (space_group) {
        1...2 => {},
        3...15 => {
            c[3] = 90;
            c[5] = 90;
        },
        16...74 => c[3..6].* = .{ 90, 90, 90 },
        75...142 => {
            c[1] = c[0];
            c[3..6].* = .{ 90, 90, 90 };
        },
        143...194 => {
            c[1] = c[0];
            c[3..6].* = .{ 90, 90, 120 };
        },
        else => c = .{ c[0], c[0], c[0], 90, 90, 90 },
    }
    return c;
}

/// Whether reflection (h, k, l) survives the lattice centring.
pub fn allowed(centring: u8, h: i32, k: i32, l: i32) bool {
    return switch (centring) {
        'F' => @mod(h, 2) == @mod(k, 2) and @mod(k, 2) == @mod(l, 2),
        'I' => @mod(h + k + l, 2) == 0,
        'A' => @mod(k + l, 2) == 0,
        'B' => @mod(h + l, 2) == 0,
        'C' => @mod(h + k, 2) == 0,
        'R' => @mod(-h + k + l, 3) == 0,
        else => true,
    };
}

/// A site's orbit under the operators: distinct images wrapped into [0, 1), in a canonical
/// order (by their positions on a grid of `eps`).
fn orbit(gpa: std.mem.Allocator, ops: []const Op, p: [3]f64) ![][3]f64 {
    const eps = 1e-4;
    var imgs: std.ArrayList([3]f64) = .empty;
    defer imgs.deinit(gpa);
    for (ops) |o| {
        var x: [3]f64 = undefined;
        for (0..3) |i| {
            var v: f64 = @as(f64, @floatFromInt(o.w[i])) / DEN;
            for (0..3) |k| v += @as(f64, @floatFromInt(o.W[i * 3 + k])) * p[k];
            if (@abs(v) < eps) v = 0;
            v = v - @floor(v);
            if (@abs(v - 1) < eps) v = 0;
            if (@abs(v) < eps) v = 0;
            x[i] = v;
        }
        try imgs.append(gpa, x);
    }
    const key = struct {
        fn of(x: [3]f64) [3]i64 {
            return .{ @intFromFloat(@round(x[0] / eps)), @intFromFloat(@round(x[1] / eps)), @intFromFloat(@round(x[2] / eps)) };
        }
        fn lt(_: void, a: [3]f64, b: [3]f64) bool {
            const ka = of(a);
            const kb = of(b);
            for (0..3) |i| if (ka[i] != kb[i]) return ka[i] < kb[i];
            for (0..3) |i| if (a[i] != b[i]) return a[i] < b[i];
            return false;
        }
    };
    std.mem.sort([3]f64, imgs.items, {}, key.lt);
    var out: std.ArrayList([3]f64) = .empty;
    errdefer out.deinit(gpa);
    for (imgs.items, 0..) |x, i| {
        if (i > 0 and std.mem.eql(i64, &key.of(x), &key.of(imgs.items[i - 1]))) continue;
        try out.append(gpa, x);
    }
    return out.toOwnedSlice(gpa);
}

/// The point group's symbol as the master is folded by, its orientation included: a few
/// point groups occur in two orientations that belong to different space groups (321 / 312,
/// 3m1 / 31m, -3m1 / -31m, -42m / -4m2, -6m2 / -62m).
pub fn foldingSymbol(pg: u32, sg: u32) []const u8 {
    return switch (sg) {
        149, 151, 153 => "312",
        157, 159 => "31m",
        162, 163 => "-31m",
        115...120 => "-4m2",
        189, 190 => "-62m",
        else => sym.point_group_names[pg - 1],
    };
}

const s3 = 0.8660254038;
const r2 = 0.7071067812;

/// The fundamental sector's bounding normals for a folding symbol (a direction is in the
/// sector when it is on the positive side of every one), the monoclinic ones for unique axis b,
/// and the second orientations of the point groups that have two.
pub fn sectorNormals(symbol: []const u8) []const [3]f64 {
    const T = struct { name: []const u8, n: []const [3]f64 };
    const all = [_]T{
        .{ .name = "1", .n = &.{} },
        .{ .name = "-1", .n = &.{.{ 0, 0, 1 }} },
        .{ .name = "2", .n = &.{.{ 1, 0, 0 }} },
        .{ .name = "m", .n = &.{.{ 0, 1, 0 }} },
        .{ .name = "2/m", .n = &.{ .{ 0, 0, 1 }, .{ 0, 1, 0 } } },
        .{ .name = "222", .n = &.{ .{ 0, 0, 1 }, .{ 0, 1, 0 }, .{ 0, 1, 0 } } },
        .{ .name = "mm2", .n = &.{ .{ 1, 0, 0 }, .{ 0, 1, 0 } } },
        .{ .name = "mmm", .n = &.{ .{ 0, 0, 1 }, .{ 0, 1, 0 }, .{ 1, 0, 0 } } },
        .{ .name = "4", .n = &.{ .{ 0, 1, 0 }, .{ 1, 0, 0 } } },
        .{ .name = "-4", .n = &.{ .{ 0, 0, 1 }, .{ 0, 1, 0 }, .{ 0, 1, 0 } } },
        .{ .name = "4/m", .n = &.{ .{ 0, 0, 1 }, .{ 0, 1, 0 }, .{ 1, 0, 0 } } },
        .{ .name = "422", .n = &.{ .{ 0, 0, 1 }, .{ 0, 1, 0 }, .{ 1, 0, 0 } } },
        .{ .name = "4mm", .n = &.{ .{ 0, 1, 0 }, .{ r2, -r2, 0 } } },
        .{ .name = "-42m", .n = &.{ .{ 0, 0, 1 }, .{ r2, r2, 0 }, .{ r2, -r2, 0 } } },
        .{ .name = "4/mmm", .n = &.{ .{ 0, 0, 1 }, .{ 0, 1, 0 }, .{ r2, -r2, 0 } } },
        .{ .name = "3", .n = &.{ .{ 0, 1, 0 }, .{ s3, 0.5, 0 } } },
        .{ .name = "-3", .n = &.{ .{ 0, 0, 1 }, .{ 0, 1, 0 }, .{ s3, 0.5, 0 } } },
        .{ .name = "32", .n = &.{ .{ 0, 0, 1 }, .{ 0, 1, 0 }, .{ s3, 0.5, 0 } } },
        .{ .name = "3m", .n = &.{ .{ 0.5, s3, 0 }, .{ 0.5, -s3, 0 } } },
        .{ .name = "-3m", .n = &.{ .{ 0, 0, 1 }, .{ 0.5, s3, 0 }, .{ 0.5, -s3, 0 } } },
        .{ .name = "6", .n = &.{ .{ 0, 1, 0 }, .{ s3, -0.5, 0 } } },
        .{ .name = "-6", .n = &.{ .{ 0, 0, 1 }, .{ 0, 1, 0 }, .{ s3, 0.5, 0 } } },
        .{ .name = "6/m", .n = &.{ .{ 0, 0, 1 }, .{ 0, 1, 0 }, .{ s3, -0.5, 0 } } },
        .{ .name = "622", .n = &.{ .{ 0, 0, 1 }, .{ 0, 1, 0 }, .{ s3, -0.5, 0 } } },
        .{ .name = "6mm", .n = &.{ .{ 0, 1, 0 }, .{ 0.5, -s3, 0 } } },
        .{ .name = "-6m2", .n = &.{ .{ 0, 0, 1 }, .{ 0.5, s3, 0 }, .{ 0.5, -s3, 0 } } },
        .{ .name = "6/mmm", .n = &.{ .{ 0, 0, 1 }, .{ 0, 1, 0 }, .{ 0.5, -s3, 0 } } },
        .{ .name = "23", .n = &.{ .{ 1, 1, 0 }, .{ 1, -1, 0 }, .{ 0, -1, 1 }, .{ 0, 1, 1 } } },
        .{ .name = "m-3", .n = &.{ .{ 1, 0, 0 }, .{ 0, -1, 1 }, .{ -1, 0, 1 }, .{ 0, 1, 0 }, .{ 0, 0, 1 } } },
        .{ .name = "432", .n = &.{ .{ 1, 0, 0 }, .{ 0, -1, 1 }, .{ -1, 0, 1 }, .{ 0, 1, 0 }, .{ 0, 0, 1 } } },
        .{ .name = "-43m", .n = &.{ .{ 1, -1, 0 }, .{ 1, 1, 0 }, .{ -1, 0, 1 } } },
        .{ .name = "m-3m", .n = &.{ .{ 1, -1, 0 }, .{ -1, 0, 1 }, .{ 0, 1, 0 } } },
        .{ .name = "312", .n = &.{ .{ 0, 0, 1 }, .{ -0.5, s3, 0 }, .{ 0.5, s3, 0 } } },
        .{ .name = "31m", .n = &.{ .{ 0, 1, 0 }, .{ s3, -0.5, 0 } } },
        .{ .name = "-31m", .n = &.{ .{ 0, 0, 1 }, .{ 0, 1, 0 }, .{ s3, -0.5, 0 } } },
        .{ .name = "-4m2", .n = &.{ .{ 0, 0, 1 }, .{ 0, 1, 0 }, .{ 1, 0, 0 } } },
        .{ .name = "-62m", .n = &.{ .{ 0, 0, 1 }, .{ 0, 1, 0 }, .{ s3, -0.5, 0 } } },
    };
    for (all) |t| if (std.mem.eql(u8, t.name, symbol)) return t.n;
    return &.{};
}

pub const centrosymmetric = [_]u32{ 2, 5, 8, 11, 15, 17, 20, 23, 27, 29, 32 };

/// One site's share of the cell, as the simulation uses it.
pub const SiteCell = struct {
    z: u32,
    /// the site's position in the asymmetric unit, fractional
    xyz: [3]f64,
    occupancy: f64,
    /// isotropic B, nm²
    b_nm2: f64,
    /// the orbit, fractional coordinates
    positions: [][3]f64,
};

pub const Cell = struct {
    gpa: std.mem.Allocator,
    space_group: u32,
    point_group: u32,
    /// lattice parameters, nm and radians
    a: f64,
    b: f64,
    c: f64,
    alpha: f64,
    beta: f64,
    gamma: f64,
    /// nm³
    volume: f64,
    /// metric tensors (row-major): the direct one, and its inverse (reciprocal)
    direct: [9]f64,
    reciprocal: [9]f64,
    /// the direct structure matrix: columns are the lattice vectors in Cartesian (a along x)
    dsm: [9]f64,
    centring: u8,
    sites: []SiteCell,
    /// g/cm³, and the averages the energy model was trained on (Σ Z·occupancy·multiplicity
    /// over the number of sites times Σ multiplicity, and the same of the atomic weights)
    density: f64,
    average_z: f64,
    average_a: f64,
    /// the point group's operators in Cartesian (proper and improper), the folding symbol and
    /// the fundamental sector's normals
    pg_ops: []rot.Mat3,
    symbol: []const u8,
    normals: []const [3]f64,
    centro: bool,
    /// how the crystal was read (cif.zig's account, JSON; owned), or empty
    reading: []const u8 = "",

    pub fn deinit(self: *Cell) void {
        for (self.sites) |s| self.gpa.free(s.positions);
        self.gpa.free(self.sites);
        self.gpa.free(self.pg_ops);
        self.gpa.free(self.reading);
    }

    /// The crystal as JSON: its group, cell (Å, degrees) and which of its parameters are free,
    /// sites (B in Å²), the figures the energy model takes, and how it was read ("reading", or
    /// null).
    pub fn json(self: *const Cell, gpa: std.mem.Allocator) ![]u8 {
        const S = struct { symbol: []const u8, z: u32, xyz: [3]f64, occupancy: f64, b_iso: f64, multiplicity: usize };
        const sites = try gpa.alloc(S, self.sites.len);
        defer gpa.free(sites);
        for (self.sites, sites) |s, *o| o.* = .{ .symbol = tables.symbols[s.z - 1], .z = s.z, .xyz = s.xyz, .occupancy = s.occupancy, .b_iso = s.b_nm2 * 100, .multiplicity = s.positions.len };
        const deg = 180.0 / std.math.pi;
        // the short name, the cubic groups with a centre written as now ("F m -3 m", not "F m 3 m")
        var name_buf: [24]u8 = undefined;
        var name = tables.sg_names[self.space_group - 1];
        if ((self.space_group >= 200 and self.space_group <= 206) or self.space_group >= 221) {
            if (std.mem.indexOf(u8, name, " 3")) |at| name = std.fmt.bufPrint(&name_buf, "{s} -3{s}", .{ name[0..at], name[at + 2 ..] }) catch name;
        }
        var out: std.Io.Writer.Allocating = .init(gpa);
        errdefer out.deinit();
        try std.json.Stringify.value(.{
            .space_group = self.space_group,
            .name = name,
            .point_group = self.symbol,
            .centring = &[1]u8{self.centring},
            .centrosymmetric = self.centro,
            // (to 1e-9: what nm and radians leave of a value as given, 3.2000000000000006)
            .cell = [6]f64{ tidy(self.a * 10), tidy(self.b * 10), tidy(self.c * 10), tidy(self.alpha * deg), tidy(self.beta * deg), tidy(self.gamma * deg) },
            .free = freeParameters(self.space_group),
            .volume_nm3 = self.volume,
            .density = self.density,
            .average_z = self.average_z,
            .average_a = self.average_a,
            .sites = sites,
        }, .{}, &out.writer);
        // the reader's account, as it is (JSON already)
        out.writer.undo(1);
        try out.writer.writeAll(",\"reading\":");
        try out.writer.writeAll(if (self.reading.len > 0) self.reading else "null");
        try out.writer.writeByte('}');
        return out.toOwnedSlice();
    }

    /// `cell`: a, b, c (Å), α, β, γ (degrees).
    pub fn init(gpa: std.mem.Allocator, space_group: u32, cell: [6]f64, sites: []const Site) !Cell {
        if (space_group < 1 or space_group > 230) return error.BadSpaceGroup;
        if (sites.len == 0) return error.NoSites;
        for (cell[0..3]) |v| if (!(v > 0)) return error.BadCell;
        const ops_list = try group.hallOps(tables.std_hall[space_group - 1]);
        const ops = ops_list.slice();
        const deg = std.math.pi / 180.0;
        const a = cell[0] * 0.1;
        const b = cell[1] * 0.1;
        const c = cell[2] * 0.1;
        const al = cell[3] * deg;
        const be = cell[4] * deg;
        const ga = cell[5] * deg;
        const ca = @cos(al);
        const cb = @cos(be);
        const cg = @cos(ga);
        const volume = a * b * c * @sqrt(@max(1 + 2 * ca * cb * cg - ca * ca - cb * cb - cg * cg, 0));
        if (!(volume > 0)) return error.BadCell;
        const direct: [9]f64 = .{ a * a, a * b * cg, a * c * cb, a * b * cg, b * b, b * c * ca, a * c * cb, b * c * ca, c * c };
        const sg_ = @sin(ga);
        const dsm: [9]f64 = .{ a, b * cg, c * cb, 0, b * sg_, -c * (cb * cg - ca) / sg_, 0, 0, volume / (a * b * sg_) };
        const reciprocal = inverse(direct) orelse return error.BadCell;

        const out = try gpa.alloc(SiteCell, sites.len);
        var made: usize = 0;
        errdefer {
            for (out[0..made]) |s| gpa.free(s.positions);
            gpa.free(out);
        }
        var m_sum: f64 = 0;
        var z_num: f64 = 0;
        var a_num: f64 = 0;
        for (sites, 0..) |s, i| {
            if (s.z < 1 or s.z > tables.n_elements) return error.UnknownElement;
            const b_iso = if (std.math.isFinite(s.b_iso) and s.b_iso > 0) s.b_iso else default_b_iso;
            out[i] = .{ .z = s.z, .xyz = s.xyz, .occupancy = s.occupancy, .b_nm2 = b_iso * 0.01, .positions = try orbit(gpa, ops, s.xyz) };
            made += 1;
            const mult: f64 = @floatFromInt(out[i].positions.len);
            const aw = tables.atomic_weights[s.z - 1];
            m_sum += mult;
            z_num += @as(f64, @floatFromInt(s.z)) * s.occupancy * mult;
            a_num += aw * s.occupancy * mult;
        }
        const n_sites: f64 = @floatFromInt(sites.len);
        const pg = sym.pointGroupOfSpaceGroup(space_group) orelse return error.BadSpaceGroup;
        const symbol = foldingSymbol(pg, space_group);
        const pg_ops = try cartesianPointGroup(gpa, ops, dsm);
        var centro = false;
        for (centrosymmetric) |p| centro = centro or p == pg;
        return .{
            .gpa = gpa,
            .space_group = space_group,
            .point_group = pg,
            .a = a,
            .b = b,
            .c = c,
            .alpha = al,
            .beta = be,
            .gamma = ga,
            .volume = volume,
            .direct = direct,
            .reciprocal = reciprocal,
            .dsm = dsm,
            .centring = group.latticeLetter(ops) orelse 'P',
            .sites = out,
            .density = a_num / (volume * 6.02214076e2),
            .average_z = if (m_sum > 0) z_num / (n_sites * m_sum) else 0,
            .average_a = if (m_sum > 0) a_num / (n_sites * m_sum) else 0,
            .pg_ops = pg_ops,
            .symbol = symbol,
            .normals = sectorNormals(symbol),
            .centro = centro,
        };
    }
};

fn tidy(v: f64) f64 {
    return @round(v * 1e9) / 1e9;
}

fn inverse(m: [9]f64) ?[9]f64 {
    const det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
    if (!(@abs(det) > 1e-30)) return null;
    const d = 1 / det;
    return .{
        (m[4] * m[8] - m[5] * m[7]) * d, (m[2] * m[7] - m[1] * m[8]) * d, (m[1] * m[5] - m[2] * m[4]) * d,
        (m[5] * m[6] - m[3] * m[8]) * d, (m[0] * m[8] - m[2] * m[6]) * d, (m[2] * m[3] - m[0] * m[5]) * d,
        (m[3] * m[7] - m[4] * m[6]) * d, (m[1] * m[6] - m[0] * m[7]) * d, (m[0] * m[4] - m[1] * m[3]) * d,
    };
}

/// The crystal's point group in Cartesian: the space group's rotation parts conjugated by the
/// direct structure matrix (M W M⁻¹), each once.
fn cartesianPointGroup(gpa: std.mem.Allocator, ops: []const Op, dsm: [9]f64) ![]rot.Mat3 {
    const inv = inverse(dsm) orelse return error.BadCell;
    var out: std.ArrayList(rot.Mat3) = .empty;
    errdefer out.deinit(gpa);
    for (ops) |o| {
        var W: [9]f64 = undefined;
        for (0..9) |i| W[i] = @floatFromInt(o.W[i]);
        const R = rot.matMul(dsm, rot.matMul(W, inv));
        var seen = false;
        for (out.items) |q| {
            var d: f64 = 0;
            for (0..9) |i| d += @abs(q[i] - R[i]);
            if (d < 1e-4) seen = true;
        }
        if (!seen) try out.append(gpa, R);
    }
    return out.toOwnedSlice(gpa);
}

test "a cell's free parameters and what its family ties" {
    try std.testing.expectEqual([6]f64{ 3.2, 3.2, 3.2, 90, 90, 90 }, constrained(225, .{ 3.2, 3.0, 4.0, 80, 90, 100 }));
    try std.testing.expectEqual([6]f64{ 3.2, 3.2, 5.2, 90, 90, 120 }, constrained(186, .{ 3.2, 3.0, 5.2, 90, 90, 90 }));
    try std.testing.expectEqual([6]f64{ 5, 6, 7, 90, 101, 90 }, constrained(14, .{ 5, 6, 7, 80, 101, 95 }));
    try std.testing.expectEqual([6]bool{ true, false, true, false, false, false }, freeParameters(166));
}

test "a cell: nickel's orbit, metrics and point group" {
    const gpa = std.testing.allocator;
    var cell = try Cell.init(gpa, 225, .{ 3.5240, 3.5240, 3.5240, 90, 90, 90 }, &.{.{ .z = 28, .xyz = .{ 0, 0, 0 } }});
    defer cell.deinit();
    try std.testing.expectEqual(@as(usize, 4), cell.sites[0].positions.len);
    try std.testing.expectEqual(@as(u32, 32), cell.point_group);
    try std.testing.expectEqual(@as(usize, 48), cell.pg_ops.len);
    try std.testing.expectEqualStrings("m-3m", cell.symbol);
    try std.testing.expectApproxEqRel(@as(f64, 0.3524 * 0.3524 * 0.3524), cell.volume, 1e-12);
    // 4 × 58.6934 g/mol in 43.76 Å³
    try std.testing.expectApproxEqRel(@as(f64, 4 * 58.6934 / (0.3524 * 0.3524 * 0.3524 * 602.214076)), cell.density, 1e-12);
    try std.testing.expect(cell.centro);
    try std.testing.expect(allowed('F', 1, 1, 1) and !allowed('F', 1, 0, 0));
}
