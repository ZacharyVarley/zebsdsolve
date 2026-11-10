//! How a crystal's unit cell is drawn (cellview.zig and cellview.wgsl draw it): every atom of
//! the cell, each site's orbit with its images on the cell's faces, edges and corners, as a
//! ball in its element's colour; the cell's lattice vectors and middle; the radius about the
//! middle that holds it all. No GPU here.
const std = @import("std");
const crystal = @import("crystal.zig");
const tables = @import("tables.zig");

/// The most atoms drawn: a cell with more is refused.
pub const MAX_ATOMS = 1 << 16;

/// One atom as the kernel takes it: its centre (Å, from the middle of the cell) and radius,
/// its colour and its site.
pub const Atom = extern struct {
    p: [4]f32,
    c: [4]f32,
};


/// The balls' radii (Å): a share of Slater's (1964) atomic radii, small enough that the
/// atoms of a dense cell stand apart.
const BALL = 0.42;
const slater = [_]f32{
    0.25, 0.31, 1.45, 1.05, 0.85, 0.70, 0.65, 0.60, 0.50, 0.38, // H … Ne
    1.80, 1.50, 1.25, 1.10, 1.00, 1.00, 1.00, 0.71, // Na … Ar
    2.20, 1.80, 1.60, 1.40, 1.35, 1.40, 1.40, 1.40, 1.35, 1.35, 1.35, 1.35, 1.30, 1.25, 1.15, 1.15, 1.15, 0.88, // K … Kr
    2.35, 2.00, 1.80, 1.55, 1.45, 1.45, 1.35, 1.30, 1.35, 1.40, 1.60, 1.55, 1.55, 1.45, 1.45, 1.40, 1.40, 1.08, // Rb … Xe
    2.60, 2.15, 1.95, 1.85, 1.85, 1.85, 1.85, 1.85, 1.85, 1.80, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, // Cs … Lu
    1.55, 1.45, 1.35, 1.35, 1.30, 1.35, 1.35, 1.35, 1.50, 1.90, 1.80, 1.60, 1.90, 1.27, 1.20, // Hf … Rn
    2.00, 2.15, 1.95, 1.80, 1.80, 1.75, 1.75, 1.75, 1.75, 1.76, 1.70, 1.70, // Fr … Cf
};

/// The elements' colours (Jmol's), 0xRRGGBB.
const colours = [_]u24{
    0xFFFFFF, 0xD9FFFF, 0xCC80FF, 0xC2FF00, 0xFFB5B5, 0x909090, 0x3050F8, 0xFF0D0D, 0x90E050, 0xB3E3F5,
    0xAB5CF2, 0x8AFF00, 0xBFA6A6, 0xF0C8A0, 0xFF8000, 0xFFFF30, 0x1FF01F, 0x80D1E3, 0x8F40D4, 0x3DFF00,
    0xE6E6E6, 0xBFC2C7, 0xA6A6AB, 0x8A99C7, 0x9C7AC7, 0xE06633, 0xF090A0, 0x50D050, 0xC88033, 0x7D80B0,
    0xC28F8F, 0x668F8F, 0xBD80E3, 0xFFA100, 0xA62929, 0x5CB8D1, 0x702EB0, 0x00FF00, 0x94FFFF, 0x94E0E0,
    0x73C2C9, 0x54B5B5, 0x3B9E9E, 0x248F8F, 0x0A7D8C, 0x006985, 0xC0C0C0, 0xFFD98F, 0xA67573, 0x668080,
    0x9E63B5, 0xD47A00, 0x940094, 0x429EB0, 0x57178F, 0x00C900, 0x70D4FF, 0xFFFFC7, 0xD9FFC7, 0xC7FFC7,
    0xA3FFC7, 0x8FFFC7, 0x61FFC7, 0x45FFC7, 0x30FFC7, 0x1FFFC7, 0x00FF9C, 0x00E675, 0x00D452, 0x00BF38,
    0x00AB24, 0x4DC2FF, 0x4DA6FF, 0x2194D6, 0x267DAB, 0x266696, 0x175487, 0xD0D0E0, 0xFFD123, 0xB8B8D0,
    0xA6544D, 0x575961, 0x9E4FB5, 0xAB5C00, 0x754F45, 0x428296, 0x420066, 0x007D00, 0x70ABFA, 0x00BAFF,
    0x00A1FF, 0x008FFF, 0x0080FF, 0x006BFF, 0x545CF2, 0x785CE3, 0x8A4FE3, 0xA136D4,
};

comptime {
    std.debug.assert(slater.len >= tables.n_elements and colours.len >= tables.n_elements);
}

pub fn ballRadius(z: u32) f32 {
    return BALL * slater[@min(z, slater.len) - 1];
}

pub fn colour(z: u32) u24 {
    return colours[@min(z, colours.len) - 1];
}

/// Where the cell lies: its lattice vectors (Å, Cartesian, the columns of the direct
/// structure matrix) and its middle, and the radius about the middle that holds everything drawn.
pub const Frame = struct {
    basis: [3][3]f64,
    centre: [3]f64,
    radius: f64,
    /// cells along each lattice vector (1: the unit cell, 3: a 3 × 3 × 3 supercell)
    n: u32 = 1,
};

fn cartesian(cell: *const crystal.Cell, f: [3]f64) [3]f64 {
    var out: [3]f64 = undefined;
    for (0..3) |i| out[i] = 10 * (cell.dsm[i * 3] * f[0] + cell.dsm[i * 3 + 1] * f[1] + cell.dsm[i * 3 + 2] * f[2]);
    return out;
}

/// Every atom drawn, at its place in the cell (or the n × n × n cells): each site's orbit in
/// every cell, a position at 0 along a lattice vector again at the far end (an atom on the
/// cell's face twice, on an edge four times, at a corner eight; with cells, the shared ones once).
pub fn atomsOf(gpa: std.mem.Allocator, cell: *const crystal.Cell, frame: *const Frame) ![]Atom {
    var out: std.ArrayList(Atom) = .empty;
    errdefer out.deinit(gpa);
    for (cell.sites, 0..) |s, si| {
        const rgb = colour(s.z);
        const col: [3]f32 = .{ @as(f32, @floatFromInt(rgb >> 16)) / 255, @as(f32, @floatFromInt((rgb >> 8) & 0xFF)) / 255, @as(f32, @floatFromInt(rgb & 0xFF)) / 255 };
        for (s.positions) |p| {
            // images along each axis: 1 or 2 (at 0 and at 1)
            var n: [3]u32 = undefined;
            for (0..3) |i| n[i] = frame.n + @intFromBool(@abs(p[i]) < 1e-4);
            for (0..n[0]) |a| for (0..n[1]) |b| for (0..n[2]) |c| {
                if (out.items.len >= MAX_ATOMS) return error.TooManyAtoms;
                const f: [3]f64 = .{ p[0] + @as(f64, @floatFromInt(a)), p[1] + @as(f64, @floatFromInt(b)), p[2] + @as(f64, @floatFromInt(c)) };
                const x = cartesian(cell, f);
                try out.append(gpa, .{
                    .p = .{ @floatCast(x[0] - frame.centre[0]), @floatCast(x[1] - frame.centre[1]), @floatCast(x[2] - frame.centre[2]), ballRadius(s.z) },
                    .c = .{ col[0], col[1], col[2], @floatFromInt(si) },
                });
            };
        }
    }
    return out.toOwnedSlice(gpa);
}

/// What tells one crystal from another for the view: a hash of its cell and its sites.
pub fn fingerprint(cell: *const crystal.Cell, n: u32) u64 {
    var h = std.hash.Wyhash.init(0x5eed + n);
    h.update(std.mem.asBytes(&cell.dsm));
    for (cell.sites) |s| {
        h.update(std.mem.asBytes(&s.z));
        h.update(std.mem.sliceAsBytes(s.positions));
    }
    return h.final();
}

pub fn frameOf(cell: *const crystal.Cell, cells: u32) Frame {
    const nn: f64 = @floatFromInt(cells);
    var fr: Frame = undefined;
    fr.n = cells;
    for (0..3) |j| fr.basis[j] = cartesian(cell, .{ @floatFromInt(@intFromBool(j == 0)), @floatFromInt(@intFromBool(j == 1)), @floatFromInt(@intFromBool(j == 2)) });
    fr.centre = cartesian(cell, .{ 0.5 * nn, 0.5 * nn, 0.5 * nn });
    // the corners, and every ball with its radius
    var r: f64 = 0;
    for (0..8) |k| {
        const x = cartesian(cell, .{ nn * @as(f64, @floatFromInt(k & 1)), nn * @as(f64, @floatFromInt((k >> 1) & 1)), nn * @as(f64, @floatFromInt(k >> 2)) });
        r = @max(r, std.math.sqrt((x[0] - fr.centre[0]) * (x[0] - fr.centre[0]) + (x[1] - fr.centre[1]) * (x[1] - fr.centre[1]) + (x[2] - fr.centre[2]) * (x[2] - fr.centre[2])));
    }
    var big: f32 = 0;
    for (cell.sites) |s| big = @max(big, ballRadius(s.z));
    fr.radius = r + big;
    return fr;
}

/// The cell's frame and its atoms' account for the app, as JSON: the lattice vectors (Å, rows
/// a, b and c in Cartesian, a along x), the middle of the cell and the radius about it that
/// holds the picture (Å; the picture's shorter side is 2 × this at zoom 1), how many atoms are
/// drawn, and each site's element, colour and ball.
pub fn info(gpa: std.mem.Allocator, cell: *const crystal.Cell, cells: u32) ![]u8 {
    const fr = frameOf(cell, cells);
    const S = struct { symbol: []const u8, z: u32, colour: [7]u8, ball: f32 };
    const sites = try gpa.alloc(S, cell.sites.len);
    defer gpa.free(sites);
    var drawn: usize = 0;
    for (cell.sites, sites) |s, *o| {
        o.* = .{ .symbol = tables.symbols[s.z - 1], .z = s.z, .colour = undefined, .ball = ballRadius(s.z) };
        _ = std.fmt.bufPrint(&o.colour, "#{x:0>6}", .{colour(s.z)}) catch unreachable;
        for (s.positions) |p| {
            var n: usize = 1;
            for (0..3) |i| n *= cells + @intFromBool(@abs(p[i]) < 1e-4);
            drawn += n;
        }
    }
    var out: std.Io.Writer.Allocating = .init(gpa);
    errdefer out.deinit();
    try std.json.Stringify.value(.{
        .basis = fr.basis,
        .centre = fr.centre,
        .radius = fr.radius,
        .atoms = drawn,
        .cells = cells,
        .sites = sites,
    }, .{}, &out.writer);
    return out.toOwnedSlice();
}

test "a cell's atoms: nickel's four, with their images, and its frame" {
    const gpa = std.testing.allocator;
    var cell = try crystal.Cell.init(gpa, 225, .{ 3.524, 3.524, 3.524, 90, 90, 90 }, &.{.{ .z = 28, .xyz = .{ 0, 0, 0 } }});
    defer cell.deinit();
    const fr = frameOf(&cell, 1);
    try std.testing.expectApproxEqAbs(@as(f64, 1.762), fr.centre[0], 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 3.524), fr.basis[0][0], 1e-9);
    // a corner (eight images) and three face centres (two each): 14
    const atoms = try atomsOf(gpa, &cell, &fr);
    defer gpa.free(atoms);
    try std.testing.expectEqual(@as(usize, 14), atoms.len);
    const j = try info(gpa, &cell, 1);
    defer gpa.free(j);
    try std.testing.expect(std.mem.indexOf(u8, j, "\"atoms\":14") != null);
    // 3 × 3 × 3 cells of fcc: the 4 × 4 × 4 corner points, and each of the three kinds of face
    // centre at 3 × 3 × 4 places
    const fr3 = frameOf(&cell, 3);
    const atoms3 = try atomsOf(gpa, &cell, &fr3);
    defer gpa.free(atoms3);
    try std.testing.expectEqual(@as(usize, 64 + 3 * 36), atoms3.len);
    try std.testing.expect(std.mem.indexOf(u8, j, "#50d050") != null);
}
