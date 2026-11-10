//! The directions a master pattern is simulated for: the pixels of the Lambert square whose
//! direction lies in the point group's fundamental sector (with a tolerance of two pixels), on
//! the northern hemisphere, and on the southern one too for a crystal without a centre of
//! symmetry. In row order, the row (ii) outer and the column (jj) inner.
const std = @import("std");
const lambert = @import("../lambert.zig");

pub const Grid = struct {
    gpa: std.mem.Allocator,
    halfw: u32,
    /// unit directions in the crystal's Cartesian frame
    khat: [][3]f32,
    /// the pixel each came from: column and row (−halfw … halfw), and the hemisphere (1 north,
    /// −1 south)
    kij: [][3]i32,

    pub fn deinit(self: *Grid) void {
        self.gpa.free(self.khat);
        self.gpa.free(self.kij);
    }

    pub fn init(gpa: std.mem.Allocator, halfw: u32, normals: []const [3]f64, centro: bool) !Grid {
        if (halfw < 2) return error.BadSetting;
        const hw: i32 = @intCast(halfw);
        const eps = 2.0 / @as(f64, @floatFromInt(halfw));
        var khat: std.ArrayList([3]f32) = .empty;
        errdefer khat.deinit(gpa);
        var kij: std.ArrayList([3]i32) = .empty;
        errdefer kij.deinit(gpa);
        for (0..@as(usize, if (centro) 1 else 2)) |hemi| {
            const south = hemi == 1;
            var ii = -hw;
            while (ii <= hw) : (ii += 1) {
                var jj = -hw;
                while (jj <= hw) : (jj += 1) {
                    const x = @as(f64, @floatFromInt(ii)) / @as(f64, @floatFromInt(hw));
                    const y = @as(f64, @floatFromInt(jj)) / @as(f64, @floatFromInt(hw));
                    const d = lambert.squareToSphere(x, y, south);
                    var inside = true;
                    for (normals) |nrm| inside = inside and d[0] * nrm[0] + d[1] * nrm[1] + d[2] * nrm[2] >= -eps;
                    if (!inside) continue;
                    try khat.append(gpa, .{ @floatCast(d[0]), @floatCast(d[1]), @floatCast(d[2]) });
                    try kij.append(gpa, .{ ii, jj, if (south) -1 else 1 });
                }
            }
        }
        return .{ .gpa = gpa, .halfw = halfw, .khat = try khat.toOwnedSlice(gpa), .kij = try kij.toOwnedSlice(gpa) };
    }
};

test "the sectors' sizes are ebsdsim's" {
    const crystal = @import("crystal.zig");
    const gpa = std.testing.allocator;
    // (ebsdsim's n_k at a half-width of 100: nickel 1914, GaN 7534, a monoclinic 20582, a
    // triclinic 40401; nickel at 250: 11017)
    const cases = [_]struct { sym: []const u8, centro: bool, hw: u32, n: usize }{
        .{ .sym = "m-3m", .centro = true, .hw = 100, .n = 1914 },
        .{ .sym = "m-3m", .centro = true, .hw = 250, .n = 11017 },
        .{ .sym = "6mm", .centro = false, .hw = 100, .n = 7534 },
        .{ .sym = "2/m", .centro = true, .hw = 100, .n = 20582 },
        .{ .sym = "-1", .centro = true, .hw = 100, .n = 40401 },
    };
    for (cases) |c| {
        var g = try Grid.init(gpa, c.hw, crystal.sectorNormals(c.sym), c.centro);
        defer g.deinit();
        try std.testing.expectEqual(c.n, g.khat.len);
    }
}
