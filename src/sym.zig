//! Crystal symmetry on the host: the 11 Laue rotation groups (orientation fundamental zones,
//! disorientation), point-group and space-group numbers, and inverse-pole-figure colours.
const std = @import("std");
const rot = @import("rot.zig");

const Quat = rot.Quat;
const Vec3 = rot.Vec3;

/// The eleven proper rotation groups, named by the Laue class each is the rotations of. A
/// value is used in two ways, and they differ for a crystal without a centre of symmetry:
///  * as a Laue class (Master.laue): the point group with the inversion added, which is what a
///    direction's colour in an inverse pole figure follows;
///  * as the rotations of the point group itself (Master.rotation, `rotationsOfPointGroup`),
///    which is what makes two orientations the same. For 6mm these are the six about c (.c6),
///    not the twelve of its Laue class 6/mmm (.d6): the two-folds across c that the inversion
///    brings would turn the polar axis over, and that is another orientation.
pub const Laue = enum(u8) {
    /// -1
    c1 = 1,
    /// 2/m (two-fold along z)
    c2 = 2,
    /// mmm
    d2 = 3,
    /// 4/m
    c4 = 4,
    /// 4/mmm
    d4 = 5,
    /// -3
    c3 = 6,
    /// -3m
    d3 = 7,
    /// 6/m
    c6 = 8,
    /// 6/mmm
    d6 = 9,
    /// m-3
    t = 10,
    /// m-3m
    o = 11,

    /// The group's own name, as a rotation group.
    pub fn rotationName(self: Laue) []const u8 {
        return switch (self) {
            .c1 => "1",
            .c2 => "2",
            .d2 => "222",
            .c4 => "4",
            .d4 => "422",
            .c3 => "3",
            .d3 => "32",
            .c6 => "6",
            .d6 => "622",
            .t => "23",
            .o => "432",
        };
    }

    /// The name of the Laue class it is the rotations of.
    pub fn name(self: Laue) []const u8 {
        return switch (self) {
            .c1 => "-1",
            .c2 => "2/m",
            .d2 => "mmm",
            .c4 => "4/m",
            .d4 => "4/mmm",
            .c3 => "-3",
            .d3 => "-3m",
            .c6 => "6/m",
            .d6 => "6/mmm",
            .t => "m-3",
            .o => "m-3m",
        };
    }
};

const r2 = 0.70710678118654752440;
const r3 = 0.86602540378443864676;

const C1 = [_]Quat{.{ 1, 0, 0, 0 }};
const C2 = [_]Quat{ .{ 1, 0, 0, 0 }, .{ 0, 0, 0, 1 } };
const D2 = [_]Quat{ .{ 1, 0, 0, 0 }, .{ 0, 0, 0, 1 }, .{ 0, 1, 0, 0 }, .{ 0, 0, 1, 0 } };
const C4 = [_]Quat{ .{ 1, 0, 0, 0 }, .{ 0, 0, 0, 1 }, .{ r2, 0, 0, r2 }, .{ r2, 0, 0, -r2 } };
const D4 = C4 ++ [_]Quat{ .{ 0, 1, 0, 0 }, .{ 0, 0, 1, 0 }, .{ 0, r2, r2, 0 }, .{ 0, -r2, r2, 0 } };
const C3 = [_]Quat{ .{ 1, 0, 0, 0 }, .{ 0.5, 0, 0, r3 }, .{ 0.5, 0, 0, -r3 } };
const D3 = C3 ++ [_]Quat{ .{ 0, 1, 0, 0 }, .{ 0, -0.5, r3, 0 }, .{ 0, 0.5, r3, 0 } };
const C6 = C3 ++ [_]Quat{ .{ 0, 0, 0, 1 }, .{ r3, 0, 0, 0.5 }, .{ r3, 0, 0, -0.5 } };
const D6 = C6 ++ [_]Quat{ .{ 0, 1, 0, 0 }, .{ 0, -0.5, r3, 0 }, .{ 0, 0.5, r3, 0 }, .{ 0, r3, 0.5, 0 }, .{ 0, -r3, 0.5, 0 }, .{ 0, 0, 1, 0 } };
const T = D2 ++ [_]Quat{
    .{ 0.5, 0.5, -0.5, 0.5 },  .{ 0.5, 0.5, 0.5, -0.5 }, .{ 0.5, 0.5, -0.5, -0.5 }, .{ 0.5, -0.5, -0.5, -0.5 },
    .{ 0.5, -0.5, 0.5, 0.5 },  .{ 0.5, -0.5, 0.5, -0.5 }, .{ 0.5, -0.5, -0.5, 0.5 }, .{ 0.5, 0.5, 0.5, 0.5 },
};
const O = T ++ [_]Quat{
    .{ r2, 0, 0, r2 }, .{ r2, 0, 0, -r2 }, .{ 0, r2, r2, 0 }, .{ 0, -r2, r2, 0 },
    .{ r2, r2, 0, 0 }, .{ r2, -r2, 0, 0 }, .{ r2, 0, r2, 0 }, .{ r2, 0, -r2, 0 },
    .{ 0, r2, 0, r2 }, .{ 0, -r2, 0, r2 }, .{ 0, 0, r2, r2 }, .{ 0, 0, -r2, r2 },
};

/// The group's rotations as quaternions (identity first).
pub fn operators(laue: Laue) []const Quat {
    return switch (laue) {
        .c1 => &C1,
        .c2 => &C2,
        .d2 => &D2,
        .c4 => &C4,
        .d4 => &D4,
        .c3 => &C3,
        .d3 => &D3,
        .c6 => &C6,
        .d6 => &D6,
        .t => &T,
        .o => &O,
    };
}

/// The symmetry-equivalent of `q` with the smallest rotation angle (its fundamental-zone
/// representative): g · q over the group, since q maps sample to crystal.
pub fn toFz(q: Quat, laue: Laue) Quat {
    var best: Quat = rot.qstd(q);
    var best_w: f64 = best[0];
    for (operators(laue)[1..]) |g| {
        const c = rot.qstd(rot.qmul(g, q));
        if (c[0] > best_w + 1e-12) {
            best_w = c[0];
            best = c;
        }
    }
    return best;
}

/// The smallest rotation angle between two orientations of one phase (radians).
pub fn disorientation(a: Quat, b: Quat, laue: Laue) f64 {
    const m = rot.qmul(a, rot.qconj(b));
    var best: Quat = m;
    for (operators(laue)) |g| {
        const c = rot.qmul(g, m);
        if (@abs(c[0]) > @abs(best[0])) best = c;
    }
    // from the vector part: the arc cosine of a scalar part near 1 loses half its digits, and
    // quaternions that arrive as f32 would never come closer than 0.04° to each other
    return 2 * std.math.atan2(@sqrt(best[1] * best[1] + best[2] * best[2] + best[3] * best[3]), @abs(best[0]));
}

/// Laue class of a crystallographic point group, numbered 1–32 in International Tables order.
pub fn laueOfPointGroup(pg: u32) ?Laue {
    return switch (pg) {
        1...2 => .c1,
        3...5 => .c2,
        6...8 => .d2,
        9...11 => .c4,
        12...15 => .d4,
        16...17 => .c3,
        18...20 => .d3,
        21...23 => .c6,
        24...27 => .d6,
        28...29 => .t,
        30...32 => .o,
        else => null,
    };
}

/// The proper rotations in a crystallographic point group (1–32, International Tables order):
/// its pure rotational subgroup, the symmetry of its orientations. In the axes the groups here
/// are written in (the unique axis along z, a two-fold of the dihedral groups along x).
pub fn rotationsOfPointGroup(pg: u32) ?Laue {
    return switch (pg) {
        1, 2, 4 => .c1, // 1, -1, m
        3, 5, 7, 10 => .c2, // 2, 2/m, mm2, -4
        6, 8, 14 => .d2, // 222, mmm, -42m
        9, 11, 13 => .c4, // 4, 4/m, 4mm
        12, 15 => .d4, // 422, 4/mmm
        16, 17, 19, 22 => .c3, // 3, -3, 3m, -6
        18, 20, 26 => .d3, // 32, -3m, -6m2
        21, 23, 25 => .c6, // 6, 6/m, 6mm
        24, 27 => .d6, // 622, 6/mmm
        28, 29, 31 => .t, // 23, m-3, -43m
        30, 32 => .o, // 432, m-3m
        else => null,
    };
}

/// Point group (1–32) of a space group (1–230).
pub fn pointGroupOfSpaceGroup(sg: u32) ?u32 {
    const first = [_]u32{ 1, 2, 3, 6, 10, 16, 25, 47, 75, 81, 83, 89, 99, 111, 123, 143, 147, 149, 156, 162, 168, 174, 175, 177, 183, 187, 191, 195, 200, 207, 215, 221 };
    if (sg < 1 or sg > 230) return null;
    var pg: u32 = 0;
    for (first, 1..) |f, i| {
        if (sg >= f) pg = @intCast(i);
    }
    return pg;
}

pub const point_group_names = [_][]const u8{
    "1",   "-1",  "2",    "m",    "2/m",  "222", "mm2", "mmm",  "4",   "-4",  "4/m",
    "422", "4mm", "-42m", "4/mmm", "3",   "-3",  "32",  "3m",   "-3m", "6",   "-6",
    "6/m", "622", "6mm",  "-6m2", "6/mmm", "23", "m-3", "432",  "-43m", "m-3m",
};

/// Parse a point-group symbol ("m-3m", "6/mmm", "m3m", …) to its number.
pub fn pointGroupFromName(s: []const u8) ?u32 {
    var buf: [16]u8 = undefined;
    var n: usize = 0;
    for (s) |ch| {
        if (ch == ' ' or ch == '_') continue;
        if (n == buf.len) return null;
        buf[n] = ch;
        n += 1;
    }
    const t = buf[0..n];
    for (point_group_names, 1..) |nm, i| if (std.mem.eql(u8, nm, t)) return @intCast(i);
    const alias = [_]struct { []const u8, u32 }{ .{ "m3m", 32 }, .{ "m3", 29 }, .{ "-3m1", 20 }, .{ "-31m", 20 }, .{ "3m1", 19 }, .{ "31m", 19 }, .{ "321", 18 }, .{ "312", 18 }, .{ "-4m2", 14 }, .{ "-62m", 26 } };
    for (alias) |a| if (std.mem.eql(u8, a[0], t)) return a[1];
    return null;
}

// ── inverse pole figure colours ──

/// How inverse pole figure colours are assigned (settings.ipf_colors):
///   classic            the cubic classes' triangle has red, green and blue at [001], [101] and
///                      [111] and mixes them between; the other classes colour the polar angle
///                      from the c axis (red) and the azimuth within the class's wedge (green
///                      to blue)
///   hue                white at the centre of the class's sector, the hue running once round
///                      it at an even rate (red, green and blue at the corners), the colour
///                      saturating linearly toward the border. For the classes whose sector is
///                      bounded by mirrors (mmm, 4/mmm, 6/mmm, m-3m) and those with a mirror
///                      across the axis (2/m, 4/m, 6/m); the others keep the classic colours.
///   perfect            the keys of Nolze and Hielscher (2016), "Orientations – perfectly
///                      colored", for the Laue class: a light centre and the hue round it, the
///                      hue's rate balanced so that red, yellow, green, cyan, blue and magenta
///                      take even shares of the sector (their appendix A), the lightness and
///                      the saturation following their equations 9 and 10 (settings
///                      ipf_lightness, ipf_gray). A class whose sector is not bounded by
///                      mirrors (m-3, -3m, 6/m, 4/m, 2/m) is coloured by the "extended" key:
///                      the half of its sector that is the sector of the class above it has a
///                      light centre, the other half a dark one, and the colours meet without a
///                      jump. For -1 and -3 no key is both free of jumps and one colour per
///                      direction; theirs jump at the equator, as the paper's do.
///   perfect_rotation   the same keys for the class's proper rotations alone (432, 622, 422,
///                      222, 23, 32, 6, 4, 3, 2, 1), which is how the paper colours an
///                      orientation rather than a direction: a direction and its opposite get
///                      different colours, every class has a light and a dark half, and none
///                      has a jump.
/// The two perfect keys are written from the paper's text and equations; they have not been
/// compared with MTEX's output, which may differ in details the paper leaves open (the
/// saturation of equation 10 is taken as a blend toward mid-gray, the one reading under which a
/// value of 1 makes the light and the dark centre the same gray, as the paper says it does).
pub const Colors = enum { classic, hue, perfect, perfect_rotation };

/// What a key shows: the directions that stand for all the others, as a region of the sphere.
pub const Shape = struct {
    kind: Kind = .wedge,
    /// a wedge's azimuths from the crystal's x, radians
    a0: f64 = 0,
    a1: f64 = 0,
    /// a wedge lies in the upper hemisphere, and where `lower` says so in the lower one too
    lower: bool = false,

    pub const Kind = enum {
        /// between [001], [101] and [111]
        triangle,
        /// between [001], [101], [111] and [011]
        quad,
        /// between [001], [111], [100] and [1-11]
        kite,
        /// from the c axis out to the equator between two azimuths
        wedge,
    };

    pub fn contains(self: Shape, d: Vec3) bool {
        const eps = 1e-9;
        switch (self.kind) {
            .triangle => return d[2] >= d[0] - eps and d[0] >= d[1] - eps and d[1] >= -eps,
            .quad => return d[2] >= d[0] - eps and d[2] >= d[1] - eps and d[0] >= -eps and d[1] >= -eps,
            .kite => return d[0] >= @abs(d[1]) - eps and d[2] >= @abs(d[1]) - eps,
            .wedge => {
                if (d[2] < -eps and !self.lower) return false;
                if (d[0] * d[0] + d[1] * d[1] < 1e-18) return true;
                var az = std.math.atan2(d[1], d[0]);
                while (az < self.a0 - eps) az += 2 * std.math.pi;
                return az <= self.a1 + eps;
            },
        }
    }

    /// The region's bounding box in stereographic projection (x0, y0, x1, y1), of its part in
    /// the upper hemisphere (projected from the south pole) or the lower one (from the north
    /// pole, the same x and y); all zero where it has no part.
    pub fn box(self: Shape, lower_half: bool) [4]f64 {
        const t8 = 0.41421356237309504880; // tan(π/8): [101]
        const h = 0.36602540378443864676; // (√3 − 1) / 2: [111]
        switch (self.kind) {
            .triangle => return if (lower_half) .{ 0, 0, 0, 0 } else .{ 0, 0, t8, h },
            .quad => return if (lower_half) .{ 0, 0, 0, 0 } else .{ 0, 0, t8, t8 },
            .kite => return if (lower_half) .{ 0, 0, 0, 0 } else .{ 0, -h, 1, h },
            .wedge => {
                if (lower_half and !self.lower) return .{ 0, 0, 0, 0 };
                var b: [4]f64 = .{ @min(0, @min(@cos(self.a0), @cos(self.a1))), @min(0, @min(@sin(self.a0), @sin(self.a1))), @max(0, @max(@cos(self.a0), @cos(self.a1))), @max(0, @max(@sin(self.a0), @sin(self.a1))) };
                // the arc's extremes along the axes, where it crosses them
                const quarter = std.math.pi / 2.0;
                var k: f64 = @ceil(self.a0 / quarter - 1e-9);
                while (k * quarter <= self.a1 + 1e-9) : (k += 1) {
                    b[0] = @min(b[0], @round(@cos(k * quarter)));
                    b[1] = @min(b[1], @round(@sin(k * quarter)));
                    b[2] = @max(b[2], @round(@cos(k * quarter)));
                    b[3] = @max(b[3], @round(@sin(k * quarter)));
                }
                return b;
            },
        }
    }
};

/// The region a key shows for a Laue class.
pub fn shapeOf(laue: Laue, colors: Colors) Shape {
    const w = struct {
        fn f(a0: f64, a1: f64, lower: bool) Shape {
            return .{ .kind = .wedge, .a0 = a0 * std.math.pi / 180.0, .a1 = a1 * std.math.pi / 180.0, .lower = lower };
        }
    }.f;
    return switch (colors) {
        .classic, .hue => switch (laue) {
            .o, .t => .{ .kind = .triangle },
            .c1 => w(0, 360, false),
            .c2 => w(0, 180, false),
            .d2, .c4 => w(0, 90, false),
            .d4 => w(0, 45, false),
            .c3 => w(0, 120, false),
            // between the mirrors either side of the two-fold axis along x
            .d3 => w(-30, 30, false),
            .c6 => w(0, 60, false),
            .d6 => w(0, 30, false),
        },
        .perfect => switch (laue) {
            .o => .{ .kind = .triangle },
            .t => .{ .kind = .quad },
            .d6 => w(0, 30, false),
            .d4 => w(0, 45, false),
            .d2, .c4 => w(0, 90, false),
            .d3 => w(-30, 30, false),
            .c6 => w(0, 60, false),
            .c2 => w(0, 180, false),
            .c3 => w(0, 120, false),
            .c1 => w(0, 360, false),
        },
        .perfect_rotation => switch (laue) {
            .o => .{ .kind = .quad },
            .t => .{ .kind = .kite },
            .d6 => w(0, 60, false),
            .d4 => w(0, 90, false),
            .d2 => w(0, 180, false),
            .d3 => w(0, 120, false),
            // one axis only: the sector runs from pole to pole
            .c6 => w(0, 60, true),
            .c4 => w(0, 90, true),
            .c2 => w(0, 180, true),
            .c3 => w(30, 150, true),
            .c1 => w(0, 360, true),
        },
    };
}

/// The corners (red, green, blue) of the classes whose sector is a triangle between mirrors.
fn hueTriangle(laue: Laue) ?[3]Vec3 {
    const r13 = 0.57735026918962576451;
    return switch (laue) {
        .o => .{ .{ 0, 0, 1 }, .{ r2, 0, r2 }, .{ r13, r13, r13 } },
        .d2 => .{ .{ 0, 0, 1 }, .{ 1, 0, 0 }, .{ 0, 1, 0 } },
        .d4 => .{ .{ 0, 0, 1 }, .{ 1, 0, 0 }, .{ r2, r2, 0 } },
        .d6 => .{ .{ 0, 0, 1 }, .{ 1, 0, 0 }, .{ r3, 0.5, 0 } },
        else => null,
    };
}

/// A fully saturated colour of hue `h` (turns) and lightness `l`.
fn hsl(h: f64, l: f64) [3]f64 {
    const ch = 1 - @abs(2 * l - 1);
    const hp = 6 * (h - @floor(h));
    const x = ch * (1 - @abs(@mod(hp, 2.0) - 1));
    const c: [3]f64 = if (hp < 1) .{ ch, x, 0 } else if (hp < 2) .{ x, ch, 0 } else if (hp < 3) .{ 0, ch, x } else if (hp < 4) .{ 0, x, ch } else if (hp < 5) .{ x, 0, ch } else .{ ch, 0, x };
    const m = l - ch / 2.0;
    return .{ c[0] + m, c[1] + m, c[2] + m };
}

fn rgb(c: [3]f64) [3]f32 {
    return .{ @floatCast(c[0]), @floatCast(c[1]), @floatCast(c[2]) };
}

/// A region of the sphere bounded by great circles, with what the perfect keys need of it.
const Sector = struct {
    /// the bounding planes' normals, pointing inward
    normals: [3]Vec3 = @splat(.{ 0, 0, 0 }),
    n_normals: usize = 0,
    /// a triangle's corners: the hue's red, green and blue
    corners: [3]Vec3 = @splat(.{ 0, 0, 0 }),
    n_corners: usize = 0,
    center: Vec3 = .{ 0, 0, 1 },
    /// the direction the azimuth about the centre is counted from
    ref: Vec3 = .{ 1, 0, 0 },

    fn triangle(a: Vec3, b: Vec3, c: Vec3) Sector {
        var s: Sector = .{ .n_normals = 3, .n_corners = 3, .corners = .{ rot.normalize(a), rot.normalize(b), rot.normalize(c) } };
        for (0..3) |i| {
            s.normals[i] = rot.normalize(rot.cross(s.corners[i], s.corners[(i + 1) % 3]));
            if (rot.dot(s.normals[i], s.corners[(i + 2) % 3]) < 0) s.normals[i] = rot.scale(s.normals[i], -1);
        }
        s.center = rot.normalize(rot.add(rot.add(s.corners[0], s.corners[1]), s.corners[2]));
        s.ref = s.corners[0];
        return s;
    }

    /// From pole to pole between two azimuths.
    fn lune(a0: f64, a1: f64) Sector {
        const am = 0.5 * (a0 + a1);
        return .{ .n_normals = 2, .normals = .{ .{ -@sin(a0), @cos(a0), 0 }, .{ @sin(a1), -@cos(a1), 0 }, .{ 0, 0, 0 } }, .center = .{ @cos(am), @sin(am), 0 }, .ref = .{ 0, 0, 1 } };
    }

    fn hemisphere() Sector {
        return .{ .n_normals = 1, .normals = .{ .{ 0, 0, 1 }, .{ 0, 0, 0 }, .{ 0, 0, 0 } } };
    }

    /// How far inside `v` is: negative outside.
    fn depth(self: *const Sector, v: Vec3) f64 {
        var d: f64 = std.math.inf(f64);
        for (self.normals[0..self.n_normals]) |n| d = @min(d, rot.dot(n, v));
        return d;
    }

    /// Along the great circle leaving the centre in direction `f` (a unit tangent there): the
    /// angle at which it meets the border.
    fn reach(self: *const Sector, f: Vec3) f64 {
        const n = rot.cross(self.center, f);
        var bound: f64 = std.math.pi;
        for (self.normals[0..self.n_normals]) |nrm| {
            var p = rot.cross(n, nrm);
            const len = rot.norm(p);
            if (len < 1e-12) continue;
            p = rot.scale(p, (if (rot.dot(p, f) < 0) @as(f64, -1) else 1) / len);
            bound = @min(bound, std.math.acos(std.math.clamp(rot.dot(self.center, p), -1.0, 1.0)));
        }
        return bound;
    }
};

/// The colours of one Laue class under one scheme: made once (the perfect keys tabulate their
/// hue), then asked direction by direction.
pub const Key = struct {
    laue: Laue,
    colors: Colors,
    /// equation 9's λ_L: 1 a lightness linear in the distance from the centre, less a wider
    /// light (and dark) centre
    lambda_l: f64 = 0.25,
    /// equation 10's λ_S: 0 white and black centres, more toward gray
    lambda_s: f64 = 0.25,
    shape: Shape,
    // ── the perfect keys' own ──
    /// the sector of the class above (the class's own where that is bounded by mirrors)
    sector: Sector = .{},
    /// what takes a direction into the sector, and whether by an operation of the class itself
    /// (a light centre) or only of the class above (a dark one)
    maps: [48]Map = undefined,
    n_maps: usize = 0,
    fast: Fast = .none,
    e1: Vec3 = .{ 1, 0, 0 },
    e2: Vec3 = .{ 0, 1, 0 },
    /// the azimuths about the centre where a stretch of hue begins (a triangle's corners: red,
    /// green, blue), closed by a whole turn
    seg: [4]f64 = .{ 0, 2 * std.math.pi, 0, 0 },
    n_seg: usize = 1,
    hue: [3][HUE_N + 1]f32 = undefined,

    const HUE_N = 256;
    const Map = struct { m: rot.Mat3, white: bool };
    /// The cubic classes above m-3m's own triangle: sorting the coordinates' sizes is the map.
    const Fast = enum { none, o_laue, t_laue, o_rotation };

    pub fn init(laue: Laue, colors: Colors, lambda_l: f64, lambda_s: f64) Key {
        var k: Key = .{ .laue = laue, .colors = colors, .lambda_l = lambda_l, .lambda_s = lambda_s, .shape = shapeOf(laue, colors) };
        if (colors != .perfect and colors != .perfect_rotation) return k;
        const proper = colors == .perfect_rotation;
        const z: Vec3 = .{ 0, 0, 1 };
        const at = struct {
            fn f(deg: f64) Vec3 {
                const a = deg * std.math.pi / 180.0;
                return .{ @cos(a), @sin(a), 0 };
            }
        }.f;
        const lune = struct {
            fn f(d0: f64, d1: f64) Sector {
                return Sector.lune(d0 * std.math.pi / 180.0, d1 * std.math.pi / 180.0);
            }
        }.f;
        // The class is P (with the inversion) or its rotations alone; the class above, P+, has a
        // sector bounded by mirrors. An operation of P+ is ±g for a rotation g; it is P's own,
        // or not, by which rotations and which signs.
        switch (laue) {
            .o, .d6, .d4, .d2 => {
                k.sector = switch (laue) {
                    .o => Sector.triangle(z, .{ 1, 0, 1 }, .{ 1, 1, 1 }),
                    .d6 => Sector.triangle(z, at(0), at(30)),
                    .d4 => Sector.triangle(z, at(0), at(45)),
                    else => Sector.triangle(z, at(0), at(90)),
                };
                if (laue == .o) k.fast = if (proper) .o_rotation else .o_laue;
                k.add(operators(laue), null, 1, true);
                k.add(operators(laue), null, -1, !proper);
            },
            .t => if (proper) {
                // 23 under -43m, whose mirrors are the {110} planes
                k.sector = Sector.triangle(z, .{ 1, 1, 1 }, .{ 1, -1, 1 });
                k.add(&T, null, 1, true);
                k.add(&O, &T, -1, false);
            } else {
                k.sector = Sector.triangle(z, .{ 1, 0, 1 }, .{ 1, 1, 1 });
                k.fast = .t_laue;
                k.both(&T, null, true);
                k.both(&O, &T, false);
            },
            .d3 => if (proper) {
                // 32 under -62m: mirrors through the two-fold axes, and across c
                k.sector = Sector.triangle(z, at(0), at(60));
                k.add(&D3, null, 1, true);
                k.add(&D6, &D3, -1, false);
            } else {
                k.sector = Sector.triangle(z, at(0), at(30));
                k.both(&D3, null, true);
                k.both(&D6, &D3, false);
            },
            .c6, .c4, .c2 => {
                const above: []const Quat = switch (laue) {
                    .c6 => &D6,
                    .c4 => &D4,
                    else => &D2,
                };
                const half: f64 = switch (laue) {
                    .c6 => 30,
                    .c4 => 45,
                    else => 90,
                };
                if (proper) {
                    // n under nmm: mirrors through the axis
                    k.sector = lune(0, half);
                    k.add(operators(laue), null, 1, true);
                    k.add(above, operators(laue), -1, false);
                } else {
                    k.sector = Sector.triangle(z, at(0), at(half));
                    k.both(operators(laue), null, true);
                    k.both(above, operators(laue), false);
                }
            },
            .c3 => if (proper) {
                // 3 under 3m, whose mirrors stand across the two-fold axes of D3
                k.sector = lune(30, 90);
                k.add(&C3, null, 1, true);
                k.add(&D3, &C3, -1, false);
            } else {
                // -3: sixths of the upper hemisphere, light and dark in turn, as the paper draws it
                k.sector = Sector.triangle(z, at(0), at(60));
                k.both(&C3, null, true);
                var turned: [3]Quat = undefined;
                for (C3, 0..) |g, i| turned[i] = rot.qmul(g, .{ 0, 0, 1, 0 });
                k.both(&turned, null, false);
            },
            .c1 => {
                k.sector = Sector.hemisphere();
                k.add(&C1, null, 1, true);
                // 1 under m (the mirror across z is minus the half turn about it); -1 keeps the
                // upper hemisphere's colours for both
                k.add(if (proper) C2[1..] else &C1, null, -1, !proper);
            },
        }
        // the azimuth about the centre, counted so that a triangle's corners come in order
        const s = &k.sector;
        k.e1 = rot.normalize(rot.sub(s.ref, rot.scale(s.center, rot.dot(s.ref, s.center))));
        k.e2 = rot.cross(s.center, k.e1);
        const two_pi = 2 * std.math.pi;
        if (s.n_corners == 3) {
            if (k.azimuth(s.corners[1]) > k.azimuth(s.corners[2])) k.e2 = rot.scale(k.e2, -1);
            k.seg = .{ 0, k.azimuth(s.corners[1]), k.azimuth(s.corners[2]), two_pi };
            k.n_seg = 3;
        }
        // The hue as a function of the azimuth (the paper's equations 5 to 8): its rate is the
        // distance to the border, so that a colour's share of the sector does not grow where the
        // border is far, times a bump at red, green and blue, which would otherwise spread; and
        // each stretch between two corners takes a third of the hues.
        for (0..k.n_seg) |i| {
            const a0 = k.seg[i];
            const a1 = k.seg[i + 1];
            var acc: [HUE_N + 1]f64 = undefined;
            acc[0] = 0;
            for (1..HUE_N + 1) |m| {
                const x = (@as(f64, @floatFromInt(m)) - 0.5) / HUE_N;
                const rho = a0 + (a1 - a0) * x;
                const f = rot.add(rot.scale(k.e1, @cos(rho)), rot.scale(k.e2, @sin(rho)));
                const along = if (k.n_seg == 3) (two_pi / 3.0) * (@as(f64, @floatFromInt(i)) + x) else rho;
                var bump: f64 = 0.5;
                for ([_]f64{ 0, two_pi / 3.0, -two_pi / 3.0 }) |c| {
                    var d = @mod(along - c + std.math.pi, two_pi) - std.math.pi;
                    d *= d;
                    bump += @exp(-4.0 / 7.0 * d);
                }
                acc[m] = acc[m - 1] + s.reach(f) * bump;
            }
            for (0..HUE_N + 1) |m| k.hue[i][m] = @floatCast((@as(f64, @floatFromInt(i)) + acc[m] / acc[HUE_N]) / @as(f64, @floatFromInt(k.n_seg)));
        }
        return k;
    }

    /// The maps ±g (by `sign`) for the rotations g of `group` that are not in `except`.
    fn add(self: *Key, group: []const Quat, except: ?[]const Quat, sign: f64, white: bool) void {
        outer: for (group) |g| {
            if (except) |ex| for (ex) |x| {
                const a = rot.qstd(g);
                const b = rot.qstd(x);
                var same: f64 = 0;
                var opposite: f64 = 0;
                for (0..4) |i| {
                    same += @abs(a[i] - b[i]);
                    opposite += @abs(a[i] + b[i]);
                }
                if (@min(same, opposite) < 1e-9) continue :outer;
            };
            var m = rot.toMat(g);
            for (&m) |*v| v.* *= sign;
            self.maps[self.n_maps] = .{ .m = m, .white = white };
            self.n_maps += 1;
        }
    }

    fn both(self: *Key, group: []const Quat, except: ?[]const Quat, white: bool) void {
        self.add(group, except, 1, white);
        self.add(group, except, -1, white);
    }

    /// The azimuth of `v` about the sector's centre, in [0, 2π).
    fn azimuth(self: *const Key, v: Vec3) f64 {
        const az = std.math.atan2(rot.dot(v, self.e2), rot.dot(v, self.e1));
        return if (az < 0) az + 2 * std.math.pi else az;
    }

    /// Colour (r, g, b in [0, 1]) of crystal direction `d` (a unit vector).
    pub fn color(self: *const Key, d: Vec3) [3]f32 {
        switch (self.colors) {
            .classic => return classicColor(d, self.laue),
            .hue => return hueColor(d, self.laue) orelse classicColor(d, self.laue),
            .perfect, .perfect_rotation => {},
        }
        const s = &self.sector;
        var v = d;
        var white = true;
        if (self.fast != .none) {
            // m-3m's triangle is z ≥ x ≥ y ≥ 0: the sizes, sorted; the operation that sorts them
            // is a permutation and three signs
            var idx = [3]usize{ 0, 1, 2 };
            const a = [3]f64{ @abs(d[0]), @abs(d[1]), @abs(d[2]) };
            var odd = false;
            inline for (.{ .{ 0, 1 }, .{ 1, 2 }, .{ 0, 1 } }) |p| {
                if (a[idx[p[0]]] > a[idx[p[1]]]) {
                    std.mem.swap(usize, &idx[p[0]], &idx[p[1]]);
                    odd = !odd;
                }
            }
            // ascending sizes go to y, x, z: one more exchange
            v = .{ a[idx[1]], a[idx[0]], a[idx[2]] };
            odd = !odd;
            var negative = false;
            for (d) |c| negative = negative != (c < 0);
            white = switch (self.fast) {
                .o_laue => true,
                .t_laue => !odd,
                else => odd == negative,
            };
        } else {
            var best: f64 = -std.math.inf(f64);
            for (self.maps[0..self.n_maps]) |*map| {
                const c = rot.matVec(map.m, d);
                const depth = s.depth(c);
                if (depth > best) {
                    best = depth;
                    v = c;
                    white = map.white;
                    if (depth >= 0) break;
                }
            }
        }
        const cos_a = std.math.clamp(rot.dot(s.center, v), -1.0, 1.0);
        const ang = std.math.acos(cos_a);
        var r: f64 = 0;
        var hue: f64 = 0;
        if (ang > 1e-9) {
            const f = rot.normalize(rot.sub(v, rot.scale(s.center, cos_a)));
            r = std.math.clamp(ang / s.reach(f), 0.0, 1.0);
            const az = self.azimuth(f);
            var i: usize = 0;
            while (i + 1 < self.n_seg and az >= self.seg[i + 1]) i += 1;
            const x = std.math.clamp((az - self.seg[i]) / (self.seg[i + 1] - self.seg[i]), 0.0, 1.0) * HUE_N;
            const m: usize = @min(@as(usize, @intFromFloat(x)), HUE_N - 1);
            const w = x - @as(f64, @floatFromInt(m));
            hue = (1 - w) * self.hue[i][m] + w * self.hue[i][m + 1];
        }
        // the sphere's colours laid on the sector: its centre the pole (light for an operation
        // of the class, dark otherwise), its border the equator
        const t = if (white) 1 - 0.5 * r else 0.5 * r;
        const sn = @sin(0.5 * std.math.pi * t);
        const l = self.lambda_l * t + (1 - self.lambda_l) * sn * sn;
        const sat = 1 - 2 * self.lambda_s * @abs(l - 0.5);
        const c = hsl(hue, l);
        return rgb(.{ sat * c[0] + (1 - sat) * 0.5, sat * c[1] + (1 - sat) * 0.5, sat * c[2] + (1 - sat) * 0.5 });
    }
};

/// The hue key's colour of crystal direction `d` (Colors.hue), null for a class it does not cover.
fn hueColor(d: Vec3, laue: Laue) ?[3]f32 {
    const two_pi = 2 * std.math.pi;
    if (hueTriangle(laue)) |tri| {
        const s = Sector.triangle(tri[0], tri[1], tri[2]);
        // the equivalent of d (or of −d: the class has the inversion) inside the triangle
        var e = d;
        find: for (operators(laue)) |g| {
            const c = rot.qapply(g, d);
            for ([_]f64{ 1, -1 }) |sgn| {
                const v = rot.scale(c, sgn);
                if (s.depth(v) > -1e-9) {
                    e = v;
                    break :find;
                }
            }
        }
        const cos_a = std.math.clamp(rot.dot(s.center, e), -1.0, 1.0);
        const ang = std.math.acos(cos_a);
        if (ang < 1e-9) return .{ 1, 1, 1 };
        const f = rot.normalize(rot.sub(e, rot.scale(s.center, cos_a)));
        const r = std.math.clamp(ang / s.reach(f), 0.0, 1.0);
        // the azimuth about the centre from the first corner, the corners a third of a turn of hue apart
        const e1 = rot.normalize(rot.sub(tri[0], rot.scale(s.center, rot.dot(tri[0], s.center))));
        const e2 = rot.cross(s.center, e1);
        var az: [3]f64 = undefined;
        for ([_]Vec3{ f, tri[1], tri[2] }, 0..) |v, i| {
            az[i] = std.math.atan2(rot.dot(v, e2), rot.dot(v, e1));
            if (az[i] < 0) az[i] += two_pi;
        }
        if (az[1] > az[2]) for (&az) |*a| {
            a.* = two_pi - a.*;
        };
        const a = az[0];
        const hue = if (a <= az[1]) a / az[1] / 3.0 else if (a <= az[2]) (1 + (a - az[1]) / (az[2] - az[1])) / 3.0 else (2 + (a - az[2]) / (two_pi - az[2])) / 3.0;
        return rgb(hsl(hue, 1 - 0.5 * r));
    }
    // an n-fold axis with a mirror across it: white along the axis, the hue once round each
    // of the n sectors, saturated at the mirror
    const fold: f64 = switch (laue) {
        .c2 => 2,
        .c4 => 4,
        .c6 => 6,
        else => return null,
    };
    const pol = std.math.acos(@min(@abs(d[2]), 1.0));
    return rgb(hsl(fold * std.math.atan2(d[1], d[0]) / two_pi, 1 - 0.5 * @min(pol / (std.math.pi / 2.0), 1.0)));
}

fn classicColor(d: Vec3, laue: Laue) [3]f32 {
    var c: [3]f64 = undefined;
    switch (laue) {
        .o, .t => {
            var a = [3]f64{ @abs(d[0]), @abs(d[1]), @abs(d[2]) };
            std.mem.sort(f64, &a, {}, std.sort.asc(f64));
            c = .{ a[2] - a[1], a[1] - a[0], a[0] * @sqrt(3.0) };
        },
        else => {
            const wedge = shapeOf(laue, .classic);
            // the equivalent in the upper hemisphere whose azimuth is the first at or past the wedge's start
            var az: f64 = std.math.inf(f64);
            var pol: f64 = 0;
            for (operators(laue)) |g| {
                var e = rot.qapply(g, d);
                if (e[2] < 0) e = rot.scale(e, -1);
                var phi = std.math.atan2(e[1], e[0]);
                while (phi < wedge.a0 - 1e-9) phi += 2 * std.math.pi;
                if (phi < az) {
                    az = phi;
                    pol = std.math.acos(@min(@max(e[2], -1.0), 1.0));
                }
            }
            const t = @min(pol / (std.math.pi / 2.0), 1.0);
            const s = @min(@max((az - wedge.a0) / (wedge.a1 - wedge.a0), 0.0), 1.0);
            c = .{ 1 - t, t * (1 - s), t * s };
        },
    }
    const m = @max(c[0], @max(c[1], c[2]));
    if (m <= 0) return .{ 1, 1, 1 };
    // a square-root tone curve keeps the sector's interior from going dark
    return .{ @floatCast(@sqrt(c[0] / m)), @floatCast(@sqrt(c[1] / m)), @floatCast(@sqrt(c[2] / m)) };
}

/// Colour (r, g, b in [0, 1]) of the crystal direction that sample direction `v` maps to under
/// orientation `q`, by `colors` (a Key made for the purpose: make one and keep it for many).
pub fn ipfColor(q: Quat, v: Vec3, laue: Laue, colors: Colors) [3]f32 {
    return Key.init(laue, colors, 0.25, 0.25).color(rot.normalize(rot.qapply(q, v)));
}

// ── tests ──

test "groups are closed and of the right order" {
    const orders = [_]usize{ 1, 2, 4, 4, 8, 3, 6, 6, 12, 12, 24 };
    inline for (std.meta.fields(Laue), 0..) |f, i| {
        const laue: Laue = @enumFromInt(f.value);
        const ops = operators(laue);
        try std.testing.expectEqual(orders[i], ops.len);
        for (ops) |a| for (ops) |b| {
            const c = rot.qstd(rot.qmul(a, b));
            var found = false;
            for (ops) |g| {
                const gs = rot.qstd(g);
                var d: f64 = 0;
                for (0..4) |k| d += @abs(gs[k] - c[k]);
                // w = 0 elements are their own negatives
                var d2: f64 = 0;
                for (0..4) |k| d2 += @abs(gs[k] + c[k]);
                if (@min(d, d2) < 1e-9) found = true;
            }
            try std.testing.expect(found);
        };
    }
}

test "fundamental zone and disorientation" {
    const q = rot.qstd(.{ 0.1, 0.7, -0.3, 0.64 });
    const f = toFz(q, .o);
    // cubic FZ: rotation angle at most 62.8°
    try std.testing.expect(2 * std.math.acos(f[0]) <= 62.81 * std.math.pi / 180.0);
    try std.testing.expectApproxEqAbs(@as(f64, 0), disorientation(q, f, .o), 1e-7);
    const g = operators(.o)[17];
    try std.testing.expectApproxEqAbs(@as(f64, 0), disorientation(rot.qmul(g, q), q, .o), 1e-7);
    const d = rot.so3Exp(.{ 0.01, 0.02, -0.015 });
    try std.testing.expectApproxEqAbs(@sqrt(0.01 * 0.01 + 0.02 * 0.02 + 0.015 * 0.015), disorientation(rot.qmul(d, q), q, .o), 1e-9);
}

test "numbers and names" {
    try std.testing.expectEqual(@as(?u32, 32), pointGroupOfSpaceGroup(225));
    try std.testing.expectEqual(@as(?u32, 27), pointGroupOfSpaceGroup(194));
    try std.testing.expectEqual(@as(?u32, 1), pointGroupOfSpaceGroup(1));
    try std.testing.expectEqual(@as(?Laue, .o), laueOfPointGroup(32));
    // the rotations of a point group: its Laue class's where it has a centre, fewer where not,
    // and always a subgroup of them of index 1 or 2
    try std.testing.expectEqual(@as(?Laue, .c6), rotationsOfPointGroup(25));
    try std.testing.expectEqual(@as(?Laue, .t), rotationsOfPointGroup(31));
    try std.testing.expectEqual(@as(?Laue, .o), rotationsOfPointGroup(32));
    for (1..33) |pg| {
        const r = rotationsOfPointGroup(@intCast(pg)).?;
        const l = laueOfPointGroup(@intCast(pg)).?;
        const centred = switch (pg) {
            2, 5, 8, 11, 15, 17, 20, 23, 27, 29, 32 => true,
            else => false,
        };
        if (centred) try std.testing.expectEqual(l, r);
        const index = operators(l).len / operators(r).len;
        try std.testing.expect(operators(l).len % operators(r).len == 0 and (index == 1 or index == 2));
    }
    try std.testing.expectEqual(@as(?u32, 32), pointGroupFromName("m-3m"));
    try std.testing.expectEqual(@as(?u32, 27), pointGroupFromName("6/mmm"));
}

test "ipf corners" {
    const id = rot.quat_identity;
    const red = ipfColor(id, .{ 0, 0, 1 }, .o, .classic);
    try std.testing.expect(red[0] == 1 and red[1] == 0 and red[2] == 0);
    const blue = ipfColor(id, .{ 1, 1, 1 }, .o, .classic);
    try std.testing.expect(blue[2] == 1 and blue[0] < 1e-6);
    // symmetry-equivalent directions get one colour
    const a = ipfColor(id, .{ 0.2, 0.5, 0.8 }, .d6, .classic);
    const b = ipfColor(operators(.d6)[7], .{ 0.2, 0.5, 0.8 }, .d6, .classic);
    for (0..3) |i| try std.testing.expectApproxEqAbs(a[i], b[i], 1e-5);
}

test "the hue key: corners, centre, symmetry and continuity" {
    const id = rot.quat_identity;
    // red, green and blue at the cubic triangle's corners, white at its centre
    const corners = [_]Vec3{ .{ 0, 0, 1 }, .{ 1, 0, 1 }, .{ 1, 1, 1 } };
    for (corners, 0..) |v, i| {
        const c = ipfColor(id, v, .o, .hue);
        for (0..3) |k| try std.testing.expectApproxEqAbs(@as(f32, if (k == i) 1 else 0), c[k], 1e-5);
    }
    const mid = rot.normalize(rot.add(rot.add(rot.normalize(corners[0]), rot.normalize(corners[1])), rot.normalize(corners[2])));
    const w = ipfColor(id, mid, .o, .hue);
    for (0..3) |k| try std.testing.expectApproxEqAbs(@as(f32, 1), w[k], 1e-5);
    var prng = std.Random.DefaultPrng.init(3);
    const r = prng.random();
    inline for (.{ Laue.o, Laue.d2, Laue.d4, Laue.d6, Laue.c2, Laue.c4, Laue.c6 }) |laue| {
        const ops = operators(laue);
        for (0..300) |_| {
            const v = rot.normalize(.{ r.floatNorm(f64), r.floatNorm(f64), r.floatNorm(f64) });
            const a = ipfColor(id, v, laue, .hue);
            // equivalent directions, and opposite ones, get one colour
            const b = ipfColor(ops[r.uintLessThan(usize, ops.len)], rot.scale(v, -1), laue, .hue);
            for (0..3) |k| try std.testing.expectApproxEqAbs(a[k], b[k], 1e-4);
            // a direction a hair away gets nearly the same colour, wherever it is (the sector's
            // borders included)
            const u = rot.normalize(rot.add(v, .{ 1e-4 * r.floatNorm(f64), 1e-4 * r.floatNorm(f64), 1e-4 * r.floatNorm(f64) }));
            const c = ipfColor(id, u, laue, .hue);
            for (0..3) |k| try std.testing.expect(@abs(a[k] - c[k]) < 0.02);
        }
    }
}

test "the perfect keys: corners, one colour per equivalent, no jumps, and what each key shows" {
    const id = rot.quat_identity;
    // m-3m: red, green and blue at the triangle's corners, a light gray at its centre
    const corners = [_]Vec3{ .{ 0, 0, 1 }, .{ 1, 0, 1 }, .{ 1, 1, 1 } };
    for (corners, 0..) |v, i| {
        const c = ipfColor(id, v, .o, .perfect);
        for (0..3) |k| try std.testing.expectApproxEqAbs(@as(f32, if (k == i) 1 else 0), c[k], 1e-4);
    }
    const mid = rot.normalize(rot.add(rot.add(rot.normalize(corners[0]), rot.normalize(corners[1])), rot.normalize(corners[2])));
    for (ipfColor(id, mid, .o, .perfect)) |c| try std.testing.expectApproxEqAbs(@as(f32, 0.875), c, 1e-4);
    // by its rotations alone a direction and its opposite differ: one centre light, one dark
    const up = ipfColor(id, mid, .o, .perfect_rotation);
    const down = ipfColor(id, rot.scale(mid, -1), .o, .perfect_rotation);
    try std.testing.expectApproxEqAbs(@as(f32, 0.875), up[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.125), down[0], 1e-4);

    var prng = std.Random.DefaultPrng.init(5);
    const r = prng.random();
    inline for (.{ Colors.perfect, Colors.perfect_rotation }) |colors| {
        inline for (std.meta.fields(Laue)) |fl| {
            const laue: Laue = @enumFromInt(fl.value);
            const key = Key.init(laue, colors, 0.25, 0.25);
            const ops = operators(laue);
            // -1 and -3 cannot be coloured by direction without a jump: theirs is at the equator
            const jumps = colors == .perfect and (laue == .c1 or laue == .c3);
            for (0..400) |_| {
                const v = rot.normalize(.{ r.floatNorm(f64), r.floatNorm(f64), r.floatNorm(f64) });
                const a = key.color(v);
                // every equivalent direction gets the colour; the opposite one too where the
                // key is the Laue class's
                const g = ops[r.uintLessThan(usize, ops.len)];
                const sign: f64 = if (colors == .perfect and r.boolean()) -1 else 1;
                const b = key.color(rot.scale(rot.qapply(g, v), sign));
                for (0..3) |k| try std.testing.expectApproxEqAbs(a[k], b[k], 2e-3);
                // a direction a hair away gets nearly the same colour
                if (!(jumps and @abs(v[2]) < 0.01)) {
                    const u = rot.normalize(rot.add(v, .{ 1e-4 * r.floatNorm(f64), 1e-4 * r.floatNorm(f64), 1e-4 * r.floatNorm(f64) }));
                    const c = key.color(u);
                    for (0..3) |k| try std.testing.expect(@abs(a[k] - c[k]) < 0.03);
                }
                // exactly one of the equivalents lies in what the key shows
                var inside: usize = 0;
                for (ops) |o| {
                    const e = rot.qapply(o, v);
                    if (key.shape.contains(e)) inside += 1;
                    if (colors == .perfect and key.shape.contains(rot.scale(e, -1))) inside += 1;
                }
                try std.testing.expectEqual(@as(usize, 1), inside);
            }
        }
    }
}

test "the classic and hue keys show one equivalent of every direction" {
    var prng = std.Random.DefaultPrng.init(9);
    const r = prng.random();
    inline for (std.meta.fields(Laue)) |fl| {
        const laue: Laue = @enumFromInt(fl.value);
        // the cubic triangle is m-3m's: m-3 shares it, two of its directions to a colour
        if (laue == .t) continue;
        const key = Key.init(laue, .classic, 0.25, 0.25);
        for (0..200) |_| {
            const v = rot.normalize(.{ r.floatNorm(f64), r.floatNorm(f64), r.floatNorm(f64) });
            var inside: usize = 0;
            for (operators(laue)) |o| {
                const e = rot.qapply(o, v);
                if (key.shape.contains(e)) inside += 1;
                if (key.shape.contains(rot.scale(e, -1))) inside += 1;
            }
            try std.testing.expectEqual(@as(usize, 1), inside);
            // and the colour is the same for every one of them
            const a = key.color(v);
            const b = key.color(rot.scale(rot.qapply(operators(laue)[r.uintLessThan(usize, operators(laue).len)], v), -1));
            for (0..3) |k| try std.testing.expectApproxEqAbs(a[k], b[k], 1e-4);
        }
    }
}
