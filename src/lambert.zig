//! The equal-area (Roşca–Lambert) map between a square and a hemisphere that master patterns
//! are stored on, as EMsoft and ebsdsim write them: a side × side image per hemisphere, pixel
//! (row, col) at square coordinates (x, y) = (2 col / (side − 1) − 1, 2 row / (side − 1) − 1).
//! shaders/common.wgsl holds the same two maps for the GPU.
const std = @import("std");
const rot = @import("rot.zig");

/// Square coordinates in [−1, 1]² → unit vector on the northern (or southern) hemisphere.
pub fn squareToSphere(x: f64, y: f64, southern: bool) rot.Vec3 {
    const sc = @sqrt(std.math.pi / 2.0);
    const xa = @abs(x) * sc;
    const ya = @abs(y) * sc;
    const swap = xa >= ya;
    const xn = if (swap) xa else ya;
    const yn = if (swap) ya else xa;
    if (xn == 0) return .{ 0, 0, if (southern) -1 else 1 };
    const r = (2 * xn / std.math.pi) * @sqrt(@max(std.math.pi - xn * xn, 0.0));
    const ang = std.math.pi * yn / (4 * xn);
    const xh = r * @cos(ang);
    const yh = r * @sin(ang);
    const z = 1 - 2 * xn * xn / std.math.pi;
    const hx = (if (swap) xh else yh) * (if (x < 0) @as(f64, -1) else 1);
    const hy = (if (swap) yh else xh) * (if (y < 0) @as(f64, -1) else 1);
    return rot.normalize(.{ hx, hy, if (southern) -z else z });
}

/// Unit vector → square coordinates of its own hemisphere (|z|).
pub fn sphereToSquare(v: rot.Vec3) [2]f64 {
    const kz = @abs(v[2]);
    if (kz >= 1 - 1e-14) return .{ 0, 0 };
    const xn = @sqrt(@max(std.math.pi * (1 - kz) / 2, 0.0));
    const ax = @abs(v[0]);
    const ay = @abs(v[1]);
    const sc = @sqrt(std.math.pi / 2.0);
    var u: f64 = undefined;
    var w: f64 = undefined;
    if (ax >= ay) {
        u = xn;
        w = (4 * xn / std.math.pi) * std.math.atan2(ay, @max(ax, 1e-300));
    } else {
        u = (4 * xn / std.math.pi) * std.math.atan2(ax, @max(ay, 1e-300));
        w = xn;
    }
    return .{ (u / sc) * (if (v[0] < 0) @as(f64, -1) else 1), (w / sc) * (if (v[1] < 0) @as(f64, -1) else 1) };
}

/// Bilinear sample of a side × side sheet at square coordinates (x, y).
pub fn sample(sheet: []const f32, side: u32, x: f64, y: f64) f64 {
    const last: f64 = @floatFromInt(side - 1);
    const fj = @min(@max((x + 1) * 0.5 * last, 0.0), last);
    const fi = @min(@max((y + 1) * 0.5 * last, 0.0), last);
    const c0: usize = @intFromFloat(@floor(fj));
    const r0: usize = @intFromFloat(@floor(fi));
    const c1 = @min(c0 + 1, side - 1);
    const r1 = @min(r0 + 1, side - 1);
    const tj = fj - @as(f64, @floatFromInt(c0));
    const ti = fi - @as(f64, @floatFromInt(r0));
    const s: usize = side;
    const a: f64 = sheet[r0 * s + c0];
    const b: f64 = sheet[r0 * s + c1];
    const c: f64 = sheet[r1 * s + c0];
    const d: f64 = sheet[r1 * s + c1];
    return (a * (1 - tj) + b * tj) * (1 - ti) + (c * (1 - tj) + d * tj) * ti;
}

test "square and sphere invert each other, with equal area" {
    const pts = [_][2]f64{ .{ 0.3, -0.7 }, .{ -0.9, 0.2 }, .{ 0.0, 0.5 }, .{ 1.0, 1.0 }, .{ -0.01, 0.01 } };
    for (pts) |p| {
        for ([_]bool{ false, true }) |south| {
            const v = squareToSphere(p[0], p[1], south);
            try std.testing.expectApproxEqAbs(@as(f64, 1), rot.norm(v), 1e-13);
            try std.testing.expect((v[2] <= 1e-12) == south or @abs(v[2]) < 1e-12);
            const q = sphereToSquare(v);
            try std.testing.expectApproxEqAbs(p[0], q[0], 1e-10);
            try std.testing.expectApproxEqAbs(p[1], q[1], 1e-10);
        }
    }
    // the square's area 4 maps to the hemisphere's 2π: |∂v/∂x × ∂v/∂y| = π/2 everywhere
    const h = 1e-6;
    for (pts[0..3]) |p| {
        const dx = rot.scale(rot.sub(squareToSphere(p[0] + h, p[1], false), squareToSphere(p[0] - h, p[1], false)), 1.0 / (2.0 * h));
        const dy = rot.scale(rot.sub(squareToSphere(p[0], p[1] + h, false), squareToSphere(p[0], p[1] - h, false)), 1.0 / (2.0 * h));
        try std.testing.expectApproxEqAbs(std.math.pi / 2.0, rot.norm(rot.cross(dx, dy)), 1e-6);
    }
}
