//! Where the detector is, and where each pattern's source point was.
//!
//! The sample frame has its origin at the source point of the map's reference cell (the centre
//! of the map), x down the tilted surface toward the detector, z along the surface normal. The
//! detector's pose maps detector coordinates (row, col, 0), in pixels, into it:
//!     x = R (row, col, 0) + t,        R = R0(sample tilt, detector tilt) · Exp(rot).
//! The pattern centre (Bruker convention: x from the left and y from the top in fractions of
//! the pattern's width and height, z the detector distance in fractions of its height) is the
//! foot of the perpendicular from the source onto the detector, so with R it gives t:
//!     t = R (0.5 − pcy · rows, 0.5 − pcx · cols, pcz · rows).
//! A pattern recorded with the beam at s (sample frame) sees the detector at t − s. That is
//! all the geometry there is: one pose, and for each pattern the point the beam stood on, which
//! the back-projection and the detector kernels are handed with it. No pattern has a pattern
//! centre of its own; the pattern centre is only how the one pose's position is written.
//!
//! Pixels here are those of the pattern being worked on: the stored pattern reduced by some
//! factor (reduce.zig). `rows` and `cols` are then the stored size over that factor, which need
//! not be whole numbers (a reduction may drop part of the last row or column; the pattern
//! centre stays where it is on the detector).
//!
//! All of this is the engine's state rather than an option: the scan file fills what it knows,
//! the caller may set any of it, and the fits (pcfit.zig, posefit.zig) refine pc and rot.
const std = @import("std");
const rot = @import("rot.zig");

pub const Geometry = extern struct {
    /// pattern centre at the reference cell, Bruker convention
    pc: [3]f64 = .{ 0.5, 0.5, 0.5 },
    sample_tilt_deg: f64 = 70,
    detector_tilt_deg: f64 = 0,
    /// rotation of the detector away from its nominal attitude (rotation vector, radians, in
    /// the detector's own axes): about its row direction (down the pattern: swing), about its
    /// column direction (across the pattern: tilt, in the detector tilt's sense) and about its
    /// normal (twist). It turns the detector about the source point, so the pattern centre
    /// stays where it is.
    rot: [3]f64 = .{ 0, 0, 0 },
    /// size of one stored pattern pixel on the detector, micrometres (0: unknown: the beam's
    /// travel cannot then be put in pixels, and every pattern is taken to come from one point)
    px_um: f64 = 0,
    step_x_um: f64 = 1,
    step_y_um: f64 = 1,

    /// The nominal attitude R0: detector tilted by `detector_tilt` facing a sample tilted by
    /// `sample_tilt` (the Bruker construction, as kikuchipy and EMsoft use it).
    pub fn nominal(self: *const Geometry) rot.Mat3 {
        const a = std.math.degreesToRadians(self.detector_tilt_deg - self.sample_tilt_deg);
        const s = @sin(a);
        const c = @cos(a);
        return .{ -s, 0, c, 0, 1, 0, -c, 0, -s };
    }

    pub fn rotation(self: *const Geometry) rot.Mat3 {
        return rot.matMul(self.nominal(), rot.toMat(rot.so3Exp(self.rot)));
    }

    /// t in pixels of a pattern of rows × cols pixels (the stored size over the reduction).
    pub fn translation(self: *const Geometry, rows: f64, cols: f64) rot.Vec3 {
        return rot.matVec(self.rotation(), .{ 0.5 - self.pc[1] * rows, 0.5 - self.pc[0] * cols, self.pc[2] * rows });
    }

    /// The pattern centre a pose (R, t in pixels) implies.
    pub fn pcOf(R: rot.Mat3, t: rot.Vec3, rows: f64, cols: f64) [3]f64 {
        const p = rot.matVec(rot.transpose(R), t);
        return .{ (0.5 - p[1]) / cols, (0.5 - p[0]) / rows, p[2] / rows };
    }

    /// Angular width of one stored pixel at the pattern centre of a pattern `rows` stored
    /// pixels high, radians (the small-angle width: 1 / the detector distance in pixels).
    pub fn pixelAngle(self: *const Geometry, rows: f64) f64 {
        return 1.0 / @max(self.pc[2] * rows, 1e-9);
    }

    /// Beam position of map cell (row, col) relative to the reference cell, in pixels of a
    /// pattern reduced by `bin` (sample frame); zero when the pixel size is unknown.
    pub fn scanOffset(self: *const Geometry, row: f64, col: f64, ref_row: f64, ref_col: f64, x_sign: i32, y_sign: i32, bin: f64) rot.Vec3 {
        if (!(self.px_um > 0)) return .{ 0, 0, 0 };
        const k = 1.0 / (self.px_um * bin);
        return .{
            @as(f64, @floatFromInt(y_sign)) * (row - ref_row) * self.step_y_um * k,
            @as(f64, @floatFromInt(x_sign)) * (col - ref_col) * self.step_x_um * k,
            0,
        };
    }
};

test "pattern centre and pose are two views of one thing" {
    var g: Geometry = .{ .pc = .{ 0.42, 0.21, 0.5 }, .rot = .{ 0.01, -0.02, 0.005 } };
    const R = g.rotation();
    const t = g.translation(60, 80);
    const pc = Geometry.pcOf(R, t, 60, 80);
    for (0..3) |i| try std.testing.expectApproxEqAbs(g.pc[i], pc[i], 1e-13);
    // the source (origin) projects onto the detector at the pattern centre, at distance pcz·rows
    const n: rot.Vec3 = .{ R[2], R[5], R[8] };
    try std.testing.expectApproxEqAbs(0.5 * 60.0, rot.dot(n, t), 1e-10);
    // a turn about the column direction is the detector tilt
    var h: Geometry = .{ .rot = .{ 0, std.math.degreesToRadians(3.0), 0 } };
    const tilted: Geometry = .{ .detector_tilt_deg = 3 };
    for (h.rotation(), tilted.rotation()) |a, b| try std.testing.expectApproxEqAbs(a, b, 1e-12);
    h.rot = .{ 0, 0, 0 };
    // nominal attitude, no extra rotation: the Bruker closed form
    g.rot = .{ 0, 0, 0 };
    const t0 = g.translation(60, 60);
    const a = std.math.degreesToRadians(-70.0);
    try std.testing.expectApproxEqAbs(0.21 * 60 * @sin(a) + 0.5 * 60 * @cos(a) - @sin(a) / 2.0, t0[0], 1e-10);
    try std.testing.expectApproxEqAbs(-60.0 * 0.42 + 0.5, t0[1], 1e-10);
    try std.testing.expectApproxEqAbs(0.21 * 60 * @cos(a) - 0.5 * 60 * @sin(a) - @cos(a) / 2.0, t0[2], 1e-10);
    // a pattern reduced by 2.5 from 61 stored rows keeps its centre on the detector: the same
    // fractions of the stored size, 24.4 reduced pixels high
    const tr = g.translation(61.0 / 2.5, 61.0 / 2.5);
    const back = Geometry.pcOf(g.rotation(), tr, 61.0 / 2.5, 61.0 / 2.5);
    for (0..3) |i| try std.testing.expectApproxEqAbs(g.pc[i], back[i], 1e-13);
}

test "the beam's travel, in kikuchipy's scan convention, shifts the detector as kikuchipy's extrapolation says" {
    // one row down and one column right, 1.5 µm steps, 70 µm pixels binned by 8, 60 × 60
    const g: Geometry = .{ .pc = .{ 0.42, 0.21, 0.5 }, .px_um = 70, .step_x_um = 1.5, .step_y_um = 1.5 };
    const R = g.rotation();
    const t = g.translation(60, 60);
    const s = g.scanOffset(1, 1, 0, 0, -1, -1, 8);
    const pc = Geometry.pcOf(R, rot.sub(t, s), 60, 60);
    const f = 1.5 / (70.0 * 8.0 * 60.0);
    const alpha = std.math.degreesToRadians(20.0);
    try std.testing.expectApproxEqAbs(0.42 - f, pc[0], 1e-12);
    try std.testing.expectApproxEqAbs(0.21 - f * @cos(alpha), pc[1], 1e-12);
    try std.testing.expectApproxEqAbs(0.5 + f * @sin(alpha), pc[2], 1e-12);
}
