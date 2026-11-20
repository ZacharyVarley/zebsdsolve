//! Oxford acquisition calibration and crystal-frame conventions.
//! Runtime implementation of source-defined transforms, checked against native MTEX.
//! Engine q: sample -> master crystal. Oxford q: CS1 -> Oxford crystal.
const std = @import("std");
const rot = @import("rot.zig");
const Geometry = @import("geometry.zig").Geometry;

pub const Metadata = struct {
    detector_euler: [3]f64 = @splat(std.math.nan(f64)),
    specimen_euler: [3]f64 = @splat(std.math.nan(f64)),
    tilt_axis: f64 = std.math.nan(f64),
    scanning_rotation: f64 = std.math.nan(f64),
    original_tilt: f64 = std.math.nan(f64),
};

pub const Status = enum { not_oxford, supported, missing_detector_orientation, missing_specimen_orientation, missing_sample_tilt, unknown_tilt_axis, unsupported_tilt_axis, unknown_scan_rotation, unsupported_scan_rotation };

/// Internal context, deliberately separate from the public Geometry ABI and its
/// fitted residual. Opening a different source replaces this context.
pub const Calibration = struct {
    metadata: ?Metadata = null,

    pub fn status(self: Calibration) Status {
        const m = self.metadata orelse return .not_oxford;
        for (m.detector_euler) |x| if (!std.math.isFinite(x)) return .missing_detector_orientation;
        for (m.specimen_euler) |x| if (!std.math.isFinite(x)) return .missing_specimen_orientation;
        if (!std.math.isFinite(m.original_tilt)) return .missing_sample_tilt;
        if (!std.math.isFinite(m.tilt_axis)) return .unknown_tilt_axis;
        if (@abs(m.tilt_axis) > 1e-6) return .unsupported_tilt_axis;
        if (!std.math.isFinite(m.scanning_rotation)) return .unknown_scan_rotation;
        if (scanDirection(m.scanning_rotation) == null) return .unsupported_scan_rotation;
        return .supported;
    }

    /// Settings signs describe the zero-angle raster basis. Oxford's scan
    /// rotation changes beam travel, not the detector pose or CS1 orientations.
    /// Apply the acquisition half turn once, retaining both caller preferences.
    pub fn scanSigns(self: Calibration, x: i32, y: i32) [2]i32 {
        const direction = if (self.status() == .supported) scanDirection(self.metadata.?.scanning_rotation).? else 1;
        return .{ direction * x, direction * y };
    }

    pub fn requireSupported(self: Calibration) !void {
        return switch (self.status()) {
            .supported => {},
            .not_oxford => error.OxfordFrameRequiresOriginalAcquisition,
            .missing_detector_orientation => error.MissingOxfordDetectorOrientation,
            .missing_specimen_orientation => error.MissingOxfordSpecimenOrientation,
            .missing_sample_tilt => error.MissingOxfordSampleTilt,
            .unknown_tilt_axis => error.UnknownOxfordTiltAxis,
            .unsupported_tilt_axis => error.UnsupportedOxfordTiltAxis,
            .unknown_scan_rotation => error.UnknownOxfordScanRotation,
            .unsupported_scan_rotation => error.UnsupportedOxfordScanRotation,
        };
    }

    /// Calibrated baseline plus an optional local detector-column tilt. Sample
    /// tilt edits recompute the Oxford chain; the sample basis B stays fixed.
    pub fn nominal(self: Calibration, g: *const Geometry) rot.Mat3 {
        if (self.status() != .supported) return g.nominal();
        const m = self.metadata.?;
        const tilt = std.math.degreesToRadians(g.sample_tilt_deg);
        // Oxford D maps detector CS3 to the microscope frame. fromBunge
        // is passive, so transpose it to obtain the source-defined active D.
        const detector = rot.transpose(rot.toMat(rot.fromBunge(m.detector_euler)));
        const a = rot.toMat(rot.so3Exp(.{ -tilt, 0, 0 }));
        const base = rot.matMul(rot.matMul(rot.matMul(rot.transpose(rot.toMat(sampleBasis())), a), detector), rasterBasis());
        return rot.matMul(base, rot.toMat(rot.so3Exp(.{ 0, std.math.degreesToRadians(g.detector_tilt_deg), 0 })));
    }

    pub fn rotation(self: Calibration, g: *const Geometry) rot.Mat3 {
        return rot.matMul(self.nominal(g), rot.toMat(rot.so3Exp(g.rot)));
    }

    pub fn translation(self: Calibration, g: *const Geometry, rows: f64, cols: f64) rot.Vec3 {
        return Geometry.translationForRotation(self.rotation(g), g.pc, rows, cols);
    }

    /// Write the full effective pose back into Oxford's detector Euler field.
    /// This includes fitted residuals without changing the fixed sample basis.
    pub fn effectiveDetectorEuler(self: Calibration, g: *const Geometry) !rot.Vec3 {
        try self.requireSupported();
        const a = rot.toMat(rot.so3Exp(.{ std.math.degreesToRadians(g.sample_tilt_deg), 0, 0 }));
        const detector = rot.matMul(rot.matMul(rot.matMul(a, rot.toMat(sampleBasis())), self.rotation(g)), rot.transpose(rasterBasis()));
        return rot.toBunge(rot.fromMat(rot.transpose(detector)));
    }

    pub fn eql(a: Calibration, b: Calibration) bool {
        const x = a.metadata orelse return b.metadata == null;
        const y = b.metadata orelse return false;
        for (x.detector_euler, y.detector_euler) |p, q| if (!same(p, q)) return false;
        for (x.specimen_euler, y.specimen_euler) |p, q| if (!same(p, q)) return false;
        return same(x.tilt_axis, y.tilt_axis) and same(x.scanning_rotation, y.scanning_rotation) and same(x.original_tilt, y.original_tilt);
    }

    pub fn hash(self: Calibration, h: anytype) void {
        h.update(@tagName(self.status()));
        if (self.metadata) |m| {
            h.update(std.mem.asBytes(&m.detector_euler));
            h.update(std.mem.asBytes(&m.specimen_euler));
            h.update(std.mem.asBytes(&m.tilt_axis));
            h.update(std.mem.asBytes(&m.scanning_rotation));
            h.update(std.mem.asBytes(&m.original_tilt));
        }
    }
};

/// Only zero and a half turn, modulo one full turn, have validated raster
/// geometry. Oxford stores radians in float32; tolerate that rounding only.
fn scanDirection(angle: f64) ?i32 {
    if (!std.math.isFinite(angle)) return null;
    // Preserve the complete previously supported interval at its exact edges.
    if (@abs(angle) <= 1e-6) return 1;
    const full = 2 * std.math.pi;
    const wrapped = @mod(angle, full);
    if (@min(wrapped, full - wrapped) <= 1e-6) return 1;
    if (@abs(wrapped - std.math.pi) <= 1e-6) return -1;
    return null;
}

fn same(a: f64, b: f64) bool {
    return a == b or (std.math.isNan(a) and std.math.isNan(b));
}

/// Fixed engine-sample -> Oxford CS1 components for the verified convention.
pub fn sampleBasis() rot.Quat {
    return rot.so3Exp(.{ 0, 0, std.math.pi / 2.0 });
}

/// Engine detector row/column/ray-depth -> Oxford CS3: (column,-row,depth).
/// Oxford x points right, y up from the bottom-left origin, z into the screen.
pub fn rasterBasis() rot.Mat3 {
    return .{ 0, 1, 0, -1, 0, 0, 0, 0, 1 };
}

/// Oxford crystal -> master crystal. Only source-verified Cartesian conventions
/// and validated orthogonal/conventional hexagonal cells are enabled initially.
pub fn crystalBasis(frame: @import("master.zig").CrystalFrame, lattice: [6]f64) !rot.Quat {
    if (frame != .x_a_z_cstar) return error.UnknownMasterCrystalConvention;
    for (lattice) |x| if (!std.math.isFinite(x) or x <= 0) return error.InvalidMasterLattice;
    const a90 = @abs(lattice[3] - 90) < 1e-4;
    const b90 = @abs(lattice[4] - 90) < 1e-4;
    if (a90 and b90 and @abs(lattice[5] - 90) < 1e-4) return rot.quat_identity;
    if (a90 and b90 and @abs(lattice[5] - 120) < 1e-4 and @abs(lattice[0] - lattice[1]) < 1e-5 * lattice[0])
        return rot.so3Exp(.{ 0, 0, std.math.pi / 6.0 });
    return error.UnsupportedMasterCrystalSetting;
}

test "Oxford residual pose and detector Euler roundtrip keep sample basis fixed" {
    const c: Calibration = .{ .metadata = .{ .detector_euler = .{ 0.00722032, 1.5679878, 0.00296268 }, .specimen_euler = .{ 0.2, 0.1, 0.3 }, .tilt_axis = 0, .scanning_rotation = 0, .original_tilt = -std.math.pi / 9.0 } };
    var g: Geometry = .{ .sample_tilt_deg = -20, .rot = .{ 0, 0, 0 } };
    const d = try c.effectiveDetectorEuler(&g);
    for (rot.toMat(rot.fromBunge(d)), rot.toMat(rot.fromBunge(c.metadata.?.detector_euler))) |a, b| try std.testing.expectApproxEqAbs(a, b, 1e-12);
    g.rot = .{ 0.01, -0.02, 0.005 };
    const changed = try c.effectiveDetectorEuler(&g);
    var copy = c;
    copy.metadata.?.detector_euler = changed;
    var baseline = g;
    baseline.rot = .{ 0, 0, 0 };
    for (copy.rotation(&baseline), c.rotation(&g)) |a, b| try std.testing.expectApproxEqAbs(a, b, 1e-12);
}

test "Oxford crystal conversion is per convention and cell setting" {
    const q = try crystalBasis(.x_a_z_cstar, .{ 2.95, 2.95, 4.68, 90, 90, 120 });
    const v = rot.qapply(q, .{ 1, 0, 0 });
    try std.testing.expectApproxEqAbs(@sqrt(3.0) / 2.0, v[0], 1e-12);
    try std.testing.expectApproxEqAbs(0.5, v[1], 1e-12);
    try std.testing.expectEqual(rot.quat_identity, try crystalBasis(.x_a_z_cstar, .{ 4.1, 4.1, 4.1, 90, 90, 90 }));
    try std.testing.expectError(error.UnknownMasterCrystalConvention, crystalBasis(.unknown, .{ 2.95, 2.95, 4.68, 90, 90, 120 }));
    try std.testing.expectError(error.UnsupportedMasterCrystalSetting, crystalBasis(.x_a_z_cstar, .{ 2.95, 3.0, 4.68, 90, 105, 90 }));
}

test "Oxford per-cell positions and plane frame use the calibrated baseline" {
    const c: Calibration = .{ .metadata = .{ .detector_euler = .{ 0.00722032, 1.5679878, 0.00296268 }, .specimen_euler = .{ 0, 0, 0 }, .tilt_axis = 0, .scanning_rotation = 0, .original_tilt = -std.math.pi / 9.0 } };
    for ([_]f64{ 0, std.math.pi }) |angle| {
        var calibrated = c;
        calibrated.metadata.?.scanning_rotation = angle;
        const signs = calibrated.scanSigns(-1, 1);
        var g: Geometry = .{ .pc = .{ 0.42, 0.21, 0.5 }, .sample_tilt_deg = -20, .rot = .{ 0.02, -0.03, 0.04 }, .px_um = 70, .step_x_um = 2, .step_y_um = 3 };
        const R = calibrated.rotation(&g);
        const t = g.positionForRotation(R, 4, 6, 3, 5, signs[0], signs[1], 2.5, 24.4, 32.4);
        const want = rot.sub(calibrated.translation(&g, 24.4, 32.4), g.scanOffset(4, 6, 3, 5, signs[0], signs[1], 2.5));
        for (t, want) |a, b| try std.testing.expectApproxEqAbs(a, b, 1e-12);
        // Construct an affine plane from calibrated step-model PCs. Fitting and decoding it
        // must recover the same residual relative to Oxford's full nominal attitude.
        const plane = @import("plane.zig");
        const rows = 30;
        const cols = 40;
        const fr = plane.Frame.of(rows, cols, 1.25);
        var points: [30]plane.Point = undefined;
        for (0..30) |k| {
            const row: f64 = @floatFromInt((k * 7) % rows);
            const col: f64 = @floatFromInt((k * 11) % cols);
            const p = g.positionForRotation(R, row, col, fr.r0, fr.c0, signs[0], signs[1], 1, 60, 75);
            const pc = Geometry.pcOf(R, p, 60, 75);
            points[k] = .{ .col = col, .row = row, .y = .{ pc[0] * fr.aspect, pc[1], pc[2] } };
        }
        g.plane = plane.fit(.affine, &points, fr).?;
        g.plane_kind = 1;
        const pc = plane.at(g.plane, 4, 6);
        const got = Geometry.pcOf(R, g.positionForRotation(R, 4, 6, fr.r0, fr.c0, signs[0], signs[1], 2.5, 24.4, 30.5), 24.4, 30.5);
        for (pc, got) |a, b| try std.testing.expectApproxEqAbs(a, b, 1e-12);
        const said = plane.saidWithNominal(g.plane, &g, calibrated.nominal(&g), rows, cols, 1.25, 60, signs[0], signs[1]);
        for (g.rot, said.rot) |a, b| try std.testing.expectApproxEqAbs(a, b, 1e-9);
        for (said.step) |step| try std.testing.expectApproxEqAbs(@as(f64, 1), step, 1e-9);
    }
}

test "Oxford zero and half-turn scan rotations preserve pose and caller signs" {
    const base: Calibration = .{ .metadata = .{ .detector_euler = .{ 6.252466, 1.594263, 6.263398 }, .specimen_euler = .{ 0, 0, 0 }, .tilt_axis = 0, .scanning_rotation = 0, .original_tilt = 1.2217305 } };
    const g: Geometry = .{ .sample_tilt_deg = 70, .px_um = 70, .step_x_um = 2, .step_y_um = 3 };
    // Fixed expectations do not derive their answer from scanDirection.
    try std.testing.expectEqual([2]i32{ -1, -1 }, base.scanSigns(-1, -1));
    var half = base;
    half.metadata.?.scanning_rotation = @as(f64, @as(f32, std.math.pi));
    try std.testing.expectEqual([2]i32{ 1, 1 }, half.scanSigns(-1, -1));
    try std.testing.expectEqual([2]i32{ 1, -1 }, half.scanSigns(-1, 1));
    const signed = half.scanSigns(-1, 1);
    const offset = g.scanOffset(4, 6, 3, 5, signed[0], signed[1], 1);
    for (offset, [_]f64{ -3.0 / 70.0, 2.0 / 70.0, 0 }) |a, expected| try std.testing.expectApproxEqAbs(expected, a, 1e-12);
    for ([_]f64{ 0, -1e-6, 1e-6, 2 * std.math.pi, -4 * std.math.pi, @as(f64, @as(f32, 4 * std.math.pi)), std.math.pi, -std.math.pi, @as(f64, @as(f32, std.math.pi)) }) |angle| {
        var c = base;
        c.metadata.?.scanning_rotation = angle;
        try std.testing.expectEqual(Status.supported, c.status());
        for (c.nominal(&g), base.nominal(&g)) |a, b| try std.testing.expectApproxEqAbs(a, b, 1e-12);
        for (try c.effectiveDetectorEuler(&g), try base.effectiveDetectorEuler(&g)) |a, b| try std.testing.expectApproxEqAbs(a, b, 1e-12);
        for ([_]i32{ -1, 1 }) |x| for ([_]i32{ -1, 1 }) |y| {
            const signs = c.scanSigns(x, y);
            const is_half = scanDirection(angle).? == -1;
            try std.testing.expectEqual(if (is_half) -x else x, signs[0]);
            try std.testing.expectEqual(if (is_half) -y else y, signs[1]);
            const turned_offset = g.scanOffset(4, 6, 3, 5, signs[0], signs[1], 1);
            const zero = g.scanOffset(4, 6, 3, 5, x, y, 1);
            for (turned_offset, zero) |a, b| try std.testing.expectApproxEqAbs(if (is_half) -b else b, a, 1e-12);
        };
    }
    for ([_]f64{ std.math.pi / 2.0, std.math.pi + 2e-6, 2 * std.math.pi + 2e-6 }) |angle| {
        var c = base;
        c.metadata.?.scanning_rotation = angle;
        try std.testing.expectEqual(Status.unsupported_scan_rotation, c.status());
        try std.testing.expectEqual([2]i32{ -1, 1 }, c.scanSigns(-1, 1));
    }
}

test "Oxford nonideal detector rays follow source CS3 to microscope calibration" {
    // Independent native MTEX matrix columns are H*(-y), H*x, H*z,
    // where H=B.inv()*Rx(-tilt)*D uses S05's float32 header.
    // No production helper supplies the expected columns.
    const c: Calibration = .{ .metadata = .{ .detector_euler = .{ 6.252466201782227, 1.594262957572937, 6.263398170471191 }, .specimen_euler = .{ 0, 0, 0 }, .tilt_axis = 0, .scanning_rotation = 3.1415927410125732, .original_tilt = 1.2217304706573486 } };
    const g: Geometry = .{ .sample_tilt_deg = 69.99999967119774 };
    const expected: rot.Mat3 = .{
        -0.9310221761103453, -0.0289316252125052, -0.3638140578005752,
        0.0190559573659648,  -0.9993467982882921, 0.0307058176866571,
        -0.3644647830444304, 0.0216549720272926,  0.9309653506478496,
    };
    for (c.nominal(&g), expected) |actual, want| try std.testing.expectApproxEqAbs(want, actual, 1e-12);
    const effective = try c.effectiveDetectorEuler(&g);
    for (rot.toMat(rot.fromBunge(effective)), rot.toMat(rot.fromBunge(c.metadata.?.detector_euler))) |actual, want| try std.testing.expectApproxEqAbs(want, actual, 1e-12);
}
