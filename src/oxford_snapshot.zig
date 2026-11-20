//! Native indexing results expressed in the Oxford CS1 and crystal bases.
const std = @import("std");
const engine = @import("engine.zig");
const oxford = @import("oxford.zig");
const plane = @import("plane.zig");
const rot = @import("rot.zig");

fn value(w: *std.Io.Writer, v: anytype) !void {
    try std.json.Stringify.value(v, .{}, w);
}

fn numbers(w: *std.Io.Writer, data: anytype) !void {
    try w.writeByte('[');
    for (data, 0..) |n, i| {
        if (i != 0) try w.writeByte(',');
        if (@typeInfo(@TypeOf(n)) == .float and !std.math.isFinite(n)) {
            try w.writeAll("null");
        } else try value(w, n);
    }
    try w.writeByte(']');
}

fn orientation(raw: []const f32, sample: rot.Quat, crystal: rot.Quat) !rot.Quat {
    const q: rot.Quat = .{ raw[0], raw[1], raw[2], raw[3] };
    const norm = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    if (!std.math.isFinite(norm) or @abs(norm - 1) > 1e-5) return error.InvalidIndexedOrientation;
    // q maps engine sample to master crystal. Change both bases without
    // reducing the result under a second set of symmetry operators.
    return rot.qstd(rot.qmul(rot.qconj(crystal), rot.qmul(q, rot.qconj(sample))));
}

pub fn snapshot(e: *engine.Engine, requested: ?rot.Quat) ![]u8 {
    const source = switch (e.source) {
        .file => |s| s,
        else => return error.OxfordFrameRequiresOriginalAcquisition,
    };
    if (source.format != .h5_oxford) return error.OxfordFrameRequiresOriginalAcquisition;
    const info = source.info;
    if (!e.res.context_consistent) return error.MixedIndexingContextsReindexScan;
    const context = e.res.context orelse return error.NoIndexedResults;
    if (context.phase_generation != e.phase_generation) return error.PhasesChangedReindexScan;
    const calibration = context.calibration;
    try calibration.requireSupported();
    const metadata = calibration.metadata.?;
    const geometry = context.geometry;
    if (geometry.plane_kind > @intFromEnum(plane.Kind.projective)) return error.InvalidOxfordGeometry;
    if (geometry.plane_kind == 0 and geometry.px_um != 0) return error.OxfordScanTranslationNotValidated;
    for (geometry.pc ++ geometry.rot ++ geometry.plane ++ [_]f64{ geometry.px_um, geometry.sample_tilt_deg, geometry.detector_tilt_deg, geometry.step_x_um, geometry.step_y_um }) |x| {
        if (!std.math.isFinite(x)) return error.InvalidOxfordGeometry;
    }
    if (geometry.pc[2] <= 0 or geometry.step_x_um <= 0 or geometry.step_y_um <= 0 or geometry.px_um < 0) return error.InvalidOxfordGeometry;
    const effective_detector = try calibration.effectiveDetectorEuler(&geometry);
    for (effective_detector) |angle| if (!std.math.isFinite(angle)) return error.InvalidOxfordGeometry;

    const sample = oxford.sampleBasis();
    if (requested) |q| {
        // The C ABI accepts an optional frame. For an original calibrated
        // acquisition only the same rotation, up to quaternion sign, is valid.
        var norm: f64 = 0;
        var dot: f64 = 0;
        for (q, sample) |x, s| {
            norm += x * x;
            dot += x * s;
        }
        if (!std.math.isFinite(norm) or @abs(norm - 1) > 1e-6) return error.InvalidExportFrame;
        if (@abs(dot) / @sqrt(norm) < 1 - 1e-10) return error.OxfordFrameOverrideConflictsWithAcquisition;
    }
    if (e.phases.items.len == 0) return error.NoPhases;
    if (std.mem.findNone(i32, e.res.phase, &.{-1}) == null) return error.NoIndexedResults;
    const crystal = try e.gpa.alloc(rot.Quat, e.phases.items.len);
    defer e.gpa.free(crystal);
    for (e.phases.items, crystal) |p, *c| {
        c.* = try oxford.crystalBasis(p.master.crystal_frame, p.master.lattice orelse return error.MissingPhaseLattice);
    }
    const preserve_source_pc = geometry.plane_kind != 0;
    const warnings: []const []const u8 = if (preserve_source_pc)
        &.{"Orientations were indexed with a fitted source plane. Standard Oxford pattern-centre fields retain acquisition values. Fitted values are stored in /zebsdsolve/PatternCentres."}
    else
        &.{};

    var out: std.Io.Writer.Allocating = .init(e.gpa);
    errdefer out.deinit();
    const w = &out.writer;
    try w.writeAll("{\"schema_version\":1,\"version\":");
    try value(w, engine.version);
    try w.writeAll(",\"scan\":");
    try value(w, .{
        .rows = info.rows,
        .cols = info.cols,
        .pat_h = info.pat_h,
        .pat_w = info.pat_w,
        .format = source.formatName(),
    });
    try w.writeAll(",\"geometry\":");
    try value(w, geometry);
    try w.writeAll(",\"calibration\":");
    try value(w, .{
        .kind = "oxford",
        .status = "supported",
        .source_detector_euler_radians = metadata.detector_euler,
        .source_specimen_euler_radians = metadata.specimen_euler,
        .source_tilt_radians = metadata.original_tilt,
        .tilt_axis_radians = metadata.tilt_axis,
        .scanning_rotation_radians = metadata.scanning_rotation,
        .effective_detector_euler_radians = effective_detector,
        .effective_tilt_radians = std.math.degreesToRadians(geometry.sample_tilt_deg),
    });
    try w.writeAll(",\"oxford_frame\":");
    try value(w, .{
        .convention_version = 2,
        .detector_pose_convention = "oxford-cs3-to-microscope-v2",
        .engine_to_cs1 = sample,
        .provenance = "Inherited Oxford acquisition: fixed engine-to-CS1 basis, full detector calibration, phase-specific master crystal conversion",
        .automatic = true,
        .phase_generation = context.phase_generation,
        .preserve_source_pattern_centres = preserve_source_pc,
        .warnings = warnings,
        .phase_crystal_to_master = crystal,
        .master_crystal_convention = "X||a,Z||c*",
        .output_crystal_convention = "X||a*,Z||c",
    });
    try w.writeAll(",\"pattern_centres\":[");
    for (0..e.res.phase.len) |i| {
        // The indexed plane is already expressed in pattern-centre coordinates.
        // Without a plane the supported geometry uses a constant reference PC.
        const pc = if (preserve_source_pc)
            plane.at(geometry.plane, @floatFromInt(i / info.cols), @floatFromInt(i % info.cols))
        else
            geometry.pc;
        if (pc[2] <= 0) return error.InvalidOxfordGeometry;
        for (pc, 0..) |x, k| {
            if (!std.math.isFinite(x)) return error.InvalidOxfordGeometry;
            if (i != 0 or k != 0) try w.writeByte(',');
            try value(w, x);
        }
    }
    try w.writeAll("],\"settings\":");
    const frozen_settings = try context.settings.toJson(e.gpa);
    defer e.gpa.free(frozen_settings);
    try w.writeAll(frozen_settings);
    try w.writeAll(",\"phases\":[");
    for (e.phases.items, 0..) |p, i| {
        if (i != 0) try w.writeByte(',');
        try value(w, .{
            .name = p.master.name,
            .space_group = p.master.space_group,
            .laue = @intFromEnum(p.master.exportLaue()),
            .laue_symbol = p.master.exportLaue().name(),
            .lattice = p.master.lattice.?,
            .master_source = p.master.source,
            .master_source_format_version = p.master.source_format_version,
            .master_crystal_convention = p.master.crystal_frame.label(),
        });
    }
    try w.writeAll("],\"results\":{");
    inline for (.{ "quaternions", "euler" }) |field| {
        if (comptime std.mem.eql(u8, field, "euler")) try w.writeByte(',');
        try w.print("\"{s}\":[", .{field});
        for (e.res.phase, 0..) |phase, i| {
            if (phase < -1 or (phase >= 0 and @as(usize, @intCast(phase)) >= crystal.len)) return error.InvalidIndexedPhase;
            const q = if (phase >= 0)
                try orientation(e.res.quat[4 * i ..][0..4], sample, crystal[@intCast(phase)])
            else
                rot.quat_identity;
            const data = if (comptime std.mem.eql(u8, field, "quaternions")) q else rot.toBunge(q);
            for (data, 0..) |x, k| {
                if (i != 0 or k != 0) try w.writeByte(',');
                try value(w, if (phase >= 0) x else 0);
            }
        }
        try w.writeByte(']');
    }
    inline for (.{ "phase", "ncc", "nmi", "z", "sweep" }) |field| {
        try w.print(",\"{s}\":", .{field});
        try numbers(w, @field(e.res, field));
    }
    try w.writeAll("}}");
    return out.toOwnedSlice();
}
