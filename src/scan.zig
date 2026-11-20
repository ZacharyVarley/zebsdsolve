//! An EBSD scan on disk: one interface over every pattern file format, read in ranges of
//! patterns so a scan far larger than memory is streamed.
//!
//! Formats, recognized by content first and file name second:
//!   h5ebsd as written by kikuchipy, Bruker (Esprit), EDAX (OIM) and Oxford (.h5oina);
//!   EMsoft's simulated pattern files (EMEBSD); EDAX .up1 / .up2; Oxford .ebsp (uncompressed);
//!   NORDIF Pattern.dat (with Setting.txt).
//! A scan may be opened from several files (the pattern file first, then side files such as
//! Setting.txt). What a file does not say (map shape of a raw stream, pattern centre) stays
//! unset here and is the caller's to supply: `Hints` for the shape, the engine's geometry for
//! the rest.
//!
//! Patterns are numbered row-major over the map (row · cols + col) and returned as stored:
//! `bytes_per_px` bytes per pixel, top row first.
const std = @import("std");
const file_mod = @import("io/file.zig");
const File = file_mod.File;
const hdf5 = @import("io/hdf5.zig");
const rot = @import("rot.zig");

const Allocator = std.mem.Allocator;

/// What the HDF5 reader could not handle when the last open failed with UnsupportedHdf5.
pub var last_why: []const u8 = "";

pub const Format = enum { h5_kikuchipy, h5_bruker, h5_edax, h5_oxford, edax_up, oxford_ebsp, nordif, h5_emsoft };

pub const OverviewKind = enum(u32) { none = 0, band_contrast = 1, image_quality = 2, radon_quality = 3, pattern_quality = 4 };

pub const Pixel = enum(u32) { u8 = 0, u16 = 1, f32 = 2 };

/// What the caller knows that the file may not say (0: not given).
pub const Hints = struct {
    rows: u32 = 0,
    cols: u32 = 0,
    pat_h: u32 = 0,
    pat_w: u32 = 0,
    /// which scan of a multi-scan HDF5 file (empty: the first)
    scan_name: []const u8 = "",
};

/// Metadata as the file gives it; NaN where it gives none.
pub const Info = struct {
    rows: u32 = 0,
    cols: u32 = 0,
    pat_h: u32 = 0,
    pat_w: u32 = 0,
    pixel: Pixel = .u8,
    step_x_um: f64 = std.math.nan(f64),
    step_y_um: f64 = std.math.nan(f64),
    sample_tilt_deg: f64 = std.math.nan(f64),
    detector_tilt_deg: f64 = std.math.nan(f64),
    /// size of one stored pattern pixel on the detector
    px_um: f64 = std.math.nan(f64),
    voltage_kv: f64 = std.math.nan(f64),
    /// the file's pattern centre in the Bruker convention (x from the left, y from the top, in
    /// fractions of the pattern's width and height; z in fractions of its height): the mean of
    /// its values where it stores one per map cell
    pc: [3]f64 = @splat(std.math.nan(f64)),
    pc_note: []const u8 = "",
    /// the detector's rotation from its nominal attitude where the file stores its orientation,
    /// as geometry.rot has it (rotation vector, radians, about the detector's row and column
    /// directions and its normal)
    detector_rot: [3]f64 = @splat(std.math.nan(f64)),
};

/// Oxford's nominal detector orientation, (0, 90°, 0): Bunge angles of the detector's axes (x
/// along the pattern's columns, y up it, z its normal toward the sample, the axes its pattern
/// centre is measured in) in the sample's.
const oxford_nominal: rot.Vec3 = .{ 0, std.math.pi / 2.0, 0 };

/// The rotation (active, row-major) Bunge angles (φ1, Φ, φ2) describe: Rz(φ1) Rx(Φ) Rz(φ2).
fn bungeMatrix(e: rot.Vec3) rot.Mat3 {
    const c1 = @cos(e[0]);
    const s1 = @sin(e[0]);
    const c = @cos(e[1]);
    const s = @sin(e[1]);
    const c2 = @cos(e[2]);
    const s2 = @sin(e[2]);
    return .{
        c1 * c2 - s1 * c * s2, -c1 * s2 - s1 * c * c2, s1 * s,
        s1 * c2 + c1 * c * s2, -s1 * s2 + c1 * c * c2, -c1 * s,
        s * s2,                s * c2,                 c,
    };
}

/// Oxford's `Detector Orientation Euler` as geometry.rot: the detector's rotation away from the
/// nominal (0, 90°, 0), in its own axes. Oxford's detector axes are those its pattern centre is
/// measured in (x along the columns, y up the pattern); the engine's are the row direction, the
/// column direction and the normal, so (x, y, z) Oxford is (column, −row, normal) here. Null
/// when the angles lie more than 15° from the nominal: then they do not say what this reading
/// takes them to. (Checked on an AZtec acquisition: its orientations and the engine's then
/// agree to 0.12°; with the detector at the nominal attitude, 3.4°.)
pub fn oxfordDetectorRot(e: rot.Vec3) ?rot.Vec3 {
    for (e) |v| if (!std.math.isFinite(v)) return null;
    const b = rot.matMul(rot.transpose(bungeMatrix(oxford_nominal)), bungeMatrix(e));
    const v = rot.so3Log(rot.fromMat(b));
    if (rot.norm(v) > std.math.degreesToRadians(15.0)) return null;
    return .{ -v[1], v[0], v[2] };
}

/// The inverse of `oxfordDetectorRot`: geometry.rot as Oxford's Bunge angles (φ1 and φ2 in
/// [0, 2π)).
pub fn oxfordDetectorEuler(r: rot.Vec3) rot.Vec3 {
    const v: rot.Vec3 = .{ r[1], -r[0], r[2] };
    const m = rot.matMul(bungeMatrix(oxford_nominal), rot.toMat(rot.so3Exp(v)));
    const tau = 2 * std.math.pi;
    const phi = std.math.acos(std.math.clamp(m[8], -1, 1));
    var p1 = std.math.atan2(m[2], -m[5]);
    var p2 = std.math.atan2(m[6], m[7]);
    if (p1 < 0) p1 += tau;
    if (p2 < 0) p2 += tau;
    return .{ p1, phi, p2 };
}

/// Orientations a file stores, as `Scan.storedOrientations` gives them (owned).
pub const Stored = struct {
    euler: []f64,
    phase: []i32,

    pub fn deinit(self: *Stored, gpa: Allocator) void {
        gpa.free(self.euler);
        gpa.free(self.phase);
        self.* = undefined;
    }
};

pub const Scan = struct {
    gpa: Allocator,
    files: []File,
    format: Format,
    info: Info,
    /// byte offset of pattern 0 (raw streams)
    data_offset: u64 = 0,
    big_endian: bool = false,
    h5: ?hdf5.Hdf5 = null,
    ds: hdf5.Dataset = .{},
    /// Oxford .ebsp: file offset of each pattern's pixels (0: not stored)
    offsets: []u64 = &.{},
    /// static background (pat_h · pat_w), when the file stores one
    background: []f32 = &.{},
    /// Optional quality map in the acquisition Data group; discovery reads metadata only.
    overview_addr: ?u64 = null,
    /// Owned scalar values from a non-HDF5 companion, in verified pattern order.
    overview_values: []f32 = &.{},
    overview_kind: OverviewKind = .none,
    /// HDF5 formats: the scan's Data and Header groups
    h5_data: ?u64 = null,
    h5_header: ?u64 = null,
    /// a companion's stored orientations (Bunge, radians, 3 per cell) and phase labels
    /// (−1: not indexed), owned; HDF5 formats read theirs on demand (storedOrientations)
    stored_euler: []f64 = &.{},
    stored_phase: []i32 = &.{},
    why: []const u8 = "",

    pub fn count(self: *const Scan) u64 {
        return @as(u64, self.info.rows) * self.info.cols;
    }

    pub fn bytesPerPx(self: *const Scan) u32 {
        return switch (self.info.pixel) {
            .u8 => 1,
            .u16 => 2,
            .f32 => 4,
        };
    }

    pub fn patternBytes(self: *const Scan) u64 {
        return @as(u64, self.info.pat_h) * self.info.pat_w * self.bytesPerPx();
    }

    pub fn readStoredOverview(self: *Scan, out: []f32) !void {
        const n: usize = @intCast(self.count());
        if (out.len != n) return error.BadLength;
        if (self.overview_values.len != 0) {
            @memcpy(out, self.overview_values);
            return;
        }
        const address = self.overview_addr orelse return error.NoStoredOverview;
        const values = try self.gpa.alloc(f64, n);
        defer self.gpa.free(values);
        if ((try self.h5.?.numbers(address, values)) != n) return error.InvalidStoredOverview;
        for (values, out) |x, *v| v.* = if (std.math.isFinite(x) and @abs(x) <= std.math.floatMax(f32)) @floatCast(x) else std.math.nan(f32);
    }

    /// The orientations the file stores from its maker's own indexing, as it stores them: Bunge
    /// angles (radians; a format that writes degrees is converted) and a phase label per cell
    /// (any number; −1: not indexed). Which frames they are in is not read here
    /// (convention.identify finds it from the data). Null where the file stores none.
    pub fn storedOrientations(self: *Scan, gpa: Allocator) !?Stored {
        const n: usize = @intCast(self.count());
        if (self.stored_euler.len != 0) {
            if (self.stored_euler.len != 3 * n or self.stored_phase.len != n) return null;
            const euler = try gpa.dupe(f64, self.stored_euler);
            errdefer gpa.free(euler);
            return .{ .euler = euler, .phase = try gpa.dupe(i32, self.stored_phase) };
        }
        const data = self.h5_data orelse return null;
        const h = &self.h5.?;
        // where each format keeps them, in what unit, and whether its phase 0 means not indexed
        const Where = struct { one: []const u8 = "", three: [3][2][]const u8 = .{ .{ "", "" }, .{ "", "" }, .{ "", "" } }, degrees: bool = false, zero_unindexed: bool };
        const where: Where = switch (self.format) {
            .h5_oxford => .{ .one = "Euler", .zero_unindexed = true },
            .h5_bruker => .{ .three = .{ .{ "phi1", "" }, .{ "PHI", "" }, .{ "phi2", "" } }, .degrees = true, .zero_unindexed = true },
            .h5_edax => .{ .three = .{ .{ "phi1", "Phi1" }, .{ "Phi", "PHI" }, .{ "phi2", "Phi2" } }, .zero_unindexed = false },
            else => return null,
        };
        const euler = try gpa.alloc(f64, 3 * n);
        errdefer gpa.free(euler);
        if (where.one.len != 0) {
            const a = (try h.child(data, where.one)) orelse {
                gpa.free(euler);
                return null;
            };
            if ((h.numbers(a, euler) catch 0) != 3 * n) {
                gpa.free(euler);
                return null;
            }
        } else {
            const column = try gpa.alloc(f64, n);
            defer gpa.free(column);
            for (where.three, 0..) |names, k| {
                var got = false;
                for (names) |nm| {
                    if (nm.len == 0 or got) continue;
                    const a = (try h.child(data, nm)) orelse continue;
                    if ((h.numbers(a, column) catch 0) != n) continue;
                    for (column, 0..) |v, i| euler[3 * i + k] = v;
                    got = true;
                }
                if (!got) {
                    gpa.free(euler);
                    return null;
                }
            }
        }
        if (where.degrees) for (euler) |*v| {
            v.* = std.math.degreesToRadians(v.*);
        };
        const phase = try gpa.alloc(i32, n);
        errdefer gpa.free(phase);
        @memset(phase, 0);
        if (try h.child(data, "Phase")) |a| {
            const vals = try gpa.alloc(f64, n);
            defer gpa.free(vals);
            if ((h.numbers(a, vals) catch 0) == n) for (vals, phase) |v, *p| {
                p.* = if (std.math.isFinite(v) and @abs(v) < 1 << 30) @intFromFloat(v) else -1;
            };
        }
        for (phase, 0..) |*p, i| {
            if (where.zero_unindexed and p.* == 0) p.* = -1;
            // (angles past a turn: the not-indexed marker some makers write, 4π)
            for (euler[3 * i ..][0..3]) |v| {
                if (!std.math.isFinite(v) or @abs(v) > 2 * std.math.pi + 0.01) p.* = -1;
            }
        }
        return .{ .euler = euler, .phase = phase };
    }

    fn discoverOverview(self: *Scan, data: u64) void {
        const Candidate = struct { name: []const u8, kind: OverviewKind };
        // Ordered preference is deterministic, with no fallback to pattern IO.
        const candidates = [_]Candidate{
            .{ .name = "Band Contrast", .kind = .band_contrast },
            .{ .name = "band_contrast", .kind = .band_contrast },
            .{ .name = "BandContrast", .kind = .band_contrast },
            .{ .name = "Image Quality", .kind = .image_quality },
            .{ .name = "IQ", .kind = .image_quality },
            .{ .name = "image_quality", .kind = .image_quality },
            .{ .name = "iq", .kind = .image_quality },
            .{ .name = "ImageQuality", .kind = .image_quality },
            .{ .name = "RadonQuality", .kind = .radon_quality },
            .{ .name = "Radon Quality", .kind = .radon_quality },
            .{ .name = "Pattern Quality", .kind = .pattern_quality },
            .{ .name = "PatternQuality", .kind = .pattern_quality },
        };
        const h = &self.h5.?;
        const original_why = h.why;
        defer h.why = original_why;
        for (candidates) |candidate| {
            const address = (h.child(data, candidate.name) catch null) orelse continue;
            var ds = h.dataset(address) catch continue;
            defer ds.deinit(self.gpa);
            const n = self.count();
            const vector = (ds.rank == 1 and ds.dims[0] == n) or
                (ds.rank == 2 and ((ds.dims[0] == n and ds.dims[1] == 1) or (ds.dims[0] == 1 and ds.dims[1] == n)));
            const raster = ds.rank == 2 and ds.dims[0] == self.info.rows and ds.dims[1] == self.info.cols;
            if (!vector and !raster) continue;
            const numeric = switch (ds.dtype.class) {
                .uint, .int => ds.dtype.size == 1 or ds.dtype.size == 2 or ds.dtype.size == 4 or ds.dtype.size == 8,
                .float => ds.dtype.size == 4 or ds.dtype.size == 8,
                else => false,
            };
            if (!numeric) continue;
            self.overview_addr = address;
            self.overview_kind = candidate.kind;
            return;
        }
    }

    pub fn formatName(self: *const Scan) []const u8 {
        return switch (self.format) {
            .h5_kikuchipy => "kikuchipy h5ebsd",
            .h5_bruker => "Bruker h5ebsd",
            .h5_edax => "EDAX h5ebsd",
            .h5_oxford => "Oxford h5oina",
            .edax_up => "EDAX up1/up2",
            .oxford_ebsp => "Oxford ebsp",
            .nordif => "NORDIF",
            .h5_emsoft => "EMsoft patterns",
        };
    }

    /// Open the scan in `paths[0]`; further paths are side files.
    pub fn open(gpa: Allocator, paths: []const []const u8, hints: Hints) !*Scan {
        if (paths.len == 0) return error.FileNotFound;
        last_why = "";
        const self = try gpa.create(Scan);
        errdefer gpa.destroy(self);
        self.* = .{ .gpa = gpa, .files = try gpa.alloc(File, paths.len), .format = .nordif, .info = .{} };
        var opened: usize = 0;
        errdefer {
            for (self.files[0..opened]) |*f| f.close();
            gpa.free(self.files);
        }
        for (paths, 0..) |p, i| {
            self.files[i] = try File.open(gpa, p);
            opened = i + 1;
        }
        errdefer {
            self.ds.deinit(gpa);
            gpa.free(self.offsets);
            gpa.free(self.background);
            gpa.free(self.overview_values);
            gpa.free(self.stored_euler);
            gpa.free(self.stored_phase);
        }
        const f = &self.files[0];
        var head: [8]u8 = @splat(0);
        try f.read(0, head[0..@intCast(@min(8, f.size))]);
        const ext = extension(paths[0]);
        if (std.mem.eql(u8, &head, "\x89HDF\r\n\x1a\n")) {
            try self.openHdf5(hints);
        } else if (eqlIgnoreCase(ext, "up1") or eqlIgnoreCase(ext, "up2")) {
            try self.openUp(eqlIgnoreCase(ext, "up2"), paths, hints);
        } else if (eqlIgnoreCase(ext, "ebsp")) {
            try self.openEbsp(hints);
        } else if (eqlIgnoreCase(ext, "dat")) {
            try self.openNordif(hints);
        } else {
            // HDF5 behind a user block, else unknown
            self.openHdf5(hints) catch return error.UnknownScanFormat;
        }
        if (self.info.rows == 0 or self.info.cols == 0 or self.info.pat_h == 0 or self.info.pat_w == 0) return error.ScanShapeUnknown;
        return self;
    }

    pub fn close(self: *Scan) void {
        const gpa = self.gpa;
        self.ds.deinit(gpa);
        gpa.free(self.offsets);
        gpa.free(self.background);
        gpa.free(self.overview_values);
        gpa.free(self.stored_euler);
        gpa.free(self.stored_phase);
        for (self.files) |*f| f.close();
        gpa.free(self.files);
        gpa.destroy(self);
    }

    /// Patterns [first, first + n) as stored bytes into `dst` (n · patternBytes()).
    pub fn read(self: *Scan, first: u64, n: u64, dst: []u8) !void {
        const pb = self.patternBytes();
        if (first + n > self.count()) return error.OutOfRange;
        if (dst.len != n * pb) return error.BadLength;
        if (n == 0) return;
        switch (self.format) {
            .edax_up, .nordif => try self.files[0].read(self.data_offset + first * pb, dst),
            .oxford_ebsp => {
                var i: u64 = 0;
                while (i < n) : (i += 1) {
                    const out = dst[@intCast(i * pb)..][0..@intCast(pb)];
                    const off = self.offsets[@intCast(first + i)];
                    if (off == 0) @memset(out, 0) else try self.files[0].read(off, out);
                }
            },
            else => {
                const h = &self.h5.?;
                if (self.ds.rank == 3) return h.readRows(&self.ds, first, n, dst);
                // (rows, cols, h, w): whole map rows through a scratch row
                const cols: u64 = self.info.cols;
                const row_bytes = cols * pb;
                const r0 = first / cols;
                const r1 = (first + n - 1) / cols;
                if (first % cols == 0 and n % cols == 0) return h.readRows(&self.ds, r0, r1 - r0 + 1, dst);
                const tmp = try self.gpa.alloc(u8, @intCast(row_bytes));
                defer self.gpa.free(tmp);
                var r = r0;
                var done: u64 = 0;
                while (r <= r1) : (r += 1) {
                    try h.readRows(&self.ds, r, 1, tmp);
                    const a = if (r == r0) first % cols else 0;
                    const b = if (r == r1) (first + n - 1) % cols + 1 else cols;
                    const len: usize = @intCast((b - a) * pb);
                    @memcpy(dst[@intCast(done)..][0..len], tmp[@intCast(a * pb)..][0..len]);
                    done += len;
                }
            },
        }
        self.toLittle(dst);
    }

    fn toLittle(self: *const Scan, dst: []u8) void {
        if (!self.big_endian or self.bytesPerPx() == 1) return;
        const bp = self.bytesPerPx();
        var i: usize = 0;
        while (i + bp <= dst.len) : (i += bp) std.mem.reverse(u8, dst[i..][0..bp]);
    }

    /// The patterns of the given cells (ascending) as stored bytes into `dst`, back to back.
    /// Cells scattered over the file (a preview lattice, the fit patterns, what a sweep left)
    /// are fetched together: one round trip where the format allows it.
    pub fn readCells(self: *Scan, cells: []const u64, dst: []u8) !void {
        const pb: usize = @intCast(self.patternBytes());
        if (dst.len != cells.len * pb) return error.BadLength;
        const total = self.count();
        for (cells, 0..) |c, i| {
            if (c >= total) return error.OutOfRange;
            if (i > 0 and c <= cells[i - 1]) return error.CellsNotAscending;
        }
        if (cells.len == 0) return;
        const gpa = self.gpa;
        switch (self.format) {
            .edax_up, .nordif => {
                var ranges: std.ArrayList(file_mod.Range) = .empty;
                defer ranges.deinit(gpa);
                var i: usize = 0;
                while (i < cells.len) {
                    var j = i + 1;
                    while (j < cells.len and cells[j] == cells[j - 1] + 1) j += 1;
                    try ranges.append(gpa, .{ .offset = self.data_offset + cells[i] * pb, .len = (j - i) * pb });
                    i = j;
                }
                try self.files[0].readSpans(ranges.items, dst);
            },
            .oxford_ebsp => {
                var ranges: std.ArrayList(file_mod.Range) = .empty;
                defer ranges.deinit(gpa);
                var missing = false;
                for (cells) |c| {
                    const off = self.offsets[@intCast(c)];
                    if (off == 0) missing = true;
                    try ranges.append(gpa, .{ .offset = off, .len = pb });
                }
                if (!missing) return self.files[0].readv(ranges.items, dst);
                for (cells, 0..) |c, i| try self.read(c, 1, dst[i * pb ..][0..pb]);
                return;
            },
            else => {
                const h = &self.h5.?;
                if (self.ds.rank == 3) {
                    try h.readRowList(&self.ds, cells, dst);
                } else {
                    // (rows, cols, h, w): each map row the cells touch, whole, through a scratch
                    const cols: u64 = self.info.cols;
                    const tmp = try gpa.alloc(u8, @intCast(cols * pb));
                    defer gpa.free(tmp);
                    var i: usize = 0;
                    while (i < cells.len) {
                        const r = cells[i] / cols;
                        try h.readRows(&self.ds, r, 1, tmp);
                        while (i < cells.len and cells[i] / cols == r) : (i += 1) {
                            @memcpy(dst[i * pb ..][0..pb], tmp[@intCast((cells[i] % cols) * pb)..][0..pb]);
                        }
                    }
                }
            },
        }
        self.toLittle(dst);
    }

    // ── HDF5 ──

    fn openHdf5(self: *Scan, hints: Hints) !void {
        const gpa = self.gpa;
        self.h5 = try hdf5.Hdf5.open(gpa, &self.files[0]);
        const h = &self.h5.?;
        errdefer last_why = h.why;
        // the scan group: the named one, else the first group with EBSD/Data and EBSD/Header
        var scan_addr: ?u64 = null;
        const names = try h.list(h.root);
        defer h.freeList(names);
        for (names) |nm| {
            if (hints.scan_name.len != 0 and !std.mem.eql(u8, nm, hints.scan_name)) continue;
            const a = (try h.child(h.root, nm)) orelse continue;
            if (!try h.isGroup(a)) continue;
            const e = (try h.child(a, "EBSD")) orelse continue;
            if ((try h.child(e, "Data")) != null and (try h.child(e, "Header")) != null) {
                scan_addr = a;
                break;
            }
        }
        const scan = scan_addr orelse {
            if (try h.find("EMData/EBSD/EBSDPatterns")) |a| return self.openEmsoft(a, hints);
            return error.NoScanInFile;
        };
        const ebsd = (try h.child(scan, "EBSD")).?;
        const data = (try h.child(ebsd, "Data")).?;
        const header = (try h.child(ebsd, "Header")).?;
        self.h5_data = data;
        self.h5_header = header;

        const V = struct {
            format: Format,
            patterns: []const []const u8,
            rows: []const u8,
            cols: []const u8,
            pat_h: []const u8,
            pat_w: []const u8,
            step_x: []const u8,
            step_y: []const u8,
            tilt: []const u8,
            det_tilt: []const u8,
            background: []const u8,
        };
        const vendors = [_]V{
            .{ .format = .h5_kikuchipy, .patterns = &.{"patterns"}, .rows = "n_rows", .cols = "n_columns", .pat_h = "pattern_height", .pat_w = "pattern_width", .step_x = "step_x", .step_y = "step_y", .tilt = "sample_tilt", .det_tilt = "elevation_angle", .background = "static_background" },
            .{ .format = .h5_bruker, .patterns = &.{"RawPatterns"}, .rows = "NROWS", .cols = "NCOLS", .pat_h = "PatternHeight", .pat_w = "PatternWidth", .step_x = "XSTEP", .step_y = "YSTEP", .tilt = "SampleTilt", .det_tilt = "CameraTilt", .background = "StaticBackground" },
            .{ .format = .h5_edax, .patterns = &.{"Pattern"}, .rows = "nRows", .cols = "nColumns", .pat_h = "Pattern Height", .pat_w = "Pattern Width", .step_x = "Step X", .step_y = "Step Y", .tilt = "Sample Tilt", .det_tilt = "Camera Elevation Angle", .background = "" },
            .{ .format = .h5_oxford, .patterns = &.{ "Processed Patterns", "Unprocessed Patterns" }, .rows = "Y Cells", .cols = "X Cells", .pat_h = "Pattern Height", .pat_w = "Pattern Width", .step_x = "X Step", .step_y = "Y Step", .tilt = "Tilt Angle", .det_tilt = "", .background = "Processed Static Background" },
        };
        var vendor: ?V = null;
        var pat_addr: u64 = 0;
        for (vendors) |v| {
            if ((try h.child(header, v.rows)) == null) continue;
            for (v.patterns) |pn| if (try h.child(data, pn)) |a| {
                pat_addr = a;
                vendor = v;
                break;
            };
            if (vendor != null) break;
        }
        const v = vendor orelse return error.NoPatternsInFile;
        self.format = v.format;
        self.ds = try h.dataset(pat_addr);
        if (self.ds.rank != 3 and self.ds.rank != 4) return error.NoPatternsInFile;
        const info = &self.info;
        info.pat_h = @intCast(self.ds.dims[self.ds.rank - 2]);
        info.pat_w = @intCast(self.ds.dims[self.ds.rank - 1]);
        info.pixel = switch (self.ds.dtype.class) {
            .uint => switch (self.ds.dtype.size) {
                1 => Pixel.u8,
                2 => Pixel.u16,
                else => return error.UnsupportedPixelType,
            },
            .float => if (self.ds.dtype.size == 4) Pixel.f32 else return error.UnsupportedPixelType,
            else => return error.UnsupportedPixelType,
        };
        self.big_endian = self.ds.dtype.big_endian;
        const n_stored: u64 = if (self.ds.rank == 4) self.ds.dims[0] * self.ds.dims[1] else self.ds.dims[0];
        const num = struct {
            fn get(hh: *hdf5.Hdf5, group: u64, name: []const u8) f64 {
                if (name.len == 0) return std.math.nan(f64);
                const a = (hh.child(group, name) catch null) orelse return std.math.nan(f64);
                return hh.number(a) catch std.math.nan(f64);
            }
            /// mean of a per-pattern array
            fn mean(hh: *hdf5.Hdf5, g: Allocator, group: u64, name: []const u8) f64 {
                const a = (hh.child(group, name) catch null) orelse return std.math.nan(f64);
                var ds = hh.dataset(a) catch return std.math.nan(f64);
                defer ds.deinit(g);
                const n: usize = @intCast(@min(ds.elements(), 1 << 24));
                if (n == 0) return std.math.nan(f64);
                const vals = g.alloc(f64, n) catch return std.math.nan(f64);
                defer g.free(vals);
                const got = hh.numbers(a, vals) catch return std.math.nan(f64);
                var s: f64 = 0;
                for (vals[0..got]) |x| s += x;
                return s / @as(f64, @floatFromInt(@max(got, 1)));
            }
        };
        const rows = num.get(h, header, v.rows);
        const cols = num.get(h, header, v.cols);
        info.rows = if (rows > 0) @intFromFloat(rows) else 0;
        info.cols = if (cols > 0) @intFromFloat(cols) else 0;
        if (hints.rows != 0 and hints.cols != 0) {
            info.rows = hints.rows;
            info.cols = hints.cols;
        }
        if (@as(u64, info.rows) * info.cols != n_stored) {
            if (self.ds.rank == 4) {
                info.rows = @intCast(self.ds.dims[0]);
                info.cols = @intCast(self.ds.dims[1]);
            } else if (info.rows == 0 or info.cols == 0) {
                info.rows = 1;
                info.cols = @intCast(n_stored);
            } else return error.ScanIsNotAFullGrid;
        }
        self.discoverOverview(data);
        info.step_x_um = num.get(h, header, v.step_x);
        info.step_y_um = num.get(h, header, v.step_y);
        info.sample_tilt_deg = num.get(h, header, v.tilt);
        if (std.math.isNan(info.sample_tilt_deg) and v.format == .h5_bruker) info.sample_tilt_deg = num.get(h, header, "Sample Tilt");
        info.detector_tilt_deg = num.get(h, header, v.det_tilt);
        const ratio = @as(f64, @floatFromInt(info.pat_w)) / @as(f64, @floatFromInt(info.pat_h));
        switch (v.format) {
            .h5_kikuchipy => {
                info.pc = .{ num.mean(h, gpa, header, "pcx"), num.mean(h, gpa, header, "pcy"), num.mean(h, gpa, header, "pcz") };
                const px = num.get(h, header, "detector_pixel_size");
                const bin = num.get(h, header, "binning");
                // kikuchipy writes 1 when the size is unknown
                if (px > 1) info.px_um = px * (if (bin > 0) bin else 1);
                if (try h.child(scan, "SEM")) |sem| if (try h.child(sem, "Header")) |sh| {
                    info.voltage_kv = num.get(h, sh, "beam_energy");
                };
            },
            .h5_bruker => {
                info.pc = .{ num.mean(h, gpa, data, "PCX"), num.mean(h, gpa, data, "PCY"), num.mean(h, gpa, data, "DD") };
                const full = num.get(h, header, "DetectorFullHeightMicrons");
                const unclipped = num.get(h, header, "UnClippedPatternHeight");
                if (full > 0 and unclipped > 0) info.px_um = full / unclipped;
                // A clipped pattern's PCY is a fraction of the height before clipping, from the
                // top: Esprit's own per-pattern pattern centres move with the beam as its tilts
                // and step say only when PCY is read so and DD as a fraction of the stored
                // height, and the patterns put their centre there (2 px from it; 9 px from PCY
                // read on the stored height).
                const stored: f64 = @floatFromInt(info.pat_h);
                if (unclipped > stored and !std.math.isNan(info.pc[1])) {
                    info.pc[1] *= unclipped / stored;
                    info.pc_note = "Bruker's PCY read as a fraction of the pattern's height before clipping (UnClippedPatternHeight), DD of its stored height";
                }
                info.voltage_kv = num.get(h, header, "KV");
            },
            .h5_edax => {
                // the pattern spans the camera's diameter (millimetres)
                const dia = num.get(h, header, "Camera Diameter");
                if (dia > 0 and info.pat_w > 0) info.px_um = 1000.0 * dia / @as(f64, @floatFromInt(info.pat_w));
                if (try h.child(header, "Pattern Center Calibration")) |pc| {
                    const x = num.get(h, pc, "x-star");
                    const y = num.get(h, pc, "y-star");
                    const z = num.get(h, pc, "z-star");
                    // EDAX measures y from the bottom, x and y in fractions of the width and
                    // height, z of the smaller of the two (as kikuchipy has it); an all-zero
                    // entry means not calibrated
                    if (x != 0 or y != 0 or z != 0) info.pc = .{ x, 1 - y, z * @min(ratio, 1) };
                }
            },
            .h5_oxford => {
                // Oxford measures y from the bottom, and y and z in fractions of the width (the
                // H5OINA specification); its tilt is in radians
                const x = num.mean(h, gpa, data, "Pattern Center X");
                const y = num.mean(h, gpa, data, "Pattern Center Y");
                const z = num.mean(h, gpa, data, "Detector Distance");
                info.pc = .{ x, 1 - y * ratio, z * ratio };
                info.sample_tilt_deg = std.math.radiansToDegrees(info.sample_tilt_deg);
                if (try h.child(header, "Detector Orientation Euler")) |a| {
                    var eu: [3]f64 = undefined;
                    if ((h.numbers(a, &eu) catch 0) == 3) if (oxfordDetectorRot(eu)) |r| {
                        info.detector_rot = r;
                    };
                }
                info.voltage_kv = num.get(h, header, "Beam Voltage");
            },
            else => unreachable,
        }
        if (v.background.len != 0) if (try h.child(header, v.background)) |a| {
            const n = @as(usize, info.pat_h) * info.pat_w;
            const vals = try gpa.alloc(f64, n);
            defer gpa.free(vals);
            if ((h.numbers(a, vals) catch 0) == n) {
                self.background = try gpa.alloc(f32, n);
                for (vals, 0..) |x, i| self.background[i] = @floatCast(x);
            }
        };
    }

    /// EMsoft's EMEBSD output: patterns (n, h, w) with the simulation's detector parameters.
    fn openEmsoft(self: *Scan, pat_addr: u64, hints: Hints) !void {
        const h = &self.h5.?;
        self.format = .h5_emsoft;
        self.ds = try h.dataset(pat_addr);
        if (self.ds.rank != 3) return error.NoPatternsInFile;
        const info = &self.info;
        info.pat_h = @intCast(self.ds.dims[1]);
        info.pat_w = @intCast(self.ds.dims[2]);
        info.pixel = switch (self.ds.dtype.class) {
            .uint => if (self.ds.dtype.size == 1) Pixel.u8 else if (self.ds.dtype.size == 2) Pixel.u16 else return error.UnsupportedPixelType,
            .float => if (self.ds.dtype.size == 4) Pixel.f32 else return error.UnsupportedPixelType,
            else => return error.UnsupportedPixelType,
        };
        self.big_endian = self.ds.dtype.big_endian;
        const n = self.ds.dims[0];
        info.rows = 1;
        info.cols = @intCast(n);
        if (hints.rows != 0 and hints.cols != 0 and @as(u64, hints.rows) * hints.cols == n) {
            info.rows = hints.rows;
            info.cols = hints.cols;
        }
        // EMsoft keeps optional per-pattern scalars beside EBSDPatterns.
        self.discoverOverview((try h.find("EMData/EBSD")).?);
        const nml = (try h.find("NMLparameters/EBSDNameList")) orelse return;
        const get = struct {
            fn f(hh: *hdf5.Hdf5, group: u64, name: []const u8) f64 {
                const a = (hh.child(group, name) catch null) orelse return std.math.nan(f64);
                return hh.number(a) catch std.math.nan(f64);
            }
        }.f;
        // EMsoft: the pattern centre as pixel offsets from the detector's middle (xpc, ypc),
        // the distance L and the pixel size delta in micrometres, at the unbinned size
        const xpc = get(h, nml, "xpc");
        const ypc = get(h, nml, "ypc");
        const L = get(h, nml, "L");
        const delta = get(h, nml, "delta");
        const numsx = get(h, nml, "numsx");
        const numsy = get(h, nml, "numsy");
        if (delta > 0 and numsx > 0 and numsy > 0 and !std.math.isNan(xpc + ypc + L)) {
            info.pc = .{ 0.5 - xpc / numsx, 0.5 - ypc / numsy, L / (numsy * delta) };
            info.px_um = delta * numsy / @as(f64, @floatFromInt(info.pat_h));
        }
        info.detector_tilt_deg = get(h, nml, "thetac");
        info.voltage_kv = get(h, nml, "energymax");
        if (try h.find("NMLparameters/MCCLNameList")) |mc| info.sample_tilt_deg = get(h, mc, "sig");
    }

    // ── EDAX .up1 / .up2 ──

    fn openUp(self: *Scan, is16: bool, paths: []const []const u8, hints: Hints) !void {
        var sidecar: ?usize = null;
        for (paths[1..], 1..) |path, k| {
            if (eqlIgnoreCase(extension(path), "ang")) {
                if (sidecar != null) return error.DuplicateANGSidecar;
                sidecar = k;
            }
        }
        const f = &self.files[0];
        if (f.size < 16) return error.BadScanFile;
        const version = try f.int(u32, 0);
        const w = try f.int(u32, 4);
        const hgt = try f.int(u32, 8);
        const off = try f.int(u32, 12);
        if ((version != 1 and version < 3) or w == 0 or hgt == 0 or off < 16 or off > f.size) return error.BadScanFile;
        const info = &self.info;
        self.format = .edax_up;
        info.pat_w = w;
        info.pat_h = hgt;
        info.pixel = if (is16) .u16 else .u8;
        self.data_offset = off;
        const pb = std.math.mul(u64, @as(u64, w) * hgt, self.bytesPerPx()) catch return error.BadScanFile;
        if ((f.size - off) % pb != 0) return error.UPIncompletePattern;
        const n = (f.size - off) / pb;
        if (n == 0 or n > std.math.maxInt(u32)) return error.ScanShapeMismatch;
        var nx: u32 = 0;
        var ny: u32 = 0;
        var dx: f64 = std.math.nan(f64);
        var dy = dx;
        var hex: u8 = 0;
        var extra: u8 = 0;
        if (version >= 3) {
            if (off < 42) return error.BadScanFile;
            // Packed v3+ header: extra-pattern flag, dimensions, hex flag, steps.
            extra = try f.int(u8, 16);
            nx = try f.int(u32, 17);
            ny = try f.int(u32, 21);
            hex = try f.int(u8, 25);
            dx = @bitCast(try f.int(u64, 26));
            dy = @bitCast(try f.int(u64, 34));
            if (hex == 0 and @as(u64, nx) * ny == n) {
                info.cols = nx;
                info.rows = ny;
                info.step_x_um = dx;
                info.step_y_um = dy;
            }
        }
        if (sidecar) |k| {
            if (hex != 0) return error.UnsupportedUPHexGrid;
            if (version >= 3) {
                if (nx == 0 or ny == 0 or @as(u64, nx) * ny != n) return if (extra != 0) error.AmbiguousUPExtraPatterns else error.UPPatternCountMismatch;
                if (!validStep(dx) or !validStep(dy)) return error.InvalidUPStep;
            }
            const ang = try @import("ang.zig").load(self.gpa, &self.files[k], n, if (version >= 3) .{ dx, dy } else null, w, hgt);
            errdefer self.gpa.free(ang.iq);
            errdefer self.gpa.free(ang.euler);
            errdefer self.gpa.free(ang.phase);
            if (version >= 3 and (nx != ang.cols or ny != ang.rows)) return error.ScanShapeMismatch;
            if (version >= 3 and (!sameStep(dx, ang.step_x_um) or !sameStep(dy, ang.step_y_um))) return error.UPANGStepMismatch;
            if ((hints.rows != 0 and hints.rows != ang.rows) or (hints.cols != 0 and hints.cols != ang.cols)) return error.ScanShapeMismatch;
            info.rows = ang.rows;
            info.cols = ang.cols;
            info.step_x_um = ang.step_x_um;
            info.step_y_um = ang.step_y_um;
            if (ang.pc) |pc| info.pc = pc;
            info.pc_note = ang.pc_note;
            self.overview_values = ang.iq;
            self.overview_kind = .image_quality;
            self.stored_euler = ang.euler;
            self.stored_phase = ang.phase;
            return;
        }
        if (hints.rows != 0 and hints.cols != 0) {
            info.rows = hints.rows;
            info.cols = hints.cols;
        }
        if (info.rows == 0) {
            info.rows = 1;
            info.cols = @intCast(n);
        }
        if (@as(u64, info.rows) * info.cols > n) return error.ScanShapeMismatch;
    }

    // ── Oxford .ebsp ──

    fn openEbsp(self: *Scan, hints: Hints) !void {
        const gpa = self.gpa;
        const f = &self.files[0];
        if (f.size < 24) return error.BadScanFile;
        const first8 = try f.int(i64, 0);
        const version: i64 = if (first8 < 0) -first8 else 0;
        const hdr: u64 = if (version < 5) 16 else 24;
        const table: u64 = if (version == 0) 0 else if (version > 3) 9 else 8;
        // The table holds one offset per pattern, in scan order; the records follow it in the
        // order they were acquired, which need not be scan order. So the table ends at the
        // smallest offset in it: read on while the entries read so far leave room for more.
        var bound: u64 = std.math.maxInt(u64);
        var seen: u64 = 0;
        while (seen < bound) : (seen += 1) {
            const at = table + seen * 8;
            if (at + 8 > f.size) break;
            const o = try f.int(i64, at);
            if (o <= 0) continue;
            const uo: u64 = @intCast(o);
            if (uo <= table) return error.BadScanFile;
            bound = @min(bound, (uo - table) / 8);
        }
        if (bound == std.math.maxInt(u64) or bound == 0) return error.BadScanFile;
        const n: usize = @intCast(bound);
        const raw = try f.alloc(gpa, table, n * 8);
        defer gpa.free(raw);
        const first_record = table + n * 8;
        const ph = first_record + (if (version >= 5) @as(u64, 8) else 0);
        const compressed = try f.int(i32, ph);
        const hgt = try f.int(i32, ph + 4);
        const w = try f.int(i32, ph + 8);
        const n_bytes = try f.int(i32, ph + 12);
        if (compressed != 0) return error.CompressedEbspUnsupported;
        if (hgt <= 0 or w <= 0) return error.BadScanFile;
        const px: i64 = @as(i64, hgt) * w;
        const info = &self.info;
        self.format = .oxford_ebsp;
        info.pat_h = @intCast(hgt);
        info.pat_w = @intCast(w);
        info.pixel = if (n_bytes == px) Pixel.u8 else if (n_bytes == 2 * px) Pixel.u16 else return error.BadScanFile;
        self.offsets = try gpa.alloc(u64, n);
        for (0..n) |i| {
            const o = std.mem.readInt(i64, raw[i * 8 ..][0..8], .little);
            self.offsets[i] = if (o <= 0) 0 else @as(u64, @intCast(o)) + hdr;
        }
        // map shape from the beam positions in the footers of the first, second and last
        // pattern of the scan (version 2 on): a row-major scan ends at the far corner
        if (version >= 2 and n > 1 and self.offsets[0] != 0 and self.offsets[1] != 0 and self.offsets[n - 1] != 0) blk: {
            const nb: u64 = @intCast(n_bytes);
            // the footer: a flag and, when set, the beam x; then the same for y
            const foot0 = self.offsets[0] + nb;
            if (try f.int(u8, foot0) == 0) break :blk;
            if (try f.int(u8, foot0 + 9) == 0) break :blk;
            const xo: u64 = 1;
            const yo: u64 = 10;
            if (self.offsets[n - 1] + nb + yo + 8 > f.size) break :blk;
            const x0: f64 = @bitCast(try f.int(u64, foot0 + xo));
            const y0: f64 = @bitCast(try f.int(u64, foot0 + yo));
            const x1: f64 = @bitCast(try f.int(u64, self.offsets[1] + nb + xo));
            const xl: f64 = @bitCast(try f.int(u64, self.offsets[n - 1] + nb + xo));
            const yl: f64 = @bitCast(try f.int(u64, self.offsets[n - 1] + nb + yo));
            const step = @abs(x1 - x0);
            if (!(step > 0) or !std.math.isFinite(xl) or !std.math.isFinite(yl)) break :blk;
            const fx = @round(@abs(xl - x0) / step) + 1;
            const fy = @round(@abs(yl - y0) / step) + 1;
            if (!(fx >= 1 and fy >= 1 and fx * fy < 1e12)) break :blk;
            const nx: u64 = @intFromFloat(fx);
            const ny: u64 = @intFromFloat(fy);
            if (nx * ny == n) {
                info.cols = @intCast(nx);
                info.rows = @intCast(ny);
                info.step_x_um = step;
                info.step_y_um = step;
            }
        }
        if (hints.rows != 0 and hints.cols != 0) {
            info.rows = hints.rows;
            info.cols = hints.cols;
        }
        if (info.rows == 0) {
            info.rows = 1;
            info.cols = @intCast(n);
        }
        if (@as(u64, info.rows) * info.cols > n) return error.ScanShapeMismatch;
    }

    // ── NORDIF ──

    fn openNordif(self: *Scan, hints: Hints) !void {
        const gpa = self.gpa;
        const info = &self.info;
        self.format = .nordif;
        info.pixel = .u8;
        if (self.files.len > 1) {
            const sf = &self.files[1];
            const text = try sf.alloc(gpa, 0, @intCast(@min(sf.size, 1 << 20)));
            defer gpa.free(text);
            var section: []const u8 = "";
            var lines = std.mem.tokenizeAny(u8, text, "\r\n");
            while (lines.next()) |line| {
                if (line.len > 0 and line[0] == '[') {
                    section = std.mem.trim(u8, line, "[]\t ");
                    continue;
                }
                var cols = std.mem.splitScalar(u8, line, '\t');
                const key = cols.next() orelse continue;
                const val = std.mem.trim(u8, cols.next() orelse continue, " ");
                if (std.mem.eql(u8, section, "Acquisition settings") and std.mem.eql(u8, key, "Resolution")) {
                    // width x height
                    const pair = splitX(val) orelse continue;
                    info.pat_w = pair[0];
                    info.pat_h = pair[1];
                } else if (std.mem.eql(u8, section, "Area") and std.mem.eql(u8, key, "Number of samples")) {
                    // rows x columns
                    const pair = splitX(val) orelse continue;
                    info.rows = pair[0];
                    info.cols = pair[1];
                } else if (std.mem.eql(u8, section, "Area") and std.mem.eql(u8, key, "Step size")) {
                    info.step_x_um = std.fmt.parseFloat(f64, val) catch info.step_x_um;
                    info.step_y_um = info.step_x_um;
                } else if (std.mem.eql(u8, section, "Microscope") and std.mem.eql(u8, key, "Accelerating voltage")) {
                    info.voltage_kv = std.fmt.parseFloat(f64, val) catch info.voltage_kv;
                } else if (std.mem.eql(u8, section, "Microscope") and std.mem.eql(u8, key, "Tilt angle")) {
                    info.sample_tilt_deg = std.fmt.parseFloat(f64, val) catch info.sample_tilt_deg;
                } else if (std.mem.eql(u8, section, "Detector angles") and std.mem.eql(u8, key, "Elevation")) {
                    info.detector_tilt_deg = std.fmt.parseFloat(f64, val) catch info.detector_tilt_deg;
                }
            }
        }
        if (hints.rows != 0 and hints.cols != 0) {
            info.rows = hints.rows;
            info.cols = hints.cols;
        }
        if (hints.pat_h != 0 and hints.pat_w != 0) {
            info.pat_h = hints.pat_h;
            info.pat_w = hints.pat_w;
        }
        if (info.rows == 0 or info.pat_h == 0) return error.ScanShapeUnknown;
        if (self.count() * self.patternBytes() > self.files[0].size) return error.ScanShapeMismatch;
    }
};

fn validStep(x: f64) bool {
    return std.math.isFinite(x) and x > 0;
}

fn sameStep(a: f64, b: f64) bool {
    return @abs(a - b) <= @max(@max(a, b) * 1e-6, @min(5e-6, @min(a, b) * 0.01));
}

fn splitX(val: []const u8) ?[2]u32 {
    const x = std.mem.indexOfScalar(u8, val, 'x') orelse return null;
    const a = std.fmt.parseInt(u32, std.mem.trim(u8, val[0..x], " "), 10) catch return null;
    const b = std.fmt.parseInt(u32, std.mem.trim(u8, val[x + 1 ..], " "), 10) catch return null;
    return .{ a, b };
}

fn extension(path: []const u8) []const u8 {
    const dot = std.mem.lastIndexOfScalar(u8, path, '.') orelse return "";
    return path[dot + 1 ..];
}

fn eqlIgnoreCase(a: []const u8, b: []const u8) bool {
    return std.ascii.eqlIgnoreCase(a, b);
}

test "the nickel scan opens with its metadata" {
    const gpa = std.testing.allocator;
    const s = Scan.open(gpa, &.{"testdata/nickel/patterns_v2.h5"}, .{}) catch |e| switch (e) {
        error.FileNotFound => return, // run from the repo root
        else => return e,
    };
    defer s.close();
    try std.testing.expectEqual(Format.h5_kikuchipy, s.format);
    try std.testing.expectEqual(@as(u32, 55), s.info.rows);
    try std.testing.expectEqual(@as(u32, 75), s.info.cols);
    try std.testing.expectEqual(@as(u32, 60), s.info.pat_h);
    try std.testing.expectEqual(Pixel.u8, s.info.pixel);
    try std.testing.expectEqual(@as(f64, 1.5), s.info.step_x_um);
    try std.testing.expectEqual(@as(f64, 70), s.info.sample_tilt_deg);
    try std.testing.expectApproxEqAbs(@as(f64, 0.42326006615963596), s.info.pc[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5020740783903902), s.info.pc[2], 1e-12);
    try std.testing.expectEqual(@as(f64, 20), s.info.voltage_kv);
    try std.testing.expectEqual(@as(usize, 3600), s.background.len);
    const buf = try gpa.alloc(u8, 3 * 3600);
    defer gpa.free(buf);
    try s.read(276, 3, buf);
    var sum: u64 = 0;
    for (buf) |v| sum += v;
    try std.testing.expectEqual(@as(u64, 1571367), sum);
    // scattered cells equal the same cells read one at a time
    const some = try gpa.alloc(u8, 3 * 3600);
    defer gpa.free(some);
    try s.readCells(&.{ 3, 277, 4000 }, some);
    try s.read(277, 1, buf[0..3600]);
    try std.testing.expectEqualSlices(u8, buf[0..3600], some[3600..7200]);
    try s.read(4000, 1, buf[0..3600]);
    try std.testing.expectEqualSlices(u8, buf[0..3600], some[7200..]);
    try std.testing.expectError(error.CellsNotAscending, s.readCells(&.{ 5, 4 }, some[0..7200]));
}

test "Oxford's detector orientation as the geometry's rotation, and back" {
    // an AZtec acquisition's (radians): 3.40° from the nominal (0, 90°, 0)
    const eu: rot.Vec3 = .{ 6.266744, 1.6277858, 6.2817445 };
    const r = oxfordDetectorRot(eu).?;
    const want = [_]f64{ 0.0163958, 0.05700002, -0.00097189 };
    for (want, r) |a, b| try std.testing.expectApproxEqAbs(a, b, 1e-6);
    const back = oxfordDetectorEuler(r);
    for (eu, back) |a, b| try std.testing.expectApproxEqAbs(a, b, 1e-6);
    // the nominal itself is no rotation, and angles far from it are not read
    for (oxfordDetectorRot(.{ 0, std.math.pi / 2.0, 0 }).?) |v| try std.testing.expectApproxEqAbs(0, v, 1e-12);
    try std.testing.expect(oxfordDetectorRot(.{ 0, std.math.degreesToRadians(20.0), 0 }) == null);
}
