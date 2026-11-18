//! Standard EDAX ANG scalar/grid companion for UP patterns; no orientations or phases.
//! Positions and steps are micrometres. Validate x-fast/y-next acquisition order relative
//! to the first position; engine geometry remains scan-local, with no origin fit or reorder.
const std = @import("std");
const File = @import("io/file.zig").File;
const Allocator = std.mem.Allocator;

pub const Metadata = struct {
    rows: u32,
    cols: u32,
    step_x_um: f64,
    step_y_um: f64,
    iq: []f32,
    pc: ?[3]f64,
    pc_note: []const u8,
};

const Lines = struct {
    f: *File,
    buf: [65536]u8 = undefined,
    pos: usize = 0,
    len: usize = 0,
    offset: u64 = 0,
    line: std.ArrayList(u8) = .empty,
    fn next(self: *Lines, gpa: Allocator) !?[]const u8 {
        self.line.clearRetainingCapacity();
        while (true) {
            if (self.pos == self.len) {
                if (self.offset == self.f.size) return if (self.line.items.len > 0) self.line.items else null;
                self.len = @intCast(@min(self.buf.len, self.f.size - self.offset));
                try self.f.read(self.offset, self.buf[0..self.len]);
                self.offset += self.len;
                self.pos = 0;
            }
            const remaining = self.buf[self.pos..self.len];
            const end = std.mem.indexOfScalar(u8, remaining, '\n') orelse remaining.len;
            if (self.line.items.len + end > 1 << 20) return error.ANGLineTooLong;
            try self.line.appendSlice(gpa, remaining[0..end]);
            self.pos += end;
            if (self.pos < self.len) {
                self.pos += 1;
                return self.line.items;
            }
        }
    }
};

fn eq(a: []const u8, b: []const u8) bool {
    return std.ascii.eqlIgnoreCase(a, b);
}
fn wordByte(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c == '_';
}
fn micro(value: []const u8) bool {
    const v = std.mem.trim(u8, value, " \t[]()");
    for ([_][]const u8{ "um", "µm", "μm", "micron", "microns", "micrometer", "micrometers", "micrometre", "micrometres" }) |unit| if (eq(v, unit)) return true;
    return false;
}
fn positive(v: []const u8) !f64 {
    const x = std.fmt.parseFloat(f64, v) catch return error.InvalidANGStep;
    if (!std.math.isFinite(x) or x <= 0) return error.InvalidANGStep;
    return x;
}
fn dimension(v: []const u8) !u32 {
    const n = std.fmt.parseInt(u32, v, 10) catch return error.InvalidANGDimensions;
    if (n == 0) return error.InvalidANGDimensions;
    return n;
}
fn checkCoordinate(actual: f64, expected: f64, origin: f64, travel: f64, step: f64) !void {
    if (!std.math.isFinite(actual) or !std.math.isFinite(expected)) return error.ANGCoordinatesNotRowMajor;
    const residual = @abs((actual - origin) - travel);
    const strict = @max(step * 1e-6, @min(5e-6, step * 0.01));
    if (residual <= strict) return; // also accepts well-resolved f64 coordinates at large origins
    // EDAX may compute positions in float32 and print five decimal places. Account
    // for origin, point, step/multiply rounding, plus two half decimal quanta.
    // This per-cell budget never grows a decimal allowance with the pixel index.
    const budget = @min(1e-5, step * 0.01) + 0x1p-24 * (@abs(origin) + @abs(expected) + 2 * @abs(travel));
    if (!std.math.isFinite(budget) or budget >= step / 8) return error.ANGCoordinatePrecisionInsufficient;
    if (residual > budget) return error.ANGCoordinatesNotRowMajor;
}

const PcDecision = struct { value: ?[3]f64 = null, note: []const u8 = "" };
/// The pattern centre in the header, and who wrote the file: what decides its convention. EDAX
/// (OIM) measures x and y in fractions of the pattern's width and height, y from the bottom, and z
/// in fractions of its smaller side; EMSphInx, and zebsdsolve up to 0.2.1, scale y and z by the
/// width. On a square pattern the two agree.
const PcHeader = struct {
    raw: [3]?f64 = .{ null, null, null },
    width_producer: bool = false,
    edax_producer: bool = false,
    software_seen: bool = false,
    header_start: bool = false,
    numeric_version: bool = false,

    fn header(self: *PcHeader, key: []const u8, value: []const u8) !void {
        const index: ?usize = if (eq(key, "x-star")) 0 else if (eq(key, "y-star")) 1 else if (eq(key, "z-star")) 2 else null;
        if (index) |k| {
            if (self.raw[k] != null) return error.DuplicateANGPatternCentre;
            const x = std.fmt.parseFloat(f64, value) catch return error.InvalidANGPatternCentre;
            if (!std.math.isFinite(x)) return error.InvalidANGPatternCentre;
            self.raw[k] = x;
        } else if (eq(key, "SOFTWARE")) {
            self.software_seen = true;
            var words = std.mem.tokenizeAny(u8, value, " \t");
            const producer = words.next() orelse return;
            if (eq(producer, "EDAX") or eq(producer, "OIM")) self.edax_producer = true;
            if (eq(producer, "EMSphInx")) self.width_producer = true;
            // (zebsdsolve names EDAX's convention after its version from 0.2.2; before, it scaled by the width)
            if (eq(producer, "zebsdsolve")) {
                if (std.mem.indexOf(u8, value, "EDAX") != null) self.edax_producer = true else self.width_producer = true;
            }
        } else if (eq(key, "HEADER") and eq(value, "Start")) {
            self.header_start = true;
        } else if (eq(key, "VERSION")) {
            _ = std.fmt.parseInt(u32, value, 10) catch return;
            self.numeric_version = true;
        }
    }

    fn resolve(self: PcHeader, width: u32, height: u32) !PcDecision {
        var present: usize = 0;
        for (self.raw) |v| present += @intFromBool(v != null);
        if (present == 0) return .{};
        if (present != 3) return error.IncompleteANGPatternCentre;
        const raw: [3]f64 = .{ self.raw[0].?, self.raw[1].?, self.raw[2].? };
        if (raw[0] == 0 and raw[1] == 0 and raw[2] == 0) return .{}; // producer's uncalibrated sentinel
        if (raw[2] <= 0 or width == 0 or height == 0) return error.InvalidANGPatternCentre;
        // who wrote it: the SOFTWARE line, else OIM's own header lines; on a rectangular pattern an
        // unknown writer's convention is not guessed at
        const conflict = self.width_producer and self.edax_producer;
        if (conflict and width != height) return .{ .note = "ANG pattern centre not applied: conflicting writers named" };
        const width_scaled = self.width_producer and !self.edax_producer;
        const edax = self.edax_producer or (!self.width_producer and !self.software_seen and self.header_start and self.numeric_version);
        if (!width_scaled and !edax and width != height) return .{ .note = "ANG pattern centre not applied: its convention is unknown for a rectangular pattern" };
        const w: f64 = @floatFromInt(width);
        const h: f64 = @floatFromInt(height);
        const pc: [3]f64 = if (width_scaled) .{ raw[0], 1 - raw[1] * w / h, raw[2] * w / h } else .{ raw[0], 1 - raw[1], raw[2] * @min(w, h) / h };
        for (pc) |x| if (!std.math.isFinite(x)) return error.InvalidANGPatternCentre;
        if (!(pc[2] > 0)) return error.InvalidANGPatternCentre;
        return .{ .value = pc, .note = if (width_scaled) "ANG pattern centre: width-scaled (EMSphInx, zebsdsolve 0.2.1 and before)" else if (edax) "ANG pattern centre: EDAX convention" else "ANG pattern centre: square pattern, the conventions agree" };
    }
};

pub fn load(gpa: Allocator, file: *File, expected_count: u64, up_steps: ?[2]f64, pattern_width: u32, pattern_height: u32) !Metadata {
    var lines: Lines = .{ .f = file };
    defer lines.line.deinit(gpa);
    var rows: u32 = 0;
    var odd: u32 = 0;
    var even: u32 = 0;
    var dx: f64 = 0;
    var dy: f64 = 0;
    var square = false;
    var pc_header: PcHeader = .{};
    var iq: []f32 = &.{};
    errdefer gpa.free(iq);
    var count: usize = 0;
    var x0: f64 = 0;
    var y0: f64 = 0;
    while (try lines.next(gpa)) |raw| {
        const line = std.mem.trim(u8, raw, " \t\r");
        if (line.len == 0) continue;
        if (line[0] == '#') {
            if (count != 0) return error.InvalidANGRecord;
            const content = std.mem.trim(u8, line[1..], " \t");
            var at: usize = 0;
            var tem = false;
            while (at < content.len) {
                if (!wordByte(content[at])) {
                    at += 1;
                    continue;
                }
                const start = at;
                while (at < content.len and wordByte(content[at])) at += 1;
                const word = content[start..at];
                if (eq(word, "ACOM") or (tem and eq(word, "data"))) return error.UnsupportedANGConvention;
                tem = eq(word, "TEM");
            }
            var pair = std.mem.tokenizeAny(u8, content, ": \t");
            const key = pair.next() orelse continue;
            const value = std.mem.trim(u8, pair.rest(), ": \t");
            try pc_header.header(key, value);
            if (eq(key, "GRID")) {
                if (square) return error.DuplicateANGMetadata;
                if (!eq(value, "SqrGrid")) return error.UnsupportedANGGrid;
                square = true;
            } else if (eq(key, "NROWS")) {
                if (rows != 0) return error.DuplicateANGMetadata;
                rows = try dimension(value);
            } else if (eq(key, "NCOLS_ODD")) {
                if (odd != 0) return error.DuplicateANGMetadata;
                odd = try dimension(value);
            } else if (eq(key, "NCOLS_EVEN")) {
                if (even != 0) return error.DuplicateANGMetadata;
                even = try dimension(value);
            } else if (eq(key, "XSTEP")) {
                if (dx != 0) return error.DuplicateANGMetadata;
                dx = try positive(value);
            } else if (eq(key, "YSTEP")) {
                if (dy != 0) return error.DuplicateANGMetadata;
                dy = try positive(value);
            } else if (eq(key, "COLUMN_HEADERS") or eq(key, "COLUMN_UNITS")) {
                return error.UnsupportedANGColumns;
            } else if (eq(key, "SCAN_UNIT") or eq(key, "SCANUNIT") or eq(key, "XUNITS") or eq(key, "YUNITS")) {
                if (!micro(value)) return error.UnsupportedANGUnits;
            }
            continue;
        }
        if (count == 0) {
            if (!square or rows == 0 or odd == 0 or even == 0) return error.ANGGridMetadataRequired;
            if (odd != even) return error.UnsupportedANGGrid;
            if (!(dx > 0) or !(dy > 0)) return error.InvalidANGStep;
            if (@as(u64, rows) * odd != expected_count or expected_count > std.math.maxInt(usize)) return error.ANGPatternCountMismatch;
            iq = try gpa.alloc(f32, @intCast(expected_count));
        }
        if (count >= iq.len) return error.ANGPatternCountMismatch;
        // Legacy ANG: phi1 Phi phi2 x y IQ CI phase [SEM fit ...]. Only x/y/IQ
        // are scalar inputs; Euler angles and phases are never interpreted.
        var values = std.mem.tokenizeAny(u8, line, " \t");
        var k: usize = 0;
        var x: f64 = std.math.nan(f64);
        var y = x;
        var quality = x;
        while (values.next()) |v| : (k += 1) {
            if (k >= 3 and k <= 5) {
                const n = std.fmt.parseFloat(f64, v) catch return error.InvalidANGRecord;
                if (k == 3) x = n else if (k == 4) y = n else quality = n;
            }
        }
        if (k < 8) return error.InvalidANGRecord;
        if (count == 0) {
            x0 = x;
            y0 = y;
        }
        const xt = @as(f64, @floatFromInt(count % odd)) * dx;
        const yt = @as(f64, @floatFromInt(count / odd)) * dy;
        try checkCoordinate(x, x0 + xt, x0, xt, dx);
        try checkCoordinate(y, y0 + yt, y0, yt, dy);
        if (up_steps) |steps| {
            const ux = @as(f64, @floatFromInt(count % odd)) * steps[0];
            const uy = @as(f64, @floatFromInt(count / odd)) * steps[1];
            checkCoordinate(x, x0 + ux, x0, ux, steps[0]) catch return error.UPANGStepMismatch;
            checkCoordinate(y, y0 + uy, y0, uy, steps[1]) catch return error.UPANGStepMismatch;
        }
        iq[count] = if (std.math.isFinite(quality) and @abs(quality) <= std.math.floatMax(f32)) @floatCast(quality) else std.math.nan(f32);
        count += 1;
    }
    if (count != expected_count or count == 0) return error.ANGPatternCountMismatch;
    const pc = try pc_header.resolve(pattern_width, pattern_height);
    return .{ .rows = rows, .cols = odd, .step_x_um = dx, .step_y_um = dy, .iq = iq, .pc = pc.value, .pc_note = pc.note };
}
