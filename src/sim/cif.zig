//! Crystals from CIF files, taken into the International Tables' standard setting the
//! simulation works in: the space group's number, the conventional cell and the sites of the
//! asymmetric unit.
//!
//! The symmetry is taken, in this order of trust: from the operators the file lists (or its
//! Hall symbol), which are authoritative; from the group it names (IT number or
//! Hermann–Mauguin symbol), whose setting is then the tabulated one that leaves the sites in
//! place and gives the most conventional cell; or, when it declares nothing (or P1), from the
//! sites alone, as the largest tabulated group that leaves them in place (a walk down the
//! subgroups by order). The change of basis to the standard setting, (P, p) with det P = +1 (a
//! mirror would take a crystal to its enantiomorph), is looked up among every setting of every
//! group whose cell keeps its system's metric, a table built from the Hall symbols the first
//! time it is needed; and searched for, over the unimodular P with entries in {−1, 0, 1} and
//! the origin shifts in 24ths, otherwise. Operators on rhombohedral axes are taken to hexagonal
//! ones first. A file that disagrees with itself is refused, with the reason, rather than
//! guessed at. (The behaviour of ebsdsim's reader, github.com/ZacharyVarley/ebsdsim.)
const std = @import("std");
const group = @import("group.zig");
const tables = @import("tables.zig");

const Op = group.Op;
const OpList = group.OpList;
const Mat = group.Mat;
const I3 = group.I3;
const DEN = group.DEN;
const MAX = group.MAX_OPS;

/// how close (fractional, nearest image) two positions are to be one
pub const default_eps = 1e-5;
/// a site's isotropic B when the file gives none or zero, Å² (zero is unphysical at any
/// temperature, and the absorption then selects pathologically many beams)
pub const default_b_iso = 0.5;
const eight_pi2 = 8.0 * std.math.pi * std.math.pi;

/// hexagonal axes from rhombohedral (obverse): its columns are a, b, c (hexagonal) on the
/// rhombohedral axes
pub const p_hex_from_rhomb: Mat = .{ 1, 0, 1, -1, 1, 1, 0, -1, 1 };

pub const Site = struct {
    label: []const u8,
    symbol: []const u8,
    z: u32,
    xyz: [3]f64,
    occupancy: f64,
    /// isotropic U, Å²
    u_iso: f64,
};

/// The change of basis applied: x′ = P⁻¹ (x − p), p in 24ths.
pub const Setting = struct {
    number: u32,
    P: Mat = I3,
    p: [3]i32 = .{ 0, 0, 0 },
    /// the file's axes were rhombohedral (taken to hexagonal before P)
    rhombohedral: bool = false,
    note: []const u8 = "",

    pub fn standard(s: Setting) bool {
        return std.mem.eql(i32, &s.P, &I3) and s.p[0] == 0 and s.p[1] == 0 and s.p[2] == 0;
    }

    /// "(a′,b′,c′) = (a | -c | b);  origin shift = (0,1/4,0)"
    pub fn describe(s: Setting, gpa: std.mem.Allocator) ![]u8 {
        var out: std.Io.Writer.Allocating = .init(gpa);
        errdefer out.deinit();
        const w = &out.writer;
        try w.writeAll("(a',b',c') = (");
        for (0..3) |j| {
            if (j > 0) try w.writeAll(" | ");
            var first = true;
            for (0..3) |i| {
                const c = s.P[i * 3 + j];
                if (c == 0) continue;
                if (c < 0) try w.writeByte('-') else if (!first) try w.writeByte('+');
                if (@abs(c) != 1) try w.print("{d}", .{@abs(c)});
                try w.writeByte("abc"[i]);
                first = false;
            }
        }
        try w.writeAll(");  origin shift = (");
        for (s.p, 0..) |v, i| {
            if (i > 0) try w.writeByte(',');
            if (v == 0) {
                try w.writeByte('0');
                continue;
            }
            const g: i32 = @intCast(std.math.gcd(@as(u32, @intCast(v)), @as(u32, DEN)));
            const num = @divExact(v, g);
            const den = @divExact(@as(i32, DEN), g);
            if (den == 1) try w.print("{d}", .{num}) else try w.print("{d}/{d}", .{ num, den });
        }
        try w.writeByte(')');
        return out.toOwnedSlice();
    }
};

/// A crystal as read, in the standard setting.
pub const Structure = struct {
    block: []const u8,
    number: u32,
    /// Å, degrees
    cell: [6]f64,
    sites: []Site,
    setting: Setting,
    /// where the symmetry came from
    provenance: []const u8,
    /// what the file declared
    declared_it: ?u32,
    declared_hall: ?[]const u8,
    declared_hm: ?[]const u8,
    /// the cell and the number of sites as the file has them
    cell_as_read: [6]f64,
    sites_as_read: usize,
    /// the sites given the default B (none in the file, or not positive), and whether the file
    /// had thermal parameters at all
    defaulted_b: usize,
    thermal_tags: bool,
};

const System = enum { triclinic, monoclinic, orthorhombic, tetragonal, trigonal, hexagonal, cubic };

fn systemOf(n: u32) System {
    return if (n <= 2) .triclinic else if (n <= 15) .monoclinic else if (n <= 74) .orthorhombic else if (n <= 142) .tetragonal else if (n <= 167) .trigonal else if (n <= 194) .hexagonal else .cubic;
}

fn origin1Hall(n: u32) ?[]const u8 {
    for (tables.origin1_hall) |e| if (e.sg == n) return e.hall;
    return null;
}

fn rhombHall(n: u32) ?[]const u8 {
    for (tables.rhomb_hall) |e| if (e.sg == n) return e.hall;
    return null;
}

// ─────────────────────────── the CIF text ───────────────────────────

/// the tags read; the rest are skipped
const wanted = [_][]const u8{
    "_cell_length_a",                  "_cell_length_b",                    "_cell_length_c",
    "_cell_angle_alpha",               "_cell_angle_beta",                  "_cell_angle_gamma",
    "_space_group_it_number",          "_symmetry_int_tables_number",       "_space_group_name_hall",
    "_symmetry_space_group_name_hall", "_space_group_name_h-m_alt",         "_symmetry_space_group_name_h-m",
    "_space_group_name_h-m_ref",       "_space_group_name_h-m_full",        "_space_group_symop_operation_xyz",
    "_symmetry_equiv_pos_as_xyz",      "_space_group_symop.operation_xyz",  "_atom_site_label",
    "_atom_site_type_symbol",          "_atom_site_fract_x",                "_atom_site_fract_y",
    "_atom_site_fract_z",              "_atom_site_occupancy",              "_atom_site_u_iso_or_equiv",
    "_atom_site_b_iso_or_equiv",
};

fn isWanted(tag: []const u8) bool {
    for (wanted) |t| if (std.mem.eql(u8, t, tag)) return true;
    return false;
}

const Value = union(enum) { one: []const u8, column: []const []const u8 };

/// A data block: its wanted tags (lower case), each a value or a loop's column.
pub const Block = struct {
    name: []const u8,
    tags: std.StringArrayHashMapUnmanaged(Value) = .empty,

    /// The first of the tags with a value that says something (not empty, ? or .).
    fn str(self: *const Block, names: []const []const u8) ?[]const u8 {
        for (names) |n| {
            const v = self.tags.get(n) orelse continue;
            const s0 = switch (v) {
                .one => |s| s,
                .column => |c| if (c.len == 0) continue else c[0],
            };
            const s = std.mem.trim(u8, s0, " \t\r\n");
            if (s.len == 0 or std.mem.eql(u8, s, "?") or std.mem.eql(u8, s, ".")) continue;
            return s;
        }
        return null;
    }

    /// The first of the tags present, as a column (a single value: a column of one).
    fn loop(self: *const Block, gpa: std.mem.Allocator, names: []const []const u8) !?[]const []const u8 {
        for (names) |n| {
            const v = self.tags.get(n) orelse continue;
            return switch (v) {
                .column => |c| c,
                .one => |s| try gpa.dupe([]const u8, &.{s}),
            };
        }
        return null;
    }
};

const Text = struct {
    s: []const u8,

    fn isWs(c: u8) bool {
        return c == ' ' or c == '\t' or c == '\n';
    }

    fn skipWs(t: Text, from: usize) usize {
        var i = from;
        while (i < t.s.len) {
            const c = t.s[i];
            if (isWs(c)) {
                i += 1;
            } else if (c == '#') {
                while (i < t.s.len and t.s[i] != '\n') i += 1;
            } else break;
        }
        return i;
    }

    /// Whether a keyword (ASCII, any case) starts at i; loop_ and global_ also need a
    /// delimiter after them.
    fn kw(t: Text, i: usize, k: []const u8) bool {
        if (i + k.len > t.s.len) return false;
        for (k, 0..) |b, j| if (std.ascii.toLower(t.s[i + j]) != b) return false;
        if (std.mem.eql(u8, k, "data_") or std.mem.eql(u8, k, "save_")) return true;
        return i + k.len == t.s.len or isWs(t.s[i + k.len]) or t.s[i + k.len] == '#';
    }

    fn endTag(t: Text, from: usize) usize {
        var j = from + 1;
        while (j < t.s.len and !isWs(t.s[j]) and t.s[j] != '#') j += 1;
        return j;
    }

    /// A value: a ;-delimited text field (at a line's start), a quoted string (closed by its
    /// quote before white space) or a bare word.
    fn value(t: Text, from: usize) struct { v: []const u8, i: usize } {
        const s = t.s;
        var i = t.skipWs(from);
        if (i >= s.len) return .{ .v = "", .i = i };
        const c = s[i];
        if (c == ';' and (i == 0 or s[i - 1] == '\n')) {
            i += 1;
            if (i < s.len and s[i] == '\n') i += 1;
            const start = i;
            while (i < s.len) : (i += 1) {
                if (s[i] == '\n' and i + 1 < s.len and s[i + 1] == ';') return .{ .v = s[start..i], .i = i + 2 };
            }
            return .{ .v = s[start..], .i = s.len };
        }
        if (c == '\'' or c == '"') {
            i += 1;
            const start = i;
            while (i < s.len) : (i += 1) {
                if (s[i] == c and (i + 1 >= s.len or isWs(s[i + 1]))) return .{ .v = s[start..i], .i = i + 1 };
            }
            return .{ .v = s[start..], .i = s.len };
        }
        var j = i;
        while (j < s.len and !isWs(s[j])) j += 1;
        return .{ .v = s[i..j], .i = j };
    }

    fn atKeyword(t: Text, i: usize) bool {
        const c = t.s[i];
        return std.mem.indexOfScalar(u8, "lLdDgGsS", c) != null and (t.kw(i, "loop_") or t.kw(i, "data_") or t.kw(i, "save_") or t.kw(i, "global_"));
    }

    fn skipLoopValues(t: Text, from: usize) usize {
        var i = from;
        while (i < t.s.len) {
            i = t.skipWs(i);
            if (i >= t.s.len or t.s[i] == '_' or t.atKeyword(i)) break;
            i = t.value(i).i;
        }
        return i;
    }

    fn skipLoop(t: Text, from: usize) usize {
        var i = from;
        while (i < t.s.len) {
            i = t.skipWs(i);
            if (i >= t.s.len or t.s[i] != '_') break;
            i = t.endTag(i);
        }
        return t.skipLoopValues(i);
    }

    /// A save_ frame's body (its "save_" read), to the bare save_ that ends it.
    fn skipSave(t: Text, from: usize) usize {
        var i = from;
        while (i < t.s.len) {
            i = t.skipWs(i);
            if (i >= t.s.len) break;
            if (t.kw(i, "save_")) {
                var j = i + 5;
                if (j >= t.s.len or isWs(t.s[j]) or t.s[j] == '#') return j;
                while (j < t.s.len and !isWs(t.s[j]) and t.s[j] != '#') j += 1;
                i = j;
                continue;
            }
            if (t.s[i] == '_') {
                i = t.value(t.endTag(i)).i;
            } else if (t.kw(i, "loop_")) {
                i = t.skipLoop(i + 5);
            } else if (t.kw(i, "data_")) {
                return i;
            } else i = t.value(i).i;
        }
        return i;
    }
};

fn lowerDup(gpa: std.mem.Allocator, s: []const u8) ![]u8 {
    const o = try gpa.alloc(u8, s.len);
    for (s, 0..) |c, i| o[i] = std.ascii.toLower(c);
    return o;
}

/// The data blocks of a CIF, in their order (a repeated name: the later block, in the first's
/// place).
pub fn parse(gpa: std.mem.Allocator, text0: []const u8) ![]Block {
    // line ends to \n
    const buf = try gpa.alloc(u8, text0.len);
    var n: usize = 0;
    var k: usize = 0;
    while (k < text0.len) : (k += 1) {
        if (text0[k] == '\r') {
            buf[n] = '\n';
            if (k + 1 < text0.len and text0[k + 1] == '\n') k += 1;
        } else buf[n] = text0[k];
        n += 1;
    }
    const t: Text = .{ .s = buf[0..n] };
    var blocks: std.ArrayList(Block) = .empty;
    var i: usize = 0;
    while (true) {
        i = t.skipWs(i);
        if (i >= t.s.len) break;
        if (t.kw(i, "data_")) {
            i += 5;
            var j = i;
            while (j < t.s.len and !Text.isWs(t.s[j]) and t.s[j] != '#') j += 1;
            var blk: Block = .{ .name = t.s[i..j] };
            i = try parseBlock(gpa, t, j, &blk);
            for (blocks.items) |*b| {
                if (std.mem.eql(u8, b.name, blk.name)) {
                    b.* = blk;
                    break;
                }
            } else try blocks.append(gpa, blk);
            continue;
        }
        if (t.s[i] == '_') {
            i = t.value(t.endTag(i)).i;
        } else if (t.kw(i, "loop_")) {
            i = t.skipLoop(i + 5);
        } else if (t.kw(i, "save_")) {
            i = t.skipSave(i + 5);
        } else if (t.kw(i, "global_")) {
            i += 7;
        } else i = t.value(i).i;
    }
    return blocks.items;
}

fn parseBlock(gpa: std.mem.Allocator, t: Text, from: usize, blk: *Block) !usize {
    var i = from;
    while (i < t.s.len) {
        i = t.skipWs(i);
        if (i >= t.s.len) break;
        const c = t.s[i];
        if (c == '_') {
            const e = t.endTag(i);
            const tag = try lowerDup(gpa, t.s[i..e]);
            const v = t.value(e);
            i = v.i;
            if (isWanted(tag)) try blk.tags.put(gpa, tag, .{ .one = v.v });
            continue;
        }
        if ((c == 'l' or c == 'L') and t.kw(i, "loop_")) {
            i = try parseLoop(gpa, t, i + 5, blk);
            continue;
        }
        // the next data_, save_ or global_ ends the block
        if ((c == 'd' or c == 'D') and t.kw(i, "data_")) break;
        if ((c == 's' or c == 'S') and t.kw(i, "save_")) break;
        if ((c == 'g' or c == 'G') and t.kw(i, "global_")) break;
        i = t.value(i).i;
    }
    return i;
}

fn parseLoop(gpa: std.mem.Allocator, t: Text, from: usize, blk: *Block) !usize {
    var i = from;
    var tags: std.ArrayList([]u8) = .empty;
    while (i < t.s.len) {
        i = t.skipWs(i);
        if (i >= t.s.len or t.s[i] != '_') break;
        const e = t.endTag(i);
        try tags.append(gpa, try lowerDup(gpa, t.s[i..e]));
        i = e;
    }
    if (tags.items.len == 0) return i;
    var any = false;
    for (tags.items) |tag| any = any or isWanted(tag);
    if (!any) return t.skipLoopValues(i);
    const cols = try gpa.alloc(std.ArrayList([]const u8), tags.items.len);
    for (cols) |*c| c.* = .empty;
    var col: usize = 0;
    while (i < t.s.len) {
        i = t.skipWs(i);
        if (i >= t.s.len or t.s[i] == '_' or t.atKeyword(i)) break;
        const v = t.value(i);
        i = v.i;
        try cols[col].append(gpa, v.v);
        col = (col + 1) % tags.items.len;
    }
    // a short last row: the rest empty
    if (col != 0) while (col < tags.items.len) : (col += 1) try cols[col].append(gpa, "");
    for (tags.items, cols) |tag, c| if (isWanted(tag)) try blk.tags.put(gpa, tag, .{ .column = c.items });
    return i;
}

/// A CIF number: "3.481(2)", "-.027", "1e-3"; ?, . or nothing: null.
fn parseNumber(v0: ?[]const u8) !?f64 {
    const v1 = v0 orelse return null;
    const v = std.mem.trim(u8, std.mem.trim(u8, v1, " \t\r\n"), "'\"");
    if (v.len == 0 or std.mem.eql(u8, v, "?") or std.mem.eql(u8, v, ".")) return null;
    const s = std.mem.trim(u8, v, " \t\r\n");
    var i: usize = 0;
    if (i < s.len and (s[i] == '+' or s[i] == '-')) i += 1;
    const d0 = i;
    while (i < s.len and std.ascii.isDigit(s[i])) i += 1;
    const whole = i - d0;
    var frac: usize = 0;
    if (i < s.len and s[i] == '.') {
        i += 1;
        const f0 = i;
        while (i < s.len and std.ascii.isDigit(s[i])) i += 1;
        frac = i - f0;
    }
    if (whole == 0 and frac == 0) return error.BadNumber;
    if (i < s.len and (s[i] == 'e' or s[i] == 'E')) {
        var j = i + 1;
        if (j < s.len and (s[j] == '+' or s[j] == '-')) j += 1;
        const e0 = j;
        while (j < s.len and std.ascii.isDigit(s[j])) j += 1;
        if (j > e0) i = j;
    }
    const mant = s[0..i];
    var rest = std.mem.trimStart(u8, s[i..], " \t");
    if (rest.len > 0 and rest[0] == '(') {
        const close = std.mem.indexOfScalar(u8, rest, ')') orelse return error.BadNumber;
        if (close == 1) return error.BadNumber;
        for (rest[1..close]) |c| if (!std.ascii.isDigit(c)) return error.BadNumber;
        rest = std.mem.trimStart(u8, rest[close + 1 ..], " \t");
    }
    if (rest.len != 0) return error.BadNumber;
    // (Zig wants a digit on both sides of the point)
    var buf: [64]u8 = undefined;
    if (mant.len + 2 > buf.len) return error.BadNumber;
    var m: usize = 0;
    for (mant, 0..) |c, ci| {
        if (c == '.' and (ci == 0 or !std.ascii.isDigit(mant[ci - 1]))) {
            buf[m] = '0';
            m += 1;
        }
        buf[m] = c;
        m += 1;
        if (c == '.' and (ci + 1 == mant.len or !std.ascii.isDigit(mant[ci + 1]))) {
            buf[m] = '0';
            m += 1;
        }
    }
    return std.fmt.parseFloat(f64, buf[0..m]) catch error.BadNumber;
}

/// The element a site's type symbol or label names: "Fe3+" → Fe, "O1" → O, "NI" → Ni.
fn elementOf(label: []const u8) ?[]const u8 {
    var buf: [64]u8 = undefined;
    var n: usize = 0;
    for (label) |c| if (std.ascii.isAlphabetic(c) and n < buf.len) {
        buf[n] = c;
        n += 1;
    };
    const s = buf[0..n];
    // two-letter symbols first; a following lower-case letter is part of a longer name
    for ([_]usize{ 2, 1 }) |len| for (tables.symbols) |e| {
        if (e.len != len or s.len < len) continue;
        if (!std.ascii.eqlIgnoreCase(s[0..len], e)) continue;
        if (s.len == len or !std.ascii.isLower(s[len])) return e;
    };
    for ([_]usize{ 2, 1 }) |len| for (tables.symbols) |e| {
        if (e.len == len and s.len >= len and std.ascii.eqlIgnoreCase(s[0..len], e)) return e;
    };
    return null;
}

fn atomicNumber(symbol: []const u8) u32 {
    for (tables.symbols, 1..) |e, z| if (std.mem.eql(u8, e, symbol)) return @intCast(z);
    return 0;
}

// ─────────────────────────── geometry ───────────────────────────

/// The cell (Å, degrees) in the basis P: G′ = Pᵀ G P.
pub fn transformCell(cell: [6]f64, P: Mat) [6]f64 {
    const d = std.math.pi / 180.0;
    const a = cell[0];
    const b = cell[1];
    const c = cell[2];
    const ca = @cos(cell[3] * d);
    const cb = @cos(cell[4] * d);
    const cg = @cos(cell[5] * d);
    const G = [9]f64{ a * a, a * b * cg, a * c * cb, a * b * cg, b * b, b * c * ca, a * c * cb, b * c * ca, c * c };
    var G2: [9]f64 = undefined;
    for (0..3) |i| for (0..3) |j| {
        var s: f64 = 0;
        for (0..3) |k| for (0..3) |l| {
            s += @as(f64, @floatFromInt(P[k * 3 + i])) * G[k * 3 + l] * @as(f64, @floatFromInt(P[l * 3 + j]));
        };
        G2[i * 3 + j] = s;
    };
    const len = [3]f64{ @sqrt(@max(G2[0], 0)), @sqrt(@max(G2[4], 0)), @sqrt(@max(G2[8], 0)) };
    const ang = struct {
        fn f(g: f64, li: f64, lj: f64) f64 {
            return std.math.acos(std.math.clamp(g / (li * lj), -1.0, 1.0)) * 180.0 / std.math.pi;
        }
    }.f;
    return .{ len[0], len[1], len[2], ang(G2[5], len[1], len[2]), ang(G2[2], len[0], len[2]), ang(G2[1], len[0], len[1]) };
}

fn inverseF(m: Mat) [9]f64 {
    const a = group.adjugate(m);
    const dt: f64 = @floatFromInt(group.det(m));
    var o: [9]f64 = undefined;
    for (0..9) |i| o[i] = @as(f64, @floatFromInt(a[i])) / dt;
    return o;
}

fn wrap(x: f64) f64 {
    return x - @floor(x);
}

/// x′ = P⁻¹ (x − p), wrapped into [0, 1).
fn transformCoords(x: [3]f64, P: Mat, p: [3]i32) [3]f64 {
    const Pi = inverseF(P);
    const y = [3]f64{ x[0] - @as(f64, @floatFromInt(p[0])) / DEN, x[1] - @as(f64, @floatFromInt(p[1])) / DEN, x[2] - @as(f64, @floatFromInt(p[2])) / DEN };
    var o: [3]f64 = undefined;
    for (0..3) |i| o[i] = wrap(Pi[i * 3] * y[0] + Pi[i * 3 + 1] * y[1] + Pi[i * 3 + 2] * y[2]);
    return o;
}

fn image(o: Op, x: [3]f64) [3]f64 {
    const y = o.apply(x);
    return .{ wrap(y[0]), wrap(y[1]), wrap(y[2]) };
}

/// The distance between two positions' nearest images, fractional.
fn nearest(a: [3]f64, b: [3]f64) f64 {
    var s: f64 = 0;
    for (0..3) |i| {
        const d = wrap(a[i] - b[i] + 0.5) - 0.5;
        s += d * d;
    }
    return @sqrt(s);
}

// ─────────────────────────── the reader ───────────────────────────

const Entry = struct { sg: u32, P: Mat, p: [3]i32 };

const Family = enum { triclinic, monoclinic, orthorhombic, tetragonal, hexagonal_family, rhombohedral, cubic };

fn inFamily(f: Family, n: u32) bool {
    const s = systemOf(n);
    return switch (f) {
        .triclinic => s == .triclinic,
        .monoclinic => s == .monoclinic,
        .orthorhombic => s == .orthorhombic,
        .tetragonal => s == .tetragonal,
        .hexagonal_family => s == .trigonal or s == .hexagonal,
        .rhombohedral => s == .trigonal and rhombHall(n) != null,
        .cubic => s == .cubic,
    };
}

/// The crystal family a cell's metric implies (not its group's).
fn metricFamily(c: [6]f64) Family {
    const eq = struct {
        fn f(x: f64, y: f64) bool {
            return @abs(x - y) < 1e-3 * @max(1.0, @abs(y));
        }
    }.f;
    const a90 = struct {
        fn f(x: f64) bool {
            return @abs(x - 90) < 1e-3;
        }
    }.f;
    if (eq(c[0], c[1]) and eq(c[1], c[2]) and a90(c[3]) and a90(c[4]) and a90(c[5])) return .cubic;
    if (eq(c[0], c[1]) and a90(c[3]) and a90(c[4]) and @abs(c[5] - 120) < 1e-3) return .hexagonal_family;
    if (eq(c[0], c[1]) and eq(c[1], c[2]) and eq(c[3], c[4]) and eq(c[4], c[5]) and !a90(c[3])) return .rhombohedral;
    if (eq(c[0], c[1]) and a90(c[3]) and a90(c[4]) and a90(c[5])) return .tetragonal;
    if (a90(c[3]) and a90(c[4]) and a90(c[5])) return .orthorhombic;
    const oblique = @as(u32, @intFromBool(!a90(c[3]))) + @intFromBool(!a90(c[4])) + @intFromBool(!a90(c[5]));
    return if (oblique <= 1) .monoclinic else .triclinic;
}

/// Whether a basis change keeps a cell of the system in the system's metric (so that it is a
/// setting someone could deposit, not merely a valid conjugation).
fn metricOk(sys: System, c: [6]f64) bool {
    const eq = struct {
        fn f(x: f64, y: f64) bool {
            return @abs(x - y) < 1e-4;
        }
    }.f;
    const a90 = struct {
        fn f(x: f64) bool {
            return @abs(x - 90) < 1e-3;
        }
    }.f;
    return switch (sys) {
        .triclinic => true,
        .monoclinic => @as(u32, @intFromBool(!a90(c[3]))) + @intFromBool(!a90(c[4])) + @intFromBool(!a90(c[5])) <= 1,
        .orthorhombic => a90(c[3]) and a90(c[4]) and a90(c[5]),
        .tetragonal => eq(c[0], c[1]) and a90(c[3]) and a90(c[4]) and a90(c[5]),
        .trigonal, .hexagonal => eq(c[0], c[1]) and a90(c[3]) and a90(c[4]) and @abs(c[5] - 120) < 1e-4,
        .cubic => eq(c[0], c[1]) and eq(c[1], c[2]) and a90(c[3]) and a90(c[4]) and a90(c[5]),
    };
}

/// a cell of each system in its standard metric, deliberately unround
const generic_cell = [7][6]f64{
    .{ 5.1, 6.2, 7.3, 83.1, 97.4, 104.7 },
    .{ 5.1, 6.2, 7.3, 90.0, 100.4, 90.0 },
    .{ 5.1, 6.2, 7.3, 90.0, 90.0, 90.0 },
    .{ 5.1, 5.1, 7.3, 90.0, 90.0, 90.0 },
    .{ 5.1, 5.1, 7.3, 90.0, 90.0, 120.0 },
    .{ 5.1, 5.1, 7.3, 90.0, 90.0, 120.0 },
    .{ 5.1, 5.1, 5.1, 90.0, 90.0, 90.0 },
};

fn rotCode(W: Mat) ?i64 {
    var c: i64 = 0;
    for (W) |v| {
        if (v < -20 or v > 20) return null;
        c = c * 41 + (v + 20);
    }
    return c;
}

fn sortedHas(items: []const i64, c: i64) bool {
    var lo: usize = 0;
    var hi: usize = items.len;
    while (lo < hi) {
        const mid = (lo + hi) / 2;
        if (items[mid] < c) lo = mid + 1 else hi = mid;
    }
    return lo < items.len and items[lo] == c;
}

fn hashOf(fp: []const i64) u64 {
    return std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(fp));
}

const Tree = struct {
    orders: [230]u32,
    children: [230][]const u32,
    tops: [7][]const u32,
};

const Symmetry = struct {
    ops: ?[]const Op,
    it: ?u32,
    provenance: []const u8,
    /// authoritative operators (a symop loop or a Hall symbol)
    useful: bool,
};

const Atoms = struct {
    cell: [6]f64,
    labels: [][]const u8,
    species: [][]const u8,
    coords: [][3]f64,
    occ: []f64,
    u: []f64,
    thermal: bool,
};

pub const Reader = struct {
    /// everything the reader makes lives here (an arena: the caller frees it at once)
    arena: std.mem.Allocator,
    eps: f64 = default_eps,
    why_buf: [700]u8 = undefined,
    why_len: usize = 0,
    cands: ?[]const Mat = null,
    legal: ?[7][]const Mat = null,
    table: ?std.AutoHashMapUnmanaged(u64, Entry) = null,
    tree: ?Tree = null,
    std_cache: [230]?[]const Op = @splat(null),
    hm_index: ?std.StringHashMapUnmanaged(u32) = null,

    /// Why the last read failed (error.Symmetry).
    pub fn why(r: *const Reader) []const u8 {
        return r.why_buf[0..r.why_len];
    }

    fn fail(r: *Reader, comptime fmt: []const u8, args: anytype) error{Symmetry} {
        const s = std.fmt.bufPrint(&r.why_buf, fmt, args) catch r.why_buf[0..];
        r.why_len = s.len;
        return error.Symmetry;
    }

    fn stdOps(r: *Reader, n: u32) ![]const Op {
        if (r.std_cache[n - 1]) |o| return o;
        const g = try group.hallOps(tables.std_hall[n - 1]);
        const o = try r.arena.dupe(Op, g.slice());
        r.std_cache[n - 1] = o;
        return o;
    }

    fn hallSlice(r: *Reader, symbol: []const u8) ![]const Op {
        const g = group.hallOps(symbol) catch |err| switch (err) {
            error.BadHallSymbol => return r.fail("cannot parse the Hall symbol '{s}'", .{symbol}),
            error.NotAGroup => return r.fail("the Hall symbol '{s}' does not generate a space group", .{symbol}),
        };
        return r.arena.dupe(Op, g.slice());
    }

    fn closeOps(r: *Reader, ops: []const Op) !OpList {
        return group.close(ops) catch r.fail("the symmetry operations do not close to a space group (past {d} operators, or not crystallographic): they are probably not a group", .{MAX});
    }

    // ── the changes of basis ──

    /// Every integer matrix with entries in {−1, 0, 1} and determinant +1 (3480): the
    /// orthorhombic permutations, the monoclinic unique axes and cell choices, R obverse and
    /// reverse, and the mild non-standard cells deposited (a′ = a + b and the like); the
    /// identity first, then the sparsest.
    fn candidates(r: *Reader) ![]const Mat {
        if (r.cands) |c| return c;
        var list: std.ArrayList(Mat) = .empty;
        for (0..19683) |code| {
            var c = code;
            var M: Mat = undefined;
            var k: usize = 9;
            while (k > 0) {
                k -= 1;
                M[k] = @as(i32, @intCast(c % 3)) - 1;
                c /= 3;
            }
            if (group.det(M) == 1) try list.append(r.arena, M);
        }
        const by = struct {
            fn key(M: Mat) u32 {
                var s: u32 = 0;
                for (M) |v| s += @abs(v);
                return (if (std.mem.eql(i32, &M, &I3)) @as(u32, 0) else 100) + s;
            }
            fn lt(_: void, a: Mat, b: Mat) bool {
                return key(a) < key(b);
            }
        };
        std.mem.sort(Mat, list.items, {}, by.lt);
        r.cands = list.items;
        return list.items;
    }

    /// Per system, the basis changes that keep its cells in its metric.
    fn legalP(r: *Reader) ![7][]const Mat {
        if (r.legal) |l| return l;
        const c = try r.candidates();
        var out: [7][]const Mat = undefined;
        for (0..7) |s| {
            var l: std.ArrayList(Mat) = .empty;
            for (c) |P| if (metricOk(@enumFromInt(s), transformCell(generic_cell[s], P))) try l.append(r.arena, P);
            out[s] = l.items;
        }
        r.legal = out;
        return out;
    }

    /// Every setting of every group whose cell keeps its system's metric, by its operators:
    /// the standard one, its origin choice 1 where there is one, and their metric-legal basis
    /// changes. (Built in this order, the first to give a set of operators keeping it.)
    fn settingsTable(r: *Reader) !*std.AutoHashMapUnmanaged(u64, Entry) {
        if (r.table == null) {
            var t: std.AutoHashMapUnmanaged(u64, Entry) = .empty;
            const legal = try r.legalP();
            var fp: [MAX]i64 = undefined;
            for (1..231) |ni| {
                const n: u32 = @intCast(ni);
                const base0 = try r.stdOps(n);
                const Base = struct { ops: []const Op, P: Mat, p: [3]i32 };
                var bases: [2]Base = undefined;
                bases[0] = .{ .ops = base0, .P = I3, .p = .{ 0, 0, 0 } };
                var nb: usize = 1;
                if (origin1Hall(n)) |h| {
                    const o1 = try r.hallSlice(h);
                    if (try r.findP(o1, base0, I3)) |p1| {
                        bases[1] = .{ .ops = o1, .P = I3, .p = p1 };
                        nb = 2;
                    }
                }
                for (bases[0..nb]) |b| for (legal[@intFromEnum(systemOf(n))]) |Q| {
                    const Qinv = group.adjugate(Q);
                    var ops: OpList = .{ .n = b.ops.len };
                    for (b.ops, 0..) |o, i| ops.ops[i] = o.transform(Q, Qinv, .{ 0, 0, 0 });
                    const f = ops.fingerprint(&fp) catch continue;
                    const slot = try t.getOrPut(r.arena, hashOf(f));
                    if (slot.found_existing) continue;
                    // (Q⁻¹, 0) undoes the basis change, then (P0, p0) reaches the standard
                    const pv = group.matVec(Qinv, b.p);
                    slot.value_ptr.* = .{ .sg = n, .P = group.matMul(Qinv, b.P), .p = .{ @mod(pv[0], DEN), @mod(pv[1], DEN), @mod(pv[2], DEN) } };
                };
            }
            r.table = t;
        }
        return &r.table.?;
    }

    /// The tabulated setting a closed operator set is, checked.
    fn lookup(r: *Reader, ops: *const OpList) !?Entry {
        const t = try r.settingsTable();
        var fp: [MAX]i64 = undefined;
        const f = ops.fingerprint(&fp) catch return null;
        const e = t.get(hashOf(f)) orelse return null;
        const moved = try ops.transformed(e.P, e.p);
        var a: [MAX]i64 = undefined;
        var b: [MAX]i64 = undefined;
        const want = try (try OpList.fromSlice(try r.stdOps(e.sg))).fingerprint(&b);
        const got = moved.fingerprint(&a) catch return null;
        return if (std.mem.eql(i64, got, want)) e else null;
    }

    /// The P (in the candidates' order) for which {P⁻¹ W P} is the standard group's rotations.
    fn matchingP(r: *Reader, cif: []const Op, stdo: []const Op) ![]const Mat {
        var target: [MAX]i64 = undefined;
        for (stdo, 0..) |o, i| target[i] = rotCode(o.W).?;
        const tms = target[0..stdo.len];
        std.mem.sort(i64, tms, {}, std.sort.asc(i64));
        var distinct: [MAX]Mat = undefined;
        var nd: usize = 0;
        for (cif) |o| {
            if (std.mem.eql(i32, &o.W, &I3)) continue;
            for (distinct[0..nd]) |d| {
                if (std.mem.eql(i32, &d, &o.W)) break;
            } else {
                distinct[nd] = o.W;
                nd += 1;
            }
        }
        var out: std.ArrayList(Mat) = .empty;
        var got: [MAX]i64 = undefined;
        next: for (try r.candidates()) |P| {
            const Pinv = group.adjugate(P);
            for (distinct[0..nd]) |W| {
                const c = rotCode(group.matMul(Pinv, group.matMul(W, P))) orelse continue :next;
                if (!sortedHas(tms, c)) continue :next;
            }
            for (cif, 0..) |o, i| got[i] = rotCode(group.matMul(Pinv, group.matMul(o.W, P))) orelse continue :next;
            std.mem.sort(i64, got[0..cif.len], {}, std.sort.asc(i64));
            if (std.mem.eql(i64, got[0..cif.len], tms)) try out.append(r.arena, P);
        }
        return out.items;
    }

    /// The origin shift p (24ths) taking the operators, in the basis P, onto the standard
    /// ones: the smallest (by its sum, then its components), or null.
    fn findP(r: *Reader, cif: []const Op, stdo: []const Op, P: Mat) !?[3]i32 {
        _ = r;
        if (cif.len != stdo.len) return null;
        const Pinv = group.inverseUnimodular(P) catch return null;
        var sc: [MAX]i64 = undefined;
        for (stdo, 0..) |o, i| sc[i] = try o.code();
        const scs = sc[0..stdo.len];
        std.mem.sort(i64, scs, {}, std.sort.asc(i64));
        // per operator: its rotation's code in the new basis; A = P⁻¹ (W − I) and b = P⁻¹ w,
        // so that its translation is A p + b
        var rc: [MAX]i64 = undefined;
        var A: [MAX]Mat = undefined;
        var b: [MAX][3]i32 = undefined;
        const cube = DEN * DEN * DEN;
        for (cif, 0..) |o, i| {
            rc[i] = rotCode(group.matMul(Pinv, group.matMul(o.W, P))) orelse return null;
            // (a rotation the standard group lacks: no p will do)
            var lo: usize = 0;
            var hi: usize = scs.len;
            const at = rc[i] * cube;
            while (lo < hi) {
                const mid = (lo + hi) / 2;
                if (scs[mid] < at) lo = mid + 1 else hi = mid;
            }
            if (lo == scs.len or @divFloor(scs[lo], cube) != rc[i]) return null;
            var Wm = o.W;
            Wm[0] -= 1;
            Wm[4] -= 1;
            Wm[8] -= 1;
            A[i] = group.matMul(Pinv, Wm);
            b[i] = group.matVec(Pinv, o.w);
        }
        var s: i32 = 0;
        while (s <= 3 * (DEN - 1)) : (s += 1) {
            var p0: i32 = 0;
            while (p0 <= @min(s, DEN - 1)) : (p0 += 1) {
                var p1: i32 = 0;
                while (p1 <= @min(s - p0, DEN - 1)) : (p1 += 1) {
                    const p = [3]i32{ p0, p1, s - p0 - p1 };
                    if (p[2] > DEN - 1) continue;
                    const ok = for (0..cif.len) |i| {
                        const t = group.matVec(A[i], p);
                        const c = ((rc[i] * DEN + @mod(t[0] + b[i][0], DEN)) * DEN + @mod(t[1] + b[i][1], DEN)) * DEN + @mod(t[2] + b[i][2], DEN);
                        if (!sortedHas(scs, c)) break false;
                    } else true;
                    if (ok) return p;
                }
            }
        }
        return null;
    }

    /// Operators on primitive rhombohedral axes, on hexagonal ones: the basis change has
    /// determinant 3 (three lattice points to the hexagonal cell), so it is applied exactly,
    /// the R centring added and the set closed.
    fn rhombToHex(r: *Reader, ops: []const Op) !OpList {
        const P = p_hex_from_rhomb;
        const adj = group.adjugate(P); // 3 P⁻¹
        var gens: OpList = .{};
        for (ops) |o| {
            const num = group.matMul(adj, group.matMul(o.W, P));
            var W: Mat = undefined;
            for (num, 0..) |v, i| {
                if (@mod(v, 3) != 0) return r.fail("the operators are not compatible with rhombohedral axes", .{});
                W[i] = @divExact(v, 3);
            }
            const tn = group.matVec(adj, o.w);
            var w: [3]i32 = undefined;
            for (tn, 0..) |v, i| {
                if (@mod(v, 3) != 0) return r.fail("the translations do not land on the 1/24 lattice on hexagonal axes", .{});
                w[i] = @divExact(v, 3);
            }
            try gens.append(Op.init(W, w));
        }
        try gens.append(.{ .W = I3, .w = .{ 16, 8, 8 } });
        try gens.append(.{ .W = I3, .w = .{ 8, 16, 16 } });
        return r.closeOps(gens.slice());
    }

    /// The change of basis taking the operators to the standard setting of their group, which
    /// is `number` when the file declares one (and must then agree).
    fn findSetting(r: *Reader, ops: []const Op, number: ?u32) !Setting {
        var closed = try r.closeOps(ops);
        if (try r.lookup(&closed)) |e| {
            if (number) |n| if (n != e.sg) return r.fail("the CIF's operators are space group {d} but it declares IT number {d}", .{ e.sg, n });
            return .{ .number = e.sg, .P = e.P, .p = e.p };
        }
        const n = number orelse return r.fail("the CIF declares no IT number and its {d} operators are not a tabulated setting of any of the 230 space groups; add _space_group_IT_number, or a Hall symbol, or check the symop loop", .{closed.n});
        const stdo = try r.stdOps(n);
        var rhomb = false;
        var note: []const u8 = "";
        if (closed.n * 3 == stdo.len) {
            closed = try r.rhombToHex(closed.slice());
            rhomb = true;
            note = "converted from rhombohedral to hexagonal axes";
            if (try r.lookup(&closed)) |e| {
                if (n != e.sg) return r.fail("the CIF's operators are space group {d} but it declares IT number {d}", .{ e.sg, n });
                return .{ .number = e.sg, .P = e.P, .p = e.p, .rhombohedral = true, .note = note };
            }
        }
        if (closed.n != stdo.len) return r.fail("the CIF's group has {d} operators but space group {d} has {d}: either the IT number is wrong or the symop list is not a group", .{ closed.n, n, stdo.len });
        const Ps = try r.matchingP(closed.slice(), stdo);
        if (Ps.len == 0) return r.fail("no change of basis (det +1) maps the CIF's rotations onto space group {d}'s: its operations and its IT number disagree", .{n});
        for (Ps) |P| if (try r.findP(closed.slice(), stdo, P)) |p| return .{ .number = n, .P = P, .p = p, .rhombohedral = rhomb, .note = note };
        return r.fail("{d} changes of basis map the rotations onto space group {d}'s but no origin shift in 24ths reproduces its operators", .{ Ps.len, n });
    }

    // ── sites ──

    /// Whether every operator maps each element's sites onto themselves (within eps).
    fn sitesInvariant(r: *Reader, ops: []const Op, species: []const []const u8, coords: []const [3]f64) bool {
        for (species, coords) |sp, x| {
            for (ops) |o| {
                const y = image(o, x);
                var best: f64 = std.math.inf(f64);
                for (species, coords) |sq, q| if (std.mem.eql(u8, sp, sq)) {
                    best = @min(best, nearest(y, q));
                };
                if (best > r.eps) return false;
            }
        }
        return true;
    }

    /// The sites' images under the operators, each element's once (within eps).
    fn expandOrbits(r: *Reader, ops: []const Op, species: []const []const u8, coords: []const [3]f64) !struct { species: [][]const u8, coords: [][3]f64 } {
        var sp_out: std.ArrayList([]const u8) = .empty;
        var x_out: std.ArrayList([3]f64) = .empty;
        for (species, 0..) |sp, i| {
            // each element once, at its first site
            const first = for (species[0..i]) |sq| {
                if (std.mem.eql(u8, sp, sq)) break false;
            } else true;
            if (!first) continue;
            const start = x_out.items.len;
            for (species, coords) |sq, x| if (std.mem.eql(u8, sp, sq)) for (ops) |o| {
                const y = image(o, x);
                const dup = for (x_out.items[start..]) |q| {
                    if (nearest(y, q) <= r.eps) break true;
                } else false;
                if (!dup) {
                    try x_out.append(r.arena, y);
                    try sp_out.append(r.arena, sp);
                }
            };
        }
        return .{ .species = sp_out.items, .coords = x_out.items };
    }

    /// One site of each orbit under the operators (the first met): the kept sites' indices.
    fn foldAsymmetric(r: *Reader, ops: []const Op, species: []const []const u8, coords: []const [3]f64) ![]usize {
        const keep = try r.arena.alloc(bool, species.len);
        @memset(keep, false);
        for (species, coords, 0..) |sp, x, i| {
            const covered = for (species[0..i], coords[0..i], 0..) |sq, y, j| {
                if (!keep[j] or !std.mem.eql(u8, sp, sq)) continue;
                const hit = for (ops) |o| {
                    if (nearest(x, image(o, y)) <= r.eps) break true;
                } else false;
                if (hit) break true;
            } else false;
            keep[i] = !covered;
        }
        var out: std.ArrayList(usize) = .empty;
        for (keep, 0..) |k, i| if (k) try out.append(r.arena, i);
        return out.items;
    }

    /// The tabulated groups by order, and per group its largest subgroups among them.
    fn descentTree(r: *Reader) !*const Tree {
        if (r.tree == null) {
            var t: Tree = undefined;
            for (1..231) |n| t.orders[n - 1] = @intCast((try r.stdOps(@intCast(n))).len);
            var kids: [230]std.ArrayList(u32) = @splat(.empty);
            for (0..7) |fi| {
                const f: Family = @enumFromInt(fi);
                var nums: std.ArrayList(u32) = .empty;
                for (1..231) |n| if (inFamily(f, @intCast(n))) try nums.append(r.arena, @intCast(n));
                var max_o: u32 = 0;
                for (nums.items) |n| max_o = @max(max_o, t.orders[n - 1]);
                var tops: std.ArrayList(u32) = .empty;
                for (nums.items) |n| if (t.orders[n - 1] == max_o) try tops.append(r.arena, n);
                t.tops[fi] = tops.items;
                for (nums.items) |n| {
                    const on = t.orders[n - 1];
                    var cands: std.ArrayList(u32) = .empty;
                    for (nums.items) |m| {
                        const om = t.orders[m - 1];
                        if (om < on and on % om == 0) try cands.append(r.arena, m);
                    }
                    var children: std.ArrayList(u32) = .empty;
                    for (cands.items) |m| {
                        const om = t.orders[m - 1];
                        const between = for (cands.items) |k| {
                            const ok_ = t.orders[k - 1];
                            if (om < ok_ and ok_ < on and on % ok_ == 0 and ok_ % om == 0) break true;
                        } else false;
                        if (!between) try children.append(r.arena, m);
                    }
                    const by = struct {
                        fn lt(orders: *const [230]u32, a: u32, b: u32) bool {
                            const oa = orders[a - 1];
                            const ob = orders[b - 1];
                            return if (oa != ob) oa > ob else a < b;
                        }
                    };
                    std.mem.sort(u32, children.items, &t.orders, by.lt);
                    for (children.items) |m| {
                        if (std.mem.indexOfScalar(u32, kids[n - 1].items, m) == null) try kids[n - 1].append(r.arena, m);
                    }
                }
            }
            for (0..230) |i| t.children[i] = kids[i].items;
            r.tree = t;
        }
        return &r.tree.?;
    }

    /// The largest tabulated group (standard setting, origin choice 1, or rhombohedral axes on
    /// a rhombohedral cell) that leaves the sites in place: down the subgroups from the
    /// largest groups the cell's metric allows, stopping at the first that does.
    fn recoverFromSites(r: *Reader, cell: [6]f64, species: []const []const u8, coords: []const [3]f64) ![]const Op {
        const tree = try r.descentTree();
        const fam = metricFamily(cell);
        var start = tree.tops[@intFromEnum(fam)];
        if (start.len == 0) start = tree.tops[@intFromEnum(Family.hexagonal_family)];
        var stack: std.ArrayList(u32) = .empty;
        try stack.appendSlice(r.arena, start);
        var visited: [231]bool = @splat(false);
        var best_n: u32 = 0;
        var best: ?[]const Op = null;
        while (stack.pop()) |n| {
            if (visited[n]) continue;
            visited[n] = true;
            var hit: ?[]const Op = null;
            const stdo = try r.stdOps(n);
            if (r.sitesInvariant(stdo, species, coords)) hit = stdo;
            if (hit == null) if (origin1Hall(n)) |h| {
                const o = try r.hallSlice(h);
                if (r.sitesInvariant(o, species, coords)) hit = o;
            };
            if (hit == null and fam == .rhombohedral) if (rhombHall(n)) |h| {
                const o = try r.hallSlice(h);
                if (r.sitesInvariant(o, species, coords)) hit = o;
            };
            if (hit) |h| {
                if (best == null or h.len > best.?.len or (h.len == best.?.len and n < best_n)) {
                    best = h;
                    best_n = n;
                }
                continue;
            }
            for (tree.children[n - 1]) |m| if (!visited[m]) try stack.append(r.arena, m);
        }
        if (best == null) for ([_]u32{ 2, 1 }) |n| {
            const o = try r.stdOps(n);
            if (r.sitesInvariant(o, species, coords)) {
                best = o;
                break;
            }
        };
        return best orelse r.fail("no tabulated group leaves the atom positions in place (the cell's metric: {s}; within {e})", .{ @tagName(fam), r.eps });
    }

    // ── a Hermann–Mauguin symbol ──

    const Hm = struct { lat: u8, toks: [3][]const u8, n: usize, axes: u8 = 0, origin: u8 = 0 };
    const HmHit = struct { n: u32, ops: []const Op, P: Mat };

    /// "C 1 2 1" / "C2" / "P2(1)/n" / "R -3 m :H": the lattice letter, three positions ("1"
    /// where nothing is; a single position is monoclinic, unique axis b), the axes (H or R)
    /// and the origin choice.
    fn normalizeHm(r: *Reader, sym0: []const u8) !Hm {
        var s = std.mem.trim(u8, std.mem.trim(u8, std.mem.trim(u8, sym0, " \t\r\n"), "'\""), " \t\r\n");
        var out: Hm = .{ .lat = 0, .toks = undefined, .n = 0 };
        // ":H", ":R", ":1", ":2" at the end
        if (s.len >= 2) {
            const last = std.ascii.toUpper(s[s.len - 1]);
            if (std.mem.indexOfScalar(u8, "HR12", last) != null) {
                const before = std.mem.trimEnd(u8, s[0 .. s.len - 1], " \t");
                if (before.len > 0 and before[before.len - 1] == ':') {
                    if (last == 'H' or last == 'R') out.axes = last else out.origin = last - '0';
                    s = std.mem.trim(u8, before[0 .. before.len - 1], " \t");
                }
            }
        }
        // "hexagonal axes" / "rhombohedral axes"
        var low: [128]u8 = undefined;
        if (s.len <= low.len) {
            for (s, 0..) |c, i| low[i] = std.ascii.toLower(c);
            const l = low[0..s.len];
            var at: ?usize = null;
            var end: usize = 0;
            var axes: u8 = 0;
            for ([_][]const u8{ "hexagonal", "rhombohedral" }) |word| {
                var from: usize = 0;
                while (std.mem.indexOfPos(u8, l, from, word)) |k| : (from = k + 1) {
                    if (k > 0 and (std.ascii.isAlphanumeric(l[k - 1]) or l[k - 1] == '_')) continue;
                    var j = k + word.len;
                    const ws0 = j;
                    while (j < l.len and std.ascii.isWhitespace(l[j])) j += 1;
                    if (j == ws0 or !std.mem.startsWith(u8, l[j..], "axes")) continue;
                    j += 4;
                    if (j < l.len and (std.ascii.isAlphanumeric(l[j]) or l[j] == '_')) continue;
                    if (at == null or k < at.?) {
                        at = k;
                        end = j;
                        axes = std.ascii.toUpper(word[0]);
                    }
                    break;
                }
            }
            if (at) |k| {
                const joined = try std.mem.concat(r.arena, u8, &.{ s[0..k], s[end..] });
                s = std.mem.trim(u8, joined, " \t\r\n");
                out.axes = axes;
            }
        }
        // brackets dropped, white space single
        var clean: std.ArrayList(u8) = .empty;
        var space = false;
        for (s) |c| {
            if (c == '(' or c == ')') continue;
            if (std.ascii.isWhitespace(c)) {
                space = true;
                continue;
            }
            if (space and clean.items.len > 0) try clean.append(r.arena, ' ');
            space = false;
            try clean.append(r.arena, c);
        }
        const t = clean.items;
        if (t.len == 0) return r.fail("empty Hermann-Mauguin symbol", .{});
        out.lat = std.ascii.toUpper(t[0]);
        if (std.mem.indexOfScalar(u8, "PABCIRSTF", out.lat) == null) return r.fail("the H-M symbol '{s}' does not start with a lattice letter", .{sym0});
        const rest = std.mem.trim(u8, t[1..], " ");
        var toks: [8][]const u8 = undefined;
        var nt: usize = 0;
        var pos: usize = 0;
        while (pos < rest.len) {
            if (rest[pos] == ' ' or rest[pos] == '_') {
                pos += 1;
                continue;
            }
            const len = hmToken(rest[pos..]);
            if (len == 0 or nt == toks.len) return r.fail("cannot read the H-M symbol '{s}' at '{s}'", .{ sym0, rest[pos..] });
            toks[nt] = rest[pos .. pos + len];
            nt += 1;
            pos += len;
        }
        if (nt == 0) return r.fail("the H-M symbol '{s}' has no symmetry positions", .{sym0});
        if (nt == 1 and (std.mem.eql(u8, toks[0], "1") or std.mem.eql(u8, toks[0], "-1"))) {
            out.toks[0] = toks[0];
            out.n = 1;
            return out;
        }
        if (nt == 1) {
            // short monoclinic: unique axis b
            out.toks = .{ "1", toks[0], "1" };
            out.n = 3;
        } else if (nt == 2) {
            // short trigonal and hexagonal (R-3c, P6/m) or cubic (P23)
            const f0 = std.mem.trimStart(u8, toks[0], "-");
            const f1 = std.mem.trimStart(u8, toks[1], "-");
            if (!(out.lat == 'R' or (std.mem.indexOfScalar(u8, "PFI", out.lat) != null and f0.len > 0 and std.mem.indexOfScalar(u8, "2346", f0[0]) != null and f1.len > 0 and std.mem.indexOfScalar(u8, "3mabcd", f1[0]) != null)))
                return r.fail("the H-M symbol '{s}' has 2 positions; expected 1 or 3", .{sym0});
            out.toks = .{ toks[0], toks[1], "" };
            out.n = 2;
        } else if (nt == 3) {
            out.toks = .{ toks[0], toks[1], toks[2] };
            out.n = 3;
        } else return r.fail("the H-M symbol '{s}' has {d} positions; expected 3", .{ sym0, nt });
        // the old "P m 3 m" for the modern "P m -3 m"
        if (out.n == 3 and std.mem.eql(u8, out.toks[1], "3") and !std.mem.eql(u8, out.toks[0], "3") and std.mem.indexOfScalar(u8, "PFI", out.lat) != null) {
            var glide = false;
            for (out.toks) |x| glide = glide or std.mem.eql(u8, x, "m") or std.mem.eql(u8, x, "n") or std.mem.eql(u8, x, "d") or std.mem.eql(u8, x, "a");
            if (glide) for (&out.toks) |*x| if (std.mem.eql(u8, x.*, "3")) {
                x.* = "-3";
            };
        }
        return out;
    }

    /// The length of the H-M position at the start of s: a rotation (−?[1-6][1-5]?) or a
    /// mirror's letter (m a b c n d e), then perhaps "/" and a mirror or rotation; 0 if none.
    fn hmToken(s: []const u8) usize {
        var i: usize = 0;
        if (i < s.len and s[i] == '-' and i + 1 < s.len and s[i + 1] >= '1' and s[i + 1] <= '6') i += 1;
        if (i < s.len and s[i] >= '1' and s[i] <= '6') {
            i += 1;
            if (i < s.len and s[i] >= '1' and s[i] <= '5') i += 1;
        } else if (i == 0 and s.len > 0 and std.mem.indexOfScalar(u8, "mabcnde", s[0]) != null) {
            i = 1;
        } else return 0;
        if (i + 1 < s.len and s[i] == '/') {
            var j = i + 1;
            if (s[j] == '-' and j + 1 < s.len and s[j + 1] >= '1' and s[j + 1] <= '6') j += 1;
            if (s[j] >= '1' and s[j] <= '6') {
                return j + 1;
            } else if (j == i + 1 and std.mem.indexOfScalar(u8, "mabcnd", s[j]) != null) return j + 1;
        }
        return i;
    }

    fn hmKey(r: *Reader, h: Hm) ![]u8 {
        return std.fmt.allocPrint(r.arena, "{c}|{s}|{s}|{s}", .{ h.lat, h.toks[0], if (h.n > 1) h.toks[1] else "", if (h.n > 2) h.toks[2] else "" });
    }

    /// The standard names, and their old cubic spellings, to their groups.
    fn hmIndex(r: *Reader) !*std.StringHashMapUnmanaged(u32) {
        if (r.hm_index == null) {
            var m: std.StringHashMapUnmanaged(u32) = .empty;
            for (tables.sg_names, 1..) |name, n| {
                var variants: [3][]const u8 = undefined;
                var nv: usize = 1;
                variants[0] = name;
                if (std.mem.indexOfScalar(u8, "PFI", name[0]) != null) {
                    if (std.mem.indexOf(u8, name, " 3 ") != null) {
                        variants[nv] = try std.mem.replaceOwned(u8, r.arena, name, " 3 ", " -3 ");
                        nv += 1;
                    }
                    if (std.mem.endsWith(u8, name, " 3")) {
                        variants[nv] = try std.mem.concat(r.arena, u8, &.{ name[0 .. name.len - 2], " -3" });
                        nv += 1;
                    }
                }
                for (variants[0..nv]) |v| {
                    const h = r.normalizeHm(v) catch continue;
                    const slot = try m.getOrPut(r.arena, try r.hmKey(h));
                    if (!slot.found_existing) slot.value_ptr.* = @intCast(n);
                }
            }
            r.hm_index = m;
        }
        return &r.hm_index.?;
    }

    /// For each axis, the H-M letters the operators allow in that position (a set: in a
    /// centred group one plane is several glides at once): a bit per m a b c n d ? 2 21 1.
    fn derivedPositions(ops: []const Op) ![3]u16 {
        var pos: [3]u16 = .{ 0, 0, 0 };
        for (0..3) |axis| {
            var mirror: Mat = I3;
            mirror[axis * 4] = -1;
            for (ops) |o| if (std.mem.eql(i32, &o.W, &mirror)) {
                const t = try o.intrinsic();
                var nz: [3]usize = undefined;
                var n: usize = 0;
                for (0..3) |i| if (t[i] != 0) {
                    nz[n] = i;
                    n += 1;
                };
                var all_half = true;
                var all_quarter = true;
                for (nz[0..n]) |i| {
                    all_half = all_half and t[i] == DEN / 2;
                    all_quarter = all_quarter and (t[i] == DEN / 4 or t[i] == 3 * DEN / 4);
                }
                const bit: u4 = if (n == 0) 0 else if (all_half) (if (n == 1) @as(u4, @intCast(1 + nz[0])) else 4) else if (all_quarter) 5 else 6;
                pos[axis] |= @as(u16, 1) << bit;
            };
            if (pos[axis] == 0) {
                var rotation: Mat = .{ -1, 0, 0, 0, -1, 0, 0, 0, -1 };
                rotation[axis * 4] = 1;
                for (ops) |o| if (std.mem.eql(i32, &o.W, &rotation)) {
                    const t = try o.intrinsic();
                    pos[axis] |= @as(u16, 1) << @as(u4, if (@mod(t[axis], DEN) != 0) 8 else 7);
                };
            }
            if (pos[axis] == 0) pos[axis] = 1 << 9;
        }
        return pos;
    }

    fn hmBit(tok: []const u8) ?u4 {
        const names = [_][]const u8{ "m", "a", "b", "c", "n", "d", "?", "2", "21", "1" };
        for (names, 0..) |n, i| if (std.mem.eql(u8, n, tok)) return @intCast(i);
        return null;
    }

    /// A non-standard monoclinic or orthorhombic H-M symbol: the groups some basis change of
    /// which has that lattice and those letters (the first change that does, per group).
    fn monoOrthoSearch(r: *Reader, h: Hm) ![]const HmHit {
        var hits: std.ArrayList(HmHit) = .empty;
        if (h.n != 3) return hits.items;
        for (3..75) |ni| {
            const n: u32 = @intCast(ni);
            const stdo = try r.stdOps(n);
            const base = try OpList.fromSlice(stdo);
            for (try r.candidates()) |P| {
                const moved = try base.transformed(P, .{ 0, 0, 0 });
                if (group.latticeLetter(moved.slice()) != h.lat) continue;
                const pos = try derivedPositions(moved.slice());
                const ok = for (0..3) |i| {
                    if (std.mem.eql(u8, h.toks[i], "1")) {
                        if (pos[i] != 1 << 9) break false;
                    } else {
                        const bit = hmBit(h.toks[i]) orelse break false;
                        if (pos[i] & (@as(u16, 1) << bit) == 0) break false;
                    }
                } else true;
                if (ok) {
                    try hits.append(r.arena, .{ .n = n, .ops = try r.arena.dupe(Op, moved.slice()), .P = P });
                    break;
                }
            }
        }
        return hits.items;
    }

    /// An H-M symbol's group and operators (its setting: the symbol's axes or origin, or a
    /// non-standard monoclinic or orthorhombic setting recognised from its letters).
    fn hmToOps(r: *Reader, sym: []const u8, it: ?u32) !struct { n: u32, ops: []const Op, note: []const u8 } {
        const h = try r.normalizeHm(sym);
        if ((try r.hmIndex()).get(try r.hmKey(h))) |n| {
            if (it) |i| if (i != n) return r.fail("the H-M symbol '{s}' is space group {d} but the CIF declares IT number {d}", .{ sym, n, i });
            if (h.axes == 'R') {
                const rh = rhombHall(n) orelse return r.fail("space group {d} has no rhombohedral setting", .{n});
                return .{ .n = n, .ops = try r.hallSlice(rh), .note = "rhombohedral axes" };
            }
            if (h.origin == 1) {
                const o1 = origin1Hall(n) orelse return r.fail("space group {d} has only one origin choice", .{n});
                return .{ .n = n, .ops = try r.hallSlice(o1), .note = "origin choice 1" };
            }
            return .{ .n = n, .ops = try r.stdOps(n), .note = "" };
        }
        const all = try r.monoOrthoSearch(h);
        var hits: std.ArrayList(HmHit) = .empty;
        for (all) |x| if (it == null or x.n == it.?) try hits.append(r.arena, x);
        if (hits.items.len == 1) {
            const x = hits.items[0];
            return .{ .n = x.n, .ops = x.ops, .note = try std.fmt.allocPrint(r.arena, "non-standard setting recognised from the H-M symbol; P = {any}", .{x.P}) };
        }
        if (hits.items.len > 1) return r.fail("the H-M symbol '{s}' is ambiguous: it matches {d} space groups; supply _space_group_IT_number or, better, the symop loop", .{ sym, hits.items.len });
        return r.fail("cannot resolve the H-M symbol '{s}' to a space group; this CIF needs a _space_group_symop_operation_xyz loop or a Hall symbol", .{sym});
    }

    // ── the file's symmetry, and its sites ──

    fn symmetryOf(r: *Reader, blk: *const Block) !Symmetry {
        var it: ?u32 = null;
        if (blk.str(&.{ "_space_group_it_number", "_symmetry_int_tables_number" })) |s| {
            const v = std.fmt.parseFloat(f64, s) catch return r.fail("cannot read the IT number '{s}'", .{s});
            if (!(v >= 1 and v < 231)) return r.fail("the IT number {d} is not one of 1 … 230", .{v});
            it = @intFromFloat(@trunc(v));
        }
        if (try blk.loop(r.arena, &.{ "_space_group_symop_operation_xyz", "_symmetry_equiv_pos_as_xyz", "_space_group_symop.operation_xyz" })) |xyz| if (xyz.len > 0) {
            var ops: std.ArrayList(Op) = .empty;
            for (xyz) |s0| {
                const s = std.mem.trim(u8, s0, " \t\r\n");
                if (s.len == 0 or std.mem.eql(u8, s, "?") or std.mem.eql(u8, s, ".")) continue;
                try ops.append(r.arena, group.parseXyz(s) catch return r.fail("cannot read the symmetry operation '{s}'", .{s}));
            }
            if (ops.items.len > 0) return .{
                .ops = ops.items,
                .it = it,
                .provenance = try std.fmt.allocPrint(r.arena, "_space_group_symop_operation_xyz ({d} listed)", .{ops.items.len}),
                .useful = !group.trivial((try r.closeOps(ops.items)).slice()),
            };
        };
        if (blk.str(&.{ "_space_group_name_hall", "_symmetry_space_group_name_hall" })) |hall| {
            const ops = try r.hallSlice(hall);
            return .{ .ops = ops, .it = it, .provenance = try std.fmt.allocPrint(r.arena, "Hall symbol '{s}'", .{hall}), .useful = !group.trivial(ops) };
        }
        if (blk.str(&.{ "_space_group_name_h-m_alt", "_symmetry_space_group_name_h-m", "_space_group_name_h-m_ref", "_space_group_name_h-m_full" })) |hm| {
            var compact: [64]u8 = undefined;
            var nc: usize = 0;
            for (std.mem.trim(u8, hm, "'\"")) |c| if (!std.ascii.isWhitespace(c) and nc < compact.len) {
                compact[nc] = std.ascii.toLower(c);
                nc += 1;
            };
            if (std.mem.eql(u8, compact[0..nc], "p1") or std.mem.eql(u8, compact[0..nc], "c1")) {
                const n = it orelse 1;
                return .{ .ops = try r.stdOps(n), .it = n, .provenance = try std.fmt.allocPrint(r.arena, "H-M symbol '{s}' (trivial)", .{hm}), .useful = false };
            }
            // an IT number is the group's identity: trusted over a (slow) H-M setting search
            if (it) |n| return .{ .ops = null, .it = n, .provenance = try std.fmt.allocPrint(r.arena, "H-M '{s}' + IT {d}", .{ hm, n }), .useful = false };
            const h = try r.hmToOps(hm, it);
            return .{ .ops = h.ops, .it = h.n, .provenance = try std.fmt.allocPrint(r.arena, "H-M symbol '{s}' -> SG{d}{s}{s}{s}", .{ hm, h.n, if (h.note.len > 0) " (" else "", h.note, if (h.note.len > 0) ")" else "" }), .useful = false };
        }
        if (it) |n| return .{ .ops = null, .it = n, .provenance = try std.fmt.allocPrint(r.arena, "IT number {d}", .{n}), .useful = false };
        return .{ .ops = null, .it = null, .provenance = "no declared symmetry", .useful = false };
    }

    fn atomsOf(r: *Reader, blk: *const Block) !Atoms {
        var a: Atoms = undefined;
        const cell_tags = [_][]const u8{ "_cell_length_a", "_cell_length_b", "_cell_length_c", "_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma" };
        for (cell_tags, 0..) |t, i| {
            const s = blk.str(&.{t});
            a.cell[i] = (parseNumber(s) catch return r.fail("cannot read {s} '{s}' as a number", .{ t, s.? })) orelse return r.fail("the CIF is missing one or more cell parameters", .{});
        }
        const labels = try blk.loop(r.arena, &.{"_atom_site_label"});
        const fx = try blk.loop(r.arena, &.{"_atom_site_fract_x"});
        const fy = try blk.loop(r.arena, &.{"_atom_site_fract_y"});
        const fz = try blk.loop(r.arena, &.{"_atom_site_fract_z"});
        if (labels == null or fx == null or fy == null or fz == null or labels.?.len == 0 or fx.?.len == 0 or fy.?.len == 0 or fz.?.len == 0)
            return r.fail("the CIF is missing the _atom_site fractional coordinate loop", .{});
        const types = try blk.loop(r.arena, &.{"_atom_site_type_symbol"});
        const occs = try blk.loop(r.arena, &.{"_atom_site_occupancy"});
        const uiso = try blk.loop(r.arena, &.{"_atom_site_u_iso_or_equiv"});
        const biso = try blk.loop(r.arena, &.{"_atom_site_b_iso_or_equiv"});
        const n = labels.?.len;
        if (fx.?.len < n or fy.?.len < n or fz.?.len < n) return r.fail("the _atom_site loop's coordinate columns are shorter than its labels", .{});
        a.labels = try r.arena.alloc([]const u8, n);
        a.species = try r.arena.alloc([]const u8, n);
        a.coords = try r.arena.alloc([3]f64, n);
        a.occ = try r.arena.alloc(f64, n);
        a.u = try r.arena.alloc(f64, n);
        a.thermal = (uiso != null and uiso.?.len > 0) or (biso != null and biso.?.len > 0);
        for (0..n) |i| {
            a.labels[i] = labels.?[i];
            // the type symbol (or the label) without its charge or number: "Fe3+" → "Fe"
            var raw = std.mem.trim(u8, std.mem.trim(u8, if (types != null and i < types.?.len) types.?[i] else labels.?[i], " \t\r\n"), "'\"");
            while (raw.len > 0 and std.mem.indexOfScalar(u8, "0123456789+-.", raw[raw.len - 1]) != null) raw = raw[0 .. raw.len - 1];
            a.species[i] = elementOf(if (raw.len > 0) raw else labels.?[i]) orelse return r.fail("cannot tell the element of the atom '{s}'", .{labels.?[i]});
            for ([_]?[]const []const u8{ fx, fy, fz }, 0..) |col, k| {
                const s = col.?[i];
                a.coords[i][k] = (parseNumber(s) catch return r.fail("cannot read the coordinate '{s}' of atom '{s}'", .{ s, labels.?[i] })) orelse return r.fail("atom '{s}' has no coordinate", .{labels.?[i]});
            }
            // (an occupancy of 0, like none, is 1)
            const o = if (occs != null and i < occs.?.len) parseNumber(occs.?[i]) catch return r.fail("cannot read the occupancy '{s}'", .{occs.?[i]}) else null;
            a.occ[i] = if (o) |v| (if (v == 0) 1 else v) else 1;
            if (uiso != null and uiso.?.len > 0) {
                if (i >= uiso.?.len) return r.fail("the U_iso column is shorter than the labels", .{});
                a.u[i] = (parseNumber(uiso.?[i]) catch return r.fail("cannot read U_iso '{s}'", .{uiso.?[i]})) orelse 0;
            } else if (biso != null and biso.?.len > 0) {
                if (i >= biso.?.len) return r.fail("the B_iso column is shorter than the labels", .{});
                a.u[i] = ((parseNumber(biso.?[i]) catch return r.fail("cannot read B_iso '{s}'", .{biso.?[i]})) orelse 0) / eight_pi2;
            } else a.u[i] = 0;
        }
        return a;
    }

    /// The candidate settings of a trusted group: the file's own operators (from its symbol),
    /// the standard ones, origin choice 1, rhombohedral axes, and the monoclinic unique axes
    /// and cell choice, and orthorhombic axis permutations, that occur in CIFs.
    fn groupCandidates(r: *Reader, n: u32, preferred: ?[]const Op) ![]const []const Op {
        var out: std.ArrayList([]const Op) = .empty;
        var seen: std.ArrayList(u64) = .empty;
        var fp: [MAX]i64 = undefined;
        const add = struct {
            fn f(rr: *Reader, o: *std.ArrayList([]const Op), sn: *std.ArrayList(u64), buf: *[MAX]i64, ops: []const Op) !void {
                const c = try rr.closeOps(ops);
                const h = hashOf(try c.fingerprint(buf));
                if (std.mem.indexOfScalar(u64, sn.items, h) != null) return;
                try sn.append(rr.arena, h);
                try o.append(rr.arena, ops);
            }
        }.f;
        const stdo = try r.stdOps(n);
        if (preferred) |p| try add(r, &out, &seen, &fp, p);
        try add(r, &out, &seen, &fp, stdo);
        if (origin1Hall(n)) |h| try add(r, &out, &seen, &fp, try r.hallSlice(h));
        if (rhombHall(n)) |h| try add(r, &out, &seen, &fp, try r.hallSlice(h));
        const mono = [_]Mat{
            .{ 0, 0, 1, 0, 1, 0, -1, 0, 0 },  .{ 0, 1, 0, 0, 0, 1, 1, 0, 0 },
            .{ 0, -1, 0, -1, 0, 0, 0, 0, -1 }, .{ -1, 0, 0, 0, 0, -1, 0, -1, 0 },
            .{ 1, 0, 1, 0, 1, 0, 0, 0, 1 },   .{ 1, 0, -1, 0, 1, 0, 0, 0, 1 },
        };
        const ortho = [_]Mat{
            .{ 0, 1, 0, 0, 0, 1, 1, 0, 0 },  .{ 0, 0, 1, 1, 0, 0, 0, 1, 0 }, .{ 0, 1, 0, 1, 0, 0, 0, 0, -1 },
            .{ 1, 0, 0, 0, 0, 1, 0, -1, 0 }, .{ 0, 0, 1, 0, 1, 0, -1, 0, 0 },
        };
        const qs: []const Mat = switch (systemOf(n)) {
            .monoclinic => &mono,
            .orthorhombic => &ortho,
            else => &.{},
        };
        const base = try OpList.fromSlice(stdo);
        for (qs) |Q| {
            const moved = try base.transformed(Q, .{ 0, 0, 0 });
            try add(r, &out, &seen, &fp, try r.arena.dupe(Op, moved.slice()));
        }
        return out.items;
    }

    /// 0 when the cell has its system's conventional metric.
    fn metricPenalty(n: u32, c: [6]f64) f64 {
        return switch (systemOf(n)) {
            .monoclinic => @abs(c[3] - 90) + @abs(c[5] - 90) + @as(f64, if (@abs(c[4] - 90) > 0.5) 0 else 10),
            .orthorhombic => @abs(c[3] - 90) + @abs(c[4] - 90) + @abs(c[5] - 90),
            .tetragonal => @abs(c[0] - c[1]) + @abs(c[3] - 90) + @abs(c[4] - 90) + @abs(c[5] - 90),
            .trigonal, .hexagonal => @abs(c[0] - c[1]) + @abs(c[3] - 90) + @abs(c[4] - 90) + @abs(c[5] - 120),
            .cubic => @abs(c[0] - c[1]) + @abs(c[1] - c[2]) + @abs(c[3] - 90) + @abs(c[4] - 90) + @abs(c[5] - 90),
            .triclinic => 0,
        };
    }

    /// A trusted group's setting from the cell and the sites: of its candidate settings, those
    /// that leave the sites in place, the one whose cell is the most conventional.
    fn resolveGroupSetting(r: *Reader, n: u32, a: *const Atoms, preferred: ?[]const Op) !Setting {
        var best: ?Setting = null;
        var best_pen: f64 = 0;
        for (try r.groupCandidates(n, preferred)) |ops| {
            const closed = try r.closeOps(ops);
            const e = try r.expandOrbits(closed.slice(), a.species, a.coords);
            if (!r.sitesInvariant(closed.slice(), e.species, e.coords)) continue;
            const st = r.findSetting(ops, n) catch |err| switch (err) {
                error.Symmetry => continue,
                else => return err,
            };
            var c2 = transformCell(a.cell, st.P);
            if (st.rhombohedral) c2 = transformCell(c2, p_hex_from_rhomb);
            const pen = metricPenalty(n, c2);
            if (best == null or pen < best_pen) {
                best = st;
                best_pen = pen;
                if (pen == 0) break;
            }
        }
        return best orelse r.fail("the cell and sites fit no tabulated setting of space group {d} (the group the CIF names is trusted; its setting could not be found)", .{n});
    }

    /// The crystal of a CIF (its block with atom sites, when it has several), in the standard
    /// setting; error.Symmetry with why() when it cannot be read or disagrees with itself.
    pub fn read(r: *Reader, text: []const u8) !Structure {
        const blocks = try parse(r.arena, text);
        if (blocks.len == 0) return r.fail("no data_ block found in the CIF", .{});
        var blk = &blocks[0];
        if (blocks.len > 1) for (blocks) |*b| if (try b.loop(r.arena, &.{"_atom_site_fract_x"})) |c| if (c.len > 0) {
            blk = b;
            break;
        };
        return r.structureOf(blk);
    }

    fn structureOf(r: *Reader, blk: *const Block) !Structure {
        const symm = try r.symmetryOf(blk);
        var a = try r.atomsOf(blk);
        var s: Structure = .{
            .block = blk.name,
            .number = 0,
            .cell = undefined,
            .sites = undefined,
            .setting = undefined,
            .provenance = symm.provenance,
            .declared_it = null,
            .declared_hall = blk.str(&.{ "_space_group_name_hall", "_symmetry_space_group_name_hall" }),
            .declared_hm = blk.str(&.{ "_space_group_name_h-m_alt", "_symmetry_space_group_name_h-m", "_space_group_name_h-m_ref", "_space_group_name_h-m_full" }),
            .cell_as_read = a.cell,
            .sites_as_read = a.species.len,
            .defaulted_b = 0,
            .thermal_tags = a.thermal,
        };
        if (blk.str(&.{ "_space_group_it_number", "_symmetry_int_tables_number" })) |t| {
            if (std.fmt.parseFloat(f64, t)) |v| {
                s.declared_it = @intFromFloat(@trunc(v));
            } else |_| {}
        }
        // a missing or non-positive U: the default
        for (a.u) |*u| if (!std.math.isFinite(u.*) or u.* <= 0) {
            u.* = default_b_iso / eight_pi2;
            s.defaulted_b += 1;
        };

        var setting: ?Setting = null;
        var from_sites: ?[]const Op = null;
        const trivial_ops = if (symm.ops) |o| group.trivial((try r.closeOps(o)).slice()) else false;
        if (symm.useful and symm.ops != null) {
            // listed operators, or a Hall symbol: authoritative
            setting = try r.findSetting(symm.ops.?, symm.it);
        } else if (symm.it != null and !trivial_ops) {
            // a trusted group (IT number or H-M symbol): its setting from the cell and sites
            setting = try r.resolveGroupSetting(symm.it.?, &a, if (symm.ops != null and !trivial_ops) symm.ops else null);
            s.provenance = try std.mem.concat(r.arena, u8, &.{ s.provenance, " | setting resolved from cell/sites" });
        } else {
            // nothing trusted (or P1): from the sites alone
            if (r.recoverFromSites(a.cell, a.species, a.coords)) |ops| {
                if (r.findSetting(ops, null)) |st| {
                    setting = st;
                    from_sites = ops;
                    s.provenance = try std.fmt.allocPrint(r.arena, "sites+Hall SG{d} (tree descent) | ops path trivial/absent", .{st.number});
                } else |err| if (err != error.Symmetry) return err;
            } else |err| if (err != error.Symmetry) return err;
            if (setting == null) {
                const inner = try r.arena.dupe(u8, r.why());
                return r.fail("cannot determine the symmetry: the CIF has no useful operators and no tabulated group leaves its sites in place ({s})", .{inner});
            }
        }
        const st = setting.?;
        if (from_sites) |ops| {
            const keep = try r.foldAsymmetric(ops, a.species, a.coords);
            inline for (.{ "labels", "species", "coords", "occ", "u" }) |field| {
                const src = @field(a, field);
                for (keep, 0..) |k, i| src[i] = src[k];
                @field(a, field) = src[0..keep.len];
            }
        }
        var cell = a.cell;
        if (st.rhombohedral) {
            cell = transformCell(cell, p_hex_from_rhomb);
            for (a.coords) |*x| x.* = transformCoords(x.*, p_hex_from_rhomb, .{ 0, 0, 0 });
        }
        s.number = st.number;
        s.setting = st;
        s.cell = transformCell(cell, st.P);
        s.sites = try r.arena.alloc(Site, a.species.len);
        for (s.sites, 0..) |*site, i| site.* = .{
            .label = a.labels[i],
            .symbol = a.species[i],
            .z = atomicNumber(a.species[i]),
            .xyz = transformCoords(a.coords[i], st.P, st.p),
            .occupancy = a.occ[i],
            .u_iso = a.u[i],
        };
        return s;
    }
};

/// What the reader did, as JSON: where the symmetry came from, the setting it applied, and
/// what the file declared and had (its cell and its number of sites as read).
pub fn account(s: *const Structure, gpa: std.mem.Allocator) ![]u8 {
    const describe = if (s.setting.standard()) try gpa.dupe(u8, "standard") else try s.setting.describe(gpa);
    defer gpa.free(describe);
    var out: std.Io.Writer.Allocating = .init(gpa);
    errdefer out.deinit();
    try std.json.Stringify.value(.{
        .block = s.block,
        .symmetry_provenance = s.provenance,
        .setting = describe,
        .setting_note = s.setting.note,
        .P = s.setting.P,
        .p_24ths = s.setting.p,
        .rhombohedral_input = s.setting.rhombohedral,
        .declared = .{ .it_number = s.declared_it, .hall = s.declared_hall, .hm = s.declared_hm },
        .cell_as_read = s.cell_as_read,
        .sites_as_read = s.sites_as_read,
        .sites_given_default_b = s.defaulted_b,
        .thermal_parameters_in_file = s.thermal_tags,
    }, .{}, &out.writer);
    return out.toOwnedSlice();
}

test "a CIF that names its group: nickel" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var r: Reader = .{ .arena = arena.allocator() };
    const s = try r.read(
        \\# a comment
        \\data_Ni
        \\_symmetry_space_group_name_H-M   'F m -3 m'
        \\_cell_length_a   3.5240(2)
        \\_cell_length_b   3.5240
        \\_cell_length_c   3.5240
        \\_cell_angle_alpha   90
        \\_cell_angle_beta   90.
        \\_cell_angle_gamma   90.000
        \\_symmetry_Int_Tables_number   225
        \\loop_
        \\ _atom_site_label
        \\ _atom_site_type_symbol
        \\ _atom_site_fract_x
        \\ _atom_site_fract_y
        \\ _atom_site_fract_z
        \\ _atom_site_occupancy
        \\  Ni1  Ni2+  0.00000000  0.00000000  .0  1
    );
    try std.testing.expectEqual(@as(u32, 225), s.number);
    try std.testing.expect(s.setting.standard());
    try std.testing.expectEqual(@as(usize, 1), s.sites.len);
    try std.testing.expectEqualStrings("Ni", s.sites[0].symbol);
    try std.testing.expectEqual(@as(u32, 28), s.sites[0].z);
    try std.testing.expectApproxEqAbs(@as(f64, 3.524), s.cell[0], 1e-12);
    try std.testing.expectEqual(@as(usize, 1), s.defaulted_b);
}

test "a P1 CIF: the group from the sites" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var r: Reader = .{ .arena = arena.allocator() };
    const s = try r.read(
        \\data_x
        \\_cell_length_a 3.524
        \\_cell_length_b 3.524
        \\_cell_length_c 3.524
        \\_cell_angle_alpha 90
        \\_cell_angle_beta 90
        \\_cell_angle_gamma 90
        \\_symmetry_space_group_name_H-M 'P 1'
        \\loop_
        \\_atom_site_label
        \\_atom_site_fract_x
        \\_atom_site_fract_y
        \\_atom_site_fract_z
        \\_atom_site_U_iso_or_equiv
        \\Ni1 0 0 0 0.005
        \\Ni2 0.5 0.5 0 0.005
        \\Ni3 0.5 0 0.5 0.005
        \\Ni4 0 0.5 0.5 0.005
    );
    try std.testing.expectEqual(@as(u32, 225), s.number);
    try std.testing.expectEqual(@as(usize, 1), s.sites.len);
    try std.testing.expectEqual(@as(usize, 0), s.defaulted_b);
}

test "every group, scrambled by a known setting, comes back" {
    // the round trip: the standard operators in the bases cab, bca and (a, b, a + c), each
    // with an origin shift, are taken back to operators equal to the standard ones
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var r: Reader = .{ .arena = arena.allocator() };
    var prng: std.Random.DefaultPrng = .init(0);
    const rand = prng.random();
    const scramble = [_]Mat{ I3, .{ 0, 0, 1, 1, 0, 0, 0, 1, 0 }, .{ 0, 1, 0, 0, 0, 1, 1, 0, 0 }, .{ 1, 0, 1, 0, 1, 0, 0, 0, 1 } };
    var a: [MAX]i64 = undefined;
    var b: [MAX]i64 = undefined;
    for (1..231) |ni| {
        const n: u32 = @intCast(ni);
        const stdo = try OpList.fromSlice(try r.stdOps(n));
        const want = try stdo.fingerprint(&b);
        for (scramble) |P| {
            const v = [3]i32{ rand.intRangeLessThan(i32, 0, DEN), rand.intRangeLessThan(i32, 0, DEN), rand.intRangeLessThan(i32, 0, DEN) };
            var moved = try stdo.transformed(try group.inverseUnimodular(P), .{ 0, 0, 0 });
            for (moved.ops[0..moved.n]) |*o| {
                const Wv = group.matVec(o.W, v);
                o.* = Op.init(o.W, .{ o.w[0] + v[0] - Wv[0], o.w[1] + v[1] - Wv[1], o.w[2] + v[2] - Wv[2] });
            }
            const st = r.findSetting(moved.slice(), n) catch |err| {
                std.debug.print("SG {d}, P {any}: {s}\n", .{ n, P, r.why() });
                return err;
            };
            const back = try (try group.close(moved.slice())).transformed(st.P, st.p);
            try std.testing.expectEqualSlices(i64, want, try back.fingerprint(&a));
        }
    }
}
