//! Space-group operators on fractional coordinates, in exact arithmetic: x ↦ W x + w / DEN,
//! with W an integer matrix and w in 24ths (origin shifts of 1/8 and 1/12 occur, so 24). Hall
//! symbols, closure under composition, changes of basis, and the integer codes by which two
//! operator sets are compared.
const std = @import("std");

/// the denominator of the translations
pub const DEN = 24;

pub const Mat = [9]i32;
pub const I3: Mat = .{ 1, 0, 0, 0, 1, 0, 0, 0, 1 };

pub fn matMul(a: Mat, b: Mat) Mat {
    var o: Mat = undefined;
    for (0..3) |i| for (0..3) |j| {
        var s: i32 = 0;
        for (0..3) |k| s += a[i * 3 + k] * b[k * 3 + j];
        o[i * 3 + j] = s;
    };
    return o;
}

pub fn matVec(a: Mat, v: [3]i32) [3]i32 {
    var o: [3]i32 = undefined;
    for (0..3) |i| o[i] = a[i * 3] * v[0] + a[i * 3 + 1] * v[1] + a[i * 3 + 2] * v[2];
    return o;
}

pub fn det(m: Mat) i32 {
    return m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
}

/// The adjugate: det(m) · m⁻¹.
pub fn adjugate(m: Mat) Mat {
    return .{
        m[4] * m[8] - m[5] * m[7], m[2] * m[7] - m[1] * m[8], m[1] * m[5] - m[2] * m[4],
        m[5] * m[6] - m[3] * m[8], m[0] * m[8] - m[2] * m[6], m[2] * m[3] - m[0] * m[5],
        m[3] * m[7] - m[4] * m[6], m[1] * m[6] - m[0] * m[7], m[0] * m[4] - m[1] * m[3],
    };
}

/// The inverse of a change of basis, which must have determinant +1 (a −1 would be a mirror,
/// and would take a crystal to its enantiomorph).
pub fn inverseUnimodular(m: Mat) !Mat {
    if (det(m) != 1) return error.NotUnimodular;
    return adjugate(m);
}

/// A Seitz operator (W, w): x ↦ W x + w / DEN, w in [0, DEN).
pub const Op = struct {
    W: Mat,
    w: [3]i32,

    pub fn init(W: Mat, w: [3]i32) Op {
        return .{ .W = W, .w = .{ @mod(w[0], DEN), @mod(w[1], DEN), @mod(w[2], DEN) } };
    }

    pub fn eql(a: Op, b: Op) bool {
        return std.mem.eql(i32, &a.W, &b.W) and std.mem.eql(i32, &a.w, &b.w);
    }

    /// a ∘ b: b first.
    pub fn compose(a: Op, b: Op) Op {
        const t = matVec(a.W, b.w);
        return init(matMul(a.W, b.W), .{ t[0] + a.w[0], t[1] + a.w[1], t[2] + a.w[2] });
    }

    pub fn isTranslation(o: Op) bool {
        return std.mem.eql(i32, &o.W, &I3);
    }

    pub fn isIdentity(o: Op) bool {
        return o.isTranslation() and o.w[0] == 0 and o.w[1] == 0 and o.w[2] == 0;
    }

    /// The order of the rotation part (1 … 6).
    pub fn order(o: Op) !u32 {
        var M = I3;
        for (1..7) |k| {
            M = matMul(M, o.W);
            if (std.mem.eql(i32, &M, &I3)) return @intCast(k);
        }
        return error.NotCrystallographic;
    }

    /// The intrinsic (screw or glide) part of the translation, in 24ths: (1/k) Σ_{j<k} Wʲ w.
    pub fn intrinsic(o: Op) ![3]i32 {
        const k: i32 = @intCast(try o.order());
        var acc: [3]i32 = .{ 0, 0, 0 };
        var M = I3;
        for (0..@intCast(k)) |_| {
            const t = matVec(M, o.w);
            for (0..3) |i| acc[i] += t[i];
            M = matMul(M, o.W);
        }
        for (acc) |v| if (@mod(v, k) != 0) return error.NotCrystallographic;
        return .{ @mod(@divExact(acc[0], k), DEN), @mod(@divExact(acc[1], k), DEN), @mod(@divExact(acc[2], k), DEN) };
    }

    /// The operator as one integer: the rotation's entries in base 41 about 20, then the
    /// translation in 24ths (41⁹ · 24³ < 2⁶³). Equal codes, equal operators; sorted codes are a
    /// set's fingerprint.
    pub fn code(o: Op) !i64 {
        var c: i64 = 0;
        for (o.W) |v| {
            if (v < -20 or v > 20) return error.NotAGroup;
            c = c * 41 + (v + 20);
        }
        return ((c * DEN + o.w[0]) * DEN + o.w[1]) * DEN + o.w[2];
    }

    /// The image of a point, fractional coordinates (not wrapped).
    pub fn apply(o: Op, x: [3]f64) [3]f64 {
        var y: [3]f64 = undefined;
        for (0..3) |i| {
            y[i] = @as(f64, @floatFromInt(o.w[i])) / DEN;
            for (0..3) |k| y[i] += @as(f64, @floatFromInt(o.W[i * 3 + k])) * x[k];
        }
        return y;
    }

    /// The operator in another basis: (P⁻¹ W P, P⁻¹ (W p + w − p)), p in 24ths.
    pub fn transform(o: Op, P: Mat, Pinv: Mat, p: [3]i32) Op {
        const Wp = matVec(o.W, p);
        const t = matVec(Pinv, .{ Wp[0] + o.w[0] - p[0], Wp[1] + o.w[1] - p[1], Wp[2] + o.w[2] - p[2] });
        return init(matMul(Pinv, matMul(o.W, P)), t);
    }
};

pub const identity: Op = .{ .W = I3, .w = .{ 0, 0, 0 } };

/// the largest space group's order with its centring (Fm-3m: 192)
pub const MAX_OPS = 192;

pub const OpList = struct {
    ops: [MAX_OPS]Op = undefined,
    n: usize = 0,

    pub fn slice(self: *const OpList) []const Op {
        return self.ops[0..self.n];
    }

    pub fn append(self: *OpList, o: Op) !void {
        if (self.n == MAX_OPS) return error.NotAGroup;
        self.ops[self.n] = o;
        self.n += 1;
    }

    pub fn fromSlice(ops: []const Op) !OpList {
        var l: OpList = .{};
        for (ops) |o| try l.append(o);
        return l;
    }

    /// The set's fingerprint: its operators' codes, sorted (with repeats).
    pub fn fingerprint(self: *const OpList, out: *[MAX_OPS]i64) ![]i64 {
        for (self.slice(), 0..) |o, i| out[i] = try o.code();
        const f = out[0..self.n];
        std.mem.sort(i64, f, {}, std.sort.asc(i64));
        return f;
    }

    /// The set in another basis (P of determinant +1).
    pub fn transformed(self: *const OpList, P: Mat, p: [3]i32) !OpList {
        const Pinv = try inverseUnimodular(P);
        var l: OpList = .{ .n = self.n };
        for (self.slice(), 0..) |o, i| l.ops[i] = o.transform(P, Pinv, p);
        return l;
    }
};

/// A set of operator codes, for closure: open addressing in 512 slots (MAX_OPS ≤ 256).
const CodeSet = struct {
    slots: [512]i64 = @splat(-1),

    fn insert(self: *CodeSet, c: i64) bool {
        var h: usize = @intCast((@as(u64, @bitCast(c)) *% 0x9E3779B97F4A7C15) >> 55);
        while (self.slots[h] != -1) {
            if (self.slots[h] == c) return false;
            h = (h + 1) & 511;
        }
        self.slots[h] = c;
        return true;
    }
};

/// The group the operators generate, translations mod 1, in the order of their codes;
/// error.NotAGroup past MAX_OPS elements.
pub fn close(gens: []const Op) !OpList {
    var g: OpList = .{};
    var seen: CodeSet = .{};
    _ = seen.insert(try identity.code());
    try g.append(identity);
    for (gens) |o0| {
        const o = Op.init(o0.W, o0.w);
        if (seen.insert(try o.code())) try g.append(o);
    }
    var i: usize = 0;
    while (i < g.n) : (i += 1) {
        for (gens) |h| {
            const o = g.ops[i].compose(Op.init(h.W, h.w));
            if (seen.insert(try o.code())) try g.append(o);
        }
    }
    sortByCode(g.ops[0..g.n]);
    return g;
}

fn sortByCode(ops: []Op) void {
    const by = struct {
        fn lt(_: void, a: Op, b: Op) bool {
            return (a.code() catch 0) < (b.code() catch 0);
        }
    };
    std.mem.sort(Op, ops, {}, by.lt);
}

/// Whether the operators are nothing but the identity (a P1 description).
pub fn trivial(ops: []const Op) bool {
    for (ops) |o| if (!o.isIdentity()) return false;
    return true;
}

/// The Bravais centring's letter from the pure translations, or null when they are not one
/// of the seven.
pub fn latticeLetter(ops: []const Op) ?u8 {
    var t: [MAX_OPS][3]i32 = undefined;
    var n: usize = 0;
    for (ops) |o| {
        if (!o.isTranslation() or o.isIdentity()) continue;
        var dup = false;
        for (t[0..n]) |q| dup = dup or std.mem.eql(i32, &q, &o.w);
        if (dup) continue;
        t[n] = o.w;
        n += 1;
    }
    const H = DEN / 2;
    const T1 = DEN / 3;
    const T2 = 2 * DEN / 3;
    const Set = struct { letter: u8, v: []const [3]i32 };
    const known = [_]Set{
        .{ .letter = 'P', .v = &.{} },
        .{ .letter = 'A', .v = &.{.{ 0, H, H }} },
        .{ .letter = 'B', .v = &.{.{ H, 0, H }} },
        .{ .letter = 'C', .v = &.{.{ H, H, 0 }} },
        .{ .letter = 'I', .v = &.{.{ H, H, H }} },
        .{ .letter = 'R', .v = &.{ .{ T2, T1, T1 }, .{ T1, T2, T2 } } },
        .{ .letter = 'F', .v = &.{ .{ 0, H, H }, .{ H, 0, H }, .{ H, H, 0 } } },
    };
    outer: for (known) |k| {
        if (k.v.len != n) continue;
        for (k.v) |v| {
            var found = false;
            for (t[0..n]) |q| found = found or std.mem.eql(i32, &q, &v);
            if (!found) continue :outer;
        }
        return k.letter;
    }
    return null;
}

/// The generators a Hall symbol names (S. R. Hall, Acta Cryst. A37 (1981) 517; with the
/// change-of-origin suffix "(0 0 4)" in twelfths). A matched pair of quotes around it is
/// dropped (a bare trailing " is the double-prime axis).
pub fn hallGenerators(symbol: []const u8) !OpList {
    const H = DEN / 2;
    const T1 = DEN / 3;
    const T2 = 2 * DEN / 3;
    const Q = DEN / 4;
    var s = std.mem.trim(u8, symbol, " \t\r\n");
    if (s.len >= 2 and s[0] == s[s.len - 1] and (s[0] == '\'' or s[0] == '"')) s = std.mem.trim(u8, s[1 .. s.len - 1], " \t");
    var shift: [3]i32 = .{ 0, 0, 0 };
    if (std.mem.endsWith(u8, s, ")")) {
        const open = std.mem.lastIndexOfScalar(u8, s, '(') orelse return error.BadHallSymbol;
        var it = std.mem.tokenizeAny(u8, s[open + 1 .. s.len - 1], " ,\t");
        for (&shift) |*v| v.* = (std.fmt.parseInt(i32, it.next() orelse return error.BadHallSymbol, 10) catch return error.BadHallSymbol) * (DEN / 12);
        if (it.next() != null) return error.BadHallSymbol;
        s = std.mem.trim(u8, s[0..open], " \t");
    }
    var toks = std.mem.tokenizeAny(u8, s, " \t");
    var lat = toks.next() orelse return error.BadHallSymbol;
    var gens: OpList = .{};
    const centro = lat[0] == '-';
    if (centro) lat = lat[1..];
    if (lat.len != 1) return error.BadHallSymbol;
    const latt: []const [3]i32 = switch (std.ascii.toUpper(lat[0])) {
        'P' => &.{},
        'A' => &.{.{ 0, H, H }},
        'B' => &.{.{ H, 0, H }},
        'C' => &.{.{ H, H, 0 }},
        'I' => &.{.{ H, H, H }},
        'R' => &.{ .{ T2, T1, T1 }, .{ T1, T2, T2 } },
        'S' => &.{ .{ T1, T1, T2 }, .{ T2, T2, T1 } },
        'T' => &.{ .{ T1, T2, T1 }, .{ T2, T1, T2 } },
        'F' => &.{ .{ 0, H, H }, .{ H, 0, H }, .{ H, H, 0 } },
        else => return error.BadHallSymbol,
    };
    var index: usize = 0;
    var prev_order: u32 = 0;
    var prev_axis: u8 = 0;
    while (toks.next()) |tok0| : (index += 1) {
        var tok = tok0;
        const neg = tok[0] == '-';
        if (neg) tok = tok[1..];
        if (tok.len == 0) return error.BadHallSymbol;
        const order: u32 = switch (tok[0]) {
            '1', '2', '3', '4', '6' => tok[0] - '0',
            else => return error.BadHallSymbol,
        };
        tok = tok[1..];
        var axis: u8 = 0;
        if (tok.len > 0 and std.mem.indexOfScalar(u8, "xyz'\"*", tok[0]) != null) {
            axis = tok[0];
            tok = tok[1..];
        }
        if (axis == 0) {
            // Hall's implicit axes
            axis = if (order == 1 or index == 0) 'z' else if (order == 3) '*' else if (order == 2 and (prev_order == 2 or prev_order == 4)) 'x' else if (order == 2 and (prev_order == 3 or prev_order == 6)) '\'' else 'z';
        }
        var W: Mat = undefined;
        if (axis == '\'' or axis == '"') {
            if (order != 2) return error.BadHallSymbol;
            const ref = if (prev_axis == 'x' or prev_axis == 'y' or prev_axis == 'z') prev_axis else 'z';
            W = switch (ref) {
                'x' => if (axis == '\'') .{ -1, 0, 0, 0, 0, -1, 0, -1, 0 } else .{ -1, 0, 0, 0, 0, 1, 0, 1, 0 },
                'y' => if (axis == '\'') .{ 0, 0, -1, 0, -1, 0, -1, 0, 0 } else .{ 0, 0, 1, 0, -1, 0, 1, 0, 0 },
                else => if (axis == '\'') .{ 0, -1, 0, -1, 0, 0, 0, 0, -1 } else .{ 0, 1, 0, 1, 0, 0, 0, 0, -1 },
            };
        } else if (axis == '*') {
            if (order != 3) return error.BadHallSymbol;
            W = .{ 0, 0, 1, 1, 0, 0, 0, 1, 0 };
        } else {
            W = switch (axis) {
                'x' => switch (order) {
                    1 => I3,
                    2 => .{ 1, 0, 0, 0, -1, 0, 0, 0, -1 },
                    3 => .{ 1, 0, 0, 0, 0, -1, 0, 1, -1 },
                    4 => .{ 1, 0, 0, 0, 0, -1, 0, 1, 0 },
                    else => .{ 1, 0, 0, 0, 1, -1, 0, 1, 0 },
                },
                'y' => switch (order) {
                    1 => I3,
                    2 => .{ -1, 0, 0, 0, 1, 0, 0, 0, -1 },
                    3 => .{ -1, 0, 1, 0, 1, 0, -1, 0, 0 },
                    4 => .{ 0, 0, 1, 0, 1, 0, -1, 0, 0 },
                    else => .{ 0, 0, 1, 0, 1, 0, -1, 0, 1 },
                },
                else => switch (order) {
                    1 => I3,
                    2 => .{ -1, 0, 0, 0, -1, 0, 0, 0, 1 },
                    3 => .{ 0, -1, 0, 1, -1, 0, 0, 0, 1 },
                    4 => .{ 0, -1, 0, 1, 0, 0, 0, 0, 1 },
                    else => .{ 1, -1, 0, 1, 0, 0, 0, 0, 1 },
                },
            };
        }
        if (neg) {
            for (&W) |*v| v.* = -v.*;
        }
        var w: [3]i32 = .{ 0, 0, 0 };
        for (tok) |ch| {
            if (ch >= '1' and ch <= '6') {
                // a screw: k / order of a turn along the axis
                const num: i32 = DEN * @as(i32, ch - '0');
                if (@mod(num, @as(i32, @intCast(order))) != 0) return error.BadHallSymbol;
                const step = @divExact(num, @as(i32, @intCast(order)));
                switch (axis) {
                    'x' => w[0] += step,
                    'y' => w[1] += step,
                    'z' => w[2] += step,
                    else => return error.BadHallSymbol,
                }
            } else {
                const tv: [3]i32 = switch (ch) {
                    'a' => .{ H, 0, 0 },
                    'b' => .{ 0, H, 0 },
                    'c' => .{ 0, 0, H },
                    'n' => .{ H, H, H },
                    'u' => .{ Q, 0, 0 },
                    'v' => .{ 0, Q, 0 },
                    'w' => .{ 0, 0, Q },
                    'd' => .{ Q, Q, Q },
                    else => return error.BadHallSymbol,
                };
                for (0..3) |j| w[j] += tv[j];
            }
        }
        try gens.append(Op.init(W, w));
        prev_order = order;
        prev_axis = axis;
    }
    if (centro) try gens.append(.{ .W = .{ -1, 0, 0, 0, -1, 0, 0, 0, -1 }, .w = .{ 0, 0, 0 } });
    for (latt) |t| try gens.append(.{ .W = I3, .w = t });
    // the origin shift: (W, w) ↦ (W, w + (I − W) v)
    for (gens.ops[0..gens.n]) |*g| {
        const Wv = matVec(g.W, shift);
        g.* = Op.init(g.W, .{ g.w[0] + shift[0] - Wv[0], g.w[1] + shift[1] - Wv[1], g.w[2] + shift[2] - Wv[2] });
    }
    return gens;
}

/// The group a Hall symbol generates.
pub fn hallOps(symbol: []const u8) !OpList {
    const g = try hallGenerators(symbol);
    return close(g.slice());
}

/// An operator from its CIF form, "x,y,z" / "-x+1/2, y, z+0.5" / "1/2+X-Y,…": integer
/// coefficients on x, y and z; a translation that is a multiple of 1/24 (decimals allowed).
pub fn parseXyz(text: []const u8) !Op {
    var s = std.mem.trim(u8, text, " \t\r\n");
    s = std.mem.trim(u8, s, "'\"");
    var parts = std.mem.splitScalar(u8, s, ',');
    var W: Mat = @splat(0);
    var w: [3]i32 = .{ 0, 0, 0 };
    var row: usize = 0;
    while (parts.next()) |part| : (row += 1) {
        if (row == 3) return error.BadOperator;
        var comp: [64]u8 = undefined;
        var n: usize = 0;
        for (part) |c| {
            if (c == ' ' or c == '\t') continue;
            if (n == comp.len) return error.BadOperator;
            comp[n] = std.ascii.toLower(c);
            n += 1;
        }
        if (n == 0) return error.BadOperator;
        // terms: a sign, then everything to the next sign
        var t_num: i64 = 0; // the translation, in 1/(24 · 10⁶)
        var i: usize = 0;
        while (i < n) {
            var sign: i64 = 1;
            if (comp[i] == '+' or comp[i] == '-') {
                if (comp[i] == '-') sign = -1;
                i += 1;
            }
            const start = i;
            while (i < n and comp[i] != '+' and comp[i] != '-') : (i += 1) {}
            var term: []const u8 = comp[start..i];
            var variable: ?usize = null;
            var buf: [64]u8 = undefined;
            for ("xyz", 0..) |v, vi| if (std.mem.indexOfScalar(u8, term, v)) |at| {
                // (the first of x, y, z named; a second one is then a bad coefficient)
                variable = vi;
                @memcpy(buf[0..at], term[0..at]);
                @memcpy(buf[at .. term.len - 1], term[at + 1 ..]);
                term = buf[0 .. term.len - 1];
                break;
            };
            term = std.mem.trim(u8, term, "*");
            // the coefficient as a fraction p / q
            var p: i64 = 1;
            var q: i64 = 1;
            if (term.len > 0) {
                if (std.mem.indexOfScalar(u8, term, '/')) |sl| {
                    p = std.fmt.parseInt(i64, term[0..sl], 10) catch return error.BadOperator;
                    q = std.fmt.parseInt(i64, term[sl + 1 ..], 10) catch return error.BadOperator;
                    if (q == 0) return error.BadOperator;
                } else if (std.mem.indexOfScalar(u8, term, '.')) |dot| {
                    // a decimal, exactly: digits after the point over a power of ten
                    const frac = term[dot + 1 ..];
                    if (frac.len > 9) return error.BadOperator;
                    const whole = if (dot == 0) 0 else std.fmt.parseInt(i64, term[0..dot], 10) catch return error.BadOperator;
                    const f = if (frac.len == 0) 0 else std.fmt.parseInt(i64, frac, 10) catch return error.BadOperator;
                    q = std.math.pow(i64, 10, @intCast(frac.len));
                    p = whole * q + f;
                } else p = std.fmt.parseInt(i64, term, 10) catch return error.BadOperator;
            }
            if (variable) |v| {
                if (@mod(p, q) != 0) return error.BadOperator;
                W[row * 3 + v] += @intCast(sign * @divExact(p, q));
            } else {
                // p / q in 24ths must be whole
                if (@mod(p * DEN, q) != 0) return error.BadOperator;
                t_num += sign * @divExact(p * DEN, q);
            }
        }
        w[row] = @intCast(@mod(t_num, DEN));
    }
    if (row != 3) return error.BadOperator;
    return Op.init(W, w);
}

test "Hall symbols close to the space groups' orders" {
    const tables = @import("tables.zig");
    // P1, P-1, P2_1/c, Fm-3m with its centring, P6_3/mmc, R-3m on hexagonal axes, Fd-3m
    const cases = [_]struct { sg: u32, order: usize, letter: u8 }{
        .{ .sg = 1, .order = 1, .letter = 'P' },   .{ .sg = 2, .order = 2, .letter = 'P' },
        .{ .sg = 14, .order = 4, .letter = 'P' },  .{ .sg = 225, .order = 192, .letter = 'F' },
        .{ .sg = 194, .order = 24, .letter = 'P' }, .{ .sg = 166, .order = 36, .letter = 'R' },
        .{ .sg = 227, .order = 192, .letter = 'F' }, .{ .sg = 38, .order = 8, .letter = 'A' },
    };
    for (cases) |c| {
        const g = try hallOps(tables.std_hall[c.sg - 1]);
        try std.testing.expectEqual(c.order, g.n);
        try std.testing.expectEqual(@as(?u8, c.letter), latticeLetter(g.slice()));
    }
    // every Hall symbol tabulated parses and closes, and its lattice letter is its name's
    for (tables.std_hall, tables.sg_names) |h, name| {
        const g = try hallOps(h);
        try std.testing.expectEqual(@as(?u8, name[0]), latticeLetter(g.slice()));
    }
    for (tables.origin1_hall) |e| _ = try hallOps(e.hall);
    for (tables.rhomb_hall) |e| _ = try hallOps(e.hall);
}

test "CIF operators parse" {
    const o = try parseXyz("-x+1/2, y , 0.25+Z");
    try std.testing.expectEqualSlices(i32, &.{ -1, 0, 0, 0, 1, 0, 0, 0, 1 }, &o.W);
    try std.testing.expectEqualSlices(i32, &.{ 12, 0, 6 }, &o.w);
    const p = try parseXyz("'x-y,x,z+5/6'");
    try std.testing.expectEqualSlices(i32, &.{ 1, -1, 0, 1, 0, 0, 0, 0, 1 }, &p.W);
    try std.testing.expectEqualSlices(i32, &.{ 0, 0, 20 }, &p.w);
    try std.testing.expectError(error.BadOperator, parseXyz("x,y"));
    try std.testing.expectError(error.BadOperator, parseXyz("x+1/5,y,z"));
}
