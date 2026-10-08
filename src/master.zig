//! A candidate phase: its master pattern on the sphere and everything derived from it once.
//!
//! Files: ebsdsim `.npz` (the fundamental sector plus the point group's operators, expanded
//! here to full hemispheres) and EMsoft / EMsoftOO master `.h5` (both hemispheres stored,
//! energies averaged with the Monte Carlo weights).
//!
//! The master is kept as two side × side Lambert sheets (lambert.zig), normalized to zero mean
//! and unit variance over the sphere, so every score computed from it is well scaled in f32.
//! `raw` is as loaded; the engine keeps what its filters make of it (one copy per pattern
//! resolution, baked on the GPU). `Harmonics` are computed on the host in f64 from such sheets.
const std = @import("std");
const rot = @import("rot.zig");
const sym = @import("sym.zig");
const sh = @import("sh.zig");
const lambert = @import("lambert.zig");
const File = @import("io/file.zig").File;
const Npz = @import("io/npz.zig").Npz;
const hdf5 = @import("io/hdf5.zig");

const Allocator = std.mem.Allocator;

pub const Master = struct {
    gpa: Allocator,
    name: []u8,
    /// sheet edge in pixels
    side: u32,
    /// [north | south], side · side each, as loaded (normalized)
    raw: []f32,
    /// crystallographic point group (1–32), 0 when the file does not say
    point_group: u32 = 0,
    space_group: u32 = 0,
    laue: sym.Laue = .o,
    /// the proper rotations of the point group: what makes two orientations the same (the Laue
    /// class's own only for a crystal with a centre of symmetry)
    rotation: sym.Laue = .o,
    voltage_kv: f64 = 0,
    /// "ebsdsim" or "emsoft"
    source: []const u8 = "",
    /// the point-group operators the master itself obeys (row-major 3 × 3 each), measured
    /// from the raw sheets by `measureSymmetry`
    ops: []rot.Mat3 = &.{},

    pub fn deinit(self: *Master) void {
        self.gpa.free(self.name);
        self.gpa.free(self.raw);
        self.gpa.free(self.ops);
    }

    /// Intensity of the raw master in direction `v`.
    fn rawAt(self: *const Master, v: rot.Vec3) f64 {
        const uv = lambert.sphereToSquare(v);
        return lambert.sample(if (v[2] >= 0) self.north(self.raw) else self.south(self.raw), self.side, uv[0], uv[1]);
    }

    /// Find which operators of the Laue class (each rotation, and each rotation with an
    /// inversion) leave the raw master unchanged, by correlating it with itself over random
    /// directions. A filter that does not commute with the crystal's symmetry is followed by an
    /// average over exactly these.
    pub fn measureSymmetry(self: *Master) !void {
        const gpa = self.gpa;
        var found: std.ArrayList(rot.Mat3) = .empty;
        errdefer found.deinit(gpa);
        var prng = std.Random.DefaultPrng.init(12345);
        const r = prng.random();
        const n = 1500;
        const dirs = try gpa.alloc(rot.Vec3, n);
        defer gpa.free(dirs);
        const base = try gpa.alloc(f64, n);
        defer gpa.free(base);
        var mean: f64 = 0;
        for (0..n) |i| {
            // uniform on the sphere
            const z = 2 * r.float(f64) - 1;
            const phi = 2 * std.math.pi * r.float(f64);
            const st = @sqrt(@max(0.0, 1 - z * z));
            dirs[i] = .{ st * @cos(phi), st * @sin(phi), z };
            base[i] = self.rawAt(dirs[i]);
            mean += base[i];
        }
        mean /= n;
        var var0: f64 = 0;
        for (base) |b| var0 += (b - mean) * (b - mean);
        for (sym.operators(self.laue)) |q| {
            const R = rot.toMat(q);
            for ([_]f64{ 1, -1 }) |sgn| {
                var g = R;
                for (&g) |*e| e.* *= sgn;
                var cov: f64 = 0;
                var var1: f64 = 0;
                var m1: f64 = 0;
                for (dirs) |d| m1 += self.rawAt(rot.matVec(g, d));
                m1 /= n;
                for (dirs, base) |d, b| {
                    const v = self.rawAt(rot.matVec(g, d)) - m1;
                    cov += v * (b - mean);
                    var1 += v * v;
                }
                if (cov / @sqrt(@max(var0 * var1, 1e-300)) > 0.97) try found.append(gpa, g);
            }
        }
        gpa.free(self.ops);
        self.ops = try found.toOwnedSlice(gpa);
    }

    pub fn north(self: *const Master, sheets: []const f32) []const f32 {
        return sheets[0 .. @as(usize, self.side) * self.side];
    }

    pub fn south(self: *const Master, sheets: []const f32) []const f32 {
        return sheets[@as(usize, self.side) * self.side ..];
    }

    /// Intensity of `sheets` (this master's raw sheets, or a filtered copy) in crystal
    /// direction `v`.
    pub fn at(self: *const Master, sheets: []const f32, v: rot.Vec3) f64 {
        const uv = lambert.sphereToSquare(v);
        return lambert.sample(if (v[2] >= 0) self.north(sheets) else self.south(sheets), self.side, uv[0], uv[1]);
    }
};

/// Zero mean, unit variance over both sheets (the Lambert map is equal-area).
fn normalize(sheets: []f32) void {
    var s: f64 = 0;
    for (sheets) |v| s += v;
    const mean = s / @as(f64, @floatFromInt(sheets.len));
    var ss: f64 = 0;
    for (sheets) |v| ss += (v - mean) * (v - mean);
    const sd = @sqrt(ss / @as(f64, @floatFromInt(sheets.len)));
    const k: f64 = if (sd > 0) 1 / sd else 1;
    for (sheets) |*v| v.* = @floatCast((v.* - mean) * k);
}

fn baseName(gpa: Allocator, path: []const u8) ![]u8 {
    var start: usize = 0;
    // the browser adapter's token for a picked file is "@<id>:<name>"
    if (path.len > 1 and path[0] == '@') {
        if (std.mem.indexOfScalar(u8, path, ':')) |c| start = c + 1;
    }
    for (path[start..], start..) |c, i| if (c == '/' or c == '\\') {
        start = i + 1;
    };
    var end = path.len;
    if (std.mem.lastIndexOfScalar(u8, path[start..], '.')) |d| end = start + d;
    return gpa.dupe(u8, path[start..end]);
}

/// Load a master pattern, by content: a ZIP is an ebsdsim .npz, HDF5 is EMsoft.
pub fn load(gpa: Allocator, path: []const u8) !Master {
    var f = try File.open(gpa, path);
    defer f.close();
    var sig: [4]u8 = undefined;
    try f.read(0, &sig);
    var m = if (std.mem.eql(u8, sig[0..2], "PK")) try loadEbsdsim(gpa, &f) else try loadEmsoft(gpa, &f);
    errdefer m.deinit();
    gpa.free(m.name);
    m.name = &.{};
    m.name = try baseName(gpa, path);
    try m.measureSymmetry();
    return m;
}

fn metaNumber(json: []const u8, key: []const u8) ?f64 {
    var pat_buf: [64]u8 = undefined;
    const pat = std.fmt.bufPrint(&pat_buf, "\"{s}\":", .{key}) catch return null;
    const at = std.mem.indexOf(u8, json, pat) orelse return null;
    var p = at + pat.len;
    while (p < json.len and json[p] == ' ') p += 1;
    var e = p;
    while (e < json.len and (std.ascii.isDigit(json[e]) or json[e] == '.' or json[e] == '-' or json[e] == 'e' or json[e] == 'E' or json[e] == '+')) e += 1;
    return std.fmt.parseFloat(f64, json[p..e]) catch null;
}

fn loadEbsdsim(gpa: Allocator, f: *File) !Master {
    var z = try Npz.open(gpa, f);
    defer z.deinit();
    var fs = (try z.array("fundamental_sector")) orelse (try z.array("integrated_fundamental_sector")) orelse return error.NotAMasterPattern;
    defer fs.deinit(gpa);
    var kij = (try z.array("fundamental_kij")) orelse return error.NotAMasterPattern;
    defer kij.deinit(gpa);
    var ops_a = (try z.array("pg_operators")) orelse return error.NotAMasterPattern;
    defer ops_a.deinit(gpa);
    var nrm_a = (try z.array("fs_normals")) orelse return error.NotAMasterPattern;
    defer nrm_a.deinit(gpa);
    var meta_a = try z.array("meta_json");
    defer if (meta_a) |*a| a.deinit(gpa);
    const meta: []const u8 = if (meta_a) |a| a.data else "";

    // values per direction: energy 0 and site 0 are the integrated slices of format 2; the
    // legacy array is (direction, site)
    const n_k = kij.shape[0];
    const vals = try gpa.alloc(f64, n_k);
    defer gpa.free(vals);
    if (fs.rank == 3) {
        if (fs.shape[2] != n_k) return error.NotAMasterPattern;
        for (0..n_k) |i| vals[i] = fs.at(i);
    } else {
        const n_sites = if (fs.rank == 2) fs.shape[1] else 1;
        for (0..n_k) |i| {
            var s: f64 = 0;
            for (0..n_sites) |j| s += fs.at(i * n_sites + j);
            vals[i] = s / @as(f64, @floatFromInt(n_sites));
        }
    }

    var hw: usize = 0;
    for (0..n_k) |i| hw = @max(hw, @as(usize, @intFromFloat(@max(@abs(kij.at(i * 3)), @abs(kij.at(i * 3 + 1))))));
    if (metaNumber(meta, "halfw")) |h| hw = @intFromFloat(h);
    const side = 2 * hw + 1;
    const plane = side * side;

    // the stored sector on each hemisphere's grid, with which pixels are stored
    const sec = try gpa.alloc(f32, 2 * plane);
    defer gpa.free(sec);
    const has = try gpa.alloc(bool, 2 * plane);
    defer gpa.free(has);
    @memset(sec, 0);
    @memset(has, false);
    var needs_south = false;
    for (0..n_k) |i| {
        const ix: i64 = @as(i64, @intFromFloat(kij.at(i * 3))) + @as(i64, @intCast(hw));
        const iy: i64 = @as(i64, @intFromFloat(kij.at(i * 3 + 1))) + @as(i64, @intCast(hw));
        if (ix < 0 or iy < 0 or ix >= side or iy >= side) continue;
        const south = kij.at(i * 3 + 2) <= 0;
        needs_south = needs_south or south;
        const at = (if (south) plane else 0) + @as(usize, @intCast(iy)) * side + @as(usize, @intCast(ix));
        sec[at] = @floatCast(vals[i]);
        has[at] = true;
    }

    const n_ops = ops_a.shape[0];
    const ops = try gpa.alloc(rot.Mat3, n_ops);
    defer gpa.free(ops);
    for (0..n_ops) |g| for (0..9) |e| {
        ops[g][e] = ops_a.at(g * 9 + e);
    };
    const n_nrm = if (nrm_a.rank == 2) nrm_a.shape[0] else 0;
    const normals = try gpa.alloc(rot.Vec3, n_nrm);
    defer gpa.free(normals);
    for (0..n_nrm) |i| normals[i] = .{ nrm_a.at(i * 3), nrm_a.at(i * 3 + 1), nrm_a.at(i * 3 + 2) };

    const sheets = try gpa.alloc(f32, 2 * plane);
    errdefer gpa.free(sheets);
    const eps = 1.0 / @as(f64, @floatFromInt(@max(hw, 1)));
    const last: f64 = @floatFromInt(side - 1);
    for (0..2) |hemi| for (0..side) |row| for (0..side) |col| {
        const x = 2 * @as(f64, @floatFromInt(col)) / last - 1;
        const y = 2 * @as(f64, @floatFromInt(row)) / last - 1;
        const d = lambert.squareToSphere(x, y, hemi == 1);
        // the symmetry-equivalent direction deepest inside the stored sector
        var rep = d;
        var best: f64 = -std.math.inf(f64);
        var best_in = false;
        for (ops) |g| {
            const t = rot.matVec(g, d);
            var margin: f64 = std.math.inf(f64);
            for (normals) |n| margin = @min(margin, rot.dot(t, n));
            const inside = margin >= -eps;
            if ((inside and !best_in) or (inside == best_in and margin > best)) {
                best = margin;
                best_in = inside;
                rep = t;
            }
            if (normals.len == 0) break;
        }
        const south = rep[2] < 0 and needs_south;
        const uv = lambert.sphereToSquare(rep);
        sheets[hemi * plane + row * side + col] = sampleStored(sec[(if (south) plane else 0)..][0..plane], has[(if (south) plane else 0)..][0..plane], side, uv[0], uv[1]);
    };
    normalize(sheets);

    var m: Master = .{
        .gpa = gpa,
        .name = try gpa.dupe(u8, ""),
        .side = @intCast(side),
        .raw = sheets,
        .source = "ebsdsim",
    };
    errdefer gpa.free(m.name);
    if (metaNumber(meta, "pg_num")) |v| m.point_group = @intFromFloat(v);
    if (metaNumber(meta, "space_group")) |v| m.space_group = @intFromFloat(v);
    if (metaNumber(meta, "voltage_kv")) |v| m.voltage_kv = v;
    if (m.point_group == 0 and m.space_group != 0) m.point_group = sym.pointGroupOfSpaceGroup(m.space_group) orelse 0;
    m.laue = sym.laueOfPointGroup(m.point_group) orelse laueFromOperators(ops);
    m.rotation = sym.rotationsOfPointGroup(m.point_group) orelse m.laue;
    return m;
}

/// Bilinear over the stored pixels only (weights renormalized), so the sector's edge does not
/// blend with pixels outside it; the nearest stored pixel when none of the four is stored.
fn sampleStored(sheet: []const f32, has: []const bool, side: usize, x: f64, y: f64) f32 {
    const last: f64 = @floatFromInt(side - 1);
    const fj = @min(@max((x + 1) * 0.5 * last, 0.0), last);
    const fi = @min(@max((y + 1) * 0.5 * last, 0.0), last);
    const c0: usize = @intFromFloat(@floor(fj));
    const r0: usize = @intFromFloat(@floor(fi));
    const c1 = @min(c0 + 1, side - 1);
    const r1 = @min(r0 + 1, side - 1);
    const tj = fj - @as(f64, @floatFromInt(c0));
    const ti = fi - @as(f64, @floatFromInt(r0));
    const idx = [4]usize{ r0 * side + c0, r0 * side + c1, r1 * side + c0, r1 * side + c1 };
    const wts = [4]f64{ (1 - ti) * (1 - tj), (1 - ti) * tj, ti * (1 - tj), ti * tj };
    var acc: f64 = 0;
    var wsum: f64 = 0;
    for (idx, wts) |i, w| if (has[i]) {
        acc += w * sheet[i];
        wsum += w;
    };
    if (wsum > 1e-9) return @floatCast(acc / wsum);
    // search outward for a stored pixel
    const ci: i64 = @intFromFloat(@round(fi));
    const cj: i64 = @intFromFloat(@round(fj));
    var r: i64 = 1;
    while (r < 8) : (r += 1) {
        var di: i64 = -r;
        while (di <= r) : (di += 1) {
            var dj: i64 = -r;
            while (dj <= r) : (dj += 1) {
                const ii = ci + di;
                const jj = cj + dj;
                if (ii < 0 or jj < 0 or ii >= side or jj >= side) continue;
                const k = @as(usize, @intCast(ii)) * side + @as(usize, @intCast(jj));
                if (has[k]) return sheet[k];
            }
        }
    }
    return 0;
}

/// The Laue class from a point group's matrices, when its number is not given: by the order of
/// its rotation group and whether it has a four-fold or a three-fold axis along z or a body
/// diagonal.
fn laueFromOperators(ops: []const rot.Mat3) sym.Laue {
    var n_rot: usize = 0;
    var has4 = false;
    var has3z = false;
    var has3d = false;
    var seen: [64]rot.Mat3 = undefined;
    for (ops) |g0| {
        var g = g0;
        if (rot.det(g) < 0) for (&g) |*e| {
            e.* = -e.*;
        };
        var dup = false;
        for (seen[0..n_rot]) |s| {
            var d: f64 = 0;
            for (0..9) |i| d += @abs(s[i] - g[i]);
            if (d < 1e-6) dup = true;
        }
        if (dup or n_rot == seen.len) continue;
        seen[n_rot] = g;
        n_rot += 1;
        const tr = g[0] + g[4] + g[8];
        if (@abs(tr - 1) < 1e-6) has4 = true; // 90° rotation
        if (@abs(tr) < 1e-6) { // 120° rotation
            if (@abs(g[8] - 1) < 1e-6) has3z = true else has3d = true;
        }
    }
    return switch (n_rot) {
        1 => .c1,
        2 => .c2,
        3 => .c3,
        4 => if (has4) .c4 else .d2,
        6 => if (has3z and !has4) (if (hasSixfold(seen[0..n_rot])) sym.Laue.c6 else .d3) else .d3,
        8 => .d4,
        12 => if (has3d) .t else .d6,
        24 => .o,
        else => .c1,
    };
}

fn hasSixfold(rots: []const rot.Mat3) bool {
    for (rots) |g| if (@abs(g[0] + g[4] + g[8] - 2) < 1e-6) return true; // 60° rotation
    return false;
}

fn loadEmsoft(gpa: Allocator, f: *File) !Master {
    var h = hdf5.Hdf5.open(gpa, f) catch return error.NotAMasterPattern;
    const nh_addr = (try h.find("EMData/EBSDmaster/mLPNH")) orelse (try h.find("EMData/TKDmaster/mLPNH")) orelse (try h.find("EMData/ECPmaster/mLPNH")) orelse return error.NotAMasterPattern;
    const dir: []const u8 = if ((try h.find("EMData/EBSDmaster/mLPNH")) != null) "EMData/EBSDmaster" else if ((try h.find("EMData/TKDmaster/mLPNH")) != null) "EMData/TKDmaster" else "EMData/ECPmaster";
    var path_buf: [64]u8 = undefined;
    const sh_addr = (try h.find(try std.fmt.bufPrint(&path_buf, "{s}/mLPSH", .{dir}))) orelse nh_addr;

    var nh = try h.dataset(nh_addr);
    defer nh.deinit(gpa);
    if (nh.rank < 2) return error.NotAMasterPattern;
    const side: usize = @intCast(nh.dims[nh.rank - 1]);
    if (nh.dims[nh.rank - 2] != side) return error.NotAMasterPattern;
    const plane = side * side;
    // leading axes: (atom sets, energies) or (energies)
    const n_e: usize = if (nh.rank >= 3) @intCast(nh.dims[nh.rank - 3]) else 1;
    const n_set: usize = if (nh.rank >= 4) @intCast(nh.dims[nh.rank - 4]) else 1;

    // energy weights: Monte Carlo counts per energy bin, when they match the master's bins
    const wts = try gpa.alloc(f64, n_e);
    defer gpa.free(wts);
    @memset(wts, 1);
    if (try h.find("EMData/MCOpenCL/accum_e")) |acc_addr| {
        var acc = try h.dataset(acc_addr);
        defer acc.deinit(gpa);
        if (acc.rank == 3 and acc.dims[2] == n_e and acc.dtype.size != 0) {
            const es: usize = acc.dtype.size;
            const raw = try gpa.alloc(u8, @intCast(acc.elements() * es));
            defer gpa.free(raw);
            try h.readAll(&acc, raw);
            @memset(wts, 0);
            const n: usize = @intCast(acc.elements());
            for (0..n) |i| wts[i % n_e] += try hdf5.toF64(acc.dtype, raw[i * es ..][0..es]);
            var total: f64 = 0;
            for (wts) |w| total += w;
            if (total <= 0) @memset(wts, 1);
        }
    }

    const sheets = try gpa.alloc(f32, 2 * plane);
    errdefer gpa.free(sheets);
    const acc = try gpa.alloc(f64, plane);
    defer gpa.free(acc);
    for ([_]u64{ nh_addr, sh_addr }, 0..) |addr, hemi| {
        var ds = try h.dataset(addr);
        defer ds.deinit(gpa);
        const es: usize = ds.dtype.size;
        const raw = try gpa.alloc(u8, @intCast(ds.elements() * es));
        defer gpa.free(raw);
        try h.readAll(&ds, raw);
        @memset(acc, 0);
        for (0..n_set) |s| for (0..n_e) |e| {
            const base = (s * n_e + e) * plane;
            for (0..plane) |i| acc[i] += wts[e] * try hdf5.toF64(ds.dtype, raw[(base + i) * es ..][0..es]);
        };
        for (0..plane) |i| sheets[hemi * plane + i] = @floatCast(acc[i]);
    }
    normalize(sheets);

    var m: Master = .{
        .gpa = gpa,
        .name = try gpa.dupe(u8, ""),
        .side = @intCast(side),
        .raw = sheets,
        .source = "emsoft",
    };
    errdefer gpa.free(m.name);
    if (try h.find("CrystalData/SpaceGroupNumber")) |a| m.space_group = @intFromFloat(h.number(a) catch 0);
    if (try h.find("NMLparameters/MCCLNameList/EkeV")) |a| m.voltage_kv = h.number(a) catch 0;
    m.point_group = sym.pointGroupOfSpaceGroup(m.space_group) orelse 0;
    m.laue = sym.laueOfPointGroup(m.point_group) orelse .c1;
    m.rotation = sym.rotationsOfPointGroup(m.point_group) orelse m.laue;
    return m;
}

// ── harmonics ──

/// The master's spherical-harmonic side, for one correlation bandwidth L.
pub const Harmonics = struct {
    L: u32,
    rules: sh.Rules,
    /// coefficients of the band-limited, zero-mean master and of its square (pairCount(L) each)
    f: []sh.Complex,
    f2: []sh.Complex,

    pub fn deinit(self: *Harmonics, gpa: Allocator) void {
        gpa.free(self.f);
        gpa.free(self.f2);
    }

    /// Analyze `sheets` (the master, filtered) on a Gauss–Legendre grid fine enough that
    /// nothing in the image aliases below L, then square the band-limited master on a grid that
    /// holds its square exactly.
    pub fn init(gpa: Allocator, m: *const Master, sheets: []const f32, L: u32) !Harmonics {
        const nt: u32 = @max(4 * L, 512);
        var grid = try sh.Grid.init(gpa, L, nt, 2 * nt);
        defer grid.deinit(gpa);
        const samples = try gpa.alloc(f64, grid.points());
        defer gpa.free(samples);
        for (0..grid.n_theta) |y| for (0..grid.n_phi) |j| {
            samples[y * grid.n_phi + j] = m.at(sheets, grid.direction(y, j));
        };
        const np = sh.pairCount(L);
        const f = try gpa.alloc(sh.Complex, np);
        errdefer gpa.free(f);
        try grid.analyze(gpa, samples, f);
        f[0] = .{ 0, 0 };
        const rules = sh.Rules.detect(L, f, 1e-5);
        // what the symmetry forbids is noise from interpolation: drop it
        for (0..L) |mm| for (mm..L) |l| {
            if (!rules.allows(@intCast(l), @intCast(mm))) f[sh.pair(L, @intCast(mm), @intCast(l))] = .{ 0, 0 };
        };

        var g2 = try sh.Grid.init(gpa, L, 2 * L, 4 * L);
        defer g2.deinit(gpa);
        const s2 = try gpa.alloc(f64, g2.points());
        defer gpa.free(s2);
        try g2.synthesize(gpa, f, s2);
        for (s2) |*v| v.* *= v.*;
        const f2 = try gpa.alloc(sh.Complex, np);
        errdefer gpa.free(f2);
        try g2.analyze(gpa, s2, f2);
        return .{ .L = L, .rules = rules, .f = f, .f2 = f2 };
    }
};

/// The largest power of two dividing the master's rotational order about z: the factor by
/// which the correlation's c axis shrinks (FFT lengths stay powers of two).
pub fn foldOf(rules: sh.Rules) u32 {
    return switch (rules.n_fold) {
        4 => 4,
        2, 6 => 2,
        else => 1,
    };
}

/// The master weights the spectrum kernel reads: W[((μ + μmax) · L + l) · L + k] =
/// i^{−m} f̃_{l,m} d^l_{k,m}(π/2), m = fold · μ, for k in [0, L) (interleaved re, im as f32; zero
/// where l < max(k, |m|)). k runs innermost because the kernel's neighbouring invocations hold
/// neighbouring k: they then read neighbouring words.
pub fn weights(gpa: Allocator, L: u32, coef: []const sh.Complex, tab: []const f64, fold: u32) ![]f32 {
    const mu_max = (L - 1) / fold;
    const mc = 2 * mu_max + 1;
    const out = try gpa.alloc(f32, @as(usize, L) * mc * L * 2);
    @memset(out, 0);
    for (0..L) |k| for (0..mc) |mi| {
        const mu: i64 = @as(i64, @intCast(mi)) - @as(i64, mu_max);
        const m: i64 = mu * @as(i64, fold);
        const am: u32 = @intCast(@abs(m));
        // i^{−m}
        const ph: sh.Complex = switch (@mod(-m, 4)) {
            0 => .{ 1, 0 },
            1 => .{ 0, 1 },
            2 => .{ -1, 0 },
            else => .{ 0, -1 },
        };
        var l: u32 = @max(@as(u32, @intCast(k)), am);
        while (l < L) : (l += 1) {
            const c = sh.signedCoef(L, coef, l, m);
            const d = sh.dHalfPi(tab, L, l, @intCast(k), m);
            const at = ((mi * L + l) * L + k) * 2;
            out[at] = @floatCast((c[0] * ph[0] - c[1] * ph[1]) * d);
            out[at + 1] = @floatCast((c[0] * ph[1] + c[1] * ph[0]) * d);
        }
    };
    return out;
}

test "the nickel master expands to a symmetric, normalized sphere" {
    const gpa = std.testing.allocator;
    var m = load(gpa, "testdata/nickel/Ni_master_20kV.npz") catch |e| switch (e) {
        error.FileNotFound => return, // run from the repo root
        else => return e,
    };
    defer m.deinit();
    try std.testing.expectEqual(@as(u32, 501), m.side);
    try std.testing.expectEqual(@as(u32, 32), m.point_group);
    try std.testing.expectEqual(sym.Laue.o, m.laue);
    try std.testing.expectEqualStrings("Ni_master_20kV", m.name);
    try std.testing.expectEqual(@as(usize, 48), m.ops.len);
    try std.testing.expectApproxEqAbs(@as(f64, 20), m.voltage_kv, 1e-9);
    var s: f64 = 0;
    var ss: f64 = 0;
    for (m.raw) |v| {
        s += v;
        ss += @as(f64, v) * v;
    }
    const n: f64 = @floatFromInt(m.raw.len);
    try std.testing.expectApproxEqAbs(@as(f64, 0), s / n, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f64, 1), ss / n, 1e-3);
    // m-3m: the same intensity in every equivalent direction (to the image's interpolation)
    const v = rot.normalize(.{ 0.31, 0.52, 0.80 });
    const ref = m.at(m.raw, v);
    for (sym.operators(.o)) |g| {
        try std.testing.expectApproxEqAbs(ref, m.at(m.raw, rot.qapply(g, v)), 0.05);
        try std.testing.expectApproxEqAbs(ref, m.at(m.raw, rot.scale(rot.qapply(g, v), -1)), 0.05);
    }
    // and the harmonics see that symmetry
    var hm = try Harmonics.init(gpa, &m, m.raw, 24);
    defer hm.deinit(gpa);
    try std.testing.expectEqual(@as(u32, 4), hm.rules.n_fold);
    try std.testing.expect(hm.rules.mirror and hm.rules.inversion);
    try std.testing.expect(sh.power(24, hm.f) > 0.05);
}
