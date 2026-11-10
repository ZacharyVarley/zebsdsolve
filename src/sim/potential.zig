//! The crystal potential's Fourier coefficients, as the dynamical solve reads them (the Bloch-
//! wave form: the structure factors' real and absorptive parts combined into one complex U).
//!
//! The reflections g are every lattice point (allowed by the centring) with a spacing above
//! d_min, the incident beam 000 first. The matrix couples beams g and h through U(g − h), so
//! the coefficients are kept for every difference of two reflections, in a table indexed by
//! a linear hash of the difference: index(Δ) = Δh · stride_h + Δk · stride_k + Δl + offset.
//! A beam's own hash is then hash(g) = h · stride_h + k · stride_k + l, and the difference of
//! two beams is index = hash(g) − hash(h) + offset: one subtraction.
//!
//! What depends on the voltage (the coefficients, the wavelength, the mean absorption, each
//! reflection's coupling scale) is `Voltage`; the rest is `Geometry`, computed once.
const std = @import("std");
const crystal = @import("crystal.zig");
const scatter = @import("scatter.zig");

const PI = std.math.pi;

pub const Geometry = struct {
    gpa: std.mem.Allocator,
    hmax: i32,
    kmax: i32,
    lmax: i32,
    stride_h: i32,
    stride_k: i32,
    table_size: u32,
    offset: i32,
    /// the reflections, 000 first (h, k, l)
    hkl: [][3]i32,
    /// every difference of two reflections in the table's box: its row in the table, |Δ|²
    /// (nm⁻²) and its structure-factor phase per site (Σ over the orbit of exp(−2πi Δ·r))
    diff_index: []u32,
    diff_g2: []f64,
    diff_phase: [][2]f64,
    n_sites: usize,
    /// per site: atomic number, occupancy (≤ 0 taken as 1), thermal displacement (nm)
    site_z: []u32,
    site_occ: []f64,
    site_sigma: []f64,
    /// prefactors of the potential: 0.04787801 / V and that × 0.664840340614319
    pref: f64,
    pre: f64,
    /// S per site and difference: Z² · occupancy · exp(−B s²) · Σ exp(2πi Δ·r), Δ-major
    /// (table_size × n_sites complex), the weights of the backscattered intensity
    sgh: []f32,

    pub fn deinit(self: *Geometry) void {
        const g = self.gpa;
        g.free(self.hkl);
        g.free(self.diff_index);
        g.free(self.diff_g2);
        g.free(self.diff_phase);
        g.free(self.site_z);
        g.free(self.site_occ);
        g.free(self.site_sigma);
        g.free(self.sgh);
    }

    pub fn hash(self: *const Geometry, h: [3]i32) i32 {
        return h[0] * self.stride_h + h[1] * self.stride_k + h[2];
    }

    pub fn init(gpa: std.mem.Allocator, cell: *const crystal.Cell, dmin: f64) !Geometry {
        const lim = limits(cell, dmin);
        const stride_k = 4 * lim[2] + 1;
        const stride_h = (4 * lim[1] + 1) * stride_k;
        const table_size: u32 = @intCast((4 * lim[0] + 1) * stride_h);
        const offset = 2 * lim[0] * stride_h + 2 * lim[1] * stride_k + 2 * lim[2];

        // the reflections, and the differences (every allowed point of the doubled box)
        var hkl: std.ArrayList([3]i32) = .empty;
        defer hkl.deinit(gpa);
        var diffs: std.ArrayList([3]i32) = .empty;
        defer diffs.deinit(gpa);
        try hkl.append(gpa, .{ 0, 0, 0 });
        try diffs.append(gpa, .{ 0, 0, 0 });
        for (0..2) |pass| {
            const big = pass == 1;
            const hr = if (big) 2 * lim[0] else lim[0];
            const kr = if (big) 2 * lim[1] else lim[1];
            const lr = if (big) 2 * lim[2] else lim[2];
            var h = -hr;
            while (h <= hr) : (h += 1) {
                var k = -kr;
                while (k <= kr) : (k += 1) {
                    var l = -lr;
                    while (l <= lr) : (l += 1) {
                        if (h == 0 and k == 0 and l == 0) continue;
                        if (!crystal.allowed(cell.centring, h, k, l)) continue;
                        if (big) {
                            try diffs.append(gpa, .{ h, k, l });
                        } else {
                            const len = @sqrt(@max(metricDot(.{ h, k, l }, cell.reciprocal), 0));
                            if (len <= 0 or 1 / len <= dmin) continue;
                            try hkl.append(gpa, .{ h, k, l });
                        }
                    }
                }
            }
        }

        const n_sites = cell.sites.len;
        var self: Geometry = .{
            .gpa = gpa,
            .hmax = lim[0],
            .kmax = lim[1],
            .lmax = lim[2],
            .stride_h = stride_h,
            .stride_k = stride_k,
            .table_size = table_size,
            .offset = offset,
            .hkl = try hkl.toOwnedSlice(gpa),
            .diff_index = &.{},
            .diff_g2 = &.{},
            .diff_phase = &.{},
            .n_sites = n_sites,
            .site_z = &.{},
            .site_occ = &.{},
            .site_sigma = &.{},
            .pref = 0.04787801 / cell.volume,
            .pre = 0.04787801 / cell.volume * 0.664840340614319,
            .sgh = &.{},
        };
        errdefer self.deinit();
        const nd = diffs.items.len;
        self.diff_index = try gpa.alloc(u32, nd);
        self.diff_g2 = try gpa.alloc(f64, nd);
        self.diff_phase = try gpa.alloc([2]f64, nd * n_sites);
        self.site_z = try gpa.alloc(u32, n_sites);
        self.site_occ = try gpa.alloc(f64, n_sites);
        self.site_sigma = try gpa.alloc(f64, n_sites);
        self.sgh = try gpa.alloc(f32, @as(usize, table_size) * n_sites * 2);
        @memset(self.sgh, 0);
        for (cell.sites, 0..) |s, i| {
            self.site_z[i] = s.z;
            self.site_occ[i] = if (s.occupancy > 0) s.occupancy else 1;
            // B = 8π² ⟨u²⟩; WK take σ in nm, ten times √(B / 8π²) with B in nm²
            self.site_sigma[i] = @max(1e-7, 10 * @sqrt(@max(s.b_nm2, 0) / (8 * PI * PI)));
        }
        for (diffs.items, 0..) |d, r| {
            const g2 = @max(metricDot(d, cell.reciprocal), 0);
            const idx: u32 = @intCast((d[0] + 2 * lim[0]) * stride_h + (d[1] + 2 * lim[1]) * stride_k + (d[2] + 2 * lim[2]));
            self.diff_index[r] = idx;
            self.diff_g2[r] = g2;
            const s_sq = 0.25 * g2;
            for (cell.sites, 0..) |s, si| {
                var re: f64 = 0;
                var im: f64 = 0;
                for (s.positions) |p| {
                    const ang = 2 * PI * (@as(f64, @floatFromInt(d[0])) * p[0] + @as(f64, @floatFromInt(d[1])) * p[1] + @as(f64, @floatFromInt(d[2])) * p[2]);
                    re += @cos(ang);
                    im += @sin(ang);
                }
                // the potential takes exp(−2πi Δ·r); S takes exp(+2πi Δ·r)
                self.diff_phase[r * n_sites + si] = .{ re, -im };
                const zf: f64 = @floatFromInt(s.z);
                const occ = if (s.occupancy != 0) s.occupancy else 1;
                const dwf = zf * zf * occ * @exp(-s.b_nm2 * s_sq);
                const at = (@as(usize, idx) * n_sites + si) * 2;
                self.sgh[at] = @floatCast(dwf * re);
                self.sgh[at + 1] = @floatCast(dwf * im);
            }
        }
        return self;
    }
};

fn metricDot(h: [3]i32, m: [9]f64) f64 {
    const v: [3]f64 = .{ @floatFromInt(h[0]), @floatFromInt(h[1]), @floatFromInt(h[2]) };
    var s: f64 = 0;
    for (0..3) |i| for (0..3) |j| {
        s += v[i] * m[i * 3 + j] * v[j];
    };
    return s;
}

/// The largest index along each axis whose spacing is still above d_min.
fn limits(cell: *const crystal.Cell, dmin: f64) [3]i32 {
    var out: [3]i32 = undefined;
    for (0..3) |axis| {
        out[axis] = 99;
        var n: i32 = 1;
        while (n < 100) : (n += 1) {
            var h: [3]i32 = .{ 0, 0, 0 };
            h[axis] = n;
            const len = @sqrt(@max(metricDot(h, cell.reciprocal), 0));
            if (len > 0 and 1 / len < dmin) {
                out[axis] = n - 1;
                break;
            }
        }
    }
    return out;
}

/// What depends on the voltage.
pub const Voltage = struct {
    gpa: std.mem.Allocator,
    voltage_kv: f64,
    /// the relativistic wavelength, nm
    mlambda: f64,
    /// the mean absorptive potential (the imaginary part of U at 000)
    diag_imag: f64,
    /// U per table row (complex, f32); zero at 000
    table: []f32,
    /// per reflection: whether its own coefficient vanishes (it is reached only by double
    /// diffraction), and its coupling scale (λ × the largest coefficient that couples it to
    /// another reflection)
    dbdiff: []u32,
    coupling: []f32,

    pub fn deinit(self: *Voltage) void {
        self.gpa.free(self.table);
        self.gpa.free(self.dbdiff);
        self.gpa.free(self.coupling);
    }

    pub fn init(gpa: std.mem.Allocator, geo: *const Geometry, voltage_kv: f64) !Voltage {
        if (!(voltage_kv > 0)) return error.BadSetting;
        const nd = geo.diff_index.len;
        const ns = geo.n_sites;
        // the scattering factors once per distinct |Δ|² and species
        const f_re = try gpa.alloc(f64, nd);
        defer gpa.free(f_re);
        const f_im = try gpa.alloc(f64, nd);
        defer gpa.free(f_im);
        const gp_re = try gpa.alloc(f64, nd);
        defer gpa.free(gp_re);
        const gp_im = try gpa.alloc(f64, nd);
        defer gpa.free(gp_im);
        @memset(f_re, 0);
        @memset(f_im, 0);
        @memset(gp_re, 0);
        @memset(gp_im, 0);
        const Key = struct { g2: u64, z: u32, sigma: u64 };
        var cache: std.AutoHashMap(Key, [2]f64) = .init(gpa);
        defer cache.deinit();
        for (0..nd) |r| {
            for (0..ns) |s| {
                const key: Key = .{ .g2 = @bitCast(geo.diff_g2[r]), .z = geo.site_z[s], .sigma = @bitCast(geo.site_sigma[s]) };
                const sf = blk: {
                    if (cache.get(key)) |v| break :blk v;
                    const v = scatter.factor(0.1 * 2 * PI * @sqrt(geo.diff_g2[r]), geo.site_z[s], geo.site_sigma[s], voltage_kv);
                    try cache.put(key, v);
                    break :blk v;
                };
                const ph = geo.diff_phase[r * ns + s];
                const occ = geo.site_occ[s];
                f_re[r] += occ * ph[0] * sf[0];
                f_im[r] += occ * ph[1] * sf[0];
                gp_re[r] += occ * ph[0] * sf[1];
                gp_im[r] += occ * ph[1] * sf[1];
            }
        }
        const vmod0 = geo.pref * std.math.hypot(f_re[0], f_im[0]);
        const psi_hat = vmod0 + voltage_kv * (1 + 0.00097847561649 * voltage_kv) * 1000;
        const mlambda = 1.22642584554 / @sqrt(psi_hat);
        const preg = 0.664840340614319;
        var self: Voltage = .{
            .gpa = gpa,
            .voltage_kv = voltage_kv,
            .mlambda = mlambda,
            .diag_imag = preg * geo.pref * std.math.hypot(gp_re[0], gp_im[0]),
            .table = try gpa.alloc(f32, @as(usize, geo.table_size) * 2),
            .dbdiff = &.{},
            .coupling = &.{},
        };
        errdefer self.deinit();
        @memset(self.table, 0);
        const dbt = try gpa.alloc(bool, geo.table_size);
        defer gpa.free(dbt);
        @memset(dbt, false);
        for (0..nd) |r| {
            const re = geo.pre * (f_re[r] - gp_im[r]);
            const im = geo.pre * (f_im[r] + gp_re[r]);
            const i = geo.diff_index[r];
            self.table[i * 2] = @floatCast(re);
            self.table[i * 2 + 1] = @floatCast(im);
            dbt[i] = std.math.hypot(re, im) <= 1e-10 and r != 0;
        }
        const off: usize = @intCast(geo.offset);
        self.table[off * 2] = 0;
        self.table[off * 2 + 1] = 0;

        const ng = geo.hkl.len;
        self.dbdiff = try gpa.alloc(u32, ng);
        for (geo.hkl, 0..) |h, g| {
            const at: usize = @intCast(geo.hash(h) + geo.offset);
            self.dbdiff[g] = @intFromBool(g != 0 and dbt[at]);
        }
        // coupling: for each reflection, the largest |U(Δ)| over the differences Δ that lead
        // to another reflection (ebsdsim's coupling_scale_from_diff_candidates)
        const Cand = struct { hash: i32, mag: f64 };
        var cands: std.ArrayList(Cand) = .empty;
        defer cands.deinit(gpa);
        var dh: i32 = -2 * geo.hmax;
        while (dh <= 2 * geo.hmax) : (dh += 1) {
            var dk: i32 = -2 * geo.kmax;
            while (dk <= 2 * geo.kmax) : (dk += 1) {
                var dl: i32 = -2 * geo.lmax;
                while (dl <= 2 * geo.lmax) : (dl += 1) {
                    const hsh = dh * geo.stride_h + dk * geo.stride_k + dl;
                    const ti = hsh + geo.offset;
                    if (ti < 0 or ti >= geo.table_size) continue;
                    const t: usize = @intCast(ti);
                    if (dbt[t]) continue;
                    const mag = std.math.hypot(@as(f64, self.table[t * 2]), @as(f64, self.table[t * 2 + 1]));
                    if (mag > 0) try cands.append(gpa, .{ .hash = hsh, .mag = mag });
                }
            }
        }
        std.mem.sort(Cand, cands.items, {}, struct {
            fn gt(_: void, a: Cand, b: Cand) bool {
                return a.mag > b.mag;
            }
        }.gt);
        var refl: std.AutoHashMap(i32, void) = .init(gpa);
        defer refl.deinit();
        for (geo.hkl) |h| try refl.put(geo.hash(h), {});
        self.coupling = try gpa.alloc(f32, ng);
        for (geo.hkl, 0..) |h, g| {
            const hi = geo.hash(h);
            var best: f64 = 0;
            for (cands.items) |c| {
                if (c.mag <= best) break;
                if (refl.contains(hi - c.hash)) {
                    best = c.mag;
                    break;
                }
            }
            self.coupling[g] = @floatCast(@max(mlambda * best, 1e-12));
        }
        return self;
    }

    /// The absorption the Lyapunov equation is shifted by for a Monte Carlo depth decay `beta`
    /// (nm⁻¹): the decay less the mean absorption already on the matrix's diagonal.
    pub fn muEff(self: *const Voltage, beta: f64) f64 {
        return beta - 2 * PI * self.mlambda * @abs(self.diag_imag);
    }
};

test "the potential's tables are ebsdsim's (nickel, GaN at 18 kV)" {
    const gpa = std.testing.allocator;
    const Case = struct { sg: u32, cell: [6]f64, sites: []const crystal.Site, n_g: usize, size: u32, offset: i32, mlambda: f64, diag_imag: f64, sum_u: f64, sum_coupling: f64, dbdiff: u32, sum_sgh: f64 };
    const cases = [_]Case{
        .{ .sg = 225, .cell = .{ 3.47514587, 3.47514587, 3.47514587, 90, 90, 90 }, .sites = &.{.{ .z = 28, .xyz = .{ 0, 0, 0 }, .b_iso = 0.5 }}, .n_g = 339, .size = 15625, .offset = 7812, .mlambda = 0.009054187146837835, .diag_imag = 2.3247171472223673, .sum_u = 1859.679443359375, .sum_coupling = 27.286223843693733, .dbdiff = 0, .sum_sgh = 3316172.0696716327 },
        .{ .sg = 186, .cell = .{ 3.1889304, 3.1889304, 5.19235725, 90, 90, 119.99999999999999 }, .sites = &.{ .{ .z = 31, .xyz = .{ 0.33333333, 0.66666667, 0.99908371 }, .b_iso = 0.5 }, .{ .z = 7, .xyz = .{ 0.33333333, 0.66666667, 0.37591629 }, .b_iso = 0.5 } }, .n_g = 1453, .size = 18081, .offset = 9040, .mlambda = 0.00905681365413938, .diag_imag = 1.6762032845675037, .sum_u = 2607.8642578125, .sum_coupling = 77.61540647223592, .dbdiff = 54, .sum_sgh = 5975175.969242225 },
    };
    for (cases) |c| {
        var cell = try crystal.Cell.init(gpa, c.sg, c.cell, c.sites);
        defer cell.deinit();
        var geo = try Geometry.init(gpa, &cell, 0.05);
        defer geo.deinit();
        var v = try Voltage.init(gpa, &geo, 18);
        defer v.deinit();
        try std.testing.expectEqual(c.n_g, geo.hkl.len);
        try std.testing.expectEqual(c.size, geo.table_size);
        try std.testing.expectEqual(c.offset, geo.offset);
        try std.testing.expectApproxEqRel(c.mlambda, v.mlambda, 1e-12);
        try std.testing.expectApproxEqRel(c.diag_imag, v.diag_imag, 1e-9);
        var su: f64 = 0;
        for (v.table) |x| su += @abs(x);
        try std.testing.expectApproxEqRel(c.sum_u, su, 1e-6);
        var sc: f64 = 0;
        for (v.coupling) |x| sc += x;
        try std.testing.expectApproxEqRel(c.sum_coupling, sc, 1e-6);
        var nd: u32 = 0;
        for (v.dbdiff) |x| nd += x;
        try std.testing.expectEqual(c.dbdiff, nd);
        var ss: f64 = 0;
        for (geo.sgh) |x| ss += @abs(x);
        try std.testing.expectApproxEqRel(c.sum_sgh, ss, 1e-6);
    }
}
