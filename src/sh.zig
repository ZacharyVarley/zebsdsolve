//! Spherical harmonics on the host, in f64: Gauss–Legendre grids, normalized associated Legendre
//! functions, analysis and synthesis, Wigner d at π/2, and the tables the SO(3) correlation
//! kernels read. Everything here runs once per phase or per geometry; the per-pattern work is
//! on the GPU (shaders/sphere.wgsl, shaders/so3.wgsl) against these tables.
//!
//! Conventions:
//!  * Y_lm(θ, φ) = P̄_l^m(cos θ) e^{imφ}, orthonormal, Condon–Shortley phase.
//!  * Coefficients a_lm = ∫ f conj(Y_lm) dΩ are stored for m ≥ 0 only (f is real:
//!    a_{l,−m} = (−1)^m conj(a_lm)), at index `pair(L, m, l)`: m-major, l = m … L−1.
//!  * A grid has nθ Gauss–Legendre rings (θ ascending from the north pole) of nφ equally spaced
//!    points, φ_j = 2π j / nφ, stored ring by ring.
//!  * D^l_{m'm}(a, b, c) = e^{−i m' a} d^l_{m'm}(b) e^{−i m c} for the rotation Rz(a) Ry(b) Rz(c),
//!    and (Λ_S f)(x) = f(S⁻¹ x) has coefficients Σ_m D^l_{m'm}(S) f_lm.
const std = @import("std");
const rot = @import("rot.zig");

const Allocator = std.mem.Allocator;
pub const Complex = [2]f64;

/// Number of stored (m ≥ 0, l ≥ m) coefficients below bandlimit L.
pub fn pairCount(L: u32) u32 {
    return L * (L + 1) / 2;
}

/// Index of (m, l), 0 ≤ m ≤ l < L.
pub fn pair(L: u32, m: u32, l: u32) u32 {
    return m * L - (m * (m -% 1)) / 2 + (l - m);
}

/// Gauss–Legendre nodes x (descending: cos θ for θ ascending) and weights on [−1, 1].
pub fn gaussLegendre(gpa: Allocator, n: u32) !struct { x: []f64, w: []f64 } {
    const x = try gpa.alloc(f64, n);
    errdefer gpa.free(x);
    const w = try gpa.alloc(f64, n);
    const nf: f64 = @floatFromInt(n);
    for (0..(n + 1) / 2) |i| {
        var z = @cos(std.math.pi * (@as(f64, @floatFromInt(i)) + 0.75) / (nf + 0.5));
        var dp: f64 = 1;
        var it: u32 = 0;
        while (it < 100) : (it += 1) {
            var p0: f64 = 1;
            var p1: f64 = z;
            var k: u32 = 2;
            while (k <= n) : (k += 1) {
                const kf: f64 = @floatFromInt(k);
                const p2 = ((2 * kf - 1) * z * p1 - (kf - 1) * p0) / kf;
                p0 = p1;
                p1 = p2;
            }
            dp = nf * (z * p1 - p0) / (z * z - 1);
            const dz = p1 / dp;
            z -= dz;
            if (@abs(dz) < 1e-16) break;
        }
        // one more evaluation for the derivative at the converged node
        var p0: f64 = 1;
        var p1: f64 = z;
        var k: u32 = 2;
        while (k <= n) : (k += 1) {
            const kf: f64 = @floatFromInt(k);
            const p2 = ((2 * kf - 1) * z * p1 - (kf - 1) * p0) / kf;
            p0 = p1;
            p1 = p2;
        }
        dp = nf * (z * p1 - p0) / (z * z - 1);
        x[i] = z;
        x[n - 1 - i] = -z;
        w[i] = 2 / ((1 - z * z) * dp * dp);
        w[n - 1 - i] = w[i];
    }
    return .{ .x = x, .w = w };
}

/// P̄_l^m(x) for every pair below L into out[pairCount(L)].
pub fn legendreAll(L: u32, x: f64, out: []f64) void {
    const s = @sqrt(@max(0.0, 1 - x * x));
    var pmm: f64 = @sqrt(1.0 / (4.0 * std.math.pi));
    var m: u32 = 0;
    while (m < L) : (m += 1) {
        if (m > 0) {
            const mf: f64 = @floatFromInt(m);
            pmm *= -@sqrt((2 * mf + 1) / (2 * mf)) * s;
        }
        const base = pair(L, m, m);
        out[base] = pmm;
        if (m + 1 < L) {
            const mf: f64 = @floatFromInt(m);
            out[base + 1] = @sqrt(2 * mf + 3) * x * pmm;
        }
        var l: u32 = m + 2;
        while (l < L) : (l += 1) {
            const lf: f64 = @floatFromInt(l);
            const mf: f64 = @floatFromInt(m);
            const a = @sqrt((4 * lf * lf - 1) / (lf * lf - mf * mf));
            const b = @sqrt(((lf - 1) * (lf - 1) - mf * mf) / (4 * (lf - 1) * (lf - 1) - 1));
            out[base + l - m] = a * (x * out[base + l - m - 1] - b * out[base + l - m - 2]);
        }
    }
}

/// A Gauss–Legendre grid with the Legendre table of bandlimit L: plm[pair · nθ + ring].
pub const Grid = struct {
    L: u32,
    n_theta: u32,
    n_phi: u32,
    /// cos θ per ring and the quadrature weight per ring (× 2π / nφ per point)
    x: []f64,
    w: []f64,
    plm: []f64,

    pub fn init(gpa: Allocator, L: u32, n_theta: u32, n_phi: u32) !Grid {
        const gl = try gaussLegendre(gpa, n_theta);
        errdefer gpa.free(gl.x);
        errdefer gpa.free(gl.w);
        const np = pairCount(L);
        const plm = try gpa.alloc(f64, @as(usize, np) * n_theta);
        errdefer gpa.free(plm);
        const tmp = try gpa.alloc(f64, np);
        defer gpa.free(tmp);
        for (0..n_theta) |y| {
            legendreAll(L, gl.x[y], tmp);
            for (0..np) |p| plm[p * n_theta + y] = tmp[p];
        }
        return .{ .L = L, .n_theta = n_theta, .n_phi = n_phi, .x = gl.x, .w = gl.w, .plm = plm };
    }

    pub fn deinit(self: *Grid, gpa: Allocator) void {
        gpa.free(self.x);
        gpa.free(self.w);
        gpa.free(self.plm);
    }

    pub fn points(self: *const Grid) usize {
        return @as(usize, self.n_theta) * self.n_phi;
    }

    /// Unit vector of ring y, point j.
    pub fn direction(self: *const Grid, y: usize, j: usize) rot.Vec3 {
        const ct = self.x[y];
        const st = @sqrt(@max(0.0, 1 - ct * ct));
        const phi = 2.0 * std.math.pi * @as(f64, @floatFromInt(j)) / @as(f64, @floatFromInt(self.n_phi));
        return .{ st * @cos(phi), st * @sin(phi), ct };
    }

    /// Analysis: samples[nθ · nφ] → coefficients[pairCount(L)].
    pub fn analyze(self: *const Grid, gpa: Allocator, samples: []const f64, out: []Complex) !void {
        const L = self.L;
        const nt = self.n_theta;
        const np = self.n_phi;
        // ring spectra F[y][m] = (2π/nφ) Σ_j f e^{−imφ_j}
        const cs = try gpa.alloc(Complex, np);
        defer gpa.free(cs);
        for (0..np) |t| {
            const ang = 2.0 * std.math.pi * @as(f64, @floatFromInt(t)) / @as(f64, @floatFromInt(np));
            cs[t] = .{ @cos(ang), @sin(ang) };
        }
        const F = try gpa.alloc(Complex, @as(usize, nt) * L);
        defer gpa.free(F);
        const sc = 2.0 * std.math.pi / @as(f64, @floatFromInt(np));
        for (0..nt) |y| {
            const row = samples[y * np ..][0..np];
            for (0..L) |m| {
                var re: f64 = 0;
                var im: f64 = 0;
                for (0..np) |j| {
                    const t = (m * j) % np;
                    re += row[j] * cs[t][0];
                    im -= row[j] * cs[t][1];
                }
                F[y * L + m] = .{ re * sc, im * sc };
            }
        }
        for (0..L) |m| {
            for (m..L) |l| {
                const p = pair(L, @intCast(m), @intCast(l));
                var re: f64 = 0;
                var im: f64 = 0;
                for (0..nt) |y| {
                    const k = self.w[y] * self.plm[@as(usize, p) * nt + y];
                    re += k * F[y * L + m][0];
                    im += k * F[y * L + m][1];
                }
                out[p] = .{ re, im };
            }
        }
    }

    /// Synthesis: coefficients[pairCount(L)] → samples[nθ · nφ].
    pub fn synthesize(self: *const Grid, gpa: Allocator, coef: []const Complex, out: []f64) !void {
        const L = self.L;
        const nt = self.n_theta;
        const np = self.n_phi;
        const cs = try gpa.alloc(Complex, np);
        defer gpa.free(cs);
        for (0..np) |t| {
            const ang = 2.0 * std.math.pi * @as(f64, @floatFromInt(t)) / @as(f64, @floatFromInt(np));
            cs[t] = .{ @cos(ang), @sin(ang) };
        }
        const G = try gpa.alloc(Complex, L);
        defer gpa.free(G);
        for (0..nt) |y| {
            for (0..L) |m| {
                var re: f64 = 0;
                var im: f64 = 0;
                for (m..L) |l| {
                    const p = pair(L, @intCast(m), @intCast(l));
                    const k = self.plm[@as(usize, p) * nt + y];
                    re += k * coef[p][0];
                    im += k * coef[p][1];
                }
                G[m] = .{ re, im };
            }
            for (0..np) |j| {
                var v = G[0][0];
                for (1..L) |m| {
                    const t = (m * j) % np;
                    v += 2 * (G[m][0] * cs[t][0] - G[m][1] * cs[t][1]);
                }
                out[y * np + j] = v;
            }
        }
    }
};

/// Σ |a_lm|² over all signed orders (the function's squared L² norm).
pub fn power(L: u32, coef: []const Complex) f64 {
    var e: f64 = 0;
    for (0..L) |m| for (m..L) |l| {
        const c = coef[pair(L, @intCast(m), @intCast(l))];
        const k: f64 = if (m == 0) 1 else 2;
        e += k * (c[0] * c[0] + c[1] * c[1]);
    };
    return e;
}

// ── Wigner d at π/2 ──

/// d^l_{k,m}(π/2) for 0 ≤ k, m, l < L at [(k · L + m) · L + l] (zero where l < max(k, m)).
pub fn wignerHalfPi(gpa: Allocator, L: u32) ![]f64 {
    const n: usize = L;
    const d = try gpa.alloc(f64, n * n * n);
    @memset(d, 0);
    // log factorials
    const lf = try gpa.alloc(f64, 2 * n + 2);
    defer gpa.free(lf);
    lf[0] = 0;
    for (1..lf.len) |i| lf[i] = lf[i - 1] + @log(@as(f64, @floatFromInt(i)));
    // k ≥ m: d^j_{m,k} = 2^{−k} sqrt((j+k)!(j−k)! / ((j+m)!(j−m)!)) P_{j−k}^{(k−m, k+m)}(0)
    for (0..n) |k| for (0..k + 1) |m| {
        const a: f64 = @floatFromInt(k - m);
        const b: f64 = @floatFromInt(k + m);
        var p2: f64 = 0;
        var p1: f64 = 0;
        var nn: usize = 0;
        while (nn + k < n) : (nn += 1) {
            const nf: f64 = @floatFromInt(nn);
            var p: f64 = undefined;
            if (nn == 0) {
                p = 1;
            } else if (nn == 1) {
                p = (a - b) / 2;
            } else {
                const c = 2 * nf + a + b;
                const num1 = (c - 1) * (a * a - b * b);
                const num2 = 2 * (nf + a - 1) * (nf + b - 1) * c;
                const den = 2 * nf * (nf + a + b) * (c - 2);
                p = (num1 * p1 - num2 * p2) / den;
            }
            const j = nn + k;
            const pref = @exp(0.5 * (lf[nn + 2 * k] + lf[nn] - lf[j + m] - lf[j - m])) * std.math.pow(f64, 2.0, -@as(f64, @floatFromInt(k)));
            // the Jacobi form gives d^j_{m,k}; d^j_{k,m} = (−1)^{k−m} d^j_{m,k}
            const v = pref * p;
            d[(m * n + k) * n + j] = v;
            d[(k * n + m) * n + j] = if ((k - m) % 2 == 0) v else -v;
            p2 = p1;
            p1 = p;
        }
    };
    return d;
}

/// Sign of d^l_{k,m}(π/2) relative to d^l_{|k|,|m|}(π/2).
pub fn dSign(l: i64, k: i64, m: i64) f64 {
    const odd = struct {
        fn f(v: i64) bool {
            return @mod(v, 2) != 0;
        }
    }.f;
    if (k < 0 and m < 0) return if (odd(k - m)) -1 else 1;
    if (m < 0) return if (odd(l + k)) -1 else 1;
    if (k < 0) return if (odd(l + m)) -1 else 1;
    return 1;
}

/// Lookup of d^l_{k,m}(π/2) for signed k, m in a `wignerHalfPi` table.
pub fn dHalfPi(tab: []const f64, L: u32, l: u32, k: i64, m: i64) f64 {
    const n: usize = L;
    const ak: usize = @intCast(@abs(k));
    const am: usize = @intCast(@abs(m));
    if (ak > l or am > l) return 0;
    return dSign(l, k, m) * tab[(ak * n + am) * n + l];
}

/// d^l_{m',m}(β) by Wigner's sum; for tests and small l only.
pub fn wignerSmallD(l: i64, mp: i64, m: i64, beta: f64) f64 {
    const fact = struct {
        fn f(n: i64) f64 {
            var r: f64 = 1;
            var i: i64 = 2;
            while (i <= n) : (i += 1) r *= @floatFromInt(i);
            return r;
        }
    }.f;
    const c = @cos(0.5 * beta);
    const s = @sin(0.5 * beta);
    const pre = @sqrt(fact(l + mp) * fact(l - mp) * fact(l + m) * fact(l - m));
    var sum: f64 = 0;
    var t: i64 = @max(0, m - mp);
    while (t <= @min(l + m, l - mp)) : (t += 1) {
        const den = fact(l + m - t) * fact(t) * fact(mp - m + t) * fact(l - mp - t);
        const sgn: f64 = if (@mod(mp - m + t, 2) == 0) 1 else -1;
        sum += sgn * std.math.pow(f64, c, @floatFromInt(2 * l + m - mp - 2 * t)) * std.math.pow(f64, s, @floatFromInt(mp - m + 2 * t)) / den;
    }
    return pre * sum;
}

/// Coefficient of a signed order from m ≥ 0 storage.
pub fn signedCoef(L: u32, coef: []const Complex, l: u32, m: i64) Complex {
    const am: u32 = @intCast(@abs(m));
    if (am > l) return .{ 0, 0 };
    const c = coef[pair(L, am, l)];
    if (m >= 0) return c;
    return if (am % 2 == 0) .{ c[0], -c[1] } else .{ -c[0], c[1] };
}

fn cmul(a: Complex, b: Complex) Complex {
    return .{ a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0] };
}

/// i^n
fn ipow(n: i64) Complex {
    return switch (@mod(n, 4)) {
        0 => .{ 1, 0 },
        1 => .{ 0, 1 },
        2 => .{ -1, 0 },
        else => .{ 0, -1 },
    };
}

/// The correlation C(a, b, c) = ∫ g(x) f(S⁻¹ x) dΩ at S = Rz(a) Ry(b) Rz(c), straight from the
/// coefficients (the sum the GPU evaluates by FFT). Reference for tests.
pub fn correlationAt(L: u32, tab: []const f64, f: []const Complex, g: []const Complex, a: f64, b: f64, c: f64) f64 {
    var total: f64 = 0;
    const Ls: i64 = L;
    var k: i64 = -(Ls - 1);
    while (k < Ls) : (k += 1) {
        var mp: i64 = -(Ls - 1);
        while (mp < Ls) : (mp += 1) {
            var m: i64 = -(Ls - 1);
            while (m < Ls) : (m += 1) {
                var t: Complex = .{ 0, 0 };
                const lmin: u32 = @intCast(@max(@abs(k), @max(@abs(mp), @abs(m))));
                var l: u32 = lmin;
                while (l < L) : (l += 1) {
                    const fl = signedCoef(L, f, l, m);
                    const gl = signedCoef(L, g, l, mp);
                    const dd = dHalfPi(tab, L, l, k, mp) * dHalfPi(tab, L, l, k, m);
                    const fg = cmul(fl, .{ gl[0], -gl[1] });
                    t[0] += fg[0] * dd;
                    t[1] += fg[1] * dd;
                }
                t = cmul(t, ipow(mp - m));
                const ang = -(@as(f64, @floatFromInt(k)) * b + @as(f64, @floatFromInt(mp)) * a + @as(f64, @floatFromInt(m)) * c);
                total += t[0] * @cos(ang) - t[1] * @sin(ang);
            }
        }
    }
    return total;
}

/// Which coefficients a crystal's symmetry leaves non-zero: orders m that are multiples of
/// `n_fold` (a rotation axis along z), l + m even (a mirror normal to z), l even (inversion).
pub const Rules = struct {
    n_fold: u32 = 1,
    mirror: bool = false,
    inversion: bool = false,

    pub fn allows(self: Rules, l: u32, m: u32) bool {
        if (m % self.n_fold != 0) return false;
        if (self.mirror and (l + m) % 2 != 0) return false;
        if (self.inversion and l % 2 != 0) return false;
        return true;
    }

    /// Read the rules off a function's own coefficients: a rule holds when the energy it
    /// forbids is below `tol` of the total.
    pub fn detect(L: u32, coef: []const Complex, tol: f64) Rules {
        var total: f64 = 0;
        var odd_l: f64 = 0;
        var odd_lm: f64 = 0;
        var off: [7]f64 = @splat(0);
        for (0..L) |m| for (m..L) |l| {
            if (l == 0) continue;
            const c = coef[pair(L, @intCast(m), @intCast(l))];
            const e = (if (m == 0) @as(f64, 1) else 2) * (c[0] * c[0] + c[1] * c[1]);
            total += e;
            if (l % 2 != 0) odd_l += e;
            if ((l + m) % 2 != 0) odd_lm += e;
            for ([_]usize{ 2, 3, 4, 6 }) |n| {
                if (m % n != 0) off[n] += e;
            }
        };
        if (total <= 0) return .{};
        var r: Rules = .{ .inversion = odd_l < tol * total, .mirror = odd_lm < tol * total };
        for ([_]u32{ 6, 4, 3, 2 }) |n| {
            if (off[n] < tol * total) {
                r.n_fold = n;
                break;
            }
        }
        return r;
    }
};

// ── tests ──

const testing = std.testing;

fn ylm(L: u32, tmp: []f64, l: u32, m: i64, v: rot.Vec3) Complex {
    legendreAll(L, v[2], tmp);
    const am: u32 = @intCast(@abs(m));
    const phi = std.math.atan2(v[1], v[0]);
    const p = tmp[pair(L, am, l)];
    const mf: f64 = @floatFromInt(m);
    if (m >= 0) return .{ p * @cos(mf * phi), p * @sin(mf * phi) };
    // Y_{l,−m} = (−1)^m conj(Y_lm)
    const s: f64 = if (am % 2 == 0) 1 else -1;
    return .{ s * p * @cos(mf * phi), s * p * @sin(mf * phi) };
}

test "gauss-legendre integrates polynomials" {
    const gl = try gaussLegendre(testing.allocator, 12);
    defer testing.allocator.free(gl.x);
    defer testing.allocator.free(gl.w);
    var s0: f64 = 0;
    var s22: f64 = 0;
    for (gl.x, gl.w) |x, w| {
        s0 += w;
        s22 += w * std.math.pow(f64, x, 22);
    }
    try testing.expectApproxEqAbs(@as(f64, 2), s0, 1e-13);
    try testing.expectApproxEqAbs(@as(f64, 2.0 / 23.0), s22, 1e-13);
    try testing.expect(gl.x[0] > gl.x[11]);
}

test "analysis inverts synthesis" {
    const gpa = testing.allocator;
    const L = 12;
    var grid = try Grid.init(gpa, L, 16, 32);
    defer grid.deinit(gpa);
    var prng = std.Random.DefaultPrng.init(7);
    const r = prng.random();
    var coef: [L * (L + 1) / 2]Complex = undefined;
    for (0..L) |m| for (m..L) |l| {
        coef[pair(L, @intCast(m), @intCast(l))] = .{ r.float(f64) - 0.5, if (m == 0) 0 else r.float(f64) - 0.5 };
    };
    const samples = try gpa.alloc(f64, grid.points());
    defer gpa.free(samples);
    try grid.synthesize(gpa, &coef, samples);
    var back: [L * (L + 1) / 2]Complex = undefined;
    try grid.analyze(gpa, samples, &back);
    for (coef, back) |a, b| {
        try testing.expectApproxEqAbs(a[0], b[0], 1e-12);
        try testing.expectApproxEqAbs(a[1], b[1], 1e-12);
    }
    // Parseval: Σ w f² = Σ |a|²
    var e: f64 = 0;
    for (0..grid.n_theta) |y| for (0..grid.n_phi) |j| {
        e += grid.w[y] * (2.0 * std.math.pi / 32.0) * samples[y * 32 + j] * samples[y * 32 + j];
    };
    try testing.expectApproxEqAbs(power(L, &coef), e, 1e-11);
}

test "wigner d at pi/2 matches the general formula and factorizes d(beta)" {
    const gpa = testing.allocator;
    const L = 9;
    const tab = try wignerHalfPi(gpa, L);
    defer gpa.free(tab);
    var l: i64 = 0;
    while (l < L) : (l += 1) {
        var k: i64 = -l;
        while (k <= l) : (k += 1) {
            var m: i64 = -l;
            while (m <= l) : (m += 1) {
                try testing.expectApproxEqAbs(wignerSmallD(l, k, m, std.math.pi / 2.0), dHalfPi(tab, L, @intCast(l), k, m), 1e-11);
            }
        }
    }
    // d^l_{m'm}(β) = i^{m'−m} Σ_k d^l_{k m'}(π/2) d^l_{k m}(π/2) e^{−ikβ}
    const beta = 0.83;
    l = 5;
    var mp: i64 = -l;
    while (mp <= l) : (mp += 1) {
        var m: i64 = -l;
        while (m <= l) : (m += 1) {
            var acc: Complex = .{ 0, 0 };
            var k: i64 = -l;
            while (k <= l) : (k += 1) {
                const dd = dHalfPi(tab, L, 5, k, mp) * dHalfPi(tab, L, 5, k, m);
                const kf: f64 = @floatFromInt(k);
                acc[0] += dd * @cos(kf * beta);
                acc[1] -= dd * @sin(kf * beta);
            }
            acc = cmul(acc, ipow(mp - m));
            try testing.expectApproxEqAbs(wignerSmallD(l, mp, m, beta), acc[0], 1e-11);
            try testing.expectApproxEqAbs(@as(f64, 0), acc[1], 1e-11);
        }
    }
}

test "harmonics rotate by D" {
    // Y_lm(S⁻¹ x) = Σ_m' D^l_{m'm}(S) Y_lm'(x), D = e^{−im'a} d(b) e^{−imc}, S = Rz(a) Ry(b) Rz(c)
    const L = 6;
    var tmp: [L * (L + 1) / 2]f64 = undefined;
    const a = 0.7;
    const b = 1.1;
    const c = -0.4;
    const S = rot.fromZyz(a, b, c);
    const x = rot.normalize(.{ 0.3, -0.5, 0.6 });
    const xr = rot.qapply(rot.qconj(S), x);
    const l: u32 = 4;
    var m: i64 = -4;
    while (m <= 4) : (m += 1) {
        const lhs = ylm(L, &tmp, l, m, xr);
        var rhs: Complex = .{ 0, 0 };
        var mp: i64 = -4;
        while (mp <= 4) : (mp += 1) {
            const ang = -(@as(f64, @floatFromInt(mp)) * a + @as(f64, @floatFromInt(m)) * c);
            const dv = wignerSmallD(4, mp, m, b);
            const D: Complex = .{ dv * @cos(ang), dv * @sin(ang) };
            const t = cmul(D, ylm(L, &tmp, l, mp, x));
            rhs[0] += t[0];
            rhs[1] += t[1];
        }
        try testing.expectApproxEqAbs(lhs[0], rhs[0], 1e-12);
        try testing.expectApproxEqAbs(lhs[1], rhs[1], 1e-12);
    }
}

test "correlation formula equals the integral" {
    const gpa = testing.allocator;
    const L = 6;
    const np = pairCount(L);
    const tab = try wignerHalfPi(gpa, L);
    defer gpa.free(tab);
    var prng = std.Random.DefaultPrng.init(3);
    const r = prng.random();
    const f = try gpa.alloc(Complex, np);
    defer gpa.free(f);
    const g = try gpa.alloc(Complex, np);
    defer gpa.free(g);
    for (0..L) |m| for (m..L) |l| {
        const p = pair(L, @intCast(m), @intCast(l));
        f[p] = .{ r.float(f64) - 0.5, if (m == 0) 0 else r.float(f64) - 0.5 };
        g[p] = .{ r.float(f64) - 0.5, if (m == 0) 0 else r.float(f64) - 0.5 };
    };
    // quadrature of ∫ g(x) f(R x) dΩ with R = S⁻¹ (R maps the frame of g to the frame of f)
    var grid = try Grid.init(gpa, L, 24, 48);
    defer grid.deinit(gpa);
    const gs = try gpa.alloc(f64, grid.points());
    defer gpa.free(gs);
    try grid.synthesize(gpa, g, gs);
    const a = 0.9;
    const b = 0.6;
    const c = 2.2;
    const R = rot.qconj(rot.fromZyz(a, b, c));
    var tmp: [L * (L + 1) / 2]f64 = undefined;
    var integral: f64 = 0;
    for (0..grid.n_theta) |y| for (0..grid.n_phi) |j| {
        const v = rot.qapply(R, grid.direction(y, j));
        // f at v from its coefficients
        legendreAll(L, v[2], &tmp);
        const phi = std.math.atan2(v[1], v[0]);
        var fv: f64 = 0;
        for (0..L) |m| for (m..L) |l| {
            const p = pair(L, @intCast(m), @intCast(l));
            const mf: f64 = @floatFromInt(m);
            const k: f64 = if (m == 0) 1 else 2;
            fv += k * tmp[p] * (f[p][0] * @cos(mf * phi) - f[p][1] * @sin(mf * phi));
        };
        integral += grid.w[y] * (2.0 * std.math.pi / 48.0) * gs[y * 48 + j] * fv;
    };
    try testing.expectApproxEqAbs(integral, correlationAt(L, tab, f, g, a, b, c), 1e-10);
}

test "rules are read from coefficients" {
    const L = 8;
    var coef: [L * (L + 1) / 2]Complex = @splat(.{ 0, 0 });
    for (0..L) |m| for (m..L) |l| {
        if (m % 4 == 0 and l % 2 == 0) coef[pair(L, @intCast(m), @intCast(l))] = .{ 1, 0.5 };
    };
    const r = Rules.detect(L, &coef, 1e-6);
    try testing.expectEqual(@as(u32, 4), r.n_fold);
    try testing.expect(r.mirror and r.inversion);
    try testing.expect(r.allows(4, 4) and !r.allows(5, 4) and !r.allows(4, 2));
}
