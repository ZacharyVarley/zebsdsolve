//! Atomic scattering factors for electrons: Weickenmeier and Kohl (Acta Cryst. A47 (1991) 590),
//! the elastic part with its Debye–Waller factor and the absorptive part from core losses and
//! thermal diffuse scattering; g in nm⁻¹ (2π |g| as WK write it), the thermal displacement in
//! nm.
const std = @import("std");
const tables = @import("tables.zig");

const PI = std.math.pi;
const EULER = 0.5772156649015329;

fn poly(c: []const f64, x: f64) f64 {
    var r = c[c.len - 1];
    var i = c.len - 1;
    while (i > 0) {
        i -= 1;
        r = r * x + c[i];
    }
    return r;
}

/// The exponential integral E1(x), x > 0.
fn exp1(x: f64) f64 {
    if (x == 0) return std.math.inf(f64);
    if (x < 0) return std.math.nan(f64);
    if (x <= 1) {
        const y = 0.6637353897094727;
        const p = [_]f64{ 0.0865197248079398, 0.03209136653035592, -0.2450882166397615, -0.03680317362579437, -0.003991671060811133, -0.00011150779292119786 };
        const q = [_]f64{ 1, 0.37091387659397013, 0.05677067710420753, 0.004273476000171037, 0.00013104990079843468, -0.5286110295202171e-6 };
        return poly(&p, x) / poly(&q, x) + x - @log(x) - y;
    }
    const p = [_]f64{ -0.12101319065772557e-18, -0.9999999999999988, -43.305866081181795, -724.5814827914625, -6046.825011271104, -27182.625446673397, -66598.26523454186, -86273.15677116495, -54844.45872264021, -14751.489578612845, -1185.4572031520103 };
    const q = [_]f64{ 1, 45.30586608118015, 809.1932149545503, 7417.3762445468955, 38129.55944848185, 113057.05869159631, 192104.04779022798, 180329.49838050182, 86722.34034673347, 18455.412473772205, 1229.2078418240305, -0.776491285282331 };
    const r = 1.0 / x;
    return (1 + poly(&p, r) / poly(&q, r)) * @exp(-x) * r;
}

/// The exponential integral Ei(x).
fn expi(x: f64) f64 {
    if (x == 0) return -std.math.inf(f64);
    if (x < 0) return -exp1(-x);
    if (x <= 6) {
        const p = [_]f64{ 2.986772243435986, 0.3563436187693774, 0.7808360762837308, 0.114670926327032, 0.04994347735765153, 0.007262245933412282, 0.001154782372278043, 0.0001164195236097652, 0.7982963656792697e-5, 0.27770562544020087e-6 };
        const q = [_]f64{ 1, -1.1709041236541391, 0.6221510984601675, -0.1951147820694954, 0.039152343139296724, -0.005048001586637057, 0.0003890340074360654, -0.1389725896017817e-4 };
        const root = 0.37250741078136663;
        const t1 = x / 3 - 1;
        const t = x - root;
        const correction = if (@abs(t) < 0.1) std.math.log1p(t / root) else @log(x / root);
        return (poly(&p, t1) / poly(&q, t1)) * t + correction;
    }
    if (x <= 10) {
        const y = 1.1589851379394531;
        const p = [_]f64{ 0.001393240861994028, -0.034992122182388874, -0.026409552075413485, -0.007612240030054764, -0.0024749620959214363, -0.00037488591794210026, -0.5540862720248818e-4, -0.3964876489248045e-5 };
        const q = [_]f64{ 1, 0.7446255668232721, 0.32906109501176706, 0.10012862497731387, 0.022385109912850635, 0.0036533419074231665, 0.00040245340851247684, 0.2636496307202557e-4 };
        const t = x / 2 - 4;
        return (y + poly(&p, t) / poly(&q, t)) * @exp(x) / x + x;
    }
    if (x <= 20) {
        const y = 1.0869731903076172;
        const p = [_]f64{ -0.008938910943569457, -0.048460773012713405, -0.06528104442222369, -0.04784475726473097, -0.02260592189237771, -0.007206036369174821, -0.0015594194703597203, -0.0002097500226602009, -0.1386522003491826e-4 };
        const q = [_]f64{ 1, 1.970172140390612, 1.8623246504307316, 1.0960143709033752, 0.43887328577308887, 0.1225377319796861, 0.02334584782757693, 0.0027817076916330367, 0.00015915028116610876 };
        const t = x / 5 - 3;
        return (y + poly(&p, t) / poly(&q, t)) * @exp(x) / x + x;
    }
    const y = 1.0130653381347656;
    const p = [_]f64{ -0.013065338134765624, 0.19029710559486577, 94.73650945371972, -2516.3532367984426, 18932.0850014926, -38703.14313620567 };
    const q = [_]f64{ 1, 61.97335928494399, -2354.562113234202, 22329.145948989308, -70126.24514039657, 54738.28331477755, 8297.162963565184 };
    const t = 1.0 / x;
    return (y + poly(&p, t) / poly(&q, t)) * @exp(x) / x + x;
}

const rih2_table = [_]f64{ 1, 1.005051, 1.010206, 1.015472, 1.020852, 1.026355, 1.031985, 1.037751, 1.043662, 1.049726, 1.055956, 1.062364, 1.068965, 1.07578, 1.08283, 1.09014, 1.097737, 1.105647, 1.113894, 1.122497, 1.13147 };

fn rih2(x: f64) f64 {
    const raw = @floor(200 / x);
    const idx: usize = @intFromFloat(std.math.clamp(raw, 0, @as(f64, @floatFromInt(rih2_table.len - 2))));
    return rih2_table[idx] + 200 * (rih2_table[idx + 1] - rih2_table[idx]) * (1 / x - 0.5e-3 * @as(f64, @floatFromInt(idx)));
}

fn rih1(x1: f64, x2: f64, x3: f64) f64 {
    if (x2 <= 20 and x3 <= 20) return @exp(-x1) * (expi(x2) - expi(x3));
    if (x2 > 20 and x3 <= 20) return @exp(x2 - x1) * rih2(x2) / x2 - @exp(-x1) * expi(x3);
    if (x2 <= 20 and x3 > 20) return @exp(-x1) * expi(x2) - @exp(x3 - x1) * rih2(x3) / x3;
    return @exp(x2 - x1) * rih2(x2) / x2 - @exp(x3 - x1) * rih2(x3) / x3;
}

const wk_i = [_]usize{ 0, 0, 0, 0, 1, 1, 1, 2, 2, 3 };
const wk_j = [_]usize{ 0, 1, 2, 3, 1, 2, 3, 2, 3, 3 };

/// The scattering factor of element `z` at |g| = `g` (WK's g: 2π over the spacing, nm⁻¹), its
/// thermal displacement `sigma` (nm) and the beam at `voltage_kv`: the elastic part with the
/// Debye–Waller factor and the relativistic correction, and the absorptive part (core losses
/// and phonons, never negative).
pub fn factor(g: f64, z: u32, sigma: f64, voltage_kv: f64) [2]f64 {
    const a = tables.wk_a[z - 1];
    const b = tables.wk_b[z - 1];
    const s = g / (4 * PI);
    const s2 = s * s;
    var f_real: f64 = 0;
    for (0..4) |i| {
        const argu = b[i] * s2;
        f_real += if (argu < 0.1)
            a[i] * b[i] * (1 - 0.5 * argu)
        else if (argu <= 20)
            (if (s2 == 0) a[i] * b[i] else a[i] * (1 - @exp(-argu)) / s2)
        else
            (if (s2 == 0) a[i] * b[i] else a[i] / s2);
    }
    const dwf = @exp(-0.5 * sigma * sigma * g * g);
    const k0 = 0.5068 * @sqrt(1022 * voltage_kv + voltage_kv * voltage_kv);

    // core losses
    const zf: f64 = @floatFromInt(z);
    const a0: f64 = 0.5289;
    const de = 6e-3 * zf;
    const theta_e = de / (2 * voltage_kv) * (2 * voltage_kv + 1022) / (voltage_kv + 1022);
    const r_val = 0.885 * a0 / std.math.cbrt(zf);
    const ta = 1 / (k0 * r_val);
    const tb = g / (2 * k0);
    const omega = 2 * tb / ta;
    const kappa = theta_e / ta;
    const k2 = kappa * kappa;
    const o2 = omega * omega;
    const sqrt_o = @sqrt(o2 + 4 * k2);
    const x1 = if (omega == 0) 0 else omega / ((1 + o2) * sqrt_o) * @log((omega + sqrt_o) / (2 * kappa));
    const qq = @sqrt((1 + o2) * (1 + o2) + 4 * k2 * o2);
    const x2 = 1 / qq * @log((1 + 2 * k2 + o2 + qq) / (2 * kappa * @sqrt(1 + k2)));
    const denom_x3 = omega * @sqrt(o2 + 4 * (1 + k2));
    const x3 = if (omega > 1e-2)
        (if (denom_x3 != 0) @log((omega + @sqrt(o2 + 4 * (1 + k2))) / (2 * @sqrt(1 + k2))) / denom_x3 else 0)
    else
        1 / (4 * (1 + k2));
    const hi = 2 * zf / (ta * ta) * (-x1 + x2 - x3);
    const f_core = 4 / (a0 * a0) * 2 * PI / (k0 * k0) * hi;

    // thermal diffuse scattering
    const m = 0.5 * sigma * sigma;
    const g2 = g * g;
    const exp_dw = @exp(-m * g2);
    var f_phon: f64 = 0;
    for (0..10) |p| {
        const ii = wk_i[p];
        const jj = wk_j[p];
        const ai = a[ii] * (4 * PI) * (4 * PI);
        const aj = a[jj] * (4 * PI) * (4 * PI);
        const bi = b[ii] / ((4 * PI) * (4 * PI));
        const bj = b[jj] / ((4 * PI) * (4 * PI));
        const mult: f64 = if (ii == jj) 1 else 2;
        var int1: f64 = undefined;
        var int2: f64 = undefined;
        if (g == 0) {
            int1 = PI * (bi * @log((bi + bj) / bi) + bj * @log((bi + bj) / bj));
            int2 = PI * ((bi + 2 * m) * @log((bi + bj + 2 * m) / (bi + 2 * m)) + bj * @log((bi + bj + 2 * m) / (bj + 2 * m)) + 2 * m * @log((2 * m) / (bj + 2 * m)));
        } else {
            const pi_by_g2 = PI / g2;
            const bi_bj = bi + bj;
            int1 = pi_by_g2 * (2 * EULER + @log(bi * g2) + @log(bj * g2) - 2 * expi(-bi * bj * g2 / bi_bj) + rih1(bi * g2, bi * bi * g2 / bi_bj, bi * g2) + rih1(bj * g2, bj * bj * g2 / bi_bj, bj * g2));
            const bi_m = bi + m;
            const bj_m = bj + m;
            const bij_m = bi + bj + 2 * m;
            int2 = pi_by_g2 * (2 * (expi(-m * bi_m * g2 / (bi + 2 * m)) + expi(-m * bj_m * g2 / (bj + 2 * m)) - expi(-bi_m * bj_m * g2 / bij_m) - expi(-0.5 * m * g2)) +
                rih1(m * g2, 0.5 * m * g2, m * m * g2 / (bi + 2 * m)) + rih1(m * g2, 0.5 * m * g2, m * m * g2 / (bj + 2 * m)) +
                rih1(bi_m * g2, bi_m * bi_m * g2 / bij_m, bi_m * bi_m * g2 / (bi + 2 * m)) + rih1(bj_m * g2, bj_m * bj_m * g2 / bij_m, bj_m * bj_m * g2 / (bj + 2 * m)));
        }
        f_phon += mult * ai * aj * (exp_dw * int1 - int2);
    }
    const gamma = (voltage_kv + 511) / 511;
    return .{ f_real * dwf * gamma, @max(0, (f_core * dwf + f_phon) * gamma * gamma / (4 * PI * k0)) };
}

test "the scattering factors are ebsdsim's" {
    // from ebsdsim's physics/scattering.py
    const cases = [_]struct { g: f64, z: u32, s: f64, v: f64, re: f64, im: f64 }{
        .{ .g = 0.0, .z = 28, .s = 0.00796, .v = 20.0, .re = 6.766123029344032, .im = 0.40423176426521557 },
        .{ .g = 3.7, .z = 28, .s = 0.00796, .v = 20.0, .re = 2.5805685330308443, .im = 0.07195005111820506 },
        .{ .g = 25.0, .z = 28, .s = 0.00796, .v = 15.0, .re = 0.14889241502800507, .im = 0.01068648757754058 },
        .{ .g = 12.0, .z = 31, .s = 0.0075, .v = 20.0, .re = 0.6247717620928248, .im = 0.017820247790479245 },
        .{ .g = 60.0, .z = 7, .s = 0.009, .v = 20.0, .re = 0.006603197885345145, .im = 0.0003420421639297825 },
        .{ .g = 8.0, .z = 79, .s = 0.006, .v = 30.0, .re = 2.399160189585665, .im = 0.07919951746738287 },
    };
    for (cases) |c| {
        const f = factor(c.g, c.z, c.s, c.v);
        try std.testing.expectApproxEqRel(c.re, f[0], 1e-12);
        try std.testing.expectApproxEqRel(c.im, f[1], 1e-10);
    }
}
