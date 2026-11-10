//! How the backscattered electrons are spread over energy and depth: a surrogate of a Monte
//! Carlo simulation of them (TinyHistNet, a small network trained on such simulations). From
//! the crystal's density, mean atomic number and weight, the sample's tilt and the beam's
//! energy, it gives, for each energy-loss bin, the depth profile of the electrons that leave
//! with that energy as an exponential, amplitude · exp(−β z). Each bin is solved at its own
//! voltage and weighted by the share of electrons it holds (`Bin.weight`); its β sets the
//! Lyapunov equation's decay. The network's weights are src/sim/surrogate.bin
//! (scripts/pack_surrogate.py).
const std = @import("std");

const blob = @embedFile("surrogate.bin");

/// the depth the profiles are normalized over: 128 bins of 1 nm (as the network was trained)
const Z_MAX = 128.0;

pub const Bin = struct {
    /// the voltage this bin is solved at (the beam's less the bin's energy loss)
    voltage_kv: f64,
    /// share of the backscattered electrons (the bins' weights add to 1)
    weight: f64,
    /// the depth profile: amplitude and decay (nm⁻¹)
    amplitude: f64,
    beta: f64,
};

const Reader = struct {
    at: usize = 4,

    fn u32_(self: *Reader) u32 {
        const v = std.mem.readInt(u32, blob[self.at..][0..4], .little);
        self.at += 4;
        return v;
    }
    fn f64_(self: *Reader) f64 {
        const v: f64 = @bitCast(std.mem.readInt(u64, blob[self.at..][0..8], .little));
        self.at += 8;
        return v;
    }
    fn f32s(self: *Reader, n: usize) []align(1) const f32 {
        const s = std.mem.bytesAsSlice(f32, blob[self.at..][0 .. n * 4]);
        self.at += n * 4;
        return s;
    }
};

/// The network's two outputs (amplitude, β) for one input row.
fn forward(gpa: std.mem.Allocator, x_in: [6]f64) ![2]f64 {
    var r: Reader = .{};
    if (!std.mem.eql(u8, blob[0..4], "ZSUR")) return error.BadSurrogate;
    _ = r.u32_();
    const n_layers = r.u32_();
    var h = try gpa.alloc(f64, 256);
    defer gpa.free(h);
    var tmp = try gpa.alloc(f64, 256);
    defer gpa.free(tmp);
    var width: usize = 6;
    // (the input's normalization is at the end of the blob: find it first)
    var skip: Reader = r;
    for (0..n_layers) |_| {
        switch (skip.u32_()) {
            0 => {
                const o = skip.u32_();
                const i = skip.u32_();
                _ = skip.f32s(@as(usize, o) * i + o);
            },
            1 => {
                const n = skip.u32_();
                _ = skip.f32s(2 * n);
            },
            else => {},
        }
    }
    for (0..2) |_| {
        const o = skip.u32_();
        const i = skip.u32_();
        _ = skip.f32s(@as(usize, o) * i + o);
    }
    var mean: [6]f64 = undefined;
    var std_: [6]f64 = undefined;
    for (&mean) |*v| v.* = skip.f64_();
    for (&std_) |*v| v.* = skip.f64_();
    _ = skip.f64_();
    const logs = skip.u32_() == 1;
    for (0..6) |i| h[i] = (x_in[i] - mean[i]) / std_[i];
    for (0..n_layers) |_| {
        switch (r.u32_()) {
            0 => {
                const o = r.u32_();
                const i = r.u32_();
                if (i != width or o > 256) return error.BadSurrogate;
                const w = r.f32s(@as(usize, o) * i);
                const b = r.f32s(o);
                for (0..o) |j| {
                    var acc: f64 = b[j];
                    for (0..i) |k| acc += @as(f64, w[j * i + k]) * h[k];
                    tmp[j] = acc;
                }
                std.mem.swap([]f64, &h, &tmp);
                width = o;
            },
            1 => {
                const n = r.u32_();
                const sc = r.f32s(n);
                const sh = r.f32s(n);
                for (0..n) |j| h[j] = h[j] * sc[j] + sh[j];
            },
            else => for (0..width) |j| {
                h[j] = h[j] / (1 + @exp(-h[j]));
            },
        }
    }
    var out: [2]f64 = undefined;
    for (0..2) |k| {
        const o = r.u32_();
        const i = r.u32_();
        if (o != 1 or i != width) return error.BadSurrogate;
        const w = r.f32s(i);
        const b = r.f32s(1);
        var acc: f64 = b[0];
        for (0..i) |j| acc += @as(f64, w[j]) * h[j];
        out[k] = if (logs) @exp(std.math.clamp(acc, -20, 20)) else std.math.log1p(@exp(-@abs(acc))) + @max(acc, 0);
    }
    return out;
}

fn header() struct { bin_kev: f64, reverse: bool } {
    // the scalars after the normalization (the last 8 + 4 + 4 bytes)
    const n = blob.len;
    return .{
        .bin_kev = @bitCast(std.mem.readInt(u64, blob[n - 16 ..][0..8], .little)),
        .reverse = std.mem.readInt(u32, blob[n - 4 ..][0..4], .little) == 1,
    };
}

/// The energy bins for a crystal (density g/cm³, mean atomic number and weight as
/// crystal.Cell has them), a sample tilt and a beam: the bins of `bin_kev` from no loss down to
/// the whole beam's energy, each with its weight and depth profile (ebsdsim's
/// infer_direct_exp_from_cell_rebinned and surrogate_to_multi_voltage_mc). Returns the bins
/// that take part (weight and amplitude above 1e-12), most energetic first.
pub fn bins(gpa: std.mem.Allocator, density: f64, average_z: f64, average_a: f64, tilt_deg: f64, beam_kv: f64, bin_kev: f64) ![]Bin {
    if (!(beam_kv > 0) or !(bin_kev > 0)) return error.BadSetting;
    const hd = header();
    const native = hd.bin_kev;
    const n_out: usize = @max(1, @as(usize, @intFromFloat(beam_kv / bin_kev)));
    // the network on its own bins
    const n_native: usize = if (@abs(bin_kev - native) < 1e-9 * native) n_out else @max(1, @as(usize, @intFromFloat(@floor(beam_kv / native))));
    const step = if (n_native == n_out) bin_kev else native;
    const amp = try gpa.alloc(f64, n_native);
    defer gpa.free(amp);
    const beta = try gpa.alloc(f64, n_native);
    defer gpa.free(beta);
    for (0..n_native) |i| {
        const e = @as(f64, @floatFromInt(i)) * step + step / 2;
        const o = try forward(gpa, .{ density, average_z, average_a, tilt_deg, beam_kv, e });
        amp[i] = o[0];
        beta[i] = o[1];
    }
    if (hd.reverse) {
        std.mem.reverse(f64, amp);
        std.mem.reverse(f64, beta);
    }
    const mass = try gpa.alloc(f64, n_native);
    defer gpa.free(mass);
    for (0..n_native) |i| {
        const nf = if (std.math.isFinite(beta[i]) and beta[i] > 0) (1 - @exp(-beta[i] * Z_MAX)) / beta[i] else 0;
        mass[i] = amp[i] * nf;
    }

    // onto the requested bins by overlap (a bin's β the mass-weighted mean of what falls in it)
    const c_mass = try gpa.alloc(f64, n_out);
    defer gpa.free(c_mass);
    const c_beta = try gpa.alloc(f64, n_out);
    defer gpa.free(c_beta);
    const c_amp = try gpa.alloc(f64, n_out);
    defer gpa.free(c_amp);
    if (n_native == n_out) {
        @memcpy(c_mass, mass);
        @memcpy(c_beta, beta);
        @memcpy(c_amp, amp);
    } else {
        @memset(c_mass, 0);
        const num = try gpa.alloc(f64, n_out);
        defer gpa.free(num);
        @memset(num, 0);
        for (0..n_out) |d| {
            const lo = @as(f64, @floatFromInt(d)) * bin_kev;
            const hi = lo + bin_kev;
            for (0..n_native) |s| {
                const s_lo = @as(f64, @floatFromInt(s)) * native;
                const ov = @min(hi, s_lo + native) - @max(lo, s_lo);
                if (ov <= 0) continue;
                const m = mass[s] * ov / native;
                c_mass[d] += m;
                num[d] += m * beta[s];
            }
            c_beta[d] = if (c_mass[d] > 0) num[d] / c_mass[d] else 0;
            const nf = if (c_mass[d] > 0 and c_beta[d] > 0) (1 - @exp(-c_beta[d] * Z_MAX)) / c_beta[d] else 0;
            c_amp[d] = if (nf > 0) c_mass[d] / nf else 0;
        }
    }
    var total: f64 = 0;
    for (c_mass) |m| total += m;
    var out: std.ArrayList(Bin) = .empty;
    errdefer out.deinit(gpa);
    for (0..n_out) |i| {
        const w = if (total > 0) c_mass[i] / total else 0;
        const v = beam_kv - @as(f64, @floatFromInt(i)) * bin_kev;
        if (w >= 1e-12 and c_amp[i] >= 1e-12 and v > 0) try out.append(gpa, .{ .voltage_kv = v, .weight = w, .amplitude = c_amp[i], .beta = c_beta[i] });
    }
    return out.toOwnedSlice(gpa);
}

/// The model's account for the app, as JSON: every bin (its voltage, weight and depth
/// profile), the depth distribution of all of them together and each bin's share of it,
/// sampled at `n_depth` depths from 0 to where the whole falls under 1 % of its value at the
/// surface (at most the 128 nm the model knows), each scaled so that the whole's largest value
/// is 1; and figures of them: the mean energy loss, the mean depth, and the depth above which
/// 90 % of the electrons leave.
pub fn describe(gpa: std.mem.Allocator, density: f64, average_z: f64, average_a: f64, tilt_deg: f64, beam_kv: f64, bin_kev: f64) ![]u8 {
    const n_depth = 96;
    const b = try bins(gpa, density, average_z, average_a, tilt_deg, beam_kv, bin_kev);
    defer gpa.free(b);
    const S = struct { kv: f64, weight: f64, beta: f64, amplitude: f64 };
    const rows = try gpa.alloc(S, b.len);
    defer gpa.free(rows);
    var loss: f64 = 0;
    for (b, rows) |x, *o| {
        o.* = .{ .kv = x.voltage_kv, .weight = x.weight, .beta = x.beta, .amplitude = x.amplitude };
        loss += x.weight * (beam_kv - x.voltage_kv);
    }
    // the whole profile Σ a e^(−β z), and its depth of 1 %
    const whole = struct {
        fn at(list: []const Bin, z: f64) f64 {
            var v: f64 = 0;
            for (list) |x| v += x.amplitude * @exp(-x.beta * z);
            return v;
        }
    };
    const top = whole.at(b, 0);
    var z_end: f64 = Z_MAX;
    {
        var z: f64 = 0;
        while (z < Z_MAX) : (z += 0.25) {
            if (whole.at(b, z) < 0.01 * top) {
                z_end = z;
                break;
            }
        }
    }
    // the mean depth and the depth of 90 %, from the profile on a fine grid
    var mass: f64 = 0;
    var first: f64 = 0;
    const fine: usize = 4096;
    const dz = Z_MAX / @as(f64, @floatFromInt(fine));
    for (0..fine) |i| {
        const z = (@as(f64, @floatFromInt(i)) + 0.5) * dz;
        const v = whole.at(b, z);
        mass += v;
        first += v * z;
    }
    var z90: f64 = Z_MAX;
    {
        var acc: f64 = 0;
        for (0..fine) |i| {
            const z = (@as(f64, @floatFromInt(i)) + 0.5) * dz;
            acc += whole.at(b, z);
            if (acc >= 0.9 * mass) {
                z90 = z;
                break;
            }
        }
    }
    const depth = try gpa.alloc(f64, n_depth);
    defer gpa.free(depth);
    const total = try gpa.alloc(f64, n_depth);
    defer gpa.free(total);
    const each = try gpa.alloc([n_depth]f64, b.len);
    defer gpa.free(each);
    const k = if (top > 0) 1 / top else 0;
    for (0..n_depth) |i| {
        const z = z_end * @as(f64, @floatFromInt(i)) / (n_depth - 1);
        depth[i] = z;
        total[i] = whole.at(b, z) * k;
        for (b, each) |x, *row| row[i] = x.amplitude * @exp(-x.beta * z) * k;
    }
    var out: std.Io.Writer.Allocating = .init(gpa);
    errdefer out.deinit();
    try std.json.Stringify.value(.{
        .beam_kv = beam_kv,
        .bin_kev = bin_kev,
        .tilt_deg = tilt_deg,
        .bins = rows,
        .mean_loss_kev = loss,
        .mean_depth_nm = if (mass > 0) first / mass else 0,
        .depth90_nm = z90,
        .depth_nm = depth,
        .depth = total,
        .depth_by_bin = each,
    }, .{}, &out.writer);
    return out.toOwnedSlice();
}

test "the surrogate's bins are ebsdsim's (nickel, 70°, 20 kV)" {
    const gpa = std.testing.allocator;
    // ebsdsim's infer_direct_exp_from_cell_rebinned: the first three bins' weights, β and
    // amplitudes at its native 1 keV, and rebinned to 0.5 and 2 keV
    const cases = [_]struct { bw: f64, w: [3]f64, beta: [3]f64, amp: [3]f64 }{
        .{ .bw = 1, .w = .{ 0.2976759003197413, 0.20341057079741232, 0.12474137550629837 }, .beta = .{ 0.22134371724322074, 0.21021261259221086, 0.21760903320188368 }, .amp = .{ 0.06846628721755274, 0.044432238225440464, 0.028206769573402426 } },
        .{ .bw = 0.5, .w = .{ 0.14488990087481135, 0.1527859994449298, 0.11576205439589048 }, .beta = .{ 0.2355990175506921, 0.20782514187640738, 0.20903567087320848 }, .amp = .{ 0.03547133166083654, 0.032994955556777436, 0.025145051570348256 } },
        .{ .bw = 2, .w = .{ 0.5010864711171535, 0.21159086060465557, 0.1137395869552417 }, .beta = .{ 0.21682516709035043, 0.2225830784554002, 0.2508634371128986 }, .amp = .{ 0.11289852544296748, 0.048938984374766646, 0.029649332309586374 } },
    };
    for (cases) |c| {
        const b = try bins(gpa, 9.289219823834271, 28.0, 58.6934, 70, 20, c.bw);
        defer gpa.free(b);
        for (0..3) |i| {
            try std.testing.expectApproxEqAbs(20 - @as(f64, @floatFromInt(i)) * c.bw, b[i].voltage_kv, 1e-12);
            try std.testing.expectApproxEqRel(c.w[i], b[i].weight, 1e-6);
            try std.testing.expectApproxEqRel(c.beta[i], b[i].beta, 1e-6);
            try std.testing.expectApproxEqRel(c.amp[i], b[i].amplitude, 1e-6);
        }
    }
}

test "the model's account: nickel's depths and losses, as JSON" {
    const gpa = std.testing.allocator;
    const j = try describe(gpa, 9.289219823834271, 28.0, 58.6934, 70, 20, 1);
    defer gpa.free(j);
    const D = struct { bins: []const struct { kv: f64, weight: f64 }, mean_depth_nm: f64, depth90_nm: f64, mean_loss_kev: f64, depth: []const f64, depth_by_bin: []const []const f64 };
    const v = try std.json.parseFromSlice(D, gpa, j, .{ .ignore_unknown_fields = true });
    defer v.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 20), v.value.bins[0].kv, 1e-12);
    var w: f64 = 0;
    for (v.value.bins) |b| w += b.weight;
    try std.testing.expectApproxEqAbs(@as(f64, 1), w, 1e-9);
    // a few nanometres (β about 0.2 nm⁻¹), 90 % deeper than the mean, the loss a few keV
    try std.testing.expect(v.value.mean_depth_nm > 2 and v.value.mean_depth_nm < 10);
    try std.testing.expect(v.value.depth90_nm > v.value.mean_depth_nm);
    try std.testing.expect(v.value.mean_loss_kev > 0.5 and v.value.mean_loss_kev < 8);
    try std.testing.expectApproxEqAbs(@as(f64, 1), v.value.depth[0], 1e-9);
    try std.testing.expectEqual(v.value.bins.len, v.value.depth_by_bin.len);
}
