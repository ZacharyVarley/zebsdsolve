//! Master patterns simulated from a crystal: dynamical Bloch-wave intensities, depth-integrated
//! under the backscattered electrons' depth profiles by a low-rank Lyapunov solve, its error
//! corrected by the dual-weighted residual (settings.sim_correct). (The method of ebsdsim,
//! github.com/ZacharyVarley/ebsdsim, whose numbers the tests here check against, with that
//! correction and a block preconditioner added.)
//!
//! A crystal (sim/crystal.zig) gives the reflections and the potential's coefficients
//! (sim/potential.zig); the energy model (sim/energy.zig) spreads the backscattered electrons
//! over energy-loss bins, each with an exponential depth profile; the directions are those of
//! the point group's fundamental sector on the Lambert square (sim/kgrid.zig). For every bin,
//! from the least loss up, each direction's depth-integrated Bloch-wave intensity is solved on
//! the GPU (shaders/simulate.wgsl), weighted by the bin's share, and added up; the bins stop
//! once one changes the pattern by less than settings.sim_stop. The result is the fundamental
//! sector, per site and per bin: a master pattern (master.fromSector) and an .npz in ebsdsim's
//! layout, which this engine and ebsdsim both read.
const std = @import("std");
const engine_mod = @import("engine.zig");
const crystal = @import("sim/crystal.zig");
const potential = @import("sim/potential.zig");
const energy = @import("sim/energy.zig");
const kgrid = @import("sim/kgrid.zig");
const master_mod = @import("master.zig");
const rot = @import("rot.zig");
const NpzWriter = @import("io/npz_write.zig").Writer;
const gpu_mod = @import("gpu/gpu.zig");
const shaders = @import("shaders");

const Engine = engine_mod.Engine;
const Gpu = gpu_mod.Gpu;
const Buf = gpu_mod.Buf;
const Bind = gpu_mod.Bind;

/// simulate.wgsl's limits: beams a direction, Krylov rank, sites
/// The beams a direction can solve for (and weak beams it can follow) in `bytes` of workgroup
/// memory: sim_solve, the most demanding pass, takes 24 bytes a beam and about 2.4 KiB besides
/// (16 KiB: 512, 32 KiB: 1024, 48 KiB: 1536).
pub fn capacity(bytes: u32) u32 {
    var cap: u32 = 512;
    for ([_]u32{ 768, 1024, 1536, 2048 }) |c| {
        if (24 * @as(u64, c) + 2560 <= bytes) cap = c;
    }
    return cap;
}

/// The least of the capacities (512, 768, 1024, 1536, 2048) that holds `need`, at most `cap`.
fn fitting(need: u32, cap: u32) u32 {
    for ([_]u32{ 512, 768, 1024, 1536, 2048 }) |c| {
        if (c >= need or c >= cap) return @min(c, cap);
    }
    return cap;
}
pub const MAX_R = 16;
pub const MAX_SITES = 16;

const P = extern struct {
    n_k: u32 = 0,
    n_g: u32 = 0,
    n: u32 = 0,
    rank: u32 = 0,
    n_sites: u32 = 0,
    table_size: u32 = 0,
    offset: i32 = 0,
    grid_x: u32 = 1,
    m0: [4]f32 = @splat(0),
    m1: [4]f32 = @splat(0),
    m2: [4]f32 = @splat(0),
    bethe: [4]f32 = @splat(0),
    phys: [4]f32 = @splat(0),
    /// the correction on; a direction's scratch (vec2 units); the weak beams a direction; unused
    flags: [4]u32 = @splat(0),
};

/// A direction's scratch in simulate.wgsl (vec2 units, its o_* offsets): header, the diagonal
/// and two core maps (n each), four core matrices and the small system, Q, ÃQ, G and ÃQ·Y,
/// BiCGSTAB's solution and shadow residual (n each); the weak beams' 1 / d_w and hashes (n_w
/// each) and their rows of T G and T Q (n_w × rank each).
fn scratchLen(n: u32, nw: u32, rank: u32) u64 {
    return 16 + 3 * @as(u64, n) + 4 * CORE_SQ + 2 * MAX_R * MAX_R + 4 * @as(u64, n) * rank + 2 * @as(u64, n) + 2 * @as(u64, nw) + 2 * @as(u64, nw) * rank;
}
const CORE_SQ = 1024;

/// What a simulation did, for the front ends.
pub const Summary = extern struct {
    /// directions of the fundamental sector, the crystal's sites, reflections
    n_k: u32 = 0,
    n_sites: u32 = 0,
    n_g: u32 = 0,
    /// energy bins the model gave and those solved
    n_bins: u32 = 0,
    n_bins_run: u32 = 0,
    /// the most beams any bin took, solved and followed as weak beams
    max_beams: u32 = 0,
    /// directions whose solve failed (their intensity left at zero), over every bin
    failed: u32 = 0,
    /// 1: the bins stopped on settings.sim_stop
    stopped: u32 = 0,
    halfw: u32 = 0,
    rank: u32 = 0,
    max_weak: u32 = 0,
    /// directions (over the bins) with more strong beams than MAX_N: their strong beams past it
    /// were followed as weak beams
    overflow: u32 = 0,
    /// the last bin's relative change of the pattern
    last_change: f64 = 0,
};

pub const Simulation = struct {
    gpa: std.mem.Allocator,
    summary: Summary,
    grid: kgrid.Grid,
    /// the fundamental sector's intensities, direction-major (n_k × n_sites): summed over the
    /// bins, and each bin's own (divided by its weight, as ebsdsim keeps them)
    integrated: []f32,
    per_bin: [][]f32,
    bin_kv: []f32,
    bin_weight: []f32,
    /// the sites' shares (occupancy × multiplicity, normalized)
    site_weight: []f32,
    pg_ops: []rot.Mat3,
    normals: [][3]f64,
    space_group: u32,
    point_group: u32,
    symbol: []const u8,
    voltage_kv: f64,
    /// a, b, c (Å) and α, β, γ (degrees)
    lattice: [6]f64,
    /// the crystal and settings as JSON, for the file's metadata
    meta: []u8,

    pub fn deinit(self: *Simulation) void {
        const g = self.gpa;
        self.grid.deinit();
        g.free(self.integrated);
        for (self.per_bin) |b| g.free(b);
        g.free(self.per_bin);
        g.free(self.bin_kv);
        g.free(self.bin_weight);
        g.free(self.site_weight);
        g.free(self.pg_ops);
        g.free(self.normals);
        g.free(self.meta);
    }

    /// Per direction, the sites combined by their shares.
    fn combined(self: *const Simulation, src: []const f32, out: []f64) void {
        const ns = self.summary.n_sites;
        for (out, 0..) |*v, k| {
            var acc: f64 = 0;
            for (0..ns) |s| acc += @as(f64, self.site_weight[s]) * src[k * ns + s];
            v.* = acc;
        }
    }

    /// The master pattern it makes (named `name`).
    pub fn master(self: *const Simulation, gpa: std.mem.Allocator, name: []const u8) !master_mod.Master {
        return self.masterOf(gpa, name, null, null);
    }

    /// The master of one energy bin (or of them all, `bin` null) and of one site (or of the
    /// sites combined by their shares, `site` null).
    pub fn masterOf(self: *const Simulation, gpa: std.mem.Allocator, name: []const u8, bin: ?usize, site: ?usize) !master_mod.Master {
        const n = self.grid.khat.len;
        const ns = self.summary.n_sites;
        if (bin) |b| if (b >= self.per_bin.len) return error.NoSuchBin;
        if (site) |s| if (s >= ns) return error.NoSuchSite;
        const src = if (bin) |b| self.per_bin[b] else self.integrated;
        const vals = try gpa.alloc(f64, n);
        defer gpa.free(vals);
        if (site) |s| {
            for (vals, 0..) |*v, k| v.* = src[k * ns + s];
        } else self.combined(src, vals);
        const normals = try gpa.alloc(rot.Vec3, self.normals.len);
        defer gpa.free(normals);
        for (self.normals, 0..) |v, i| normals[i] = v;
        var m = try master_mod.fromSector(gpa, vals, self.grid.kij, self.grid.halfw, self.pg_ops, normals, .{ .point_group = self.point_group, .space_group = self.space_group, .voltage_kv = self.voltage_kv, .lattice = self.lattice });
        errdefer m.deinit();
        gpa.free(m.name);
        m.name = &.{};
        m.name = try gpa.dupe(u8, name);
        try m.measureSymmetry();
        return m;
    }

    /// The .npz file, in ebsdsim's layout (format 3): the fundamental sector as (energy, site,
    /// direction), energy 0 summed over the bins and 1 + b bin b (when there are several), site 0
    /// the sites combined and 1 + s site s (when there are several).
    pub fn npz(self: *const Simulation, gpa: std.mem.Allocator) ![]u8 {
        const n_k = self.grid.khat.len;
        const ns = self.summary.n_sites;
        const nb = self.per_bin.len;
        const e_dim: usize = if (nb > 1) 1 + nb else 1;
        const s_dim: usize = if (ns > 1) 1 + ns else 1;
        const fs = try gpa.alloc(f32, e_dim * s_dim * n_k);
        defer gpa.free(fs);
        const tmp = try gpa.alloc(f64, n_k);
        defer gpa.free(tmp);
        for (0..e_dim) |e| {
            const src = if (e == 0) self.integrated else self.per_bin[e - 1];
            const slab = fs[e * s_dim * n_k ..][0 .. s_dim * n_k];
            if (ns > 1) {
                self.combined(src, tmp);
                for (0..n_k) |k| slab[k] = @floatCast(tmp[k]);
                for (0..ns) |s| for (0..n_k) |k| {
                    slab[(1 + s) * n_k + k] = src[k * ns + s];
                };
            } else @memcpy(slab, src);
        }
        var w = NpzWriter.init(gpa);
        defer w.deinit();
        try w.addSlice("fundamental_sector", f32, &.{ e_dim, s_dim, n_k }, fs);
        try w.addSlice("fundamental_kij", i32, &.{ n_k, 3 }, std.mem.bytesAsSlice(i32, std.mem.sliceAsBytes(self.grid.kij)));
        try w.addSlice("fundamental_khat", f32, &.{ n_k, 3 }, std.mem.bytesAsSlice(f32, std.mem.sliceAsBytes(self.grid.khat)));
        try w.addSlice("pg_operators", f64, &.{ self.pg_ops.len, 3, 3 }, std.mem.bytesAsSlice(f64, std.mem.sliceAsBytes(self.pg_ops)));
        try w.addSlice("fs_normals", f64, &.{ self.normals.len, 3 }, std.mem.bytesAsSlice(f64, std.mem.sliceAsBytes(self.normals)));
        try w.addSlice("bin_voltages_kv", f32, &.{self.bin_kv.len}, self.bin_kv);
        try w.addSlice("bin_weights", f32, &.{self.bin_weight.len}, self.bin_weight);
        try w.addSlice("site_weights", f32, &.{self.site_weight.len}, self.site_weight);
        try w.addSlice("meta_json", u8, &.{self.meta.len}, self.meta);
        return w.finish();
    }
};

/// The change the latest bin made to the pattern: before and after (both normalized to their
/// own range), the norm of the difference over the norm of the after (ebsdsim's
/// relative_normalized_view_change).
fn relativeChange(sum: []const f64, delta: []const f64) f64 {
    var lo0: f64 = std.math.inf(f64);
    var hi0: f64 = -std.math.inf(f64);
    var lo1: f64 = std.math.inf(f64);
    var hi1: f64 = -std.math.inf(f64);
    for (sum, delta) |s, d| {
        lo0 = @min(lo0, s);
        hi0 = @max(hi0, s);
        lo1 = @min(lo1, s + d);
        hi1 = @max(hi1, s + d);
    }
    const sp0 = hi0 - lo0;
    const sp1 = hi1 - lo1;
    var dd: f64 = 0;
    var tt: f64 = 0;
    for (sum, delta) |s, d| {
        const a = if (sp0 > 1e-30) (s - lo0) / sp0 else 0;
        const b = if (sp1 > 1e-30) (s + d - lo1) / sp1 else 0;
        dd += (b - a) * (b - a);
        tt += b * b;
    }
    if (!(tt > 1e-60)) return std.math.inf(f64);
    return @sqrt(dd) / @sqrt(tt);
}

/// A kernel of simulate.wgsl compiled for `cap` beams a direction and `bpt` rows a thread.
fn pipe(g: *Gpu, comptime entry: []const u8, cap: u32, bpt: u32) u32 {
    var kb: [96]u8 = undefined;
    const k = std.fmt.bufPrint(&kb, "simulate/" ++ entry ++ "@{d}/{d}", .{ cap, bpt }) catch return gpu_mod.NO_PIPE;
    if (g.pipes.get(k)) |p| return p;
    const code = std.fmt.allocPrint(g.gpa, "const MAX_N: u32 = {d}u;\nconst BPT: u32 = {d}u;\n{s}", .{ cap, bpt, shaders.simulate_body }) catch return gpu_mod.NO_PIPE;
    defer g.gpa.free(code);
    return g.pipeline(k, code, entry) catch gpu_mod.NO_PIPE;
}

/// Simulate the master pattern of `cell` under the engine's settings (sim_*).
pub fn run(e: *Engine, cell: *const crystal.Cell) !Simulation {
    const gpa = e.gpa;
    const g = &e.g;
    const s = &e.set;
    e.cancel = false;
    if (cell.sites.len > MAX_SITES) return error.TooManySites;

    var geo = try potential.Geometry.init(gpa, cell, s.sim_dmin_nm);
    defer geo.deinit();
    const bins = try energy.bins(gpa, cell.density, cell.average_z, cell.average_a, s.sim_tilt_deg, s.sim_voltage_kv, s.sim_bin_kev);
    defer gpa.free(bins);
    if (bins.len == 0) return error.NoEnergyBins;
    var grid = try kgrid.Grid.init(gpa, s.sim_halfw, cell.normals, cell.centro);
    errdefer grid.deinit();
    const n_k: u32 = @intCast(grid.khat.len);
    const n_g: u32 = @intCast(geo.hkl.len);
    const ns: u32 = @intCast(cell.sites.len);
    const rank = s.sim_rank;
    // the beams a direction can take, from the device's workgroup memory
    const cap = capacity(e.g.lim.max_workgroup_storage);
    e.events.log("simulation: space group {d} (point group {s}), {d} reflections to {d} nm, {d} directions on a {d} × {d} Lambert square, {d} energy bins from {d} kV; up to {d} beams a direction (and as many weak)", .{ cell.space_group, cell.symbol, n_g, s.sim_dmin_nm, n_k, 2 * s.sim_halfw + 1, 2 * s.sim_halfw + 1, bins.len, s.sim_voltage_kv, cap });

    // what does not change from bin to bin
    var hkl_b: Buf = .{};
    var sgh_b: Buf = .{};
    var kv_b: Buf = .{};
    var tab_b: Buf = .{};
    var cpl_b: Buf = .{};
    var dbd_b: Buf = .{};
    var cnt_b: Buf = .{};
    var keys_b: Buf = .{};
    var beams_b: Buf = .{};
    var work_b: Buf = .{};
    var out_b: Buf = .{};
    defer inline for (.{ &hkl_b, &sgh_b, &kv_b, &tab_b, &cpl_b, &dbd_b, &cnt_b, &keys_b, &beams_b, &work_b, &out_b }) |b| g.release(b);
    {
        const h = try gpa.alloc([4]i32, n_g);
        defer gpa.free(h);
        for (geo.hkl, 0..) |v, i| h[i] = .{ v[0], v[1], v[2], geo.hash(v) };
        hkl_b = g.storage(@as(u64, n_g) * 16);
        g.writeSlice(hkl_b, 0, [4]i32, h);
    }
    sgh_b = g.storage(@as(u64, geo.sgh.len) * 4);
    g.writeSlice(sgh_b, 0, f32, geo.sgh);
    tab_b = g.storage(@as(u64, geo.table_size) * 8);
    cpl_b = g.storage(@as(u64, n_g) * 4);
    dbd_b = g.storage(@as(u64, n_g) * 4);
    cnt_b = g.storage(16);
    kv_b = g.storage(@as(u64, n_k) * 16);
    out_b = g.storage(@as(u64, n_k) * ns * 4);

    var p: P = .{ .n_g = n_g, .n_sites = ns, .table_size = geo.table_size, .offset = geo.offset, .rank = rank };
    const m = cell.reciprocal;
    p.m0 = .{ @floatCast(m[0]), @floatCast(m[1]), @floatCast(m[2]), 0 };
    p.m1 = .{ @floatCast(m[3]), @floatCast(m[4]), @floatCast(m[5]), 0 };
    p.m2 = .{ @floatCast(m[6]), @floatCast(m[7]), @floatCast(m[8]), 0 };
    // (the weak cutoff never under the strong one: a direction's strong beams past MAX_N are weak)
    p.bethe = .{ @floatCast(s.sim_c_strong), @floatCast(s.sim_c_cutoff), @floatCast(s.sim_dbdiff_cutoff), @floatCast(@max(s.sim_c_strong, s.sim_c_weak)) };

    const integrated = try gpa.alloc(f32, @as(usize, n_k) * ns);
    errdefer gpa.free(integrated);
    @memset(integrated, 0);
    var per_bin: std.ArrayList([]f32) = .empty;
    errdefer {
        for (per_bin.items) |b| gpa.free(b);
        per_bin.deinit(gpa);
    }
    var bin_kv: std.ArrayList(f32) = .empty;
    errdefer bin_kv.deinit(gpa);
    var bin_w: std.ArrayList(f32) = .empty;
    errdefer bin_w.deinit(gpa);
    const sum_view = try gpa.alloc(f64, n_k);
    defer gpa.free(sum_view);
    @memset(sum_view, 0);
    const delta_view = try gpa.alloc(f64, n_k);
    defer gpa.free(delta_view);
    const kv = try gpa.alloc([4]f32, n_k);
    defer gpa.free(kv);
    const bin_out = try gpa.alloc(f32, @as(usize, n_k) * ns);
    defer gpa.free(bin_out);

    var summary: Summary = .{ .n_k = n_k, .n_sites = ns, .n_g = n_g, .n_bins = @intCast(bins.len), .halfw = s.sim_halfw, .rank = rank };
    // the directions are solved a tile at a time: as many as fit a scratch of up to 256 MiB
    // (the binding limit allowing), and a multiple of 64 so that a tile's slices of the
    // per-direction buffers start where bindings may
    const room = @min(@min(e.g.lim.max_storage_binding, e.g.lim.max_buffer) / 2, 256 << 20);
    var done_tiles: u32 = 0;
    for (bins, 0..) |bin, bi| {
        if (e.cancel) return error.Cancelled;
        var volt = try potential.Voltage.init(gpa, &geo, bin.voltage_kv);
        defer volt.deinit();
        g.writeSlice(tab_b, 0, f32, volt.table);
        g.writeSlice(cpl_b, 0, f32, volt.coupling);
        g.writeSlice(dbd_b, 0, u32, volt.dbdiff);
        // k = k̂ · DSM / λ (a direction's wave vector in reciprocal-lattice components)
        for (grid.khat, 0..) |d, i| {
            var v: [4]f32 = .{ 0, 0, 0, 0 };
            for (0..3) |c| {
                var acc: f64 = 0;
                for (0..3) |j| acc += @as(f64, d[j]) * cell.dsm[j * 3 + c];
                v[c] = @floatCast(acc / volt.mlambda);
            }
            kv[i] = v;
        }
        g.writeSlice(kv_b, 0, [4]f32, kv);
        const mu = volt.muEff(bin.beta);
        p.phys = .{ @floatCast(std.math.pi * volt.mlambda), @floatCast(volt.diag_imag), @floatCast(mu), @floatCast(bin.amplitude * bin.weight) };

        // the beams this bin takes: as many as the direction with the most strong ones
        g.writeSlice(cnt_b, 0, u32, &.{ 0, 0, 0, 0 });
        p.n_k = n_k;
        {
            const groups = (n_k + 63) / 64;
            const f = Gpu.flat(groups);
            p.grid_x = f[0];
            g.dispatch(pipe(g, "strong_count", cap, 1), f[0], f[1], 1, &.{ g.uniform(0, std.mem.asBytes(&p)), hkl_b.at(1), cpl_b.at(2), dbd_b.at(3), kv_b.at(4), cnt_b.at(7) });
        }
        var cnt: [4]u32 = undefined;
        g.read(cnt_b, 0, std.mem.asBytes(&cnt));
        try g.wait();
        // the beams: as many as the direction with the most strong ones has (at most MAX_N), and
        // the weak ones after them, as many as any direction has within the weak cutoff
        const n = @max(@min(cnt[0], n_g, cap), 1);
        const nw: u32 = @min(cnt[2] -| n, cap, n_g - n);
        // the bin's kernels sized for what it needs, not the device's most (workgroup memory a
        // workgroup holds and no other can use: a small bin runs faster in a small size)
        const bcap = fitting(@max(n, nw), cap);
        if (cnt[3] > 0) {
            e.events.log("simulation: at {d} kV, {d} directions have more strong beams than the {d} a direction can solve (up to {d}): each solves its {d} nearest, and follows the rest as weak beams", .{ bin.voltage_kv, cnt[3], cap, cnt[0], cap });
            summary.overflow += cnt[3];
        }
        if (cnt[2] -| n > cap) e.events.log("simulation: at {d} kV, up to {d} weak beams a direction; the {d} nearest are followed", .{ bin.voltage_kv, cnt[2] - n, cap });
        summary.max_beams = @max(summary.max_beams, n);
        summary.max_weak = @max(summary.max_weak, nw);
        p.n = n;
        p.flags[2] = nw;
        const bpt = (n + 255) / 256;
        const stride = scratchLen(n, nw, rank);
        const per_k: u64 = stride * 8 + @as(u64, n_g) * 4;
        var tile: u32 = @intCast(@min(@max(room / per_k, 64), 16384));
        tile = @max(tile / 64 * 64, 64);
        g.ensure(&work_b, @as(u64, tile) * stride * 8);
        // the solve's rows a thread: one when the beams fit a workgroup
        const solve_pipe = pipe(g, "sim_solve", bcap, bpt);
        g.ensure(&keys_b, @as(u64, tile) * n_g * 4);
        g.ensure(&beams_b, @as(u64, tile) * (n + nw) * 4);
        const n_tiles = (n_k + tile - 1) / tile;
        var k0: u32 = 0;
        while (k0 < n_k) : (k0 += tile) {
            if (e.cancel) return error.Cancelled;
            const cnt_k = @min(tile, n_k - k0);
            var pt = p;
            pt.n_k = cnt_k;
            const f = Gpu.flat(cnt_k);
            pt.grid_x = f[0];
            pt.flags = .{ @intFromBool(s.sim_correct), @intCast(stride), nw, 0 };
            const kv_bind = kv_b.range(4, @as(u64, k0) * 16, @as(u64, cnt_k) * 16);
            g.dispatch(pipe(g, "select_beams", bcap, bpt), f[0], f[1], 1, &.{ g.uniform(0, std.mem.asBytes(&pt)), hkl_b.at(1), cpl_b.at(2), dbd_b.at(3), kv_bind, keys_b.at(5), beams_b.at(6) });
            const u = g.uniform(0, std.mem.asBytes(&pt));
            g.dispatch(pipe(g, "sim_prep", bcap, bpt), f[0], f[1], 1, &.{ u, hkl_b.at(1), kv_bind, beams_b.at(6), tab_b.at(8), work_b.at(10) });
            g.dispatch(solve_pipe, f[0], f[1], 1, &.{ u, hkl_b.at(1), beams_b.at(6), tab_b.at(8), work_b.at(10) });
            if (s.sim_correct) g.dispatch(pipe(g, "sim_core", bcap, bpt), f[0], f[1], 1, &.{ u, hkl_b.at(1), beams_b.at(6), tab_b.at(8), work_b.at(10) });
            g.dispatch(pipe(g, "sim_intensity", bcap, bpt), f[0], f[1], 1, &.{ u, hkl_b.at(1), beams_b.at(6), cnt_b.at(7), sgh_b.at(9), work_b.at(10), out_b.range(11, @as(u64, k0) * ns * 4, @as(u64, cnt_k) * ns * 4) });
            g.submit();
            done_tiles += 1;
            e.events.emit(.{ .kind = "progress", .stage = "simulate", .done = done_tiles, .total = @as(u32, @intCast(bins.len)) * n_tiles });
        }
        g.read(out_b, 0, std.mem.sliceAsBytes(bin_out));
        g.read(cnt_b, 0, std.mem.asBytes(&cnt));
        try g.wait();
        summary.failed += cnt[1];

        // the bin's share into the sum; the bin itself kept as ebsdsim keeps it (unweighted)
        const own = try gpa.alloc(f32, bin_out.len);
        errdefer gpa.free(own);
        const wgt = bin.weight;
        for (bin_out, 0..) |v, i| {
            const x: f32 = if (std.math.isFinite(v)) v else 0;
            integrated[i] += x;
            own[i] = if (wgt != 0) @floatCast(@as(f64, x) / wgt) else 0;
        }
        try per_bin.append(gpa, own);
        try bin_kv.append(gpa, @floatCast(bin.voltage_kv));
        try bin_w.append(gpa, @floatCast(wgt));
        for (0..n_k) |k| {
            var acc: f64 = 0;
            for (0..ns) |si| {
                const v = bin_out[k * ns + si];
                acc += if (std.math.isFinite(v)) v else 0;
            }
            delta_view[k] = acc;
        }
        const change = relativeChange(sum_view, delta_view);
        for (sum_view, delta_view) |*a, d| a.* += d;
        summary.last_change = change;
        summary.n_bins_run = @intCast(bi + 1);
        e.events.log("simulation: bin {d} of {d}, {d} kV ({d:.3} of the electrons), {d} beams and {d} weak, μ {d:.4} nm⁻¹: the pattern changed by {d:.4}", .{ bi + 1, bins.len, bin.voltage_kv, bin.weight, n, nw, mu, change });
        if (s.sim_stop > 0 and bi > 0 and change <= s.sim_stop) {
            summary.stopped = 1;
            break;
        }
    }

    // the sites' shares
    const site_w = try gpa.alloc(f32, ns);
    errdefer gpa.free(site_w);
    {
        var tot: f64 = 0;
        for (cell.sites) |si| tot += si.occupancy * @as(f64, @floatFromInt(si.positions.len));
        for (cell.sites, 0..) |si, i| site_w[i] = @floatCast(if (tot > 0) si.occupancy * @as(f64, @floatFromInt(si.positions.len)) / tot else 1.0 / @as(f64, @floatFromInt(ns)));
    }
    const pg_ops = try gpa.dupe(rot.Mat3, cell.pg_ops);
    errdefer gpa.free(pg_ops);
    const normals = try gpa.dupe([3]f64, cell.normals);
    errdefer gpa.free(normals);
    const meta = try metaJson(gpa, e, cell, &summary, bin_kv.items, bin_w.items);
    e.events.log("simulation: {d} of {d} bins, at most {d} beams a direction; {d} directions failed", .{ summary.n_bins_run, summary.n_bins, summary.max_beams, summary.failed });
    return .{
        .gpa = gpa,
        .summary = summary,
        .grid = grid,
        .integrated = integrated,
        .per_bin = try per_bin.toOwnedSlice(gpa),
        .bin_kv = try bin_kv.toOwnedSlice(gpa),
        .bin_weight = try bin_w.toOwnedSlice(gpa),
        .site_weight = site_w,
        .pg_ops = pg_ops,
        .normals = normals,
        .space_group = cell.space_group,
        .point_group = cell.point_group,
        .symbol = cell.symbol,
        .voltage_kv = s.sim_voltage_kv,
        .lattice = .{ 10 * cell.a, 10 * cell.b, 10 * cell.c, std.math.radiansToDegrees(cell.alpha), std.math.radiansToDegrees(cell.beta), std.math.radiansToDegrees(cell.gamma) },
        .meta = meta,
    };
}

/// The file's meta_json: ebsdsim's keys where they mean the same.
fn metaJson(gpa: std.mem.Allocator, e: *Engine, cell: *const crystal.Cell, sm: *const Summary, kv: []const f32, w: []const f32) ![]u8 {
    _ = kv;
    _ = w;
    const s = &e.set;
    const Site = struct { index: usize, atomic_number: u32, symbol: []const u8, occupancy: f64, b_iso_nm_sq: f64, multiplicity: usize };
    const sites = try gpa.alloc(Site, cell.sites.len);
    defer gpa.free(sites);
    for (cell.sites, 0..) |si, i| sites[i] = .{ .index = i, .atomic_number = si.z, .symbol = @import("sim/tables.zig").symbols[si.z - 1], .occupancy = si.occupancy, .b_iso_nm_sq = si.b_nm2, .multiplicity = si.positions.len };
    const deg = 180.0 / std.math.pi;
    const n_bins: usize = sm.n_bins_run;
    const multi_bin = n_bins > 1;
    const multi_site = cell.sites.len > 1;
    const bin_index = try gpa.alloc(usize, n_bins);
    defer gpa.free(bin_index);
    for (bin_index, 0..) |*v, i| v.* = if (multi_bin) 1 + i else 0;
    const site_index = try gpa.alloc(usize, if (multi_site) cell.sites.len else 1);
    defer gpa.free(site_index);
    for (site_index, 0..) |*v, i| v.* = if (multi_site) 1 + i else 0;
    var out: std.Io.Writer.Allocating = .init(gpa);
    errdefer out.deinit();
    try std.json.Stringify.value(.{
        .format = "ebsdsim-master-pattern",
        .format_version = 3,
        .source = "zebsdsolve",
        .voltage_kv = s.sim_voltage_kv,
        .grid_size = 2 * s.sim_halfw + 1,
        .halfw = s.sim_halfw,
        .dmin = s.sim_dmin_nm,
        .energy_binwidth_keV = s.sim_bin_kev,
        .mode = "bloch",
        .solver = "galerkin",
        .rank = s.sim_rank,
        .correction = if (s.sim_correct) "dual-weighted residual, exact on a 30-beam core" else "none",
        .bethe_c_strong = s.sim_c_strong,
        .bethe_c_cutoff = s.sim_c_cutoff,
        .dbdiff_sg_cutoff = s.sim_dbdiff_cutoff,
        .relative_image_stop = s.sim_stop,
        .mc_backend = "surrogate",
        .sigma_deg = s.sim_tilt_deg,
        .n_mc_bins = sm.n_bins,
        .n_bins_run = sm.n_bins_run,
        .n_bin_patterns = sm.n_bins_run,
        .stopped_by_relative_change = sm.stopped != 0,
        .last_relative_change = sm.last_change,
        .pg_num = cell.point_group,
        .pg_symbol = cell.symbol,
        .space_group = cell.space_group,
        .is_centrosymmetric = cell.centro,
        .n_k = sm.n_k,
        .n_sites = sm.n_sites,
        .fail_k = sm.failed,
        .cell = .{
            .a_angstrom = cell.a * 10,
            .b_angstrom = cell.b * 10,
            .c_angstrom = cell.c * 10,
            .alpha_deg = cell.alpha * deg,
            .beta_deg = cell.beta * deg,
            .gamma_deg = cell.gamma * deg,
            .volume_angstrom3 = cell.volume * 1000,
            .density_g_cm3 = cell.density,
            .lattice_centering = &[1]u8{cell.centring},
            .space_group = cell.space_group,
            .pg_num = cell.point_group,
            .pg_symbol = cell.symbol,
            .average_atomic_number = cell.average_z,
            .average_atomic_weight = cell.average_a,
            .n_sites = cell.sites.len,
            .sites = sites,
        },
        .fundamental_sector_axes = .{
            .dims = [_][]const u8{ "energy", "site", "direction" },
            .energy_integrated_index = 0,
            .bin_to_energy_index = bin_index,
            .site_integrated_index = 0,
            .site_to_index = site_index,
        },
    }, .{}, &out.writer);
    // how the crystal was read (the CIF reader's account; JSON already), or null
    out.writer.undo(1);
    try out.writer.writeAll(",\"crystal_reading\":");
    try out.writer.writeAll(if (cell.reading.len > 0) cell.reading else "null");
    try out.writer.writeByte('}');
    return out.toOwnedSlice();
}

