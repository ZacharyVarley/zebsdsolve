//! Dictionary indexing in a learned subspace (shaders/pca.wgsl): the schedule.
//!
//! A dictionary's simulated patterns are compared with an experimental pattern pixel by pixel
//! in plain dictionary indexing, and simulated again for every batch. Here the dictionary is
//! simulated once: as it streams by, Oja's rule learns an orthonormal basis W of the k
//! directions that hold most of its energy, and every entry is kept as its k coefficients. A
//! batch of experimental patterns is then projected onto W and matched by coefficients: n · k
//! multiplications per comparison instead of the pixel count, and nothing simulated.
//!
//! Oja's step on a batch X of dictionary patterns is W ← W + η (W Xᵀ) X / n, after which the
//! rows are made orthonormal again: the device forms W Wᵀ, the host factors it (Cholesky, in
//! f64) and the device applies the inverse factor. The step size falls as 1 / t (Oja's and
//! Krasulina's schedule); early on it is large enough that a step is nearly a power iteration
//! on the batch's covariance.
const std = @import("std");
const gpu_mod = @import("gpu/gpu.zig");
const shaders = @import("shaders");

const Gpu = gpu_mod.Gpu;
const Buf = gpu_mod.Buf;
const Bind = gpu_mod.Bind;

const P = extern struct {
    n: u32 = 0,
    p: u32 = 0,
    k: u32 = 0,
    grid_x: u32 = 1,
    row0: u32 = 0,
    n_dict: u32 = 0,
    pad0: u32 = 0,
    pad1: u32 = 0,
    eta: [4]f32 = @splat(0),
};

pub const Pca = struct {
    g: *Gpu,
    /// coefficients of the batch in hand, the basis before it is orthonormal, W Wᵀ and its
    /// inverse factor
    y: Buf = .{},
    w2: Buf = .{},
    gram: Buf = .{},
    rot: Buf = .{},
    steps: u32 = 0,

    pub fn init(g: *Gpu) Pca {
        return .{ .g = g };
    }

    pub fn deinit(self: *Pca) void {
        inline for (.{ "y", "w2", "gram", "rot" }) |f| self.g.release(&@field(self, f));
    }

    fn run(self: *Pca, comptime entry: []const u8, p0: P, groups: u64, binds: []const Bind) void {
        var p = p0;
        const f = Gpu.flat(@intCast(groups));
        p.grid_x = f[0];
        var all: [6]Bind = undefined;
        all[0] = self.g.uniform(0, std.mem.asBytes(&p));
        @memcpy(all[1..][0..binds.len], binds);
        const pipe = self.g.pipeline("pca/" ++ entry, shaders.pca, entry) catch gpu_mod.NO_PIPE;
        self.g.dispatch(pipe, f[0], f[1], 1, all[0 .. binds.len + 1]);
    }

    /// Workgroups of `project`: an invocation takes four patterns by four components.
    fn projectGroups(n: u32, k: u32) u64 {
        return (@as(u64, (n + 3) / 4) * ((k + 3) / 4) + 63) / 64;
    }

    /// A random basis of k rows of p pixels in `w` (seeded, so a run repeats), orthonormal.
    pub fn begin(self: *Pca, w: *Buf, k: u32, p: u32) !void {
        const g = self.g;
        const gpa = g.gpa;
        const host = try gpa.alloc(f32, @as(usize, k) * p);
        defer gpa.free(host);
        var prng = std.Random.DefaultPrng.init(0x7a6562);
        const r = prng.random();
        for (host) |*v| v.* = r.floatNorm(f32);
        g.ensure(w, host.len * 4);
        g.ensure(&self.w2, host.len * 4);
        g.writeSlice(self.w2, 0, f32, host);
        self.steps = 0;
        try self.orthonormalize(w.*, k, p);
    }

    /// `w` ← the rows of `w2` made orthonormal (one readback of k² numbers).
    fn orthonormalize(self: *Pca, w: Buf, k: u32, p: u32) !void {
        const g = self.g;
        const gpa = g.gpa;
        const kk: usize = @as(usize, k) * k;
        g.ensure(&self.gram, kk * 4);
        g.ensure(&self.rot, kk * 4);
        const prm: P = .{ .p = p, .k = k };
        self.run("gram", prm, (kk + 63) / 64, &.{ self.w2.at(2), self.gram.at(6) });
        const gh = try gpa.alloc(f32, kk);
        defer gpa.free(gh);
        g.read(self.gram, 0, std.mem.sliceAsBytes(gh));
        try g.wait();
        // W2 W2ᵀ = L Lᵀ; the rows of L⁻¹ W2 are orthonormal
        const L = try gpa.alloc(f64, kk);
        defer gpa.free(L);
        var jitter: f64 = 0;
        var trace: f64 = 0;
        for (0..k) |i| trace += gh[i * k + i];
        factor: while (true) {
            @memset(L, 0);
            for (0..k) |i| {
                for (0..i + 1) |j| {
                    var s: f64 = gh[i * k + j];
                    if (i == j) s += jitter;
                    for (0..j) |m| s -= L[i * k + m] * L[j * k + m];
                    if (i == j) {
                        if (!(s > 1e-12 * trace / @as(f64, @floatFromInt(k)))) {
                            // rows nearly dependent: lift the diagonal and factor again
                            jitter = if (jitter == 0) 1e-6 * trace / @as(f64, @floatFromInt(k)) else jitter * 10;
                            if (jitter > trace) return error.PcaBasisDegenerate;
                            continue :factor;
                        }
                        L[i * k + i] = @sqrt(s);
                    } else L[i * k + j] = s / L[j * k + j];
                }
            }
            break;
        }
        // L⁻¹, lower triangular, column by column
        const inv = try gpa.alloc(f32, kk);
        defer gpa.free(inv);
        @memset(inv, 0);
        const col = try gpa.alloc(f64, k);
        defer gpa.free(col);
        for (0..k) |j| {
            for (j..k) |i| {
                var s: f64 = if (i == j) 1 else 0;
                for (j..i) |m| s -= L[i * k + m] * col[m];
                col[i] = s / L[i * k + i];
                inv[i * k + j] = @floatCast(col[i]);
            }
        }
        g.writeSlice(self.rot, 0, f32, inv);
        self.run("rotate", prm, (@as(u64, k) * p + 63) / 64, &.{ self.w2.at(2), w.at(5), self.rot.at(7) });
    }

    /// One step of Oja's rule on the n patterns (rows of p pixels) in `x`.
    pub fn learn(self: *Pca, w: Buf, x: Buf, n: u32, k: u32, p: u32) !void {
        const g = self.g;
        g.ensure(&self.y, @as(u64, n) * k * 4);
        var prm: P = .{ .n = n, .p = p, .k = k };
        self.run("project", prm, projectGroups(n, k), &.{ x.at(1), w.at(2), self.y.at(3) });
        // unit-norm patterns spread a total variance of 1 over the directions, so a leading
        // direction holds a few times 1 / k of it: η of 16 k grows those by an order of
        // magnitude a step at first, and then falls as 1 / t
        const eta = 16.0 * @as(f64, @floatFromInt(k)) / (1.0 + @as(f64, @floatFromInt(self.steps)) / 8.0);
        prm.eta[0] = @floatCast(eta / @as(f64, @floatFromInt(n)));
        self.run("accumulate", prm, (@as(u64, k) * p + 63) / 64, &.{ x.at(1), w.at(2), self.y.at(4), self.w2.at(5) });
        self.steps += 1;
        try self.orthonormalize(w, k, p);
    }

    /// Record: the coefficients of the n patterns in `x` into rows row0 … of `out`.
    pub fn project(self: *Pca, w: Buf, x: Buf, n: u32, k: u32, p: u32, out: Buf, row0: u32) void {
        const prm: P = .{ .n = n, .p = p, .k = k, .row0 = row0 };
        self.run("project", prm, projectGroups(n, k), &.{ x.at(1), w.at(2), out.at(3) });
    }

    /// Record: fold the best of the `n_dict` dictionary entries in `coef` (indices from `first`)
    /// for each of the n experimental coefficient rows in `y` into `best`.
    pub fn match(self: *Pca, coef: Bind, y: Buf, n: u32, k: u32, n_dict: u32, first: u32, best: Buf) void {
        const prm: P = .{ .n = n, .k = k, .row0 = first, .n_dict = n_dict };
        var c = coef;
        c.slot = 1;
        self.run("nearest", prm, (n + 7) / 8, &.{ c, y.at(4), best.at(8) });
    }
};
