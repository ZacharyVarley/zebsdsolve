//! Pattern preprocessing on the GPU (shaders/prep.wgsl): patterns in (as stored, or already
//! reduced on the host), filtered patterns out, zero-mean and unit-norm under the mask. The
//! filters run in a fixed order: static background, dynamic background, CLAHE, band pass, then
//! the mask and the normalization. CLAHE alone also filters a master pattern's two hemisphere
//! images; a master's other filters act on the sphere (`sphereBlur`, `symmetrize`).
const std = @import("std");
const gpu_mod = @import("gpu/gpu.zig");
const shaders = @import("shaders");
const settings_mod = @import("settings.zig");

const Gpu = gpu_mod.Gpu;
const Buf = gpu_mod.Buf;
const Background = settings_mod.Background;

const P = extern struct {
    n: u32 = 0,
    h: u32 = 0,
    w: u32 = 0,
    grid_x: u32 = 1,
    pixel: u32 = 0,
    op: u32 = 0,
    tiles: u32 = 1,
    bins: u32 = 2,
    radius: u32 = 0,
    pad_h: u32 = 0,
    pad_w: u32 = 0,
    tile_h: u32 = 1,
    tile_w: u32 = 1,
    bin: u32 = 1,
    cpad: u32 = 0,
    pad2: u32 = 0,
    k: [4]f32 = @splat(0),
    mask: [4]f32 = @splat(0),
};

pub const Options = struct {
    /// the static background at the images' size, subtracted (null: none)
    static: ?Buf = null,
    dynamic: Background = .none,
    /// sigma of the dynamic background's blur, pixels
    dynamic_sigma: f64 = 0,
    clahe: bool = false,
    clahe_grid: u32 = 4,
    clahe_bins: u32 = 64,
    clahe_clip: f64 = 40,
    bandpass: bool = false,
    /// sigmas in pixels of these images
    bp_fine: f64 = 0,
    bp_coarse: f64 = 0,
    /// mask centre (row, col) and squared radius in pixels (≤ 0: none)
    mask: [3]f32 = .{ 0, 0, 0 },
    /// finish with zero mean and unit norm under the mask (off: the filtered values as they are)
    normalize: bool = true,
};

pub const Prep = struct {
    g: *Gpu,
    /// scratch images, each as large as the batch
    tmp: [3]Buf = .{ .{}, .{}, .{} },
    hist: Buf = .{},
    cdf: Buf = .{},
    /// a wide blur's coarse grids
    coarse: [2]Buf = .{ .{}, .{} },

    pub fn init(g: *Gpu) Prep {
        return .{ .g = g };
    }

    pub fn deinit(self: *Prep) void {
        for (&self.tmp) |*b| self.g.release(b);
        for (&self.coarse) |*b| self.g.release(b);
        self.g.release(&self.hist);
        self.g.release(&self.cdf);
    }

    fn pipe(self: *Prep, comptime entry: []const u8) u32 {
        return self.g.pipeline("prep/" ++ entry, shaders.prep, entry) catch gpu_mod.NO_PIPE;
    }

    /// One invocation per element of `count` (the batch's pixels unless given).
    fn perCount(self: *Prep, comptime entry: []const u8, p0: P, count: u64, binds: []const gpu_mod.Bind) void {
        var p = p0;
        const f = Gpu.flat(@intCast((count + 255) / 256));
        p.grid_x = f[0];
        var all: [8]gpu_mod.Bind = undefined;
        all[0] = self.g.uniform(0, std.mem.asBytes(&p));
        @memcpy(all[1..][0..binds.len], binds);
        self.g.dispatch(self.pipe(entry), f[0], f[1], 1, all[0 .. binds.len + 1]);
    }

    fn perPixel(self: *Prep, comptime entry: []const u8, p0: P, binds: []const gpu_mod.Bind) void {
        self.perCount(entry, p0, @as(u64, p0.n) * p0.h * p0.w, binds);
    }

    fn perImage(self: *Prep, comptime entry: []const u8, p0: P, binds: []const gpu_mod.Bind) void {
        var p = p0;
        const f = Gpu.flat(p.n);
        p.grid_x = f[0];
        var all: [8]gpu_mod.Bind = undefined;
        all[0] = self.g.uniform(0, std.mem.asBytes(&p));
        @memcpy(all[1..][0..binds.len], binds);
        self.g.dispatch(self.pipe(entry), f[0], f[1], 1, all[0 .. binds.len + 1]);
    }

    /// Gaussian blur of `src` into `dst` through `mid`. A wide one (sigma of 12 pixels or
    /// more) is computed from sums over boxes of sigma / 6 pixels (prep.wgsl says how, and how
    /// close it comes).
    fn blur(self: *Prep, p0: P, sigma: f64, src: Buf, mid: Buf, dst: Buf) void {
        var p = p0;
        const s: u32 = @intFromFloat(@floor(sigma / 6));
        if (s < 2) {
            p.radius = @intFromFloat(@max(1.0, @ceil(3 * sigma)));
            p.k[0] = @floatCast(1.0 / (2 * sigma * sigma));
            self.perPixel("blur_rows", p, &.{ src.at(3), mid.at(2) });
            self.perPixel("blur_cols", p, &.{ mid.at(3), dst.at(2) });
            return;
        }
        // the variances add: the box's, the coarse blur's and the spread's (deviation s)
        const sf: f64 = @floatFromInt(s);
        const coarse_sigma = @sqrt(sigma * sigma - sf * sf - (sf * sf - 1) / 12);
        p.bin = s;
        p.cpad = 3;
        p.radius = @intFromFloat(@ceil(4 * coarse_sigma));
        p.k[0] = @floatCast(1.0 / (2 * coarse_sigma * coarse_sigma));
        p.k[2] = @floatCast(1.0 / (2 * sf * sf));
        const n: u64 = p.n;
        const hs: u64 = (p.h + s - 1) / s;
        const ws: u64 = (p.w + s - 1) / s;
        const hp = hs + 2 * p.cpad;
        const wp = ws + 2 * p.cpad;
        for (&self.coarse) |*b| self.g.ensure(b, n * hp * @max(wp, p.w) * 8);
        const co = self.coarse;
        self.perCount("box_sum", p, n * hs * ws, &.{ src.at(3), co[0].at(9) });
        self.perCount("coarse_rows", p, n * hs * wp, &.{ co[0].at(8), co[1].at(9) });
        self.perCount("coarse_cols", p, n * hp * wp, &.{ co[1].at(8), co[0].at(9) });
        self.perCount("spread_rows", p, n * hp * p.w, &.{ co[0].at(8), co[1].at(9) });
        self.perPixel("spread_cols", p, &.{ co[1].at(8), dst.at(2) });
    }

    /// Record the filter chain on the n images of h × w in `img` (in place).
    pub fn filter(self: *Prep, img: Buf, n: u32, h: u32, w: u32, o: Options) void {
        const g = self.g;
        const bytes: u64 = @as(u64, n) * h * w * 4;
        for (&self.tmp) |*b| g.ensure(b, bytes);
        var p: P = .{ .n = n, .h = h, .w = w, .mask = .{ o.mask[0], o.mask[1], o.mask[2], 0 } };
        // `cur` holds the image so far; the others are free
        var cur = img;
        var free: [3]Buf = self.tmp;
        const take = struct {
            /// Make `out` the current image and return the old one to the free list.
            fn swap(c: *Buf, fr: *[3]Buf, i: usize) void {
                const t = c.*;
                c.* = fr[i];
                fr[i] = t;
            }
        }.swap;

        if (o.static) |bg| self.perPixel("background", p, &.{ cur.at(2), bg.at(4) });
        if (o.dynamic != .none and o.dynamic_sigma > 0) {
            self.blur(p, o.dynamic_sigma, cur, free[0], free[1]);
            p.op = if (o.dynamic == .divide) 1 else 0;
            self.perPixel("combine", p, &.{ cur.at(3), free[1].at(4), free[0].at(2) });
            take(&cur, &free, 0);
        }
        if (o.clahe) {
            p.tiles = o.clahe_grid;
            p.bins = o.clahe_bins;
            p.k[1] = @floatCast(o.clahe_clip);
            p.pad_h = (p.tiles - h % p.tiles) % p.tiles;
            p.pad_w = (p.tiles - w % p.tiles) % p.tiles;
            p.tile_h = @max(1, (h + p.pad_h) / p.tiles);
            p.tile_w = @max(1, (w + p.pad_w) / p.tiles);
            const cells: u64 = @as(u64, n) * p.tiles * p.tiles * p.bins * 4;
            g.ensure(&self.hist, cells);
            g.ensure(&self.cdf, cells);
            self.perImage("stretch", p, &.{cur.at(2)});
            g.clear(self.hist);
            self.perCount("clahe_hist", p, @as(u64, n) * (h + p.pad_h) * (w + p.pad_w), &.{ cur.at(3), self.hist.at(5) });
            {
                var pc = p;
                const f = Gpu.flat((n * p.tiles * p.tiles + 63) / 64);
                pc.grid_x = f[0];
                g.dispatch(self.pipe("clahe_cdf"), f[0], f[1], 1, &.{ g.uniform(0, std.mem.asBytes(&pc)), self.hist.at(5), self.cdf.at(6) });
            }
            self.perPixel("clahe_apply", p, &.{ cur.at(3), self.cdf.at(7), free[0].at(2) });
            take(&cur, &free, 0);
        }
        if (o.bandpass and o.bp_coarse > 0) {
            // fine into free[1] (or the image itself), coarse into free[2], difference into free[0]
            var fine = cur;
            if (o.bp_fine > 0.05) {
                self.blur(p, o.bp_fine, cur, free[0], free[1]);
                fine = free[1];
            }
            self.blur(p, o.bp_coarse, cur, free[0], free[2]);
            p.op = 0;
            self.perPixel("combine", p, &.{ fine.at(3), free[2].at(4), free[0].at(2) });
            take(&cur, &free, 0);
        }
        if (o.normalize) self.perImage("normalize", p, &.{cur.at(2)});
        if (cur.id != img.id) g.copy(cur, 0, img, 0, bytes);
    }

    fn msPipe(self: *Prep, comptime entry: []const u8) u32 {
        return self.g.pipeline("mastersph/" ++ entry, shaders.mastersph, entry) catch gpu_mod.NO_PIPE;
    }

    const MsP = extern struct { side: u32, grid_x: u32, radius: u32, n_ops: u32, k: [4]f32 };

    /// Record: Gaussian smoothing on the sphere of the master in `img` ([north | south], side ×
    /// side each) by `sigma` sheet pixels at the pole, in place. Passes of doubling width, each
    /// sampling the last one's output on a lattice fine enough for it.
    pub fn sphereBlur(self: *Prep, img: Buf, side: u32, sigma: f64) void {
        const g = self.g;
        if (!(sigma > 0.05)) return;
        const bytes: u64 = 2 * @as(u64, side) * side * 4;
        g.ensure(&self.tmp[0], bytes);
        const px_rad = @sqrt(2.0 * std.math.pi) / @as(f64, @floatFromInt(side));
        const f = Gpu.flat(@intCast((bytes / 4 + 255) / 256));
        var left = sigma * sigma;
        var s: f64 = @min(sigma, 2.0);
        var src = img;
        var dst = self.tmp[0];
        while (left > 1e-6) {
            if (s * s > left or left - s * s < 0.02 * left) s = @sqrt(left);
            left -= s * s;
            // lattice step 0.6 σ, out to ±2.4 σ
            const sr = s * px_rad;
            const p: MsP = .{ .side = side, .grid_x = f[0], .radius = 4, .n_ops = 0, .k = .{ @floatCast(0.6 * sr), @floatCast(1.0 / (2 * sr * sr)), 0, 0 } };
            g.dispatch(self.msPipe("sphere_blur"), f[0], f[1], 1, &.{ g.uniform(0, std.mem.asBytes(&p)), src.at(1), dst.at(2) });
            std.mem.swap(Buf, &src, &dst);
            s *= 2;
        }
        if (src.id != img.id) g.copy(src, 0, img, 0, bytes);
    }

    /// Record: replace the master in `img` by its mean over the point-group operators in `ops`
    /// (three vec4 rows each), in place.
    pub fn symmetrize(self: *Prep, img: Buf, side: u32, ops: Buf, n_ops: u32) void {
        const g = self.g;
        const bytes: u64 = 2 * @as(u64, side) * side * 4;
        g.ensure(&self.tmp[0], bytes);
        const f = Gpu.flat(@intCast((bytes / 4 + 255) / 256));
        const p: MsP = .{ .side = side, .grid_x = f[0], .radius = 0, .n_ops = n_ops, .k = @splat(0) };
        g.dispatch(self.msPipe("symmetrize"), f[0], f[1], 1, &.{ g.uniform(0, std.mem.asBytes(&p)), img.at(1), self.tmp[0].at(2), ops.at(3) });
        g.copy(self.tmp[0], 0, img, 0, bytes);
    }

    /// Record: the gradient on the sphere of the master in `img` at every sheet pixel, into
    /// `out` (a vec4 per pixel, grown to hold them).
    pub fn gradient(self: *Prep, img: Buf, side: u32, out: *Buf) void {
        const g = self.g;
        const n: u64 = 2 * @as(u64, side) * side;
        g.ensure(out, n * 16);
        const f = Gpu.flat(@intCast((n + 255) / 256));
        const p: MsP = .{ .side = side, .grid_x = f[0], .radius = 0, .n_ops = 0, .k = .{ @floatCast(@sqrt(2.0 * std.math.pi) / @as(f64, @floatFromInt(side))), 0, 0, 0 } };
        g.dispatch(self.msPipe("gradient"), f[0], f[1], 1, &.{ g.uniform(0, std.mem.asBytes(&p)), img.at(1), out.at(4) });
    }

    /// Record: unpack the n patterns of h × w in `raw` (pixel: 0 u8, 1 u16, 2 f32) into `dst`.
    pub fn load(self: *Prep, raw: Buf, n: u32, h: u32, w: u32, pixel: u32, dst: Buf) void {
        const p: P = .{ .n = n, .h = h, .w = w, .pixel = pixel };
        self.perPixel("load", p, &.{ raw.at(1), dst.at(2) });
    }

    /// Record: unpack the patterns as `load` does, then the filter chain on them.
    pub fn run(self: *Prep, raw: Buf, n: u32, h: u32, w: u32, pixel: u32, o: Options, dst: Buf) void {
        self.load(raw, n, h, w, pixel, dst);
        self.filter(dst, n, h, w, o);
    }
};
