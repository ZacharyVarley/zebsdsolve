//! The detector side on the GPU: the schedule over shaders/detector.wgsl. Simulated patterns,
//! the sums behind the pose fit, orientation refinement run wholly on the device, mutual
//! information, and dictionary scoring. As elsewhere, calls record work; the caller reads.
const std = @import("std");
const gpu_mod = @import("gpu/gpu.zig");
const shaders = @import("shaders");
const spherical = @import("spherical.zig");

const Gpu = gpu_mod.Gpu;
const Buf = gpu_mod.Buf;
const Bind = gpu_mod.Bind;
pub const Frame = spherical.Frame;

/// Sums per item (detector.wgsl NS).
pub const NS = 43;
/// Floats of a refinement record: orientation (4), NCC, NMI, the size of the last refinement
/// step that improved the match (radians), and one the host fills (how far the NMI stands
/// above chance).
pub const RECORD = 8;

/// An item: detector position for the pattern (pixels), the pattern's index, an orientation.
pub const Pose = extern struct {
    t: [3]f32,
    pat: f32,
    q: [4]f32,
};

const P = extern struct {
    n_items: u32 = 0,
    pat_h: u32 = 0,
    pat_w: u32 = 0,
    side: u32 = 0,
    grid_x: u32 = 1,
    n_chunks: u32 = 1,
    flags: u32 = 0,
    n_bins: u32 = 16,
    n_dict: u32 = 0,
    dict0: u32 = 0,
    n_exp: u32 = 0,
    out0: u32 = 0,
    c0: [4]f32 = @splat(0),
    c1: [4]f32 = @splat(0),
    c2: [4]f32 = @splat(0),
    mask: [4]f32 = @splat(0),
    num: [4]f32 = @splat(0),
    t0: [4]f32 = @splat(0),
};

/// A phase's master on the device: [north | south] sheets of side × side.
pub const MasterGpu = struct {
    buf: Buf = .{},
    side: u32 = 0,
    /// its gradient on the sphere, a vec4 per sheet pixel
    grad: Buf = .{},
};

pub const RefineOptions = struct {
    iters: u32 = 8,
    /// largest rotation of one iteration, radians
    max_step: f32 = 0.035,
    /// the scale of the Gauss–Newton curvature a refinement starts with, and its bounds (the
    /// steps adapt it; see refine_step)
    scale0: f32 = 1,
    scale_min: f32 = 0.02,
    scale_max: f32 = 1e4,
    nmi_bins: u32 = 16,
};

pub const Detector = struct {
    g: *Gpu,
    /// pixels of a pattern one invocation of the sums adds up (settings.refine_pixels)
    chunk_px: u32 = 64,
    /// invocations per workgroup of the sums (settings.wg_refine)
    wg: u32 = shaders.wg_default,
    poses: Buf = .{},
    sims: Buf = .{},
    part: Buf = .{},
    sums: Buf = .{},
    state: Buf = .{},
    score: Buf = .{},
    dict_best: Buf = .{},
    /// decoy items, and their mutual information (draw-major)
    decoys: Buf = .{},
    chance: Buf = .{},

    pub fn init(g: *Gpu) Detector {
        return .{ .g = g };
    }

    pub fn deinit(self: *Detector) void {
        inline for (.{ "poses", "sims", "part", "sums", "state", "score", "dict_best", "decoys", "chance" }) |f| self.g.release(&@field(self, f));
    }

    fn pipe(self: *Detector, comptime entry: []const u8) u32 {
        return self.g.pipeline("detector/" ++ entry, shaders.detector, entry) catch gpu_mod.NO_PIPE;
    }

    fn params(self: *const Detector, m: MasterGpu, fr: Frame, n_items: u32) P {
        const hw = fr.pat_h * fr.pat_w;
        return .{
            .n_items = n_items,
            .pat_h = fr.pat_h,
            .pat_w = fr.pat_w,
            .side = m.side,
            .n_chunks = std.math.clamp((hw + self.chunk_px - 1) / @max(self.chunk_px, 1), 1, 1024),
            .c0 = .{ fr.c0[0], fr.c0[1], fr.c0[2], 0 },
            .c1 = .{ fr.c1[0], fr.c1[1], fr.c1[2], 0 },
            .c2 = .{ fr.c2[0], fr.c2[1], fr.c2[2], 0 },
            .mask = .{ fr.mask[0], fr.mask[1], fr.mask[2], 0 },
            // differences on the sphere a master pixel apart
            .num = .{ @floatCast(@sqrt(2.0 * std.math.pi) / @as(f64, @floatFromInt(@max(m.side, 2)))), 0, 0, 0 },
        };
    }

    fn run(self: *Detector, comptime entry: []const u8, p0: P, groups: u32, binds: []const Bind) void {
        var p = p0;
        const f = Gpu.flat(groups);
        p.grid_x = f[0];
        var all: [8]Bind = undefined;
        all[0] = self.g.uniform(0, std.mem.asBytes(&p));
        @memcpy(all[1..][0..binds.len], binds);
        self.g.dispatch(self.pipe(entry), f[0], f[1], 1, all[0 .. binds.len + 1]);
    }

    /// Upload the items.
    pub fn setPoses(self: *Detector, poses: []const Pose) void {
        self.g.ensure(&self.poses, poses.len * @sizeOf(Pose));
        self.g.writeSlice(self.poses, 0, Pose, poses);
    }

    /// Record: simulated patterns of the n items into `sims` (n · h · w).
    pub fn project(self: *Detector, m: MasterGpu, fr: Frame, n: u32) void {
        const hw: u64 = @as(u64, fr.pat_h) * fr.pat_w;
        self.g.ensure(&self.sims, n * hw * 4);
        self.run("project", self.params(m, fr, n), @intCast((n * hw + 255) / 256), &.{ m.buf.at(1), self.poses.at(2), self.sims.at(4) });
    }

    /// Record: the NS sums of each item against its pattern in `pats`, into `sums`.
    /// `rotation_only` leaves out the three detector-shift directions (orientation refinement
    /// does not use them).
    pub fn sumsOf(self: *Detector, m: MasterGpu, fr: Frame, pats: Buf, n: u32, rotation_only: bool) void {
        var p = self.params(m, fr, n);
        if (rotation_only) p.flags = 2;
        self.g.ensure(&self.part, @as(u64, n) * p.n_chunks * NS * 4);
        self.g.ensure(&self.sums, @as(u64, n) * NS * 4);
        // (the master's values lie beside its gradient: one table)
        {
            var ps = p;
            const f = Gpu.flat((n * p.n_chunks + self.wg - 1) / self.wg);
            ps.grid_x = f[0];
            const pipe_id = self.g.pipelineWith("detector/sums", shaders.sums_wg, self.wg, shaders.detector_body, "sums") catch gpu_mod.NO_PIPE;
            self.g.dispatch(pipe_id, f[0], f[1], 1, &.{ self.g.uniform(0, std.mem.asBytes(&ps)), self.poses.at(2), pats.at(3), self.part.at(5), m.grad.at(21) });
        }
        self.run("sums_reduce", p, (n * NS + 63) / 64, &.{ self.part.at(6), self.sums.at(7) });
    }

    /// Record: refine the n items' orientations against their patterns, score them (NCC and
    /// NMI), and write one record per item to `records` starting at record `out0`.
    pub fn refine(self: *Detector, m: MasterGpu, fr: Frame, pats: Buf, n: u32, o: RefineOptions, records: Buf, out0: u32) void {
        const g = self.g;
        var p = self.params(m, fr, n);
        p.mask[3] = o.scale0;
        p.num[1] = o.max_step;
        p.num[2] = o.scale_min;
        p.num[3] = o.scale_max;
        p.n_bins = o.nmi_bins;
        p.out0 = out0;
        g.ensure(&self.state, @as(u64, n) * 24 * 4);
        g.ensure(&self.score, @as(u64, n) * 4);
        self.run("refine_init", p, (n + 63) / 64, &.{ self.state.at(9), self.poses.at(10) });
        var it: u32 = 0;
        while (it <= o.iters) : (it += 1) {
            self.sumsOf(m, fr, pats, n, true);
            var ps = p;
            ps.flags = if (it == o.iters) 1 else 0;
            self.run("refine_step", ps, (n + 63) / 64, &.{ self.sums.at(8), self.state.at(9), self.poses.at(10) });
        }
        self.run("nmi", p, n, &.{ m.buf.at(1), self.poses.at(2), pats.at(3), self.score.at(11) });
        self.run("collect", p, (n + 63) / 64, &.{ self.state.at(14), self.score.at(15), records.at(16) });
    }

    /// Record: make the n items from `items` ((t, pattern index) each) and the orientations
    /// that start records `first`, `first + 1`, … of `peaks` (spherical.zig's peaks, or
    /// refinement records: both begin with the orientation).
    pub fn seed(self: *Detector, fr: Frame, items: Buf, peaks: Buf, first: u32, n: u32) void {
        self.g.ensure(&self.poses, @as(u64, n) * @sizeOf(Pose));
        var p = self.params(.{}, fr, n);
        p.out0 = first;
        self.run("seed", p, (n + 63) / 64, &.{ self.poses.at(10), items.at(17), peaks.at(18) });
    }

    /// Record: what mutual information chance gives the n items: each item's pattern against
    /// the master at `draws` random orientations, into `chance` (draws · n values, draw-major).
    pub fn chanceOf(self: *Detector, m: MasterGpu, fr: Frame, pats: Buf, n: u32, draws: u32, nmi_bins: u32) void {
        const g = self.g;
        g.ensure(&self.decoys, @as(u64, n) * @sizeOf(Pose));
        g.ensure(&self.chance, @as(u64, n) * draws * 4);
        var p = self.params(m, fr, n);
        p.n_bins = nmi_bins;
        for (0..draws) |d| {
            p.n_exp = @intCast(d);
            p.dict0 = 0;
            self.run("decoy", p, (n + 63) / 64, &.{ self.poses.at(2), self.decoys.at(10) });
            p.dict0 = @intCast(d * n);
            self.run("nmi", p, n, &.{ m.buf.at(1), self.decoys.at(2), pats.at(3), self.chance.at(11) });
        }
    }

    /// Record: make the n items from `items` and each pattern's dictionary hit.
    pub fn seedDict(self: *Detector, fr: Frame, items: Buf, dict_q: Buf, n: u32) void {
        self.g.ensure(&self.poses, @as(u64, n) * @sizeOf(Pose));
        self.run("seed_dict", self.params(.{}, fr, n), (n + 63) / 64, &.{ self.poses.at(10), items.at(17), dict_q.at(19), self.dict_best.at(20) });
    }

    /// Record: reset the dictionary's running best for `n_exp` patterns.
    pub fn dictBegin(self: *Detector, n_exp: u32) !void {
        const g = self.g;
        g.ensure(&self.dict_best, @as(u64, n_exp) * 8);
        const start = try g.gpa.alloc(u32, @as(usize, n_exp) * 2);
        defer g.gpa.free(start);
        for (0..n_exp) |i| {
            start[2 * i] = @bitCast(@as(f32, -3.4e38));
            start[2 * i + 1] = 0;
        }
        g.writeSlice(self.dict_best, 0, u32, start);
    }

    /// Record: simulate dictionary entries [dict0, dict0 + n) of `dict_q` at detector position
    /// `t` into `sims`, zero-mean and unit-norm under the mask.
    pub fn dictSims(self: *Detector, m: MasterGpu, fr: Frame, t: [3]f32, dict_q: Buf, n: u32, dict0: u32) void {
        const hw: u64 = @as(u64, fr.pat_h) * fr.pat_w;
        self.g.ensure(&self.sims, n * hw * 4);
        var p = self.params(m, fr, n);
        p.dict0 = dict0;
        p.t0 = .{ t[0], t[1], t[2], 0 };
        self.run("project_dict", p, @intCast((n * hw + 255) / 256), &.{ m.buf.at(1), self.sims.at(4), dict_q.at(19) });
        self.run("sim_norm", p, n, &.{self.sims.at(4)});
    }

    /// Record: simulate dictionary entries [dict0, dict0 + n) and fold their inner products
    /// with the first `n_exp` patterns into the running best.
    pub fn dictChunk(self: *Detector, m: MasterGpu, fr: Frame, t: [3]f32, pats: Buf, dict_q: Buf, n: u32, dict0: u32, n_exp: u32) void {
        self.dictSims(m, fr, t, dict_q, n, dict0);
        var p = self.params(m, fr, n);
        p.dict0 = dict0;
        p.n_dict = n;
        p.n_exp = n_exp;
        self.run("dict_score", p, n_exp, &.{ pats.at(3), self.sims.at(12), self.dict_best.at(13) });
    }
};
