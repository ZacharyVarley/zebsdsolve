//! Non-local pattern averaging (NLPAR) on the GPU: the schedule over shaders/nlpar.wgsl, which
//! states the method. The engine hands over a window: the cells whose averaged patterns are
//! wanted, then ring after ring of the cells around them, with each cell's surroundings as
//! indices into that list. A pass needs its input `radius` rings further out than its output,
//! and the noise of every stored pattern that went into what it compares, each measured against
//! that pattern's own nearest neighbours: `radius · passes + 1` rings in all. Each pass works on
//! a shorter prefix of the list than the one before, and the last on the wanted cells alone.
const std = @import("std");
const gpu_mod = @import("gpu/gpu.zig");
const shaders = @import("shaders");

const Gpu = gpu_mod.Gpu;
const Buf = gpu_mod.Buf;
const Bind = gpu_mod.Bind;

/// No neighbour (outside the map): nlpar.wgsl's NONE.
pub const NONE: u32 = 0xffffffff;

const P = extern struct {
    n: u32 = 0,
    px: u32 = 0,
    radius: u32 = 0,
    reach: u32 = 0,
    grid_x: u32 = 1,
    out0: u32 = 0,
    spread: u32 = 0,
    pad0: u32 = 0,
    num: [4]f32 = @splat(0),
};

pub const Window = struct {
    /// per cell of the list, the list indices (or NONE) of the cells up to `reach()` away each
    /// way: a square of `side()` × `side()`, rows first, the cell itself in the middle
    nbr: []const u32,
    /// counts[d]: the cells within d rings of the wanted ones (counts[0] of them are wanted),
    /// for d up to `rings()`
    counts: []const u32,
    radius: u32,
    passes: u32,

    /// How far a pattern's average draws on stored patterns.
    pub fn reachOf(radius: u32, passes: u32) u32 {
        return radius * passes;
    }

    /// Rings of cells around the wanted ones that the passes need.
    pub fn ringsOf(radius: u32, passes: u32) u32 {
        return radius * passes + 1;
    }

    pub fn side(self: Window) u32 {
        return 2 * reachOf(self.radius, self.passes) + 1;
    }
};

pub const Nlpar = struct {
    g: *Gpu,
    /// the window's patterns (f32) before a pass and after it, in turn
    a: Buf = .{},
    b: Buf = .{},
    nbr: Buf = .{},
    d2: Buf = .{},
    wts: Buf = .{},
    /// the stored patterns' noise (variance per pixel), per cell of the list
    sig: Buf = .{},
    /// each pattern's shares of the stored ones, before a pass and after it, in turn
    sa: Buf = .{},
    sb: Buf = .{},
    /// which of the two holds the averages' shares after `average`
    final: Buf = .{},

    pub fn init(g: *Gpu) Nlpar {
        return .{ .g = g };
    }

    pub fn deinit(self: *Nlpar) void {
        inline for (.{ "a", "b", "nbr", "d2", "wts", "sig", "sa", "sb" }) |f| self.g.release(&@field(self, f));
    }

    fn run(self: *Nlpar, comptime entry: []const u8, p0: P, groups: u64, binds: []const Bind) void {
        var p = p0;
        const f = Gpu.flat(@intCast(@max(groups, 1)));
        p.grid_x = f[0];
        var all: [6]Bind = undefined;
        all[0] = self.g.uniform(0, std.mem.asBytes(&p));
        @memcpy(all[1..][0..binds.len], binds);
        const pipe = self.g.pipeline("nlpar/" ++ entry, shaders.nlpar, entry) catch gpu_mod.NO_PIPE;
        self.g.dispatch(pipe, f[0], f[1], 1, all[0 .. binds.len + 1]);
    }

    /// Record: the passes over the window's patterns in `self.a` (`px` f32 each, in the list's
    /// order), leaving the wanted cells' averages in `out` from pattern `out0` on. Afterwards
    /// `sig` holds every listed cell's noise and `final` the wanted cells' shares of the stored
    /// patterns (`side()`² each).
    pub fn average(self: *Nlpar, w: Window, px: u32, lambda: f64, out: Buf, out0: u32) void {
        const g = self.g;
        const rings = Window.ringsOf(w.radius, w.passes);
        const n_win: u64 = w.counts[rings];
        const k: u64 = (2 * w.radius + 1) * (2 * w.radius + 1);
        const f: u64 = w.side() * w.side();
        if (w.passes > 1) g.ensure(&self.b, n_win * px * 4);
        g.ensure(&self.nbr, n_win * f * 4);
        g.ensure(&self.d2, n_win * k * 4);
        g.ensure(&self.wts, n_win * k * 4);
        g.ensure(&self.sig, n_win * 4);
        g.ensure(&self.sa, n_win * f * 4);
        g.ensure(&self.sb, n_win * f * 4);
        g.writeSlice(self.nbr, 0, u32, w.nbr);
        var p: P = .{ .px = px, .radius = w.radius, .reach = Window.reachOf(w.radius, w.passes) };
        p.num = .{ @floatCast(1.0 / (lambda * lambda)), @floatCast(@sqrt(0.5 * @as(f64, @floatFromInt(px)))), 0, 0 };
        var src = self.a;
        var dst = self.b;
        var s_in = self.sa;
        var s_out = self.sb;
        var t: u32 = 1;
        while (t <= w.passes) : (t += 1) {
            // the cells this pass leaves averaged: what the passes after it read
            const d = (w.passes - t) * w.radius;
            p.spread = (t - 1) * w.radius;
            // differences of every pair a weight below reads (and, from the stored patterns,
            // of every pair a noise reads: all of them)
            p.n = w.counts[if (t == 1) rings else d + w.radius];
            self.run("distance", p, @as(u64, p.n) * (k / 2), &.{ src.at(1), self.nbr.at(3), self.d2.at(4) });
            if (t == 1) {
                p.n = w.counts[rings - 1];
                self.run("sigma", p, (p.n + 63) / 64, &.{ self.nbr.at(3), self.d2.at(5), self.sig.at(6) });
            }
            p.n = w.counts[d];
            self.run("weights", p, (p.n + 63) / 64, &.{ self.nbr.at(3), self.d2.at(5), self.sig.at(7), self.wts.at(8), s_in.at(10) });
            const last = t == w.passes;
            p.out0 = if (last) out0 else 0;
            self.run("average", p, (@as(u64, p.n) * px + 255) / 256, &.{ src.at(1), (if (last) out else dst).at(2), self.nbr.at(3), self.wts.at(9) });
            p.out0 = 0;
            self.run("shares", p, (@as(u64, p.n) * f + 255) / 256, &.{ self.nbr.at(3), self.wts.at(9), s_in.at(10), s_out.at(11) });
            std.mem.swap(Buf, &src, &dst);
            std.mem.swap(Buf, &s_in, &s_out);
        }
        self.final = s_in;
    }
};
