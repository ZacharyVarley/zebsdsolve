//! A master pattern in a picture, for the app: a globe (an orthographic camera looking at a
//! unit sphere whose surface carries the master), the two Lambert squares, the two
//! stereographic discs, or the whole sphere in one square (the joint Lambert projection), each
//! of the view turned by a rotation the caller holds (the mouse's). The sheets of the master
//! shown are kept on the device between calls, so that a drag only renders (mastersph.wgsl,
//! sphere_view).
const std = @import("std");
const engine_mod = @import("engine.zig");
const gpu_mod = @import("gpu/gpu.zig");
const shaders = @import("shaders");

const Engine = engine_mod.Engine;
const Buf = gpu_mod.Buf;

/// What the device holds for the view: a master's sheets and what they were made from.
pub const Cache = struct {
    /// the source's identity: a hash of samples of its sheets (a new master, a new simulation
    /// or a refiltered one has other sheets; a freed one's address may come again)
    key: u64 = 0,
    /// the simulation's: a hash of its sector's intensities (the sheets are made from them)
    sim_key: u64 = 0,
    len: usize = 0,
    side: u32 = 0,
    /// the contrast: the 0.5 % and 99.5 % quantiles (as the app draws its other grey images)
    lo: f32 = 0,
    hi: f32 = 1,
    sheets: Buf = .{},
    pix: Buf = .{},

    pub fn deinit(self: *Cache, g: *gpu_mod.Gpu) void {
        g.release(&self.sheets);
        g.release(&self.pix);
        self.* = .{};
    }
};

const MsP = extern struct {
    side: u32,
    grid_x: u32,
    /// sphere_view: the picture's height
    radius: u32 = 0,
    /// sphere_view: the projection
    n_ops: u32 = 0,
    k: [4]f32,
};

pub const Projection = enum(u32) { globe = 0, lambert = 1, stereographic = 2, joint = 3 };

/// Render `width` × `height` RGBA8 pixels (one u32 each, R in the low byte) of a master:
/// `source` ≥ 0 a phase's, as patterns are compared with it (or, `stored`, as it was loaded),
/// −1 the last simulation's (of energy bin `bin` alone when ≥ 0, of site `site` alone when ≥
/// 0). `layout` is (pan across, pan down, hemispheres shown: 0 both, 1 upper, 2 lower, 1 for the
/// difference of the upper and the inverted lower; see sphere_view). `rot` turns view directions (x right, y up, z toward the viewer) into the crystal's
/// frame, row-major; `zoom` 1 fits the projection to the picture.
pub fn render(e: *Engine, source: i32, stored: bool, bin: i32, site: i32, projection: Projection, rot: [9]f32, layout: [4]f32, zoom: f32, width: u32, height: u32, out: []u32) !void {
    if (width < 8 or height < 8 or width > 4096 or height > 4096) return error.BadSize;
    if (out.len != @as(usize, width) * height) return error.BadLength;
    if (!(zoom > 0.05 and zoom < 100)) return error.BadSetting;
    const g = &e.g;
    // a slot for the phases' masters, one for the simulation's
    const c = &e.view[if (source >= 0) 0 else 1];
    if (source >= 0) {
        const p: usize = @intCast(source);
        if (p >= e.phases.items.len) return error.NoSuchPhase;
        if (stored) {
            try upload(e, c, e.phases.items[p].master.raw);
        } else {
            const lv = e.levelOf(.search);
            try e.prepareMasters(&.{lv});
            try upload(e, c, e.phases.items[p].work[@intFromEnum(lv)]);
        }
    } else {
        const s = if (e.sim) |*x| x else return error.NoSimulation;
        const b: ?usize = if (bin >= 0) @intCast(bin) else null;
        const st: ?usize = if (site >= 0) @intCast(site) else null;
        // (which part of which simulation: its sector's intensities, and the part asked for)
        const key = fingerprint(s.integrated) ^ (@as(u64, @bitCast(@as(i64, bin))) *% 0x9E3779B97F4A7C15) ^ (@as(u64, @bitCast(@as(i64, site))) *% 0xC2B2AE3D27D4EB4F);
        if (key != c.sim_key or c.side == 0) {
            var m = try s.masterOf(e.gpa, "", b, st);
            defer m.deinit();
            try upload(e, c, m.raw);
            c.sim_key = key;
        }
    }
    g.ensure(&c.pix, @as(u64, width) * height * 4);
    // (the rotation's three rows, and the layout: pan across and down in shorter sides, the
    // hemispheres shown, the difference of the upper and the inverted lower)
    var rows: [16]f32 = @splat(0);
    rows[12..16].* = layout;
    for (0..3) |i| for (0..3) |j| {
        rows[i * 4 + j] = rot[i * 3 + j];
    };
    const p: MsP = .{ .side = c.side, .grid_x = width, .radius = height, .n_ops = @intFromEnum(projection), .k = .{ zoom, c.lo, 1.0 / @max(c.hi - c.lo, 1e-20), 0 } };
    const pipe = g.pipeline("mastersph/sphere_view", shaders.mastersph, "sphere_view") catch gpu_mod.NO_PIPE;
    g.reserve(@sizeOf(MsP), @sizeOf(@TypeOf(rows)));
    g.dispatch(pipe, (width + 15) / 16, (height + 15) / 16, 1, &.{ g.uniform(0, std.mem.asBytes(&p)), c.sheets.at(1), g.stage(3, std.mem.sliceAsBytes(&rows)), c.pix.at(5) });
    g.read(c.pix, 0, std.mem.sliceAsBytes(out));
    try g.wait();
}

/// A hash of the length and of about 4096 values spread over `v`.
fn fingerprint(v: []const f32) u64 {
    var h = std.hash.Wyhash.init(v.len);
    const step = @max(1, v.len / 4096);
    var i: usize = 0;
    while (i < v.len) : (i += step) h.update(std.mem.asBytes(&v[i]));
    return h.final();
}

/// Put `sheets` (two side × side Lambert sheets) on the device for the view, with their
/// contrast, unless they are there already.
fn upload(e: *Engine, c: *Cache, sheets: []const f32) !void {
    const key = fingerprint(sheets);
    if (key == c.key and sheets.len == c.len and c.side != 0) return;
    const side: u32 = @intFromFloat(@round(@sqrt(@as(f64, @floatFromInt(sheets.len / 2)))));
    if (side < 2 or 2 * @as(usize, side) * side != sheets.len) return error.BadLength;
    e.g.ensure(&c.sheets, sheets.len * 4);
    e.g.writeSlice(c.sheets, 0, f32, sheets);
    // the 0.5 % and 99.5 % quantiles of a sample
    const step = @max(1, sheets.len / 20000);
    var sample: std.ArrayList(f32) = .empty;
    defer sample.deinit(e.gpa);
    var i: usize = 0;
    while (i < sheets.len) : (i += step) if (std.math.isFinite(sheets[i])) try sample.append(e.gpa, sheets[i]);
    var lo: f32 = 0;
    var hi: f32 = 1;
    if (sample.items.len > 0) {
        std.mem.sort(f32, sample.items, {}, std.sort.asc(f32));
        const last = sample.items.len - 1;
        lo = sample.items[last * 5 / 1000];
        hi = sample.items[last * 995 / 1000];
    }
    c.* = .{ .key = key, .sim_key = c.sim_key, .len = sheets.len, .side = side, .lo = lo, .hi = hi, .sheets = c.sheets, .pix = c.pix };
}
