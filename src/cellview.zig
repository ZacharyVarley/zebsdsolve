//! The crystal to simulate, as its unit cell in a picture, for the app: every atom of the cell
//! (each site's orbit, with its images on the cell's faces, edges and corners) as a ball in its
//! element's colour, and the cell's edges as rods, seen by an orthographic camera turned by a
//! rotation the caller holds (the mouse's). The atoms are kept on the device between calls, so
//! that a drag only renders (cellview.wgsl).
const std = @import("std");
const engine_mod = @import("engine.zig");
const gpu_mod = @import("gpu/gpu.zig");
const shaders = @import("shaders");
const cellframe = @import("sim/cellframe.zig");

const Engine = engine_mod.Engine;
const Buf = gpu_mod.Buf;

/// What the device holds for the view: the atoms of a crystal and which crystal they are of.
pub const Cache = struct {
    /// the crystal's identity (cellframe.fingerprint)
    key: u64 = 0,
    n: u32 = 0,
    atoms: Buf = .{},
    pix: Buf = .{},

    pub fn deinit(self: *Cache, g: *gpu_mod.Gpu) void {
        g.release(&self.atoms);
        g.release(&self.pix);
        self.* = .{};
    }
};

const CvP = extern struct {
    width: u32,
    height: u32,
    n_atoms: u32,
    highlight: u32,
    rot: [12]f32,
    basis: [12]f32,
    origin: [4]f32,
    k: [4]f32,
};

/// Render `width` × `height` RGBA8 pixels (one u32 each, R in the low byte) of the crystal's
/// cell. `rot` turns view directions (x right, y up, z toward the viewer) into the crystal's
/// Cartesian frame, row-major; `zoom` 1 fits the cell to the shorter side; `highlight` what to
/// pick out (cellview.wgsl: 1–3 an edge a, b, c; 4–6 an angle α, β, γ; 16 + k site k); `cells`
/// the cells along each lattice vector (1, or 3 for a 3 × 3 × 3 supercell).
pub fn render(e: *Engine, rot: [9]f32, zoom: f32, highlight: u32, cells: u32, width: u32, height: u32, out: []u32) !void {
    if (cells < 1 or cells > 5) return error.BadSetting;
    if (width < 8 or height < 8 or width > 4096 or height > 4096) return error.BadSize;
    if (out.len != @as(usize, width) * height) return error.BadLength;
    if (!(zoom > 0.05 and zoom < 100)) return error.BadSetting;
    const cell = if (e.crystal) |*c| c else return error.NoCrystal;
    const g = &e.g;
    const c = &e.cell_view;
    const fr = cellframe.frameOf(cell, cells);
    const key = cellframe.fingerprint(cell, cells);
    if (key != c.key or c.n == 0) {
        const atoms = try cellframe.atomsOf(e.gpa, cell, &fr);
        defer e.gpa.free(atoms);
        g.ensure(&c.atoms, @max(atoms.len, 1) * @sizeOf(cellframe.Atom));
        g.writeSlice(c.atoms, 0, cellframe.Atom, atoms);
        c.key = key;
        c.n = @intCast(atoms.len);
    }
    g.ensure(&c.pix, @as(u64, width) * height * 4);
    const radius: f32 = @floatCast(fr.radius);
    const scale = 0.5 * @as(f32, @floatFromInt(@min(width, height))) * zoom / radius;
    var p: CvP = .{
        .width = width,
        .height = height,
        .n_atoms = c.n,
        .highlight = highlight,
        .rot = @splat(0),
        .basis = @splat(0),
        // (the cell's origin from the middle; and, in .w, the cells along each vector)
        .origin = .{ @floatCast(-fr.centre[0]), @floatCast(-fr.centre[1]), @floatCast(-fr.centre[2]), @floatFromInt(cells) },
        // rods about 2 % of the picture's radius and never under a pixel; rays from 2 radii out
        .k = .{ scale, @max(0.02 * radius, 1.0 / scale), 1.0 / (2 * radius), 2 * radius },
    };
    for (0..3) |i| for (0..3) |j| {
        p.rot[i * 4 + j] = rot[i * 3 + j];
        p.basis[i * 4 + j] = @floatCast(fr.basis[i][j]);
    };
    const pipe = g.pipeline("cellview/cell_view", shaders.cellview, "cell_view") catch gpu_mod.NO_PIPE;
    g.dispatch(pipe, (width + 15) / 16, (height + 15) / 16, 1, &.{ g.uniform(0, std.mem.asBytes(&p)), c.atoms.at(1), c.pix.at(2) });
    g.read(c.pix, 0, std.mem.sliceAsBytes(out));
    try g.wait();
}
