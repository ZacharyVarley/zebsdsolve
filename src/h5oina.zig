//! The results as an Oxford .h5oina, made from the acquisition the scan was opened from.
//!
//! The file is new (io/hdf5_write.zig), not the acquisition edited: every group, dataset and
//! attribute of the acquisition is copied through the engine's own reader, except the patterns
//! and anything else stored per map cell with more than 64 values (spectra), which a results
//! file has no use for, and except the indexing, which is this engine's: the orientations in the
//! engine's convention (convention.zig: found from the acquisition's own orientations, or set),
//! the phase of each cell as the acquisition numbers its phases (new ones added for phases it
//! does not have), whether a cell was indexed, and the pattern centre of each cell in Oxford's
//! convention. The engine's own scores go under /zebsdsolve, with the convention used, the
//! geometry and what was left out.
//!
//! Oxford's layout, as its files have it: /<n>/EBSD/Header (Phases/<id>/...) and
//! /<n>/EBSD/Data, a value per map cell, row-major; Euler as passive Bunge angles in radians;
//! Phase 0 where a cell has no solution; Error 1 where it has one; Pattern Center X a fraction of
//! the pattern's width from its left, Pattern Center Y of its width from its bottom, Detector
//! Distance of its width.
const std = @import("std");
const engine = @import("engine.zig");
const rot = @import("rot.zig");
const hdf5 = @import("io/hdf5.zig");
const W = @import("io/hdf5_write.zig");
const convention = @import("convention.zig");
const Geometry = @import("geometry.zig").Geometry;
const scan_mod = @import("scan.zig");

const Engine = engine.Engine;
const Allocator = std.mem.Allocator;

/// The datasets of the acquisition's Data group that are its own indexing, written anew.
const replaced = [_][]const u8{ "Euler", "Phase", "Error", "Mean Angular Deviation", "Bands", "Pattern Center X", "Pattern Center Y", "Detector Distance" };

const Copy = struct {
    gpa: Allocator,
    h: *hdf5.Hdf5,
    out: *W.File,
    cells: u64,
    data_src: u64,
    header_src: u64,
    data: ?*W.Node = null,
    header: ?*W.Node = null,
    /// the acquisition's attributes of the datasets written anew, by name
    kept: std.StringHashMapUnmanaged([]hdf5.Attribute) = .empty,
    omitted: std.ArrayList(u8) = .empty,

    fn omit(self: *Copy, path: []const u8, why: []const u8) !void {
        try self.omitted.print(self.gpa, "{s}: {s}\n", .{ path, why });
    }

    fn deinit(self: *Copy) void {
        var it = self.kept.iterator();
        while (it.next()) |kv| {
            self.gpa.free(kv.key_ptr.*);
            for (kv.value_ptr.*) |*a| a.deinit(self.gpa);
            self.gpa.free(kv.value_ptr.*);
        }
        self.kept.deinit(self.gpa);
        self.omitted.deinit(self.gpa);
    }

    fn attributes(self: *Copy, src: u64, dst: *W.Node) !void {
        const attrs = self.h.attributes(src) catch return;
        defer {
            for (attrs) |*a| a.deinit(self.gpa);
            self.gpa.free(attrs);
        }
        for (attrs) |a| try self.out.attribute(dst, a.name, valueOf(a.dtype, a.dims[0..a.rank], a.data) orelse continue);
    }

    fn group(self: *Copy, src: u64, dst: *W.Node, path: []const u8) !void {
        if (src == self.data_src) self.data = dst;
        if (src == self.header_src) self.header = dst;
        try self.attributes(src, dst);
        const names = try self.h.list(src);
        defer self.h.freeList(names);
        for (names) |name| {
            const sub = try std.fmt.allocPrint(self.gpa, "{s}/{s}", .{ path, name });
            defer self.gpa.free(sub);
            const addr = (try self.h.child(src, name)) orelse continue;
            if (self.h.isGroup(addr) catch false) {
                try self.group(addr, try self.out.group(dst, name), sub);
                continue;
            }
            if (src == self.data_src and isReplaced(name)) {
                // (its attributes go on the dataset written in its place)
                const attrs = self.h.attributes(addr) catch try self.gpa.alloc(hdf5.Attribute, 0);
                try self.kept.put(self.gpa, try self.gpa.dupe(u8, name), attrs);
                continue;
            }
            self.dataset(addr, dst, name, sub) catch |err| try self.omit(sub, @errorName(err));
        }
    }

    fn dataset(self: *Copy, addr: u64, dst: *W.Node, name: []const u8, path: []const u8) !void {
        const gpa = self.gpa;
        var ds = try self.h.dataset(addr);
        defer ds.deinit(gpa);
        const rank = ds.rank;
        if (rank >= 1 and ds.dims[0] == self.cells and ds.rowElements() > 64) return self.omit(path, "a value per map cell of more than 64 values (patterns, spectra)");
        if (ds.dtype.class == .other) return self.omit(path, "a datatype other than a number or a string");
        const total = ds.elements() * ds.dtype.size;
        if (total > 256 << 20) return self.omit(path, "over 256 MB");
        const raw = try gpa.alloc(u8, @intCast(total));
        defer gpa.free(raw);
        try self.h.readAll(&ds, raw);
        const dims = ds.dims[0..rank];
        var node: *W.Node = undefined;
        if (ds.dtype.class == .vlen_string) {
            // as fixed-length strings, as long as the longest
            const n: usize = @intCast(ds.elements());
            const es: usize = ds.dtype.size;
            var strs = try gpa.alloc([]u8, n);
            var made: usize = 0;
            defer {
                for (strs[0..made]) |s| gpa.free(s);
                gpa.free(strs);
            }
            var longest: usize = 1;
            var buf: [65536]u8 = undefined;
            for (0..n) |i| {
                strs[i] = try gpa.dupe(u8, try self.h.vlenString(raw[i * es ..][0..es], &buf));
                made += 1;
                longest = @max(longest, strs[i].len);
            }
            const fixed = try gpa.alloc(u8, n * longest);
            defer gpa.free(fixed);
            @memset(fixed, 0);
            for (strs, 0..) |s, i| @memcpy(fixed[i * longest ..][0..s.len], s);
            node = try self.out.dataset(dst, name, .{ .dtype = W.Type.str(longest), .dims = dims, .bytes = fixed });
        } else {
            if (ds.dtype.big_endian and ds.dtype.class != .string) swap(raw, ds.dtype.size);
            node = try self.out.dataset(dst, name, valueOf(ds.dtype, dims, raw) orelse return self.omit(path, "an element size no writer type has"));
        }
        try self.attributes(addr, node);
    }
};

fn isReplaced(name: []const u8) bool {
    for (replaced) |r| if (std.mem.eql(u8, r, name)) return true;
    return false;
}

fn swap(raw: []u8, size: u32) void {
    if (size < 2) return;
    var i: usize = 0;
    while (i + size <= raw.len) : (i += size) std.mem.reverse(u8, raw[i..][0..size]);
}

fn valueOf(t: hdf5.Dtype, dims: []const u64, bytes: []const u8) ?W.Value {
    const class: W.Class = switch (t.class) {
        .uint => .uint,
        .int => .int,
        .float => .float,
        .string => .string,
        else => return null,
    };
    const wt: W.Type = .{ .class = class, .size = t.size };
    switch (class) {
        .uint, .int => if (t.size != 1 and t.size != 2 and t.size != 4 and t.size != 8) return null,
        .float => if (t.size != 4 and t.size != 8) return null,
        .string => if (t.size == 0) return null,
    }
    return .{ .dtype = wt, .dims = dims, .bytes = bytes };
}

fn putAttrs(out: *W.File, node: *W.Node, attrs: ?[]hdf5.Attribute) !void {
    for (attrs orelse return) |a| try out.attribute(node, a.name, valueOf(a.dtype, a.dims[0..a.rank], a.data) orelse continue);
}

/// The .h5oina's bytes (caller frees).
pub fn write(e: *Engine) ![]u8 {
    const gpa = e.gpa;
    const s = switch (e.source) {
        .file => |s| s,
        else => return error.NotAnOxfordAcquisition,
    };
    if (s.format != .h5_oxford) return error.NotAnOxfordAcquisition;
    if (e.phases.items.len == 0) return error.NoPhases;
    for (e.phases.items) |p| if (p.master.lattice == null) return error.MissingPhaseLattice;
    const info = s.info;
    const n: usize = @as(usize, info.rows) * info.cols;
    if (e.res.phase.len != n) return error.NoResults;
    // the convention: found from the acquisition's own orientations unless one is set (what the
    // file's angles mean is the acquisition's, and nothing else says it)
    if (e.convention.isEngine()) {
        const got = try e.identifyFrame(null);
        if (!got.installed) return error.FrameNotFound;
    }
    const frame = e.convention;
    var stored = (try s.storedOrientations(gpa)) orelse return error.NoStoredOrientations;
    defer stored.deinit(gpa);
    const matched = try convention.matchLabels(gpa, e.res.phase, stored.phase, e.phases.items.len);

    var out = try W.File.init(gpa);
    defer out.deinit();
    const h = &s.h5.?;
    var copy: Copy = .{ .gpa = gpa, .h = h, .out = &out, .cells = n, .data_src = s.h5_data.?, .header_src = s.h5_header.? };
    defer copy.deinit();
    try copy.group(h.root, out.root, "");
    const data = copy.data orelse return error.NoScanInFile;
    const header = copy.header orelse return error.NoScanInFile;

    // the phases: the acquisition's as it numbers them, and the engine's it does not have
    const phases_node = try out.group(header, "Phases");
    var next: i32 = 1;
    for (phases_node.links.items) |l| next = @max(next, (std.fmt.parseInt(i32, l.name, 10) catch 0) + 1);
    var ids: [convention.max_phases]i32 = @splat(0);
    for (e.phases.items, 0..) |*p, i| {
        if (i >= convention.max_phases) break;
        if (matched[i] > 0) {
            ids[i] = matched[i];
            continue;
        }
        ids[i] = next;
        next += 1;
        var nb: [16]u8 = undefined;
        const g = try out.group(phases_node, try std.fmt.bufPrint(&nb, "{d}", .{ids[i]}));
        const m = &p.master;
        _ = try out.text(g, "Phase Name", if (m.name.len != 0) m.name else "phase");
        _ = try out.text(g, "Reference", "zebsdsolve master pattern");
        const one = [_]u64{1};
        const sg: i32 = @intCast(m.space_group);
        _ = try out.dataset(g, "Space Group", .{ .dtype = W.Type.i32_, .dims = &one, .bytes = std.mem.asBytes(&sg) });
        const laue: i32 = @intFromEnum(m.laue);
        const lg = try out.dataset(g, "Laue Group", .{ .dtype = W.Type.i32_, .dims = &one, .bytes = std.mem.asBytes(&laue) });
        const lname = m.laue.name();
        try out.attribute(lg, "Symbol", .{ .dtype = W.Type.str(lname.len), .bytes = lname });
        const l = m.lattice.?;
        const dims13 = [_]u64{ 1, 3 };
        const lengths = [_]f32{ @floatCast(l[0]), @floatCast(l[1]), @floatCast(l[2]) };
        const angles = [_]f32{ @floatCast(std.math.degreesToRadians(l[3])), @floatCast(std.math.degreesToRadians(l[4])), @floatCast(std.math.degreesToRadians(l[5])) };
        const ld = try out.dataset(g, "Lattice Dimensions", .{ .dtype = W.Type.f32_, .dims = &dims13, .bytes = std.mem.sliceAsBytes(&lengths) });
        try out.attribute(ld, "Unit", .{ .dtype = W.Type.str(8), .bytes = "angstrom" });
        const la = try out.dataset(g, "Lattice Angles", .{ .dtype = W.Type.f32_, .dims = &dims13, .bytes = std.mem.sliceAsBytes(&angles) });
        try out.attribute(la, "Unit", .{ .dtype = W.Type.str(3), .bytes = "rad" });
    }
    if (next > 256) return error.TooManyPhases;

    // the indexing, the engine's
    const crystal = try gpa.alloc(rot.Quat, e.phases.items.len);
    defer gpa.free(crystal);
    for (e.phases.items, crystal) |*p, *c| c.* = frame.crystalOf(p.master.lattice) orelse return error.MissingPhaseLattice;
    const euler = try gpa.alloc(f32, 3 * n);
    defer gpa.free(euler);
    const phase = try gpa.alloc(u8, n);
    defer gpa.free(phase);
    const err = try gpa.alloc(u8, n);
    defer gpa.free(err);
    var indexed: usize = 0;
    for (0..n) |i| {
        const ph = e.res.phase[i];
        if (ph < 0 or ph >= e.phases.items.len or ph >= convention.max_phases) {
            @memset(euler[3 * i ..][0..3], 0);
            phase[i] = 0;
            err[i] = 0;
            continue;
        }
        const q = e.res.quat[4 * i ..][0..4];
        const b = rot.toBunge(frame.toFrame(.{ q[0], q[1], q[2], q[3] }, crystal[@intCast(ph)]));
        for (0..3) |k| euler[3 * i + k] = @floatCast(b[k]);
        phase[i] = @intCast(ids[@intCast(ph)]);
        err[i] = 1;
        indexed += 1;
    }
    const cells = [_]u64{n};
    const cells3 = [_]u64{ n, 3 };
    try putAttrs(&out, try out.dataset(data, "Euler", .{ .dtype = W.Type.f32_, .dims = &cells3, .bytes = std.mem.sliceAsBytes(euler) }), copy.kept.get("Euler"));
    try putAttrs(&out, try out.dataset(data, "Phase", .{ .dtype = W.Type.u8_, .dims = &cells, .bytes = phase }), copy.kept.get("Phase"));
    try putAttrs(&out, try out.dataset(data, "Error", .{ .dtype = W.Type.u8_, .dims = &cells, .bytes = err }), copy.kept.get("Error"));
    // each cell's pattern centre as the geometry has the beam there, in Oxford's convention
    {
        const geo = e.geo;
        const R = geo.rotation();
        const sz = try e.sizeOf(.full);
        const ratio = sz[0] / sz[1];
        const pcs = try gpa.alloc(f32, 3 * n);
        defer gpa.free(pcs);
        for (0..n) |i| {
            const pc = Geometry.pcOf(R, try e.placeOf(&geo, i, .full), sz[0], sz[1]);
            pcs[i] = @floatCast(pc[0]);
            pcs[n + i] = @floatCast((1 - pc[1]) * ratio);
            pcs[2 * n + i] = @floatCast(pc[2] * ratio);
        }
        for ([_][]const u8{ "Pattern Center X", "Pattern Center Y", "Detector Distance" }, 0..) |name, k| {
            try putAttrs(&out, try out.dataset(data, name, .{ .dtype = W.Type.f32_, .dims = &cells, .bytes = std.mem.sliceAsBytes(pcs[k * n ..][0..n]) }), copy.kept.get(name));
        }
    }
    // the detector's orientation the orientations were found under, where the acquisition
    // stores one
    for (header.links.items) |*l| if (std.mem.eql(u8, l.name, "Detector Orientation Euler")) if (l.node.value) |*v| {
        if (v.dtype.class != .float or v.bytes.len != 3 * @as(usize, v.dtype.size)) continue;
        const eu = scan_mod.oxfordDetectorEuler(e.geo.rot);
        const bytes = try out.arena.allocator().alloc(u8, v.bytes.len);
        for (0..3) |k| {
            if (v.dtype.size == 4) std.mem.writeInt(u32, bytes[4 * k ..][0..4], @bitCast(@as(f32, @floatCast(eu[k]))), .little) else std.mem.writeInt(u64, bytes[8 * k ..][0..8], @bitCast(eu[k]), .little);
        }
        v.bytes = bytes;
    };
    // the share indexed, where the acquisition says its own
    for (header.links.items) |*l| if (std.mem.eql(u8, l.name, "Hit Rate")) if (l.node.value) |*v| {
        const rate = 100 * @as(f64, @floatFromInt(indexed)) / @as(f64, @floatFromInt(@max(n, 1)));
        const bytes = try out.arena.allocator().alloc(u8, v.bytes.len);
        @memset(bytes, 0);
        switch (v.dtype.class) {
            .float => if (v.dtype.size == 4) std.mem.writeInt(u32, bytes[0..4], @bitCast(@as(f32, @floatCast(rate))), .little) else std.mem.writeInt(u64, bytes[0..8], @bitCast(rate), .little),
            else => {},
        }
        if (v.dtype.class == .float) v.bytes = bytes;
    };

    // the engine's own
    const z = try out.group(out.root, "zebsdsolve");
    _ = try out.text(z, "Version", engine.version);
    var line: [256]u8 = undefined;
    var lw = std.Io.Writer.fixed(&line);
    try frame.describe(&lw);
    _ = try out.text(z, "Convention", lw.buffered());
    _ = try out.dataset(z, "NCC", .{ .dtype = W.Type.f32_, .dims = &cells, .bytes = std.mem.sliceAsBytes(e.res.ncc) });
    _ = try out.dataset(z, "NMI", .{ .dtype = W.Type.f32_, .dims = &cells, .bytes = std.mem.sliceAsBytes(e.res.nmi) });
    _ = try out.dataset(z, "Z", .{ .dtype = W.Type.f32_, .dims = &cells, .bytes = std.mem.sliceAsBytes(e.res.z) });
    _ = try out.dataset(z, "Sweep", .{ .dtype = W.Type.i32_, .dims = &cells, .bytes = std.mem.sliceAsBytes(e.res.sweep) });
    const pc = [_]f64{ e.geo.pc[0], e.geo.pc[1], e.geo.pc[2] };
    const three = [_]u64{3};
    _ = try out.dataset(z, "Pattern Centre", .{ .dtype = W.Type.f64_, .dims = &three, .bytes = std.mem.sliceAsBytes(&pc) });
    if (e.geo.plane_kind != 0) {
        const four3 = [_]u64{ 4, 3 };
        _ = try out.dataset(z, "Plane", .{ .dtype = W.Type.f64_, .dims = &four3, .bytes = std.mem.sliceAsBytes(&e.geo.plane) });
    }
    _ = try out.text(z, "Left Out", copy.omitted.items);
    return out.bytes(gpa);
}
