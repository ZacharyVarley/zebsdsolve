//! A reader for the part of HDF5 that EBSD files use, written against the file format
//! specification so the engine needs no HDF5 library (and so the browser reads a 100 GB file
//! in place). It follows groups, reads dataset headers and reads datasets whole, by ranges of
//! their first axis, or by a list of steps along it (the patterns of a preview lattice). A
//! chunked dataset's index is read one tree level per round trip, and the chunks a request
//! needs in one more.
//!
//! Covered: superblock versions 0–3; object headers versions 1 and 2; groups stored as symbol
//! tables (the default of every HDF5 writer) or as compact link messages; fixed-point, floating
//! point, fixed and variable-length string types; compact, contiguous and chunked layouts
//! (version 1 B-tree; version 4 single-chunk and implicit indexes); the deflate, shuffle,
//! Fletcher-32 and LZF filters.
//! Not covered, each reported by name: dense groups (fractal heaps), the fixed and extensible
//! array and version 2 B-tree chunk indexes of files written with libver="latest", external
//! links and other filters.
const std = @import("std");
const file_mod = @import("file.zig");
const File = file_mod.File;
const Range = file_mod.Range;
const codec = @import("codec.zig");

const Allocator = std.mem.Allocator;
const UNDEF: u64 = std.math.maxInt(u64);
pub const MAX_RANK = 8;

pub const Class = enum { uint, int, float, string, vlen_string, other };

pub const Dtype = struct {
    class: Class = .other,
    size: u32 = 0,
    big_endian: bool = false,
};

const Filter = struct { id: u16, elem: u32 = 0 };

const Layout = union(enum) {
    /// data stored in the object header: file offset and size
    compact: struct { offset: u64, size: u64 },
    contiguous: struct { addr: u64, size: u64 },
    /// version 1 B-tree of chunks
    chunked: struct { btree: u64 },
    /// one chunk holding the whole dataset
    single: struct { addr: u64, size: u64, mask: u32 },
    /// unfiltered chunks stored back to back in chunk order
    implicit: struct { addr: u64 },
};

/// Where a chunk is stored: its address (UNDEF: never written), stored size and filter mask.
const Slot = struct {
    addr: u64 = UNDEF,
    size: u32 = 0,
    mask: u32 = 0,
};

/// Chunks read together are bounded by this many stored bytes.
const SPAN_BYTES = 64 << 20;

pub const Dataset = struct {
    rank: u8 = 0,
    dims: [MAX_RANK]u64 = @splat(1),
    dtype: Dtype = .{},
    layout: Layout = .{ .contiguous = .{ .addr = UNDEF, .size = 0 } },
    /// chunk shape (chunked layouts)
    chunk: [MAX_RANK]u64 = @splat(1),
    filters: [4]Filter = undefined,
    n_filters: u8 = 0,
    /// Chunk slots can be populated by targeted reads before the full tree is ready.
    index: []Slot = &.{},
    known: []bool = &.{},
    index_ready: bool = false,
    index_failed: bool = false,
    frontier: std.ArrayList(u64) = .empty,
    frontier_at: usize = 0,
    warmed_nodes: u64 = 0,
    nodes_seen: std.AutoHashMapUnmanaged(u64, void) = .empty,
    node_len: usize = 0,
    /// chunks along each axis
    grid: [MAX_RANK]u64 = @splat(1),
    /// the last chunk decoded (single-chunk layout)
    cache_addr: u64 = UNDEF,
    cache: []u8 = &.{},

    pub fn deinit(self: *Dataset, gpa: Allocator) void {
        gpa.free(self.index);
        gpa.free(self.known);
        self.frontier.deinit(gpa);
        self.nodes_seen.deinit(gpa);
        gpa.free(self.cache);
        self.* = .{};
    }

    pub fn elements(self: *const Dataset) u64 {
        var n: u64 = 1;
        for (self.dims[0..self.rank]) |d| n *= d;
        return n;
    }

    /// Elements in one step of the first axis.
    pub fn rowElements(self: *const Dataset) u64 {
        var n: u64 = 1;
        for (self.dims[1..@max(self.rank, 1)]) |d| n *= d;
        return n;
    }

    fn chunkElements(self: *const Dataset) u64 {
        var n: u64 = 1;
        for (self.chunk[0..self.rank]) |d| n *= d;
        return n;
    }
};

pub const Hdf5 = struct {
    gpa: Allocator,
    f: *File,
    /// sizes of file offsets and of lengths, in bytes
    so: u8 = 8,
    sl: u8 = 8,
    base: u64 = 0,
    root: u64 = 0,
    /// why the last call failed, when the format is the reason
    why: []const u8 = "",

    pub fn open(gpa: Allocator, f: *File) !Hdf5 {
        var self: Hdf5 = .{ .gpa = gpa, .f = f };
        // the superblock is at 0 or at a power of two ≥ 512 (after a user block)
        var at: u64 = 0;
        const sig = "\x89HDF\r\n\x1a\n";
        while (true) {
            var b: [8]u8 = undefined;
            f.read(at, &b) catch return error.NotHdf5;
            if (std.mem.eql(u8, &b, sig)) break;
            at = if (at == 0) 512 else at * 2;
            if (at >= f.size) return error.NotHdf5;
        }
        const version = try f.int(u8, at + 8);
        if (version <= 1) {
            self.so = try f.int(u8, at + 13);
            self.sl = try f.int(u8, at + 14);
            var p: u64 = at + 24;
            if (version == 1) p += 4;
            self.base = try self.offset(p);
            // base, free-space info, end of file, driver info, then the root symbol table entry
            p += 4 * @as(u64, self.so);
            self.root = try self.offset(p + self.so);
        } else if (version <= 3) {
            self.so = try f.int(u8, at + 9);
            self.sl = try f.int(u8, at + 10);
            self.base = try self.offset(at + 12);
            self.root = try self.offset(at + 12 + 3 * @as(u64, self.so));
        } else return self.unsupported("superblock version");
        if (self.so != 8 and self.so != 4) return self.unsupported("offset size");
        return self;
    }

    fn unsupported(self: *Hdf5, what: []const u8) error{UnsupportedHdf5} {
        self.why = what;
        return error.UnsupportedHdf5;
    }

    fn offset(self: *Hdf5, at: u64) !u64 {
        const v = try self.f.uintN(at, self.so);
        if (self.so == 4 and v == 0xffffffff) return UNDEF;
        return v;
    }

    fn length(self: *Hdf5, at: u64) !u64 {
        return self.f.uintN(at, self.sl);
    }

    // ── object headers ──

    const Message = struct { type: u16, at: u64, size: u32 };

    /// Every message of the object header at `addr` (caller frees).
    fn messages(self: *Hdf5, addr: u64) ![]Message {
        var out: std.ArrayList(Message) = .empty;
        errdefer out.deinit(self.gpa);
        const a = addr + self.base;
        var sig: [4]u8 = undefined;
        try self.f.read(a, &sig);
        if (std.mem.eql(u8, &sig, "OHDR")) {
            const flags = try self.f.int(u8, a + 5);
            var p: u64 = a + 6;
            if (flags & 0x20 != 0) p += 16;
            if (flags & 0x10 != 0) p += 4;
            const size_bytes: u8 = @as(u8, 1) << @intCast(flags & 3);
            const chunk0 = try self.f.uintN(p, size_bytes);
            p += size_bytes;
            try self.messagesV2(&out, p, chunk0, flags & 0x04 != 0);
        } else {
            if (sig[0] != 1) return self.unsupported("object header version");
            const n = try self.f.int(u16, a + 2);
            const size = try self.f.int(u32, a + 8);
            try self.messagesV1(&out, a + 16, size, n);
        }
        return out.toOwnedSlice(self.gpa);
    }

    fn messagesV1(self: *Hdf5, out: *std.ArrayList(Message), start: u64, size: u64, limit: u32) !void {
        var p = start;
        const end = start + size;
        while (p + 8 <= end and out.items.len < limit + 64) {
            const t = try self.f.int(u16, p);
            const sz = try self.f.int(u16, p + 2);
            const body = p + 8;
            if (t == 0x10) {
                const c_off = try self.offset(body);
                const c_len = try self.length(body + self.so);
                try self.messagesV1(out, c_off + self.base, c_len, limit);
            } else if (t != 0) {
                try out.append(self.gpa, .{ .type = t, .at = body, .size = sz });
            }
            p = body + sz;
        }
    }

    fn messagesV2(self: *Hdf5, out: *std.ArrayList(Message), start: u64, size: u64, order: bool) !void {
        var p = start;
        const end = start + size;
        const head: u64 = if (order) 6 else 4;
        while (p + head <= end) {
            const t = try self.f.int(u8, p);
            const sz = try self.f.int(u16, p + 1);
            const body = p + head;
            if (t == 0x10) {
                const c_off = try self.offset(body);
                const c_len = try self.length(body + self.so);
                // a continuation block: "OCHK", messages, checksum
                try self.messagesV2(out, c_off + self.base + 4, c_len - 8, order);
            } else if (t != 0) {
                try out.append(self.gpa, .{ .type = t, .at = body, .size = sz });
            }
            p = body + sz;
        }
    }

    // ── groups ──

    /// Address of the object named `name` in the group at `group`, or null.
    pub fn child(self: *Hdf5, group: u64, name: []const u8) !?u64 {
        var found: ?u64 = null;
        const Ctx = struct {
            want: []const u8,
            found: *?u64,
            fn visit(ctx: *@This(), nm: []const u8, addr: u64) bool {
                if (!std.mem.eql(u8, nm, ctx.want)) return true;
                ctx.found.* = addr;
                return false;
            }
        };
        var ctx: Ctx = .{ .want = name, .found = &found };
        try self.walk(group, &ctx, Ctx.visit);
        return found;
    }

    /// Names of the group's members (caller frees each and the list).
    pub fn list(self: *Hdf5, group: u64) ![][]u8 {
        var names: std.ArrayList([]u8) = .empty;
        errdefer {
            for (names.items) |n| self.gpa.free(n);
            names.deinit(self.gpa);
        }
        const Ctx = struct {
            gpa: Allocator,
            names: *std.ArrayList([]u8),
            failed: bool = false,
            fn visit(ctx: *@This(), nm: []const u8, addr: u64) bool {
                _ = addr;
                const d = ctx.gpa.dupe(u8, nm) catch {
                    ctx.failed = true;
                    return false;
                };
                ctx.names.append(ctx.gpa, d) catch {
                    ctx.gpa.free(d);
                    ctx.failed = true;
                    return false;
                };
                return true;
            }
        };
        var ctx: Ctx = .{ .gpa = self.gpa, .names = &names };
        try self.walk(group, &ctx, Ctx.visit);
        if (ctx.failed) return error.OutOfMemory;
        return names.toOwnedSlice(self.gpa);
    }

    pub fn freeList(self: *Hdf5, names: [][]u8) void {
        for (names) |n| self.gpa.free(n);
        self.gpa.free(names);
    }

    /// Resolve a path of names separated by '/', from the root.
    pub fn find(self: *Hdf5, path: []const u8) !?u64 {
        var at = self.root;
        var it = std.mem.tokenizeScalar(u8, path, '/');
        while (it.next()) |part| at = (try self.child(at, part)) orelse return null;
        return at;
    }

    /// Whether the object at `addr` is a group.
    pub fn isGroup(self: *Hdf5, addr: u64) !bool {
        const msgs = try self.messages(addr);
        defer self.gpa.free(msgs);
        for (msgs) |m| if (m.type == 0x11 or m.type == 0x02 or m.type == 0x06) return true;
        return false;
    }

    fn walk(self: *Hdf5, group: u64, ctx: anytype, comptime visit: anytype) !void {
        const msgs = try self.messages(group);
        defer self.gpa.free(msgs);
        for (msgs) |m| switch (m.type) {
            0x11 => {
                const btree = try self.offset(m.at);
                const heap = try self.offset(m.at + self.so);
                // local heap: "HEAP", version, reserved(3), data size, free list head, data address
                const heap_data = try self.offset(heap + self.base + 8 + 2 * @as(u64, self.sl));
                if (!try self.walkTree(btree, heap_data + self.base, ctx, visit)) return;
            },
            0x06 => {
                // link message
                const flags = try self.f.int(u8, m.at + 1);
                var p = m.at + 2;
                var link_type: u8 = 0;
                if (flags & 0x08 != 0) {
                    link_type = try self.f.int(u8, p);
                    p += 1;
                }
                if (flags & 0x04 != 0) p += 8;
                if (flags & 0x10 != 0) p += 1;
                const nb: u8 = @as(u8, 1) << @intCast(flags & 3);
                const nlen = try self.f.uintN(p, nb);
                p += nb;
                if (link_type != 0 or nlen > 1024) continue;
                var name: [1024]u8 = undefined;
                try self.f.read(p, name[0..@intCast(nlen)]);
                const target = try self.offset(p + nlen);
                if (!visit(ctx, name[0..@intCast(nlen)], target)) return;
            },
            0x02 => {
                // link info: dense storage when the fractal heap address is defined
                const flags = try self.f.int(u8, m.at + 1);
                const p = m.at + 2 + (if (flags & 1 != 0) @as(u64, 8) else 0);
                if (try self.offset(p) != UNDEF) return self.unsupported("dense group (written with libver=\"latest\")");
            },
            else => {},
        };
    }

    fn walkTree(self: *Hdf5, node: u64, heap_data: u64, ctx: anytype, comptime visit: anytype) !bool {
        const a = node + self.base;
        var sig: [4]u8 = undefined;
        try self.f.read(a, &sig);
        if (!std.mem.eql(u8, &sig, "TREE")) return error.BadHdf5;
        const level = try self.f.int(u8, a + 5);
        const used = try self.f.int(u16, a + 6);
        var p = a + 8 + 2 * @as(u64, self.so);
        var i: u32 = 0;
        while (i < used) : (i += 1) {
            // key (a heap offset), then the child
            const kid = try self.offset(p + self.sl);
            p += @as(u64, self.sl) + self.so;
            if (level > 0) {
                if (!try self.walkTree(kid, heap_data, ctx, visit)) return false;
                continue;
            }
            const s = kid + self.base;
            try self.f.read(s, &sig);
            if (!std.mem.eql(u8, &sig, "SNOD")) return error.BadHdf5;
            const n = try self.f.int(u16, s + 6);
            var e = s + 8;
            var k: u32 = 0;
            while (k < n) : (k += 1) {
                const name_off = try self.offset(e);
                const obj = try self.offset(e + self.so);
                e += 2 * @as(u64, self.so) + 24;
                var name: [1024]u8 = undefined;
                const avail: usize = @intCast(@min(name.len, self.f.size - (heap_data + name_off)));
                try self.f.read(heap_data + name_off, name[0..avail]);
                const len = std.mem.indexOfScalar(u8, name[0..avail], 0) orelse avail;
                if (!visit(ctx, name[0..len], obj)) return false;
            }
        }
        return true;
    }

    // ── datasets ──

    /// The header of the dataset at `addr`.
    pub fn dataset(self: *Hdf5, addr: u64) !Dataset {
        const msgs = try self.messages(addr);
        defer self.gpa.free(msgs);
        var ds: Dataset = .{};
        var have_layout = false;
        for (msgs) |m| switch (m.type) {
            0x01 => {
                const version = try self.f.int(u8, m.at);
                ds.rank = try self.f.int(u8, m.at + 1);
                if (ds.rank > MAX_RANK) return self.unsupported("dataset rank");
                const p = m.at + (if (version == 1) @as(u64, 8) else 4);
                for (0..ds.rank) |i| ds.dims[i] = try self.length(p + i * self.sl);
            },
            0x03 => ds.dtype = try self.datatype(m.at),
            0x08 => {
                have_layout = true;
                try self.layoutMessage(&ds, m.at);
            },
            0x0b => {
                const version = try self.f.int(u8, m.at);
                const n = try self.f.int(u8, m.at + 1);
                if (n > ds.filters.len) return self.unsupported("filter pipeline length");
                var p = m.at + (if (version == 1) @as(u64, 8) else 2);
                for (0..n) |i| {
                    const id = try self.f.int(u16, p);
                    var name_len: u16 = 0;
                    if (version == 1 or id >= 256) {
                        name_len = try self.f.int(u16, p + 2);
                        p += 2;
                    }
                    const n_cd = try self.f.int(u16, p + 4);
                    p += 6;
                    p += if (version == 1) std.mem.alignForward(u64, name_len, 8) else name_len;
                    var fl: Filter = .{ .id = id };
                    if (id == 2 and n_cd >= 1) fl.elem = try self.f.int(u32, p);
                    p += 4 * @as(u64, n_cd);
                    if (version == 1 and n_cd % 2 == 1) p += 4;
                    ds.filters[i] = fl;
                    switch (id) {
                        1, 2, 3, 32000 => {},
                        else => return self.unsupported("compression filter"),
                    }
                }
                ds.n_filters = n;
            },
            else => {},
        };
        if (!have_layout) return error.NotADataset;
        return ds;
    }

    fn datatype(self: *Hdf5, at: u64) !Dtype {
        const b0 = try self.f.int(u8, at);
        const bits = try self.f.int(u8, at + 1);
        const size = try self.f.int(u32, at + 4);
        return switch (b0 & 0x0f) {
            0 => .{ .class = if (bits & 0x08 != 0) .int else .uint, .size = size, .big_endian = bits & 1 != 0 },
            1 => .{ .class = .float, .size = size, .big_endian = bits & 1 != 0 },
            3 => .{ .class = .string, .size = size },
            // enumerations (h5py booleans) read as their base integer
            8 => blk: {
                var base = try self.datatype(at + 8);
                base.size = size;
                break :blk base;
            },
            9 => .{ .class = if (bits & 0x0f == 1) .vlen_string else .other, .size = size },
            else => .{ .class = .other, .size = size },
        };
    }

    fn layoutMessage(self: *Hdf5, ds: *Dataset, at: u64) !void {
        const version = try self.f.int(u8, at);
        if (version == 3) {
            switch (try self.f.int(u8, at + 1)) {
                0 => ds.layout = .{ .compact = .{ .offset = at + 4, .size = try self.f.int(u16, at + 2) } },
                1 => ds.layout = .{ .contiguous = .{ .addr = try self.offset(at + 2), .size = try self.length(at + 2 + self.so) } },
                2 => {
                    const nd = try self.f.int(u8, at + 2);
                    if (nd == 0 or nd - 1 > MAX_RANK) return self.unsupported("chunk rank");
                    ds.layout = .{ .chunked = .{ .btree = try self.offset(at + 3) } };
                    for (0..nd - 1) |i| ds.chunk[i] = try self.f.int(u32, at + 3 + self.so + 4 * i);
                },
                else => return self.unsupported("layout class"),
            }
        } else if (version == 4) {
            switch (try self.f.int(u8, at + 1)) {
                0 => ds.layout = .{ .compact = .{ .offset = at + 4, .size = try self.f.int(u16, at + 2) } },
                1 => ds.layout = .{ .contiguous = .{ .addr = try self.offset(at + 2), .size = try self.length(at + 2 + self.so) } },
                2 => {
                    const flags = try self.f.int(u8, at + 2);
                    const nd = try self.f.int(u8, at + 3);
                    const enc = try self.f.int(u8, at + 4);
                    if (nd == 0 or nd - 1 > MAX_RANK) return self.unsupported("chunk rank");
                    var p = at + 5;
                    for (0..nd) |i| {
                        const v = try self.f.uintN(p, enc);
                        if (i < nd - 1) ds.chunk[i] = v;
                        p += enc;
                    }
                    switch (try self.f.int(u8, p)) {
                        1 => {
                            p += 1;
                            var size: u64 = 0;
                            var mask: u32 = 0;
                            if (flags & 0x02 != 0) {
                                size = try self.length(p);
                                mask = try self.f.int(u32, p + self.sl);
                                p += @as(u64, self.sl) + 4;
                            }
                            ds.layout = .{ .single = .{ .addr = try self.offset(p), .size = size, .mask = mask } };
                        },
                        2 => ds.layout = .{ .implicit = .{ .addr = try self.offset(p + 1) } },
                        3 => return self.unsupported("fixed-array chunk index (written with libver=\"latest\")"),
                        4 => return self.unsupported("extensible-array chunk index (written with libver=\"latest\")"),
                        5 => return self.unsupported("version 2 B-tree chunk index (written with libver=\"latest\")"),
                        else => return self.unsupported("chunk index"),
                    }
                },
                else => return self.unsupported("layout class"),
            }
        } else if (version == 1 or version == 2) {
            const nd = try self.f.int(u8, at + 1);
            const class = try self.f.int(u8, at + 2);
            var p = at + 8;
            var addr: u64 = UNDEF;
            if (class != 0) {
                addr = try self.offset(p);
                p += self.so;
            }
            switch (class) {
                1 => {
                    var size: u64 = 1;
                    for (0..nd) |i| size *= try self.f.int(u32, p + 4 * i);
                    ds.layout = .{ .contiguous = .{ .addr = addr, .size = size } };
                },
                2 => {
                    if (nd == 0 or nd - 1 > MAX_RANK) return self.unsupported("chunk rank");
                    ds.layout = .{ .chunked = .{ .btree = addr } };
                    for (0..nd - 1) |i| ds.chunk[i] = try self.f.int(u32, p + 4 * i);
                },
                else => {
                    const size = try self.f.int(u32, p + 4 * @as(u64, nd));
                    ds.layout = .{ .compact = .{ .offset = p + 4 * @as(u64, nd) + 4, .size = size } };
                },
            }
        } else return self.unsupported("layout version");
    }

    /// Read steps [first, first + count) of the dataset's first axis (every other axis whole)
    /// into `dst`, as stored bytes (`count · rowElements · dtype.size`).
    pub fn readRows(self: *Hdf5, ds: *Dataset, first: u64, count: u64, dst: []u8) !void {
        const es: u64 = ds.dtype.size;
        const row = ds.rowElements() * es;
        if (ds.rank == 0) {
            if (first != 0 or count != 1) return error.OutOfRange;
        } else if (first + count > ds.dims[0]) return error.OutOfRange;
        if (dst.len != count * row) return error.BadLength;
        switch (ds.layout) {
            .compact => |c| try self.f.read(c.offset + first * row, dst),
            .contiguous => |c| {
                if (c.addr == UNDEF) {
                    @memset(dst, 0);
                    return;
                }
                try self.f.read(c.addr + self.base + first * row, dst);
            },
            .single => |c| {
                const whole = try self.chunkBytes(ds, c.addr, if (c.size == 0) ds.elements() * es else c.size, c.mask, ds.elements() * es);
                @memcpy(dst, whole[@intCast(first * row)..][0..dst.len]);
            },
            .chunked, .implicit => {
                const rows = try self.gpa.alloc(u64, @intCast(count));
                defer self.gpa.free(rows);
                for (rows, 0..) |*r, i| r.* = first + i;
                try self.readChunked(ds, rows, dst);
            },
        }
    }

    /// Read the given steps of the first axis (ascending, each once) into `dst` back to back:
    /// what `readRows` would give for each, in as few round trips as the layout allows.
    pub fn readRowList(self: *Hdf5, ds: *Dataset, rows: []const u64, dst: []u8) !void {
        const es: u64 = ds.dtype.size;
        const row: usize = @intCast(ds.rowElements() * es);
        if (ds.rank == 0) return error.OutOfRange;
        if (dst.len != rows.len * row) return error.BadLength;
        for (rows, 0..) |r, i| {
            if (r >= ds.dims[0] or (i > 0 and r <= rows[i - 1])) return error.OutOfRange;
        }
        if (rows.len == 0) return;
        switch (ds.layout) {
            .contiguous => |c| {
                if (c.addr == UNDEF) {
                    @memset(dst, 0);
                    return;
                }
                // runs of consecutive rows, all in one round trip
                var ranges: std.ArrayList(Range) = .empty;
                defer ranges.deinit(self.gpa);
                var i: usize = 0;
                while (i < rows.len) {
                    var j = i + 1;
                    while (j < rows.len and rows[j] == rows[j - 1] + 1) j += 1;
                    try ranges.append(self.gpa, .{ .offset = c.addr + self.base + rows[i] * row, .len = (j - i) * row });
                    i = j;
                }
                try self.f.readSpans(ranges.items, dst);
            },
            .chunked, .implicit => try self.readChunked(ds, rows, dst),
            .compact, .single => for (rows, 0..) |r, i| try self.readRows(ds, r, 1, dst[i * row ..][0..row]),
        }
    }

    pub fn readAll(self: *Hdf5, ds: *Dataset, dst: []u8) !void {
        return self.readRows(ds, 0, if (ds.rank == 0) 1 else ds.dims[0], dst);
    }

    fn initIndex(self: *Hdf5, ds: *Dataset) !void {
        if (ds.index.len != 0 or ds.index_ready) return;
        switch (ds.layout) {
            .chunked, .implicit => {},
            else => {
                ds.index_ready = true;
                return;
            },
        }
        var n: u64 = 1;
        for (0..ds.rank) |d| {
            if (ds.chunk[d] == 0) return error.BadHdf5;
            ds.grid[d] = ds.dims[d] / ds.chunk[d] + @intFromBool(ds.dims[d] % ds.chunk[d] != 0);
            n = try std.math.mul(u64, n, ds.grid[d]);
        }
        errdefer {
            ds.frontier.clearRetainingCapacity();
            ds.nodes_seen.clearRetainingCapacity();
        }
        const index = try self.gpa.alloc(Slot, std.math.cast(usize, n) orelse return error.BadHdf5);
        errdefer self.gpa.free(index);
        const known = try self.gpa.alloc(bool, index.len);
        errdefer self.gpa.free(known);
        @memset(index, .{});
        @memset(known, false);
        switch (ds.layout) {
            .chunked => |c| {
                if (c.btree != UNDEF) {
                    try ds.nodes_seen.put(self.gpa, c.btree, {});
                    try ds.frontier.append(self.gpa, c.btree);
                } else {
                    @memset(known, true);
                    ds.index_ready = true;
                }
            },
            .implicit => |c| {
                const bytes = try std.math.mul(u64, ds.chunkElements(), ds.dtype.size);
                for (index, 0..) |*slot, k| slot.* = .{ .addr = try std.math.add(u64, c.addr, try std.math.mul(u64, k, bytes)), .size = std.math.cast(u32, bytes) orelse return error.BadHdf5, .mask = 0xffffffff };
                @memset(known, true);
                ds.index_ready = true;
            },
            else => ds.index_ready = true,
        }
        ds.index = index;
        ds.known = known;
        ds.node_len = 8 + 2 * @as(usize, self.so) + 64 * self.treeEntry(ds) + self.treeKey(ds);
    }

    fn treeKey(_: *Hdf5, ds: *const Dataset) usize {
        return 8 + 8 * (@as(usize, ds.rank) + 1);
    }
    fn treeEntry(self: *Hdf5, ds: *const Dataset) usize {
        return self.treeKey(ds) + self.so;
    }
    fn treeHead(self: *Hdf5) usize {
        return 8 + 2 * @as(usize, self.so);
    }

    fn nodeRange(self: *Hdf5, addr: u64, len: usize) !Range {
        const at = try std.math.add(u64, addr, self.base);
        if (at >= self.f.size) return error.BadHdf5;
        return .{ .offset = at, .len = @intCast(@min(len, self.f.size - at)) };
    }

    /// Includes the terminal upper-bound key, needed for safe targeted descent.
    fn nodeRequired(self: *Hdf5, ds: *Dataset, b: []const u8) !usize {
        if (b.len < self.treeHead() or !std.mem.eql(u8, b[0..4], "TREE") or b[4] != 1) return error.BadHdf5;
        const used: usize = std.mem.readInt(u16, b[6..8], .little);
        return self.treeHead() + used * self.treeEntry(ds) + self.treeKey(ds);
    }

    fn chunkChild(self: *Hdf5, ds: *Dataset, p: []const u8) u64 {
        const key = self.treeKey(ds);
        const addr = if (self.so == 8) std.mem.readInt(u64, p[key..][0..8], .little) else std.mem.readInt(u32, p[key..][0..4], .little);
        return if (self.so == 4 and addr == 0xffffffff) UNDEF else addr;
    }

    fn leaf(self: *Hdf5, ds: *Dataset, b: []const u8) !void {
        const used: usize = std.mem.readInt(u16, b[6..8], .little);
        for (0..used) |i| {
            const p = b[self.treeHead() + i * self.treeEntry(ds) ..];
            var slot: u64 = 0;
            for (0..ds.rank) |d| {
                const o = std.mem.readInt(u64, p[8 + 8 * d ..][0..8], .little);
                if (o >= ds.dims[d] or o % ds.chunk[d] != 0) return error.BadHdf5;
                slot = slot * ds.grid[d] + o / ds.chunk[d];
            }
            ds.index[@intCast(slot)] = .{ .addr = self.chunkChild(ds, p), .size = std.mem.readInt(u32, p[0..4], .little), .mask = std.mem.readInt(u32, p[4..8], .little) };
            ds.known[@intCast(slot)] = true;
        }
    }

    /// Warm a bounded tree batch. Exactly one backend readv per call; an oversized
    /// node updates the next call's range instead of performing an unbounded retry.
    pub fn prepareIndexStep(self: *Hdf5, ds: *Dataset, max_nodes: usize, max_bytes: usize) !bool {
        if (max_nodes == 0 or max_bytes == 0) return error.BadLength;
        errdefer |err| {
            if (err != error.ChunkIndexBudgetTooSmall) ds.index_failed = true;
        }
        try self.initIndex(ds);
        if (ds.index_failed) return error.ChunkIndexFailed;
        if (ds.index_ready) return true;
        var ranges: std.ArrayList(Range) = .empty;
        defer ranges.deinit(self.gpa);
        var total: usize = 0;
        const left = ds.frontier.items.len - ds.frontier_at;
        for (ds.frontier.items[ds.frontier_at..][0..@min(left, max_nodes)]) |addr| {
            const range = try self.nodeRange(addr, ds.node_len);
            if (range.len > max_bytes) return error.ChunkIndexBudgetTooSmall;
            if (range.len > max_bytes - total) break;
            try ranges.append(self.gpa, range);
            total += range.len;
        }
        const buf = try self.gpa.alloc(u8, total);
        defer self.gpa.free(buf);
        try self.f.readv(ranges.items, buf);
        var off: usize = 0;
        var need: usize = 0;
        for (ranges.items) |range| {
            const b = buf[off..][0..range.len];
            off += range.len;
            const required = try self.nodeRequired(ds, b);
            if (required > b.len) need = @max(need, required);
        }
        if (need != 0) {
            if (need <= ds.node_len) return error.BadHdf5;
            ds.node_len = need;
            return false;
        }
        off = 0;
        for (ranges.items) |range| {
            const b = buf[off..][0..range.len];
            off += range.len;
            if (b[5] == 0) try self.leaf(ds, b) else {
                const used: usize = std.mem.readInt(u16, b[6..8], .little);
                for (0..used) |i| {
                    const addr = self.chunkChild(ds, b[self.treeHead() + i * self.treeEntry(ds) ..]);
                    if (addr == UNDEF) return error.BadHdf5;
                    const seen = try ds.nodes_seen.getOrPut(self.gpa, addr);
                    if (seen.found_existing) return error.BadHdf5;
                    try ds.frontier.append(self.gpa, addr);
                }
            }
        }
        ds.warmed_nodes += ranges.items.len;
        ds.frontier_at += ranges.items.len;
        if (ds.frontier_at == ds.frontier.items.len) {
            @memset(ds.known, true);
            ds.index_ready = true;
            ds.frontier.clearAndFree(self.gpa);
            ds.nodes_seen.clearAndFree(self.gpa);
            ds.frontier_at = 0;
        }
        return ds.index_ready;
    }

    fn compareOrigin(ds: *const Dataset, slot: usize, key: []const u8) std.math.Order {
        var origin: [MAX_RANK]u64 = @splat(0);
        var rem = slot;
        var d: usize = ds.rank;
        while (d > 0) {
            d -= 1;
            origin[d] = (rem % ds.grid[d]) * ds.chunk[d];
            rem /= @intCast(ds.grid[d]);
        }
        for (origin[0..ds.rank], 0..) |o, k| {
            const boundary = std.mem.readInt(u64, key[8 + 8 * k ..][0..8], .little);
            if (o != boundary) return std.math.order(o, boundary);
        }
        return .eq;
    }

    /// Resolve only requested slots, batching requests that share B-tree nodes.
    /// Unknown slots become missing only after a verified path/leaf has excluded them.
    fn lookupSlots(self: *Hdf5, ds: *Dataset, slots: []const usize) !void {
        if (ds.index_ready) return;
        if (ds.index_failed) return error.ChunkIndexFailed;
        errdefer ds.index_failed = true;
        const Query = struct { addr: u64, slot: usize, depth: u16 = 0 };
        var queries: std.ArrayList(Query) = .empty;
        defer queries.deinit(self.gpa);
        for (slots) |slot| if (!ds.known[slot]) {
            try queries.append(self.gpa, .{ .addr = ds.layout.chunked.btree, .slot = slot });
        };
        while (queries.items.len != 0) {
            std.mem.sort(Query, queries.items, {}, struct {
                fn lt(_: void, a: Query, b: Query) bool {
                    return a.addr < b.addr;
                }
            }.lt);
            var next: std.ArrayList(Query) = .empty;
            errdefer next.deinit(self.gpa);
            var ranges: std.ArrayList(Range) = .empty;
            defer ranges.deinit(self.gpa);
            var last = UNDEF;
            var total: usize = 0;
            for (queries.items) |q| if (q.addr != last) {
                const range = try self.nodeRange(q.addr, ds.node_len);
                try ranges.append(self.gpa, range);
                total += range.len;
                last = q.addr;
            };
            const buf = try self.gpa.alloc(u8, total);
            defer self.gpa.free(buf);
            if (ranges.items.len == 1) try self.f.read(ranges.items[0].offset, buf) else try self.f.readv(ranges.items, buf);
            var off: usize = 0;
            var need: usize = 0;
            for (ranges.items) |range| {
                const required = try self.nodeRequired(ds, buf[off..][0..range.len]);
                if (required > range.len) need = @max(need, required);
                off += range.len;
            }
            if (need != 0) {
                if (need <= ds.node_len) return error.BadHdf5;
                ds.node_len = need;
                continue;
            }
            off = 0;
            var qi: usize = 0;
            for (ranges.items) |range| {
                const b = buf[off..][0..range.len];
                off += range.len;
                const addr = queries.items[qi].addr;
                const used: usize = std.mem.readInt(u16, b[6..8], .little);
                if (b[5] == 0) try self.leaf(ds, b);
                while (qi < queries.items.len and queries.items[qi].addr == addr) : (qi += 1) {
                    const q = queries.items[qi];
                    if (ds.known[q.slot]) continue;
                    if (q.depth >= 64) return error.BadHdf5;
                    var found = false;
                    if (b[5] != 0) for (0..used) |j| {
                        const p = b[self.treeHead() + j * self.treeEntry(ds) ..];
                        const upper = b[self.treeHead() + (j + 1) * self.treeEntry(ds) ..];
                        if (compareOrigin(ds, q.slot, p) != .lt and compareOrigin(ds, q.slot, upper) == .lt) {
                            const kid = self.chunkChild(ds, p);
                            if (kid == UNDEF) return error.BadHdf5;
                            try next.append(self.gpa, .{ .addr = kid, .slot = q.slot, .depth = q.depth + 1 });
                            found = true;
                            break;
                        }
                    };
                    if (!found) ds.known[q.slot] = true;
                }
            }
            queries.deinit(self.gpa);
            queries = next;
        }
    }

    /// Undo a chunk's filters: `stored` as read from the file → the chunk's `out_len` bytes
    /// (caller frees). A set mask bit means that filter was skipped when the chunk was written.
    fn unfilter(self: *Hdf5, ds: *const Dataset, stored: []const u8, mask: u32, out_len: usize) ![]u8 {
        var cur = try self.gpa.dupe(u8, stored);
        errdefer self.gpa.free(cur);
        var i: usize = ds.n_filters;
        while (i > 0) {
            i -= 1;
            if (mask & (@as(u32, 1) << @intCast(i)) != 0) continue;
            const fl = ds.filters[i];
            switch (fl.id) {
                1, 32000 => {
                    const next = try self.gpa.alloc(u8, out_len);
                    errdefer self.gpa.free(next);
                    if (fl.id == 1) try codec.inflate(cur, next, false) else try codec.lzf(cur, next);
                    self.gpa.free(cur);
                    cur = next;
                },
                2 => {
                    const next = try self.gpa.alloc(u8, cur.len);
                    codec.unshuffle(cur, next, if (fl.elem != 0) fl.elem else ds.dtype.size);
                    self.gpa.free(cur);
                    cur = next;
                },
                // Fletcher-32: the checksum trails the data
                3 => cur = try self.gpa.realloc(cur, cur.len - 4),
                else => unreachable,
            }
        }
        if (cur.len != out_len) return error.BadHdf5;
        return cur;
    }

    /// The decoded bytes of the one chunk of a single-chunk dataset (kept for the next call).
    fn chunkBytes(self: *Hdf5, ds: *Dataset, addr: u64, stored: u64, mask: u32, out_len: u64) ![]const u8 {
        if (ds.cache_addr == addr and ds.cache.len == out_len) return ds.cache;
        ds.cache_addr = UNDEF;
        self.gpa.free(ds.cache);
        ds.cache = &.{};
        if (ds.n_filters == 0 or mask == 0xffffffff) {
            ds.cache = try self.f.alloc(self.gpa, addr + self.base, @intCast(out_len));
        } else {
            const raw = try self.f.alloc(self.gpa, addr + self.base, @intCast(stored));
            defer self.gpa.free(raw);
            ds.cache = try self.unfilter(ds, raw, mask, @intCast(out_len));
        }
        ds.cache_addr = addr;
        return ds.cache;
    }

    const Want = struct {
        slot: usize,
        addr: u64,
        size: usize,
        /// where its stored bytes start in the span buffer
        at: usize = 0,
    };

    /// The given steps of the first axis (ascending) of a chunked dataset into `dst`.
    fn readChunked(self: *Hdf5, ds: *Dataset, rows: []const u64, dst: []u8) !void {
        try self.initIndex(ds);
        if (ds.index_failed) return error.ChunkIndexFailed;
        const gpa = self.gpa;
        const es: usize = ds.dtype.size;
        const rank: usize = ds.rank;
        @memset(dst, 0); // chunks never written read as the fill value
        if (ds.index.len == 0 or rows.len == 0) return;
        // element strides of a row of the output and of a chunk
        var ostride: [MAX_RANK]u64 = undefined;
        var cstride: [MAX_RANK]u64 = undefined;
        var os: u64 = 1;
        var cs: u64 = 1;
        var d = rank;
        while (d > 0) {
            d -= 1;
            ostride[d] = os;
            cstride[d] = cs;
            os *= ds.dims[d];
            cs *= ds.chunk[d];
        }
        const chunk_bytes: usize = @intCast(cs * es);
        const filtered = ds.n_filters != 0;
        const c0 = ds.chunk[0];
        // chunks in one slab of the grid's first axis
        const slab_len: usize = @intCast(ds.index.len / ds.grid[0]);
        if (!ds.index_ready and rows.len == ds.dims[0]) {
            // Whole-dataset readers need every slot: drain the existing bounded walker
            // instead of allocating one targeted query per chunk.
            while (!try self.prepareIndexStep(ds, 2048, 8 * 1024 * 1024)) {}
        } else if (!ds.index_ready) {
            // At most eight unknown slots per lookup keeps even oversized v1 nodes
            // below the payload path's 64 MiB scratch limit.
            var slots: [8]usize = undefined;
            var used: usize = 0;
            var last_slab: u64 = UNDEF;
            for (rows) |r| {
                const slab = r / c0;
                if (slab == last_slab) continue;
                last_slab = slab;
                for (0..slab_len) |k| {
                    const slot = @as(usize, @intCast(slab)) * slab_len + k;
                    if (ds.known[slot]) continue;
                    slots[used] = slot;
                    used += 1;
                    if (used == slots.len) {
                        try self.lookupSlots(ds, &slots);
                        used = 0;
                    }
                }
            }
            if (used != 0) try self.lookupSlots(ds, slots[0..used]);
        }

        var want: std.ArrayList(Want) = .empty;
        defer want.deinit(gpa);
        var ranges: std.ArrayList(Range) = .empty;
        defer ranges.deinit(gpa);
        var span: std.ArrayList(u8) = .empty;
        defer span.deinit(gpa);
        var order: std.ArrayList(u32) = .empty;
        defer order.deinit(gpa);

        var i: usize = 0;
        while (i < rows.len) {
            // the chunks of as many slabs as fit one span; they cover rows[i0..i]
            const begin = i;
            want.clearRetainingCapacity();
            var bytes: usize = 0;
            while (i < rows.len) {
                const slab: usize = @intCast(rows[i] / c0);
                var slab_bytes: usize = 0;
                for (ds.index[slab * slab_len ..][0..slab_len]) |sl| {
                    if (sl.addr != UNDEF) slab_bytes += sl.size;
                }
                if (want.items.len != 0 and bytes + slab_bytes > SPAN_BYTES) break;
                for (ds.index[slab * slab_len ..][0..slab_len], 0..) |sl, k| {
                    if (sl.addr == UNDEF) continue;
                    if ((!filtered or sl.mask == 0xffffffff) and sl.size != chunk_bytes) return error.BadHdf5;
                    try want.append(gpa, .{ .slot = slab * slab_len + k, .addr = sl.addr + self.base, .size = sl.size });
                }
                bytes += slab_bytes;
                while (i < rows.len and rows[i] / c0 == slab) i += 1;
            }
            if (want.items.len == 0) continue;

            // their stored bytes in file order, neighbours as one range, all in one round trip
            try order.resize(gpa, want.items.len);
            for (order.items, 0..) |*o, k| o.* = @intCast(k);
            std.mem.sort(u32, order.items, want.items, struct {
                fn lt(w: []Want, a: u32, b: u32) bool {
                    return w[a].addr < w[b].addr;
                }
            }.lt);
            ranges.clearRetainingCapacity();
            var total: usize = 0;
            for (order.items) |k| {
                const w = &want.items[k];
                w.at = total;
                try ranges.append(gpa, .{ .offset = w.addr, .len = w.size });
                total += w.size;
            }
            try span.resize(gpa, total);
            try self.f.readSpans(ranges.items, span.items);

            const part = rows[begin..i];
            for (want.items) |w| {
                const stored = span.items[w.at..][0..w.size];
                const sl = ds.index[w.slot];
                var owned: ?[]u8 = null;
                defer if (owned) |o| gpa.free(o);
                var data: []const u8 = stored;
                if (filtered and sl.mask != 0xffffffff) {
                    owned = try self.unfilter(ds, stored, sl.mask, chunk_bytes);
                    data = owned.?;
                }
                // the chunk's origin from its place in the grid
                var off: [MAX_RANK]u64 = @splat(0);
                var rem: u64 = w.slot;
                var k = rank;
                while (k > 0) {
                    k -= 1;
                    off[k] = (rem % ds.grid[k]) * ds.chunk[k];
                    rem /= ds.grid[k];
                }
                // the part of the chunk inside the dataset along the other axes, copied in runs
                // along the last one, for each wanted row the chunk holds
                var hi: [MAX_RANK]u64 = @splat(1);
                for (1..rank) |a| hi[a] = @min(off[a] + ds.chunk[a], ds.dims[a]);
                var lo: usize = 0;
                var up: usize = part.len;
                while (lo < up) {
                    const mid = (lo + up) / 2;
                    if (part[mid] < off[0]) lo = mid + 1 else up = mid;
                }
                var j = lo;
                while (j < part.len and part[j] < off[0] + c0) : (j += 1) {
                    const src_row = (part[j] - off[0]) * cstride[0];
                    const dst_row = (begin + j) * ostride[0];
                    if (rank == 1) {
                        @memcpy(dst[@intCast(dst_row * es)..][0..es], data[@intCast(src_row * es)..][0..es]);
                        continue;
                    }
                    const last = rank - 1;
                    const run: usize = @intCast((hi[last] - off[last]) * es);
                    var pos = off;
                    outer: while (true) {
                        var src: u64 = src_row;
                        var out: u64 = dst_row;
                        for (1..rank) |a| {
                            src += (pos[a] - off[a]) * cstride[a];
                            out += pos[a] * ostride[a];
                        }
                        @memcpy(dst[@intCast(out * es)..][0..run], data[@intCast(src * es)..][0..run]);
                        var a = last;
                        while (a > 1) {
                            a -= 1;
                            pos[a] += 1;
                            if (pos[a] < hi[a]) continue :outer;
                            pos[a] = off[a];
                        }
                        break;
                    }
                }
            }
        }
    }

    // ── values ──

    /// Up to `out.len` values of a numeric dataset, converted to f64; returns the count read.
    pub fn numbers(self: *Hdf5, addr: u64, out: []f64) !usize {
        var ds = try self.dataset(addr);
        defer ds.deinit(self.gpa);
        const n: usize = @intCast(@min(ds.elements(), out.len));
        if (n == 0) return 0;
        const es: usize = ds.dtype.size;
        const total: usize = @intCast(ds.elements() * es);
        const raw = try self.gpa.alloc(u8, total);
        defer self.gpa.free(raw);
        try self.readAll(&ds, raw);
        for (0..n) |i| out[i] = try toF64(ds.dtype, raw[i * es ..][0..es]);
        return n;
    }

    /// The first value of a numeric dataset.
    pub fn number(self: *Hdf5, addr: u64) !f64 {
        var v: [1]f64 = undefined;
        if (try self.numbers(addr, &v) == 0) return error.Empty;
        return v[0];
    }

    /// The first string of a string dataset into `buf`; returns its length.
    pub fn string(self: *Hdf5, addr: u64, buf: []u8) ![]u8 {
        var ds = try self.dataset(addr);
        defer ds.deinit(self.gpa);
        if (ds.elements() == 0) return buf[0..0];
        const es: usize = ds.dtype.size;
        const raw = try self.gpa.alloc(u8, @intCast(ds.elements() * es));
        defer self.gpa.free(raw);
        try self.readAll(&ds, raw);
        switch (ds.dtype.class) {
            .string => {
                const s = raw[0..es];
                const len = std.mem.indexOfScalar(u8, s, 0) orelse es;
                var n = @min(len, buf.len);
                @memcpy(buf[0..n], s[0..n]);
                while (n > 0 and buf[n - 1] == ' ') n -= 1;
                return buf[0..n];
            },
            .vlen_string => {
                // length, global heap collection address, object index
                const len = std.mem.readInt(u32, raw[0..4], .little);
                const coll = std.mem.readVarInt(u64, raw[4..][0..self.so], .little);
                const idx = std.mem.readInt(u32, raw[4 + self.so ..][0..4], .little);
                return self.heapObject(coll, idx, len, buf);
            },
            else => return error.NotAString,
        }
    }

    fn heapObject(self: *Hdf5, coll: u64, index: u32, len: u32, buf: []u8) ![]u8 {
        const a = coll + self.base;
        var sig: [4]u8 = undefined;
        try self.f.read(a, &sig);
        if (!std.mem.eql(u8, &sig, "GCOL")) return error.BadHdf5;
        const size = try self.length(a + 8);
        var p = a + 8 + self.sl;
        while (p + 8 + self.sl <= a + size) {
            const id = try self.f.int(u16, p);
            const osize = try self.length(p + 8);
            const body = p + 8 + self.sl;
            if (id == 0) break;
            if (id == index) {
                const n: usize = @intCast(@min(@min(osize, len), buf.len));
                try self.f.read(body, buf[0..n]);
                return buf[0..n];
            }
            p = body + std.mem.alignForward(u64, osize, 8);
        }
        return error.BadHdf5;
    }
};

pub fn toF64(t: Dtype, b: []const u8) !f64 {
    const e: std.builtin.Endian = if (t.big_endian) .big else .little;
    return switch (t.class) {
        .uint => switch (t.size) {
            1 => @floatFromInt(b[0]),
            2 => @floatFromInt(std.mem.readInt(u16, b[0..2], e)),
            4 => @floatFromInt(std.mem.readInt(u32, b[0..4], e)),
            8 => @floatFromInt(std.mem.readInt(u64, b[0..8], e)),
            else => error.UnsupportedType,
        },
        .int => switch (t.size) {
            1 => @floatFromInt(@as(i8, @bitCast(b[0]))),
            2 => @floatFromInt(std.mem.readInt(i16, b[0..2], e)),
            4 => @floatFromInt(std.mem.readInt(i32, b[0..4], e)),
            8 => @floatFromInt(std.mem.readInt(i64, b[0..8], e)),
            else => error.UnsupportedType,
        },
        .float => switch (t.size) {
            4 => @as(f32, @bitCast(std.mem.readInt(u32, b[0..4], e))),
            8 => @bitCast(std.mem.readInt(u64, b[0..8], e)),
            else => error.UnsupportedType,
        },
        else => error.UnsupportedType,
    };
}

test "read the nickel scan like h5py" {
    const gpa = std.testing.allocator;
    var f = File.open(gpa, "testdata/nickel/patterns_v2.h5") catch return; // run from the repo root
    defer f.close();
    var h = try Hdf5.open(gpa, &f);
    var buf: [64]u8 = undefined;
    try std.testing.expectEqualStrings("kikuchipy", try h.string((try h.find("manufacturer")).?, &buf));
    try std.testing.expectEqual(@as(f64, 55), try h.number((try h.find("Scan 1/EBSD/Header/n_rows")).?));
    try std.testing.expectEqual(@as(f64, 1.5), try h.number((try h.find("Scan 1/EBSD/Header/step_x")).?));
    try std.testing.expect((try h.find("Scan 1/EBSD/Nope")) == null);
    try std.testing.expect(try h.isGroup((try h.find("Scan 1/EBSD")).?));
    const names = try h.list(h.root);
    defer h.freeList(names);
    try std.testing.expectEqual(@as(usize, 3), names.len);

    var ds = try h.dataset((try h.find("Scan 1/EBSD/Data/patterns")).?);
    defer ds.deinit(gpa);
    try std.testing.expectEqual(@as(u8, 3), ds.rank);
    try std.testing.expectEqual(@as(u64, 4125), ds.dims[0]);
    try std.testing.expectEqual(@as(u64, 277), ds.chunk[0]);
    try std.testing.expectEqual(Class.uint, ds.dtype.class);
    // sums checked against h5py: patterns[0], patterns[276:279], patterns[4124]
    const one = try gpa.alloc(u8, 3600);
    defer gpa.free(one);
    const sum = struct {
        fn of(b: []const u8) u64 {
            var s: u64 = 0;
            for (b) |v| s += v;
            return s;
        }
    }.of;
    try h.readRows(&ds, 0, 1, one);
    try std.testing.expectEqual(@as(u64, 531330), sum(one));
    const three = try gpa.alloc(u8, 3 * 3600);
    defer gpa.free(three);
    try h.readRows(&ds, 276, 3, three);
    try std.testing.expectEqual(@as(u64, 1571367), sum(three));
    try h.readRows(&ds, 4124, 1, one);
    try std.testing.expectEqual(@as(u64, 523040), sum(one));
    try std.testing.expectError(error.OutOfRange, h.readRows(&ds, 4125, 1, one));
    // a list of rows across chunks equals the rows read one by one
    const list = [_]u64{ 0, 276, 277, 278, 2000, 4124 };
    const many = try gpa.alloc(u8, list.len * 3600);
    defer gpa.free(many);
    try h.readRowList(&ds, &list, many);
    try std.testing.expectEqual(@as(u64, 531330), sum(many[0..3600]));
    try std.testing.expectEqualSlices(u8, three, many[3600 .. 4 * 3600]);
    try std.testing.expectEqualSlices(u8, one, many[5 * 3600 ..]);
    try h.readRows(&ds, 2000, 1, one);
    try std.testing.expectEqualSlices(u8, one, many[4 * 3600 .. 5 * 3600]);
    try std.testing.expectError(error.OutOfRange, h.readRowList(&ds, &.{ 5, 5 }, many[0 .. 2 * 3600]));

    var pcx: [4125]f64 = undefined;
    try std.testing.expectEqual(@as(usize, 4125), try h.numbers((try h.find("Scan 1/EBSD/Header/pcx")).?, &pcx));
    try std.testing.expectApproxEqAbs(@as(f64, 0.42771687591335866), pcx[0], 1e-15);
}
