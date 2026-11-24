//! A writer for the part of HDF5 that results need: a new file of groups, datasets of
//! little-endian numbers and of fixed- and variable-length strings, stored contiguously, and
//! attributes of the same, laid out as HDF5 1.8 and later read it: superblock version 2, version
//! 2 object headers, groups as link messages in their headers, the variable-length strings in
//! global heap collections right after the superblock. Written against the file format specification, as
//! io/hdf5.zig reads it, so the engine needs no HDF5 library to write either. The whole file is
//! built in memory (results are maps, not patterns).
const std = @import("std");

const Allocator = std.mem.Allocator;
const UNDEF: u64 = std.math.maxInt(u64);

pub const Class = enum { uint, int, float, string, vstring };

/// An element type: a number of 1, 2, 4 or 8 bytes (floats 4 or 8), a string of `size` bytes
/// (UTF-8, padded with nulls), or a variable-length string (UTF-8, as h5py and Oxford's files
/// write them; `vstr`).
pub const Type = struct {
    class: Class,
    size: u32,

    pub const u8_: Type = .{ .class = .uint, .size = 1 };
    pub const i32_: Type = .{ .class = .int, .size = 4 };
    pub const u32_: Type = .{ .class = .uint, .size = 4 };
    pub const f32_: Type = .{ .class = .float, .size = 4 };
    pub const f64_: Type = .{ .class = .float, .size = 8 };

    /// (an element on disk: its length, the address of its heap collection, its index there)
    pub const vstr: Type = .{ .class = .vstring, .size = 16 };

    pub fn str(len: usize) Type {
        return .{ .class = .string, .size = @intCast(@max(len, 1)) };
    }

    fn valid(self: Type) bool {
        return switch (self.class) {
            .uint, .int => self.size == 1 or self.size == 2 or self.size == 4 or self.size == 8,
            .float => self.size == 4 or self.size == 8,
            .string => self.size >= 1,
            .vstring => self.size == 16,
        };
    }
};

/// A value: its type, its shape (empty: a scalar) and its bytes, little-endian, row-major; for
/// variable-length strings, the strings themselves, separated by nulls.
pub const Value = struct {
    dtype: Type,
    dims: []const u64 = &.{},
    bytes: []const u8,

    fn count(self: Value) u64 {
        var n: u64 = 1;
        for (self.dims) |d| n *= d;
        return n;
    }
};

const Attr = struct { name: []const u8, value: Value };
const Link = struct { name: []const u8, node: *Node };

pub const Node = struct {
    /// a dataset's value; null for a group
    value: ?Value = null,
    links: std.ArrayList(Link) = .empty,
    attrs: std.ArrayList(Attr) = .empty,

    pub fn isGroup(self: *const Node) bool {
        return self.value == null;
    }
};

/// A global heap collection: its address and its strings (object i + 1 is strings[i]).
const Heap = struct {
    at: u64,
    strings: std.ArrayList([]const u8) = .empty,
    /// its bytes so far: its header and each object's, with the object padded to 8
    used: u64 = 16,

    /// The collection's size: what it holds, and the free space after it (at least an object's
    /// header; at least 4096 bytes in all, as libhdf5 reads a collection).
    fn size(self: *const Heap) u64 {
        return @max(4096, self.used + 16);
    }
};

/// Where the first heap collection goes: after the superblock (version 2, 48 bytes).
const HEAPS_AT: u64 = 48;

pub const File = struct {
    arena: std.heap.ArenaAllocator,
    root: *Node,
    heaps: std.ArrayList(Heap) = .empty,

    pub fn init(gpa: Allocator) !File {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const root = try arena.allocator().create(Node);
        root.* = .{};
        return .{ .arena = arena, .root = root };
    }

    pub fn deinit(self: *File) void {
        self.arena.deinit();
    }

    fn find(parent: *Node, name: []const u8) ?*Node {
        for (parent.links.items) |l| if (std.mem.eql(u8, l.name, name)) return l.node;
        return null;
    }

    /// The group `name` in `parent`, made if it is not there.
    pub fn group(self: *File, parent: *Node, name: []const u8) !*Node {
        if (find(parent, name)) |n| {
            if (!n.isGroup()) return error.NameTaken;
            return n;
        }
        const a = self.arena.allocator();
        const n = try a.create(Node);
        n.* = .{};
        try parent.links.append(a, .{ .name = try a.dupe(u8, name), .node = n });
        return n;
    }

    /// The group at a path of names separated by '/', from the root, made as needed.
    pub fn path(self: *File, p: []const u8) !*Node {
        var node = self.root;
        var it = std.mem.tokenizeScalar(u8, p, '/');
        while (it.next()) |name| node = try self.group(node, name);
        return node;
    }

    /// A dataset `name` in `parent`, its value copied.
    pub fn dataset(self: *File, parent: *Node, name: []const u8, v: Value) !*Node {
        if (find(parent, name) != null) return error.NameTaken;
        const a = self.arena.allocator();
        const n = try a.create(Node);
        n.* = .{ .value = try self.stored(v) };
        try parent.links.append(a, .{ .name = try a.dupe(u8, name), .node = n });
        return n;
    }

    /// A string dataset (one variable-length string, a scalar).
    pub fn text(self: *File, parent: *Node, name: []const u8, s: []const u8) !*Node {
        return self.dataset(parent, name, .{ .dtype = Type.vstr, .bytes = s });
    }

    /// An attribute of a group or a dataset, its value copied.
    pub fn attribute(self: *File, node: *Node, name: []const u8, v: Value) !void {
        const a = self.arena.allocator();
        try node.attrs.append(a, .{ .name = try a.dupe(u8, name), .value = try self.stored(v) });
    }

    /// An attribute whose value is one this file already stores (another node's), as it is.
    pub fn attributeStored(self: *File, node: *Node, name: []const u8, v: Value) !void {
        const a = self.arena.allocator();
        try node.attrs.append(a, .{ .name = try a.dupe(u8, name), .value = .{ .dtype = v.dtype, .dims = try a.dupe(u64, v.dims), .bytes = try a.dupe(u8, v.bytes) } });
    }

    /// The text of a stored value of one string (null: not a string, or more than one).
    pub fn textOf(self: *const File, v: Value) ?[]const u8 {
        if (v.count() != 1) return null;
        switch (v.dtype.class) {
            .string => return std.mem.sliceTo(v.bytes, 0),
            .vstring => {
                const len = std.mem.readInt(u32, v.bytes[0..4], .little);
                if (len == 0) return "";
                const at = std.mem.readInt(u64, v.bytes[4..12], .little);
                const idx = std.mem.readInt(u32, v.bytes[12..16], .little);
                for (self.heaps.items) |h| if (h.at == at) return if (idx >= 1 and idx <= h.strings.items.len) h.strings.items[idx - 1] else null;
                return null;
            },
            else => return null,
        }
    }

    /// The value as it is stored: copied, and variable-length strings put in a heap
    /// collection, the value their elements (an empty string has none: length and address 0).
    fn stored(self: *File, v: Value) !Value {
        if (!v.dtype.valid()) return error.UnsupportedType;
        const a = self.arena.allocator();
        const dims = try a.dupe(u64, v.dims);
        if (v.dtype.class != .vstring) {
            if (v.bytes.len != v.count() * v.dtype.size) return error.BadLength;
            return .{ .dtype = v.dtype, .dims = dims, .bytes = try a.dupe(u8, v.bytes) };
        }
        const n = v.count();
        if (if (n == 0) v.bytes.len != 0 else std.mem.count(u8, v.bytes, &.{0}) + 1 != n) return error.BadLength;
        const el = try a.alloc(u8, @intCast(16 * n));
        @memset(el, 0);
        var it = std.mem.splitScalar(u8, v.bytes, 0);
        for (0..@intCast(n)) |i| {
            const s = it.next().?;
            if (s.len == 0) continue;
            if (s.len > std.math.maxInt(u32)) return error.BadLength;
            if (self.heaps.items.len == 0 or self.heaps.items[self.heaps.items.len - 1].strings.items.len == std.math.maxInt(u16)) {
                const at = if (self.heaps.items.len == 0) HEAPS_AT else blk: {
                    const last = &self.heaps.items[self.heaps.items.len - 1];
                    break :blk last.at + last.size();
                };
                try self.heaps.append(a, .{ .at = at });
            }
            const h = &self.heaps.items[self.heaps.items.len - 1];
            try h.strings.append(a, try a.dupe(u8, s));
            h.used += 16 + std.mem.alignForward(u64, s.len, 8);
            const e = el[16 * i ..][0..16];
            std.mem.writeInt(u32, e[0..4], @intCast(s.len), .little);
            std.mem.writeInt(u64, e[4..12], h.at, .little);
            std.mem.writeInt(u32, e[12..16], @intCast(h.strings.items.len), .little);
        }
        return .{ .dtype = v.dtype, .dims = dims, .bytes = el };
    }

    /// The file's bytes (caller frees).
    pub fn bytes(self: *File, gpa: Allocator) ![]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(gpa);
        try out.appendNTimes(gpa, 0, HEAPS_AT);
        for (self.heaps.items) |*h| try heap(gpa, &out, h);
        const root = try emit(gpa, &out, self.root);
        // superblock version 2: signature, version, sizes of offsets and lengths, flags, base
        // address, extension address, end of file, root group's header, checksum
        var sb: [48]u8 = undefined;
        @memcpy(sb[0..8], "\x89HDF\r\n\x1a\n");
        sb[8] = 2;
        sb[9] = 8;
        sb[10] = 8;
        sb[11] = 0;
        std.mem.writeInt(u64, sb[12..20], 0, .little);
        std.mem.writeInt(u64, sb[20..28], UNDEF, .little);
        std.mem.writeInt(u64, sb[28..36], out.items.len, .little);
        std.mem.writeInt(u64, sb[36..44], root, .little);
        std.mem.writeInt(u32, sb[44..48], lookup3(sb[0..44], 0), .little);
        @memcpy(out.items[0..48], &sb);
        return out.toOwnedSlice(gpa);
    }
};

// ── layout ──

fn pad8(gpa: Allocator, out: *std.ArrayList(u8)) !void {
    while (out.items.len % 8 != 0) try out.append(gpa, 0);
}

/// A global heap collection, version 1: its header (signature, version, its size), each object
/// (index, reference count, its size, its bytes padded to 8), then object 0, the free space to
/// its end.
fn heap(gpa: Allocator, out: *std.ArrayList(u8), h: *const Heap) !void {
    if (out.items.len != h.at) return error.BadLayout;
    const size = h.size();
    try out.appendSlice(gpa, &.{ 'G', 'C', 'O', 'L', 1, 0, 0, 0 });
    try out.appendSlice(gpa, &std.mem.toBytes(std.mem.nativeToLittle(u64, size)));
    for (h.strings.items, 1..) |s, i| {
        try out.appendSlice(gpa, &std.mem.toBytes(std.mem.nativeToLittle(u16, @intCast(i))));
        try out.appendNTimes(gpa, 0, 6);
        try out.appendSlice(gpa, &std.mem.toBytes(std.mem.nativeToLittle(u64, s.len)));
        try out.appendSlice(gpa, s);
        try pad8(gpa, out);
    }
    try out.appendNTimes(gpa, 0, 8);
    try out.appendSlice(gpa, &std.mem.toBytes(std.mem.nativeToLittle(u64, size - h.used)));
    try out.appendNTimes(gpa, 0, @intCast(size - h.used - 16));
    if (out.items.len != h.at + size) return error.BadLayout;
}

/// Write a node (its children first, then its data, then its header); returns the header's address.
fn emit(gpa: Allocator, out: *std.ArrayList(u8), node: *Node) !u64 {
    var msgs: std.ArrayList(u8) = .empty;
    defer msgs.deinit(gpa);
    if (node.value) |v| {
        var data_addr: u64 = UNDEF;
        if (v.bytes.len != 0) {
            try pad8(gpa, out);
            data_addr = out.items.len;
            try out.appendSlice(gpa, v.bytes);
        }
        const ds = try dataspace(gpa, v.dims);
        defer gpa.free(ds);
        try message(gpa, &msgs, 0x01, ds);
        const dt = try datatype(gpa, v.dtype);
        defer gpa.free(dt);
        try message(gpa, &msgs, 0x03, dt);
        // fill value, version 3: allocated late, written if set, none defined
        try message(gpa, &msgs, 0x05, &.{ 3, 0x0a });
        // layout, version 3: contiguous at the address, of the size
        var lay: [18]u8 = undefined;
        lay[0] = 3;
        lay[1] = 1;
        std.mem.writeInt(u64, lay[2..10], data_addr, .little);
        std.mem.writeInt(u64, lay[10..18], v.bytes.len, .little);
        try message(gpa, &msgs, 0x08, &lay);
    } else {
        const addrs = try gpa.alloc(u64, node.links.items.len);
        defer gpa.free(addrs);
        for (node.links.items, addrs) |l, *a| a.* = try emit(gpa, out, l.node);
        // link info, version 0: no creation order, no dense storage
        var li: [18]u8 = undefined;
        li[0] = 0;
        li[1] = 0;
        std.mem.writeInt(u64, li[2..10], UNDEF, .little);
        std.mem.writeInt(u64, li[10..18], UNDEF, .little);
        try message(gpa, &msgs, 0x02, &li);
        // group info, version 0, with the link phase change values: every link compact
        var gi: [6]u8 = undefined;
        gi[0] = 0;
        gi[1] = 1;
        std.mem.writeInt(u16, gi[2..4], @intCast(@max(8, @min(node.links.items.len, 65535))), .little);
        std.mem.writeInt(u16, gi[4..6], 6, .little);
        try message(gpa, &msgs, 0x0a, &gi);
        for (node.links.items, addrs) |l, a| {
            // link, version 1: a hard link; the name's length in one byte or two
            var m: std.ArrayList(u8) = .empty;
            defer m.deinit(gpa);
            if (l.name.len == 0 or l.name.len > 65535) return error.BadName;
            const two = l.name.len > 255;
            try m.appendSlice(gpa, &.{ 1, if (two) 0x01 else 0x00 });
            if (two) try m.appendSlice(gpa, &std.mem.toBytes(std.mem.nativeToLittle(u16, @intCast(l.name.len)))) else try m.append(gpa, @intCast(l.name.len));
            try m.appendSlice(gpa, l.name);
            try m.appendSlice(gpa, &std.mem.toBytes(std.mem.nativeToLittle(u64, a)));
            try message(gpa, &msgs, 0x06, m.items);
        }
    }
    for (node.attrs.items) |at| {
        // attribute, version 3: name (with its null), type, space, value; the name UTF-8
        const dt = try datatype(gpa, at.value.dtype);
        defer gpa.free(dt);
        const ds = try dataspace(gpa, at.value.dims);
        defer gpa.free(ds);
        var m: std.ArrayList(u8) = .empty;
        defer m.deinit(gpa);
        try m.appendSlice(gpa, &.{ 3, 0 });
        for ([_]usize{ at.name.len + 1, dt.len, ds.len }) |n| try m.appendSlice(gpa, &std.mem.toBytes(std.mem.nativeToLittle(u16, @intCast(n))));
        try m.append(gpa, 1);
        try m.appendSlice(gpa, at.name);
        try m.append(gpa, 0);
        try m.appendSlice(gpa, dt);
        try m.appendSlice(gpa, ds);
        try m.appendSlice(gpa, at.value.bytes);
        try message(gpa, &msgs, 0x0c, m.items);
    }
    // object header, version 2: signature, version, flags (the chunk's size in 4 bytes),
    // the chunk's size, its messages, checksum
    try pad8(gpa, out);
    const addr = out.items.len;
    const start = out.items.len;
    try out.appendSlice(gpa, "OHDR");
    try out.appendSlice(gpa, &.{ 2, 0x02 });
    try out.appendSlice(gpa, &std.mem.toBytes(std.mem.nativeToLittle(u32, @intCast(msgs.items.len))));
    try out.appendSlice(gpa, msgs.items);
    const sum = lookup3(out.items[start..], 0);
    try out.appendSlice(gpa, &std.mem.toBytes(std.mem.nativeToLittle(u32, sum)));
    return addr;
}

/// A version 2 header message: type, size, flags, body (messages of these types are never
/// shared and need no flags).
fn message(gpa: Allocator, msgs: *std.ArrayList(u8), kind: u8, body: []const u8) !void {
    if (body.len > 65535) return error.MessageTooLarge;
    try msgs.append(gpa, kind);
    try msgs.appendSlice(gpa, &std.mem.toBytes(std.mem.nativeToLittle(u16, @intCast(body.len))));
    try msgs.append(gpa, 0);
    try msgs.appendSlice(gpa, body);
}

/// A dataspace message, version 2: a scalar, or a simple space of these dimensions.
fn dataspace(gpa: Allocator, dims: []const u64) ![]u8 {
    var m: std.ArrayList(u8) = .empty;
    errdefer m.deinit(gpa);
    try m.appendSlice(gpa, &.{ 2, @intCast(dims.len), 0, if (dims.len == 0) 0 else 1 });
    for (dims) |d| try m.appendSlice(gpa, &std.mem.toBytes(std.mem.nativeToLittle(u64, d)));
    return m.toOwnedSlice(gpa);
}

/// A datatype message, version 1, little-endian.
fn datatype(gpa: Allocator, t: Type) ![]u8 {
    var m: std.ArrayList(u8) = .empty;
    errdefer m.deinit(gpa);
    const cls: u8 = switch (t.class) {
        .uint, .int => 0,
        .float => 1,
        .string => 3,
        .vstring => 9,
    };
    const bits: [3]u8 = switch (t.class) {
        .uint => .{ 0, 0, 0 },
        .int => .{ 0x08, 0, 0 },
        // IEEE: mantissa normalized with an implied bit; the sign's bit
        .float => .{ 0x20, @intCast(8 * t.size - 1), 0 },
        // null padded, UTF-8
        .string => .{ 0x11, 0, 0 },
        // a string, null terminated, UTF-8
        .vstring => .{ 0x01, 0x01, 0 },
    };
    try m.appendSlice(gpa, &.{ 0x10 | cls, bits[0], bits[1], bits[2] });
    try m.appendSlice(gpa, &std.mem.toBytes(std.mem.nativeToLittle(u32, t.size)));
    switch (t.class) {
        .uint, .int => {
            try m.appendSlice(gpa, &std.mem.toBytes(std.mem.nativeToLittle(u16, 0)));
            try m.appendSlice(gpa, &std.mem.toBytes(std.mem.nativeToLittle(u16, @intCast(8 * t.size))));
        },
        .float => {
            const double = t.size == 8;
            try m.appendSlice(gpa, &std.mem.toBytes(std.mem.nativeToLittle(u16, 0)));
            try m.appendSlice(gpa, &std.mem.toBytes(std.mem.nativeToLittle(u16, @intCast(8 * t.size))));
            // exponent's place and size, mantissa's place and size, exponent bias
            try m.appendSlice(gpa, if (double) &[_]u8{ 52, 11, 0, 52 } else &[_]u8{ 23, 8, 0, 23 });
            try m.appendSlice(gpa, &std.mem.toBytes(std.mem.nativeToLittle(u32, if (double) 1023 else 127)));
        },
        .string => {},
        // of bytes (an unsigned byte, as libhdf5 has a variable-length string's characters)
        .vstring => try m.appendSlice(gpa, &.{ 0x10, 0, 0, 0, 1, 0, 0, 0, 0, 0, 8, 0 }),
    }
    return m.toOwnedSlice(gpa);
}

/// Bob Jenkins' lookup3 hash (hashlittle), HDF5's checksum of its metadata.
pub fn lookup3(key: []const u8, init: u32) u32 {
    const rot = std.math.rotl;
    var a: u32 = 0xdeadbeef +% @as(u32, @truncate(key.len)) +% init;
    var b = a;
    var c = a;
    var k = key;
    while (k.len > 12) {
        a +%= std.mem.readInt(u32, k[0..4], .little);
        b +%= std.mem.readInt(u32, k[4..8], .little);
        c +%= std.mem.readInt(u32, k[8..12], .little);
        // mix
        a -%= c;
        a ^= rot(u32, c, 4);
        c +%= b;
        b -%= a;
        b ^= rot(u32, a, 6);
        a +%= c;
        c -%= b;
        c ^= rot(u32, b, 8);
        b +%= a;
        a -%= c;
        a ^= rot(u32, c, 16);
        c +%= b;
        b -%= a;
        b ^= rot(u32, a, 19);
        a +%= c;
        c -%= b;
        c ^= rot(u32, b, 4);
        b +%= a;
        k = k[12..];
    }
    if (k.len == 0) return c;
    // the last block, zero-padded (the missing bytes add nothing)
    var last: [12]u8 = @splat(0);
    @memcpy(last[0..k.len], k);
    a +%= std.mem.readInt(u32, last[0..4], .little);
    b +%= std.mem.readInt(u32, last[4..8], .little);
    c +%= std.mem.readInt(u32, last[8..12], .little);
    // final
    c ^= b;
    c -%= rot(u32, b, 14);
    a ^= c;
    a -%= rot(u32, c, 11);
    b ^= a;
    b -%= rot(u32, a, 25);
    c ^= b;
    c -%= rot(u32, b, 16);
    a ^= c;
    a -%= rot(u32, c, 4);
    b ^= a;
    b -%= rot(u32, a, 14);
    c ^= b;
    c -%= rot(u32, b, 24);
    return c;
}

test "lookup3 as Jenkins publishes it" {
    try std.testing.expectEqual(@as(u32, 0xdeadbeef), lookup3("", 0));
    try std.testing.expectEqual(@as(u32, 0x17770551), lookup3("Four score and seven years ago", 0));
    try std.testing.expectEqual(@as(u32, 0xcd628161), lookup3("Four score and seven years ago", 1));
}

test "variable-length strings, as h5py and Oxford's files store them, read back" {
    const hdf5 = @import("hdf5.zig");
    const io_file = @import("file.zig");
    const testing = std.testing;
    const gpa = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var f = try File.init(gpa);
    defer f.deinit();
    const g = try f.path("a/b");
    _ = try f.text(g, "scalar", "zebsdsolve");
    _ = try f.text(g, "empty", "");
    const d = try f.dataset(g, "one", .{ .dtype = Type.vstr, .dims = &[_]u64{1}, .bytes = "Zr" });
    try f.attribute(d, "Unit", .{ .dtype = Type.vstr, .bytes = "rad" });
    try testing.expectEqualStrings("rad", f.textOf(d.attrs.items[0].value).?);
    try testing.expectError(error.BadLength, f.dataset(g, "short", .{ .dtype = Type.vstr, .dims = &[_]u64{2}, .bytes = "x" }));
    // more strings than one collection holds (65535)
    const n = 70000;
    var many: std.ArrayList(u8) = .empty;
    defer many.deinit(gpa);
    for (0..n) |i| {
        if (i != 0) try many.append(gpa, 0);
        try many.print(gpa, "s{d}", .{i});
    }
    _ = try f.dataset(g, "many", .{ .dtype = Type.vstr, .dims = &[_]u64{n}, .bytes = many.items });
    try testing.expectEqual(@as(usize, 2), f.heaps.items.len);
    const bytes = try f.bytes(gpa);
    defer gpa.free(bytes);
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "s.h5", .data = bytes });
    const path = try tmp.dir.realPathFileAlloc(testing.io, "s.h5", gpa);
    defer gpa.free(path);
    var rf = try io_file.File.open(gpa, path);
    defer rf.close();
    var h = try hdf5.Hdf5.open(gpa, &rf);
    var buf: [64]u8 = undefined;
    try testing.expectEqualStrings("zebsdsolve", try h.string((try h.find("a/b/scalar")).?, &buf));
    try testing.expectEqualStrings("", try h.string((try h.find("a/b/empty")).?, &buf));
    try testing.expectEqualStrings("Zr", try h.string((try h.find("a/b/one")).?, &buf));
    const attrs = try h.attributes((try h.find("a/b/one")).?);
    defer {
        for (attrs) |*a| a.deinit(gpa);
        gpa.free(attrs);
    }
    try testing.expect(attrs.len == 1 and attrs[0].vlen);
    try testing.expectEqualStrings("rad", std.mem.sliceTo(attrs[0].data, 0));
    var ds = try h.dataset((try h.find("a/b/many")).?);
    defer ds.deinit(gpa);
    const raw = try gpa.alloc(u8, n * 16);
    defer gpa.free(raw);
    try h.readAll(&ds, raw);
    try testing.expectEqualStrings("s0", try h.vlenString(raw[0..16], &buf));
    // (in the second collection)
    try testing.expectEqualStrings("s69999", try h.vlenString(raw[(n - 1) * 16 ..][0..16], &buf));
}
