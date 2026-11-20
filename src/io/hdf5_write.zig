//! A writer for the part of HDF5 that results need: a new file of groups, datasets of
//! little-endian numbers and fixed-length strings, stored contiguously, and attributes of the
//! same, laid out as HDF5 1.8 and later read it: superblock version 2, version 2 object headers,
//! groups as link messages in their headers. Written against the file format specification, as
//! io/hdf5.zig reads it, so the engine needs no HDF5 library to write either. The whole file is
//! built in memory (results are maps, not patterns).
const std = @import("std");

const Allocator = std.mem.Allocator;
const UNDEF: u64 = std.math.maxInt(u64);

pub const Class = enum { uint, int, float, string };

/// An element type: a number of 1, 2, 4 or 8 bytes (floats 4 or 8), or a string of `size` bytes
/// (UTF-8, padded with nulls).
pub const Type = struct {
    class: Class,
    size: u32,

    pub const u8_: Type = .{ .class = .uint, .size = 1 };
    pub const i32_: Type = .{ .class = .int, .size = 4 };
    pub const u32_: Type = .{ .class = .uint, .size = 4 };
    pub const f32_: Type = .{ .class = .float, .size = 4 };
    pub const f64_: Type = .{ .class = .float, .size = 8 };

    pub fn str(len: usize) Type {
        return .{ .class = .string, .size = @intCast(@max(len, 1)) };
    }

    fn valid(self: Type) bool {
        return switch (self.class) {
            .uint, .int => self.size == 1 or self.size == 2 or self.size == 4 or self.size == 8,
            .float => self.size == 4 or self.size == 8,
            .string => self.size >= 1,
        };
    }
};

/// A value: its type, its shape (empty: a scalar) and its bytes, little-endian, row-major.
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

pub const File = struct {
    arena: std.heap.ArenaAllocator,
    root: *Node,

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
        if (!v.dtype.valid()) return error.UnsupportedType;
        if (v.bytes.len != v.count() * v.dtype.size) return error.BadLength;
        if (find(parent, name) != null) return error.NameTaken;
        const a = self.arena.allocator();
        const n = try a.create(Node);
        n.* = .{ .value = .{ .dtype = v.dtype, .dims = try a.dupe(u64, v.dims), .bytes = try a.dupe(u8, v.bytes) } };
        try parent.links.append(a, .{ .name = try a.dupe(u8, name), .node = n });
        return n;
    }

    /// A string dataset (one string, a scalar).
    pub fn text(self: *File, parent: *Node, name: []const u8, s: []const u8) !*Node {
        return self.dataset(parent, name, .{ .dtype = Type.str(s.len), .bytes = if (s.len == 0) "\x00" else s });
    }

    /// An attribute of a group or a dataset, its value copied.
    pub fn attribute(self: *File, node: *Node, name: []const u8, v: Value) !void {
        if (!v.dtype.valid()) return error.UnsupportedType;
        if (v.bytes.len != v.count() * v.dtype.size) return error.BadLength;
        const a = self.arena.allocator();
        try node.attrs.append(a, .{ .name = try a.dupe(u8, name), .value = .{ .dtype = v.dtype, .dims = try a.dupe(u64, v.dims), .bytes = try a.dupe(u8, v.bytes) } });
    }

    /// The file's bytes (caller frees).
    pub fn bytes(self: *File, gpa: Allocator) ![]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(gpa);
        try out.appendNTimes(gpa, 0, 48);
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
        try message(gpa, &msgs, 0x01, try dataspace(gpa, v.dims));
        try message(gpa, &msgs, 0x03, try datatype(gpa, v.dtype));
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
    };
    const bits: [3]u8 = switch (t.class) {
        .uint => .{ 0, 0, 0 },
        .int => .{ 0x08, 0, 0 },
        // IEEE: mantissa normalized with an implied bit; the sign's bit
        .float => .{ 0x20, @intCast(8 * t.size - 1), 0 },
        // null padded, UTF-8
        .string => .{ 0x11, 0, 0 },
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
