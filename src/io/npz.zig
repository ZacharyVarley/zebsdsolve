//! NumPy .npz archives (a ZIP of .npy arrays, stored or deflated), read through `File`.
const std = @import("std");
const File = @import("file.zig").File;
const codec = @import("codec.zig");

const Allocator = std.mem.Allocator;

pub const Array = struct {
    /// 'f' float, 'i' signed, 'u' unsigned, 'b' bool, 'S' bytes
    kind: u8 = 0,
    /// bytes per element
    size: u8 = 0,
    rank: u8 = 0,
    shape: [8]usize = @splat(1),
    fortran: bool = false,
    big_endian: bool = false,
    data: []u8 = &.{},
    /// the allocation `data` is a tail of
    raw: []u8 = &.{},

    pub fn deinit(self: *Array, gpa: Allocator) void {
        gpa.free(self.raw);
        self.* = .{};
    }

    pub fn count(self: *const Array) usize {
        var n: usize = 1;
        for (self.shape[0..self.rank]) |d| n *= d;
        return n;
    }

    pub fn at(self: *const Array, i: usize) f64 {
        const b = self.data[i * self.size ..][0..self.size];
        const e: std.builtin.Endian = if (self.big_endian) .big else .little;
        return switch (self.kind) {
            'f' => if (self.size == 4) @as(f32, @bitCast(std.mem.readInt(u32, b[0..4], e))) else @bitCast(std.mem.readInt(u64, b[0..8], e)),
            'i' => switch (self.size) {
                1 => @floatFromInt(@as(i8, @bitCast(b[0]))),
                2 => @floatFromInt(std.mem.readInt(i16, b[0..2], e)),
                4 => @floatFromInt(std.mem.readInt(i32, b[0..4], e)),
                else => @floatFromInt(std.mem.readInt(i64, b[0..8], e)),
            },
            else => switch (self.size) {
                1 => @floatFromInt(b[0]),
                2 => @floatFromInt(std.mem.readInt(u16, b[0..2], e)),
                4 => @floatFromInt(std.mem.readInt(u32, b[0..4], e)),
                else => @floatFromInt(std.mem.readInt(u64, b[0..8], e)),
            },
        };
    }
};

const Entry = struct { name: []u8, method: u16, comp: u64, size: u64, local: u64 };

pub const Npz = struct {
    gpa: Allocator,
    f: *File,
    entries: []Entry,

    pub fn open(gpa: Allocator, f: *File) !Npz {
        // the end-of-central-directory record is in the last 64 KiB
        const tail_len: usize = @intCast(@min(f.size, 65536 + 22));
        const tail = try f.alloc(gpa, f.size - tail_len, tail_len);
        defer gpa.free(tail);
        var at: ?usize = null;
        var i = tail_len;
        while (i >= 22) {
            i -= 1;
            if (i + 22 <= tail_len and std.mem.readInt(u32, tail[i..][0..4], .little) == 0x06054b50) {
                at = i;
                break;
            }
        }
        const eocd = at orelse return error.NotAZip;
        var n: u64 = std.mem.readInt(u16, tail[eocd + 10 ..][0..2], .little);
        var cd_size: u64 = std.mem.readInt(u32, tail[eocd + 12 ..][0..4], .little);
        var cd_off: u64 = std.mem.readInt(u32, tail[eocd + 16 ..][0..4], .little);
        if (n == 0xffff or cd_off == 0xffffffff) {
            // ZIP64: the locator sits just before the record
            if (eocd < 20 or std.mem.readInt(u32, tail[eocd - 20 ..][0..4], .little) != 0x07064b50) return error.BadZip;
            const z64 = std.mem.readInt(u64, tail[eocd - 12 ..][0..8], .little);
            n = try f.int(u64, z64 + 32);
            cd_size = try f.int(u64, z64 + 40);
            cd_off = try f.int(u64, z64 + 48);
        }
        const cd = try f.alloc(gpa, cd_off, @intCast(cd_size));
        defer gpa.free(cd);
        var list: std.ArrayList(Entry) = .empty;
        errdefer {
            for (list.items) |e| gpa.free(e.name);
            list.deinit(gpa);
        }
        var p: usize = 0;
        var k: u64 = 0;
        while (k < n and p + 46 <= cd.len) : (k += 1) {
            if (std.mem.readInt(u32, cd[p..][0..4], .little) != 0x02014b50) return error.BadZip;
            var e: Entry = .{
                .name = undefined,
                .method = std.mem.readInt(u16, cd[p + 10 ..][0..2], .little),
                .comp = std.mem.readInt(u32, cd[p + 20 ..][0..4], .little),
                .size = std.mem.readInt(u32, cd[p + 24 ..][0..4], .little),
                .local = std.mem.readInt(u32, cd[p + 42 ..][0..4], .little),
            };
            const name_len: usize = std.mem.readInt(u16, cd[p + 28 ..][0..2], .little);
            const extra_len: usize = std.mem.readInt(u16, cd[p + 30 ..][0..2], .little);
            const comment_len: usize = std.mem.readInt(u16, cd[p + 32 ..][0..2], .little);
            // ZIP64 extra field: the 64-bit values of whichever fields are saturated, in order
            var x = p + 46 + name_len;
            const x_end = x + extra_len;
            while (x + 4 <= x_end) {
                const id = std.mem.readInt(u16, cd[x..][0..2], .little);
                const len: usize = std.mem.readInt(u16, cd[x + 2 ..][0..2], .little);
                if (id == 1) {
                    var q = x + 4;
                    if (e.size == 0xffffffff) {
                        e.size = std.mem.readInt(u64, cd[q..][0..8], .little);
                        q += 8;
                    }
                    if (e.comp == 0xffffffff) {
                        e.comp = std.mem.readInt(u64, cd[q..][0..8], .little);
                        q += 8;
                    }
                    if (e.local == 0xffffffff) e.local = std.mem.readInt(u64, cd[q..][0..8], .little);
                }
                x += 4 + len;
            }
            e.name = try gpa.dupe(u8, cd[p + 46 ..][0..name_len]);
            list.append(gpa, e) catch |err| {
                gpa.free(e.name);
                return err;
            };
            p += 46 + name_len + extra_len + comment_len;
        }
        return .{ .gpa = gpa, .f = f, .entries = try list.toOwnedSlice(gpa) };
    }

    pub fn deinit(self: *Npz) void {
        for (self.entries) |e| self.gpa.free(e.name);
        self.gpa.free(self.entries);
    }

    pub fn has(self: *const Npz, name: []const u8) bool {
        return self.find(name) != null;
    }

    fn find(self: *const Npz, name: []const u8) ?Entry {
        for (self.entries) |e| {
            if (e.name.len == name.len + 4 and std.mem.startsWith(u8, e.name, name) and std.mem.endsWith(u8, e.name, ".npy")) return e;
        }
        return null;
    }

    /// The array stored as `name`.npy (caller deinits), or null.
    pub fn array(self: *Npz, name: []const u8) !?Array {
        const e = self.find(name) orelse return null;
        const name_len = try self.f.int(u16, e.local + 26);
        const extra_len = try self.f.int(u16, e.local + 28);
        const start = e.local + 30 + name_len + extra_len;
        const raw = try self.gpa.alloc(u8, @intCast(e.size));
        errdefer self.gpa.free(raw);
        switch (e.method) {
            0 => try self.f.read(start, raw),
            8 => {
                const comp = try self.f.alloc(self.gpa, start, @intCast(e.comp));
                defer self.gpa.free(comp);
                try codec.inflate(comp, raw, true);
            },
            else => return error.UnsupportedZipMethod,
        }
        var a = try parseNpy(raw);
        a.raw = raw;
        return a;
    }
};

fn parseNpy(raw: []u8) !Array {
    if (raw.len < 10 or !std.mem.eql(u8, raw[0..6], "\x93NUMPY")) return error.NotNpy;
    const hl: usize, const hs: usize = if (raw[6] == 1) .{ std.mem.readInt(u16, raw[8..10], .little), 10 } else .{ std.mem.readInt(u32, raw[8..12], .little), 12 };
    if (hs + hl > raw.len) return error.NotNpy;
    const head = raw[hs .. hs + hl];
    var a: Array = .{};
    // 'descr': '<f4'
    const d = std.mem.indexOf(u8, head, "'descr'") orelse return error.NotNpy;
    const q0 = (std.mem.indexOfScalarPos(u8, head, d + 7, '\'') orelse return error.NotNpy) + 1;
    const q1 = std.mem.indexOfScalarPos(u8, head, q0, '\'') orelse return error.NotNpy;
    var descr = head[q0..q1];
    if (descr.len >= 1 and (descr[0] == '<' or descr[0] == '>' or descr[0] == '|' or descr[0] == '=')) {
        a.big_endian = descr[0] == '>';
        descr = descr[1..];
    }
    if (descr.len < 2) return error.UnsupportedNpy;
    a.kind = descr[0];
    a.size = std.fmt.parseInt(u8, descr[1..], 10) catch return error.UnsupportedNpy;
    switch (a.kind) {
        'f', 'i', 'u', 'b', 'S' => {},
        else => return error.UnsupportedNpy,
    }
    a.fortran = std.mem.indexOf(u8, head, "'fortran_order': True") != null;
    const s = std.mem.indexOf(u8, head, "'shape'") orelse return error.NotNpy;
    const p0 = (std.mem.indexOfScalarPos(u8, head, s, '(') orelse return error.NotNpy) + 1;
    const p1 = std.mem.indexOfScalarPos(u8, head, p0, ')') orelse return error.NotNpy;
    var it = std.mem.tokenizeAny(u8, head[p0..p1], ", ");
    while (it.next()) |tok| {
        if (a.rank == a.shape.len) return error.UnsupportedNpy;
        a.shape[a.rank] = std.fmt.parseInt(usize, tok, 10) catch return error.NotNpy;
        a.rank += 1;
    }
    a.data = raw[hs + hl ..];
    const elem: usize = if (a.kind == 'S') 1 else a.size;
    if (a.kind == 'S') a.size = 1;
    if (a.data.len < a.count() * elem) return error.NotNpy;
    return a;
}

test "read the nickel master like numpy" {
    const gpa = std.testing.allocator;
    var f = File.open(gpa, "testdata/nickel/Ni_master_20kV.npz") catch return; // run from the repo root
    defer f.close();
    var z = try Npz.open(gpa, &f);
    defer z.deinit();
    try std.testing.expect(z.has("fundamental_sector") and !z.has("nope"));
    var fs = (try z.array("fundamental_sector")).?;
    defer fs.deinit(gpa);
    try std.testing.expectEqual(@as(u8, 3), fs.rank);
    try std.testing.expectEqual(@as(usize, 11017), fs.shape[2]);
    try std.testing.expect(fs.kind == 'f' and fs.size == 4);
    var kij = (try z.array("fundamental_kij")).?;
    defer kij.deinit(gpa);
    try std.testing.expectEqual(@as(f64, -2), kij.at(0));
    try std.testing.expectEqual(@as(f64, 1), kij.at(2));
    var ops = (try z.array("pg_operators")).?;
    defer ops.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 48), ops.shape[0]);
}
