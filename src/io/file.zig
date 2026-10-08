//! Random-access reads of large files, one interface over two backends, like the GPU layer:
//! natively the operating system (positioned reads, no libc on Windows); in the browser the
//! adapter (web/zebsdsolve.js) slices a `File` the user picked. A read is ordinary blocking
//! code: in the browser the module suspends on it (JS Promise Integration) while the slice
//! loads, so a 100 GB scan is read in pieces and never held in memory.
//!
//! Format readers go through `File.read`, which serves small requests from a block cache:
//! parsing an HDF5 tree makes thousands of tiny reads, and each uncached one is a round trip.
//! `File.readv` reads many ranges in one round trip, and `File.readSpans` also reads neighbours
//! as one range, gap and all: patterns scattered over a file (a preview lattice, the chunks of
//! an HDF5 dataset, what a sweep left unindexed) are asked for together. In Chrome a round trip
//! costs about 0.8 ms and each slice of a file, even in parallel, about 0.25 ms, while a large
//! slice reads at about a gigabyte per second: a slice costs what a quarter megabyte does, so
//! gaps up to half of that are read through.
const std = @import("std");
const builtin = @import("builtin");

pub const is_web = builtin.cpu.arch == .wasm32;
const Allocator = std.mem.Allocator;

const BLOCK = 64 * 1024;
const N_BLOCKS = 32;

/// Ranges closer than this are read as one: where reading the gap costs less than asking again.
pub const merge_gap: usize = if (is_web) 128 << 10 else 16 << 10;

const Block = struct {
    /// block index, or none
    index: u64 = std.math.maxInt(u64),
    /// last use
    stamp: u64 = 0,
    data: []u8 = &.{},
};

/// A byte range of a file.
pub const Range = struct { offset: u64, len: usize };

pub const File = struct {
    gpa: Allocator,
    h: Handle,
    size: u64,
    blocks: [N_BLOCKS]Block = @splat(.{}),
    clock: u64 = 0,
    /// bytes fetched from the backend and backend calls (for the front ends' statistics)
    bytes_read: u64 = 0,
    reads: u64 = 0,

    /// Open `name`: a path natively; in the browser the token the adapter gave a picked file.
    pub fn open(gpa: Allocator, name: []const u8) !File {
        const r = try backend.open(gpa, name);
        return .{ .gpa = gpa, .h = r.h, .size = r.size };
    }

    pub fn close(self: *File) void {
        for (&self.blocks) |*b| if (b.data.len != 0) self.gpa.free(b.data);
        backend.close(self.h);
        self.* = undefined;
    }

    /// Fill `dst` from `offset`; error.EndOfFile if the range leaves the file.
    pub fn read(self: *File, offset: u64, dst: []u8) !void {
        if (dst.len == 0) return;
        if (offset > self.size or dst.len > self.size - offset) return error.EndOfFile;
        if (dst.len >= BLOCK) return self.raw(offset, dst);
        var done: usize = 0;
        while (done < dst.len) {
            const pos = offset + done;
            const blk = try self.block(pos / BLOCK);
            const in: usize = @intCast(pos % BLOCK);
            const n = @min(dst.len - done, blk.len - in);
            @memcpy(dst[done..][0..n], blk[in..][0..n]);
            done += n;
        }
    }

    fn raw(self: *File, offset: u64, dst: []u8) !void {
        self.reads += 1;
        self.bytes_read += dst.len;
        try backend.read(self.h, offset, dst);
    }

    /// Fill `dst` with the ranges back to back, in one round trip (the cache is bypassed).
    pub fn readv(self: *File, ranges: []const Range, dst: []u8) !void {
        var total: usize = 0;
        for (ranges) |r| {
            if (r.offset > self.size or r.len > self.size - r.offset) return error.EndOfFile;
            total += r.len;
        }
        if (total != dst.len) return error.BadLength;
        if (ranges.len == 0) return;
        self.reads += 1;
        self.bytes_read += total;
        try backend.readv(self.gpa, self.h, ranges, dst);
    }

    /// Fill `dst` with the ranges back to back, like `readv`, the ranges being in ascending
    /// order and apart: neighbours closer than `merge_gap` are fetched as one range.
    pub fn readSpans(self: *File, ranges: []const Range, dst: []u8) !void {
        const gpa = self.gpa;
        var merged: std.ArrayList(Range) = .empty;
        defer merged.deinit(gpa);
        var total: usize = 0;
        var gaps = false;
        for (ranges) |r| {
            if (merged.items.len != 0) {
                const last = &merged.items[merged.items.len - 1];
                const end = last.offset + last.len;
                if (r.offset < end) return error.RangesNotAscending;
                if (r.offset - end <= merge_gap) {
                    const gap: usize = @intCast(r.offset - end);
                    if (gap != 0) gaps = true;
                    last.len += gap + r.len;
                    total += gap + r.len;
                    continue;
                }
            }
            try merged.append(gpa, r);
            total += r.len;
        }
        if (!gaps) return self.readv(merged.items, dst);
        const tmp = try gpa.alloc(u8, total);
        defer gpa.free(tmp);
        try self.readv(merged.items, tmp);
        var m: usize = 0;
        var base: usize = 0;
        var out: usize = 0;
        for (ranges) |r| {
            while (r.offset >= merged.items[m].offset + merged.items[m].len) {
                base += merged.items[m].len;
                m += 1;
            }
            @memcpy(dst[out..][0..r.len], tmp[base + @as(usize, @intCast(r.offset - merged.items[m].offset)) ..][0..r.len]);
            out += r.len;
        }
    }

    fn block(self: *File, index: u64) ![]const u8 {
        self.clock += 1;
        var oldest: usize = 0;
        for (&self.blocks, 0..) |*b, i| {
            if (b.index == index) {
                b.stamp = self.clock;
                return b.data[0..self.blockLen(index)];
            }
            if (b.stamp < self.blocks[oldest].stamp) oldest = i;
        }
        const b = &self.blocks[oldest];
        if (b.data.len == 0) b.data = try self.gpa.alloc(u8, BLOCK);
        b.index = std.math.maxInt(u64);
        const n = self.blockLen(index);
        try self.raw(index * BLOCK, b.data[0..n]);
        b.index = index;
        b.stamp = self.clock;
        return b.data[0..n];
    }

    fn blockLen(self: *const File, index: u64) usize {
        return @intCast(@min(BLOCK, self.size - index * BLOCK));
    }

    /// Little-endian integer at `offset`.
    pub fn int(self: *File, comptime T: type, offset: u64) !T {
        var b: [@sizeOf(T)]u8 = undefined;
        try self.read(offset, &b);
        return std.mem.readInt(T, &b, .little);
    }

    /// Little-endian unsigned integer of `size` bytes (1, 2, 4 or 8).
    pub fn uintN(self: *File, offset: u64, size: u8) !u64 {
        var b: [8]u8 = @splat(0);
        try self.read(offset, b[0..size]);
        return std.mem.readInt(u64, &b, .little);
    }

    /// Allocate and read [offset, offset + len).
    pub fn alloc(self: *File, gpa: Allocator, offset: u64, len: usize) ![]u8 {
        const out = try gpa.alloc(u8, len);
        errdefer gpa.free(out);
        try self.read(offset, out);
        return out;
    }
};

pub const Handle = backend.H;
const backend = if (is_web) web else if (builtin.os.tag == .windows) windows else posix;

const windows = struct {
    const H = *anyopaque;
    const OVERLAPPED = extern struct { Internal: usize = 0, InternalHigh: usize = 0, Offset: u32, OffsetHigh: u32, hEvent: ?*anyopaque = null };
    extern "kernel32" fn CreateFileW(name: [*:0]const u16, access: u32, share: u32, sec: ?*anyopaque, disp: u32, flags: u32, tmpl: ?*anyopaque) callconv(.winapi) ?*anyopaque;
    extern "kernel32" fn ReadFile(h: *anyopaque, buf: [*]u8, n: u32, got: ?*u32, ov: ?*OVERLAPPED) callconv(.winapi) i32;
    extern "kernel32" fn GetFileSizeEx(h: *anyopaque, size: *i64) callconv(.winapi) i32;
    extern "kernel32" fn CloseHandle(h: *anyopaque) callconv(.winapi) i32;
    const GENERIC_READ = 0x80000000;
    const SHARE_ALL = 0x7;
    const OPEN_EXISTING = 3;
    const INVALID: usize = std.math.maxInt(usize);

    fn open(gpa: Allocator, name: []const u8) !struct { h: H, size: u64 } {
        const w = try std.unicode.utf8ToUtf16LeAllocZ(gpa, name);
        defer gpa.free(w);
        const h = CreateFileW(w.ptr, GENERIC_READ, SHARE_ALL, null, OPEN_EXISTING, 0, null) orelse return error.FileNotFound;
        if (@intFromPtr(h) == INVALID) return error.FileNotFound;
        var size: i64 = 0;
        if (GetFileSizeEx(h, &size) == 0) {
            _ = CloseHandle(h);
            return error.ReadFailed;
        }
        return .{ .h = h, .size = @intCast(size) };
    }

    fn read(h: H, offset: u64, dst: []u8) !void {
        var done: usize = 0;
        while (done < dst.len) {
            const pos = offset + done;
            var ov: OVERLAPPED = .{ .Offset = @truncate(pos), .OffsetHigh = @truncate(pos >> 32) };
            var got: u32 = 0;
            const want: u32 = @intCast(@min(dst.len - done, 1 << 30));
            if (ReadFile(h, dst.ptr + done, want, &got, &ov) == 0 or got == 0) return error.ReadFailed;
            done += got;
        }
    }

    fn readv(gpa: Allocator, h: H, ranges: []const Range, dst: []u8) !void {
        _ = gpa;
        var at: usize = 0;
        for (ranges) |r| {
            try read(h, r.offset, dst[at..][0..r.len]);
            at += r.len;
        }
    }

    fn close(h: H) void {
        _ = CloseHandle(h);
    }
};

const posix = struct {
    const H = c_int;
    const c = struct {
        extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
        extern "c" fn pread(fd: c_int, buf: [*]u8, n: usize, off: i64) isize;
        extern "c" fn lseek(fd: c_int, off: i64, whence: c_int) i64;
        extern "c" fn close(fd: c_int) c_int;
    };

    fn open(gpa: Allocator, name: []const u8) !struct { h: H, size: u64 } {
        const z = try gpa.dupeZ(u8, name);
        defer gpa.free(z);
        const fd = c.open(z.ptr, 0);
        if (fd < 0) return error.FileNotFound;
        const size = c.lseek(fd, 0, 2);
        if (size < 0) {
            _ = c.close(fd);
            return error.ReadFailed;
        }
        return .{ .h = fd, .size = @intCast(size) };
    }

    fn read(h: H, offset: u64, dst: []u8) !void {
        var done: usize = 0;
        while (done < dst.len) {
            const got = c.pread(h, dst.ptr + done, dst.len - done, @intCast(offset + done));
            if (got <= 0) return error.ReadFailed;
            done += @intCast(got);
        }
    }

    fn readv(gpa: Allocator, h: H, ranges: []const Range, dst: []u8) !void {
        _ = gpa;
        var at: usize = 0;
        for (ranges) |r| {
            try read(h, r.offset, dst[at..][0..r.len]);
            at += r.len;
        }
    }

    fn close(h: H) void {
        _ = c.close(h);
    }
};

const web = struct {
    const H = u32;
    const js = struct {
        /// id of the file registered under this token, or 0
        extern "zeio" fn open(name: [*]const u8, len: usize) u32;
        extern "zeio" fn size(id: u32) f64;
        /// suspends while the slice loads; 0 on success
        extern "zeio" fn read(id: u32, offset: f64, dst: [*]u8, len: usize) i32;
        /// n slices (offsets as f64, lengths as u32) loaded together into dst back to back;
        /// suspends until all have; 0 on success
        extern "zeio" fn readv(id: u32, n: usize, offsets: [*]const f64, lens: [*]const u32, dst: [*]u8) i32;
        extern "zeio" fn close(id: u32) void;
    };

    fn open(gpa: Allocator, name: []const u8) !struct { h: H, size: u64 } {
        _ = gpa;
        const id = js.open(name.ptr, name.len);
        if (id == 0) return error.FileNotFound;
        return .{ .h = id, .size = @intFromFloat(js.size(id)) };
    }

    fn read(h: H, offset: u64, dst: []u8) !void {
        if (js.read(h, @floatFromInt(offset), dst.ptr, dst.len) != 0) return error.ReadFailed;
    }

    fn readv(gpa: Allocator, h: H, ranges: []const Range, dst: []u8) !void {
        const offs = try gpa.alloc(f64, ranges.len);
        defer gpa.free(offs);
        const lens = try gpa.alloc(u32, ranges.len);
        defer gpa.free(lens);
        for (ranges, 0..) |r, i| {
            offs[i] = @floatFromInt(r.offset);
            lens[i] = @intCast(r.len);
        }
        if (js.readv(h, ranges.len, offs.ptr, lens.ptr, dst.ptr) != 0) return error.ReadFailed;
    }

    fn close(h: H) void {
        js.close(h);
    }
};

test "block cache serves small and large reads" {
    if (is_web) return;
    const gpa = std.testing.allocator;
    var f = File.open(gpa, "testdata/nickel/patterns_v2.h5") catch return; // run from the repo root
    defer f.close();
    var sig: [8]u8 = undefined;
    try f.read(0, &sig);
    try std.testing.expectEqualSlices(u8, "\x89HDF\r\n\x1a\n", &sig);
    // a read spanning two cache blocks equals the direct read
    const a = try gpa.alloc(u8, 3000);
    defer gpa.free(a);
    const b = try gpa.alloc(u8, BLOCK + 3000);
    defer gpa.free(b);
    try f.read(BLOCK - 1500, a);
    try f.read(BLOCK - 1500, b);
    try std.testing.expectEqualSlices(u8, a, b[0..3000]);
    try std.testing.expectError(error.EndOfFile, f.read(f.size - 10, a));
    // ranges read together equal the same ranges read one by one
    const v = try gpa.alloc(u8, 3000);
    defer gpa.free(v);
    try f.readv(&.{ .{ .offset = BLOCK - 1500, .len = 1000 }, .{ .offset = 7, .len = 1 }, .{ .offset = BLOCK - 500, .len = 1999 } }, v);
    try std.testing.expectEqualSlices(u8, a[0..1000], v[0..1000]);
    try std.testing.expectEqual(sig[7], v[1000]);
    try std.testing.expectEqualSlices(u8, a[1000..2999], v[1001..3000]);
    // neighbours read as one range come out as the ranges themselves
    const w = try gpa.alloc(u8, 2500);
    defer gpa.free(w);
    try f.readSpans(&.{ .{ .offset = BLOCK - 1500, .len = 1000 }, .{ .offset = BLOCK - 400, .len = 500 }, .{ .offset = BLOCK + 500, .len = 1000 } }, w);
    try std.testing.expectEqualSlices(u8, a[0..1000], w[0..1000]);
    try std.testing.expectEqualSlices(u8, a[1100..1600], w[1000..1500]);
    try std.testing.expectEqualSlices(u8, a[2000..3000], w[1500..2500]);
    try std.testing.expectError(error.RangesNotAscending, f.readSpans(&.{ .{ .offset = 100, .len = 10 }, .{ .offset = 105, .len = 10 } }, w[0..20]));
}
