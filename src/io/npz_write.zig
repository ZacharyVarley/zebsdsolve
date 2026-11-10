//! Writing NumPy .npz archives: a ZIP of .npy arrays, stored (no compression), little-endian.
const std = @import("std");

const Allocator = std.mem.Allocator;

pub const Writer = struct {
    gpa: Allocator,
    out: std.ArrayList(u8) = .empty,
    dir: std.ArrayList(u8) = .empty,
    count: u16 = 0,

    pub fn init(gpa: Allocator) Writer {
        return .{ .gpa = gpa };
    }

    pub fn deinit(self: *Writer) void {
        self.out.deinit(self.gpa);
        self.dir.deinit(self.gpa);
    }

    /// Add array `name` (".npy" is appended) of numpy type `descr` ("<f4", "<i4", "<f8", "|u1")
    /// and `shape`, its elements' bytes in C order.
    pub fn add(self: *Writer, name: []const u8, descr: []const u8, shape: []const usize, bytes: []const u8) !void {
        const gpa = self.gpa;
        // the .npy: magic, version 1.0, a header padded with spaces to a multiple of 64
        var hdr: std.ArrayList(u8) = .empty;
        defer hdr.deinit(gpa);
        try hdr.print(gpa, "{{'descr': '{s}', 'fortran_order': False, 'shape': (", .{descr});
        for (shape, 0..) |d, i| try hdr.print(gpa, "{d}{s}", .{ d, if (shape.len == 1 or i + 1 < shape.len) ", " else "" });
        try hdr.appendSlice(gpa, "), }");
        const pre = 10;
        const total = std.mem.alignForward(usize, pre + hdr.items.len + 1, 64);
        while (pre + hdr.items.len + 1 < total) try hdr.append(gpa, ' ');
        try hdr.append(gpa, '\n');
        var npy: std.ArrayList(u8) = .empty;
        defer npy.deinit(gpa);
        try npy.appendSlice(gpa, "\x93NUMPY\x01\x00");
        try npy.appendSlice(gpa, &std.mem.toBytes(std.mem.nativeToLittle(u16, @intCast(hdr.items.len))));
        try npy.appendSlice(gpa, hdr.items);
        try npy.appendSlice(gpa, bytes);

        var fname_buf: [128]u8 = undefined;
        const fname = try std.fmt.bufPrint(&fname_buf, "{s}.npy", .{name});
        var crc = std.hash.Crc32.init();
        crc.update(npy.items);
        const c = crc.final();
        const size: u32 = @intCast(npy.items.len);
        const local: u32 = @intCast(self.out.items.len);
        // local file header
        try self.u32le(&self.out, 0x04034b50);
        try self.u16le(&self.out, 20);
        try self.u16le(&self.out, 0);
        try self.u16le(&self.out, 0);
        try self.u16le(&self.out, 0);
        try self.u16le(&self.out, 0x21);
        try self.u32le(&self.out, c);
        try self.u32le(&self.out, size);
        try self.u32le(&self.out, size);
        try self.u16le(&self.out, @intCast(fname.len));
        try self.u16le(&self.out, 0);
        try self.out.appendSlice(gpa, fname);
        try self.out.appendSlice(gpa, npy.items);
        // its central-directory entry
        try self.u32le(&self.dir, 0x02014b50);
        try self.u16le(&self.dir, 20);
        try self.u16le(&self.dir, 20);
        try self.u16le(&self.dir, 0);
        try self.u16le(&self.dir, 0);
        try self.u16le(&self.dir, 0);
        try self.u16le(&self.dir, 0x21);
        try self.u32le(&self.dir, c);
        try self.u32le(&self.dir, size);
        try self.u32le(&self.dir, size);
        try self.u16le(&self.dir, @intCast(fname.len));
        try self.u16le(&self.dir, 0);
        try self.u16le(&self.dir, 0);
        try self.u16le(&self.dir, 0);
        try self.u16le(&self.dir, 0);
        try self.u32le(&self.dir, 0);
        try self.u32le(&self.dir, local);
        try self.dir.appendSlice(gpa, fname);
        self.count += 1;
    }

    pub fn addSlice(self: *Writer, name: []const u8, comptime T: type, shape: []const usize, data: []const T) !void {
        const descr = switch (T) {
            f32 => "<f4",
            f64 => "<f8",
            i32 => "<i4",
            u32 => "<u4",
            u8 => "|u1",
            else => @compileError("npz: unsupported element type"),
        };
        try self.add(name, descr, shape, std.mem.sliceAsBytes(data));
    }

    /// The archive's bytes (the caller owns them); the writer is spent.
    pub fn finish(self: *Writer) ![]u8 {
        const gpa = self.gpa;
        const at: u32 = @intCast(self.out.items.len);
        const dir_len: u32 = @intCast(self.dir.items.len);
        try self.out.appendSlice(gpa, self.dir.items);
        try self.u32le(&self.out, 0x06054b50);
        try self.u16le(&self.out, 0);
        try self.u16le(&self.out, 0);
        try self.u16le(&self.out, self.count);
        try self.u16le(&self.out, self.count);
        try self.u32le(&self.out, dir_len);
        try self.u32le(&self.out, at);
        try self.u16le(&self.out, 0);
        return self.out.toOwnedSlice(gpa);
    }

    fn u16le(self: *Writer, to: *std.ArrayList(u8), v: u16) !void {
        try to.appendSlice(self.gpa, &std.mem.toBytes(std.mem.nativeToLittle(u16, v)));
    }
    fn u32le(self: *Writer, to: *std.ArrayList(u8), v: u32) !void {
        try to.appendSlice(self.gpa, &std.mem.toBytes(std.mem.nativeToLittle(u32, v)));
    }
};
