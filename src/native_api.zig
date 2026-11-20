//! Native shared library: the C ABI the Python package calls through ctypes. The shared calls
//! are in exports.zig; creation (which opens wgpu-native) is here.
const std = @import("std");
pub const gpu = @import("gpu/gpu.zig");
pub const shaders = @import("shaders");
pub const engine = @import("engine.zig");

comptime {
    _ = @import("exports.zig");
}

const Engine = engine.Engine;
const gpa = std.heap.smp_allocator;

var create_err: [512]u8 = undefined;
var create_err_len: usize = 0;

export fn ze_version() [*:0]const u8 {
    return engine.version;
}

/// Open wgpu-native (`lib_path`: the library file or its directory) and create an engine.
/// Null on failure; ze_create_error says why.
export fn ze_create(lib_path: [*:0]const u8) ?*Engine {
    create_err_len = 0;
    const dev = gpu.Backend.Device.init(gpa, std.mem.span(lib_path)) catch |e| {
        const m = std.fmt.bufPrint(&create_err, "GPU init failed: {s}", .{@errorName(e)}) catch "GPU init failed";
        create_err_len = m.len;
        return null;
    };
    return Engine.create(gpa, dev) catch |e| {
        const m = std.fmt.bufPrint(&create_err, "engine init failed: {s}", .{@errorName(e)}) catch "engine init failed";
        create_err_len = m.len;
        dev.deinit();
        return null;
    };
}

export fn ze_create_error(buf: [*]u8, cap: usize) usize {
    const n = @min(create_err_len, cap);
    @memcpy(buf[0..n], create_err[0..n]);
    return n;
}

export fn ze_destroy(e: ?*Engine) void {
    const self = e orelse return;
    const dev = self.dev;
    self.destroy();
    dev.deinit();
}

/// Oxford CS1 export snapshot. A null buffer queries its size.
export fn ze_export_oxford_snapshot(e: *Engine, frame: ?[*]const f64, buf: ?[*]u8, cap: usize) i64 {
    const requested = if (frame) |p| p[0..4].* else null;
    const bytes = @import("oxford_snapshot.zig").snapshot(e, requested) catch |err| {
        e.fail("export_oxford_snapshot: {s}", .{@errorName(err)});
        return -1;
    };
    defer e.gpa.free(bytes);
    if (buf) |p| {
        if (cap < bytes.len) {
            e.fail("export_oxford_snapshot: BufferTooSmall", .{});
            return -1;
        }
        @memcpy(p[0..bytes.len], bytes);
    }
    return @intCast(bytes.len);
}
