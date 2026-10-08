//! Browser module (wasm32-freestanding): the engine's exports for web/zebsdsolve.js. GPU calls
//! and file reads are imported from the adapter (src/gpu/web.zig, src/io/file.zig). Exports that
//! wait for the GPU or for a file suspend through JS Promise Integration; the adapter calls
//! them through WebAssembly.promising. The shared calls are in exports.zig; memory and creation
//! are here.
const std = @import("std");
const gpu = @import("gpu/gpu.zig");
const engine = @import("engine.zig");

comptime {
    _ = @import("exports.zig");
}

const Engine = engine.Engine;
const gpa = std.heap.wasm_allocator;

extern "env" fn ze_log(ptr: [*]const u8, len: usize) void;

pub const std_options: std.Options = .{ .logFn = log };

fn log(comptime level: std.log.Level, comptime scope: @EnumLiteral(), comptime fmt: []const u8, args: anytype) void {
    _ = scope;
    var buf: [512]u8 = undefined;
    const m = std.fmt.bufPrint(&buf, "[" ++ @tagName(level) ++ "] " ++ fmt, args) catch return;
    ze_log(m.ptr, m.len);
}

pub fn panic(msg: []const u8, _: ?*std.builtin.StackTrace, _: ?usize) noreturn {
    ze_log(msg.ptr, msg.len);
    @trap();
}

export fn ze_version() [*:0]const u8 {
    return engine.version;
}

export fn ze_alloc(n: usize) ?[*]u8 {
    const s = gpa.alloc(u8, n) catch return null;
    return s.ptr;
}

export fn ze_free(p: [*]u8, n: usize) void {
    gpa.free(p[0..n]);
}

export fn ze_create() ?*Engine {
    const dev = gpu.Backend.Device.init(gpa) catch return null;
    return Engine.create(gpa, dev) catch null;
}

export fn ze_destroy(e: *Engine) void {
    e.destroy();
}
