//! Browser backend: every call is forwarded to web/zebsdsolve.js (import module "zegpu"), which
//! owns the GPUDevice and maps integer handles to GPU objects. Sizes and offsets cross as f64
//! (exact to 2^53) so the adapter never sees BigInts.
//!
//! `wait` is a suspending import (WebAssembly JavaScript Promise Integration): the module's
//! stack is parked while the adapter awaits the submitted reads, then resumes, so the schedule
//! is written as ordinary blocking code, exactly as natively.
const std = @import("std");
const api = @import("api.zig");

const js = struct {
    extern "zegpu" fn limits(out: *api.Limits) void;
    extern "zegpu" fn info(buf: [*]u8, cap: usize) usize;
    extern "zegpu" fn createBuffer(size: f64, usage: u32) u32;
    extern "zegpu" fn destroyBuffer(id: u32) void;
    extern "zegpu" fn writeBuffer(id: u32, offset: f64, ptr: [*]const u8, len: usize) void;
    extern "zegpu" fn createPipeline(code: [*]const u8, code_len: usize, entry: [*]const u8, entry_len: usize) u32;
    extern "zegpu" fn dispatch(pipe: u32, x: u32, y: u32, z: u32, binds: [*]const api.Bind, n: usize) void;
    extern "zegpu" fn dispatchIndirect(pipe: u32, buf: u32, offset: f64, binds: [*]const api.Bind, n: usize) void;
    extern "zegpu" fn clearBuffer(id: u32, offset: f64, size: f64) void;
    extern "zegpu" fn copyBuffer(src: u32, soff: f64, dst: u32, doff: f64, size: f64) void;
    extern "zegpu" fn readBuffer(src: u32, offset: f64, size: f64, dst: [*]u8) void;
    extern "zegpu" fn submit() void;
    extern "zegpu" fn takeError(buf: [*]u8, cap: usize) usize;
    /// suspends until every read requested so far has landed; 0 ok, else a failure code
    extern "zegpu" fn wait() i32;
};

pub const Device = struct {
    info_buf: [256]u8 = undefined,
    info_len: usize = 0,
    err_buf: [1024]u8 = undefined,

    pub fn init(gpa: std.mem.Allocator) !*Device {
        const self = try gpa.create(Device);
        self.* = .{};
        self.info_len = js.info(&self.info_buf, self.info_buf.len);
        return self;
    }

    pub fn info(self: *Device) []const u8 {
        return self.info_buf[0..self.info_len];
    }

    pub fn limits(self: *Device) api.Limits {
        _ = self;
        var l: api.Limits = undefined;
        js.limits(&l);
        return l;
    }

    pub fn takeError(self: *Device) ?[]const u8 {
        const n = js.takeError(&self.err_buf, self.err_buf.len);
        return if (n == 0) null else self.err_buf[0..n];
    }

    pub fn createBuffer(self: *Device, size: u64, usage: api.Usage) u32 {
        _ = self;
        return js.createBuffer(@floatFromInt(size), @intFromEnum(usage));
    }

    pub fn destroyBuffer(self: *Device, id: u32) void {
        _ = self;
        js.destroyBuffer(id);
    }

    pub fn writeBuffer(self: *Device, id: u32, offset: u64, data: []const u8) void {
        _ = self;
        js.writeBuffer(id, @floatFromInt(offset), data.ptr, data.len);
    }

    pub fn createPipeline(self: *Device, code: []const u8, entry: []const u8) !u32 {
        _ = self;
        const p = js.createPipeline(code.ptr, code.len, entry.ptr, entry.len);
        return if (p == 0) error.PipelineFailed else p;
    }

    pub fn dispatch(self: *Device, pipe: u32, x: u32, y: u32, z: u32, binds: []const api.Bind) void {
        _ = self;
        js.dispatch(pipe, x, y, z, binds.ptr, binds.len);
    }

    /// A dispatch whose workgroup counts (3 × u32) the GPU reads from `buf` at byte `offset`.
    pub fn dispatchIndirect(self: *Device, pipe: u32, buf: u32, offset: u64, binds: []const api.Bind) void {
        _ = self;
        js.dispatchIndirect(pipe, buf, @floatFromInt(offset), binds.ptr, binds.len);
    }

    pub fn clearBuffer(self: *Device, id: u32, offset: u64, size: u64) void {
        _ = self;
        js.clearBuffer(id, @floatFromInt(offset), @floatFromInt(size));
    }

    pub fn copyBuffer(self: *Device, src: u32, soff: u64, dst: u32, doff: u64, size: u64) void {
        _ = self;
        js.copyBuffer(src, @floatFromInt(soff), dst, @floatFromInt(doff), @floatFromInt(size));
    }

    pub fn readBuffer(self: *Device, src: u32, off: u64, size: u64, dst: [*]u8) void {
        _ = self;
        js.readBuffer(src, @floatFromInt(off), @floatFromInt(size), dst);
    }

    pub fn submit(self: *Device) void {
        _ = self;
        js.submit();
    }

    /// Per-dispatch GPU timing is not offered in the browser (the adapter's ze.stats() counts
    /// round trips instead).
    pub fn setProfile(self: *Device, on: bool) bool {
        _ = self;
        _ = on;
        return false;
    }

    pub fn takeProfile(self: *Device, out: []api.ProfEntry) usize {
        _ = self;
        _ = out;
        return 0;
    }

    /// Submit and block (suspend) until every requested read has been copied into memory.
    pub fn wait(self: *Device) !void {
        _ = self;
        js.submit();
        if (js.wait() != 0) return error.ReadFailed;
    }
};
