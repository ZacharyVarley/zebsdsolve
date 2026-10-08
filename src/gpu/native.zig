//! Native backend: webgpu.h, served by wgpu-native loaded at run time (std.DynLib).
//!
//! Objects are addressed by small integer handles (the same scheme the browser adapter uses),
//! so the scheduling code above never sees a backend type. Dispatches are recorded into one
//! command encoder (inside one compute pass while consecutive) until `submit`. Readbacks copy
//! into a staging buffer in the same encoder and map after the submit; `wait` polls the device
//! until every requested read has landed in its destination.
const std = @import("std");
const c = @import("webgpu_c");
const api = @import("api.zig");
const DynLib = @import("dynlib.zig").Lib;

const Allocator = std.mem.Allocator;

// ── the webgpu.h functions we call, resolved by name ─────────────────────────────────────────
const DevicePollFn = *const fn (c.WGPUDevice, c.WGPUBool, ?*const c.WGPUSubmissionIndex) callconv(.c) c.WGPUBool;

const Procs = struct {
    CreateInstance: c.WGPUProcCreateInstance,
    InstanceRequestAdapter: c.WGPUProcInstanceRequestAdapter,
    InstanceProcessEvents: c.WGPUProcInstanceProcessEvents,
    InstanceRelease: c.WGPUProcInstanceRelease,
    AdapterRequestDevice: c.WGPUProcAdapterRequestDevice,
    AdapterGetLimits: c.WGPUProcAdapterGetLimits,
    AdapterGetInfo: c.WGPUProcAdapterGetInfo,
    AdapterInfoFreeMembers: c.WGPUProcAdapterInfoFreeMembers,
    AdapterRelease: c.WGPUProcAdapterRelease,
    AdapterHasFeature: c.WGPUProcAdapterHasFeature,
    DeviceGetQueue: c.WGPUProcDeviceGetQueue,
    DeviceGetLimits: c.WGPUProcDeviceGetLimits,
    DeviceCreateBuffer: c.WGPUProcDeviceCreateBuffer,
    DeviceCreateShaderModule: c.WGPUProcDeviceCreateShaderModule,
    DeviceCreateComputePipeline: c.WGPUProcDeviceCreateComputePipeline,
    DeviceCreateBindGroup: c.WGPUProcDeviceCreateBindGroup,
    DeviceCreateCommandEncoder: c.WGPUProcDeviceCreateCommandEncoder,
    DeviceRelease: c.WGPUProcDeviceRelease,
    ComputePipelineGetBindGroupLayout: c.WGPUProcComputePipelineGetBindGroupLayout,
    ComputePipelineRelease: c.WGPUProcComputePipelineRelease,
    BindGroupLayoutRelease: c.WGPUProcBindGroupLayoutRelease,
    BindGroupRelease: c.WGPUProcBindGroupRelease,
    ShaderModuleRelease: c.WGPUProcShaderModuleRelease,
    BufferDestroy: c.WGPUProcBufferDestroy,
    BufferRelease: c.WGPUProcBufferRelease,
    BufferMapAsync: c.WGPUProcBufferMapAsync,
    BufferGetConstMappedRange: c.WGPUProcBufferGetConstMappedRange,
    BufferUnmap: c.WGPUProcBufferUnmap,
    QueueWriteBuffer: c.WGPUProcQueueWriteBuffer,
    QueueSubmit: c.WGPUProcQueueSubmit,
    QueueRelease: c.WGPUProcQueueRelease,
    CommandEncoderBeginComputePass: c.WGPUProcCommandEncoderBeginComputePass,
    CommandEncoderCopyBufferToBuffer: c.WGPUProcCommandEncoderCopyBufferToBuffer,
    CommandEncoderClearBuffer: c.WGPUProcCommandEncoderClearBuffer,
    CommandEncoderFinish: c.WGPUProcCommandEncoderFinish,
    CommandEncoderRelease: c.WGPUProcCommandEncoderRelease,
    CommandBufferRelease: c.WGPUProcCommandBufferRelease,
    ComputePassEncoderSetPipeline: c.WGPUProcComputePassEncoderSetPipeline,
    ComputePassEncoderSetBindGroup: c.WGPUProcComputePassEncoderSetBindGroup,
    ComputePassEncoderDispatchWorkgroups: c.WGPUProcComputePassEncoderDispatchWorkgroups,
    ComputePassEncoderDispatchWorkgroupsIndirect: c.WGPUProcComputePassEncoderDispatchWorkgroupsIndirect,
    ComputePassEncoderEnd: c.WGPUProcComputePassEncoderEnd,
    ComputePassEncoderRelease: c.WGPUProcComputePassEncoderRelease,
    DevicePoll: DevicePollFn,
    DeviceCreateQuerySet: c.WGPUProcDeviceCreateQuerySet,
    CommandEncoderResolveQuerySet: c.WGPUProcCommandEncoderResolveQuerySet,
    QuerySetRelease: c.WGPUProcQuerySetRelease,
    QueueGetTimestampPeriod: *const fn (c.WGPUQueue) callconv(.c) f32,
};

/// Timestamp queries per submit while profiling (two per dispatch).
const PROF_QUERIES = 4096; // wgpu's maximum query set size

/// GPU time of one pipeline, summed over the dispatches since profiling was last read.
pub const ProfEntry = api.ProfEntry;

fn loadProcs(lib: *DynLib) error{MissingSymbol}!Procs {
    var p: Procs = undefined;
    inline for (@typeInfo(Procs).@"struct".fields) |f| {
        const T = @typeInfo(f.type);
        const Fn = if (T == .optional) T.optional.child else f.type;
        const ptr = lib.lookup(Fn, "wgpu" ++ f.name) orelse return error.MissingSymbol;
        @field(p, f.name) = ptr;
    }
    return p;
}

fn sv(s: []const u8) c.WGPUStringView {
    return .{ .data = s.ptr, .length = s.len };
}

fn svSlice(v: c.WGPUStringView) []const u8 {
    const d = v.data orelse return "";
    if (v.length == std.math.maxInt(usize)) return std.mem.span(@as([*:0]const u8, @ptrCast(d)));
    return @as([*]const u8, @ptrCast(d))[0..v.length];
}

const Pending = struct {
    staging: c.WGPUBuffer,
    size: usize,
    dst: [*]u8,
    state: enum { recorded, mapping, done, failed } = .recorded,
};

const ProfBatch = struct { ticks: []u64, pipes: []u32 };

const Pipe = struct {
    pipeline: c.WGPUComputePipeline,
    layout: c.WGPUBindGroupLayout,
};

pub const Device = struct {
    gpa: Allocator,
    lib: DynLib,
    p: Procs,
    instance: c.WGPUInstance,
    adapter: c.WGPUAdapter,
    device: c.WGPUDevice,
    queue: c.WGPUQueue,
    lim: api.Limits,
    info_buf: [256]u8 = undefined,
    info_len: usize = 0,

    buffers: std.ArrayList(c.WGPUBuffer) = .empty,
    free_ids: std.ArrayList(u32) = .empty,
    modules: std.AutoHashMapUnmanaged(u64, c.WGPUShaderModule) = .empty,
    pipes: std.ArrayList(Pipe) = .empty,

    encoder: c.WGPUCommandEncoder = null,
    pass: c.WGPUComputePassEncoder = null,
    pending: std.ArrayList(*Pending) = .empty, // heap entries: map callbacks keep their address

    err_buf: [1024]u8 = undefined,
    err_len: usize = 0,
    /// errors reported so far, and the latest one's message
    err_count: u64 = 0,
    last_buf: [1024]u8 = undefined,
    last_len: usize = 0,
    /// the first error since createPipeline began
    since_buf: [2048]u8 = undefined,
    since_len: usize = 0,
    lost: bool = false,

    /// profiling (setProfile): every dispatch in its own pass with timestamps
    has_ts: bool = false,
    prof_on: bool = false,
    qset: c.WGPUQuerySet = null,
    qbuf: c.WGPUBuffer = null,
    /// pipelines of the timed dispatches in the open batch
    prof_pipes: std.ArrayList(u32) = .empty,
    /// submitted batches whose timestamps land at the next wait
    prof_batches: std.ArrayList(ProfBatch) = .empty,
    prof_totals: std.AutoArrayHashMapUnmanaged(u32, ProfEntry) = .empty,

    /// Open wgpu-native from `lib_path` (a file, or a directory holding the platform's library)
    /// and create a high-performance device with the adapter's full limits.
    pub fn init(gpa: Allocator, lib_path: []const u8) !*Device {
        const self = try gpa.create(Device);
        errdefer gpa.destroy(self);
        self.* = .{
            .gpa = gpa,
            .lib = undefined,
            .p = undefined,
            .instance = null,
            .adapter = null,
            .device = null,
            .queue = null,
            .lim = undefined,
        };
        self.lib = try openLib(gpa, lib_path);
        errdefer self.lib.close();
        self.p = try loadProcs(&self.lib);

        self.instance = self.p.CreateInstance.?(null) orelse return error.NoInstance;

        // adapter
        const AdapterWait = struct {
            done: bool = false,
            adapter: c.WGPUAdapter = null,
            fn cb(status: c.WGPURequestAdapterStatus, a: c.WGPUAdapter, msg: c.WGPUStringView, ud1: ?*anyopaque, ud2: ?*anyopaque) callconv(.c) void {
                _ = msg;
                _ = ud2;
                const w: *@This() = @ptrCast(@alignCast(ud1.?));
                if (status == c.WGPURequestAdapterStatus_Success) w.adapter = a;
                w.done = true;
            }
        };
        var aw: AdapterWait = .{};
        var aopt = std.mem.zeroes(c.WGPURequestAdapterOptions);
        aopt.powerPreference = c.WGPUPowerPreference_HighPerformance;
        var acb = std.mem.zeroes(c.WGPURequestAdapterCallbackInfo);
        acb.mode = c.WGPUCallbackMode_AllowProcessEvents;
        acb.callback = AdapterWait.cb;
        acb.userdata1 = &aw;
        _ = self.p.InstanceRequestAdapter.?(self.instance, &aopt, acb);
        while (!aw.done) self.p.InstanceProcessEvents.?(self.instance);
        self.adapter = aw.adapter orelse return error.NoAdapter;

        var ainfo = std.mem.zeroes(c.WGPUAdapterInfo);
        _ = self.p.AdapterGetInfo.?(self.adapter, &ainfo);
        const txt = std.fmt.bufPrint(&self.info_buf, "{s} ({s}, backend {d})", .{
            svSlice(ainfo.device), svSlice(ainfo.description), ainfo.backendType,
        }) catch self.info_buf[0..0];
        self.info_len = txt.len;
        self.p.AdapterInfoFreeMembers.?(ainfo);

        // device: ask for everything the adapter offers (as the browser app does for the
        // storage-binding, buffer and workgroup-memory limits the maps need).
        var alim = std.mem.zeroes(c.WGPULimits);
        _ = self.p.AdapterGetLimits.?(self.adapter, &alim);
        const DeviceWait = struct {
            done: bool = false,
            device: c.WGPUDevice = null,
            msg: [256]u8 = undefined,
            msg_len: usize = 0,
            fn cb(status: c.WGPURequestDeviceStatus, d: c.WGPUDevice, msg: c.WGPUStringView, ud1: ?*anyopaque, ud2: ?*anyopaque) callconv(.c) void {
                _ = ud2;
                const w: *@This() = @ptrCast(@alignCast(ud1.?));
                if (status == c.WGPURequestDeviceStatus_Success) w.device = d else {
                    const m = svSlice(msg);
                    w.msg_len = @min(m.len, w.msg.len);
                    @memcpy(w.msg[0..w.msg_len], m[0..w.msg_len]);
                }
                w.done = true;
            }
        };
        var dw: DeviceWait = .{};
        var ddesc = std.mem.zeroes(c.WGPUDeviceDescriptor);
        ddesc.requiredLimits = &alim;
        const has_f16 = self.p.AdapterHasFeature.?(self.adapter, c.WGPUFeatureName_ShaderF16) != 0;
        self.has_ts = self.p.AdapterHasFeature.?(self.adapter, c.WGPUFeatureName_TimestampQuery) != 0;
        var feats: [2]c.WGPUFeatureName = undefined;
        var nf: usize = 0;
        if (has_f16) {
            feats[nf] = c.WGPUFeatureName_ShaderF16;
            nf += 1;
        }
        if (self.has_ts) {
            feats[nf] = c.WGPUFeatureName_TimestampQuery;
            nf += 1;
        }
        ddesc.requiredFeatureCount = nf;
        ddesc.requiredFeatures = &feats;
        ddesc.uncapturedErrorCallbackInfo.callback = onError;
        ddesc.uncapturedErrorCallbackInfo.userdata1 = self;
        ddesc.deviceLostCallbackInfo.mode = c.WGPUCallbackMode_AllowProcessEvents;
        ddesc.deviceLostCallbackInfo.callback = onLost;
        ddesc.deviceLostCallbackInfo.userdata1 = self;
        var dcb = std.mem.zeroes(c.WGPURequestDeviceCallbackInfo);
        dcb.mode = c.WGPUCallbackMode_AllowProcessEvents;
        dcb.callback = DeviceWait.cb;
        dcb.userdata1 = &dw;
        _ = self.p.AdapterRequestDevice.?(self.adapter, &ddesc, dcb);
        while (!dw.done) self.p.InstanceProcessEvents.?(self.instance);
        self.device = dw.device orelse {
            self.setError(dw.msg[0..dw.msg_len]);
            return error.NoDevice;
        };
        self.queue = self.p.DeviceGetQueue.?(self.device);

        var dlim = std.mem.zeroes(c.WGPULimits);
        _ = self.p.DeviceGetLimits.?(self.device, &dlim);
        self.lim = .{
            .max_storage_binding = dlim.maxStorageBufferBindingSize,
            .max_buffer = dlim.maxBufferSize,
            .max_workgroup_storage = dlim.maxComputeWorkgroupStorageSize,
            .max_storage_per_stage = dlim.maxStorageBuffersPerShaderStage,
            .uniform_align = dlim.minUniformBufferOffsetAlignment,
            .storage_align = dlim.minStorageBufferOffsetAlignment,
            .has_f16 = @intFromBool(has_f16),
            // naga compiles dot4U8Packed (checked against wgpu-native 29)
            .packed_dot = 1,
        };
        try self.buffers.append(gpa, null); // handle 0 = none
        try self.pipes.append(gpa, .{ .pipeline = null, .layout = null });
        return self;
    }

    /// `path` is the library itself or the directory that holds it.
    fn openLib(gpa: Allocator, path: []const u8) !DynLib {
        const name = switch (@import("builtin").os.tag) {
            .windows => "wgpu_native.dll",
            .macos => "libwgpu_native.dylib",
            else => "libwgpu_native.so",
        };
        if (std.mem.endsWith(u8, path, name)) return DynLib.open(gpa, path);
        const joined = try std.fs.path.join(gpa, &.{ path, name });
        defer gpa.free(joined);
        return DynLib.open(gpa, joined);
    }

    fn onError(dev: [*c]const c.WGPUDevice, kind: c.WGPUErrorType, msg: c.WGPUStringView, ud1: ?*anyopaque, ud2: ?*anyopaque) callconv(.c) void {
        _ = dev;
        _ = kind;
        _ = ud2;
        const self: *Device = @ptrCast(@alignCast(ud1.?));
        self.setError(svSlice(msg));
    }

    fn onLost(dev: [*c]const c.WGPUDevice, reason: c.WGPUDeviceLostReason, msg: c.WGPUStringView, ud1: ?*anyopaque, ud2: ?*anyopaque) callconv(.c) void {
        _ = dev;
        _ = ud2;
        const self: *Device = @ptrCast(@alignCast(ud1.?));
        if (reason != c.WGPUDeviceLostReason_Destroyed and reason != c.WGPUDeviceLostReason_CallbackCancelled) {
            self.lost = true;
            self.setError(svSlice(msg));
        }
    }

    fn setError(self: *Device, m: []const u8) void {
        self.err_count += 1;
        if (self.since_len == 0) {
            self.since_len = @min(m.len, self.since_buf.len);
            @memcpy(self.since_buf[0..self.since_len], m[0..self.since_len]);
        }
        self.last_len = @min(m.len, self.last_buf.len);
        @memcpy(self.last_buf[0..self.last_len], m[0..self.last_len]);
        if (self.err_len != 0) return; // keep the first error
        self.err_len = @min(m.len, self.err_buf.len);
        @memcpy(self.err_buf[0..self.err_len], m[0..self.err_len]);
    }

    /// The latest GPU error's message (empty if none yet).
    pub fn lastError(self: *const Device) []const u8 {
        return self.last_buf[0..self.last_len];
    }

    /// The first error of the last createPipeline (the shader module's, if it failed first).
    pub fn failMessage(self: *const Device) []const u8 {
        return self.since_buf[0..self.since_len];
    }

    /// The first GPU error since the last call, if any.
    pub fn takeError(self: *Device) ?[]const u8 {
        if (self.err_len == 0) return null;
        const m = self.err_buf[0..self.err_len];
        self.err_len = 0;
        return m;
    }

    pub fn info(self: *Device) []const u8 {
        return self.info_buf[0..self.info_len];
    }

    pub fn limits(self: *Device) api.Limits {
        return self.lim;
    }

    pub fn deinit(self: *Device) void {
        if (self.qset != null) self.p.QuerySetRelease.?(self.qset);
        if (self.qbuf != null) {
            self.p.BufferDestroy.?(self.qbuf);
            self.p.BufferRelease.?(self.qbuf);
        }
        self.prof_pipes.deinit(self.gpa);
        for (self.prof_batches.items) |b| self.freeBatch(b);
        self.prof_batches.deinit(self.gpa);
        self.prof_totals.deinit(self.gpa);
        if (self.pass != null) self.p.ComputePassEncoderRelease.?(self.pass);
        if (self.encoder != null) self.p.CommandEncoderRelease.?(self.encoder);
        for (self.pending.items) |r| {
            self.p.BufferDestroy.?(r.staging);
            self.p.BufferRelease.?(r.staging);
            self.gpa.destroy(r);
        }
        self.pending.deinit(self.gpa);
        for (self.pipes.items) |pp| {
            if (pp.layout != null) self.p.BindGroupLayoutRelease.?(pp.layout);
            if (pp.pipeline != null) self.p.ComputePipelineRelease.?(pp.pipeline);
        }
        self.pipes.deinit(self.gpa);
        var it = self.modules.valueIterator();
        while (it.next()) |m| self.p.ShaderModuleRelease.?(m.*);
        self.modules.deinit(self.gpa);
        for (self.buffers.items) |b| if (b != null) {
            self.p.BufferDestroy.?(b);
            self.p.BufferRelease.?(b);
        };
        self.buffers.deinit(self.gpa);
        self.free_ids.deinit(self.gpa);
        if (self.queue != null) self.p.QueueRelease.?(self.queue);
        if (self.device != null) self.p.DeviceRelease.?(self.device);
        if (self.adapter != null) self.p.AdapterRelease.?(self.adapter);
        if (self.instance != null) self.p.InstanceRelease.?(self.instance);
        // wgpu-native stays loaded: GPU drivers keep threads alive past the instance, and
        // unloading their code under them crashes the process at exit.
        self.gpa.destroy(self);
    }

    // ── buffers ──
    pub fn createBuffer(self: *Device, size: u64, usage: api.Usage) u32 {
        var d = std.mem.zeroes(c.WGPUBufferDescriptor);
        d.size = size;
        d.usage = switch (usage) {
            .storage => c.WGPUBufferUsage_Storage | c.WGPUBufferUsage_CopySrc | c.WGPUBufferUsage_CopyDst | c.WGPUBufferUsage_Indirect,
            .uniform => c.WGPUBufferUsage_Uniform | c.WGPUBufferUsage_CopyDst,
        };
        const b = self.p.DeviceCreateBuffer.?(self.device, &d);
        if (self.free_ids.pop()) |id| {
            self.buffers.items[id] = b;
            return id;
        }
        self.buffers.append(self.gpa, b) catch return 0;
        return @intCast(self.buffers.items.len - 1);
    }

    pub fn destroyBuffer(self: *Device, id: u32) void {
        if (id == 0 or id >= self.buffers.items.len) return;
        const b = self.buffers.items[id];
        if (b == null) return;
        // Destroy waits for queued work that uses it (webgpu.h semantics), so this is safe mid-batch
        // only after submit; callers release between operations.
        self.p.BufferDestroy.?(b);
        self.p.BufferRelease.?(b);
        self.buffers.items[id] = null;
        self.free_ids.append(self.gpa, id) catch {};
    }

    pub fn writeBuffer(self: *Device, id: u32, offset: u64, data: []const u8) void {
        self.p.QueueWriteBuffer.?(self.queue, self.buffers.items[id], offset, data.ptr, data.len);
    }

    // ── pipelines ──
    pub fn createPipeline(self: *Device, code: []const u8, entry: []const u8) !u32 {
        const key = std.hash.Wyhash.hash(0, code);
        const before = self.err_count;
        self.since_len = 0;
        const gop = try self.modules.getOrPut(self.gpa, key);
        if (!gop.found_existing) {
            var src = std.mem.zeroes(c.WGPUShaderSourceWGSL);
            src.chain.sType = c.WGPUSType_ShaderSourceWGSL;
            src.code = sv(code);
            var md = std.mem.zeroes(c.WGPUShaderModuleDescriptor);
            md.nextInChain = @ptrCast(&src.chain);
            gop.value_ptr.* = self.p.DeviceCreateShaderModule.?(self.device, &md);
        }
        var pd = std.mem.zeroes(c.WGPUComputePipelineDescriptor);
        pd.label = sv(entry);
        pd.compute.module = gop.value_ptr.*;
        pd.compute.entryPoint = sv(entry);
        const pl = self.p.DeviceCreateComputePipeline.?(self.device, &pd);
        if (pl == null or self.err_count != before) {
            // wgpu's message (the shader compiler's, or a validation error), for CI logs
            std.log.err("pipeline {s}: {s}", .{ entry, self.failMessage() });
            return error.PipelineFailed;
        }
        const layout = self.p.ComputePipelineGetBindGroupLayout.?(pl, 0);
        try self.pipes.append(self.gpa, .{ .pipeline = pl, .layout = layout });
        return @intCast(self.pipes.items.len - 1);
    }

    // ── recording ──
    fn enc(self: *Device) c.WGPUCommandEncoder {
        if (self.encoder == null) self.encoder = self.p.DeviceCreateCommandEncoder.?(self.device, null);
        return self.encoder;
    }

    fn endPass(self: *Device) void {
        if (self.pass == null) return;
        self.p.ComputePassEncoderEnd.?(self.pass);
        self.p.ComputePassEncoderRelease.?(self.pass);
        self.pass = null;
    }

    // ── profiling ──
    /// Time every dispatch on the GPU from now on (false: stop). Returns false when the adapter
    /// has no timestamp queries.
    pub fn setProfile(self: *Device, on: bool) bool {
        if (on and !self.has_ts) return false;
        if (on and self.qset == null) {
            var qd = std.mem.zeroes(c.WGPUQuerySetDescriptor);
            qd.type = c.WGPUQueryType_Timestamp;
            qd.count = PROF_QUERIES;
            self.qset = self.p.DeviceCreateQuerySet.?(self.device, &qd);
            var bd = std.mem.zeroes(c.WGPUBufferDescriptor);
            bd.size = PROF_QUERIES * 8;
            bd.usage = c.WGPUBufferUsage_QueryResolve | c.WGPUBufferUsage_CopySrc;
            self.qbuf = self.p.DeviceCreateBuffer.?(self.device, &bd);
        }
        self.prof_on = on;
        return true;
    }

    /// GPU time per pipeline since the last call (after a wait), then reset.
    pub fn takeProfile(self: *Device, out: []ProfEntry) usize {
        const n = @min(out.len, self.prof_totals.count());
        @memcpy(out[0..n], self.prof_totals.values()[0..n]);
        self.prof_totals.clearRetainingCapacity();
        return n;
    }

    fn freeBatch(self: *Device, b: ProfBatch) void {
        self.gpa.free(b.ticks);
        self.gpa.free(b.pipes);
    }

    /// Resolve the open batch's timestamps into a readback that lands at the next wait.
    fn resolveProfile(self: *Device) void {
        const n: u32 = @intCast(self.prof_pipes.items.len);
        if (n == 0) return;
        defer self.prof_pipes.clearRetainingCapacity();
        const ticks = self.gpa.alloc(u64, 2 * n) catch return;
        const pipes = self.gpa.dupe(u32, self.prof_pipes.items) catch {
            self.gpa.free(ticks);
            return;
        };
        const e = self.enc();
        self.p.CommandEncoderResolveQuerySet.?(e, self.qset, 0, 2 * n, self.qbuf, 0);
        var d = std.mem.zeroes(c.WGPUBufferDescriptor);
        d.size = 16 * @as(u64, n);
        d.usage = c.WGPUBufferUsage_MapRead | c.WGPUBufferUsage_CopyDst;
        const st = self.p.DeviceCreateBuffer.?(self.device, &d);
        self.p.CommandEncoderCopyBufferToBuffer.?(e, self.qbuf, 0, st, 0, d.size);
        const r = self.gpa.create(Pending) catch return;
        r.* = .{ .staging = st, .size = @intCast(d.size), .dst = @ptrCast(ticks.ptr) };
        self.pending.append(self.gpa, r) catch self.gpa.destroy(r);
        self.prof_batches.append(self.gpa, .{ .ticks = ticks, .pipes = pipes }) catch self.freeBatch(.{ .ticks = ticks, .pipes = pipes });
    }

    /// Fold the landed batches into the per-pipeline totals.
    fn foldProfile(self: *Device) void {
        const period: f64 = self.p.QueueGetTimestampPeriod(self.queue);
        for (self.prof_batches.items) |b| {
            for (b.pipes, 0..) |pipe, i| {
                const t0 = b.ticks[2 * i];
                const t1 = b.ticks[2 * i + 1];
                const gop = self.prof_totals.getOrPut(self.gpa, pipe) catch continue;
                if (!gop.found_existing) gop.value_ptr.* = .{ .pipe = pipe, .count = 0, .ns = 0 };
                gop.value_ptr.count += 1;
                if (t1 > t0) gop.value_ptr.ns += @as(f64, @floatFromInt(t1 - t0)) * period;
            }
            self.freeBatch(b);
        }
        self.prof_batches.clearRetainingCapacity();
    }

    pub fn dispatch(self: *Device, pipe: u32, x: u32, y: u32, z: u32, binds: []const api.Bind) void {
        self.launch(pipe, .{ .direct = .{ x, y, z } }, binds);
    }

    /// A dispatch whose workgroup counts (3 × u32) the GPU reads from `buf` at byte `offset`.
    pub fn dispatchIndirect(self: *Device, pipe: u32, buf: u32, offset: u64, binds: []const api.Bind) void {
        self.launch(pipe, .{ .indirect = .{ .buf = buf, .offset = offset } }, binds);
    }

    const Launch = union(enum) { direct: [3]u32, indirect: struct { buf: u32, offset: u64 } };

    fn launch(self: *Device, pipe: u32, how: Launch, binds: []const api.Bind) void {
        const pp = self.pipes.items[pipe];
        const timed = self.prof_on and self.prof_pipes.items.len < PROF_QUERIES / 2;
        if (timed) {
            self.endPass();
            const k: u32 = @intCast(self.prof_pipes.items.len);
            var tw = std.mem.zeroes(c.WGPUPassTimestampWrites);
            tw.querySet = self.qset;
            tw.beginningOfPassWriteIndex = 2 * k;
            tw.endOfPassWriteIndex = 2 * k + 1;
            var pd = std.mem.zeroes(c.WGPUComputePassDescriptor);
            pd.timestampWrites = &tw;
            self.pass = self.p.CommandEncoderBeginComputePass.?(self.enc(), &pd);
            self.prof_pipes.append(self.gpa, pipe) catch {};
        }
        defer if (timed) self.endPass();
        if (self.pass == null) self.pass = self.p.CommandEncoderBeginComputePass.?(self.enc(), null);
        var entries: [16]c.WGPUBindGroupEntry = undefined;
        for (binds, 0..) |b, i| {
            entries[i] = std.mem.zeroes(c.WGPUBindGroupEntry);
            entries[i].binding = b.slot;
            entries[i].buffer = self.buffers.items[b.buf];
            entries[i].offset = b.offset;
            entries[i].size = if (b.size == 0) c.WGPU_WHOLE_SIZE else b.size;
        }
        var bd = std.mem.zeroes(c.WGPUBindGroupDescriptor);
        bd.layout = pp.layout;
        bd.entryCount = binds.len;
        bd.entries = &entries;
        const bg = self.p.DeviceCreateBindGroup.?(self.device, &bd);
        self.p.ComputePassEncoderSetPipeline.?(self.pass, pp.pipeline);
        self.p.ComputePassEncoderSetBindGroup.?(self.pass, 0, bg, 0, null);
        switch (how) {
            .direct => |n| self.p.ComputePassEncoderDispatchWorkgroups.?(self.pass, @max(n[0], 1), @max(n[1], 1), @max(n[2], 1)),
            .indirect => |ind| self.p.ComputePassEncoderDispatchWorkgroupsIndirect.?(self.pass, self.buffers.items[ind.buf], ind.offset),
        }
        self.p.BindGroupRelease.?(bg);
    }

    pub fn clearBuffer(self: *Device, id: u32, offset: u64, size: u64) void {
        self.endPass();
        self.p.CommandEncoderClearBuffer.?(self.enc(), self.buffers.items[id], offset, size);
    }

    pub fn copyBuffer(self: *Device, src: u32, soff: u64, dst: u32, doff: u64, size: u64) void {
        self.endPass();
        self.p.CommandEncoderCopyBufferToBuffer.?(self.enc(), self.buffers.items[src], soff, self.buffers.items[dst], doff, size);
    }

    /// Queue a copy of src[off..off+size] into host memory at `dst`; it lands during `wait`.
    pub fn readBuffer(self: *Device, src: u32, off: u64, size: u64, dst: [*]u8) void {
        self.endPass();
        var d = std.mem.zeroes(c.WGPUBufferDescriptor);
        d.size = (size + 3) & ~@as(u64, 3);
        d.usage = c.WGPUBufferUsage_MapRead | c.WGPUBufferUsage_CopyDst;
        const st = self.p.DeviceCreateBuffer.?(self.device, &d);
        self.p.CommandEncoderCopyBufferToBuffer.?(self.enc(), self.buffers.items[src], off, st, 0, d.size);
        const r = self.gpa.create(Pending) catch return;
        r.* = .{ .staging = st, .size = @intCast(size), .dst = dst };
        self.pending.append(self.gpa, r) catch self.gpa.destroy(r);
    }

    pub fn submit(self: *Device) void {
        self.endPass();
        if (self.prof_pipes.items.len > 0) self.resolveProfile();
        if (self.encoder != null) {
            const cb = self.p.CommandEncoderFinish.?(self.encoder, null);
            self.p.QueueSubmit.?(self.queue, 1, &cb);
            self.p.CommandBufferRelease.?(cb);
            self.p.CommandEncoderRelease.?(self.encoder);
            self.encoder = null;
        }
        for (self.pending.items) |r| {
            if (r.state != .recorded) continue;
            r.state = .mapping;
            var ci = std.mem.zeroes(c.WGPUBufferMapCallbackInfo);
            ci.mode = c.WGPUCallbackMode_AllowProcessEvents;
            ci.callback = onMapped;
            ci.userdata1 = r;
            _ = self.p.BufferMapAsync.?(r.staging, c.WGPUMapMode_Read, 0, (r.size + 3) & ~@as(usize, 3), ci);
        }
    }

    fn onMapped(status: c.WGPUMapAsyncStatus, msg: c.WGPUStringView, ud1: ?*anyopaque, ud2: ?*anyopaque) callconv(.c) void {
        _ = msg;
        _ = ud2;
        const r: *Pending = @ptrCast(@alignCast(ud1.?));
        r.state = if (status == c.WGPUMapAsyncStatus_Success) .done else .failed;
    }

    /// Block until every submitted read has been copied to its destination.
    pub fn wait(self: *Device) !void {
        self.submit();
        while (true) {
            var busy = false;
            for (self.pending.items) |r| {
                if (r.state == .mapping) busy = true;
            }
            if (!busy) break;
            _ = self.p.DevicePoll(self.device, 1, null);
            self.p.InstanceProcessEvents.?(self.instance);
        }
        var failed = false;
        for (self.pending.items) |r| {
            if (r.state == .done) {
                const m = self.p.BufferGetConstMappedRange.?(r.staging, 0, (r.size + 3) & ~@as(usize, 3));
                if (m) |ptr| @memcpy(r.dst[0..r.size], @as([*]const u8, @ptrCast(ptr))[0..r.size]);
                self.p.BufferUnmap.?(r.staging);
            } else failed = true;
            self.p.BufferDestroy.?(r.staging);
            self.p.BufferRelease.?(r.staging);
            self.gpa.destroy(r);
        }
        self.pending.clearRetainingCapacity();
        if (self.prof_batches.items.len > 0) self.foldProfile();
        if (self.lost) return error.DeviceLost;
        if (failed) return error.ReadFailed;
    }
};
