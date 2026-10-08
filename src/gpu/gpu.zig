//! The GPU layer the indexing code schedules through. One implementation above two
//! backends: native (webgpu.h via wgpu-native) and web (imports forwarded to browser WebGPU).
//!
//! Policies that live here, once, for both targets:
//!  * Buffers are handles; algorithm code allocates them once and reuses them (`Scratch`).
//!  * Every dispatch gets its own uniform slot from an arena that is uploaded with a single
//!    write just before the submit, so a whole schedule is one command buffer.
//!  * Pipelines compile once per (source, entry point) and are looked up by name.
//!  * Nothing reads back implicitly. `read` queues a copy to host memory; `wait` submits and
//!    blocks until it has landed (in the browser: suspends through JS Promise Integration).
//!    A readback costs a round trip (~3 ms in a browser), so schedules read rarely and batch.
//!  * `write` is ordered with recorded work: during an open batch it goes through the storage
//!    arena as a recorded copy (a plain queue write would land before the batch runs).
//!  * A dispatch over more than 65535 workgroups in a dimension is refused with an error that
//!    names the kernel (wgpu-native would abort the process); 1-D kernels over pixels fold
//!    into (x, y) (`flat`).
const std = @import("std");
const builtin = @import("builtin");
pub const api = @import("api.zig");

pub const is_web = builtin.cpu.arch == .wasm32;
pub const Backend = if (is_web) @import("web.zig") else @import("native.zig");
pub const Bind = api.Bind;
pub const Limits = api.Limits;

const Allocator = std.mem.Allocator;

/// A storage buffer handle with its size.
pub const Buf = struct {
    id: u32 = 0,
    size: u64 = 0,

    pub fn at(self: Buf, slot: u32) Bind {
        return .{ .slot = slot, .buf = self.id };
    }
    pub fn range(self: Buf, slot: u32, offset: u64, size: u64) Bind {
        return .{ .slot = slot, .buf = self.id, .offset = offset, .size = size };
    }
};

const UNIFORM_ARENA = 256 * 1024;
/// maxComputeWorkgroupsPerDimension (the WebGPU default every adapter meets)
pub const MAX_GROUPS = 65535;
const STORAGE_ARENA = 1024 * 1024;

/// Gated recording: dispatch slots per gate session.
pub const GATE_SLOTS = 4096;

/// A gated dispatch waits on the done flag only.
pub const NO_FLAG: u32 = 0xffffffff;

/// The pipeline index callers use for a pipeline that failed to build: its dispatches are
/// skipped and the next wait fails (error.PipelineFailed) instead of running another pipeline.
pub const NO_PIPE: u32 = 0xffffffff;

pub const Gate = struct {
    flag: Buf,
    flag_word: u32,
    /// the second flag word gating the dispatches recorded now (gateOn), or NO_FLAG
    also: u32 = NO_FLAG,
    /// slots recorded; the first `flushed` are already uploaded
    n: u32 = 0,
    flushed: u32 = 0,
};

const GATE_WGSL =
    \\@group(0) @binding(0) var<storage, read> base: array<u32>;
    \\@group(0) @binding(1) var<storage, read> flag: array<f32>;
    \\@group(0) @binding(2) var<storage, read_write> live: array<u32>;
    \\@group(0) @binding(3) var<uniform> u: vec4<u32>;
    \\// slot s (base: x, y, z, its own flag word) runs its recorded workgroups unless the done
    \\// flag (flag[u.y]) or its own flag is set (> 0.5)
    \\@compute @workgroup_size(256)
    \\fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    \\    let s = g.x;
    \\    if (s >= u.x) { return; }
    \\    let f = base[s * 4u + 3u];
    \\    let off = flag[u.y] > 0.5 || (f != 0xffffffffu && flag[f] > 0.5);
    \\    for (var c = 0u; c < 3u; c++) { live[s * 3u + c] = select(base[s * 4u + c], 0u, off); }
    \\}
;

/// GPU traffic counters: dispatches recorded, submits, reads queued and waits (round trips).
pub const Counts = extern struct { dispatches: u32 = 0, submits: u32 = 0, reads: u32 = 0, waits: u32 = 0 };

pub const Gpu = struct {
    gpa: Allocator,
    dev: *Backend.Device,
    lim: Limits,
    /// uniform arena: host staging, one GPU buffer, 256-byte slots
    ustage: []align(16) u8,
    ubuf: u32,
    uoff: usize = 0,
    ualign: usize,
    /// storage arena: small per-dispatch read-only data (pose matrices, lists)
    sstage: []align(16) u8,
    sbuf: u32,
    soff: usize = 0,
    salign: usize,
    /// buffers released since the last submit (destroyed after it)
    retired: std.ArrayList(u32) = .empty,
    pipes: std.StringHashMapUnmanaged(u32) = .empty,
    /// keys of `pipes` (owned)
    names: std.ArrayList([]u8) = .empty,
    /// the first dispatch skipped since the last wait because a dimension exceeded MAX_GROUPS
    /// (natively an invalid dispatch aborts the process, so it never reaches the device)
    oversize: ?struct { pipe: u32, n: [3]u32 } = null,
    /// a dispatch named a pipeline that failed to build (NO_PIPE) since the last wait
    bad_pipe: bool = false,
    /// work recorded since the last submit (writes then go through the arena, in order)
    open: bool = false,
    /// GPU traffic since the last countsTake (the browser adapter's ze.stats, natively)
    counts: Counts = .{},
    /// gated recording (beginGate … endGate)
    gate: ?Gate = null,
    gate_base: Buf = .{},
    gate_live: Buf = .{},
    gate_counts: std.ArrayList(u32) = .empty,

    pub fn init(gpa: Allocator, dev: *Backend.Device) !Gpu {
        const lim = dev.limits();
        const ustage = try gpa.alignedAlloc(u8, .@"16", UNIFORM_ARENA);
        const sstage = try gpa.alignedAlloc(u8, .@"16", STORAGE_ARENA);
        return .{
            .gpa = gpa,
            .dev = dev,
            .lim = lim,
            .ustage = ustage,
            .ubuf = dev.createBuffer(UNIFORM_ARENA, .uniform),
            .ualign = @max(@as(usize, lim.uniform_align), 16),
            .sstage = sstage,
            .sbuf = dev.createBuffer(STORAGE_ARENA, .storage),
            .salign = @max(@as(usize, lim.storage_align), 16),
        };
    }

    pub fn deinit(self: *Gpu) void {
        self.release(&self.gate_base);
        self.release(&self.gate_live);
        self.gate_counts.deinit(self.gpa);
        for (self.retired.items) |id| self.dev.destroyBuffer(id);
        self.retired.deinit(self.gpa);
        self.dev.destroyBuffer(self.ubuf);
        self.gpa.free(self.ustage);
        self.dev.destroyBuffer(self.sbuf);
        self.gpa.free(self.sstage);
        for (self.names.items) |n| self.gpa.free(n);
        self.names.deinit(self.gpa);
        self.pipes.deinit(self.gpa);
    }

    // ── buffers ──
    pub fn storage(self: *Gpu, bytes: u64) Buf {
        const size = std.mem.alignForward(u64, @max(bytes, 16), 16);
        return .{ .id = self.dev.createBuffer(size, .storage), .size = size };
    }

    /// Release a buffer. It is destroyed after the next submit, so work already recorded in the
    /// open batch may still use it.
    pub fn release(self: *Gpu, b: *Buf) void {
        if (b.id != 0) self.retired.append(self.gpa, b.id) catch self.dev.destroyBuffer(b.id);
        b.* = .{};
    }

    /// `b` with at least `bytes`; reallocated only when it must grow (contents not kept).
    pub fn ensure(self: *Gpu, b: *Buf, bytes: u64) void {
        if (b.id != 0 and b.size >= bytes) return;
        self.release(b);
        b.* = self.storage(bytes);
    }

    /// Write bytes into `b`, in order with the recorded work: queue writes take effect before
    /// the open batch runs, so while one is open the bytes go through the storage arena and a
    /// recorded copy (or, when they do not fit, the batch is submitted first).
    pub fn write(self: *Gpu, b: Buf, offset: u64, data: []const u8) void {
        if (!self.open or data.len == 0) return self.dev.writeBuffer(b.id, offset, data);
        const need = std.mem.alignForward(usize, data.len, 16);
        if (data.len % 4 != 0 or offset % 4 != 0 or need > self.sstage.len / 4) {
            self.submit();
            return self.dev.writeBuffer(b.id, offset, data);
        }
        if (self.soff + need > self.sstage.len) self.submit();
        const off = self.soff;
        @memcpy(self.sstage[off..][0..data.len], data);
        self.soff = std.mem.alignForward(usize, off + need, self.salign);
        self.dev.copyBuffer(self.sbuf, off, b.id, offset, data.len);
        self.open = true;
    }

    pub fn writeSlice(self: *Gpu, b: Buf, offset: u64, comptime T: type, data: []const T) void {
        self.write(b, offset, std.mem.sliceAsBytes(data));
    }

    pub fn clear(self: *Gpu, b: Buf) void {
        self.dev.clearBuffer(b.id, 0, b.size);
        self.open = true;
    }

    /// Zero bytes [offset, offset + size) (multiples of 4).
    pub fn clearRange(self: *Gpu, b: Buf, offset: u64, size: u64) void {
        self.dev.clearBuffer(b.id, offset, size);
        self.open = true;
    }

    /// Copy bytes; the size rounds up to 4 (WebGPU copies whole words; half-precision images
    /// of odd size end in a padding pair), within both buffers.
    pub fn copy(self: *Gpu, src: Buf, soff: u64, dst: Buf, doff: u64, size: u64) void {
        var n = std.mem.alignForward(u64, size, 4);
        if (src.size > 0 and dst.size > 0) n = @min(n, src.size - soff, dst.size - doff);
        self.dev.copyBuffer(src.id, soff, dst.id, doff, n);
        self.open = true;
    }

    /// Queue a read of `src[offset..]` into `dst`; valid after the next submit completes.
    pub fn read(self: *Gpu, src: Buf, offset: u64, dst: []u8) void {
        self.counts.reads += 1;
        self.dev.readBuffer(src.id, offset, dst.len, dst.ptr);
        self.open = true;
    }

    // ── pipelines ──
    /// Pipeline for `entry` of `code`, compiled on first use and cached under `key`
    /// (e.g. "so3/spectrum", "fft/line").
    pub fn pipeline(self: *Gpu, key: []const u8, code: []const u8, entry: []const u8) !u32 {
        if (self.pipes.get(key)) |p| return p;
        const p = try self.dev.createPipeline(code, entry);
        const owned = try self.gpa.dupe(u8, key);
        errdefer self.gpa.free(owned);
        try self.names.append(self.gpa, owned);
        try self.pipes.put(self.gpa, owned, p);
        return p;
    }

    /// Pipeline for `entry` of `body` with the line `const <constant>: u32 = <value>u;` before
    /// it: a kernel whose workgroup size is chosen at run time. Cached under key@value.
    pub fn pipelineWith(self: *Gpu, comptime key: []const u8, comptime constant: []const u8, value: u32, body: []const u8, entry: []const u8) !u32 {
        var kb: [96]u8 = undefined;
        const k = try std.fmt.bufPrint(&kb, key ++ "@{d}", .{value});
        if (self.pipes.get(k)) |p| return p;
        const code = try std.fmt.allocPrint(self.gpa, "const " ++ constant ++ ": u32 = {d}u;\n{s}", .{ value, body });
        defer self.gpa.free(code);
        return self.pipeline(k, code, entry);
    }

    // ── dispatch ──
    /// A uniform slot holding `bytes`, bound at `slot`. Flushes the batch if the arena is full.
    pub fn uniform(self: *Gpu, slot: u32, bytes: []const u8) Bind {
        const need = std.mem.alignForward(usize, bytes.len, 16);
        if (self.uoff + need > self.ustage.len) self.submit();
        const off = self.uoff;
        @memcpy(self.ustage[off..][0..bytes.len], bytes);
        @memset(self.ustage[off + bytes.len ..][0 .. need - bytes.len], 0);
        self.uoff = std.mem.alignForward(usize, off + need, self.ualign);
        return .{ .slot = slot, .buf = self.ubuf, .offset = off, .size = need };
    }

    /// Read-only data for one dispatch (e.g. a pose matrix), bound at `slot` from the storage
    /// arena: each call gets its own slot, so no buffer is needed per dispatch.
    pub fn stage(self: *Gpu, slot: u32, bytes: []const u8) Bind {
        const need = std.mem.alignForward(usize, @max(bytes.len, 16), 16);
        if (self.soff + need > self.sstage.len) self.submit();
        const off = self.soff;
        @memcpy(self.sstage[off..][0..bytes.len], bytes);
        @memset(self.sstage[off + bytes.len ..][0 .. need - bytes.len], 0);
        self.soff = std.mem.alignForward(usize, off + need, self.salign);
        return .{ .slot = slot, .buf = self.sbuf, .offset = off, .size = need };
    }

    /// A 1-D grid of `groups` workgroups as (x, y) within the per-dimension limit, for kernels
    /// that index gid.x + gid.y · num_workgroups.x · 256 (flat_index in the WGSL).
    pub fn flat(groups: u32) [2]u32 {
        if (groups <= MAX_GROUPS) return .{ @max(groups, 1), 1 };
        return .{ MAX_GROUPS, (groups + MAX_GROUPS - 1) / MAX_GROUPS };
    }

    /// Dispatch a flat_index kernel over `groups` workgroups in z layers.
    pub fn dispatchFlat(self: *Gpu, pipe: u32, groups: u32, z: u32, binds: []const Bind) void {
        const f = flat(groups);
        self.dispatch(pipe, f[0], f[1], z, binds);
    }

    pub fn dispatch(self: *Gpu, pipe: u32, x: u32, y: u32, z: u32, binds: []const Bind) void {
        self.counts.dispatches += 1;
        if (pipe == NO_PIPE) {
            self.bad_pipe = true;
            return;
        }
        if (x > MAX_GROUPS or y > MAX_GROUPS or z > MAX_GROUPS) {
            if (self.oversize == null) self.oversize = .{ .pipe = pipe, .n = .{ x, y, z } };
            return;
        }
        if (self.gate) |*gt| if (gt.n < GATE_SLOTS) {
            // gated: the GPU reads this dispatch's workgroup counts from its live slot
            self.gate_counts.appendSlice(self.gpa, &.{ @max(x, 1), @max(y, 1), @max(z, 1), gt.also }) catch {};
            const slot = gt.n;
            gt.n += 1;
            self.dev.dispatchIndirect(pipe, self.gate_live.id, @as(u64, slot) * 12, binds);
            self.open = true;
            return;
        };
        self.dev.dispatch(pipe, x, y, z, binds);
        self.open = true;
    }

    // ── gated recording ──
    // A long schedule recorded ahead (the GPU-resident climb) stops costing GPU work once a flag
    // the GPU itself sets says it is done: between beginGate and endGate every dispatch is
    // recorded indirect, its workgroup counts read from a live table that gateRefresh fills
    // from the recorded counts, or with zeros once the flag is set.

    /// Start gating on the f32 flag at word `flag_word` of `flag` (set: > 0.5).
    pub fn beginGate(self: *Gpu, flag: Buf, flag_word: u32) void {
        self.ensure(&self.gate_base, GATE_SLOTS * 16);
        self.ensure(&self.gate_live, GATE_SLOTS * 12);
        self.gate_counts.clearRetainingCapacity();
        self.gate = .{ .flag = flag, .flag_word = flag_word };
    }

    /// Gate the dispatches recorded from now on also on the flag at word `word` of the gate's
    /// buffer (null: on the done flag only).
    pub fn gateOn(self: *Gpu, word: ?u32) void {
        if (self.gate) |*gt| gt.also = word orelse NO_FLAG;
    }

    /// Record the refresh of the live table from the flags (call wherever a flag may have
    /// changed, before the gated dispatches that follow).
    pub fn gateRefresh(self: *Gpu) void {
        const gt = self.gate orelse return;
        const pipe = self.pipeline("gpu/gate", GATE_WGSL, "main") catch return;
        const u = [4]u32{ GATE_SLOTS, gt.flag_word, 0, 0 };
        self.gate = null; // the refresh itself is not gated
        defer self.gate = gt;
        self.dispatch(pipe, (GATE_SLOTS + 255) / 256, 1, 1, &.{ self.gate_base.at(0), gt.flag.at(1), self.gate_live.at(2), self.uniform(3, std.mem.asBytes(&u)) });
    }

    /// Record a dispatch outside the gate (direct) while gating: for kernels that read the gate's
    /// flags themselves and skip their work (gateWord says which flag besides the done flag).
    pub fn dispatchUngated(self: *Gpu, pipe: u32, x: u32, y: u32, z: u32, binds: []const Bind) void {
        const gt = self.gate;
        self.gate = null;
        defer self.gate = gt;
        self.dispatch(pipe, x, y, z, binds);
    }

    /// Pause gating (the dispatches until resumeGate run whatever the flags say); returns the
    /// session for resumeGate.
    pub fn pauseGate(self: *Gpu) ?Gate {
        const gt = self.gate;
        self.gate = null;
        return gt;
    }

    pub fn resumeGate(self: *Gpu, gt: ?Gate) void {
        self.gate = gt;
    }

    /// The flag word gating the dispatches recorded now besides the done flag (NO_FLAG: none,
    /// or not gating).
    pub fn gateWord(self: *const Gpu) u32 {
        return if (self.gate) |gt| gt.also else NO_FLAG;
    }

    /// Upload the counts of the slots recorded since the last upload (queue writes, which land
    /// before the work submitted after them).
    fn gateFlush(self: *Gpu) void {
        if (self.gate == null) return;
        const g = &self.gate.?;
        if (g.n > g.flushed) {
            self.dev.writeBuffer(self.gate_base.id, @as(u64, g.flushed) * 16, std.mem.sliceAsBytes(self.gate_counts.items[g.flushed * 4 .. g.n * 4]));
            g.flushed = g.n;
        }
    }

    /// Stop gating (the session's slots stay valid until its work is submitted).
    pub fn endGate(self: *Gpu) void {
        self.gateFlush();
        self.gate = null;
    }

    /// Time every dispatch on the GPU (native only; false when unavailable).
    pub fn setProfile(self: *Gpu, on: bool) bool {
        return self.dev.setProfile(on);
    }

    /// GPU time per pipeline since the last call, slowest first, as text lines
    /// "ms  count  pipeline" into `out`; returns the length.
    pub fn profileReport(self: *Gpu, out: []u8) usize {
        var buf: [512]api.ProfEntry = undefined;
        const n = self.dev.takeProfile(&buf);
        const es = buf[0..n];
        std.sort.block(api.ProfEntry, es, {}, struct {
            fn gt(_: void, a: api.ProfEntry, b: api.ProfEntry) bool {
                return a.ns > b.ns;
            }
        }.gt);
        var total: f64 = 0;
        var calls: u64 = 0;
        for (es) |e| {
            total += e.ns;
            calls += e.count;
        }
        var w: std.Io.Writer = .fixed(out);
        w.print("{d:9.3} ms {d:6} dispatches  total\n", .{ total * 1e-6, calls }) catch return w.end;
        for (es) |e| w.print("{d:9.3} ms {d:6}  {s}\n", .{ e.ns * 1e-6, e.count, self.pipeName(e.pipe) }) catch break;
        return w.end;
    }

    /// Name of pipeline `pipe` (its cache key), for error messages.
    pub fn pipeName(self: *Gpu, pipe: u32) []const u8 {
        var it = self.pipes.iterator();
        while (it.next()) |kv| if (kv.value_ptr.* == pipe) return kv.key_ptr.*;
        return "?";
    }

    /// Upload the uniform arena, then submit everything recorded since the last submit.
    pub fn submit(self: *Gpu) void {
        self.counts.submits += 1;
        self.gateFlush();
        if (self.uoff > 0) {
            self.dev.writeBuffer(self.ubuf, 0, self.ustage[0..self.uoff]);
            self.uoff = 0;
        }
        if (self.soff > 0) {
            self.dev.writeBuffer(self.sbuf, 0, self.sstage[0..self.soff]);
            self.soff = 0;
        }
        self.dev.submit();
        self.open = false;
        for (self.retired.items) |id| self.dev.destroyBuffer(id);
        self.retired.clearRetainingCapacity();
    }

    /// The counters since the last call, then zeroed.
    pub fn countsTake(self: *Gpu) Counts {
        const c = self.counts;
        self.counts = .{};
        return c;
    }

    /// Submit, then block until every queued `read` has landed (natively by polling the device;
    /// in the browser the module suspends while the adapter awaits the reads).
    pub fn wait(self: *Gpu) !void {
        self.counts.waits += 1;
        self.submit();
        try self.dev.wait();
        if (self.oversize != null) return error.DispatchTooLarge;
        if (self.bad_pipe) {
            self.bad_pipe = false;
            return error.PipelineFailed;
        }
    }
};
