//! Native GPU probe: open wgpu-native, report the adapter and limits, run one kernel.
//!   zig build probe            (the path to wgpu-native's lib/ is passed by build.zig)
const std = @import("std");
const ze = @import("zebsdsolve");
const gpu = ze.gpu;

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    var args = try init.minimal.args.iterateAllocator(gpa);
    defer args.deinit();
    _ = args.next();
    const lib_path = args.next() orelse ".deps/wgpu-native/win_amd64/lib";

    const dev = try gpu.Backend.Device.init(gpa, lib_path);
    defer dev.deinit();
    var g = try gpu.Gpu.init(gpa, dev);
    defer g.deinit();
    const lim = g.lim;
    std.debug.print("adapter: {s}\nlimits: storage binding {d} MiB, buffer {d} MiB, workgroup storage {d} B, storage/stage {d}, uniform align {d}\n", .{
        dev.info(), lim.max_storage_binding >> 20, lim.max_buffer >> 20, lim.max_workgroup_storage, lim.max_storage_per_stage, lim.uniform_align,
    });

    // fft.wgsl `fft_lines`: the forward FFT of a unit impulse at point 1 of an 8-point line is
    // e^{−2πi k / 8}.
    const pipe = try g.pipeline("fft/fft_lines", ze.shaders.fft, "fft_lines");
    if (dev.takeError()) |e| std.debug.print("error: {s}\n", .{e});
    var data = g.storage(16 * 4);
    defer g.release(&data);
    var tw = g.storage(1024 * 8);
    defer g.release(&tw);
    var line: [16]f32 = @splat(0);
    line[2] = 1;
    g.writeSlice(data, 0, f32, &line);
    var tw_h: [2048]f32 = undefined;
    for (0..1024) |t| {
        const ang = -2.0 * std.math.pi * @as(f64, @floatFromInt(t)) / 1024.0;
        tw_h[2 * t] = @floatCast(@cos(ang));
        tw_h[2 * t + 1] = @floatCast(@sin(ang));
    }
    g.writeSlice(tw, 0, f32, &tw_h);
    const u = [8]u32{ 8, 3, 1, 1, 8, 1, 1024, 0 };
    var ub: [48]u8 = undefined;
    @memcpy(ub[0..32], std.mem.asBytes(&u));
    @memcpy(ub[32..48], std.mem.asBytes(&[4]f32{ -1, 1, 0, 0 }));
    g.dispatch(pipe, 1, 1, 1, &.{ g.uniform(0, &ub), data.at(1), tw.at(2) });
    var host: [16]f32 = undefined;
    g.read(data, 0, std.mem.sliceAsBytes(&host));
    try g.wait();
    if (dev.takeError()) |e| std.debug.print("error: {s}\n", .{e});
    std.debug.print("fft of an impulse: X[1] = {d:.4} {d:.4}i (expect 0.7071 -0.7071i), X[2] = {d:.4} {d:.4}i (expect 0 -1i)\n", .{ host[2], host[3], host[4], host[5] });
}
