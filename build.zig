const std = @import("std");

// zebsdsolve build.
//   zig build -Doptimize=ReleaseFast   native shared library (zig-out/bin or lib) + probe
//   zig build wasm                     zig-out/web/zebsdsolve.wasm (browser; adapter: web/zebsdsolve.js)
//   zig build probe                    run the native GPU probe
//   zig build test                     host-side unit tests (math, file readers; no GPU)
// The native library loads wgpu-native at run time (no link-time dependency), so only the
// WebGPU header is needed to build. -Dwgpu=<dir> points at an unpacked wgpu-native release.
pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const wasm_optimize = b.option(std.builtin.OptimizeMode, "wasm-optimize", "optimization of zebsdsolve.wasm (default ReleaseSmall)") orelse .ReleaseSmall;
    const wgpu_dir = b.option([]const u8, "wgpu", "unpacked wgpu-native release (include/webgpu)") orelse
        ".deps/wgpu-native/win_amd64";

    const shaders = b.createModule(.{ .root_source_file = b.path("shaders/shaders.zig") });

    // webgpu.h → Zig declarations (types and proc typedefs; functions are resolved at run time).
    const tc = b.addTranslateC(.{
        .root_source_file = b.path("src/gpu/webgpu_c.h"),
        .target = target,
        .optimize = optimize,
        .link_libc = false,
    });
    tc.addIncludePath(b.path("src/gpu/cshim"));
    tc.addIncludePath(.{ .cwd_relative = b.pathJoin(&.{ wgpu_dir, "include" }) });
    const webgpu_c = tc.createModule();

    // Native shared library (Python loads it with ctypes).
    // (stripped outside Debug: debug information names the directories of the machine that built it)
    const lib_mod = b.createModule(.{
        .root_source_file = b.path("src/native_api.zig"),
        .target = target,
        .optimize = optimize,
        .strip = optimize != .Debug,
    });
    lib_mod.addImport("webgpu_c", webgpu_c);
    // dlopen on Linux / macOS needs libc; Windows loads through kernel32 and links none.
    if (target.result.os.tag != .windows) lib_mod.link_libc = true;
    lib_mod.addImport("shaders", shaders);
    const lib = b.addLibrary(.{ .name = "zebsdsolve", .root_module = lib_mod, .linkage = .dynamic });
    b.installArtifact(lib);

    // Native probe: open the GPU, print the adapter, run a tiny kernel.
    const probe_mod = b.createModule(.{
        .root_source_file = b.path("tests/probe.zig"),
        .strip = optimize != .Debug,
        .target = target,
        .optimize = optimize,
    });
    probe_mod.addImport("webgpu_c", webgpu_c);
    probe_mod.addImport("shaders", shaders);
    probe_mod.addImport("zebsdsolve", lib_mod);
    const probe = b.addExecutable(.{ .name = "probe", .root_module = probe_mod });
    b.installArtifact(probe);
    const run_probe = b.addRunArtifact(probe);
    run_probe.addArg(b.pathJoin(&.{ wgpu_dir, "lib" }));
    b.step("probe", "run the native GPU probe").dependOn(&run_probe.step);

    // Host-side unit tests: everything that needs no GPU (run from the repo root: the file
    // readers' tests open testdata/).
    const test_step = b.step("test", "run the host-side unit tests");
    for ([_][]const u8{ "src/sh.zig", "src/master.zig", "src/scan.zig", "src/geometry.zig", "src/settings.zig", "src/reduce.zig" }) |f| {
        const t = b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path(f), .target = target, .optimize = optimize, .link_libc = target.result.os.tag != .windows }) });
        test_step.dependOn(&b.addRunArtifact(t).step);
    }

    // Browser module: wasm32-freestanding, GPU calls imported from web/zebsdsolve.js.
    const wasm_target = b.resolveTargetQuery(.{ .cpu_arch = .wasm32, .os_tag = .freestanding });
    const wasm_mod = b.createModule(.{
        .root_source_file = b.path("src/web_api.zig"),
        .target = wasm_target,
        .optimize = wasm_optimize,
    });
    wasm_mod.addImport("shaders", shaders);
    const wasm = b.addExecutable(.{ .name = "zebsdsolve", .root_module = wasm_mod });
    wasm.entry = .disabled;
    wasm.rdynamic = true;
    const install_wasm = b.addInstallArtifact(wasm, .{ .dest_dir = .{ .override = .{ .custom = "web" } } });
    b.step("wasm", "build zig-out/web/zebsdsolve.wasm").dependOn(&install_wasm.step);
}
