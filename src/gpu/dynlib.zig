//! Run-time library loading. Windows: kernel32 directly (std.DynLib has no Windows loader in
//! Zig 0.16, and the native library links no libc there). Elsewhere: dlopen (libc is linked on
//! Linux and macOS for this).
const std = @import("std");
const builtin = @import("builtin");

pub const Lib = if (builtin.os.tag == .windows) WinLib else PosixLib;

const WinLib = struct {
    handle: *anyopaque,

    extern "kernel32" fn LoadLibraryExW(name: [*:0]const u16, file: ?*anyopaque, flags: u32) callconv(.winapi) ?*anyopaque;
    extern "kernel32" fn GetProcAddress(module: *anyopaque, name: [*:0]const u8) callconv(.winapi) ?*anyopaque;
    extern "kernel32" fn FreeLibrary(module: *anyopaque) callconv(.winapi) i32;
    // Resolve the library's own dependencies from its directory, not the process's.
    const LOAD_WITH_ALTERED_SEARCH_PATH = 0x8;

    pub fn open(gpa: std.mem.Allocator, path: []const u8) !WinLib {
        const w = try std.unicode.utf8ToUtf16LeAllocZ(gpa, path);
        defer gpa.free(w);
        const h = LoadLibraryExW(w.ptr, null, LOAD_WITH_ALTERED_SEARCH_PATH) orelse return error.FileNotFound;
        return .{ .handle = h };
    }

    pub fn lookup(self: *WinLib, comptime T: type, name: [:0]const u8) ?T {
        const p = GetProcAddress(self.handle, name.ptr) orelse return null;
        return @ptrCast(@alignCast(p));
    }

    pub fn close(self: *WinLib) void {
        _ = FreeLibrary(self.handle);
    }
};

const PosixLib = struct {
    inner: std.DynLib,

    pub fn open(gpa: std.mem.Allocator, path: []const u8) !PosixLib {
        _ = gpa;
        return .{ .inner = try std.DynLib.open(path) };
    }

    pub fn lookup(self: *PosixLib, comptime T: type, name: [:0]const u8) ?T {
        return self.inner.lookup(T, name);
    }

    pub fn close(self: *PosixLib) void {
        self.inner.close();
    }
};
