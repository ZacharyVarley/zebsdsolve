//! Patterns reduced on the CPU as they are read: the average of the detector area each output
//! pixel covers. A whole factor that divides the pattern is plain binning; any other factor
//! weights the pixels an output pixel only partly covers by the part covered (area resampling:
//! the exact box filter, so nothing aliases and a flat pattern stays flat). The output keeps
//! square pixels: `floor(size / factor)` each way, from the top-left corner, and whatever is
//! left of the last row or column is dropped.
//!
//! This runs on the host, not in a kernel: a reduced pattern is all the search needs of a
//! pattern, and reducing before the upload keeps most of a large scan's bytes off the GPU.
const std = @import("std");

/// Size of a pattern of `src` pixels along one axis after reduction by `factor`.
pub fn size(src: u32, factor: f64) u32 {
    return @intFromFloat(@floor(@as(f64, @floatFromInt(src)) / factor + 1e-9));
}

pub fn shape(src_h: u32, src_w: u32, factor: f64) [2]u32 {
    return .{ size(src_h, factor), size(src_w, factor) };
}

/// True when `factor` is a whole number (then every output pixel is a block's mean).
pub fn isWhole(factor: f64) bool {
    return @abs(factor - @round(factor)) < 1e-9;
}

fn pixel(comptime T: type, bytes: []const u8, i: usize) f32 {
    return switch (T) {
        u8 => @floatFromInt(bytes[i]),
        u16 => @floatFromInt(std.mem.readInt(u16, bytes[2 * i ..][0..2], .little)),
        f32 => @bitCast(std.mem.readInt(u32, bytes[4 * i ..][0..4], .little)),
        else => unreachable,
    };
}

/// Each output pixel from the `kc` column sums it spans.
fn across(comptime kc: usize, band: []const u32, out: []f32, inv: f32) void {
    for (out, 0..) |*o, c| {
        var acc: u32 = 0;
        inline for (0..kc) |j| acc += band[c * kc + j];
        o.* = @as(f32, @floatFromInt(acc)) * inv;
    }
}

/// Reduce one pattern of `src_h × src_w` pixels of type T (stored bytes, little-endian) by
/// `factor` into `dst` (`shape(...)` f32). `row` is scratch of the output's width, `sums` of the
/// pattern's.
pub fn pattern(comptime T: type, src: []const u8, src_h: u32, src_w: u32, factor: f64, dst: []f32, row: []f32, sums: []u32) void {
    const hw = shape(src_h, src_w, factor);
    const h: usize = hw[0];
    const w: usize = hw[1];
    std.debug.assert(dst.len == h * w and row.len == w and sums.len >= src_w);
    if (isWhole(factor)) {
        const k: usize = @intFromFloat(@round(factor));
        const inv = 1.0 / @as(f32, @floatFromInt(k * k));
        if (T != f32 and k <= 256) {
            // Whole pixels of integers: the columns of each band of k rows are summed as
            // integers, a plain loop over the row that the compiler turns into vector adds,
            // and k neighbouring sums make a pixel. (Indexing is otherwise bound by this loop:
            // a pattern a pixel at a time through floats cost more than its search.)
            const cols = w * k;
            const px: []align(1) const T = std.mem.bytesAsSlice(T, src[0 .. @as(usize, src_h) * src_w * @sizeOf(T)]);
            for (0..h) |r| {
                const band = sums[0..cols];
                for (band, px[r * k * src_w ..][0..cols]) |*a, v| a.* = v;
                for (1..k) |i| {
                    for (band, px[(r * k + i) * src_w ..][0..cols]) |*a, v| a.* += v;
                }
                const out = dst[r * w ..][0..w];
                switch (k) {
                    // (a stride the compiler knows is one it can vectorize)
                    inline 2, 3, 4, 5, 6, 8 => |kc| across(kc, band, out, inv),
                    else => for (out, 0..) |*o, c| {
                        var acc: u32 = 0;
                        for (band[c * k ..][0..k]) |v| acc += v;
                        o.* = @as(f32, @floatFromInt(acc)) * inv;
                    },
                }
            }
            return;
        }
        for (0..h) |r| {
            const out = dst[r * w ..][0..w];
            @memset(out, 0);
            for (0..k) |i| {
                const line = (r * k + i) * src_w;
                for (0..w) |c| {
                    var acc: f32 = 0;
                    for (0..k) |j| acc += pixel(T, src, line + c * k + j);
                    out[c] += acc;
                }
            }
            for (out) |*v| v.* *= inv;
        }
        return;
    }
    @memset(dst, 0);
    const inv: f32 = @floatCast(1.0 / (factor * factor));
    // source rows the output covers
    const rows: usize = @intFromFloat(@ceil(@as(f64, @floatFromInt(h)) * factor - 1e-9));
    for (0..rows) |r| {
        // this source row reduced along its length
        const line = r * src_w;
        for (0..w) |c| {
            const x0 = @as(f64, @floatFromInt(c)) * factor;
            const x1 = x0 + factor;
            var j: usize = @intFromFloat(@floor(x0));
            var acc: f64 = 0;
            while (@as(f64, @floatFromInt(j)) < x1 - 1e-12) : (j += 1) {
                const a = @max(x0, @as(f64, @floatFromInt(j)));
                const b = @min(x1, @as(f64, @floatFromInt(j + 1)));
                acc += (b - a) * pixel(T, src, line + j);
            }
            row[c] = @floatCast(acc);
        }
        // and shared between the one or two output rows it lies in
        const y0: f64 = @floatFromInt(r);
        const top: usize = @intFromFloat(@floor(y0 / factor + 1e-12));
        const edge = @as(f64, @floatFromInt(top + 1)) * factor;
        const w0: f32 = @floatCast(@min(y0 + 1, edge) - y0);
        if (top < h) {
            for (dst[top * w ..][0..w], row) |*v, x| v.* += w0 * x;
        }
        if (w0 < 1 and top + 1 < h) {
            for (dst[(top + 1) * w ..][0..w], row) |*v, x| v.* += (1 - w0) * x;
        }
    }
    for (dst) |*v| v.* *= inv;
}

/// Reduce `n` patterns stored back to back; `pixel_bytes` is 1 (u8), 2 (u16) or 4 (f32).
pub fn patterns(gpa: std.mem.Allocator, src: []const u8, n: usize, src_h: u32, src_w: u32, pixel_bytes: u32, factor: f64, dst: []f32) !void {
    const hw = shape(src_h, src_w, factor);
    const out: usize = @as(usize, hw[0]) * hw[1];
    const in: usize = @as(usize, src_h) * src_w * pixel_bytes;
    std.debug.assert(dst.len == n * out);
    const row = try gpa.alloc(f32, hw[1]);
    defer gpa.free(row);
    const sums = try gpa.alloc(u32, src_w);
    defer gpa.free(sums);
    for (0..n) |k| {
        const a = src[k * in ..][0..in];
        const b = dst[k * out ..][0..out];
        switch (pixel_bytes) {
            1 => pattern(u8, a, src_h, src_w, factor, b, row, sums),
            2 => pattern(u16, a, src_h, src_w, factor, b, row, sums),
            else => pattern(f32, a, src_h, src_w, factor, b, row, sums),
        }
    }
}

test "a whole factor is binning" {
    var src: [6 * 8]u8 = undefined;
    for (&src, 0..) |*v, i| v.* = @intCast((i * 7 + 3) % 251);
    var dst: [3 * 4]f32 = undefined;
    var row: [4]f32 = undefined;
    var sums: [8]u32 = undefined;
    pattern(u8, &src, 6, 8, 2, &dst, &row, &sums);
    for (0..3) |r| for (0..4) |c| {
        var acc: f32 = 0;
        for (0..2) |i| for (0..2) |j| {
            acc += @floatFromInt(src[(2 * r + i) * 8 + 2 * c + j]);
        };
        try std.testing.expectApproxEqAbs(acc / 4, dst[r * 4 + c], 1e-5);
    };
    try std.testing.expectEqual([2]u32{ 3, 4 }, shape(7, 9, 2));
    // the same from 16-bit pixels and from floats, with a remainder dropped (7 × 9 by 3 → 2 × 3)
    var wide: [7 * 9 * 2]u8 = undefined;
    var real: [7 * 9 * 4]u8 = undefined;
    for (0..63) |i| {
        const v: u16 = @intCast((i * 977 + 31) % 60000);
        std.mem.writeInt(u16, wide[2 * i ..][0..2], v, .little);
        std.mem.writeInt(u32, real[4 * i ..][0..4], @bitCast(@as(f32, @floatFromInt(v))), .little);
    }
    var a: [6]f32 = undefined;
    var b: [6]f32 = undefined;
    var row3: [3]f32 = undefined;
    var sums9: [9]u32 = undefined;
    pattern(u16, &wide, 7, 9, 3, &a, &row3, &sums9);
    pattern(f32, &real, 7, 9, 3, &b, &row3, &sums9);
    for (a, b) |x, y| try std.testing.expectApproxEqRel(y, x, 1e-6);
}

test "any factor averages the area it covers" {
    const gpa = std.testing.allocator;
    // 10 × 10 of u16 by 2.5 → 4 × 4, nothing dropped: the total is conserved
    var src: [200]u8 = undefined;
    var total: f64 = 0;
    for (0..100) |i| {
        const v: u16 = @intCast((i * 37 + 11) % 1000);
        std.mem.writeInt(u16, src[2 * i ..][0..2], v, .little);
        total += @floatFromInt(v);
    }
    var dst: [16]f32 = undefined;
    try patterns(gpa, &src, 1, 10, 10, 2, 2.5, &dst);
    var sum: f64 = 0;
    for (dst) |v| sum += v;
    try std.testing.expectApproxEqRel(total, sum * 2.5 * 2.5, 1e-5);
    // the first output pixel by hand: rows and columns 0, 1 whole and half of 2
    var want: f64 = 0;
    for (0..3) |r| for (0..3) |c| {
        const wr: f64 = if (r == 2) 0.5 else 1;
        const wc: f64 = if (c == 2) 0.5 else 1;
        want += wr * wc * @as(f64, @floatFromInt(std.mem.readInt(u16, src[2 * (r * 10 + c) ..][0..2], .little)));
    };
    try std.testing.expectApproxEqRel(want / 6.25, @as(f64, dst[0]), 1e-5);
    // a flat pattern stays flat, with a remainder dropped (10 by 3 → 3 × 3 of the first 9)
    var flat: [100]u8 = @splat(7);
    var out: [9]f32 = undefined;
    try patterns(gpa, &flat, 1, 10, 10, 1, 3, &out);
    for (out) |v| try std.testing.expectApproxEqAbs(@as(f32, 7), v, 1e-5);
    var out2: [36]f32 = undefined;
    try patterns(gpa, &flat, 1, 10, 10, 1, 1.6, &out2);
    for (out2) |v| try std.testing.expectApproxEqAbs(@as(f32, 7), v, 1e-5);
}
