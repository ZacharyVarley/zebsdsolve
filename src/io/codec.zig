//! Decoders for the byte filters scan files use: deflate (HDF5 gzip, NPZ members), LZF (h5py),
//! byte shuffle (HDF5), and the LZW that Oxford and Bruker use for compressed patterns is not
//! here because those files store one stream per pattern in formats read elsewhere.
const std = @import("std");

/// Inflate `src` (zlib-wrapped, or raw deflate) into `dst`, which must be exactly the output size.
pub fn inflate(src: []const u8, dst: []u8, raw: bool) !void {
    var in: std.Io.Reader = .fixed(src);
    var dec: std.compress.flate.Decompress = .init(&in, if (raw) .raw else .zlib, &.{});
    var out: std.Io.Writer = .fixed(dst);
    const n = dec.reader.streamRemaining(&out) catch return error.BadDeflateStream;
    if (n != dst.len) return error.BadDeflateStream;
}

/// LZF (liblzf, as the HDF5 filter 32000 stores chunks) into `dst`, exactly the output size.
pub fn lzf(src: []const u8, dst: []u8) !void {
    var ip: usize = 0;
    var op: usize = 0;
    while (ip < src.len) {
        const ctrl: usize = src[ip];
        ip += 1;
        if (ctrl < 32) {
            const n = ctrl + 1;
            if (ip + n > src.len or op + n > dst.len) return error.BadLzfStream;
            @memcpy(dst[op..][0..n], src[ip..][0..n]);
            ip += n;
            op += n;
        } else {
            var len = ctrl >> 5;
            if (len == 7) {
                if (ip >= src.len) return error.BadLzfStream;
                len += src[ip];
                ip += 1;
            }
            if (ip >= src.len) return error.BadLzfStream;
            const back = ((ctrl & 0x1f) << 8) + src[ip] + 1;
            ip += 1;
            len += 2;
            if (back > op or op + len > dst.len) return error.BadLzfStream;
            // the ranges may overlap (run-length encoding): copy forward byte by byte
            var k: usize = 0;
            while (k < len) : (k += 1) dst[op + k] = dst[op - back + k];
            op += len;
        }
    }
    if (op != dst.len) return error.BadLzfStream;
}

/// Undo the HDF5 shuffle filter: `src` holds byte 0 of every element, then byte 1, ….
pub fn unshuffle(src: []const u8, dst: []u8, elem: usize) void {
    if (elem <= 1) {
        @memcpy(dst, src);
        return;
    }
    const n = src.len / elem;
    for (0..elem) |b| for (0..n) |i| {
        dst[i * elem + b] = src[b * n + i];
    };
    const tail = n * elem;
    @memcpy(dst[tail..], src[tail..]);
}

test "lzf literals and back references" {
    // "abcabcabcabc": literal run of 3, then a 9-byte copy from 3 back
    const src = [_]u8{ 2, 'a', 'b', 'c', (7 << 5) | 0, 0, 2 };
    var out: [12]u8 = undefined;
    try lzf(&src, &out);
    try std.testing.expectEqualStrings("abcabcabcabc", &out);
    try std.testing.expectError(error.BadLzfStream, lzf(src[0..5], &out));
}

test "inflate a stored zlib stream" {
    // zlib header, one stored block "hello", adler32
    const src = [_]u8{ 0x78, 0x01, 0x01, 0x05, 0x00, 0xfa, 0xff, 'h', 'e', 'l', 'l', 'o', 0x06, 0x2c, 0x02, 0x15 };
    var out: [5]u8 = undefined;
    try inflate(&src, &out, false);
    try std.testing.expectEqualStrings("hello", &out);
}

test "unshuffle" {
    const src = [_]u8{ 1, 3, 5, 2, 4, 6 };
    var out: [6]u8 = undefined;
    unshuffle(&src, &out, 2);
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4, 5, 6 }, &out);
}
