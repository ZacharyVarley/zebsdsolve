//! Types shared by the backends (native webgpu.h, browser imports) and the layer above them.

pub const Usage = enum(u32) {
    /// read-write storage, copy source and destination
    storage = 0,
    /// uniform, copy destination
    uniform = 1,
};

/// One bind group entry (group 0): `buf` (a buffer handle) at `slot`, bytes [offset, offset+size);
/// size 0 = to the end of the buffer.
pub const Bind = extern struct {
    slot: u32,
    buf: u32,
    offset: u64 = 0,
    size: u64 = 0,
};

pub const Limits = extern struct {
    max_storage_binding: u64,
    max_buffer: u64,
    max_workgroup_storage: u32,
    max_storage_per_stage: u32,
    uniform_align: u32,
    storage_align: u32,
    /// shader-f16 enabled on the device (half-precision texels)
    has_f16: u32 = 0,
    /// WGSL packed_4x8_integer_dot_product (dot4U8Packed)
    packed_dot: u32 = 0,
};

/// GPU time of one pipeline, summed over its dispatches (profiling).
pub const ProfEntry = struct { pipe: u32, count: u32, ns: f64 };
