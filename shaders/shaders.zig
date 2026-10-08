//! WGSL sources, embedded at compile time. Each module is one text (one shader module on the
//! device); the kernels of a module are its entry points.
const common = @embedFile("common.wgsl");
const fft_src = @embedFile("fft.wgsl");

// Two workgroup sizes are not written in the texts: the engine puts a line that sets each
// before the text it compiles (`gpu.pipelineWith`), 64 by default and whatever tuning chose
// otherwise. The `_body` texts lack that line; the others carry the default.
pub const fft_wg = "FFT_WG";
pub const sums_wg = "SUMS_WG";
pub const wg_default = 64;
const fft_line = "const FFT_WG: u32 = 64u;\n";
const sums_line = "const SUMS_WG: u32 = 64u;\n";

/// Line FFT (power-of-two lengths; one text for every length).
pub const fft_body = fft_src;
pub const fft = fft_line ++ fft_body;
/// Pattern preprocessing: load, background, blur, CLAHE, normalize.
pub const prep = common ++ "\n" ++ @embedFile("prep.wgsl");
/// Patterns onto the sphere and into harmonics.
pub const sphere = common ++ "\n" ++ @embedFile("sphere.wgsl");
/// Normalized cross-correlation over SO(3) (with the FFT it runs along its last axis).
pub const so3_body = common ++ "\n" ++ fft_src ++ "\n" ++ @embedFile("so3.wgsl");
pub const so3 = fft_line ++ so3_body;
/// Filters on a master pattern as a function on the sphere.
pub const mastersph = common ++ "\n" ++ @embedFile("mastersph.wgsl");
/// Master patterns projected on the detector: simulation, fit sums, refinement, NMI, dictionary.
pub const detector_body = common ++ "\n" ++ @embedFile("detector.wgsl");
pub const detector = sums_line ++ detector_body;

/// A dictionary reduced to its leading components (Oja's rule), and matching in them.
pub const pca = common ++ "\n" ++ @embedFile("pca.wgsl");
/// Patterns averaged with the neighbours in the map that resemble them (NLPAR).
pub const nlpar = common ++ "\n" ++ @embedFile("nlpar.wgsl");

/// Every module by name (tests/check_shaders.py builds each entry point of each).
pub const modules = .{
    .{ "fft", fft },
    .{ "prep", prep },
    .{ "sphere", sphere },
    .{ "so3", so3 },
    .{ "detector", detector },
    .{ "mastersph", mastersph },
    .{ "pca", pca },
    .{ "nlpar", nlpar },
};
