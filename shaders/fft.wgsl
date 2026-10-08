// Radix-2 line FFT for power-of-two lengths up to 1024, one text for every length: the length
// and the layout of the lines are uniforms, so the engine compiles this once (every distinct
// shader text is a compile on the user's machine).
//
// A workgroup holds as many whole lines as fit its memory (1024 / n), loads them bit-reversed,
// runs log2(n) butterfly stages in place with a barrier between stages, and stores them. Every
// length here is a power of two, so the indices are shifts and masks: a division by a number
// the compiler does not know costs more than the butterfly it would address.
//
// Lines lie back to back: point p of line i at i · stored + p. `stored` < n reads the missing
// points as zero (padding); only the first `keep` points of each line are written back.

struct FftP {
    n: u32,
    log2n: u32,
    keep: u32,
    n_lines: u32,
    stored: u32,
    grid_x: u32,
    // length of the twiddle table (a power of two ≥ n): tw[t] = e^{−2πi t / tw_len}
    tw_len: u32,
    pad0: u32,
    // −1 forward (e^{−i…}), +1 inverse (e^{+i…}); the output is multiplied by scale
    sign: f32,
    scale: f32,
    pad1: f32,
    pad2: f32,
};

@group(0) @binding(0) var<uniform> fp: FftP;
@group(0) @binding(1) var<storage, read_write> fdata: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> ftw: array<vec2<f32>>;

const FFT_MAX: u32 = 1024u;
// FFT_WG, the invocations of a workgroup (a power of two up to 256), is set in a line the engine
// puts before this text (shaders.zig): 64 unless tuning found this GPU faster with another.
var<workgroup> fs: array<vec2<f32>, 1024>;

// The butterfly stages over the lpw bit-reversed lines in fs.
fn fft_core(tid: u32, lpw: u32) {
    let hn = fp.n >> 1u;
    let total = lpw * hn;
    let lh = fp.log2n - 1u;
    for (var st = 0u; st < fp.log2n; st = st + 1u) {
        let half = 1u << st;
        let tstep = fp.tw_len >> (st + 1u);
        for (var e = tid; e < total; e = e + FFT_WG) {
            let r = e & (hn - 1u);
            let j = r & (half - 1u);
            let i0 = ((e >> lh) << fp.log2n) + ((r >> st) << (st + 1u)) + j;
            let i1 = i0 + half;
            let t = ftw[j * tstep];
            let w = vec2<f32>(t.x, -fp.sign * t.y);
            let x = fs[i1];
            let b = vec2<f32>(x.x * w.x - x.y * w.y, x.x * w.y + x.y * w.x);
            let a = fs[i0];
            fs[i0] = a + b;
            fs[i1] = a - b;
        }
        workgroupBarrier();
    }
}

@compute @workgroup_size(FFT_WG)
fn fft_lines(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let n = fp.n;
    let lpw = FFT_MAX / n;
    let shift = 32u - fp.log2n;
    let line0 = (wid.y * fp.grid_x + wid.x) * lpw;
    for (var e = tid; e < lpw * n; e = e + FFT_WG) {
        let pnt = e & (n - 1u);
        let line = line0 + (e >> fp.log2n);
        var v = vec2<f32>(0.0, 0.0);
        if (line < fp.n_lines && pnt < fp.stored) { v = fdata[line * fp.stored + pnt]; }
        fs[(e - pnt) + (reverseBits(pnt) >> shift)] = v;
    }
    workgroupBarrier();
    fft_core(tid, lpw);
    for (var e = tid; e < lpw * n; e = e + FFT_WG) {
        let pnt = e & (n - 1u);
        let line = line0 + (e >> fp.log2n);
        if (line < fp.n_lines && pnt < fp.keep) { fdata[line * fp.stored + pnt] = fs[e] * fp.scale; }
    }
}
