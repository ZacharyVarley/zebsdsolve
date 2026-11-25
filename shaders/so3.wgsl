// Normalized cross-correlation over SO(3) from spherical-harmonic coefficients (after
// common.wgsl and fft.wgsl).
//
// For a pattern g (sample frame) and a master f (crystal frame),
//     C(a, b, c) = ∫ g(x) f(S⁻¹ x) dΩ,   S = Rz(a) Ry(b) Rz(c),
// is a three-dimensional Fourier series whose coefficients are sums over l (sh.zig):
//     C = Σ T[k, m', m] e^{−i (k b + m' a + m c)},
//     T = i^{m'−m} Σ_l f_lm conj(g_lm') d^l_{k m'}(π/2) d^l_{k m}(π/2).
// The orientation at a peak is S⁻¹ (sample → crystal).
//
// Two symmetries each halve the work, and the order of the axes is chosen to use both:
//   * d^l_{−k m} = (−1)^{l+m} d^l_{k m} gives T[−k, m', m] = (−1)^{m'+m} T[k, m', m], which says
//     C(a, −b, c) = C(a + π, b, c + π): b in [0, π] holds every orientation. So the b axis is
//     transformed first, from k ≥ 0, and only its half is kept.
//   * C is real, so after that T[·, −m', −m] is the conjugate of T[·, m', m]: only m' ≥ 0 is
//     ever computed, and the a axis, transformed last, is a real transform.
// A master with a z axis of order `fold` has only orders m that are its multiples, and the c
// axis then spans 2π / fold. The master's factors come as one table,
// W[m, l, k] = i^{−m} f_lm d^l_{k m}, and d^l_{k m'} as another, both with k innermost and
// zero where l < k: the invocations of a workgroup hold neighbouring k and step through l
// together, so each step they read a few neighbouring words between them.
//
//   spectrum     T for k ≥ 0 and m' ≥ 0, columns along k into the workgroup's memory (two
//                m' to a column, one even in k and one odd), and from there at once their
//                transform along k, U[j; m', m] = Σ_k T e^{−i k b_j} for b_j = 2π j / n,
//                j ≤ n / 2. T itself is never stored.
//   a_peak_m     an (item, j) slab of U at a time in the workgroup's memory: along c, then
//                along a as below, and the maximum of each (b, c) line; where the slab fits
//   (fft.wgsl)   otherwise: forward FFT along m → V[j; m', c], and then
//   a_peak       the last axis: C(a) = V[0] + 2 Re Σ_{m' > 0} V[m'] e^{−i m' a} at all n values
//                of a from one complex FFT of half that length (the real values come out in
//                pairs, as real and imaginary parts), times the reciprocal denominator of the
//                normalized correlation, and the maximum of each (b, c) line
//   a_volume     the same lines written out instead (the denominators are built from these)
//   rden_build   1 / sqrt(⟨w, f²⟩ − ⟨w, f⟩² / ∫w) from the window's correlations with f and f²
//   peak_reduce  the best line per item
//   peak_finish  parabolic interpolation around the peak and the orientation quaternion
//                (peak_finish_m on a_peak_m's path: a workgroup an item, V along c from U)
//
// Layouts: U and V at [((b·nb + j)·L + m')·nc + iq] for item b, nb = n / 2 + 1 and
// iq = (m / fold) mod nc; correlation volumes at [(j·nc + iq)·n + ia]. A line of every
// transform but the first is then contiguous or a short stride apart, and the first reads only
// the tables (small enough to stay in the cache) and the item's own coefficients.

struct SoP {
    n_items: u32,
    bw: u32,
    n: u32,
    nc: u32,
    // master orders stored: μ in [−mu_max, mu_max], mc = 2 mu_max + 1 of them, m = fold · μ
    mc: u32,
    mu_max: u32,
    fold: u32,
    // 2 when the master has no odd l (inversion), else 1
    l_step: u32,
    n_pairs: u32,
    grid_x: u32,
    // items per denominator volume: item b reads volume b / rden_div
    rden_div: u32,
    // n / 2 + 1 values of b
    nb: u32,
    log2nc: u32,
    // 1: spec holds U, not yet transformed along c (a_peak_m's path)
    u_in_spec: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(3) var<uniform> so: SoP;
@group(0) @binding(4) var<storage, read> wmaster: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read> dq: array<f32>;
@group(0) @binding(6) var<storage, read> coef: array<vec2<f32>>;
@group(0) @binding(7) var<storage, read_write> spec: array<vec2<f32>>;
@group(0) @binding(8) var<storage, read> rden: array<f32>;
@group(0) @binding(9) var<storage, read_write> cand_v: array<f32>;
@group(0) @binding(10) var<storage, read_write> cand_i: array<u32>;
// per item: bitcast of the best value, its volume index
@group(0) @binding(11) var<storage, read_write> best: array<u32>;
@group(0) @binding(12) var<storage, read_write> vol: array<f32>;
@group(0) @binding(13) var<storage, read> vol_f: array<f32>;
@group(0) @binding(14) var<storage, read> vol_f2: array<f32>;
// per item of the window batch: the window's area
@group(0) @binding(15) var<storage, read> area: array<f32>;
@group(0) @binding(16) var<storage, read_write> rden_out: array<f32>;
// per item: quaternion (w, x, y, z), score, a, b, c
@group(0) @binding(17) var<storage, read_write> peaks: array<f32>;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// i^{m'} x
fn i_pow(x: vec2<f32>, am: u32) -> vec2<f32> {
    switch (am & 3u) {
        case 1u: { return vec2<f32>(-x.y, x.x); }
        case 2u: { return -x; }
        case 3u: { return vec2<f32>(x.y, -x.x); }
        default: { return x; }
    }
}

// the master's signed order μ of wrapped index iq
fn mu_of(iq: u32) -> i32 {
    if (iq >= (so.nc + 1u) / 2u) { return i32(iq) - i32(so.nc); }
    return i32(iq);
}

// acc + W conj(g) d, a term of spectrum's sums
fn sp_term(acc: vec2<f32>, w: vec2<f32>, g: vec2<f32>, d: f32) -> vec2<f32> {
    return acc + cmul(w, vec2<f32>(g.x, -g.y)) * d;
}

// spectrum's T at k and −k of the pair (line i of fs) from the two lines' sums ta, tb of
// column iq (zero where the master has no such order)
fn sp_put(keep: bool, i: u32, k: u32, ta: vec2<f32>, tb: vec2<f32>, am: u32, iq: u32) {
    let n = fp.n;
    let shift = 32u - fp.log2n;
    let amu = u32(abs(mu_of(iq)));
    // T[−k] = (−1)^{m'+m} T[k]: odd for the line m' = am, else for am + 1
    let odd = ((am + amu * so.fold) & 1u) == 1u;
    var xa = i_pow(ta, am);
    var xb = i_pow(tb, am + 1u);
    if (amu > so.mu_max) { xa = vec2<f32>(0.0, 0.0); }
    if (amu > so.mu_max || am + 1u >= so.bw) { xb = vec2<f32>(0.0, 0.0); }
    if (keep) {
        fs[fsw(i * n + (reverseBits(k) >> shift))] = xa + xb;
        if (k > 0u) { fs[fsw(i * n + (reverseBits(n - k) >> shift))] = select(xa, -xa, odd) + select(xb, -xb, !odd); }
    }
}

// W's row for column iq (μ clamped to the table: a column beyond it is zeroed by sp_put) and the
// lowest l it is nonzero at (L for none)
fn sp_wrow(iq: u32) -> vec2<u32> {
    let mu = mu_of(iq);
    let amu = u32(abs(mu));
    let row = u32(clamp(mu, -i32(so.mu_max), i32(so.mu_max)) + i32(so.mu_max)) * so.bw * so.bw;
    return vec2<u32>(row, select(so.bw, amu * so.fold, amu <= so.mu_max));
}

// Lines here are columns along k, a pair of them to one transform: m' = 2h and 2h + 1 at the
// same (item, m). Their T are one even and one odd in k (m' + m is odd for one of them), so the
// transform of their sum holds both: the even part of what comes out is the even one's
// transform, the odd part the odd one's. A workgroup takes qb = min(4, lpw) neighbouring m of
// lpw / qb items, all at one pair: pair line ((g·⌈L/2⌉ + h)·(nc / qb) + q)·lpw + t·qb + c holds
// item g·(lpw / qb) + t at iq = q·qb + c. Its stores are then runs of qb neighbouring words, and
// its sums share their rows of W and d.
@compute @workgroup_size(FFT_WG)
fn spectrum(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let n = fp.n;
    let lpw = FFT_MAX / n;
    let log2lpw = 10u - fp.log2n;
    let shift = 32u - fp.log2n;
    let wg = wid.y * fp.grid_x + wid.x;
    let L = so.bw;
    let Lh = (L + 1u) / 2u;
    let qb = min(4u, lpw);
    let log2qb = select(1u, 2u, qb == 4u);
    let per = lpw >> log2qb;
    let log2g = so.log2nc - log2qb;
    let iq0 = (wg & ((1u << log2g) - 1u)) << log2qb;
    let rest = wg >> log2g;
    let grp = rest / Lh;
    let am = 2u * (rest - grp * Lh);
    let amb = min(am + 1u, L - 1u);
    // k runs over −L < k < L in a transform of length n ≥ 2L: zero between
    for (var e = tid; e < lpw * n; e = e + FFT_WG) {
        let p = e & (n - 1u);
        if (p >= L && p <= n - L) { fs[fsw((e - p) + (reverseBits(p) >> shift))] = vec2<f32>(0.0, 0.0); }
    }
    // T[k] = Σ_l W[μ, l, k] conj(g_{l m'}) d^l_{k m'}: an invocation takes one k, the qb columns
    // and two of the items (one when the workgroup has one), for both lines of the pair: each W
    // it reads serves four sums, each d eight (the items' coefficients are words all its
    // neighbours read too). Each sum starts at the lowest l of the four columns (the tables are
    // zero below its own: W where l < |m|, d where l < max(k, m')) and runs in the order of l.
    // (With two columns, n = 512, it sums two more that it does not keep.)
    let r0 = sp_wrow(iq0);
    let r1 = sp_wrow(iq0 + 1u);
    let r2 = sp_wrow(iq0 + min(2u, qb - 1u));
    let r3 = sp_wrow(iq0 + min(3u, qb - 1u));
    let lo = min(min(r0.y, r1.y), min(r2.y, r3.y));
    let two = per > 1u;
    let nch = max(per >> 1u, 1u);
    for (var e = tid; e < nch * L; e = e + FFT_WG) {
        let ch = e / L;
        let k = e - ch * L;
        let t0 = ch * 2u;
        let b0 = grp * per + t0;
        var a00 = vec2<f32>(0.0, 0.0);
        var b00 = vec2<f32>(0.0, 0.0);
        var a01 = vec2<f32>(0.0, 0.0);
        var b01 = vec2<f32>(0.0, 0.0);
        var a02 = vec2<f32>(0.0, 0.0);
        var b02 = vec2<f32>(0.0, 0.0);
        var a03 = vec2<f32>(0.0, 0.0);
        var b03 = vec2<f32>(0.0, 0.0);
        var a10 = vec2<f32>(0.0, 0.0);
        var b10 = vec2<f32>(0.0, 0.0);
        var a11 = vec2<f32>(0.0, 0.0);
        var b11 = vec2<f32>(0.0, 0.0);
        var a12 = vec2<f32>(0.0, 0.0);
        var b12 = vec2<f32>(0.0, 0.0);
        var a13 = vec2<f32>(0.0, 0.0);
        var b13 = vec2<f32>(0.0, 0.0);
        if (b0 < so.n_items && lo < L) {
            var l = max(k & ~15u, max(am, lo));
            if (so.l_step == 2u) { l = l + (l & 1u); }
            // (an item past the last reads the last's coefficients: its lines are not stored)
            let last = so.n_items - 1u;
            let ga = am * L - (am * (am + 1u)) / 2u;
            let gb = amb * L - (amb * (amb + 1u)) / 2u;
            let cA = min(b0, last) * so.n_pairs;
            let cB = min(b0 + 1u, last) * so.n_pairs;
            let da = am * L * L + k;
            let db = amb * L * L + k;
            for (; l < L; l = l + so.l_step) {
                let at = l * L + k;
                let w0 = wmaster[r0.x + at];
                let w1 = wmaster[r1.x + at];
                let w2 = wmaster[r2.x + at];
                let w3 = wmaster[r3.x + at];
                let d0 = dq[da + l * L];
                let d1 = dq[db + l * L];
                let g0 = coef[cA + ga + l];
                let g1 = coef[cA + gb + l];
                a00 = sp_term(a00, w0, g0, d0);
                b00 = sp_term(b00, w0, g1, d1);
                a01 = sp_term(a01, w1, g0, d0);
                b01 = sp_term(b01, w1, g1, d1);
                a02 = sp_term(a02, w2, g0, d0);
                b02 = sp_term(b02, w2, g1, d1);
                a03 = sp_term(a03, w3, g0, d0);
                b03 = sp_term(b03, w3, g1, d1);
                if (two) {
                    let g2 = coef[cB + ga + l];
                    let g3 = coef[cB + gb + l];
                    a10 = sp_term(a10, w0, g2, d0);
                    b10 = sp_term(b10, w0, g3, d1);
                    a11 = sp_term(a11, w1, g2, d0);
                    b11 = sp_term(b11, w1, g3, d1);
                    a12 = sp_term(a12, w2, g2, d0);
                    b12 = sp_term(b12, w2, g3, d1);
                    a13 = sp_term(a13, w3, g2, d0);
                    b13 = sp_term(b13, w3, g3, d1);
                }
            }
        }
        sp_put(true, (t0 + 0u) * qb + 0u, k, a00, b00, am, iq0 + 0u);
        sp_put(true, (t0 + 0u) * qb + 1u, k, a01, b01, am, iq0 + 1u);
        sp_put(2u < qb, (t0 + 0u) * qb + 2u, k, a02, b02, am, iq0 + 2u);
        sp_put(3u < qb, (t0 + 0u) * qb + 3u, k, a03, b03, am, iq0 + 3u);
        sp_put(two, (t0 + 1u) * qb + 0u, k, a10, b10, am, iq0 + 0u);
        sp_put(two, (t0 + 1u) * qb + 1u, k, a11, b11, am, iq0 + 1u);
        sp_put(two && 2u < qb, (t0 + 1u) * qb + 2u, k, a12, b12, am, iq0 + 2u);
        sp_put(two && 3u < qb, (t0 + 1u) * qb + 3u, k, a13, b13, am, iq0 + 3u);
    }
    workgroupBarrier();
    fft_core(tid, lpw);
    // the pair apart, each into its line
    let nb = so.nb;
    for (var e = tid; e < lpw * nb; e = e + FFT_WG) {
        let i = e & (lpw - 1u);
        let j = e >> log2lpw;
        let b = grp * per + (i >> log2qb);
        if (b < so.n_items) {
            let iq = iq0 + (i & (qb - 1u));
            let zj = fs[fsw(i * n + j)];
            let zr = fs[fsw(i * n + ((n - j) & (n - 1u)))];
            let ev = 0.5 * (zj + zr);
            let od = 0.5 * (zj - zr);
            // which of the two is odd: m' even here, so m alone says (m = fold · μ)
            let a_odd = ((u32(abs(mu_of(iq))) * so.fold) & 1u) == 1u;
            let at = ((b * nb + j) * L + am) * so.nc + iq;
            spec[at] = select(ev, od, a_odd);
            if (am + 1u < L) { spec[at + so.nc] = select(od, ev, a_odd); }
        }
    }
}

// Lines from here on run along m', one per (item, j, c): line = (b·nb + j)·nc + iq, and fp.n is
// half the orientation grid, h = n / 2.
//
// A real sequence x_a = Σ X_m' e^{−2πi m' a / n} (X_{n−m'} = conj X_m') comes out of a complex
// transform of half its length as z_p = x_{2p} + i x_{2p+1}, when that transform is given
//     Z_q = (X_q + X_{q+h}) + i e^{−2πi q / n} (X_q − X_{q+h}),   X_{q+h} = conj X_{h−q}.
// Here X_m' = V[m'] for m' < L (≤ h) and zero above. Load lpw such lines bit-reversed into fs.
fn a_load(tid: u32, line0: u32, lpw: u32) {
    let h = fp.n;
    let shift = 32u - fp.log2n;
    let L = so.bw;
    let tws = fp.tw_len / (2u * h);
    // (neighbouring invocations take neighbouring lines, which lie side by side in `spec`)
    let log2lpw = 10u - fp.log2n;
    for (var e = tid; e < lpw * h; e = e + FFT_WG) {
        let ll = e & (lpw - 1u);
        let q = e >> log2lpw;
        let line = line0 + ll;
        var z = vec2<f32>(0.0, 0.0);
        if (line < fp.n_lines) {
            let base = (line >> so.log2nc) * L * so.nc + (line & (so.nc - 1u));
            var x = vec2<f32>(0.0, 0.0);
            var y = vec2<f32>(0.0, 0.0);
            if (q < L) { x = spec[base + q * so.nc]; }
            if (q == 0u) {
                x.y = 0.0;
            } else if (h - q < L) {
                let v = spec[base + (h - q) * so.nc];
                y = vec2<f32>(v.x, -v.y);
            }
            let w = ftw[q * tws];
            let d = x - y;
            z = x + y + vec2<f32>(-(w.x * d.y + w.y * d.x), w.x * d.x - w.y * d.y);
        }
        fs[fsw((ll << fp.log2n) + (reverseBits(q) >> shift))] = z;
    }
}

var<workgroup> seg_v: array<f32, FFT_WG>;
var<workgroup> seg_i: array<u32, FFT_WG>;

// The largest normalized correlation among points p0, p0 + dp, … below p1 of line `line` (line
// ll of the workgroup), each holding two values of a; .y: its index along a, as a float's bits
// (the first of equal ones).
fn a_scan(line: u32, ll: u32, p0: u32, p1: u32, dp: u32) -> vec2<f32> {
    let per = so.nb * so.nc;
    let b = line / per;
    let rbase = (line - (b - b / so.rden_div) * per) * so.n;
    var bv = -3.4e38;
    var bi = 0u;
    for (var p = p0; p < p1; p = p + dp) {
        let z = fs[fsw((ll << fp.log2n) + p)];
        let even = z.x * rden[rbase + 2u * p];
        let odd = z.y * rden[rbase + 2u * p + 1u];
        if (even > bv) {
            bv = even;
            bi = 2u * p;
        }
        if (odd > bv) {
            bv = odd;
            bi = 2u * p + 1u;
        }
    }
    return vec2<f32>(bv, bitcast<f32>(bi));
}

fn a_keep(line: u32, bv: f32, ia: u32) {
    let per = so.nb * so.nc;
    cand_v[line] = bv;
    cand_i[line] = (line % per) * so.n + ia;
}

@compute @workgroup_size(FFT_WG)
fn a_peak(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let h = fp.n;
    let lpw = FFT_MAX / h;
    let line0 = (wid.y * fp.grid_x + wid.x) * lpw;
    a_load(tid, line0, lpw);
    workgroupBarrier();
    fft_core(tid, lpw);
    if (lpw >= FFT_WG) {
        // short lines: an invocation scans whole ones
        for (var ll = tid; ll < lpw; ll = ll + FFT_WG) {
            let line = line0 + ll;
            if (line < fp.n_lines) {
                let r = a_scan(line, ll, 0u, h, 1u);
                a_keep(line, r.x, bitcast<u32>(r.y));
            }
        }
    } else {
        // segs invocations side by side scan a line, each every segs-th point (neighbouring
        // invocations read neighbouring words); the line's first keeps the best of them
        let segs = FFT_WG / lpw;
        let ll = tid / segs;
        let seg = tid & (segs - 1u);
        let line = line0 + ll;
        var r = vec2<f32>(-3.4e38, 0.0);
        if (line < fp.n_lines) { r = a_scan(line, ll, seg, h, segs); }
        seg_v[tid] = r.x;
        seg_i[tid] = bitcast<u32>(r.y);
        workgroupBarrier();
        if (seg == 0u && line < fp.n_lines) {
            var bv = seg_v[tid];
            var bi = seg_i[tid];
            for (var s = 1u; s < segs; s = s + 1u) {
                let v = seg_v[tid + s];
                let i = seg_i[tid + s];
                if (v > bv || (v == bv && i < bi)) {
                    bv = v;
                    bi = i;
                }
            }
            a_keep(line, bv, bi);
        }
    }
}

// ── the transforms along c and a, and the peaks, one (item, j) slab at a time ──
// spectrum's lines along c and a_peak's along a together: a workgroup takes the whole slab of
// U[j; m', ·] (h · nc ≤ AM_SLOTS points) into its memory, transforms it along c there, builds
// a_load's input in place and transforms that along a, and scans it as a_peak does. The volume
// then goes to memory once, from spectrum, and is read once, here (where spectrum, fftLines and
// a_peak write and read it twice each). The engine takes this path where the slab fits; corr_at
// then transforms what it reads along c itself (so.u_in_spec). The arithmetic of each transform
// is fft_core's, in its order.
const AM_WG: u32 = 256u;
const AM_SLOTS: u32 = 2048u;
// the most (point, mirror) pairs an invocation holds while the slab is rearranged
const AM_PAIRS: u32 = 5u;
var<workgroup> fm: array<vec2<f32>, AM_SLOTS>;

// The slab's point (row: m', or the point along a; col: c), each row's columns permuted by the
// row (an XOR) so that the invocations reading along a row or down a column meet every bank.
fn am_at(row: u32, col: u32) -> u32 {
    return row * so.nc + (col ^ (row & (so.nc - 1u)));
}

// fft_core over `count` (at most 2^lgc) lines of length 2^lg in fm: along rows (`down` false:
// line = row, points its columns) or down columns (line = column, points its rows).
fn am_fft(tid: u32, count: u32, lgc: u32, lg: u32, down: bool) {
    let len = 1u << lg;
    let hn = len >> 1u;
    let qn = len >> 2u;
    let lmask = (1u << lgc) - 1u;
    var st0 = 0u;
    for (; st0 + 1u < lg; st0 = st0 + 2u) {
        let half = 1u << st0;
        let t1 = fp.tw_len >> (st0 + 1u);
        let t2 = fp.tw_len >> (st0 + 2u);
        for (var e = tid; e < (qn << lgc); e = e + AM_WG) {
            let line = e & lmask;
            if (line < count) {
                let r = e >> lgc;
                let j = r & (half - 1u);
                let p0 = ((r >> st0) << (st0 + 2u)) + j;
                // (rows and columns: the point steps `half` along the line)
                let row = select(line, p0, down);
                let col = select(p0, line, down);
                let i0 = am_at(row, col);
                let i1 = select(am_at(row, col + half), am_at(row + half, col), down);
                let i2 = select(am_at(row, col + 2u * half), am_at(row + 2u * half, col), down);
                let i3 = select(am_at(row, col + 3u * half), am_at(row + 3u * half, col), down);
                let w1 = ftw[j * t1];
                let x0 = fm[i0];
                let x2 = fm[i2];
                let b1 = twiddled(fm[i1], w1);
                let b3 = twiddled(fm[i3], w1);
                let a0 = x0 + b1;
                let a1 = x0 - b1;
                let a2 = x2 + b3;
                let a3 = x2 - b3;
                let c2 = twiddled(a2, ftw[j * t2]);
                let c3 = twiddled(a3, ftw[(j + half) * t2]);
                fm[i0] = a0 + c2;
                fm[i2] = a0 - c2;
                fm[i1] = a1 + c3;
                fm[i3] = a1 - c3;
            }
        }
        workgroupBarrier();
    }
    for (var st = st0; st < lg; st = st + 1u) {
        let half = 1u << st;
        let tstep = fp.tw_len >> (st + 1u);
        for (var e = tid; e < (hn << lgc); e = e + AM_WG) {
            let line = e & lmask;
            if (line < count) {
                let r = e >> lgc;
                let j = r & (half - 1u);
                let p0 = ((r >> st) << (st + 1u)) + j;
                let row = select(line, p0, down);
                let col = select(p0, line, down);
                let i0 = am_at(row, col);
                let i1 = select(am_at(row, col + half), am_at(row + half, col), down);
                let b = twiddled(fm[i1], ftw[j * tstep]);
                let a = fm[i0];
                fm[i0] = a + b;
                fm[i1] = a - b;
            }
        }
        workgroupBarrier();
    }
}

// a_load's point: Z from x and y (= conj X_{h−q}) at twiddle w
fn am_z(x: vec2<f32>, y: vec2<f32>, w: vec2<f32>) -> vec2<f32> {
    let d = x - y;
    return x + y + vec2<f32>(-(w.x * d.y + w.y * d.x), w.x * d.x - w.y * d.y);
}

@compute @workgroup_size(AM_WG)
fn a_peak_m(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let h = fp.n;
    let log2h = fp.log2n;
    let nc = so.nc;
    let log2nc = so.log2nc;
    let L = so.bw;
    let slab = wid.y * fp.grid_x + wid.x;
    let live = slab < so.n_items * so.nb;
    let base = slab * L * nc;
    // U, each row (m') bit-reversed along c, rows from L on zero
    let shift_c = 32u - log2nc;
    for (var e = tid; e < h * nc; e = e + AM_WG) {
        let row = e >> log2nc;
        let col = e & (nc - 1u);
        var v = vec2<f32>(0.0, 0.0);
        if (live && row < L) { v = spec[base + e]; }
        fm[am_at(row, reverseBits(col) >> shift_c)] = v;
    }
    workgroupBarrier();
    // along c (what fftLines does to spectrum's output): V, row q, column c
    am_fft(tid, L, log2h, log2nc, false);
    // a_load's points, down each column, bit-reversed for the transform along a: each from a
    // point and its mirror h − q, held until every invocation has read its own
    let tws = fp.tw_len / (2u * h);
    let shift_a = 32u - log2h;
    let pairs = nc * (h / 2u + 1u);
    var zq: array<vec2<f32>, AM_PAIRS>;
    var zr: array<vec2<f32>, AM_PAIRS>;
    for (var k = 0u; k < AM_PAIRS; k = k + 1u) {
        let e = tid + k * AM_WG;
        if (e < pairs) {
            let col = e & (nc - 1u);
            let q = e >> log2nc;
            var x = fm[am_at(q, col)];
            var y = vec2<f32>(0.0, 0.0);
            if (q == 0u) {
                x.y = 0.0;
            } else {
                let v = fm[am_at(h - q, col)];
                y = vec2<f32>(v.x, -v.y);
            }
            zq[k] = am_z(x, y, ftw[q * tws]);
            // its mirror's, from the same two points
            if (q != 0u && h - q != q) {
                let xm = fm[am_at(h - q, col)];
                let ym = fm[am_at(q, col)];
                zr[k] = am_z(xm, vec2<f32>(ym.x, -ym.y), ftw[(h - q) * tws]);
            }
        }
    }
    workgroupBarrier();
    for (var k = 0u; k < AM_PAIRS; k = k + 1u) {
        let e = tid + k * AM_WG;
        if (e < pairs) {
            let col = e & (nc - 1u);
            let q = e >> log2nc;
            fm[am_at(reverseBits(q) >> shift_a, col)] = zq[k];
            if (q != 0u && h - q != q) { fm[am_at(reverseBits(h - q) >> shift_a, col)] = zr[k]; }
        }
    }
    workgroupBarrier();
    // along a, down each column
    am_fft(tid, nc, log2nc, log2h, true);
    // each column's best, as a_scan finds it: segs invocations side by side to a column, each
    // every segs-th point
    let segs = AM_WG / nc;
    let col = tid / segs;
    let seg = tid & (segs - 1u);
    let line = slab * nc + col;
    let per = so.nb * nc;
    let b = line / per;
    let rbase = (line - (b - b / so.rden_div) * per) * so.n;
    var bv = -3.4e38;
    var bi = 0u;
    if (live) {
        for (var p = seg; p < h; p = p + segs) {
            let z = fm[am_at(p, col)];
            let even = z.x * rden[rbase + 2u * p];
            let odd = z.y * rden[rbase + 2u * p + 1u];
            if (even > bv) {
                bv = even;
                bi = 2u * p;
            }
            if (odd > bv) {
                bv = odd;
                bi = 2u * p + 1u;
            }
        }
    }
    workgroupBarrier();
    fm[tid] = vec2<f32>(bv, bitcast<f32>(bi));
    workgroupBarrier();
    if (live && seg == 0u) {
        for (var s = 1u; s < segs; s = s + 1u) {
            let o = fm[tid + s];
            let i = bitcast<u32>(o.y);
            if (o.x > bv || (o.x == bv && i < bi)) {
                bv = o.x;
                bi = i;
            }
        }
        a_keep(line, bv, bi);
    }
}

@compute @workgroup_size(FFT_WG)
fn a_volume(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let h = fp.n;
    let lpw = FFT_MAX / h;
    let line0 = (wid.y * fp.grid_x + wid.x) * lpw;
    a_load(tid, line0, lpw);
    workgroupBarrier();
    fft_core(tid, lpw);
    for (var e = tid; e < lpw * h; e = e + FFT_WG) {
        let p = e & (h - 1u);
        let line = line0 + (e >> fp.log2n);
        if (line < fp.n_lines) {
            let z = fs[fsw(e)];
            vol[line * so.n + 2u * p] = z.x;
            vol[line * so.n + 2u * p + 1u] = z.y;
        }
    }
}

@compute @workgroup_size(256)
fn rden_build(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let tid = flat_id(wid, lid, so.grid_x, 256u);
    let per_item = so.nb * so.n * so.nc;
    if (tid >= so.n_items * per_item) { return; }
    let a = max(area[(tid / per_item) * 4u], 1e-12);
    let m = vol_f[tid];
    let v = vol_f2[tid] - m * m / a;
    // a window the master is flat under has no defined correlation: score it 0
    var r = 0.0;
    if (v > 1e-6 * a) { r = inverseSqrt(v); }
    rden_out[tid] = r;
}

var<workgroup> rv: array<f32, 256>;
var<workgroup> ri: array<u32, 256>;

@compute @workgroup_size(256)
fn peak_reduce(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let b = wid.y * so.grid_x + wid.x;
    let per = so.nb * so.nc;
    var bv = -3.4e38;
    var bi = 0u;
    if (b < so.n_items) {
        for (var i = lid; i < per; i = i + 256u) {
            let v = cand_v[b * per + i];
            if (v > bv) {
                bv = v;
                bi = cand_i[b * per + i];
            }
        }
    }
    rv[lid] = bv;
    ri[lid] = bi;
    workgroupBarrier();
    for (var off = 128u; off > 0u; off = off >> 1u) {
        if (lid < off) {
            let o = lid + off;
            if (rv[o] > rv[lid] || (rv[o] == rv[lid] && ri[o] < ri[lid])) {
                rv[lid] = rv[o];
                ri[lid] = ri[o];
            }
        }
        workgroupBarrier();
    }
    if (lid == 0u && b < so.n_items) {
        best[b * 2u] = bitcast<u32>(rv[0]);
        best[b * 2u + 1u] = ri[0];
    }
}

// V[j; m', c] of item b at row (b·nb + j)·L + m' of spec: as it is, or (so.u_in_spec) U's
// transform along c taken directly.
fn v_at(row: u32, c: u32) -> vec2<f32> {
    let base = row * so.nc;
    if (so.u_in_spec == 0u) { return spec[base + c]; }
    let ts = fp.tw_len / so.nc;
    var acc = vec2<f32>(0.0, 0.0);
    for (var x = 0u; x < so.nc; x = x + 1u) { acc = acc + cmul(spec[base + x], ftw[((x * c) & (so.nc - 1u)) * ts]); }
    return acc;
}

// The normalized correlation of item b at volume cell (j, iq, ia), summed directly along m'.
fn corr_at(b: u32, j: u32, iq: u32, ia: u32) -> f32 {
    let row0 = (b * so.nb + j) * so.bw;
    var acc = v_at(row0, iq).x;
    for (var m = 1u; m < so.bw; m = m + 1u) {
        let s = v_at(row0 + m, iq);
        let ang = TAU * f32((m * ia) & (so.n - 1u)) / f32(so.n);
        acc = acc + 2.0 * (s.x * cos(ang) + s.y * sin(ang));
    }
    return acc * rden[((((b / so.rden_div) * so.nb + j) * so.nc + iq) * so.n) + ia];
}

// Offset of a parabola's maximum through three equally spaced values, within half a step.
fn vertex(lo: f32, mid: f32, hi: f32) -> f32 {
    let den = lo - 2.0 * mid + hi;
    if (den >= -1e-12) { return 0.0; }
    return clamp(0.5 * (lo - hi) / den, -0.5, 0.5);
}

@compute @workgroup_size(64)
fn peak_finish(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let b = flat_id(wid, lid, so.grid_x, 64u);
    if (b >= so.n_items) { return; }
    let idx = best[b * 2u + 1u];
    let ia = idx & (so.n - 1u);
    let iq = (idx / so.n) & (so.nc - 1u);
    let j = idx / (so.n * so.nc);
    let v0 = bitcast<f32>(best[b * 2u]);
    let da = vertex(corr_at(b, j, iq, (ia + so.n - 1u) & (so.n - 1u)), v0, corr_at(b, j, iq, (ia + 1u) & (so.n - 1u)));
    let dc = vertex(corr_at(b, j, (iq + so.nc - 1u) & (so.nc - 1u), ia), v0, corr_at(b, j, (iq + 1u) & (so.nc - 1u), ia));
    var dj = 0.0;
    if (j > 0u && j < so.n / 2u) { dj = vertex(corr_at(b, j - 1u, iq, ia), v0, corr_at(b, j + 1u, iq, ia)); }
    put_peak(b, ia, j, iq, da, dj, dc, v0);
}

// peak_finish where spec holds U (a_peak_m's path): a workgroup an item, an invocation an m'
// at a time, each taking V along c for the five lines its neighbours need and its terms of their
// six sums; the workgroup adds them.
var<workgroup> pf: array<f32, 384>;

@compute @workgroup_size(64)
fn peak_finish_m(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let b = wid.y * so.grid_x + wid.x;
    let live = b < so.n_items;
    let bb = min(b, so.n_items - 1u);
    let idx = best[bb * 2u + 1u];
    let ia = idx & (so.n - 1u);
    let iq = (idx / so.n) & (so.nc - 1u);
    let j = idx / (so.n * so.nc);
    let inner = j > 0u && j < so.n / 2u;
    let iqm = (iq + so.nc - 1u) & (so.nc - 1u);
    let iqp = (iq + 1u) & (so.nc - 1u);
    let iam = (ia + so.n - 1u) & (so.n - 1u);
    let iap = (ia + 1u) & (so.n - 1u);
    var s: array<f32, 6>;
    for (var k = 0u; k < 6u; k = k + 1u) { s[k] = 0.0; }
    let row0 = (bb * so.nb + j) * so.bw;
    for (var m = lid; m < so.bw; m = m + 64u) {
        let f = select(2.0, 1.0, m == 0u);
        let v0 = v_at(row0 + m, iq);
        let vm = v_at(row0 + m, iqm);
        let vp = v_at(row0 + m, iqp);
        s[0] = s[0] + f * a_term(v0, m, iam);
        s[1] = s[1] + f * a_term(v0, m, iap);
        s[2] = s[2] + f * a_term(vm, m, ia);
        s[3] = s[3] + f * a_term(vp, m, ia);
        if (inner) {
            s[4] = s[4] + f * a_term(v_at(row0 - so.bw + m, iq), m, ia);
            s[5] = s[5] + f * a_term(v_at(row0 + so.bw + m, iq), m, ia);
        }
    }
    for (var k = 0u; k < 6u; k = k + 1u) { pf[k * 64u + lid] = s[k]; }
    workgroupBarrier();
    for (var off = 32u; off > 0u; off = off >> 1u) {
        if (lid < off) {
            for (var k = 0u; k < 6u; k = k + 1u) { pf[k * 64u + lid] = pf[k * 64u + lid] + pf[k * 64u + lid + off]; }
        }
        workgroupBarrier();
    }
    if (!live || lid != 0u) { return; }
    let v0 = bitcast<f32>(best[b * 2u]);
    let rb = (b / so.rden_div) * so.nb;
    let da = vertex(pf[0] * rden_at(rb, j, iq, iam), v0, pf[64] * rden_at(rb, j, iq, iap));
    let dc = vertex(pf[128] * rden_at(rb, j, iqm, ia), v0, pf[192] * rden_at(rb, j, iqp, ia));
    var dj = 0.0;
    if (inner) { dj = vertex(pf[256] * rden_at(rb, j - 1u, iq, ia), v0, pf[320] * rden_at(rb, j + 1u, iq, ia)); }
    put_peak(b, ia, j, iq, da, dj, dc, v0);
}

// Re(V e^{i m a}) at grid point ia along a (corr_at's term)
fn a_term(v: vec2<f32>, m: u32, ia: u32) -> f32 {
    let ang = TAU * f32((m * ia) & (so.n - 1u)) / f32(so.n);
    return v.x * cos(ang) + v.y * sin(ang);
}

fn rden_at(rb: u32, j: u32, iq: u32, ia: u32) -> f32 {
    return rden[(((rb + j) * so.nc + iq) * so.n) + ia];
}

// The peak's orientation, interpolated by (da, dj, dc) grid steps, into `peaks`.
fn put_peak(b: u32, ia: u32, j: u32, iq: u32, da: f32, dj: f32, dc: f32, v0: f32) {
    let step = TAU / f32(so.n);
    let a = (f32(ia) + da) * step;
    let bb = (f32(j) + dj) * step;
    let c = (f32(iq) + dc) * step;
    // S = Rz(a) Ry(bb) Rz(c); the orientation is its inverse
    let qa = vec4<f32>(cos(0.5 * a), 0.0, 0.0, sin(0.5 * a));
    let qb = vec4<f32>(cos(0.5 * bb), 0.0, sin(0.5 * bb), 0.0);
    let qc = vec4<f32>(cos(0.5 * c), 0.0, 0.0, sin(0.5 * c));
    var q = qconj(qmul(qa, qmul(qb, qc)));
    if (q.x < 0.0) { q = -q; }
    let o = b * 8u;
    peaks[o] = q.x;
    peaks[o + 1u] = q.y;
    peaks[o + 2u] = q.z;
    peaks[o + 3u] = q.w;
    peaks[o + 4u] = v0;
    peaks[o + 5u] = a;
    peaks[o + 6u] = bb;
    peaks[o + 7u] = c;
}
