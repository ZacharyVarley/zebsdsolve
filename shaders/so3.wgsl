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
//   spectrum     T for k ≥ 0 and m' ≥ 0, a column along k at a time into the workgroup's
//                memory, and from there at once its transform along k,
//                U[j; m', m] = Σ_k T e^{−i k b_j} for b_j = 2π j / n, j ≤ n / 2. T itself is
//                never stored.
//   (fft.wgsl)   forward FFT along m → V[j; m', c]
//   a_peak       the last axis: C(a) = V[0] + 2 Re Σ_{m' > 0} V[m'] e^{−i m' a} at all n values
//                of a from one complex FFT of half that length (the real values come out in
//                pairs, as real and imaginary parts), times the reciprocal denominator of the
//                normalized correlation, and the maximum of each (b, c) line
//   a_volume     the same lines written out instead (the denominators are built from these)
//   rden_build   1 / sqrt(⟨w, f²⟩ − ⟨w, f⟩² / ∫w) from the window's correlations with f and f²
//   peak_reduce  the best line per item
//   peak_finish  parabolic interpolation around the peak and the orientation quaternion
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
    pad0: u32,
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

// Lines here are columns along k, one per (item, m', m): line = (b·L + m')·nc + iq.
@compute @workgroup_size(FFT_WG)
fn spectrum(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let n = fp.n;
    let lpw = FFT_MAX / n;
    let shift = 32u - fp.log2n;
    let line0 = (wid.y * fp.grid_x + wid.x) * lpw;
    let L = so.bw;
    // k runs over −L < k < L in a transform of length n ≥ 2L: zero between
    for (var e = tid; e < lpw * n; e = e + FFT_WG) {
        let p = e & (n - 1u);
        if (p >= L && p <= n - L) { fs[(e - p) + (reverseBits(p) >> shift)] = vec2<f32>(0.0, 0.0); }
    }
    for (var e = tid; e < lpw * L; e = e + FFT_WG) {
        let col = e / L;
        let k = e - col * L;
        let line = line0 + col;
        var acc = vec2<f32>(0.0, 0.0);
        var odd = false;
        if (line < fp.n_lines) {
            let iq = line & (so.nc - 1u);
            let rest = line >> so.log2nc;
            let b = rest / L;
            let am = rest - b * L;
            // the master's signed order from its wrapped index
            var mu = i32(iq);
            if (iq >= (so.nc + 1u) / 2u) { mu = i32(iq) - i32(so.nc); }
            let amu = u32(abs(mu));
            if (amu <= so.mu_max) {
                // from the same l for sixteen k at a time (the tables are zero below l = k)
                var l = max(k & ~15u, max(am, amu * so.fold));
                if (so.l_step == 2u) { l = l + (l & 1u); }
                let wbase = u32(mu + i32(so.mu_max)) * L * L + k;
                let dbase = am * L * L + k;
                let gbase = b * so.n_pairs + am * L - (am * (am + 1u)) / 2u;
                for (; l < L; l = l + so.l_step) {
                    let g = coef[gbase + l];
                    acc = acc + cmul(wmaster[wbase + l * L], vec2<f32>(g.x, -g.y)) * dq[dbase + l * L];
                }
                // i^{m'}
                switch (am & 3u) {
                    case 1u: { acc = vec2<f32>(-acc.y, acc.x); }
                    case 2u: { acc = -acc; }
                    case 3u: { acc = vec2<f32>(acc.y, -acc.x); }
                    default: {}
                }
                odd = ((am + amu * so.fold) & 1u) == 1u;
            }
        }
        let at = col * n;
        fs[at + (reverseBits(k) >> shift)] = acc;
        // T[−k] = (−1)^{m'+m} T[k]
        if (k > 0u) { fs[at + (reverseBits(n - k) >> shift)] = select(acc, -acc, odd); }
    }
    workgroupBarrier();
    fft_core(tid, lpw);
    let nb = so.nb;
    for (var e = tid; e < lpw * nb; e = e + FFT_WG) {
        let col = e / nb;
        let j = e - col * nb;
        let line = line0 + col;
        if (line < fp.n_lines) {
            let rest = line >> so.log2nc;
            let b = rest / L;
            spec[((b * nb + j) * L + (rest - b * L)) * so.nc + (line & (so.nc - 1u))] = fs[col * n + j];
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
    for (var e = tid; e < lpw * h; e = e + FFT_WG) {
        let q = e & (h - 1u);
        let line = line0 + (e >> fp.log2n);
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
        fs[(e - q) + (reverseBits(q) >> shift)] = z;
    }
}

var<workgroup> seg_v: array<f32, FFT_WG>;
var<workgroup> seg_i: array<u32, FFT_WG>;

// The largest normalized correlation among points [p0, p1) of line `line` (line ll of the
// workgroup), each holding two values of a; .y: its index along a, as a float's bits.
fn a_scan(line: u32, ll: u32, p0: u32, p1: u32) -> vec2<f32> {
    let per = so.nb * so.nc;
    let b = line / per;
    let rbase = (line - (b - b / so.rden_div) * per) * so.n;
    var bv = -3.4e38;
    var bi = 0u;
    for (var p = p0; p < p1; p = p + 1u) {
        let z = fs[(ll << fp.log2n) + p];
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
                let r = a_scan(line, ll, 0u, h);
                a_keep(line, r.x, bitcast<u32>(r.y));
            }
        }
    } else {
        // every invocation scans a piece of a line; the line's first one keeps the best piece
        let segs = FFT_WG / lpw;
        let ll = tid & (lpw - 1u);
        let seg = tid / lpw;
        let len = h / segs;
        let line = line0 + ll;
        var r = vec2<f32>(-3.4e38, 0.0);
        if (line < fp.n_lines) { r = a_scan(line, ll, seg * len, (seg + 1u) * len); }
        seg_v[tid] = r.x;
        seg_i[tid] = bitcast<u32>(r.y);
        workgroupBarrier();
        if (tid < lpw && line < fp.n_lines) {
            var bv = seg_v[tid];
            var bi = seg_i[tid];
            for (var s = 1u; s < segs; s = s + 1u) {
                let v = seg_v[tid + s * lpw];
                if (v > bv) {
                    bv = v;
                    bi = seg_i[tid + s * lpw];
                }
            }
            a_keep(line, bv, bi);
        }
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
            let z = fs[e];
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

// The normalized correlation of item b at volume cell (j, iq, ia), summed directly along m'.
fn corr_at(b: u32, j: u32, iq: u32, ia: u32) -> f32 {
    let base = (b * so.nb + j) * so.bw * so.nc + iq;
    var acc = spec[base].x;
    for (var m = 1u; m < so.bw; m = m + 1u) {
        let s = spec[base + m * so.nc];
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
