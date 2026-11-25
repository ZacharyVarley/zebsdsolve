// Patterns onto the sphere and into spherical harmonics (after common.wgsl and fft.wgsl).
//
// An item is one (geometry, pattern) pair: the detector's position t for that pattern (the
// pattern centre, shifted by where the beam was on the sample) and which pattern to read. The
// sphere is sampled on a Gauss–Legendre grid, nθ rings of nφ points, in the sample frame: each
// grid direction → where its ray meets the detector → the bilinear pattern value and whether
// it landed on the detector and inside the mask (the window). The grid itself is never stored.
//
//   sphere_sums    the window's quadrature-weighted sums (∫w, ∫w·value) over a chunk of an
//                  item's grid, for its mean
//   rings          each ring with its mirror in the equator, the window mean subtracted and
//                  zero outside the window, as one complex line (the ring real, its mirror
//                  imaginary) transformed along φ; the two apart again, and of their orders
//                  m < L the sum and the difference, what the Legendre sums take
//   legendre       contract the ring spectra with the weighted Legendre table → a_lm; a ring
//                  and its mirror together, since P̄_l^m(−x) = (−1)^{l+m} P̄_l^m(x)
//   sh_norm        scale each item's coefficients to unit norm (and its window's area and mean
//                  into its stats)
//
// Coefficients are stored for m ≥ 0 at pair(m, l) = m·L − m(m−1)/2 + (l − m).

struct SphP {
    n_items: u32,
    n_theta: u32,
    n_phi: u32,
    bw: u32,
    pat_h: u32,
    pat_w: u32,
    n_pairs: u32,
    grid_x: u32,
    // 1: subtract the window mean   2: scale to unit harmonic norm
    // 4: project the window itself (ones) instead of a pattern
    flags: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    // the detector's row direction, column direction and normal, in the sample frame
    c0: vec4<f32>,
    c1: vec4<f32>,
    c2: vec4<f32>,
    // circular mask: centre (row, col), squared radius in pixels (≤ 0: none)
    mask: vec4<f32>,
};

// (fft.wgsl has bindings 0 to 2)
@group(0) @binding(3) var<uniform> sp: SphP;
// (t in pixels, pattern index)
@group(0) @binding(4) var<storage, read> items: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> pats: array<f32>;
// (cos θ, quadrature weight) per ring
@group(0) @binding(6) var<storage, read> ring: array<vec2<f32>>;
// per item and northern ring y < nθ / 2, for each order m < L: the ring's spectrum plus its
// mirror's (.xy) and minus it (.zw), at [(b · nθ / 2 + y) · L + m]
@group(0) @binding(7) var<storage, read_write> spectra: array<vec4<f32>>;
// w_y · P̄_l^m(cos θ_y) · 2π / nφ at [y · n_pairs + pair] for the northern rings y < nθ / 2: the
// pair innermost, because neighbouring invocations hold neighbouring pairs
@group(0) @binding(8) var<storage, read> wplm: array<f32>;
// (m << 16) | l per pair
@group(0) @binding(9) var<storage, read> pairs: array<u32>;
@group(0) @binding(10) var<storage, read_write> alm: array<vec2<f32>>;
// per item: window area ∫w dΩ, window mean, harmonic norm, a_00
@group(0) @binding(11) var<storage, read_write> stats: array<f32>;
// per item and chunk of its grid (SUMS_CHUNK points): Σ w·wq·value, Σ w·wq
@group(0) @binding(12) var<storage, read_write> partials: array<vec2<f32>>;

// The grid's point (ring y, point j along φ) of item b: (pattern value, window).
fn sample(b: u32, y: u32, j: u32) -> vec2<f32> {
    let ct = ring[y].x;
    let st = sqrt(max(1.0 - ct * ct, 0.0));
    let phi = TAU * f32(j) / f32(sp.n_phi);
    let d = vec3<f32>(st * cos(phi), st * sin(phi), ct);
    let it = items[b];
    let nd = dot(sp.c2.xyz, d);
    var val = 0.0;
    var w = 0.0;
    if (nd > 1e-4) {
        // the ray λ d meets the detector plane {c0 row + c1 col + t}
        let h = (dot(sp.c2.xyz, it.xyz) / nd) * d - it.xyz;
        let row = dot(sp.c0.xyz, h);
        let col = dot(sp.c1.xyz, h);
        let hmax = f32(sp.pat_h - 1u);
        let wmax = f32(sp.pat_w - 1u);
        let dr = row - sp.mask.x;
        let dc = col - sp.mask.y;
        if (row >= 0.0 && row <= hmax && col >= 0.0 && col <= wmax && (sp.mask.z <= 0.0 || dr * dr + dc * dc <= sp.mask.z)) {
            w = 1.0;
            val = 1.0;
            if ((sp.flags & 4u) == 0u) {
                let r0 = min(u32(row), sp.pat_h - 2u);
                let q0 = min(u32(col), sp.pat_w - 2u);
                let fr = row - f32(r0);
                let fc = col - f32(q0);
                let o = u32(it.w) * sp.pat_h * sp.pat_w + r0 * sp.pat_w + q0;
                let top = mix(pats[o], pats[o + 1u], fc);
                let bot = mix(pats[o + sp.pat_w], pats[o + sp.pat_w + 1u], fc);
                val = mix(top, bot, fr);
            }
        }
    }
    return vec2<f32>(val, w);
}

var<workgroup> wsum: array<vec2<f32>, 256>;

// points of an item's grid a sums workgroup takes (a grid holds a whole number of them, or is
// smaller and taken whole)
const SUMS_CHUNK: u32 = 2048u;

fn sums_chunks() -> u32 {
    let per = sp.n_theta * sp.n_phi;
    return per / min(per, SUMS_CHUNK);
}

@compute @workgroup_size(256)
fn sphere_sums(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let w = wid.y * sp.grid_x + wid.x;
    let per = sp.n_theta * sp.n_phi;
    let chunk = min(per, SUMS_CHUNK);
    let nch = per / chunk;
    let b = w / nch;
    let r0 = (w - b * nch) * chunk;
    var acc = vec2<f32>(0.0, 0.0);
    if (b < sp.n_items) {
        for (var i = lid; i < chunk; i = i + 256u) {
            let r = r0 + i;
            let y = r / sp.n_phi;
            let g = sample(b, y, r - y * sp.n_phi);
            let wq = ring[y].y;
            acc = acc + vec2<f32>(g.x * g.y * wq, g.y * wq);
        }
    }
    wsum[lid] = acc;
    workgroupBarrier();
    for (var off = 128u; off > 0u; off = off >> 1u) {
        if (lid < off) { wsum[lid] = wsum[lid] + wsum[lid + off]; }
        workgroupBarrier();
    }
    if (lid == 0u && b < sp.n_items) { partials[w] = wsum[0]; }
}

// item b's window sums (Σ w·wq·value, Σ w·wq) from its chunks, and its window mean
fn window_sums(b: u32, nch: u32) -> vec2<f32> {
    var tot = vec2<f32>(0.0, 0.0);
    for (var c = 0u; c < nch; c = c + 1u) { tot = tot + partials[b * nch + c]; }
    return tot;
}

fn window_mean(tot: vec2<f32>) -> f32 {
    if ((sp.flags & 1u) != 0u) { return tot.x / max(tot.y, 1e-20); }
    return 0.0;
}

// per line of a rings workgroup (at most FFT_MAX / 32 of them, nφ ≥ 32): its item's window mean
var<workgroup> rmean: array<f32, 64>;

// Lines here are (item, northern ring y): line b · nθ / 2 + y, of length nφ (fp.n).
@compute @workgroup_size(FFT_WG)
fn rings(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let n = fp.n;
    let lpw = FFT_MAX / n;
    let shift = 32u - fp.log2n;
    let nh = sp.n_theta / 2u;
    let line0 = (wid.y * fp.grid_x + wid.x) * lpw;
    let nch = sums_chunks();
    // each line's window mean, from its item's sums
    for (var i = tid; i < lpw; i = i + FFT_WG) {
        let line = line0 + i;
        var mean = 0.0;
        if (line < fp.n_lines) { mean = window_mean(window_sums(line / nh, nch)); }
        rmean[i] = mean;
    }
    workgroupBarrier();
    // the ring (real) and its mirror (imaginary), centred and windowed, bit-reversed
    for (var e = tid; e < lpw * n; e = e + FFT_WG) {
        let j = e & (n - 1u);
        let i = e >> fp.log2n;
        let line = line0 + i;
        var v = vec2<f32>(0.0, 0.0);
        if (line < fp.n_lines) {
            let b = line / nh;
            let y = line - b * nh;
            let mean = rmean[i];
            let gn = sample(b, y, j);
            let gs = sample(b, sp.n_theta - 1u - y, j);
            v = vec2<f32>((gn.x - mean) * gn.y, (gs.x - mean) * gs.y);
        }
        fs[fsw((e - j) + (reverseBits(j) >> shift))] = v;
    }
    workgroupBarrier();
    fft_core(tid, lpw);
    // the two apart: Z = N + i S for real N, S gives N_m = (Z_m + conj Z_{n−m}) / 2 and
    // S_m = (Z_m − conj Z_{n−m}) / 2i
    let L = sp.bw;
    for (var e = tid; e < lpw * L; e = e + FFT_WG) {
        let i = e / L;
        let m = e - i * L;
        let line = line0 + i;
        if (line < fp.n_lines) {
            let z = fs[fsw(i * n + m)];
            let zr = fs[fsw(i * n + ((n - m) & (n - 1u)))];
            let rn = 0.5 * vec2<f32>(z.x + zr.x, z.y - zr.y);
            let dd = vec2<f32>(z.x - zr.x, z.y + zr.y);
            let rs = 0.5 * vec2<f32>(dd.y, -dd.x);
            spectra[line * L + m] = vec4<f32>(rn + rs, rn - rs);
        }
    }
}

@compute @workgroup_size(256)
fn legendre(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let tid = flat_id(wid, lid, sp.grid_x, 256u);
    if (tid >= sp.n_items * sp.n_pairs) { return; }
    let b = tid / sp.n_pairs;
    let p = tid % sp.n_pairs;
    let ml = pairs[p];
    // the mirror counts with (−1)^{l+m}
    let odd = (((ml >> 16u) + (ml & 0xffffu)) & 1u) == 1u;
    let nh = sp.n_theta / 2u;
    let base = b * nh * sp.bw + (ml >> 16u);
    var acc = vec2<f32>(0.0, 0.0);
    for (var y = 0u; y < nh; y = y + 1u) {
        let r = spectra[base + y * sp.bw];
        acc = acc + wplm[y * sp.n_pairs + p] * select(r.xy, r.zw, odd);
    }
    alm[tid] = acc;
}

@compute @workgroup_size(256)
fn sh_norm(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let b = wid.y * sp.grid_x + wid.x;
    let live = b < sp.n_items;
    var e = 0.0;
    if (live) {
        for (var p = lid; p < sp.n_pairs; p = p + 256u) {
            let a = alm[b * sp.n_pairs + p];
            var k = 2.0;
            if ((pairs[p] >> 16u) == 0u) { k = 1.0; }
            e = e + k * dot(a, a);
        }
    }
    wsum[lid] = vec2<f32>(e, 0.0);
    workgroupBarrier();
    for (var off = 128u; off > 0u; off = off >> 1u) {
        if (lid < off) { wsum[lid] = wsum[lid] + wsum[lid + off]; }
        workgroupBarrier();
    }
    let nrm = sqrt(max(wsum[0].x, 1e-30));
    if (!live) { return; }
    if ((sp.flags & 2u) != 0u) {
        for (var p = lid; p < sp.n_pairs; p = p + 256u) {
            alm[b * sp.n_pairs + p] = alm[b * sp.n_pairs + p] / nrm;
        }
    }
    if (lid == 0u) {
        // (the window's area and mean here rather than in rings, which binds six buffers)
        let tot = window_sums(b, sums_chunks());
        stats[b * 4u] = tot.y * TAU / f32(sp.n_phi);
        stats[b * 4u + 1u] = window_mean(tot);
        stats[b * 4u + 2u] = nrm;
        stats[b * 4u + 3u] = alm[b * sp.n_pairs].x;
    }
}
