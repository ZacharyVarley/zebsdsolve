// Patterns onto the sphere and into spherical harmonics (after common.wgsl).
//
// An item is one (geometry, pattern) pair: the detector's position t for that pattern (the
// pattern centre, shifted by where the beam was on the sample) and which pattern to read. The
// sphere is sampled on a Gauss–Legendre grid, nθ rings of nφ points, in the sample frame.
//
//   backproject    each grid direction → where its ray meets the detector → bilinear pattern
//                  value (.x) and whether it landed on the detector and inside the mask (.y)
//   window_center  subtract the window's mean (quadrature weighted), zero outside the window
//   (fft.wgsl)     forward FFT of every ring along φ
//   legendre       contract the ring spectra with the weighted Legendre table → a_lm; a ring
//                  and its mirror in the equator together, since P̄_l^m(−x) = (−1)^{l+m} P̄_l^m(x)
//   sh_norm        scale each item's coefficients to unit norm
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

@group(0) @binding(0) var<uniform> sp: SphP;
// (t in pixels, pattern index)
@group(0) @binding(1) var<storage, read> items: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> pats: array<f32>;
// (cos θ, quadrature weight) per ring
@group(0) @binding(3) var<storage, read> ring: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> grid: array<vec2<f32>>;
// w_y · P̄_l^m(cos θ_y) · 2π / nφ at [y · n_pairs + pair] for the northern rings y < nθ / 2: the
// pair innermost, because neighbouring invocations hold neighbouring pairs
@group(0) @binding(5) var<storage, read> wplm: array<f32>;
// (m << 16) | l per pair
@group(0) @binding(6) var<storage, read> pairs: array<u32>;
@group(0) @binding(7) var<storage, read_write> alm: array<vec2<f32>>;
// per item: window area ∫w dΩ, window mean, harmonic norm, a_00
@group(0) @binding(8) var<storage, read_write> stats: array<f32>;

@compute @workgroup_size(256)
fn backproject(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let tid = flat_id(wid, lid, sp.grid_x, 256u);
    let per = sp.n_theta * sp.n_phi;
    if (tid >= sp.n_items * per) { return; }
    let b = tid / per;
    let r = tid % per;
    let ct = ring[r / sp.n_phi].x;
    let st = sqrt(max(1.0 - ct * ct, 0.0));
    let phi = TAU * f32(r % sp.n_phi) / f32(sp.n_phi);
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
    grid[tid] = vec2<f32>(val, w);
}

var<workgroup> wsum: array<vec2<f32>, 256>;

@compute @workgroup_size(256)
fn window_center(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let b = wid.y * sp.grid_x + wid.x;
    let per = sp.n_theta * sp.n_phi;
    let live = b < sp.n_items;
    var acc = vec2<f32>(0.0, 0.0);
    if (live) {
        for (var i = lid; i < per; i = i + 256u) {
            let g = grid[b * per + i];
            let wq = ring[i / sp.n_phi].y;
            acc = acc + vec2<f32>(g.x * g.y * wq, g.y * wq);
        }
    }
    wsum[lid] = acc;
    workgroupBarrier();
    for (var off = 128u; off > 0u; off = off >> 1u) {
        if (lid < off) { wsum[lid] = wsum[lid] + wsum[lid + off]; }
        workgroupBarrier();
    }
    let tot = wsum[0];
    if (!live) { return; }
    var mean = 0.0;
    if ((sp.flags & 1u) != 0u) { mean = tot.x / max(tot.y, 1e-20); }
    for (var i = lid; i < per; i = i + 256u) {
        let g = grid[b * per + i];
        grid[b * per + i] = vec2<f32>((g.x - mean) * g.y, 0.0);
    }
    if (lid == 0u) {
        stats[b * 4u] = tot.y * TAU / f32(sp.n_phi);
        stats[b * 4u + 1u] = mean;
    }
}

@compute @workgroup_size(256)
fn legendre(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let tid = flat_id(wid, lid, sp.grid_x, 256u);
    if (tid >= sp.n_items * sp.n_pairs) { return; }
    let b = tid / sp.n_pairs;
    let p = tid % sp.n_pairs;
    let ml = pairs[p];
    let base = b * sp.n_theta * sp.n_phi + (ml >> 16u);
    let south = select(1.0, -1.0, (((ml >> 16u) + (ml & 0xffffu)) & 1u) == 1u);
    let last = base + (sp.n_theta - 1u) * sp.n_phi;
    var acc = vec2<f32>(0.0, 0.0);
    for (var y = 0u; y < sp.n_theta / 2u; y = y + 1u) {
        acc = acc + wplm[y * sp.n_pairs + p] * (grid[base + y * sp.n_phi] + south * grid[last - y * sp.n_phi]);
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
        stats[b * 4u + 2u] = nrm;
        stats[b * 4u + 3u] = alm[b * sp.n_pairs].x;
    }
}
