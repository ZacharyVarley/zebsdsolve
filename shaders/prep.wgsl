// Pattern preprocessing on a batch of images [n][h][w] of f32 (after common.wgsl). The same
// kernels filter experimental patterns and, as two images, a master pattern's hemispheres.
//
//   load         stored pixels (8 or 16 bit integers or f32, packed in words) → f32
//   background   subtract a static background image
//   blur_rows, blur_cols   separable Gaussian, renormalized at the edges
//   box_sum, coarse_rows, coarse_cols, spread_rows, spread_cols   the same blur for a wide
//                Gaussian, from sums over boxes of pixels (see there)
//   combine      a − b, a / b (dynamic background, band pass)
//   stretch      each image to [0, 1] between its extremes
//   clahe_hist, clahe_cdf, clahe_apply   contrast-limited adaptive histogram equalization
//   normalize    zero mean and unit norm under the circular mask, zero outside it

struct PrepP {
    n: u32,
    h: u32,
    w: u32,
    grid_x: u32,
    // load: pixel type (0: u8, 1: u16, 2: f32)
    pixel: u32,
    // combine: 0 subtract, 1 divide
    op: u32,
    // CLAHE: tiles per side and histogram levels
    tiles: u32,
    bins: u32,
    // blur: taps on each side
    radius: u32,
    // CLAHE: rows and columns of reflected padding that make the image a whole number of
    // tiles, and a tile's size
    pad_h: u32,
    pad_w: u32,
    tile_h: u32,
    tile_w: u32,
    // wide blur: the side of a box in pixels, and the cells of margin around the coarse grid
    bin: u32,
    cpad: u32,
    pad2: u32,
    // blur: 1 / (2 σ²); CLAHE: clip limit (multiples of the mean count per level); wide blur:
    // 1 / (2 σ²) of the Gaussian that spreads the coarse grid back over the pixels
    k: vec4<f32>,
    // mask centre (row, col), squared radius in pixels (≤ 0: none)
    mask: vec4<f32>,
};

@group(0) @binding(0) var<uniform> pp: PrepP;
@group(0) @binding(1) var<storage, read> raw: array<u32>;
@group(0) @binding(2) var<storage, read_write> img: array<f32>;
@group(0) @binding(3) var<storage, read> src: array<f32>;
@group(0) @binding(4) var<storage, read> other: array<f32>;
@group(0) @binding(5) var<storage, read_write> hist: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> cdf: array<f32>;
@group(0) @binding(7) var<storage, read> cdf_in: array<f32>;
// wide blur: per coarse cell, a weighted sum of pixels and the sum of the weights
@group(0) @binding(8) var<storage, read> cin: array<vec2<f32>>;
@group(0) @binding(9) var<storage, read_write> cout: array<vec2<f32>>;

@compute @workgroup_size(256)
fn load(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let tid = flat_id(wid, lid, pp.grid_x, 256u);
    if (tid >= pp.n * pp.h * pp.w) { return; }
    var v = 0.0;
    if (pp.pixel == 0u) {
        v = f32((raw[tid >> 2u] >> ((tid & 3u) * 8u)) & 0xffu);
    } else if (pp.pixel == 1u) {
        v = f32((raw[tid >> 1u] >> ((tid & 1u) * 16u)) & 0xffffu);
    } else {
        v = bitcast<f32>(raw[tid]);
    }
    img[tid] = v;
}

@compute @workgroup_size(256)
fn background(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let tid = flat_id(wid, lid, pp.grid_x, 256u);
    let hw = pp.h * pp.w;
    if (tid >= pp.n * hw) { return; }
    img[tid] = img[tid] - other[tid % hw];
}

@compute @workgroup_size(256)
fn combine(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let tid = flat_id(wid, lid, pp.grid_x, 256u);
    if (tid >= pp.n * pp.h * pp.w) { return; }
    if (pp.op == 0u) { img[tid] = src[tid] - other[tid]; } else { img[tid] = src[tid] / max(other[tid], 1e-6); }
}

@compute @workgroup_size(256)
fn blur_rows(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let tid = flat_id(wid, lid, pp.grid_x, 256u);
    if (tid >= pp.n * pp.h * pp.w) { return; }
    let col = i32(tid % pp.w);
    let line = tid - u32(col);
    var acc = 0.0;
    var wsum = 0.0;
    let r = i32(pp.radius);
    for (var d = -r; d <= r; d = d + 1) {
        let c = col + d;
        if (c < 0 || c >= i32(pp.w)) { continue; }
        let g = exp(-f32(d * d) * pp.k.x);
        acc = acc + g * src[line + u32(c)];
        wsum = wsum + g;
    }
    img[tid] = acc / wsum;
}

@compute @workgroup_size(256)
fn blur_cols(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let tid = flat_id(wid, lid, pp.grid_x, 256u);
    let hw = pp.h * pp.w;
    if (tid >= pp.n * hw) { return; }
    let row = i32((tid % hw) / pp.w);
    var acc = 0.0;
    var wsum = 0.0;
    let r = i32(pp.radius);
    for (var d = -r; d <= r; d = d + 1) {
        let q = row + d;
        if (q < 0 || q >= i32(pp.h)) { continue; }
        let g = exp(-f32(d * d) * pp.k.x);
        acc = acc + g * src[u32(i32(tid) + d * i32(pp.w))];
        wsum = wsum + g;
    }
    img[tid] = acc / wsum;
}

// A wide Gaussian costs its width in taps at every pixel, twice, and a dynamic background's is
// a sixth of the pattern wide. But a Gaussian of deviation σ is so smooth that sums over boxes
// of s = σ / 6 pixels hold all it needs: the pixels are summed over boxes (box_sum), the box
// sums are blurred on their own coarse grid (coarse_rows, coarse_cols) and spread back over
// the pixels by a narrow Gaussian of deviation s (spread_rows, spread_cols); the variances of
// the box, the coarse blur and the spread add up to σ². Sums of pixels and counts of pixels
// travel side by side and are divided at the end, which is the renormalization at the edges;
// the coarse grid carries a margin of cells outside the image so that the spread is never cut
// off there. On real patterns the result lies within 0.2 % (rms) of the Gaussian carried to 6 σ,
// closer than blur_rows / blur_cols, which stop at 3 σ, come to it (0.5 %).

// Centre (in pixels) and width of coarse cell `jr` along an axis of n pixels; cells outside the
// image are whole.
fn cell(jr: i32, n: u32) -> vec2<f32> {
    let s = i32(pp.bin);
    let lo = jr * s;
    var hi = lo + s;
    if (jr >= 0 && lo < i32(n)) { hi = min(hi, i32(n)); }
    return vec2<f32>(0.5 * f32(lo + hi - 1), f32(hi - lo));
}

@compute @workgroup_size(256)
fn box_sum(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let tid = flat_id(wid, lid, pp.grid_x, 256u);
    let ws = (pp.w + pp.bin - 1u) / pp.bin;
    let hs = (pp.h + pp.bin - 1u) / pp.bin;
    if (tid >= pp.n * hs * ws) { return; }
    let j = tid % ws;
    let i = (tid / ws) % hs;
    let base = (tid / (ws * hs)) * pp.h * pp.w;
    let r1 = min((i + 1u) * pp.bin, pp.h);
    let c1 = min((j + 1u) * pp.bin, pp.w);
    var acc = 0.0;
    for (var r = i * pp.bin; r < r1; r = r + 1u) {
        for (var c = j * pp.bin; c < c1; c = c + 1u) { acc = acc + src[base + r * pp.w + c]; }
    }
    cout[tid] = vec2<f32>(acc, f32((r1 - i * pp.bin) * (c1 - j * pp.bin)));
}

// [n][hs][ws] → [n][hs][ws + 2 cpad]: blurred along the rows, onto the grid with its margin
@compute @workgroup_size(256)
fn coarse_rows(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let tid = flat_id(wid, lid, pp.grid_x, 256u);
    let ws = (pp.w + pp.bin - 1u) / pp.bin;
    let hs = (pp.h + pp.bin - 1u) / pp.bin;
    let wp = ws + 2u * pp.cpad;
    if (tid >= pp.n * hs * wp) { return; }
    let line = tid / wp;
    let c = cell(i32(tid % wp) - i32(pp.cpad), pp.w).x;
    let reach = f32(pp.radius);
    let j0 = max(i32(floor((c - reach) / f32(pp.bin))), 0);
    let j1 = min(i32(floor((c + reach) / f32(pp.bin))), i32(ws) - 1);
    var acc = vec2<f32>(0.0, 0.0);
    for (var j = j0; j <= j1; j = j + 1) {
        let d = cell(j, pp.w).x - c;
        acc = acc + exp(-d * d * pp.k.x) * cin[line * ws + u32(j)];
    }
    cout[tid] = acc;
}

// [n][hs][wp] → [n][hs + 2 cpad][wp]
@compute @workgroup_size(256)
fn coarse_cols(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let tid = flat_id(wid, lid, pp.grid_x, 256u);
    let wp = (pp.w + pp.bin - 1u) / pp.bin + 2u * pp.cpad;
    let hs = (pp.h + pp.bin - 1u) / pp.bin;
    let hp = hs + 2u * pp.cpad;
    if (tid >= pp.n * hp * wp) { return; }
    let col = tid % wp;
    let base = (tid / (wp * hp)) * hs * wp + col;
    let c = cell(i32((tid / wp) % hp) - i32(pp.cpad), pp.h).x;
    let reach = f32(pp.radius);
    let i0 = max(i32(floor((c - reach) / f32(pp.bin))), 0);
    let i1 = min(i32(floor((c + reach) / f32(pp.bin))), i32(hs) - 1);
    var acc = vec2<f32>(0.0, 0.0);
    for (var i = i0; i <= i1; i = i + 1) {
        let d = cell(i, pp.h).x - c;
        acc = acc + exp(-d * d * pp.k.x) * cin[base + u32(i) * wp];
    }
    cout[tid] = acc;
}

// [n][hp][wp] → [n][hp][w]: each pixel column from the cpad cells on either side of its own
@compute @workgroup_size(256)
fn spread_rows(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let tid = flat_id(wid, lid, pp.grid_x, 256u);
    let wp = (pp.w + pp.bin - 1u) / pp.bin + 2u * pp.cpad;
    let hp = (pp.h + pp.bin - 1u) / pp.bin + 2u * pp.cpad;
    if (tid >= pp.n * hp * pp.w) { return; }
    let x = tid % pp.w;
    let base = (tid / pp.w) * wp;
    let own = x / pp.bin;
    var acc = vec2<f32>(0.0, 0.0);
    for (var t = 0u; t <= 2u * pp.cpad; t = t + 1u) {
        let cw = cell(i32(own + t) - i32(pp.cpad), pp.w);
        let d = f32(x) - cw.x;
        acc = acc + cw.y * exp(-d * d * pp.k.z) * cin[base + own + t];
    }
    cout[tid] = acc;
}

// [n][hp][w] → the blurred images [n][h][w]
@compute @workgroup_size(256)
fn spread_cols(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let tid = flat_id(wid, lid, pp.grid_x, 256u);
    let hw = pp.h * pp.w;
    if (tid >= pp.n * hw) { return; }
    let hp = (pp.h + pp.bin - 1u) / pp.bin + 2u * pp.cpad;
    let x = tid % pp.w;
    let y = (tid % hw) / pp.w;
    let base = (tid / hw) * hp * pp.w + x;
    let own = y / pp.bin;
    var acc = vec2<f32>(0.0, 0.0);
    for (var t = 0u; t <= 2u * pp.cpad; t = t + 1u) {
        let cw = cell(i32(own + t) - i32(pp.cpad), pp.h);
        let d = f32(y) - cw.x;
        acc = acc + cw.y * exp(-d * d * pp.k.z) * cin[base + (own + t) * pp.w];
    }
    img[tid] = acc.x / acc.y;
}

var<workgroup> red: array<vec2<f32>, 256>;

@compute @workgroup_size(256)
fn stretch(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let b = wid.y * pp.grid_x + wid.x;
    let hw = pp.h * pp.w;
    let live = b < pp.n;
    var ext = vec2<f32>(3.4e38, -3.4e38);
    if (live) {
        for (var i = lid; i < hw; i = i + 256u) {
            let v = img[b * hw + i];
            ext = vec2<f32>(min(ext.x, v), max(ext.y, v));
        }
    }
    red[lid] = ext;
    workgroupBarrier();
    for (var off = 128u; off > 0u; off = off >> 1u) {
        if (lid < off) { red[lid] = vec2<f32>(min(red[lid].x, red[lid + off].x), max(red[lid].y, red[lid + off].y)); }
        workgroupBarrier();
    }
    let lo = red[0].x;
    let k = 1.0 / max(red[0].y - lo, 1e-20);
    if (live) {
        for (var i = lid; i < hw; i = i + 256u) { img[b * hw + i] = (img[b * hw + i] - lo) * k; }
    }
}

// CLAHE, for a batch. The image is
// padded by reflection at its top and left to a whole number of tiles; each tile's histogram is
// clipped and what was cut is spread over its levels; a pixel takes the equalized value from
// the four nearest tiles' tables, each read between its two nearest levels.
fn clahe_reflect(i: u32, n: u32, pad: u32) -> u32 {
    if (n <= 1u) { return 0u; }
    var x = i32(i) - i32(pad);
    if (x < 0) { x = -x; }
    if (x >= i32(n)) { x = 2 * i32(n) - 2 - x; }
    return u32(clamp(x, 0, i32(n) - 1));
}

@compute @workgroup_size(256)
fn clahe_hist(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let tid = flat_id(wid, lid, pp.grid_x, 256u);
    let hp = pp.h + pp.pad_h;
    let wp = pp.w + pp.pad_w;
    if (tid >= pp.n * hp * wp) { return; }
    let b = tid / (hp * wp);
    let py = (tid % (hp * wp)) / wp;
    let px = tid % wp;
    let oy = clahe_reflect(py, pp.h, pp.pad_h);
    let ox = clahe_reflect(px, pp.w, pp.pad_w);
    let v = clamp(src[b * pp.h * pp.w + oy * pp.w + ox], 0.0, 1.0);
    let bin = u32(clamp(v * f32(pp.bins - 1u), 0.0, f32(pp.bins - 1u)));
    let tile = (py / pp.tile_h) * pp.tiles + px / pp.tile_w;
    atomicAdd(&hist[(b * pp.tiles * pp.tiles + tile) * pp.bins + bin], 1u);
}

@compute @workgroup_size(64)
fn clahe_cdf(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let tid = flat_id(wid, lid, pp.grid_x, 64u);
    if (tid >= pp.n * pp.tiles * pp.tiles) { return; }
    let base = tid * pp.bins;
    let voxels = f32(max(pp.tile_h * pp.tile_w, 1u));
    let nb = f32(pp.bins);
    let limit = max(pp.k.y * voxels / nb, 1.0);
    var sumc = 0.0;
    for (var i = 0u; i < pp.bins; i = i + 1u) { sumc = sumc + min(f32(atomicLoad(&hist[base + i])), limit); }
    // what the clip cut, spread evenly; the remainder one count each to the first levels
    let clipped = voxels - sumc;
    let residual = i32(clipped - nb * floor(clipped / nb));
    let redist = (clipped - f32(residual)) / nb;
    var acc = 0.0;
    for (var i = 0u; i < pp.bins; i = i + 1u) {
        var c = min(f32(atomicLoad(&hist[base + i])), limit) + redist;
        if (i32(i) < residual) { c = c + 1.0; }
        acc = acc + c;
        cdf[base + i] = clamp(acc / voxels, 0.0, 1.0);
    }
}

@compute @workgroup_size(256)
fn clahe_apply(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let tid = flat_id(wid, lid, pp.grid_x, 256u);
    let hw = pp.h * pp.w;
    if (tid >= pp.n * hw) { return; }
    let b = tid / hw;
    let row = (tid % hw) / pp.w;
    let col = tid % pp.w;
    let v = clamp(src[tid], 0.0, 1.0);
    let gy = (f32(row) + 0.5) / f32(pp.h) * f32(pp.tiles - 1u);
    let gx = (f32(col) + 0.5) / f32(pp.w) * f32(pp.tiles - 1u);
    let y0 = u32(floor(gy));
    let x0 = u32(floor(gx));
    let y1 = min(y0 + 1u, pp.tiles - 1u);
    let x1 = min(x0 + 1u, pp.tiles - 1u);
    let fy = gy - f32(y0);
    let fx = gx - f32(x0);
    let bf = v * f32(pp.bins - 1u);
    let b0 = u32(clamp(floor(bf), 0.0, f32(pp.bins - 2u)));
    let fb = bf - f32(b0);
    let b1 = b0 + 1u;
    let tb = b * pp.tiles * pp.tiles;
    let i00 = (tb + y0 * pp.tiles + x0) * pp.bins;
    let i10 = (tb + y0 * pp.tiles + x1) * pp.bins;
    let i01 = (tb + y1 * pp.tiles + x0) * pp.bins;
    let i11 = (tb + y1 * pp.tiles + x1) * pp.bins;
    let c00 = mix(cdf_in[i00 + b0], cdf_in[i00 + b1], fb);
    let c10 = mix(cdf_in[i10 + b0], cdf_in[i10 + b1], fb);
    let c01 = mix(cdf_in[i01 + b0], cdf_in[i01 + b1], fb);
    let c11 = mix(cdf_in[i11 + b0], cdf_in[i11 + b1], fb);
    img[tid] = mix(mix(c00, c10, fx), mix(c01, c11, fx), fy);
}

fn masked(i: u32) -> bool {
    let dr = f32(i / pp.w) - pp.mask.x;
    let dc = f32(i % pp.w) - pp.mask.y;
    return pp.mask.z <= 0.0 || dr * dr + dc * dc <= pp.mask.z;
}

@compute @workgroup_size(256)
fn normalize(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let b = wid.y * pp.grid_x + wid.x;
    let hw = pp.h * pp.w;
    let live = b < pp.n;
    var acc = vec2<f32>(0.0, 0.0);
    if (live) {
        for (var i = lid; i < hw; i = i + 256u) {
            if (masked(i)) { acc = acc + vec2<f32>(img[b * hw + i], 1.0); }
        }
    }
    red[lid] = acc;
    workgroupBarrier();
    for (var off = 128u; off > 0u; off = off >> 1u) {
        if (lid < off) { red[lid] = red[lid] + red[lid + off]; }
        workgroupBarrier();
    }
    let mean = red[0].x / max(red[0].y, 1.0);
    workgroupBarrier();
    var ss = 0.0;
    if (live) {
        for (var i = lid; i < hw; i = i + 256u) {
            var v = 0.0;
            if (masked(i)) { v = img[b * hw + i] - mean; }
            img[b * hw + i] = v;
            ss = ss + v * v;
        }
    }
    red[lid] = vec2<f32>(ss, 0.0);
    workgroupBarrier();
    for (var off = 128u; off > 0u; off = off >> 1u) {
        if (lid < off) { red[lid] = red[lid] + red[lid + off]; }
        workgroupBarrier();
    }
    let inv = inverseSqrt(max(red[0].x, 1e-30));
    if (live) {
        for (var i = lid; i < hw; i = i + 256u) { img[b * hw + i] = img[b * hw + i] * inv; }
    }
}
