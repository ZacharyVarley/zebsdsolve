// Non-local pattern averaging (NLPAR; Brewick, Wright and Rowenhorst, Ultramicroscopy 200
// (2019) 50), after common.wgsl. A pattern is replaced by a weighted mean of itself and its
// neighbours in the map, and a neighbour weighs as much as it looks like the same pattern under
// other noise:
//
//   d²(i, j)  the mean squared difference of patterns i and j over their N pixels
//   σ²(i)     the noise of pattern i: half the smallest d² to one of its eight nearest
//             neighbours (two patterns of the same crystal differ by twice the noise)
//   e(i, j)   the d² that noise alone would give: σ²(i) + σ²(j)
//   d̃(i, j)   sqrt(N / 2) · (d² / e − 1): how far d² lies above that, in its standard deviations
//   w(i, j)   exp(−max(d̃, 0) / λ²), and 1 for the pattern itself
//   out(i)    Σ w(i, j) pattern(j) / Σ w(i, j) over the window of `radius` cells each way
//
// One pass of this over a wide window is the published method. Here it may run in several
// passes over a small window, each on the patterns the pass before left. A pattern then
// reaches radius × passes cells, but only through neighbours it resembles at every step, and the
// later passes decide their weights on patterns already quieter: a grain a cell or two wide,
// which one wide pass can average into the grains around it while its weights are still
// noisy, keeps its own pattern.
//
// Averaged patterns are not independent (neighbours share what went into them), so measuring σ²
// again from them would be wrong, and is not needed: each averaged pattern is a known mix of
// stored ones, pattern(i) = Σ_u a_i[u] stored(i + u), and its shares a_i are kept beside it.
// The noise in the difference of two averages is then exactly
//     e(i, j) = Σ_u (a_i[u] − a_j[u − o])² σ²(i + u),   o the offset from i to j,
// with σ² measured once, on the stored patterns. For those (a_i = 1 at u = 0) it is the
// σ²(i) + σ²(j) above, and d̃ keeps its meaning in every pass.
//
// The cells of a batch come as a list: the batch's own cells first, then ring after ring of
// the cells around them (engine.zig), so that every pass works on a prefix of the list.
// `nbr[cell · F + slot]` is the list index of the cell at an offset of up to `reach` =
// radius × passes cells each way (rows of that square, top left first; the cell itself in the
// middle), or NONE outside the map; shares are laid out the same way. d² and weights have one
// slot per cell of the window, (2 radius + 1)² of them.
//
//   distance  d² of each cell with the neighbours after it in the window (the pairs before it
//             are the same pairs seen from the other cell, and are written from there)
//   sigma     σ² per cell, from the stored patterns
//   weights   the normalized weights per cell
//   average   the weighted means
//   shares    each mean's shares of the stored patterns

const NONE: u32 = 0xffffffffu;

struct NlP {
    // cells this dispatch covers (a prefix of the list), pixels per pattern
    n: u32,
    px: u32,
    radius: u32,
    // radius × passes: how far the neighbour table and the shares reach
    reach: u32,
    grid_x: u32,
    // average: the mean of cell c goes to pattern out0 + c of the output
    out0: u32,
    // how far the shares of the patterns going in reach: radius × the passes done (0: the
    // stored patterns themselves)
    spread: u32,
    pad0: u32,
    // 1 / λ², sqrt(N / 2)
    num: vec4<f32>,
};

@group(0) @binding(0) var<uniform> nl: NlP;
@group(0) @binding(1) var<storage, read> pin: array<f32>;
@group(0) @binding(2) var<storage, read_write> pout: array<f32>;
@group(0) @binding(3) var<storage, read> nbr: array<u32>;
@group(0) @binding(4) var<storage, read_write> d2: array<f32>;
@group(0) @binding(5) var<storage, read> d2_in: array<f32>;
@group(0) @binding(6) var<storage, read_write> sig: array<f32>;
@group(0) @binding(7) var<storage, read> sig_in: array<f32>;
@group(0) @binding(8) var<storage, read_write> wts: array<f32>;
@group(0) @binding(9) var<storage, read> wts_in: array<f32>;
@group(0) @binding(10) var<storage, read> share_in: array<f32>;
@group(0) @binding(11) var<storage, read_write> share: array<f32>;

// Slot of offset (dy, dx) in the neighbour table and the shares.
fn wide(dy: i32, dx: i32) -> u32 {
    let r = i32(nl.reach);
    return u32((dy + r) * (2 * r + 1) + dx + r);
}

fn wide_count() -> u32 {
    return (2u * nl.reach + 1u) * (2u * nl.reach + 1u);
}

// Offset of window slot q (d², weights).
fn offset_of(q: u32) -> vec2<i32> {
    let side = 2u * nl.radius + 1u;
    return vec2<i32>(i32(q / side) - i32(nl.radius), i32(q % side) - i32(nl.radius));
}

fn within(v: vec2<i32>, r: u32) -> bool {
    return abs(v.x) <= i32(r) && abs(v.y) <= i32(r);
}

var<workgroup> part: array<f32, 256>;

// One workgroup per (cell, later neighbour): 256 invocations share the pixels.
@compute @workgroup_size(256)
fn distance(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let pair = wid.y * nl.grid_x + wid.x;
    let side = 2u * nl.radius + 1u;
    let k = side * side;
    let later = k / 2u;
    let c = pair / later;
    let q = later + 1u + pair % later;
    var j = NONE;
    if (c < nl.n) {
        let o = offset_of(q);
        j = nbr[c * wide_count() + wide(o.x, o.y)];
    }
    var acc = 0.0;
    if (j != NONE) {
        for (var p = lid; p < nl.px; p = p + 256u) {
            let d = pin[c * nl.px + p] - pin[j * nl.px + p];
            acc = acc + d * d;
        }
    }
    part[lid] = acc;
    workgroupBarrier();
    for (var off = 128u; off > 0u; off = off >> 1u) {
        if (lid < off) { part[lid] = part[lid] + part[lid + off]; }
        workgroupBarrier();
    }
    if (lid == 0u && j != NONE) {
        let v = part[0] / f32(nl.px);
        d2[c * k + q] = v;
        d2[j * k + (k - 1u - q)] = v;
    }
}

@compute @workgroup_size(64)
fn sigma(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let c = flat_id(wid, lid, nl.grid_x, 64u);
    if (c >= nl.n) { return; }
    let side = 2u * nl.radius + 1u;
    var least = -1.0;
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            if ((dy == 0 && dx == 0) || nbr[c * wide_count() + wide(dy, dx)] == NONE) { continue; }
            let v = d2_in[c * side * side + u32(dy + i32(nl.radius)) * side + u32(dx + i32(nl.radius))];
            if (least < 0.0 || v < least) { least = v; }
        }
    }
    // (a cell without neighbours has no estimate: 0, and then only itself in its average)
    sig[c] = 0.5 * max(least, 0.0);
}

// The noise in the difference of the patterns of cell c and its neighbour j at offset o.
fn expected(c: u32, j: u32, o: vec2<i32>) -> f32 {
    if (nl.spread == 0u) { return sig_in[c] + sig_in[j]; }
    let f = wide_count();
    let m = i32(nl.spread + nl.radius);
    var e = 0.0;
    for (var uy = -m; uy <= m; uy = uy + 1) {
        for (var ux = -m; ux <= m; ux = ux + 1) {
            let at = nbr[c * f + wide(uy, ux)];
            if (at == NONE) { continue; }
            var mine = 0.0;
            if (within(vec2<i32>(uy, ux), nl.spread)) { mine = share_in[c * f + wide(uy, ux)]; }
            var theirs = 0.0;
            let v = vec2<i32>(uy, ux) - o;
            if (within(v, nl.spread)) { theirs = share_in[j * f + wide(v.x, v.y)]; }
            e = e + (mine - theirs) * (mine - theirs) * sig_in[at];
        }
    }
    return e;
}

@compute @workgroup_size(64)
fn weights(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let c = flat_id(wid, lid, nl.grid_x, 64u);
    if (c >= nl.n) { return; }
    let side = 2u * nl.radius + 1u;
    let k = side * side;
    let own = k / 2u;
    var total = 1.0;
    for (var q = 0u; q < k; q = q + 1u) {
        if (q == own) { continue; }
        let o = offset_of(q);
        let j = nbr[c * wide_count() + wide(o.x, o.y)];
        var w = 0.0;
        if (j != NONE) {
            let e = expected(c, j, o);
            let d = d2_in[c * k + q];
            if (d <= e) {
                w = 1.0;
            } else if (e > 0.0) {
                w = exp(-nl.num.y * (d / e - 1.0) * nl.num.x);
            }
        }
        wts[c * k + q] = w;
        total = total + w;
    }
    for (var q = 0u; q < k; q = q + 1u) {
        if (q == own) { wts[c * k + q] = 1.0 / total; } else { wts[c * k + q] = wts[c * k + q] / total; }
    }
}

@compute @workgroup_size(256)
fn average(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let tid = flat_id(wid, lid, nl.grid_x, 256u);
    if (tid >= nl.n * nl.px) { return; }
    let c = tid / nl.px;
    let p = tid - c * nl.px;
    let side = 2u * nl.radius + 1u;
    let k = side * side;
    var acc = 0.0;
    for (var q = 0u; q < k; q = q + 1u) {
        let w = wts_in[c * k + q];
        if (w == 0.0) { continue; }
        let o = offset_of(q);
        acc = acc + w * pin[nbr[c * wide_count() + wide(o.x, o.y)] * nl.px + p];
    }
    pout[(nl.out0 + c) * nl.px + p] = acc;
}

// a'_c[u] = Σ_j w(c, j) a_j[u − o_j]: the mean's share of the stored pattern at offset u.
@compute @workgroup_size(256)
fn shares(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let tid = flat_id(wid, lid, nl.grid_x, 256u);
    let f = wide_count();
    if (tid >= nl.n * f) { return; }
    let c = tid / f;
    let slot = tid - c * f;
    let wn = 2u * nl.reach + 1u;
    let u = vec2<i32>(i32(slot / wn) - i32(nl.reach), i32(slot % wn) - i32(nl.reach));
    let side = 2u * nl.radius + 1u;
    let k = side * side;
    var acc = 0.0;
    for (var q = 0u; q < k; q = q + 1u) {
        let w = wts_in[c * k + q];
        if (w == 0.0) { continue; }
        let o = offset_of(q);
        let v = u - o;
        if (nl.spread == 0u) {
            if (v.x == 0 && v.y == 0) { acc = acc + w; }
        } else if (within(v, nl.spread)) {
            acc = acc + w * share_in[nbr[c * f + wide(o.x, o.y)] * f + wide(v.x, v.y)];
        }
    }
    share[tid] = acc;
}
