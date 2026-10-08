// A dictionary in a few hundred numbers per pattern (after common.wgsl). The simulated patterns
// of a dictionary lie close to a subspace of far fewer dimensions than they have pixels; its
// basis W (k rows of p pixels, orthonormal) is learned from the dictionary as it streams by,
// by Oja's rule, and patterns are then matched by their k coefficients instead of their pixels.
//
//   project      Y = X Wᵀ: the coefficients of n patterns (dictionary entries or experimental)
//   accumulate   Oja's step W ← W + η (Yᵀ X) / n on a batch of dictionary patterns
//   gram         W Wᵀ (k × k), which the host factors to make the rows orthonormal again
//   rotate       W ← L⁻¹ W, with L⁻¹ lower triangular from that factor
//   nearest      per experimental pattern, the dictionary entry whose coefficients have the
//                largest inner product with its own, folded into a running best

struct PcaP {
    // rows of X (patterns), pixels per pattern, components
    n: u32,
    p: u32,
    k: u32,
    grid_x: u32,
    // project: the first output row; match: the index of the first dictionary entry bound
    row0: u32,
    // match: dictionary entries bound
    n_dict: u32,
    pad0: u32,
    pad1: u32,
    // accumulate: η / n
    eta: vec4<f32>,
};

@group(0) @binding(0) var<uniform> pc: PcaP;
// patterns [n][p]; in match, the dictionary's coefficients [n_dict][k]
@group(0) @binding(1) var<storage, read> px: array<f32>;
// the basis [k][p]
@group(0) @binding(2) var<storage, read> pw: array<f32>;
@group(0) @binding(3) var<storage, read_write> py: array<f32>;
@group(0) @binding(4) var<storage, read> py_in: array<f32>;
@group(0) @binding(5) var<storage, read_write> pw_out: array<f32>;
@group(0) @binding(6) var<storage, read_write> pgram: array<f32>;
@group(0) @binding(7) var<storage, read> prot: array<f32>;
// per experimental pattern: best score (f32 bits), dictionary index
@group(0) @binding(8) var<storage, read_write> pbest: array<u32>;

// Four patterns against four components per invocation: every pixel of a pattern and of a
// component is then read once for sixteen products, not once for each (the kernel is bound by
// memory traffic, not arithmetic).
@compute @workgroup_size(64)
fn project(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let tid = flat_id(wid, lid, pc.grid_x, 64u);
    let kq = (pc.k + 3u) / 4u;
    if (tid >= ((pc.n + 3u) / 4u) * kq) { return; }
    let e0 = (tid / kq) * 4u;
    let c0 = (tid % kq) * 4u;
    var xb: array<u32, 4>;
    var wb: array<u32, 4>;
    for (var j = 0u; j < 4u; j = j + 1u) {
        xb[j] = min(e0 + j, pc.n - 1u) * pc.p;
        wb[j] = min(c0 + j, pc.k - 1u) * pc.p;
    }
    var acc: array<vec4<f32>, 4>;
    for (var j = 0u; j < 4u; j = j + 1u) { acc[j] = vec4<f32>(0.0); }
    for (var i = 0u; i < pc.p; i = i + 1u) {
        let wv = vec4<f32>(pw[wb[0] + i], pw[wb[1] + i], pw[wb[2] + i], pw[wb[3] + i]);
        acc[0] = acc[0] + px[xb[0] + i] * wv;
        acc[1] = acc[1] + px[xb[1] + i] * wv;
        acc[2] = acc[2] + px[xb[2] + i] * wv;
        acc[3] = acc[3] + px[xb[3] + i] * wv;
    }
    for (var j = 0u; j < 4u; j = j + 1u) {
        if (e0 + j >= pc.n) { break; }
        for (var m = 0u; m < 4u; m = m + 1u) {
            if (c0 + m < pc.k) { py[(pc.row0 + e0 + j) * pc.k + c0 + m] = acc[j][m]; }
        }
    }
}

@compute @workgroup_size(64)
fn accumulate(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let tid = flat_id(wid, lid, pc.grid_x, 64u);
    if (tid >= pc.k * pc.p) { return; }
    let c = tid / pc.p;
    let pix = tid % pc.p;
    var acc = 0.0;
    for (var i = 0u; i < pc.n; i = i + 1u) { acc = acc + py_in[i * pc.k + c] * px[i * pc.p + pix]; }
    pw_out[tid] = pw[tid] + pc.eta.x * acc;
}

@compute @workgroup_size(64)
fn gram(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let tid = flat_id(wid, lid, pc.grid_x, 64u);
    if (tid >= pc.k * pc.k) { return; }
    let a = (tid / pc.k) * pc.p;
    let b = (tid % pc.k) * pc.p;
    var acc = 0.0;
    for (var i = 0u; i < pc.p; i = i + 1u) { acc = acc + pw[a + i] * pw[b + i]; }
    pgram[tid] = acc;
}

@compute @workgroup_size(64)
fn rotate(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let tid = flat_id(wid, lid, pc.grid_x, 64u);
    if (tid >= pc.k * pc.p) { return; }
    let c = tid / pc.p;
    let pix = tid % pc.p;
    var acc = 0.0;
    for (var j = 0u; j <= c; j = j + 1u) { acc = acc + prot[c * pc.k + j] * pw[j * pc.p + pix]; }
    pw_out[tid] = acc;
}

var<workgroup> pred: array<f32, 2048>;
var<workgroup> predi: array<u32, 2048>;

// Eight experimental patterns per workgroup: each invocation walks its share of the dictionary
// once and scores all eight against every entry it reads.
@compute @workgroup_size(256)
fn nearest(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let e0 = (wid.y * pc.grid_x + wid.x) * 8u;
    var yb: array<u32, 8>;
    var bv: array<f32, 8>;
    var bi: array<u32, 8>;
    for (var j = 0u; j < 8u; j = j + 1u) {
        yb[j] = min(e0 + j, pc.n - 1u) * pc.k;
        bv[j] = -3.4e38;
        bi[j] = 0u;
    }
    if (e0 < pc.n) {
        for (var d = lid; d < pc.n_dict; d = d + 256u) {
            let db = d * pc.k;
            var a0 = vec4<f32>(0.0);
            var a1 = vec4<f32>(0.0);
            for (var c = 0u; c < pc.k; c = c + 1u) {
                let w = px[db + c];
                a0 = a0 + w * vec4<f32>(py_in[yb[0] + c], py_in[yb[1] + c], py_in[yb[2] + c], py_in[yb[3] + c]);
                a1 = a1 + w * vec4<f32>(py_in[yb[4] + c], py_in[yb[5] + c], py_in[yb[6] + c], py_in[yb[7] + c]);
            }
            for (var j = 0u; j < 4u; j = j + 1u) {
                if (a0[j] > bv[j]) {
                    bv[j] = a0[j];
                    bi[j] = d;
                }
                if (a1[j] > bv[j + 4u]) {
                    bv[j + 4u] = a1[j];
                    bi[j + 4u] = d;
                }
            }
        }
    }
    for (var j = 0u; j < 8u; j = j + 1u) {
        pred[j * 256u + lid] = bv[j];
        predi[j * 256u + lid] = bi[j];
    }
    workgroupBarrier();
    for (var off = 128u; off > 0u; off = off >> 1u) {
        if (lid < off) {
            for (var j = 0u; j < 8u; j = j + 1u) {
                let a = j * 256u + lid;
                let o = a + off;
                if (pred[o] > pred[a] || (pred[o] == pred[a] && predi[o] < predi[a])) {
                    pred[a] = pred[o];
                    predi[a] = predi[o];
                }
            }
        }
        workgroupBarrier();
    }
    if (lid < 8u && e0 + lid < pc.n && pred[lid * 256u] > bitcast<f32>(pbest[(e0 + lid) * 2u])) {
        pbest[(e0 + lid) * 2u] = bitcast<u32>(pred[lid * 256u]);
        pbest[(e0 + lid) * 2u + 1u] = pc.row0 + predi[lid * 256u];
    }
}
