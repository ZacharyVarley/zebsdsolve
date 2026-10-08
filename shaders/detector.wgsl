// The detector's side: master patterns projected onto detector pixels, and what the fits and
// scores need from comparing them with experimental patterns (after common.wgsl).
//
// An item is a pattern with its geometry and an orientation: two vec4 in `poses`, the
// detector's position t for that pattern in pixels with the pattern's index in .w, then the
// orientation (w, x, y, z). Pixel (row, col) lies at u = c0 · row + c1 · col + t from the source
// point, its ray is u / |u|, and the simulated intensity is the master in direction q · ray.
//
//   project       simulated patterns
//   project_dict  simulated patterns of dictionary orientations at one detector position
//   seed          items with the orientations a search found (spherical peaks or dictionary hits)
//   decoy         items with random orientations, for what mutual information chance gives
//   sums          per item, the sums from which the host (pose fit) and refine_step build the
//                 normalized cross-correlation with the item's pattern, its gradient and its
//                 Gauss–Newton matrix over six directions: a shift of the detector (3) and a
//                 rotation of the crystal (3), or over the rotation alone (flags bit 1), which
//                 is all orientation refinement uses. Chunks of pixels in parallel;
//                 sums_reduce adds the chunks.
//   refine_init, refine_step   Gauss–Newton on each item's orientation, on the device: a step
//                 from the best point so far, accepted when the next sums score higher, with
//                 the model's curvature rescaled by how each gain compares with its promise
//   collect       each item's refined orientation, NCC, NMI and last improving step as one
//                 record
//   nmi           normalized mutual information between each item's pattern and simulation
//   sim_norm      zero mean and unit norm of simulated patterns under the mask
//   dict_score    best dictionary entry per pattern, by inner product
//
// Experimental patterns arrive already zero-mean and unit-norm under the mask (prep.wgsl) and
// zero outside it, so NCC = Σ (I − Ī) b / |I − Ī|.

const NS: u32 = 43u;

struct DetP {
    n_items: u32,
    pat_h: u32,
    pat_w: u32,
    side: u32,
    grid_x: u32,
    // pixel chunks per item (sums)
    n_chunks: u32,
    // refine_step: 1 = last call (keep the best, take no step)
    flags: u32,
    n_bins: u32,
    // dictionary: entries in this chunk, index of its first entry, patterns scored
    // (nmi: the score of item b goes to dict0 + b; decoy: n_exp numbers the draw)
    n_dict: u32,
    dict0: u32,
    n_exp: u32,
    // collect: the record of item b goes to out0 + b; seed: its orientation comes from there
    out0: u32,
    c0: vec4<f32>,
    c1: vec4<f32>,
    c2: vec4<f32>,
    // mask centre (row, col), squared radius in pixels (≤ 0: none); .w: the curvature scale a
    // refinement starts with
    mask: vec4<f32>,
    // finite-difference step on the sphere (radians), largest refinement step (radians),
    // smallest and largest curvature scale
    num: vec4<f32>,
    // project_dict: the detector position every dictionary pattern is simulated at
    t0: vec4<f32>,
};

@group(0) @binding(0) var<uniform> dp: DetP;
@group(0) @binding(1) var<storage, read> master: array<f32>;
@group(0) @binding(2) var<storage, read> poses: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> pats: array<f32>;
@group(0) @binding(4) var<storage, read_write> sims: array<f32>;
@group(0) @binding(5) var<storage, read_write> part: array<f32>;
@group(0) @binding(6) var<storage, read> part_in: array<f32>;
@group(0) @binding(7) var<storage, read_write> sums_out: array<f32>;
@group(0) @binding(8) var<storage, read> sums_in: array<f32>;
// refinement state per item (24 floats): current q (0–3), best q (4–7), best NCC (8), the
// curvature scale (9), the best point's gradient (10–12) and matrix (13–18), the size of the
// last step that improved (19), the gain the model promised for the step being tried (20) and
// that step's size (21), the count of points accepted (22)
@group(0) @binding(9) var<storage, read_write> state: array<f32>;
@group(0) @binding(10) var<storage, read_write> poses_rw: array<vec4<f32>>;
@group(0) @binding(11) var<storage, read_write> score: array<f32>;
@group(0) @binding(12) var<storage, read> sims_in: array<f32>;
// per pattern: bitcast best inner product, entry index
@group(0) @binding(13) var<storage, read_write> dict_best: array<u32>;
@group(0) @binding(14) var<storage, read> state_in: array<f32>;
@group(0) @binding(15) var<storage, read> score_in: array<f32>;
// per item: orientation (w, x, y, z), NCC, NMI, accepted iterations, 0
@group(0) @binding(16) var<storage, read_write> records: array<f32>;
// seed: (t, pattern index) per item; spherical peaks (8 floats per item, the orientation first)
@group(0) @binding(17) var<storage, read> seed_items: array<vec4<f32>>;
@group(0) @binding(18) var<storage, read> seed_peaks: array<f32>;
// the dictionary's orientations
@group(0) @binding(19) var<storage, read> dict_q: array<vec4<f32>>;
@group(0) @binding(20) var<storage, read> dict_hit: array<u32>;
@group(0) @binding(21) var<storage, read> mgrad: array<vec4<f32>>;

fn in_mask(row: f32, col: f32) -> bool {
    let dr = row - dp.mask.x;
    let dc = col - dp.mask.y;
    return dp.mask.z <= 0.0 || dr * dr + dc * dc <= dp.mask.z;
}

// Bilinear master intensity in crystal direction v (its own hemisphere's sheet).
fn master_at(v: vec3<f32>) -> f32 {
    let uv = sphere_to_square(v);
    let last = f32(dp.side - 1u);
    let fj = clamp((uv.x + 1.0) * 0.5 * last, 0.0, last);
    let fi = clamp((uv.y + 1.0) * 0.5 * last, 0.0, last);
    let c0 = min(u32(fj), dp.side - 2u);
    let r0 = min(u32(fi), dp.side - 2u);
    var o = r0 * dp.side + c0;
    if (v.z < 0.0) { o = o + dp.side * dp.side; }
    let tj = fj - f32(c0);
    let top = mix(master[o], master[o + 1u], tj);
    let bot = mix(master[o + dp.side], master[o + dp.side + 1u], tj);
    return mix(top, bot, fi - f32(r0));
}

// The master's gradient on the sphere (.xyz) and its intensity (.w) in crystal direction v:
// mastersph.wgsl laid them side by side at the sheets' pixels, so one bilinear lookup reads
// both from the same four places.
fn grad_at(v: vec3<f32>) -> vec4<f32> {
    let uv = sphere_to_square(v);
    let last = f32(dp.side - 1u);
    let fj = clamp((uv.x + 1.0) * 0.5 * last, 0.0, last);
    let fi = clamp((uv.y + 1.0) * 0.5 * last, 0.0, last);
    let c0 = min(u32(fj), dp.side - 2u);
    let r0 = min(u32(fi), dp.side - 2u);
    var o = r0 * dp.side + c0;
    if (v.z < 0.0) { o = o + dp.side * dp.side; }
    let tj = fj - f32(c0);
    let top = mix(mgrad[o], mgrad[o + 1u], tj);
    let bot = mix(mgrad[o + dp.side], mgrad[o + dp.side + 1u], tj);
    return mix(top, bot, fi - f32(r0));
}

fn ray_of(b: u32, row: u32, col: u32) -> vec3<f32> {
    return dp.c0.xyz * f32(row) + dp.c1.xyz * f32(col) + poses[b * 2u].xyz;
}

@compute @workgroup_size(256)
fn project(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let tid = flat_id(wid, lid, dp.grid_x, 256u);
    let hw = dp.pat_h * dp.pat_w;
    if (tid >= dp.n_items * hw) { return; }
    let b = tid / hw;
    let pix = tid % hw;
    let u = ray_of(b, pix / dp.pat_w, pix % dp.pat_w);
    sims[tid] = master_at(qrot(poses[b * 2u + 1u], normalize(u)));
}

@compute @workgroup_size(256)
fn project_dict(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let tid = flat_id(wid, lid, dp.grid_x, 256u);
    let hw = dp.pat_h * dp.pat_w;
    if (tid >= dp.n_items * hw) { return; }
    let pix = tid % hw;
    let u = dp.c0.xyz * f32(pix / dp.pat_w) + dp.c1.xyz * f32(pix % dp.pat_w) + dp.t0.xyz;
    sims[tid] = master_at(qrot(dict_q[dp.dict0 + tid / hw], normalize(u)));
}

@compute @workgroup_size(64)
fn seed(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let b = flat_id(wid, lid, dp.grid_x, 64u);
    if (b >= dp.n_items) { return; }
    poses_rw[b * 2u] = seed_items[b];
    let o = (dp.out0 + b) * 8u;
    poses_rw[b * 2u + 1u] = vec4<f32>(seed_peaks[o], seed_peaks[o + 1u], seed_peaks[o + 2u], seed_peaks[o + 3u]);
}

fn hash_u32(x: u32) -> u32 {
    let v = x * 747796405u + 2891336453u;
    let w = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    return (w >> 22u) ^ w;
}

fn unit_rand(x: u32) -> f32 {
    return f32(hash_u32(x) >> 8u) / 16777216.0;
}

// Each item of `poses` again with an orientation drawn uniformly over all rotations (Shoemake),
// a different one for every item and every draw.
@compute @workgroup_size(64)
fn decoy(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let b = flat_id(wid, lid, dp.grid_x, 64u);
    if (b >= dp.n_items) { return; }
    let s = hash_u32(b + 1u) ^ (dp.n_exp * 0x9e3779b9u);
    let u1 = unit_rand(s);
    let a2 = unit_rand(s + 1u) * TAU;
    let a3 = unit_rand(s + 2u) * TAU;
    let ra = sqrt(1.0 - u1);
    let rb = sqrt(u1);
    poses_rw[b * 2u] = poses[b * 2u];
    poses_rw[b * 2u + 1u] = vec4<f32>(ra * sin(a2), ra * cos(a2), rb * sin(a3), rb * cos(a3));
}

@compute @workgroup_size(64)
fn seed_dict(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let b = flat_id(wid, lid, dp.grid_x, 64u);
    if (b >= dp.n_items) { return; }
    poses_rw[b * 2u] = seed_items[b];
    poses_rw[b * 2u + 1u] = dict_q[dict_hit[b * 2u + 1u]];
}

// (SUMS_WG, this kernel's invocations per workgroup, is set in a line the engine puts before
// this text: 64 unless tuning found this GPU faster with another)
@compute @workgroup_size(SUMS_WG)
fn sums(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let tid = flat_id(wid, lid, dp.grid_x, SUMS_WG);
    if (tid >= dp.n_items * dp.n_chunks) { return; }
    let b = tid / dp.n_chunks;
    let chunk = tid % dp.n_chunks;
    let hw = dp.pat_h * dp.pat_w;
    let per = (hw + dp.n_chunks - 1u) / dp.n_chunks;
    let q = poses[b * 2u + 1u];
    let qi = qconj(q);
    let pbase = u32(poses[b * 2u].w) * hw;
    let start = chunk * per;
    let stop = min(start + per, hw);
    // (row, col) of the pixel, kept by counting: no division per pixel
    var row = start / dp.pat_w;
    var col = start - row * dp.pat_w;
    // sum i of chunk c at [(b · NS + i) · n_chunks + c]: neighbouring invocations hold
    // neighbouring chunks and so write neighbouring words, and sums_reduce reads each sum's
    // chunks in a row
    let out = b * NS * dp.n_chunks + chunk;
    let nc = dp.n_chunks;
    if ((dp.flags & 2u) != 0u) {
        // Orientation refinement needs the rotation's three directions only: their sums are
        // few enough to stay in registers (an indexed array of 43 lives in memory).
        var s0 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        var sd = vec3<f32>(0.0, 0.0, 0.0);
        var sv = vec3<f32>(0.0, 0.0, 0.0);
        var se = vec3<f32>(0.0, 0.0, 0.0);
        var hd = vec3<f32>(0.0, 0.0, 0.0);
        var ho = vec3<f32>(0.0, 0.0, 0.0);
        for (var pix = start; pix < stop; pix = pix + 1u) {
            if (in_mask(f32(row), f32(col))) {
                let c = qrot(q, normalize(ray_of(b, row, col)));
                let m = grad_at(c);
                let d = cross(c, m.xyz);
                let e = pats[pbase + pix];
                s0 = s0 + vec4<f32>(1.0, m.w, m.w * m.w, m.w * e);
                sd = sd + d;
                sv = sv + m.w * d;
                se = se + e * d;
                hd = hd + d * d;
                ho = ho + d.xxy * d.yzz;
            }
            col = col + 1u;
            if (col == dp.pat_w) {
                col = 0u;
                row = row + 1u;
            }
        }
        part[out] = s0.x;
        part[out + 1u * nc] = s0.y;
        part[out + 2u * nc] = s0.z;
        part[out + 3u * nc] = s0.w;
        part[out + 7u * nc] = sd.x;
        part[out + 8u * nc] = sd.y;
        part[out + 9u * nc] = sd.z;
        part[out + 13u * nc] = sv.x;
        part[out + 14u * nc] = sv.y;
        part[out + 15u * nc] = sv.z;
        part[out + 19u * nc] = se.x;
        part[out + 20u * nc] = se.y;
        part[out + 21u * nc] = se.z;
        part[out + 37u * nc] = hd.x;
        part[out + 38u * nc] = ho.x;
        part[out + 39u * nc] = ho.y;
        part[out + 40u * nc] = hd.y;
        part[out + 41u * nc] = ho.z;
        part[out + 42u * nc] = hd.z;
        return;
    }
    var s: array<f32, 43>;
    for (var i = 0u; i < NS; i = i + 1u) { s[i] = 0.0; }
    for (var pix = start; pix < stop; pix = pix + 1u) {
        if (in_mask(f32(row), f32(col))) {
            let u = ray_of(b, row, col);
            let dist = length(u);
            let c = qrot(q, u / dist);
            let m = grad_at(c);
            let val = m.w;
            // d = (∂I/∂ detector shift, ∂I/∂ crystal rotation)
            let a = cross(c, m.xyz);
            let w = qrot(qi, m.xyz) / dist;
            var d: array<f32, 6>;
            d[0] = w.x;
            d[1] = w.y;
            d[2] = w.z;
            d[3] = a.x;
            d[4] = a.y;
            d[5] = a.z;
            let e = pats[pbase + pix];
            s[0] = s[0] + 1.0;
            s[1] = s[1] + val;
            s[2] = s[2] + val * val;
            s[3] = s[3] + val * e;
            var o = 22u;
            for (var i = 0u; i < 6u; i = i + 1u) {
                s[4u + i] = s[4u + i] + d[i];
                s[10u + i] = s[10u + i] + val * d[i];
                s[16u + i] = s[16u + i] + e * d[i];
                for (var j = i; j < 6u; j = j + 1u) {
                    s[o] = s[o] + d[i] * d[j];
                    o = o + 1u;
                }
            }
        }
        col = col + 1u;
        if (col == dp.pat_w) {
            col = 0u;
            row = row + 1u;
        }
    }
    for (var i = 0u; i < NS; i = i + 1u) { part[out + i * nc] = s[i]; }
}

// Which of the sums orientation refinement leaves out (the detector shift's).
fn shift_sum(i: u32) -> bool {
    return (i >= 4u && i < 7u) || (i >= 10u && i < 13u) || (i >= 16u && i < 19u) || (i >= 22u && i < 37u);
}

@compute @workgroup_size(64)
fn sums_reduce(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let tid = flat_id(wid, lid, dp.grid_x, 64u);
    if (tid >= dp.n_items * NS) { return; }
    let i = tid % NS;
    var acc = 0.0;
    if ((dp.flags & 2u) == 0u || !shift_sum(i)) {
        for (var c = 0u; c < dp.n_chunks; c = c + 1u) { acc = acc + part_in[tid * dp.n_chunks + c]; }
    }
    sums_out[tid] = acc;
}

// Solve the symmetric 3 × 3 system (h: xx, xy, xz, yy, yz, zz) by Cramer's rule.
fn solve3(h: array<f32, 6>, g: vec3<f32>) -> vec3<f32> {
    let m = mat3x3<f32>(vec3<f32>(h[0], h[1], h[2]), vec3<f32>(h[1], h[3], h[4]), vec3<f32>(h[2], h[4], h[5]));
    let det = determinant(m);
    if (abs(det) < 1e-30) { return vec3<f32>(0.0, 0.0, 0.0); }
    let x = determinant(mat3x3<f32>(g, m[1], m[2]));
    let y = determinant(mat3x3<f32>(m[0], g, m[2]));
    let z = determinant(mat3x3<f32>(m[0], m[1], g));
    return vec3<f32>(x, y, z) / det;
}

@compute @workgroup_size(64)
fn refine_init(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let b = flat_id(wid, lid, dp.grid_x, 64u);
    if (b >= dp.n_items) { return; }
    let st = b * 24u;
    let q = poses_rw[b * 2u + 1u];
    for (var i = 0u; i < 24u; i = i + 1u) { state[st + i] = 0.0; }
    state[st + 4u] = q.x;
    state[st + 5u] = q.y;
    state[st + 6u] = q.z;
    state[st + 7u] = q.w;
    state[st + 8u] = -2.0;
    state[st + 9u] = dp.mask.w;
}

@compute @workgroup_size(64)
fn collect(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let b = flat_id(wid, lid, dp.grid_x, 64u);
    if (b >= dp.n_items) { return; }
    let st = b * 24u;
    let o = (dp.out0 + b) * 8u;
    records[o] = state_in[st + 4u];
    records[o + 1u] = state_in[st + 5u];
    records[o + 2u] = state_in[st + 6u];
    records[o + 3u] = state_in[st + 7u];
    records[o + 4u] = state_in[st + 8u];
    records[o + 5u] = score_in[b];
    records[o + 6u] = state_in[st + 19u];
    records[o + 7u] = 0.0;
}

@compute @workgroup_size(64)
fn refine_step(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let b = flat_id(wid, lid, dp.grid_x, 64u);
    if (b >= dp.n_items) { return; }
    let sb = b * NS;
    let st = b * 24u;
    let n = max(sums_in[sb], 1.0);
    let mean = sums_in[sb + 1u] / n;
    let var_n = sums_in[sb + 2u] - n * mean * mean;
    var ncc = -2.0;
    var grad = vec3<f32>(0.0, 0.0, 0.0);
    var hess: array<f32, 6>;
    if (var_n > 1e-12) {
        let inv = inverseSqrt(var_n);
        ncc = sums_in[sb + 3u] * inv;
        // crystal-rotation components (3, 4, 5) of the six directions
        let sd = vec3<f32>(sums_in[sb + 7u], sums_in[sb + 8u], sums_in[sb + 9u]);
        let au = (vec3<f32>(sums_in[sb + 13u], sums_in[sb + 14u], sums_in[sb + 15u]) - mean * sd) * inv;
        let bd = vec3<f32>(sums_in[sb + 19u], sums_in[sb + 20u], sums_in[sb + 21u]);
        grad = (bd - ncc * au) * inv;
        // Σ d_i d_j for i, j in 3..5 sit at 37 (3,3) 38 (3,4) 39 (3,5) 40 (4,4) 41 (4,5) 42 (5,5)
        let iv = inv * inv;
        hess[0] = (sums_in[sb + 37u] - sd.x * sd.x / n - au.x * au.x) * iv;
        hess[1] = (sums_in[sb + 38u] - sd.x * sd.y / n - au.x * au.y) * iv;
        hess[2] = (sums_in[sb + 39u] - sd.x * sd.z / n - au.x * au.z) * iv;
        hess[3] = (sums_in[sb + 40u] - sd.y * sd.y / n - au.y * au.y) * iv;
        hess[4] = (sums_in[sb + 41u] - sd.y * sd.z / n - au.y * au.z) * iv;
        hess[5] = (sums_in[sb + 42u] - sd.z * sd.z / n - au.z * au.z) * iv;
    }
    // The Gauss–Newton matrix J'J is the curvature of the NCC only for a pattern that is its
    // simulation: for a real one the curvature is smaller (by the NCC itself, and again where
    // the pattern lacks the simulation's fine detail), so plain steps fall short every time. The
    // matrix is therefore scaled by mu, kept where the gains match what the model promises:
    // on a quadratic whose curvature is mu* times J'J, a step with mu gains (2 − mu* / mu)
    // times its promise, which gives mu* from the gain just seen.
    var mu = state[st + 9u];
    let cur = poses_rw[b * 2u + 1u];
    if (ncc > state[st + 8u]) {
        let promised = state[st + 20u];
        if (state[st + 22u] > 0.5 && promised > 1e-6) {
            let rho = (ncc - state[st + 8u]) / promised;
            mu = clamp(mu * (2.0 - rho), mu / 3.0, mu * 3.0);
        }
        // accept: the current point becomes the best
        state[st + 4u] = cur.x;
        state[st + 5u] = cur.y;
        state[st + 6u] = cur.z;
        state[st + 7u] = cur.w;
        state[st + 8u] = ncc;
        state[st + 10u] = grad.x;
        state[st + 11u] = grad.y;
        state[st + 12u] = grad.z;
        for (var i = 0u; i < 6u; i = i + 1u) { state[st + 13u + i] = hess[i]; }
        // (the starting point is accepted without a step: size 0)
        state[st + 19u] = state[st + 21u];
        state[st + 22u] = state[st + 22u] + 1.0;
    } else {
        // reject: step again from the best point, more cautiously
        mu = mu * 3.0;
    }
    mu = clamp(mu, dp.num.z, dp.num.w);
    state[st + 9u] = mu;
    let qbest = vec4<f32>(state[st + 4u], state[st + 5u], state[st + 6u], state[st + 7u]);
    if ((dp.flags & 1u) != 0u) {
        poses_rw[b * 2u + 1u] = qbest;
        return;
    }
    let g = vec3<f32>(state[st + 10u], state[st + 11u], state[st + 12u]);
    var h: array<f32, 6>;
    for (var i = 0u; i < 6u; i = i + 1u) { h[i] = state[st + 13u + i]; }
    for (var i = 0u; i < 6u; i = i + 1u) { h[i] = h[i] * mu; }
    let tr = max(h[0] + h[3] + h[5], 1e-20);
    h[0] = h[0] + 1e-6 * tr;
    h[3] = h[3] + 1e-6 * tr;
    h[5] = h[5] + 1e-6 * tr;
    var step = solve3(h, g);
    let len = length(step);
    if (len > dp.num.y) { step = step * (dp.num.y / len); }
    // what the model promises this step gains: g·p − ½ p'(mu J'J)p
    let quad = h[0] * step.x * step.x + h[3] * step.y * step.y + h[5] * step.z * step.z
        + 2.0 * (h[1] * step.x * step.y + h[2] * step.x * step.z + h[4] * step.y * step.z);
    state[st + 20u] = dot(g, step) - 0.5 * quad;
    state[st + 21u] = length(step);
    poses_rw[b * 2u + 1u] = qnormalize(qmul(qexp(step), qbest));
}

@compute @workgroup_size(256)
fn sim_norm(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let b = wid.y * dp.grid_x + wid.x;
    let hw = dp.pat_h * dp.pat_w;
    let live = b < dp.n_items;
    var acc = vec2<f32>(0.0, 0.0);
    if (live) {
        for (var i = lid; i < hw; i = i + 256u) {
            if (in_mask(f32(i / dp.pat_w), f32(i % dp.pat_w))) { acc = acc + vec2<f32>(sims[b * hw + i], 1.0); }
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
            if (in_mask(f32(i / dp.pat_w), f32(i % dp.pat_w))) { v = sims[b * hw + i] - mean; }
            sims[b * hw + i] = v;
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
        for (var i = lid; i < hw; i = i + 256u) { sims[b * hw + i] = sims[b * hw + i] * inv; }
    }
}

var<workgroup> red: array<vec2<f32>, 256>;
var<workgroup> redi: array<u32, 256>;

@compute @workgroup_size(256)
fn dict_score(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let e = wid.y * dp.grid_x + wid.x;
    let hw = dp.pat_h * dp.pat_w;
    var bv = -3.4e38;
    var bi = 0u;
    if (e < dp.n_exp) {
        for (var d = lid; d < dp.n_dict; d = d + 256u) {
            var acc = 0.0;
            for (var i = 0u; i < hw; i = i + 1u) { acc = acc + pats[e * hw + i] * sims_in[d * hw + i]; }
            if (acc > bv) {
                bv = acc;
                bi = d;
            }
        }
    }
    red[lid] = vec2<f32>(bv, 0.0);
    redi[lid] = bi;
    workgroupBarrier();
    for (var off = 128u; off > 0u; off = off >> 1u) {
        if (lid < off) {
            let o = lid + off;
            if (red[o].x > red[lid].x || (red[o].x == red[lid].x && redi[o] < redi[lid])) {
                red[lid] = red[o];
                redi[lid] = redi[o];
            }
        }
        workgroupBarrier();
    }
    if (lid == 0u && e < dp.n_exp && red[0].x > bitcast<f32>(dict_best[e * 2u])) {
        dict_best[e * 2u] = bitcast<u32>(red[0].x);
        dict_best[e * 2u + 1u] = dp.dict0 + redi[0];
    }
}

var<workgroup> hist: array<atomic<u32>, 1024>;
var<workgroup> lim: array<vec4<f32>, 256>;

// Normalized mutual information 2 I(A; B) / (H(A) + H(B)) between each item's pattern and its
// simulation, both binned between their own extremes under the mask.
@compute @workgroup_size(256)
fn nmi(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let b = wid.y * dp.grid_x + wid.x;
    let hw = dp.pat_h * dp.pat_w;
    let live = b < dp.n_items;
    let nb = dp.n_bins;
    var pbase = 0u;
    var q = vec4<f32>(1.0, 0.0, 0.0, 0.0);
    if (live) {
        pbase = u32(poses[b * 2u].w) * hw;
        q = poses[b * 2u + 1u];
    }
    // extremes of the pattern (.xy) and of the simulation (.zw)
    var ext = vec4<f32>(3.4e38, -3.4e38, 3.4e38, -3.4e38);
    if (live) {
        for (var i = lid; i < hw; i = i + 256u) {
            let row = i / dp.pat_w;
            let col = i % dp.pat_w;
            if (!in_mask(f32(row), f32(col))) { continue; }
            let e = pats[pbase + i];
            let s = master_at(qrot(q, normalize(ray_of(b, row, col))));
            ext = vec4<f32>(min(ext.x, e), max(ext.y, e), min(ext.z, s), max(ext.w, s));
        }
    }
    lim[lid] = ext;
    for (var i = lid; i < nb * nb; i = i + 256u) { atomicStore(&hist[i], 0u); }
    workgroupBarrier();
    for (var off = 128u; off > 0u; off = off >> 1u) {
        if (lid < off) {
            let o = lim[lid + off];
            let m = lim[lid];
            lim[lid] = vec4<f32>(min(m.x, o.x), max(m.y, o.y), min(m.z, o.z), max(m.w, o.w));
        }
        workgroupBarrier();
    }
    let r = lim[0];
    workgroupBarrier();
    let ke = f32(nb) / max(r.y - r.x, 1e-20);
    let ks = f32(nb) / max(r.w - r.z, 1e-20);
    if (live) {
        for (var i = lid; i < hw; i = i + 256u) {
            let row = i / dp.pat_w;
            let col = i % dp.pat_w;
            if (!in_mask(f32(row), f32(col))) { continue; }
            let e = pats[pbase + i];
            let s = master_at(qrot(q, normalize(ray_of(b, row, col))));
            let be = min(u32((e - r.x) * ke), nb - 1u);
            let bs = min(u32((s - r.z) * ks), nb - 1u);
            atomicAdd(&hist[be * nb + bs], 1u);
        }
    }
    workgroupBarrier();
    // joint entropy over the cells; marginal entropies over the rows and columns (one
    // invocation each)
    var hj = 0.0;
    var tot = 0.0;
    for (var i = lid; i < nb * nb; i = i + 256u) {
        let c = f32(atomicLoad(&hist[i]));
        tot = tot + c;
        if (c > 0.0) { hj = hj - c * log(c); }
    }
    var hm = 0.0;
    if (lid < nb) {
        var rs = 0.0;
        var cs = 0.0;
        for (var k = 0u; k < nb; k = k + 1u) {
            rs = rs + f32(atomicLoad(&hist[lid * nb + k]));
            cs = cs + f32(atomicLoad(&hist[k * nb + lid]));
        }
        if (rs > 0.0) { hm = hm - rs * log(rs); }
        if (cs > 0.0) { hm = hm - cs * log(cs); }
    }
    lim[lid] = vec4<f32>(hj, hm, tot, 0.0);
    workgroupBarrier();
    for (var off = 128u; off > 0u; off = off >> 1u) {
        if (lid < off) { lim[lid] = lim[lid] + lim[lid + off]; }
        workgroupBarrier();
    }
    if (lid == 0u && live) {
        // with counts c and N = Σc: H = log N − (Σ c log c) / N
        let t = lim[0];
        let nn = max(t.z, 1.0);
        let h_joint = log(nn) + t.x / nn;
        let h_marg = 2.0 * log(nn) + t.y / nn;
        var v = 0.0;
        if (h_marg > 1e-12) { v = 2.0 * (h_marg - h_joint) / h_marg; }
        score[dp.dict0 + b] = v;
    }
}
