// Filters on a master pattern as a function on the sphere (after common.wgsl). A master is two
// equal-area sheets, [north | south], side × side each (lambert.zig). Filtering the sheets as
// flat images would smooth by different amounts in different directions (the map is not
// conformal) and so break the symmetry of the crystal; these kernels work in directions instead.
//
//   sphere_blur   Gaussian smoothing on the sphere: each pixel direction, averaged over a
//                 square lattice of directions around it in its tangent plane
//   symmetrize    the mean over the point group of the crystal (after a filter that does not
//                 commute with it, such as CLAHE on the sheets)
//   gradient      the master's gradient on the sphere at every sheet pixel, a tangent vector,
//                 with the master's value beside it (.w): computed once per filtered master,
//                 so that refinement looks intensity and Jacobian up together, in one place,
//                 instead of differencing the master at every pixel of every iteration

struct MsP {
    side: u32,
    grid_x: u32,
    // sphere_blur: lattice points on each side of the centre
    radius: u32,
    // symmetrize: number of operators
    n_ops: u32,
    // sphere_blur: lattice step (radians), 1 / (2 σ²) in radians⁻²
    k: vec4<f32>,
};

@group(0) @binding(0) var<uniform> ms: MsP;
@group(0) @binding(1) var<storage, read> msrc: array<f32>;
@group(0) @binding(2) var<storage, read_write> mdst: array<f32>;
// point-group operators, three vec4 rows each
@group(0) @binding(3) var<storage, read> mops: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> mgrad: array<vec4<f32>>;

// Square coordinates in [−1, 1]² → unit vector on the northern or southern hemisphere.
fn square_to_sphere(p: vec2<f32>, south: bool) -> vec3<f32> {
    let sc = 1.2533141373155003; // sqrt(π / 2)
    let xa = abs(p.x) * sc;
    let ya = abs(p.y) * sc;
    let swap = xa >= ya;
    let xn = max(xa, ya);
    let yn = min(xa, ya);
    var v = vec3<f32>(0.0, 0.0, 1.0);
    if (xn > 0.0) {
        let r = (2.0 * xn / PI) * sqrt(max(PI - xn * xn, 0.0));
        let ang = PI * yn / (4.0 * xn);
        var hx = r * cos(ang);
        var hy = r * sin(ang);
        if (!swap) {
            let t = hx;
            hx = hy;
            hy = t;
        }
        v = normalize(vec3<f32>(hx * select(1.0, -1.0, p.x < 0.0), hy * select(1.0, -1.0, p.y < 0.0), 1.0 - 2.0 * xn * xn / PI));
    }
    if (south) { v.z = -v.z; }
    return v;
}

fn msrc_at(v: vec3<f32>) -> f32 {
    let uv = sphere_to_square(v);
    let last = f32(ms.side - 1u);
    let fj = clamp((uv.x + 1.0) * 0.5 * last, 0.0, last);
    let fi = clamp((uv.y + 1.0) * 0.5 * last, 0.0, last);
    let c0 = min(u32(fj), ms.side - 2u);
    let r0 = min(u32(fi), ms.side - 2u);
    var o = r0 * ms.side + c0;
    if (v.z < 0.0) { o = o + ms.side * ms.side; }
    let tj = fj - f32(c0);
    let top = mix(msrc[o], msrc[o + 1u], tj);
    let bot = mix(msrc[o + ms.side], msrc[o + ms.side + 1u], tj);
    return mix(top, bot, fi - f32(r0));
}

fn pixel_dir(tid: u32) -> vec3<f32> {
    let plane = ms.side * ms.side;
    let pix = tid % plane;
    let last = f32(ms.side - 1u);
    let p = vec2<f32>(2.0 * f32(pix % ms.side) / last - 1.0, 2.0 * f32(pix / ms.side) / last - 1.0);
    return square_to_sphere(p, tid >= plane);
}

@compute @workgroup_size(256)
fn sphere_blur(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let tid = flat_id(wid, lid, ms.grid_x, 256u);
    if (tid >= 2u * ms.side * ms.side) { return; }
    let d = pixel_dir(tid);
    var ref_axis = vec3<f32>(1.0, 0.0, 0.0);
    if (abs(d.x) > 0.8) { ref_axis = vec3<f32>(0.0, 1.0, 0.0); }
    let e1 = normalize(cross(d, ref_axis));
    let e2 = cross(d, e1);
    let r = i32(ms.radius);
    var acc = 0.0;
    var wsum = 0.0;
    for (var i = -r; i <= r; i = i + 1) {
        for (var j = -r; j <= r; j = j + 1) {
            let a = f32(i) * ms.k.x;
            let b = f32(j) * ms.k.x;
            let w = exp(-(a * a + b * b) * ms.k.y);
            acc = acc + w * msrc_at(normalize(d + a * e1 + b * e2));
            wsum = wsum + w;
        }
    }
    mdst[tid] = acc / wsum;
}

@compute @workgroup_size(256)
fn symmetrize(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let tid = flat_id(wid, lid, ms.grid_x, 256u);
    if (tid >= 2u * ms.side * ms.side) { return; }
    let d = pixel_dir(tid);
    var acc = 0.0;
    for (var g = 0u; g < ms.n_ops; g = g + 1u) {
        let v = vec3<f32>(dot(mops[g * 3u].xyz, d), dot(mops[g * 3u + 1u].xyz, d), dot(mops[g * 3u + 2u].xyz, d));
        acc = acc + msrc_at(normalize(v));
    }
    mdst[tid] = acc / f32(max(ms.n_ops, 1u));
}

// Central differences a sheet pixel apart (ms.k.x radians) along two tangent directions.
@compute @workgroup_size(256)
fn gradient(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let tid = flat_id(wid, lid, ms.grid_x, 256u);
    if (tid >= 2u * ms.side * ms.side) { return; }
    let c = pixel_dir(tid);
    var ref_axis = vec3<f32>(1.0, 0.0, 0.0);
    if (abs(c.x) > 0.8) { ref_axis = vec3<f32>(0.0, 1.0, 0.0); }
    let e1 = normalize(cross(c, ref_axis));
    let e2 = cross(c, e1);
    let h = ms.k.x;
    let g1 = (msrc_at(normalize(c + h * e1)) - msrc_at(normalize(c - h * e1))) / (2.0 * h);
    let g2 = (msrc_at(normalize(c + h * e2)) - msrc_at(normalize(c - h * e2))) / (2.0 * h);
    mgrad[tid] = vec4<f32>(g1 * e1 + g2 * e2, msrc[tid]);
}
