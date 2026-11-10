// Filters on a master pattern as a function on the sphere (after common.wgsl). A master is two
// equal-area sheets, [north | south], side × side each (lambert.zig). Filtering the sheets as
// flat images would smooth by different amounts in different directions (the map is not
// conformal) and so break the symmetry of the crystal; these kernels work in directions instead.
//
//   sphere_blur   Gaussian smoothing on the sphere: each pixel direction, averaged over a
//                 square lattice of directions around it in its tangent plane
//   symmetrize    the mean over the point group of the crystal (after a filter that does not
//                 commute with it, such as CLAHE on the sheets)
//   sphere_view   a picture of the master for the app, in one of four projections (a lit
//                 globe, the Lambert squares, the stereographic discs, or the whole sphere in
//                 one square): each pixel a direction of the view, turned into the crystal's
//                 frame and looked up in the sheets
//   gradient      the master's gradient on the sphere at every sheet pixel, a tangent vector,
//                 with the master's value beside it (.w): computed once per filtered master,
//                 so that refinement looks intensity and Jacobian up together, in one place,
//                 instead of differencing the master at every pixel of every iteration

struct MsP {
    side: u32,
    grid_x: u32,
    // sphere_blur: lattice points on each side of the centre
    radius: u32,
    // symmetrize: number of operators; sphere_view: the projection
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
// sphere_view: the picture, one packed RGBA8 pixel each
@group(0) @binding(5) var<storage, read_write> mpix: array<u32>;

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

// The master in a picture: grid_x × radius pixels (width × height) of one of four projections
// (ms.n_ops): 0 a globe, orthographic and lit; 1 the Lambert squares and 2 the stereographic
// discs, both hemispheres (the upper beside, or over, the lower, each seen from above) or one
// of them; 3 the whole sphere in one square, the joint Lambert projection: the upper square in
// the middle as a diamond, the lower square's four triangles folded out around it into the
// corners. The view's rotation (crystal ← view, three rows) in mops[0 … 2]; mops[3] = (the
// picture's pan across and down, in shorter sides; the hemispheres shown, 0 both, 1 upper,
// 2 lower; 1: show I(d) − I(−d), the upper hemisphere less the lower one's inversion, signed);
// ms.k = (zoom, the contrast's low end, 1 / its range, unused). Zoom scales the whole layout
// about the picture's middle (then the pan moves it). Outside the projection the well's colour;
// its edges antialiased over a pixel.
fn in_panel(mode: u32, q: vec2<f32>, step: f32) -> bool {
    if (mode == 1u || mode == 3u) {
        return max(abs(q.x), abs(q.y)) < 1.0 + step;
    }
    return length(q) < 1.0 + step;
}

@compute @workgroup_size(16, 16)
fn sphere_view(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = ms.grid_x;
    let h = ms.radius;
    if (gid.x >= w || gid.y >= h) { return; }
    let fw = f32(w);
    let fh = f32(h);
    let m = min(fw, fh);
    let px = vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5);
    let bg = vec3<f32>(10.0, 10.0, 11.0) / 255.0;
    let mode = ms.n_ops;
    let z = ms.k.x;
    let pan = mops[3].xy * m;
    let shown = u32(mops[3].z + 0.5);
    let diff = mops[3].w > 0.5;
    let centre = 0.5 * vec2<f32>(fw, fh) + pan;
    let flat_view = mode == 1u || mode == 2u;
    let two = flat_view && shown == 0u;
    var c0 = centre;
    var c1 = centre;
    var half = 0.5 * m * z;
    var lower = flat_view && shown == 2u;
    if (two) {
        let wide = fw >= fh;
        half = 0.5 * select(min(fw, 0.5 * fh), min(0.5 * fw, fh), wide) * z;
        let off = select(vec2<f32>(0.0, 0.25 * fh), vec2<f32>(0.25 * fw, 0.0), wide) * z;
        c0 = centre - off;
        c1 = centre + off;
    }
    var q = vec2<f32>(px.x - c0.x, c0.y - px.y) / half;
    let step = 1.0 / half;
    if (two && !in_panel(mode, q, step)) {
        lower = true;
        q = vec2<f32>(px.x - c1.x, c1.y - px.y) / half;
    }
    var col = bg;
    var d = vec3<f32>(0.0, 0.0, 1.0);
    // how much of the pixel the projection covers, and the light on it
    var cover = 0.0;
    var lit = 1.0;
    if (mode == 0u) {
        let r = length(q);
        if (r < 1.0 + step) {
            d = vec3<f32>(q, sqrt(max(1.0 - r * r, 0.0)));
            if (r >= 1.0) { d = vec3<f32>(q / r, 0.0); }
            cover = clamp((1.0 - r) / step + 0.5, 0.0, 1.0);
            // soft light from the upper left, and the rim a little darker (the globe reads as round)
            let light = normalize(vec3<f32>(-0.35, 0.45, 0.82));
            lit = (0.56 + 0.44 * max(dot(d, light), 0.0)) * (0.8 + 0.2 * d.z);
        }
    } else if (mode == 1u) {
        let e = max(abs(q.x), abs(q.y));
        if (e < 1.0 + step) {
            d = square_to_sphere(clamp(q, vec2<f32>(-1.0), vec2<f32>(1.0)), lower);
            cover = clamp((1.0 - e) / step + 0.5, 0.0, 1.0);
        }
    } else if (mode == 2u) {
        let r = length(q);
        if (r < 1.0 + step) {
            let c = q / max(r, 1.0);
            let r2 = dot(c, c);
            d = vec3<f32>(2.0 * c, 1.0 - r2) / (1.0 + r2);
            if (lower) { d.z = -d.z; }
            cover = clamp((1.0 - r) / step + 0.5, 0.0, 1.0);
        }
    } else {
        let e = max(abs(q.x), abs(q.y));
        if (e < 1.0 + step) {
            // the square's coordinates turned back 45° onto the diamond |u| + |v| ≤ 2
            let c = clamp(q, vec2<f32>(-1.0), vec2<f32>(1.0));
            let u = c.x + c.y;
            let v = c.y - c.x;
            if (abs(u) <= 1.0 && abs(v) <= 1.0) {
                d = square_to_sphere(vec2<f32>(u, v), false);
            } else {
                // a fold: the lower square's triangle, reflected across the upper square's edge
                var s = vec2<f32>(u, -2.0 - v);
                if (u > 1.0) { s = vec2<f32>(2.0 - u, v); }
                else if (u < -1.0) { s = vec2<f32>(-2.0 - u, v); }
                else if (v > 1.0) { s = vec2<f32>(u, 2.0 - v); }
                d = square_to_sphere(clamp(s, vec2<f32>(-1.0), vec2<f32>(1.0)), true);
            }
            cover = clamp((1.0 - e) / step + 0.5, 0.0, 1.0);
        }
    }
    if (cover > 0.0) {
        let v = normalize(vec3<f32>(dot(mops[0].xyz, d), dot(mops[1].xyz, d), dot(mops[2].xyz, d)));
        var shade = vec3<f32>(0.0);
        if (diff) {
            // orange where the direction is brighter than its inversion, blue where darker
            let a = (msrc_at(v) - msrc_at(-v)) * ms.k.z * 2.0;
            shade = mix(vec3<f32>(0.04), select(vec3<f32>(0.31, 0.55, 0.88), vec3<f32>(0.95, 0.6, 0.22), a > 0.0), clamp(abs(a), 0.0, 1.0)) * lit;
        } else {
            let t = clamp((msrc_at(v) - ms.k.y) * ms.k.z, 0.0, 1.0);
            shade = vec3<f32>(t * lit);
        }
        col = mix(bg, shade, cover);
    }
    mpix[gid.y * w + gid.x] = pack4x8unorm(vec4<f32>(col, 1.0));
}
