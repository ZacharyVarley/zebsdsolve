// The crystal's unit cell in a picture, for the app (cellview.zig): its atoms as balls and its
// twelve edges as rods, seen by an orthographic camera, lit from the upper left. Every pixel
// casts four rays (2 × 2 within it) and keeps the nearest thing each meets.
//
//   cell_view   width × height packed RGBA8 pixels

struct CvP {
    width: u32,
    height: u32,
    n_atoms: u32,
    // what to pick out: 0 nothing; 1, 2, 3 the edge a, b or c; 4, 5, 6 the angle α, β or γ
    // (its two edges); 16 + k the atoms of site k
    highlight: u32,
    // the view's rotation, crystal ← view (three rows)
    rot: array<vec4<f32>, 3>,
    // the lattice vectors a, b, c (Å, Cartesian), and the cell's corner at the origin of the
    // fractional coordinates, all from the middle of the cell
    basis: array<vec4<f32>, 3>,
    origin: vec4<f32>,
    // pixels per Å, the rods' radius (Å), 1 / the depth of the scene (Å⁻¹), how far in front of
    // the middle of the cell the rays start (Å)
    k: vec4<f32>,
};

// one atom: its centre from the middle of the cell (Å) and radius; its colour and site
struct Atom {
    p: vec4<f32>,
    c: vec4<f32>,
};

@group(0) @binding(0) var<uniform> cv: CvP;
@group(0) @binding(1) var<storage, read> atoms: array<Atom>;
@group(0) @binding(2) var<storage, read_write> cpix: array<u32>;

const NONE: f32 = 1e30;
const EDGE_GREY: vec3<f32> = vec3<f32>(0.52, 0.52, 0.56);
const PICK: vec3<f32> = vec3<f32>(0.91, 0.78, 0.49);

// the nearest positive distance along the ray (unit direction) at which it meets a ball, or NONE
fn ball(ro: vec3<f32>, rd: vec3<f32>, c: vec3<f32>, r: f32) -> f32 {
    let oc = ro - c;
    let b = dot(oc, rd);
    let h = b * b - (dot(oc, oc) - r * r);
    if (h < 0.0) { return NONE; }
    let t = -b - sqrt(h);
    return select(NONE, t, t > 0.0);
}

// … at which it meets the rod from pa to pb of radius r (a cylinder with a ball at each end)
fn rod(ro: vec3<f32>, rd: vec3<f32>, pa: vec3<f32>, pb: vec3<f32>, r: f32) -> f32 {
    var best = min(ball(ro, rd, pa, r), ball(ro, rd, pb, r));
    let ba = pb - pa;
    let oa = ro - pa;
    let baba = dot(ba, ba);
    let bard = dot(ba, rd);
    let baoa = dot(ba, oa);
    let a = baba - bard * bard;
    let b = baba * dot(rd, oa) - baoa * bard;
    let c = baba * dot(oa, oa) - baoa * baoa - r * r * baba;
    let h = b * b - a * c;
    // (a ray along the rod meets only its ends)
    if (h >= 0.0 && a > 1e-9 * baba) {
        let t = (-b - sqrt(h)) / a;
        let y = baoa + t * bard;
        if (y > 0.0 && y < baba && t > 0.0) { best = min(best, t); }
    }
    return best;
}

// edge e of the twelve: its ends (from the middle of the cell) and which lattice vector it runs along
// The lines of the cells' grid: with n cells along each vector, (n + 1)² lines along each,
// each n cells long: 12 lines for the one cell.
fn cells_n() -> u32 {
    return max(u32(cv.origin.w + 0.5), 1u);
}

fn edge(e: u32) -> array<vec3<f32>, 2> {
    let n = cells_n();
    let m = n + 1u;
    let axis = e / (m * m);
    let j = e % (m * m);
    let a = cv.basis[axis].xyz;
    let u = cv.basis[(axis + 1u) % 3u].xyz;
    let v = cv.basis[(axis + 2u) % 3u].xyz;
    let p0 = cv.origin.xyz + f32(j % m) * u + f32(j / m) * v;
    return array<vec3<f32>, 2>(p0, p0 + f32(n) * a);
}

// whether a line is on the outside of the cells (the others are the cells' inner faces)
fn outer(e: u32) -> bool {
    let n = cells_n();
    let m = n + 1u;
    let j = e % (m * m);
    let ju = j % m;
    let jv = j / m;
    return (ju == 0u || ju == n) && (jv == 0u || jv == n);
}

// whether edge e is picked out: the edge from the origin along the lattice vector asked for, or
// the two that make the angle asked for
fn picked(e: u32) -> bool {
    let hl = cv.highlight;
    let m = cells_n() + 1u;
    let axis = e / (m * m);
    return e % (m * m) == 0u && ((hl >= 1u && hl <= 3u && axis == hl - 1u) || (hl >= 4u && hl <= 6u && axis != hl - 4u));
}

fn edge_colour(e: u32) -> vec3<f32> {
    if (picked(e)) { return PICK; }
    // the three edges from the origin: a, b and c in the colours of x, y and z on the globe
    let m = cells_n() + 1u;
    if (e % (m * m) != 0u) { return select(EDGE_GREY * 0.6, EDGE_GREY, outer(e)); }
    let axis = e / (m * m);
    return select(select(vec3<f32>(0.42, 0.62, 0.78), vec3<f32>(0.5, 0.75, 0.43), axis == 1u), vec3<f32>(0.88, 0.44, 0.37), axis == 0u);
}

// the colour seen along one ray (origin and direction in the crystal's frame)
fn trace(ro: vec3<f32>, rd: vec3<f32>) -> vec4<f32> {
    var best = NONE;
    var normal = vec3<f32>(0.0, 0.0, 1.0);
    var base = vec3<f32>(0.0);
    var glow = 0.0;
    var dim = false;
    let hl = cv.highlight;
    for (var i = 0u; i < cv.n_atoms; i = i + 1u) {
        let at = atoms[i];
        let t = ball(ro, rd, at.p.xyz, at.p.w);
        if (t < best) {
            best = t;
            normal = (ro + t * rd - at.p.xyz) / at.p.w;
            base = at.c.rgb;
            let site = u32(at.c.w + 0.5);
            glow = select(0.0, 1.0, hl >= 16u && site == hl - 16u);
            dim = hl >= 16u && site != hl - 16u;
        }
    }
    let r = cv.k.y;
    let lines = 3u * (cells_n() + 1u) * (cells_n() + 1u);
    for (var e = 0u; e < lines; e = e + 1u) {
        let ends = edge(e);
        let t = rod(ro, rd, ends[0], ends[1], select(select(0.45 * r, r, outer(e)), 1.6 * r, picked(e)));
        if (t < best) {
            best = t;
            let p = ro + t * rd;
            let ba = ends[1] - ends[0];
            let foot = ends[0] + clamp(dot(p - ends[0], ba) / dot(ba, ba), 0.0, 1.0) * ba;
            normal = normalize(p - foot);
            base = edge_colour(e);
            glow = 0.0;
            dim = false;
        }
    }
    if (best >= NONE) { return vec4<f32>(0.0); }
    // light in the view's frame: the normal turned back (the rotation's transpose)
    let n = normalize(vec3<f32>(
        cv.rot[0].x * normal.x + cv.rot[1].x * normal.y + cv.rot[2].x * normal.z,
        cv.rot[0].y * normal.x + cv.rot[1].y * normal.y + cv.rot[2].y * normal.z,
        cv.rot[0].z * normal.x + cv.rot[1].z * normal.y + cv.rot[2].z * normal.z));
    let light = normalize(vec3<f32>(-0.4, 0.55, 0.75));
    let diffuse = max(dot(n, light), 0.0);
    let half_v = normalize(light + vec3<f32>(0.0, 0.0, 1.0));
    let spec = pow(max(dot(n, half_v), 0.0), 40.0);
    var col = base * (0.28 + 0.72 * diffuse) + vec3<f32>(0.35 * spec);
    if (dim) { col = mix(col, vec3<f32>(0.08), 0.62); }
    col = mix(col, PICK, 0.35 * glow) + 0.12 * glow * vec3<f32>(spec);
    // far things a little darker: depth reads at a glance (k.w is how far in front of the
    // middle of the cell the rays start, k.z one over the depth of the whole picture)
    let depth = clamp((best - cv.k.w) * cv.k.z + 0.5, 0.0, 1.0);
    col = col * (1.0 - 0.32 * depth);
    return vec4<f32>(col, 1.0);
}

@compute @workgroup_size(16, 16)
fn cell_view(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = cv.width;
    let h = cv.height;
    if (gid.x >= w || gid.y >= h) { return; }
    let bg = vec3<f32>(10.0, 10.0, 11.0) / 255.0;
    // rays toward −z of the view, from in front of everything
    let rd = vec3<f32>(cv.rot[0].z, cv.rot[1].z, cv.rot[2].z) * -1.0;
    let far = cv.k.w;
    var acc = vec3<f32>(0.0);
    for (var s = 0u; s < 4u; s = s + 1u) {
        let sub = vec2<f32>(0.25 + 0.5 * f32(s & 1u), 0.25 + 0.5 * f32(s >> 1u));
        let x = (f32(gid.x) + sub.x - 0.5 * f32(w)) / cv.k.x;
        let y = (0.5 * f32(h) - f32(gid.y) - sub.y) / cv.k.x;
        let o = vec3<f32>(x, y, far);
        let ro = vec3<f32>(dot(cv.rot[0].xyz, o), dot(cv.rot[1].xyz, o), dot(cv.rot[2].xyz, o));
        let c = trace(ro, rd);
        acc = acc + mix(bg, c.rgb, c.a);
    }
    cpix[gid.y * w + gid.x] = pack4x8unorm(vec4<f32>(acc * 0.25, 1.0));
}
