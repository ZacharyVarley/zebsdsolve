// The dynamical part of a master pattern's simulation: for every direction of the fundamental
// sector, the backscattered intensity of a Bloch-wave field, depth-integrated under an
// exponential depth profile (src/simulate.zig has the schedule).
//
// A direction's wave vector k (reciprocal-lattice components) excites the reflections g with
// excitation errors s_g. The beams are the n reflections whose |s_g| over their coupling
// scale is least (000 always among them, first). Their amplitudes ψ(z) obey ψ' = A ψ with
//     A_ii = −πλ U'_0 + 2πi s_i,      A_ij = iπλ U(g_i − g_j),
// and the depth-integrated density X = ∫ e^{Ãz} e₀ e₀ᴴ e^{Ãᴴz} dz, Ã = A − (μ/2) I (μ: the
// Monte Carlo depth decay less the mean absorption), solves the Lyapunov equation
//     Ã X + X Ãᴴ + e₀ e₀ᴴ = 0.
// The intensity of site s is f_s = Σ_ij Re[S_s(g_j − g_i) X_ji] = tr(S_s X).
// X is taken in a rational Krylov subspace of a few columns (Druskin and Simoncini's RKSM, one
// repeated pole) by Galerkin projection, and the projection's error in f corrected to second
// order by the dual-weighted residual (the passes are described at "the solve" below).
//
// Bethe's weak beams: the reflections after the n strong ones whose score is within c_weak
// (and, when a direction has more strong beams than MAX_N, the strong ones past MAX_N) are not
// solved for but followed: ψ_w = W ψ, W = −diag(1 / d_w) A_ws with d_w = 2πi s_w, so that the
// strong beams solve with A_eff = A_ss + A_sw W and the intensity runs over strong and weak
// beams together, X_full = T X Tᴴ with T = [I; W] (in the low-rank factors: T G and T Q).
//
//   strong_count   per direction, the beams with |s|/coupling ≤ c_strong, and those within the
//                  weak cutoff: the bin's n and n_w are the most any direction has
//   select_beams   per direction (a workgroup), the n least scores and then the n_w next ones,
//                  by an exact radix select, each set in the reflections' order (000 first)
//   sim_prep, sim_solve, sim_core, sim_intensity: the solve
//
// BPT (rows a thread in sim_solve, ⌈n / 256⌉) is set by a line the engine puts before the text.

const WG: u32 = 256u;
// MAX_N (the beams a direction solves for, and the weak beams it follows) is set by a line the
// engine puts before the text, from the device's workgroup memory: 512 in WebGPU's default
// 16 KiB, 1024 in 32 KiB, 1536 in 48 KiB
const MAX_W: u32 = MAX_N;
// an empty weak beam's slot
const NONE: u32 = 0xffffffffu;
const MAX_R: u32 = 16u;
const MAX_SITES: u32 = 16u;
const PI: f32 = 3.141592653589793;
const KEY_INF: u32 = 0xffffffffu;

struct SimP {
    n_k: u32,
    n_g: u32,
    n: u32,
    rank: u32,
    n_sites: u32,
    table_size: u32,
    offset: i32,
    grid_x: u32,
    // the reciprocal metric's rows
    m0: vec4<f32>,
    m1: vec4<f32>,
    m2: vec4<f32>,
    // c_strong, c_cutoff, the double-diffraction |s| cutoff, the weak beams' cutoff (at least
    // c_strong: strong beams past MAX_N are weak)
    bethe: vec4<f32>,
    // πλ, the mean absorption U'_0, μ, the amplitude (the bin's weight × its depth amplitude)
    phys: vec4<f32>,
    // 1: the dual-weighted correction; a direction's scratch length (vec2 units); the weak
    // beams a direction (n_w); unused
    flags: vec4<u32>,
};

@group(0) @binding(0) var<uniform> sp: SimP;
// (h, k, l, the beam's hash h·stride_h + k·stride_k + l)
@group(0) @binding(1) var<storage, read> hkl: array<vec4<i32>>;
@group(0) @binding(2) var<storage, read> coupling: array<f32>;
@group(0) @binding(3) var<storage, read> dbdiff: array<u32>;
// per direction of this dispatch: k in reciprocal-lattice components (.w unused)
@group(0) @binding(4) var<storage, read> kvecs: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> keys: array<u32>;
@group(0) @binding(6) var<storage, read_write> beams: array<u32>;
@group(0) @binding(7) var<storage, read_write> counter: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read> table: array<vec2<f32>>;
@group(0) @binding(9) var<storage, read> sgh: array<vec2<f32>>;
@group(0) @binding(10) var<storage, read_write> work: array<vec2<f32>>;
@group(0) @binding(11) var<storage, read_write> intensity: array<f32>;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
fn cconj(a: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x, -a.y);
}
fn cabs2(a: vec2<f32>) -> f32 {
    return a.x * a.x + a.y * a.y;
}
fn cdiv(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let d = max(cabs2(b), 1.0e-30);
    return vec2<f32>((a.x * b.x + a.y * b.y) / d, (a.y * b.x - a.x * b.y) / d);
}

fn dot_metric(a: vec3<f32>, b: vec3<f32>) -> f32 {
    let mb = vec3<f32>(dot(sp.m0.xyz, b), dot(sp.m1.xyz, b), dot(sp.m2.xyz, b));
    return dot(a, mb);
}

// The excitation error of reflection h for wave vector k (ebsdsim's, in the crystal's metric).
fn excitation(k: vec3<f32>, h: vec3<f32>) -> f32 {
    let kpg = k + h;
    let tkpg = 2.0 * k + h;
    let xnom = -dot_metric(h, tkpg);
    let q1 = sqrt(max(dot_metric(kpg, kpg), 0.0));
    let klen = sqrt(max(dot_metric(k, k), 0.0));
    let kpg_k = dot_metric(kpg, k);
    let xden = 2.0 * q1 * kpg_k / (q1 * klen + 1e-16);
    return xnom / (xden + 1e-16);
}

// A reflection's score for a direction: its |s| over its coupling scale, and whether it is a
// candidate at all.
fn score(k: vec3<f32>, g: u32) -> vec2<f32> {
    if (g == 0u) {
        return vec2<f32>(0.0, 1.0);
    }
    let h = hkl[g];
    let s = abs(excitation(k, vec3<f32>(f32(h.x), f32(h.y), f32(h.z))));
    let ratio = s / max(coupling[g], 1e-12);
    var cand = ratio <= sp.bethe.y;
    if (dbdiff[g] != 0u) {
        cand = s <= sp.bethe.z;
    }
    return vec2<f32>(ratio, select(0.0, 1.0, cand));
}

fn flat_group(wid: vec3<u32>) -> u32 {
    return wid.x + wid.y * sp.grid_x;
}

@compute @workgroup_size(64)
fn strong_count(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let b = flat_group(wid) * 64u + lid;
    if (b >= sp.n_k) {
        return;
    }
    let k = kvecs[b].xyz;
    var strong = 1u;
    var within = 1u;
    for (var g = 1u; g < sp.n_g; g = g + 1u) {
        let sc = score(k, g);
        if (sc.y > 0.5) {
            strong = strong + select(0u, 1u, sc.x <= sp.bethe.x);
            within = within + select(0u, 1u, sc.x <= sp.bethe.w);
        }
    }
    atomicMax(&counter[0], strong);
    atomicMax(&counter[2], within);
    if (strong > MAX_N) {
        atomicAdd(&counter[3], 1u);
    }
}

// ── beam selection ──

var<workgroup> hist: array<atomic<u32>, 256>;
var<workgroup> scan: array<u32, 256>;
var<workgroup> rs: array<u32, 4>; // prefix, mask, still wanted, total

// Monotone u32 keys for non-negative f32 scores; not a candidate: the worst.
fn key_of(sc: vec2<f32>) -> u32 {
    if (sc.y < 0.5 || sc.x != sc.x) {
        return KEY_INF - 1u;
    }
    return bitcast<u32>(max(sc.x, 0.0));
}

// Exclusive prefix sum of scan[0..256) in place; returns the total. All threads call it.
fn exclusive_scan(lid: u32) -> u32 {
    workgroupBarrier();
    if (lid == 0u) {
        var acc = 0u;
        for (var i = 0u; i < WG; i = i + 1u) {
            let v = scan[i];
            scan[i] = acc;
            acc = acc + v;
        }
        rs[3] = acc;
    }
    workgroupBarrier();
    return rs[3];
}

// The `want` least keys of a direction, by an exact radix select eight bits at a time: leaves
// the threshold key in rs[0] and how many keys at it are wanted in rs[2]. All threads call it.
fn radix_select(lid: u32, base: u32, n_g: u32, want: u32) {
    workgroupBarrier();
    if (lid == 0u) {
        rs[0] = 0u;
        rs[1] = 0u;
        rs[2] = want;
    }
    storageBarrier();
    workgroupBarrier();
    for (var round = 0u; round < 4u; round = round + 1u) {
        let shift = 24u - 8u * round;
        atomicStore(&hist[lid], 0u);
        workgroupBarrier();
        let prefix = rs[0];
        let mask = rs[1];
        for (var g = lid; g < n_g; g = g + WG) {
            let key = keys[base + g];
            if ((key & mask) == prefix) {
                atomicAdd(&hist[(key >> shift) & 255u], 1u);
            }
        }
        workgroupBarrier();
        if (lid == 0u) {
            var want_left = rs[2];
            var acc = 0u;
            var digit = 255u;
            for (var d = 0u; d < 256u; d = d + 1u) {
                let c = atomicLoad(&hist[d]);
                if (acc + c >= want_left) {
                    digit = d;
                    break;
                }
                acc = acc + c;
            }
            rs[0] = prefix | (digit << shift);
            rs[1] = mask | (255u << shift);
            rs[2] = want_left - acc;
        }
        workgroupBarrier();
    }
}

// The keys radix_select chose, their reflections written to beams[out …] in the reflections'
// order; each taken key becomes KEY_INF (a later selection passes it by); a key above `limit`
// leaves its slot as it is. All threads call it.
fn take_selected(lid: u32, base: u32, n_g: u32, out: u32, limit: u32) {
    let threshold = rs[0];
    let ties = rs[2];
    // each thread a run of reflections: those below the threshold, and the first `ties` at it
    let run = (n_g + WG - 1u) / WG;
    let g0 = min(lid * run, n_g);
    let g1 = min(g0 + run, n_g);
    var below = 0u;
    var at = 0u;
    for (var g = g0; g < g1; g = g + 1u) {
        let key = keys[base + g];
        below = below + select(0u, 1u, key < threshold);
        at = at + select(0u, 1u, key == threshold);
    }
    scan[lid] = at;
    _ = exclusive_scan(lid);
    let at_before = scan[lid];
    let take_at = min(at, select(0u, ties - at_before, ties > at_before));
    workgroupBarrier();
    scan[lid] = below + take_at;
    _ = exclusive_scan(lid);
    var pos = scan[lid];
    var taken_at = 0u;
    for (var g = g0; g < g1; g = g + 1u) {
        let key = keys[base + g];
        var take = key < threshold;
        if (key == threshold && taken_at < take_at) {
            take = true;
            taken_at = taken_at + 1u;
        }
        if (take) {
            if (key <= limit) {
                beams[out + pos] = g;
            }
            keys[base + g] = KEY_INF;
            pos = pos + 1u;
        }
    }
}

@compute @workgroup_size(256)
fn select_beams(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let b = flat_group(wid);
    if (b >= sp.n_k) {
        return;
    }
    let k = kvecs[b].xyz;
    let n_g = sp.n_g;
    let base = b * n_g;
    let nw = sp.flags.z;
    let out = b * (sp.n + nw);
    for (var g = lid; g < n_g; g = g + WG) {
        keys[base + g] = key_of(score(k, g));
    }
    // the weak slots start empty (a direction may have fewer weak beams than the bin's n_w)
    for (var w = lid; w < nw; w = w + WG) {
        beams[out + sp.n + w] = NONE;
    }
    let strong = min(sp.n, n_g);
    radix_select(lid, base, n_g, strong);
    take_selected(lid, base, n_g, out, KEY_INF);
    if (nw > 0u) {
        storageBarrier();
        radix_select(lid, base, n_g, min(nw, n_g - strong));
        take_selected(lid, base, n_g, out + sp.n, bitcast<u32>(sp.bethe.w));
    }
}

// ── the solve ──
//
// Per direction (a workgroup each), in four passes that share the direction's scratch (`work`,
// laid out by the offsets below):
//
//   sim_prep       the beams' diagonal (−πλU'₀ + 2πi s), the pole q, and the preconditioner:
//                  the CORE beams nearest the Bragg condition (least |s|, 000 always) keep their
//                  mutual coupling exactly, M_c = q₁ I − (D + C) on them inverted (Gauss–Jordan)
//   sim_solve      the Cayley columns z_k (one repeated pole), each (qI − Ã) z = w by BiCGSTAB
//                  with the left Jacobi scaling L = diag(q₁ − d)^(−1/2) and the right
//                  preconditioner P = block-diag(M_c⁻¹, diag(M)⁻¹) L⁻¹ (= L off the core);
//                  Gram–Schmidt twice; Ã Q; H = Qᴴ Ã Q, b = Qᴴ e₀; H Y + Y Hᴴ + b bᴴ = 0 by
//                  squared Smith; G = Q Y and AG = (Ã Q) Y
//   sim_core       the correction's core: the CORE beams where the Galerkin residual
//                  R = Ã X + X Ãᴴ + e₀ e₀ᴴ (X = Q Y Qᴴ) is largest against |d|, and on them
//                  Ã_c X̃ + X̃ Ã_cᴴ + R_c = 0 by Cayley and Smith doubling
//   sim_intensity  I_s = Σ_ij Re[S_s(g_j − g_i) Xc_ji] over half the pairs (S_s and Xc are
//                  Hermitian), Xc = X corrected by the dual-weighted residual: off the core
//                  Xc_ji = X_ji − R_ji / (conj d_i + d_j) (the dual taken diagonal), on core pairs
//                  Xc_ji = X_ji + X̃_ji (the dual exact there)
//
// The correction: the Galerkin estimate's error is f − f_G = tr(Y R) with the dual
// Ãᴴ Y + Y Ã + S = 0; with an approximate dual the error left is second order. On the core the
// dual's part is, by the same duality, tr(S_c X̃).
// Within WebGPU's default limits: 256 invocations, 16 KiB of workgroup memory, eight storage
// buffers; the coupling U(g_i − g_j) is read from the table (it stays in cache a direction).

const CORE: u32 = 30u;       // beams in either core
const CS: u32 = 31u;         // the core matrices' row stride in workgroup memory (no bank conflicts)
const CORE_SQ: u32 = 1024u;  // a core matrix's room in scratch (stride CORE)
const BICG_TOL: f32 = 5.0e-3;
const BICG_MAX: u32 = 256u;
const DOUBLINGS: u32 = 8u;

// scratch offsets (vec2 units) for a direction of n beams at rank r
fn o_d(n: u32) -> u32 { return 16u; }
fn o_cp(n: u32) -> u32 { return 16u + n; }
fn o_c2(n: u32) -> u32 { return 16u + 2u * n; }
fn o_k(n: u32) -> u32 { return 16u + 3u * n; }
fn o_xt(n: u32) -> u32 { return o_k(n) + CORE_SQ; }
fn o_xs(n: u32) -> u32 { return o_xt(n) + CORE_SQ; }
fn o_gr(n: u32) -> u32 { return o_xs(n) + CORE_SQ; }
fn o_sm(n: u32) -> u32 { return o_gr(n) + CORE_SQ; }
fn o_q(n: u32) -> u32 { return o_sm(n) + 2u * MAX_R * MAX_R; }
fn o_aq(n: u32, r: u32) -> u32 { return o_q(n) + n * r; }
fn o_g(n: u32, r: u32) -> u32 { return o_q(n) + 2u * n * r; }
fn o_ag(n: u32, r: u32) -> u32 { return o_q(n) + 3u * n * r; }
// BiCGSTAB's solution and shadow residual (in scratch: fewer registers a thread)
fn o_bv(n: u32, r: u32) -> u32 { return o_q(n) + 4u * n * r; }
// the weak beams (n_w = sp.flags.z): 1 / d_w (0 in an empty slot), their hashes (bitcast, .x),
// and their rows of T G and T Q (column-major, n_w a column)
fn o_wi(n: u32, r: u32) -> u32 { return o_bv(n, r) + 2u * n; }
fn o_wh(n: u32, r: u32) -> u32 { return o_wi(n, r) + sp.flags.z; }
fn o_gw(n: u32, r: u32) -> u32 { return o_wh(n, r) + sp.flags.z; }
fn o_qw(n: u32, r: u32) -> u32 { return o_gw(n, r) + sp.flags.z * r; }
// header: [0] (q, q₁), [1] (√(2q), μ), [2] (kept columns, failed)

fn csqrt(z: vec2<f32>) -> vec2<f32> {
    let r = sqrt(sqrt(max(cabs2(z), 1.0e-30)));
    let h = 0.5 * atan2(z.y, z.x);
    return vec2<f32>(r * cos(h), r * sin(h));
}

// U(g_i − g_j)·iπλ for two beams by their hashes (zero off the table)
fn coupling_h(hi: i32, hj: i32) -> vec2<f32> {
    let t = hi - hj + sp.offset;
    if (t < 0 || u32(t) >= sp.table_size) {
        return vec2<f32>(0.0, 0.0);
    }
    let u = table[u32(t)];
    return vec2<f32>(-sp.phys.x * u.y, sp.phys.x * u.x);
}

fn beam_hash(b: u32, i: u32) -> i32 {
    return hkl[beams[b * (sp.n + sp.flags.z) + i]].w;
}

var<workgroup> red: array<vec2<f32>, WG>;
var<workgroup> s_scalar: vec4<f32>;
var<workgroup> s_uint: u32;
var<workgroup> s_flag: u32;

// Sum over the workgroup, the same value returned to every thread.
fn sum_c(lid: u32, v: vec2<f32>) -> vec2<f32> {
    workgroupBarrier();
    red[lid] = v;
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s = s >> 1u) {
        if (lid < s) {
            red[lid] = red[lid] + red[lid + s];
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        s_scalar = vec4<f32>(red[0], 0.0, 0.0);
    }
    return workgroupUniformLoad(&s_scalar).xy;
}

// ── the core matrices (sim_prep, sim_core) ──

var<workgroup> m_a: array<vec2<f32>, 930>;   // CORE * CS
var<workgroup> m_b: array<vec2<f32>, 930>;

// sum_c through m_a (sim_prep and sim_core reduce only while m_a is free)
fn sum_ma(lid: u32, v: vec2<f32>) -> vec2<f32> {
    workgroupBarrier();
    m_a[lid] = v;
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s = s >> 1u) {
        if (lid < s) {
            m_a[lid] = m_a[lid] + m_a[lid + s];
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        s_scalar = vec4<f32>(m_a[0], 0.0, 0.0);
    }
    return workgroupUniformLoad(&s_scalar).xy;
}
var<workgroup> c_idx: array<u32, CORE>;
var<workgroup> c_fac: array<vec2<f32>, CORE>;
var<workgroup> c_inv: vec2<f32>;
var<workgroup> c_piv: u32;

var<workgroup> k_key: array<f32, MAX_N>;

// The m least keys (k_key[i], i < n) by counting; writes each beam's core position
// (or −1) to scratch at `at` and the core's beams to c_idx.
fn pick_core(lid: u32, n: u32, m: u32, at: u32) {
    for (var i = lid; i < n; i = i + WG) {
        let k = k_key[i];
        var cnt = 0u;
        for (var j = 0u; j < n; j = j + 1u) {
            let kj = k_key[j];
            if (kj < k || (kj == k && j < i)) {
                cnt = cnt + 1u;
            }
        }
        if (cnt < m) {
            c_idx[cnt] = i;
            work[at + i] = vec2<f32>(f32(cnt), 0.0);
        } else {
            work[at + i] = vec2<f32>(-1.0, 0.0);
        }
    }
    workgroupBarrier();
}

// m_b ← m_a⁻¹ (m × m, Gauss–Jordan with partial pivoting; m_a is destroyed)
fn invert_core(lid: u32, m: u32) {
    for (var e = lid; e < m * m; e = e + WG) {
        let i = e / m;
        let c = e - i * m;
        m_b[i * CS + c] = select(vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), i == c);
    }
    workgroupBarrier();
    for (var k = 0u; k < m; k = k + 1u) {
        if (lid == 0u) {
            var p = k;
            var best = -1.0;
            for (var i = 0u; i < CORE; i = i + 1u) {
                let g = m_a[i * CS + k];
                let v = select(-1.0, cabs2(g), i >= k && i < m);
                if (v > best) {
                    best = v;
                    p = i;
                }
            }
            c_piv = p;
            c_inv = cdiv(vec2<f32>(1.0, 0.0), m_a[p * CS + k]);
        }
        workgroupBarrier();
        let p = c_piv;
        let inv = c_inv;
        for (var c = lid; c < m; c = c + WG) {
            let ak = m_a[k * CS + c];
            let bk = m_b[k * CS + c];
            m_a[k * CS + c] = cmul(m_a[p * CS + c], inv);
            m_b[k * CS + c] = cmul(m_b[p * CS + c], inv);
            if (p != k) {
                m_a[p * CS + c] = ak;
                m_b[p * CS + c] = bk;
            }
        }
        workgroupBarrier();
        for (var i = lid; i < m; i = i + WG) {
            c_fac[i] = select(m_a[i * CS + k], vec2<f32>(0.0, 0.0), i == k);
        }
        workgroupBarrier();
        for (var e = lid; e < m * m; e = e + WG) {
            let i = e / m;
            let c = e - i * m;
            if (i != k) {
                m_a[i * CS + c] = m_a[i * CS + c] - cmul(c_fac[i], m_a[k * CS + c]);
                m_b[i * CS + c] = m_b[i * CS + c] - cmul(c_fac[i], m_b[k * CS + c]);
            }
        }
        workgroupBarrier();
    }
}

@compute @workgroup_size(256)
fn sim_prep(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let b = flat_group(wid);
    if (b >= sp.n_k) {
        return;
    }
    let n = sp.n;
    let wb = b * sp.flags.y;
    let mu = sp.phys.z;
    let k = kvecs[b].xyz;
    // the diagonal (A's, without −μ/2)
    for (var i = lid; i < n; i = i + WG) {
        let g = beams[b * (n + sp.flags.z) + i];
        let h = hkl[g];
        var s = 0.0;
        if (g != 0u) {
            s = excitation(k, vec3<f32>(f32(h.x), f32(h.y), f32(h.z)));
        }
        work[wb + o_d(n) + i] = vec2<f32>(-sp.phys.x * sp.phys.y, 2.0 * PI * s);
    }
    // the weak beams: 1 / d_w = 1 / (2πi s_w), and their hashes
    let nw = sp.flags.z;
    for (var w = lid; w < nw; w = w + WG) {
        let g = beams[b * (n + nw) + n + w];
        var inv = vec2<f32>(0.0, 0.0);
        var hw = 0;
        if (g != NONE) {
            let h = hkl[g];
            hw = h.w;
            let s = excitation(k, vec3<f32>(f32(h.x), f32(h.y), f32(h.z)));
            if (abs(s) > 1.0e-7) {
                inv = vec2<f32>(0.0, -1.0 / (2.0 * PI * s));
            }
        }
        work[wb + o_wi(n, sp.rank) + w] = inv;
        work[wb + o_wh(n, sp.rank) + w] = vec2<f32>(bitcast<f32>(hw), 0.0);
    }
    storageBarrier();
    workgroupBarrier();
    // the pole: q = √(‖A‖_F / n · mean decay), ebsdsim's choice
    // (a thread's rows against every j, all threads at the same j: the j side is a broadcast)
    var part = vec2<f32>(0.0, 0.0);
    for (var i = lid; i < n; i = i + WG) {
        let d = work[wb + o_d(n) + i];
        part = part + vec2<f32>(cabs2(d), -d.x);
        let hi = beam_hash(b, i);
        var cs = -cabs2(coupling_h(hi, hi));
        for (var j = 0u; j < n; j = j + 1u) {
            cs = cs + cabs2(coupling_h(hi, beam_hash(b, j)));
        }
        part.x = part.x + cs;
    }
    let tot = sum_ma(lid, part);
    let nf = f32(n);
    let fro = sqrt(max(tot.x, 0.0)) / nf;
    let decay = tot.y / nf + 0.5 * mu;
    let q = sqrt(max(fro * max(decay, 1e-12), 1e-12));
    let q1 = q + 0.5 * mu;
    if (lid == 0u) {
        work[wb] = vec2<f32>(q, q1);
        work[wb + 1u] = vec2<f32>(sqrt(2.0 * q), mu);
        work[wb + 2u] = vec2<f32>(0.0, 0.0);
    }
    // the preconditioner's core: least |s| (000, first, always)
    let m = min(n, CORE);
    for (var i = lid; i < n; i = i + WG) {
        k_key[i] = select(abs(work[wb + o_d(n) + i].y), -1.0, i == 0u);
    }
    workgroupBarrier();
    pick_core(lid, n, m, wb + o_cp(n));
    for (var e = lid; e < m * m; e = e + WG) {
        let a = e / m;
        let c = e - a * m;
        let ia = c_idx[a];
        let ic = c_idx[c];
        var v: vec2<f32>;
        if (a == c) {
            v = vec2<f32>(q1, 0.0) - work[wb + o_d(n) + ia];
        } else {
            v = -coupling_h(beam_hash(b, ia), beam_hash(b, ic));
        }
        m_a[a * CS + c] = v;
    }
    workgroupBarrier();
    invert_core(lid, m);
    // K = M_c⁻¹ diag(1/scl_c), 1/scl = (q₁ − d)^(1/2)
    for (var e = lid; e < m * m; e = e + WG) {
        let a = e / m;
        let c = e - a * m;
        let ic = c_idx[c];
        work[wb + o_k(n) + a * CORE + c] = cmul(m_b[a * CS + c], csqrt(vec2<f32>(q1, 0.0) - work[wb + o_d(n) + ic]));
    }
}

// ── sim_solve ──

var<workgroup> s_hash: array<i32, MAX_N>;
var<workgroup> s_x: array<vec2<f32>, MAX_N>;   // the vector a product takes
var<workgroup> s_y: array<vec2<f32>, MAX_N>;   // the small system (H, T, P) once the columns are done
var<workgroup> s_stage: array<vec2<f32>, CORE>;
var<workgroup> s_wh: array<i32, MAX_W>;      // the weak beams' hashes

// The weak beams' part of a product: s_y[w] = (A_ws x)_w / d_w, x in s_x (s_y is free while
// the columns are made). All threads call it.
fn weak_pass(lid: u32, n: u32, nw: u32, wi: u32) {
    for (var w = lid; w < nw; w = w + WG) {
        let hw = s_wh[w];
        var a0 = vec2<f32>(0.0, 0.0);
        var a1 = vec2<f32>(0.0, 0.0);
        let n2 = n & ~1u;
        for (var j = 0u; j < n2; j = j + 2u) {
            a0 = a0 + cmul(coupling_h(hw, s_hash[j]), s_x[j]);
            a1 = a1 + cmul(coupling_h(hw, s_hash[j + 1u]), s_x[j + 1u]);
        }
        if (n2 < n) {
            a0 = a0 + cmul(coupling_h(hw, s_hash[n2]), s_x[n2]);
        }
        s_y[w] = cmul(work[wi + w], a0 + a1);
    }
    workgroupBarrier();
}

// y_i = (M x)_i = (q₁ − d_i) x_i − Σ_(j≠i) C_ij x_j + Σ_w C_iw s_y[w], x in s_x (row i of this
// thread): M = q₁ I − A_eff, A_eff = A + A_sw W
fn m_times(i: u32, n: u32, nw: u32, q1: f32, d: vec2<f32>) -> vec2<f32> {
    let hi = s_hash[i];
    var a0 = vec2<f32>(0.0, 0.0);
    var a1 = vec2<f32>(0.0, 0.0);
    let n2 = n & ~1u;
    for (var j = 0u; j < n2; j = j + 2u) {
        a0 = a0 + cmul(coupling_h(hi, s_hash[j]), s_x[j]);
        a1 = a1 + cmul(coupling_h(hi, s_hash[j + 1u]), s_x[j + 1u]);
    }
    if (n2 < n) {
        a0 = a0 + cmul(coupling_h(hi, s_hash[n2]), s_x[n2]);
    }
    // the j = i term was added with U(0): take it back
    let self_c = coupling_h(hi, hi);
    let acc = a0 + a1 - cmul(self_c, s_x[i]);
    var wk = vec2<f32>(0.0, 0.0);
    for (var w = 0u; w < nw; w = w + 1u) {
        wk = wk + cmul(coupling_h(hi, s_wh[w]), s_y[w]);
    }
    return cmul(vec2<f32>(q1, 0.0) - d, s_x[i]) - acc + wk;
}

// s_x ← P v (the right preconditioner) from each thread's rows v[t]
fn precondition(lid: u32, n: u32, wb: u32, v: array<vec2<f32>, BPT>, scl: array<vec2<f32>, BPT>, cpt: array<i32, BPT>) {
    workgroupBarrier();
    for (var t = 0u; t < BPT; t = t + 1u) {
        let i = lid + t * WG;
        if (i < n) {
            s_x[i] = cmul(scl[t], v[t]);
            if (cpt[t] >= 0) {
                s_stage[u32(cpt[t])] = v[t];
            }
        }
    }
    workgroupBarrier();
    let m = min(n, CORE);
    for (var t = 0u; t < BPT; t = t + 1u) {
        let i = lid + t * WG;
        if (i < n && cpt[t] >= 0) {
            let kb = wb + o_k(n) + u32(cpt[t]) * CORE;
            var acc = vec2<f32>(0.0, 0.0);
            for (var c = 0u; c < m; c = c + 1u) {
                acc = acc + cmul(work[kb + c], s_stage[c]);
            }
            s_x[i] = acc;
        }
    }
    workgroupBarrier();
}

@compute @workgroup_size(256)
fn sim_solve(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let b = flat_group(wid);
    if (b >= sp.n_k) {
        return;
    }
    let n = sp.n;
    let r = sp.rank;
    let wb = b * sp.flags.y;
    let hq = work[wb];
    let q = hq.x;
    let q1 = hq.y;
    let scale = work[wb + 1u].x;
    let mu = work[wb + 1u].y;
    if (lid == 0u) {
        s_flag = 0u;
    }
    for (var i = lid; i < n; i = i + WG) {
        s_hash[i] = beam_hash(b, i);
    }
    let nw = sp.flags.z;
    let wi = wb + o_wi(n, r);
    for (var w = lid; w < nw; w = w + WG) {
        s_wh[w] = bitcast<i32>(work[wb + o_wh(n, r) + w].x);
    }
    var dd: array<vec2<f32>, BPT>;
    var scl: array<vec2<f32>, BPT>;
    var cpt: array<i32, BPT>;
    var wd: array<vec2<f32>, BPT>;
    for (var t = 0u; t < BPT; t = t + 1u) {
        let i = lid + t * WG;
        dd[t] = vec2<f32>(0.0, 0.0);
        scl[t] = vec2<f32>(0.0, 0.0);
        cpt[t] = -1;
        wd[t] = vec2<f32>(0.0, 0.0);
        if (i < n) {
            dd[t] = work[wb + o_d(n) + i];
            let diag = vec2<f32>(q1, 0.0) - dd[t];
            scl[t] = cdiv(vec2<f32>(1.0, 0.0), csqrt(diag));
            cpt[t] = i32(work[wb + o_cp(n) + i].x);
            wd[t] = select(vec2<f32>(0.0, 0.0), vec2<f32>(scale, 0.0), i == 0u);
        }
    }
    workgroupBarrier();

    // the Cayley columns: (qI − Ã) z = w, w ← 2q z − w
    for (var col = 0u; col < r; col = col + 1u) {
        // y (the solution in the preconditioned variable) and r̂ live in scratch
        let yb = wb + o_bv(n, r);
        let hb = yb + n;
        var rr: array<vec2<f32>, BPT>;
        var pv: array<vec2<f32>, BPT>;
        var vv: array<vec2<f32>, BPT>;
        var ss: array<vec2<f32>, BPT>;
        var r0 = 0.0;
        for (var t = 0u; t < BPT; t = t + 1u) {
            rr[t] = cmul(scl[t], wd[t]);
            let i = lid + t * WG;
            if (i < n) {
                work[yb + i] = vec2<f32>(0.0, 0.0);
                work[hb + i] = rr[t];
            }
            pv[t] = vec2<f32>(0.0, 0.0);
            vv[t] = vec2<f32>(0.0, 0.0);
            ss[t] = vec2<f32>(0.0, 0.0);
            r0 = r0 + cabs2(rr[t]);
        }
        var rho_o = vec2<f32>(1.0, 0.0);
        var alpha = vec2<f32>(1.0, 0.0);
        var omega = vec2<f32>(1.0, 0.0);
        if (sqrt(sum_c(lid, vec2<f32>(r0, 0.0)).x) > BICG_TOL) {
            for (var it = 0u; it < BICG_MAX; it = it + 1u) {
                var lr = vec2<f32>(0.0, 0.0);
                for (var t = 0u; t < BPT; t = t + 1u) {
                    let i = lid + t * WG;
                    if (i < n) {
                        lr = lr + cmul(cconj(work[hb + i]), rr[t]);
                    }
                }
                let rho = sum_c(lid, lr);
                if (cabs2(rho) < 1.0e-30) {
                    break;
                }
                let beta = cmul(cdiv(rho, rho_o), cdiv(alpha, omega));
                for (var t = 0u; t < BPT; t = t + 1u) {
                    pv[t] = rr[t] + cmul(beta, pv[t] - cmul(omega, vv[t]));
                }
                precondition(lid, n, wb, pv, scl, cpt);
                if (nw > 0u) {
                    weak_pass(lid, n, nw, wi);
                }
                var lv = vec2<f32>(0.0, 0.0);
                for (var t = 0u; t < BPT; t = t + 1u) {
                    let i = lid + t * WG;
                    vv[t] = vec2<f32>(0.0, 0.0);
                    if (i < n) {
                        vv[t] = cmul(scl[t], m_times(i, n, nw, q1, dd[t]));
                        lv = lv + cmul(cconj(work[hb + i]), vv[t]);
                    }
                }
                let rv = sum_c(lid, lv);
                if (cabs2(rv) < 1.0e-30) {
                    break;
                }
                alpha = cdiv(rho, rv);
                var ls = 0.0;
                for (var t = 0u; t < BPT; t = t + 1u) {
                    let i = lid + t * WG;
                    if (i < n) {
                        work[yb + i] = work[yb + i] + cmul(alpha, pv[t]);
                    }
                    ss[t] = rr[t] - cmul(alpha, vv[t]);
                    ls = ls + cabs2(ss[t]);
                }
                let sn = sqrt(sum_c(lid, vec2<f32>(ls, 0.0)).x);
                if (sn != sn || sn > 1.0e20) {
                    if (lid == 0u) {
                        s_flag = 1u;
                    }
                    break;
                }
                if (sn <= BICG_TOL) {
                    break;
                }
                precondition(lid, n, wb, ss, scl, cpt);
                if (nw > 0u) {
                    weak_pass(lid, n, nw, wi);
                }
                var tt = vec2<f32>(0.0, 0.0);
                var ts = vec2<f32>(0.0, 0.0);
                var td: array<vec2<f32>, BPT>;
                for (var t = 0u; t < BPT; t = t + 1u) {
                    let i = lid + t * WG;
                    td[t] = vec2<f32>(0.0, 0.0);
                    if (i < n) {
                        td[t] = cmul(scl[t], m_times(i, n, nw, q1, dd[t]));
                    }
                    tt = tt + cmul(cconj(td[t]), td[t]);
                    ts = ts + cmul(cconj(td[t]), ss[t]);
                }
                let ttv = sum_c(lid, tt);
                let tsv = sum_c(lid, ts);
                if (cabs2(ttv) < 1.0e-30) {
                    break;
                }
                omega = cdiv(tsv, ttv);
                var lr2 = 0.0;
                for (var t = 0u; t < BPT; t = t + 1u) {
                    let i = lid + t * WG;
                    if (i < n) {
                        work[yb + i] = work[yb + i] + cmul(omega, ss[t]);
                    }
                    rr[t] = ss[t] - cmul(omega, td[t]);
                    lr2 = lr2 + cabs2(rr[t]);
                }
                let rn = sqrt(sum_c(lid, vec2<f32>(lr2, 0.0)).x);
                if (rn != rn || rn > 1.0e20) {
                    if (lid == 0u) {
                        s_flag = 1u;
                    }
                    break;
                }
                if (rn <= BICG_TOL || cabs2(omega) < 1.0e-30) {
                    break;
                }
                rho_o = rho;
            }
        }
        // z = P y
        var y: array<vec2<f32>, BPT>;
        for (var t = 0u; t < BPT; t = t + 1u) {
            let i = lid + t * WG;
            y[t] = select(vec2<f32>(0.0, 0.0), work[yb + min(i, n - 1u)], i < n);
        }
        precondition(lid, n, wb, y, scl, cpt);
        for (var t = 0u; t < BPT; t = t + 1u) {
            let i = lid + t * WG;
            if (i < n) {
                var z = s_x[i];
                if (z.x != z.x || z.y != z.y || abs(z.x) > 1e3 || abs(z.y) > 1e3) {
                    z = vec2<f32>(0.0, 0.0);
                    s_flag = 1u;
                }
                work[wb + o_q(n) + col * n + i] = z;
                wd[t] = 2.0 * q * z - wd[t];
            }
        }
    }
    storageBarrier();
    workgroupBarrier();

    // Gram–Schmidt twice over; a column with less than 1e-4 of its norm left is dropped
    let qb = wb + o_q(n);
    var kept = 0u;
    for (var j = 0u; j < r; j = j + 1u) {
        var qc: array<vec2<f32>, BPT>;
        var n0 = 0.0;
        for (var t = 0u; t < BPT; t = t + 1u) {
            let i = lid + t * WG;
            qc[t] = vec2<f32>(0.0, 0.0);
            if (i < n) {
                qc[t] = work[qb + j * n + i];
            }
            n0 = n0 + cabs2(qc[t]);
        }
        let nrm0 = sqrt(sum_c(lid, vec2<f32>(n0, 0.0)).x);
        for (var sweep = 0u; sweep < 2u; sweep = sweep + 1u) {
            for (var c = 0u; c < kept; c = c + 1u) {
                var acc = vec2<f32>(0.0, 0.0);
                for (var t = 0u; t < BPT; t = t + 1u) {
                    let i = lid + t * WG;
                    if (i < n) {
                        acc = acc + cmul(cconj(work[qb + c * n + i]), qc[t]);
                    }
                }
                let d = sum_c(lid, acc);
                for (var t = 0u; t < BPT; t = t + 1u) {
                    let i = lid + t * WG;
                    if (i < n) {
                        qc[t] = qc[t] - cmul(d, work[qb + c * n + i]);
                    }
                }
            }
        }
        var nl = 0.0;
        for (var t = 0u; t < BPT; t = t + 1u) {
            nl = nl + cabs2(qc[t]);
        }
        let nrm = sqrt(sum_c(lid, vec2<f32>(nl, 0.0)).x);
        if (nrm > 1e-4 * nrm0 && nrm > 0.0) {
            for (var t = 0u; t < BPT; t = t + 1u) {
                let i = lid + t * WG;
                if (i < n) {
                    work[qb + kept * n + i] = qc[t] / nrm;
                }
            }
            kept = kept + 1u;
        }
        storageBarrier();
    }
    let m = kept;
    let mm = m * m;

    // Ã Q
    let aqb = wb + o_aq(n, r);
    for (var j = 0u; j < m; j = j + 1u) {
        workgroupBarrier();
        for (var t = 0u; t < BPT; t = t + 1u) {
            let i = lid + t * WG;
            if (i < n) {
                s_x[i] = work[qb + j * n + i];
            }
        }
        workgroupBarrier();
        if (nw > 0u) {
            weak_pass(lid, n, nw, wi);
        }
        for (var t = 0u; t < BPT; t = t + 1u) {
            let i = lid + t * WG;
            if (i < n) {
                // M = q₁ I − A, so Ã x = A x − (μ/2) x = q₁ x − M x − (μ/2) x = q x − M x
                work[aqb + j * n + i] = q * s_x[i] - m_times(i, n, nw, q1, dd[t]);
            }
        }
    }
    storageBarrier();
    workgroupBarrier();
    // H = Qᴴ Ã Q (a thread an entry) and b = Qᴴ e₀ (the incident beam is beam 0)
    var hij = vec2<f32>(0.0, 0.0);
    if (lid < mm) {
        let i = lid / m;
        let j = lid % m;
        for (var l = 0u; l < n; l = l + 1u) {
            hij = hij + cmul(cconj(work[qb + i * n + l]), work[aqb + j * n + l]);
        }
    }
    var bi = vec2<f32>(0.0, 0.0);
    if (lid < m) {
        bi = cconj(work[qb + lid * n]);
    }
    workgroupBarrier();
    // the small system (m ≤ 16): H then P at s_y[0 …], T at s_y[256 …]; S at s_x[0 …], b at
    // s_x[256 …], B at s_x[272 …] (s_x is done with once Ã Q is)
    if (lid < mm) {
        s_y[lid] = hij;
    }
    if (lid < m) {
        s_x[256u + lid] = bi;
    }
    workgroupBarrier();

    // H Y + Y Hᴴ + b bᴴ = 0 by squared Smith: C = (pI − H)⁻¹, T = 2p C − I,
    // Y = Σ_k T^k B Bᴴ T^kᴴ, B = √(2p) C b; S ← S + T S Tᴴ, T ← T²
    var hpart = vec2<f32>(0.0, 0.0);
    if (lid < mm) {
        hpart = vec2<f32>(cabs2(s_y[lid]), select(0.0, -s_y[lid].x, lid / m == lid % m));
    }
    let ht = sum_c(lid, hpart);
    let mf = max(f32(m), 1.0);
    let p = sqrt(max(sqrt(max(ht.x, 0.0)) / mf * max(ht.y / mf, 1e-12), 1e-12));
    let ab = wb + o_sm(n);
    let w2 = 2u * m;
    for (var e = lid; e < m * w2; e = e + WG) {
        let i = e / w2;
        let j = e % w2;
        var v = vec2<f32>(0.0, 0.0);
        if (j < m) {
            v = -s_y[i * m + j];
            if (i == j) {
                v = v + vec2<f32>(p, 0.0);
            }
        } else if (j - m == i) {
            v = vec2<f32>(1.0, 0.0);
        }
        work[ab + e] = v;
    }
    storageBarrier();
    workgroupBarrier();
    for (var c = 0u; c < m; c = c + 1u) {
        if (lid == 0u) {
            var at = c;
            var best = -1.0;
            for (var i = c; i < m; i = i + 1u) {
                let a = cabs2(work[ab + i * w2 + c]);
                if (a > best) {
                    best = a;
                    at = i;
                }
            }
            if (at != c) {
                for (var j = 0u; j < w2; j = j + 1u) {
                    let a = work[ab + c * w2 + j];
                    work[ab + c * w2 + j] = work[ab + at * w2 + j];
                    work[ab + at * w2 + j] = a;
                }
            }
            if (best < 1e-36) {
                s_flag = 1u;
            }
            let inv = cdiv(vec2<f32>(1.0, 0.0), work[ab + c * w2 + c]);
            for (var j = 0u; j < w2; j = j + 1u) {
                work[ab + c * w2 + j] = cmul(work[ab + c * w2 + j], inv);
            }
        }
        storageBarrier();
        workgroupBarrier();
        var f = vec2<f32>(0.0, 0.0);
        if (lid < m && lid != c) {
            f = work[ab + lid * w2 + c];
        }
        storageBarrier();
        workgroupBarrier();
        if (lid < m && lid != c) {
            for (var j = 0u; j < w2; j = j + 1u) {
                work[ab + lid * w2 + j] = work[ab + lid * w2 + j] - cmul(f, work[ab + c * w2 + j]);
            }
        }
        storageBarrier();
        workgroupBarrier();
    }
    var tij = vec2<f32>(0.0, 0.0);
    if (lid < mm) {
        let i = lid / m;
        let j = lid % m;
        tij = 2.0 * p * work[ab + i * w2 + m + j];
        if (i == j) {
            tij = tij - vec2<f32>(1.0, 0.0);
        }
    }
    var bb = vec2<f32>(0.0, 0.0);
    if (lid < m) {
        for (var j = 0u; j < m; j = j + 1u) {
            bb = bb + cmul(work[ab + lid * w2 + m + j], s_x[256u + j]);
        }
        bb = sqrt(2.0 * p) * bb;
    }
    workgroupBarrier();
    if (lid < mm) {
        s_y[256u + lid] = tij;
    }
    if (lid < m) {
        s_x[272u + lid] = bb;
    }
    workgroupBarrier();
    if (lid < mm) {
        s_x[lid] = cmul(s_x[272u + lid / m], cconj(s_x[272u + lid % m]));
    }
    workgroupBarrier();
    for (var it = 0u; it < 48u; it = it + 1u) {
        var pij = vec2<f32>(0.0, 0.0);
        if (lid < mm) {
            let i = lid / m;
            let j = lid % m;
            for (var l = 0u; l < m; l = l + 1u) {
                pij = pij + cmul(s_y[256u + i * m + l], s_x[l * m + j]);
            }
        }
        workgroupBarrier();
        if (lid < mm) {
            s_y[lid] = pij;
        }
        workgroupBarrier();
        var add = vec2<f32>(0.0, 0.0);
        var t2 = vec2<f32>(0.0, 0.0);
        if (lid < mm) {
            let i = lid / m;
            let j = lid % m;
            for (var l = 0u; l < m; l = l + 1u) {
                add = add + cmul(s_y[i * m + l], cconj(s_y[256u + j * m + l]));
                t2 = t2 + cmul(s_y[256u + i * m + l], s_y[256u + l * m + j]);
            }
        }
        workgroupBarrier();
        var tn = 0.0;
        if (lid < mm) {
            s_x[lid] = s_x[lid] + add;
            s_y[256u + lid] = t2;
            tn = cabs2(t2);
        }
        let size = sum_c(lid, vec2<f32>(tn, 0.0)).x;
        if (size != size || size > 1e12) {
            if (lid == 0u) {
                s_flag = 1u;
            }
            break;
        }
        if (size < 1e-14) {
            break;
        }
    }
    workgroupBarrier();

    // G = Q Y and AG = (Ã Q) Y (n × m, Y in s_x[0 …])
    let gb = wb + o_g(n, r);
    let agb = wb + o_ag(n, r);
    for (var t = 0u; t < BPT; t = t + 1u) {
        let i = lid + t * WG;
        if (i < n) {
            for (var j = 0u; j < m; j = j + 1u) {
                var g = vec2<f32>(0.0, 0.0);
                var ag = vec2<f32>(0.0, 0.0);
                for (var l = 0u; l < m; l = l + 1u) {
                    let y = s_x[l * m + j];
                    g = g + cmul(work[qb + l * n + i], y);
                    ag = ag + cmul(work[aqb + l * n + i], y);
                }
                work[gb + j * n + i] = g;
                work[agb + j * n + i] = ag;
            }
        }
    }
    if (nw > 0u) {
        storageBarrier();
        workgroupBarrier();
        // the weak beams' rows: (W G)_w = −(1 / d_w) Σ_s A_ws G_s, and of Q likewise
        let gwb = wb + o_gw(n, r);
        let qwb = wb + o_qw(n, r);
        for (var w = lid; w < nw; w = w + WG) {
            let hw = s_wh[w];
            let inv = -work[wi + w];
            for (var c = 0u; c < m; c = c + 1u) {
                var ga = vec2<f32>(0.0, 0.0);
                var qa = vec2<f32>(0.0, 0.0);
                for (var s = 0u; s < n; s = s + 1u) {
                    let a = coupling_h(hw, s_hash[s]);
                    ga = ga + cmul(a, work[gb + c * n + s]);
                    qa = qa + cmul(a, work[qb + c * n + s]);
                }
                work[gwb + c * nw + w] = cmul(inv, ga);
                work[qwb + c * nw + w] = cmul(inv, qa);
            }
        }
        // A_eff's diagonal, d_i − Σ_w A_iw A_wi / d_w, for the correction (the products are done)
        for (var t = 0u; t < BPT; t = t + 1u) {
            let i = lid + t * WG;
            if (i < n) {
                let hi = s_hash[i];
                var acc = vec2<f32>(0.0, 0.0);
                for (var w = 0u; w < nw; w = w + 1u) {
                    let hw = s_wh[w];
                    acc = acc + cmul(cmul(coupling_h(hi, hw), work[wi + w]), coupling_h(hw, hi));
                }
                work[wb + o_d(n) + i] = dd[t] - acc;
            }
        }
    }
    let failed = workgroupUniformLoad(&s_flag);
    if (lid == 0u) {
        work[wb + 2u] = vec2<f32>(f32(m), f32(failed));
    }
}

// ── sim_core ──

// R_ij = Σ_c v_ic conj(u_jc), v = [AG, G], u = [Q, AQ] (2m columns)
fn core_u(wb: u32, n: u32, r: u32, j: u32, c: u32, m: u32) -> vec2<f32> {
    if (c < m) {
        return work[wb + o_q(n) + c * n + j];
    }
    return work[wb + o_aq(n, r) + (c - m) * n + j];
}
fn core_v(wb: u32, n: u32, r: u32, i: u32, c: u32, m: u32) -> vec2<f32> {
    if (c < m) {
        return work[wb + o_ag(n, r) + c * n + i];
    }
    return work[wb + o_g(n, r) + (c - m) * n + i];
}

@compute @workgroup_size(256)
fn sim_core(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let b = flat_group(wid);
    if (b >= sp.n_k) {
        return;
    }
    let n = sp.n;
    let r = sp.rank;
    let wb = b * sp.flags.y;
    let mu = sp.phys.z;
    // (the header through workgroup memory: what it says steers loops that hold barriers)
    if (lid == 0u) {
        s_scalar = vec4<f32>(work[wb + 2u], 0.0, 0.0);
    }
    let hdr = workgroupUniformLoad(&s_scalar).xy;
    let m = u32(hdr.x);
    let rr = 2u * m;
    let mc = min(n, CORE);
    // the residual's row norms ‖R_i‖² = Σ_kl v_ik conj(v_il) Gram_kl, Gram_kl = Σ_j conj(u_jk) u_jl
    let gr = wb + o_gr(n);
    for (var kl = 0u; kl < rr * rr; kl = kl + 1u) {
        let kk = kl / rr;
        let ll = kl - kk * rr;
        var part = vec2<f32>(0.0, 0.0);
        for (var j = lid; j < n; j = j + WG) {
            part = part + cmul(cconj(core_u(wb, n, r, j, kk, m)), core_u(wb, n, r, j, ll, m));
        }
        let s = sum_ma(lid, part);
        if (lid == 0u) {
            work[gr + kl] = s;
        }
    }
    storageBarrier();
    workgroupBarrier();
    for (var i = lid; i < n; i = i + WG) {
        var s2 = 0.0;
        for (var k = 0u; k < rr; k = k + 1u) {
            let vk = core_v(wb, n, r, i, k, m);
            for (var l = 0u; l < rr; l = l + 1u) {
                s2 = s2 + cmul(cmul(vk, cconj(core_v(wb, n, r, i, l, m))), work[gr + k * rr + l]).x;
            }
        }
        let di = work[wb + o_d(n) + i] - vec2<f32>(0.5 * mu, 0.0);
        let key = -sqrt(max(s2, 0.0)) / max(sqrt(cabs2(di)), 1e-20);
        k_key[i] = select(key, -3.0e38, i == 0u);
    }
    workgroupBarrier();
    pick_core(lid, n, mc, wb + o_c2(n));
    // Ã_c and R_c (R_c to scratch, to be X̃'s start); qc = ‖Ã_c‖_F / √m
    var fl = 0.0;
    for (var e = lid; e < mc * mc; e = e + WG) {
        let a = e / mc;
        let c = e - a * mc;
        let ia = c_idx[a];
        let ic = c_idx[c];
        var v: vec2<f32>;
        if (a == c) {
            v = work[wb + o_d(n) + ia] - vec2<f32>(0.5 * mu, 0.0);
        } else {
            let ha = beam_hash(b, ia);
            let hc = beam_hash(b, ic);
            v = coupling_h(ha, hc);
            // A_eff = A − A_sw diag(1 / d_w) A_ws on the core
            for (var w = 0u; w < sp.flags.z; w = w + 1u) {
                let hw = bitcast<i32>(work[wb + o_wh(n, r) + w].x);
                v = v - cmul(cmul(coupling_h(ha, hw), work[wb + o_wi(n, r) + w]), coupling_h(hw, hc));
            }
        }
        m_b[a * CS + c] = v;
        fl = fl + cabs2(v);
        var rv = vec2<f32>(0.0, 0.0);
        for (var k = 0u; k < rr; k = k + 1u) {
            rv = rv + cmul(core_v(wb, n, r, ia, k, m), cconj(core_u(wb, n, r, ic, k, m)));
        }
        if (ia == 0u && ic == 0u) {
            rv = rv + vec2<f32>(1.0, 0.0);
        }
        work[wb + o_xs(n) + a * CORE + c] = rv;
    }
    let qc = sqrt(sum_ma(lid, vec2<f32>(fl, 0.0)).x / f32(mc));
    // m_a = qc I − Ã_c, inverted: m_b = C = (qc − Ã_c)⁻¹
    for (var e = lid; e < mc * mc; e = e + WG) {
        let a = e / mc;
        let c = e - a * mc;
        m_a[a * CS + c] = select(vec2<f32>(0.0, 0.0), vec2<f32>(qc, 0.0), a == c) - m_b[a * CS + c];
    }
    storageBarrier();
    workgroupBarrier();
    invert_core(lid, mc);
    // m_a = C R; X̃ = 2qc (C R) Cᴴ into scratch; m_b = T = 2qc C − I
    let xt = wb + o_xt(n);
    let xs = wb + o_xs(n);
    for (var e = lid; e < mc * mc; e = e + WG) {
        let i = e / mc;
        let c = e - i * mc;
        var s = vec2<f32>(0.0, 0.0);
        for (var l = 0u; l < mc; l = l + 1u) {
            s = s + cmul(m_b[i * CS + l], work[xs + l * CORE + c]);
        }
        m_a[i * CS + c] = s;
    }
    workgroupBarrier();
    for (var e = lid; e < mc * mc; e = e + WG) {
        let i = e / mc;
        let c = e - i * mc;
        var s = vec2<f32>(0.0, 0.0);
        for (var l = 0u; l < mc; l = l + 1u) {
            s = s + cmul(m_a[i * CS + l], cconj(m_b[c * CS + l]));
        }
        work[xt + i * CORE + c] = 2.0 * qc * s;
    }
    storageBarrier();
    workgroupBarrier();
    for (var e = lid; e < mc * mc; e = e + WG) {
        let i = e / mc;
        let c = e - i * mc;
        m_b[i * CS + c] = 2.0 * qc * m_b[i * CS + c] - select(vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), i == c);
    }
    workgroupBarrier();
    // doublings: P = T X̃ (m_a), X̃ += P Tᴴ, T ← T² (via m_a)
    for (var it = 0u; it < DOUBLINGS; it = it + 1u) {
        for (var e = lid; e < mc * mc; e = e + WG) {
            let i = e / mc;
            let c = e - i * mc;
            var s = vec2<f32>(0.0, 0.0);
            for (var l = 0u; l < mc; l = l + 1u) {
                s = s + cmul(m_b[i * CS + l], work[xt + l * CORE + c]);
            }
            m_a[i * CS + c] = s;
        }
        storageBarrier();
        workgroupBarrier();
        for (var e = lid; e < mc * mc; e = e + WG) {
            let i = e / mc;
            let c = e - i * mc;
            var s = vec2<f32>(0.0, 0.0);
            for (var l = 0u; l < mc; l = l + 1u) {
                s = s + cmul(m_a[i * CS + l], cconj(m_b[c * CS + l]));
            }
            work[xt + i * CORE + c] = work[xt + i * CORE + c] + s;
        }
        storageBarrier();
        workgroupBarrier();
        if (it + 1u < DOUBLINGS) {
            for (var e = lid; e < mc * mc; e = e + WG) {
                let i = e / mc;
                let c = e - i * mc;
                var s = vec2<f32>(0.0, 0.0);
                for (var l = 0u; l < mc; l = l + 1u) {
                    s = s + cmul(m_b[i * CS + l], m_b[l * CS + c]);
                }
                m_a[i * CS + c] = s;
            }
            workgroupBarrier();
            for (var e = lid; e < mc * mc; e = e + WG) {
                let i = e / mc;
                let c = e - i * mc;
                m_b[i * CS + c] = m_a[i * CS + c];
            }
            workgroupBarrier();
        }
    }
}

// ── sim_intensity ──

var<workgroup> i_hash: array<i32, 2u * MAX_N>;  // the strong beams, then the weak
var<workgroup> i_d: array<vec2<f32>, MAX_N>;
var<workgroup> i_cp: array<i32, MAX_N>;
var<workgroup> i_red: array<f32, WG>;

// Row i of a factor (column c): a strong beam's from `strong_at` (n a column), a weak beam's
// from `weak_at` (n_w a column).
fn row_at(strong_at: u32, weak_at: u32, n: u32, nw: u32, c: u32, i: u32) -> vec2<f32> {
    if (i < n) {
        return work[strong_at + c * n + i];
    }
    return work[weak_at + c * nw + i - n];
}

@compute @workgroup_size(256)
fn sim_intensity(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let b = flat_group(wid);
    if (b >= sp.n_k) {
        return;
    }
    let n = sp.n;
    let r = sp.rank;
    let wb = b * sp.flags.y;
    let mu = sp.phys.z;
    // (the header through workgroup memory: what it says steers loops that hold barriers)
    if (lid == 0u) {
        s_scalar = vec4<f32>(work[wb + 2u], 0.0, 0.0);
    }
    let hdr = workgroupUniformLoad(&s_scalar).xy;
    let m = u32(hdr.x);
    let failed = hdr.y > 0.5;
    let correct = sp.flags.x != 0u;
    let qb = wb + o_q(n);
    let aqb = wb + o_aq(n, r);
    let gb = wb + o_g(n, r);
    let agb = wb + o_ag(n, r);
    let xt = wb + o_xt(n);
    let nw = sp.flags.z;
    let gwb = wb + o_gw(n, r);
    let qwb = wb + o_qw(n, r);
    for (var i = lid; i < n; i = i + WG) {
        i_hash[i] = beam_hash(b, i);
        i_d[i] = work[wb + o_d(n) + i] - vec2<f32>(0.5 * mu, 0.0);
        i_cp[i] = select(-1, i32(work[wb + o_c2(n) + i].x), correct);
    }
    for (var w = lid; w < nw; w = w + WG) {
        i_hash[n + w] = bitcast<i32>(work[wb + o_wh(n, r) + w].x);
    }
    workgroupBarrier();
    var acc: array<f32, MAX_SITES>;
    for (var s = 0u; s < MAX_SITES; s = s + 1u) {
        acc[s] = 0.0;
    }
    let ns = sp.n_sites;
    // a thread takes rows i and N − 1 − i, each against the j ≥ i (balanced triangles), over
    // the N = n + n_w strong and weak beams; the weak beams' rows of X are T G (T Q)ᴴ's, and
    // the correction is the strong beams' alone
    let nn = n + nw;
    let half = (nn + 1u) / 2u;
    for (var row = lid; row < half; row = row + WG) {
        for (var side = 0u; side < 2u; side = side + 1u) {
            let i = select(row, nn - 1u - row, side == 1u);
            if (side == 1u && i == row) {
                break;
            }
            let hi = i_hash[i];
            let strong_i = i < n;
            let di = cconj(i_d[min(i, n - 1u)]);
            let ci = select(-1, i_cp[min(i, n - 1u)], strong_i);
            for (var j = i; j < nn; j = j + 1u) {
                // X_ji = Σ_c G_jc conj(Q_ic);  R_ji = Σ_c AG_jc conj(Q_ic) + G_jc conj(AQ_ic)
                let both = j < n;
                var x = vec2<f32>(0.0, 0.0);
                var rv = vec2<f32>(0.0, 0.0);
                for (var c = 0u; c < m; c = c + 1u) {
                    let qi = cconj(row_at(qb, qwb, n, nw, c, i));
                    let gj = row_at(gb, gwb, n, nw, c, j);
                    x = x + cmul(gj, qi);
                    if (correct && both) {
                        rv = rv + cmul(work[agb + c * n + j], qi) + cmul(gj, cconj(work[aqb + c * n + i]));
                    }
                }
                var v = x;
                if (correct && both) {
                    if (i == 0u && j == 0u) {
                        rv = rv + vec2<f32>(1.0, 0.0);
                    }
                    let cj = i_cp[j];
                    if (ci >= 0 && cj >= 0) {
                        v = x + work[xt + u32(cj) * CORE + u32(ci)];
                    } else {
                        v = x - cdiv(rv, di + i_d[j]);
                    }
                }
                let t = i_hash[j] - hi + sp.offset;
                if (t >= 0 && u32(t) < sp.table_size) {
                    let wgt = select(2.0, 1.0, j == i);
                    for (var s = 0u; s < ns; s = s + 1u) {
                        let sh = sgh[u32(t) * ns + s];
                        acc[s] = acc[s] + wgt * (sh.x * v.x - sh.y * v.y);
                    }
                }
            }
        }
    }
    for (var s = 0u; s < ns; s = s + 1u) {
        workgroupBarrier();
        i_red[lid] = acc[s];
        workgroupBarrier();
        for (var st = WG / 2u; st > 0u; st = st >> 1u) {
            if (lid < st) {
                i_red[lid] = i_red[lid] + i_red[lid + st];
            }
            workgroupBarrier();
        }
        if (lid == 0u) {
            let v = i_red[0];
            intensity[b * ns + s] = select(sp.phys.w * v, 0.0, failed || v != v);
        }
    }
    if (lid == 0u && failed) {
        atomicAdd(&counter[1], 1u);
    }
}
