//! Rotations and rigid motions in f64 (host side).
//!
//! Conventions, the same throughout the engine and both front ends:
//!  * A quaternion is (w, x, y, z), unit, and acts on a vector as q v q*.
//!  * An orientation q maps sample-frame directions to crystal-frame directions:
//!    v_crystal = q v_sample q*. Crystal symmetry therefore multiplies on the left (g · q).
//!  * Bunge angles (phi1, Phi, phi2) are radians, ZXZ, passive: `fromBunge`.
//!  * A rigid motion is [t(3), q(4)]: x' = q x q* + t. Tangent vectors are [tau(3), phi(3)].
const std = @import("std");

pub const Quat = [4]f64;
pub const Vec3 = [3]f64;
/// row-major 3×3
pub const Mat3 = [9]f64;
/// [t, q]
pub const Se3 = [7]f64;

pub const quat_identity: Quat = .{ 1, 0, 0, 0 };
pub const se3_identity: Se3 = .{ 0, 0, 0, 1, 0, 0, 0 };

pub fn dot(a: Vec3, b: Vec3) f64 {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

pub fn cross(a: Vec3, b: Vec3) Vec3 {
    return .{ a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0] };
}

pub fn add(a: Vec3, b: Vec3) Vec3 {
    return .{ a[0] + b[0], a[1] + b[1], a[2] + b[2] };
}

pub fn sub(a: Vec3, b: Vec3) Vec3 {
    return .{ a[0] - b[0], a[1] - b[1], a[2] - b[2] };
}

pub fn scale(a: Vec3, s: f64) Vec3 {
    return .{ a[0] * s, a[1] * s, a[2] * s };
}

pub fn norm(a: Vec3) f64 {
    return @sqrt(dot(a, a));
}

pub fn normalize(a: Vec3) Vec3 {
    const n = norm(a);
    return if (n > 0) scale(a, 1.0 / n) else a;
}

pub fn qmul(a: Quat, b: Quat) Quat {
    return .{
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    };
}

pub fn qconj(q: Quat) Quat {
    return .{ q[0], -q[1], -q[2], -q[3] };
}

/// Unit norm with a non-negative real part.
pub fn qstd(q: Quat) Quat {
    const n = @sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    const s: f64 = if (n == 0) 1 else (if (q[0] < 0) -1.0 / n else 1.0 / n);
    return .{ q[0] * s, q[1] * s, q[2] * s, q[3] * s };
}

/// q v q*
pub fn qapply(q: Quat, v: Vec3) Vec3 {
    const u: Vec3 = .{ q[1], q[2], q[3] };
    const t = scale(cross(u, v), 2);
    return add(add(v, scale(t, q[0])), cross(u, t));
}

/// The matrix of v ↦ q v q*.
pub fn toMat(q: Quat) Mat3 {
    const w = q[0];
    const x = q[1];
    const y = q[2];
    const z = q[3];
    return .{
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y),
        2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y),
    };
}

/// The quaternion of a rotation matrix (Shepperd's method).
pub fn fromMat(m: Mat3) Quat {
    const tr = m[0] + m[4] + m[8];
    var q: Quat = undefined;
    if (tr > 0) {
        const s = @sqrt(tr + 1.0) * 2;
        q = .{ 0.25 * s, (m[7] - m[5]) / s, (m[2] - m[6]) / s, (m[3] - m[1]) / s };
    } else if (m[0] > m[4] and m[0] > m[8]) {
        const s = @sqrt(1.0 + m[0] - m[4] - m[8]) * 2;
        q = .{ (m[7] - m[5]) / s, 0.25 * s, (m[1] + m[3]) / s, (m[2] + m[6]) / s };
    } else if (m[4] > m[8]) {
        const s = @sqrt(1.0 + m[4] - m[0] - m[8]) * 2;
        q = .{ (m[2] - m[6]) / s, (m[1] + m[3]) / s, 0.25 * s, (m[5] + m[7]) / s };
    } else {
        const s = @sqrt(1.0 + m[8] - m[0] - m[4]) * 2;
        q = .{ (m[3] - m[1]) / s, (m[2] + m[6]) / s, (m[5] + m[7]) / s, 0.25 * s };
    }
    return qstd(q);
}

pub fn matMul(a: Mat3, b: Mat3) Mat3 {
    var o: Mat3 = undefined;
    for (0..3) |i| for (0..3) |j| {
        o[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
    };
    return o;
}

pub fn matVec(m: Mat3, v: Vec3) Vec3 {
    return .{
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
        m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
        m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
    };
}

pub fn transpose(m: Mat3) Mat3 {
    return .{ m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8] };
}

pub fn det(m: Mat3) f64 {
    return m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
}

/// Bunge (phi1, Phi, phi2) → quaternion.
pub fn fromBunge(e: Vec3) Quat {
    const sigma = 0.5 * (e[0] + e[2]);
    const delta = 0.5 * (e[0] - e[2]);
    const c = @cos(0.5 * e[1]);
    const s = @sin(0.5 * e[1]);
    return qstd(.{ c * @cos(sigma), -s * @cos(delta), -s * @sin(delta), -c * @sin(sigma) });
}

/// Quaternion → Bunge (phi1, Phi, phi2), phi1 and phi2 in [0, 2π), Phi in [0, π].
pub fn toBunge(q0: Quat) Vec3 {
    const q = qstd(q0);
    const q03 = q[0] * q[0] + q[3] * q[3];
    const q12 = q[1] * q[1] + q[2] * q[2];
    const chi = @sqrt(q03 * q12);
    var e: Vec3 = undefined;
    if (chi < 1e-14) {
        if (q12 <= q03) {
            e = .{ std.math.atan2(-2 * q[0] * q[3], q[0] * q[0] - q[3] * q[3]), 0, 0 };
        } else {
            e = .{ std.math.atan2(2 * q[1] * q[2], q[1] * q[1] - q[2] * q[2]), std.math.pi, 0 };
        }
    } else {
        e = .{
            std.math.atan2((q[1] * q[3] - q[0] * q[2]) / chi, (-q[0] * q[1] - q[2] * q[3]) / chi),
            std.math.atan2(2 * chi, q03 - q12),
            std.math.atan2((q[0] * q[2] + q[1] * q[3]) / chi, (q[2] * q[3] - q[0] * q[1]) / chi),
        };
    }
    if (e[0] < 0) e[0] += 2 * std.math.pi;
    if (e[2] < 0) e[2] += 2 * std.math.pi;
    return e;
}

/// ZYZ angles (a, b, c): the rotation Rz(a) Ry(b) Rz(c) (active) as a quaternion.
pub fn fromZyz(a: f64, b: f64, c: f64) Quat {
    const qa: Quat = .{ @cos(0.5 * a), 0, 0, @sin(0.5 * a) };
    const qb: Quat = .{ @cos(0.5 * b), 0, @sin(0.5 * b), 0 };
    const qc: Quat = .{ @cos(0.5 * c), 0, 0, @sin(0.5 * c) };
    return qmul(qa, qmul(qb, qc));
}

/// Rotation angle between two orientations (radians), ignoring symmetry.
pub fn angleBetween(a: Quat, b: Quat) f64 {
    const d = @abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]);
    return 2 * std.math.acos(@min(d, 1.0));
}

// ── so(3) ──

/// exp of a rotation vector.
pub fn so3Exp(phi: Vec3) Quat {
    const th2 = dot(phi, phi);
    const th = @sqrt(th2);
    // sin(th/2)/th, series near zero
    const k = if (th < 1e-6) 0.5 - th2 / 48.0 else @sin(0.5 * th) / th;
    return .{ @cos(0.5 * th), k * phi[0], k * phi[1], k * phi[2] };
}

/// log of a unit quaternion: the rotation vector with angle in [0, π].
pub fn so3Log(q0: Quat) Vec3 {
    const q = qstd(q0);
    const s2 = q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    const s = @sqrt(s2);
    const k = if (s < 1e-9) 2.0 else 2.0 * std.math.atan2(s, q[0]) / s;
    return .{ k * q[1], k * q[2], k * q[3] };
}

fn hat(v: Vec3) Mat3 {
    return .{ 0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0 };
}

const mat_identity: Mat3 = .{ 1, 0, 0, 0, 1, 0, 0, 0, 1 };

fn matAxpy(a: Mat3, s: f64, b: Mat3) Mat3 {
    var o = a;
    for (0..9) |i| o[i] += s * b[i];
    return o;
}

/// Left Jacobian of SO(3): Exp(phi + d) ≈ Exp(Jl d) Exp(phi).
pub fn so3LeftJacobian(phi: Vec3) Mat3 {
    const th2 = dot(phi, phi);
    const th = @sqrt(th2);
    const K = hat(phi);
    const K2 = matMul(K, K);
    const a = if (th < 1e-5) 0.5 - th2 / 24.0 else (1 - @cos(th)) / th2;
    const b = if (th < 1e-5) 1.0 / 6.0 - th2 / 120.0 else (th - @sin(th)) / (th2 * th);
    return matAxpy(matAxpy(mat_identity, a, K), b, K2);
}

fn so3LeftJacobianInv(phi: Vec3) Mat3 {
    const th2 = dot(phi, phi);
    const th = @sqrt(th2);
    const K = hat(phi);
    const K2 = matMul(K, K);
    const c = if (th < 1e-5) 1.0 / 12.0 + th2 / 720.0 else (1.0 / th2) * (1 - th * @sin(th) / (2 * (1 - @cos(th))));
    return matAxpy(matAxpy(mat_identity, -0.5, K), c, K2);
}

// ── se(3) ──

/// exp of [tau, phi]: rotation Exp(phi), translation Jl(phi) tau.
pub fn se3Exp(xi: [6]f64) Se3 {
    const phi: Vec3 = .{ xi[3], xi[4], xi[5] };
    const t = matVec(so3LeftJacobian(phi), .{ xi[0], xi[1], xi[2] });
    const q = so3Exp(phi);
    return .{ t[0], t[1], t[2], q[0], q[1], q[2], q[3] };
}

pub fn se3Log(T: Se3) [6]f64 {
    const phi = so3Log(.{ T[3], T[4], T[5], T[6] });
    const tau = matVec(so3LeftJacobianInv(phi), .{ T[0], T[1], T[2] });
    return .{ tau[0], tau[1], tau[2], phi[0], phi[1], phi[2] };
}

/// a ∘ b: apply b, then a.
pub fn se3Mul(a: Se3, b: Se3) Se3 {
    const qa: Quat = .{ a[3], a[4], a[5], a[6] };
    const qb: Quat = .{ b[3], b[4], b[5], b[6] };
    const t = add(qapply(qa, .{ b[0], b[1], b[2] }), .{ a[0], a[1], a[2] });
    const q = qstd(qmul(qa, qb));
    return .{ t[0], t[1], t[2], q[0], q[1], q[2], q[3] };
}

pub fn se3Inv(a: Se3) Se3 {
    const qi = qconj(.{ a[3], a[4], a[5], a[6] });
    const t = scale(qapply(qi, .{ a[0], a[1], a[2] }), -1);
    return .{ t[0], t[1], t[2], qi[0], qi[1], qi[2], qi[3] };
}

pub fn se3Act(T: Se3, p: Vec3) Vec3 {
    return add(qapply(.{ T[3], T[4], T[5], T[6] }, p), .{ T[0], T[1], T[2] });
}

/// Left Jacobian of SE(3) at xi (6×6, row-major): Exp(xi + d) ≈ Exp(Jl d) Exp(xi). By central
/// differences of the group's own exp and log, which is accurate to ~1e-9 and has no formula
/// of its own to get wrong.
pub fn se3LeftJacobian(xi: [6]f64) [36]f64 {
    var J: [36]f64 = undefined;
    const h = 1e-6;
    const base_inv = se3Inv(se3Exp(xi));
    for (0..6) |j| {
        var xp = xi;
        var xm = xi;
        xp[j] += h;
        xm[j] -= h;
        const dp = se3Log(se3Mul(se3Exp(xp), base_inv));
        const dm = se3Log(se3Mul(se3Exp(xm), base_inv));
        for (0..6) |i| J[i * 6 + j] = (dp[i] - dm[i]) / (2 * h);
    }
    return J;
}

// ── cubochoric ↔ quaternion (Roşca, Morawiec, De Graef 2014) ──

/// Half edge of the cubochoric cube: (π^(2/3)) / 2.
pub const cubochoric_half_edge: f64 = 1.0725146985555481;

/// Cubochoric vector → quaternion (equal-volume map of the cube onto SO(3)).
pub fn fromCubochoric(cu: Vec3) Quat {
    return fromHomochoric(cuToHo(cu));
}

fn cuToHo(cu: Vec3) Vec3 {
    const pi = std.math.pi;
    const ax: [3]f64 = .{ @abs(cu[0]), @abs(cu[1]), @abs(cu[2]) };
    if (@max(ax[0], @max(ax[1], ax[2])) == 0) return .{ 0, 0, 0 };
    // the pyramid: (p[0], p[1]) span its base, p[2] is its axis
    var p: [3]usize = .{ 0, 1, 2 };
    if (ax[0] >= ax[1] and ax[0] >= ax[2]) p = .{ 1, 2, 0 } else if (ax[1] >= ax[0] and ax[1] >= ax[2]) p = .{ 2, 0, 1 };
    // cube of edge π^(2/3) → cube of edge a = π^(5/6) / 6^(1/6)
    const sc = std.math.pow(f64, pi, 1.0 / 6.0) / std.math.pow(f64, 6.0, 1.0 / 6.0);
    const x = cu[p[0]] * sc;
    const y = cu[p[1]] * sc;
    const z = cu[p[2]] * sc;
    const pref = @sqrt(6.0 / pi);
    var h: Vec3 = undefined;
    if (x == 0 and y == 0) {
        h[p[0]] = 0;
        h[p[1]] = 0;
        h[p[2]] = pref * z;
        return h;
    }
    // square → curved square (equal area), then the inverse Lambert projection onto the ball
    const sq2 = @sqrt(2.0);
    const beta = 0.5 * std.math.pow(f64, pi, 5.0 / 6.0) / std.math.pow(f64, 6.0, 1.0 / 6.0);
    const prek = std.math.cbrt(@as(f64, 3.0 * pi / 4.0)) * std.math.pow(f64, 2.0, 0.25) / beta;
    var t1: f64 = undefined;
    var t2: f64 = undefined;
    if (@abs(y) <= @abs(x)) {
        const q = (pi / 12.0) * y / x;
        const k = prek * x / @sqrt(sq2 - @cos(q));
        t1 = (sq2 * @cos(q) - 1) * k;
        t2 = sq2 * @sin(q) * k;
    } else {
        const q = (pi / 12.0) * x / y;
        const k = prek * y / @sqrt(sq2 - @cos(q));
        t1 = sq2 * @sin(q) * k;
        t2 = (sq2 * @cos(q) - 1) * k;
    }
    const c = t1 * t1 + t2 * t2;
    const q = @sqrt(@max(0.0, 1.0 - pi * c / (24.0 * z * z)));
    h[p[0]] = t1 * q;
    h[p[1]] = t2 * q;
    h[p[2]] = pref * z - @sqrt(pi) * c / (@sqrt(24.0) * z);
    return h;
}

/// Homochoric vector → quaternion: |h|³ = (3/4)(θ − sin θ).
pub fn fromHomochoric(h: Vec3) Quat {
    const r = norm(h);
    if (r < 1e-12) return quat_identity;
    // solve (3/4)(θ − sin θ) = r³ for θ in [0, π]: Newton from θ ≈ 2r (exact to third order)
    const target = r * r * r;
    var th: f64 = @min(std.math.pi, 2.0 * r);
    var it: u32 = 0;
    while (it < 50) : (it += 1) {
        const d = 0.75 * (1 - @cos(th));
        if (d < 1e-300) break;
        const step = (0.75 * (th - @sin(th)) - target) / d;
        th = @min(@max(th - step, 0.0), std.math.pi);
        if (@abs(step) < 1e-15) break;
    }
    const s = @sin(0.5 * th) / r;
    return .{ @cos(0.5 * th), h[0] * s, h[1] * s, h[2] * s };
}

// ── tests ──

const tol = 1e-12;

fn expectVec(a: Vec3, b: Vec3, eps: f64) !void {
    for (0..3) |i| try std.testing.expectApproxEqAbs(a[i], b[i], eps);
}

test "quaternion acts as its matrix" {
    const q = qstd(.{ 0.3, -0.5, 0.7, 0.2 });
    const v: Vec3 = .{ 0.2, -1.3, 0.8 };
    try expectVec(qapply(q, v), matVec(toMat(q), v), tol);
    const q2 = fromMat(toMat(q));
    for (0..4) |i| try std.testing.expectApproxEqAbs(q[i], q2[i], tol);
    try std.testing.expectApproxEqAbs(@as(f64, 1), det(toMat(q)), tol);
}

test "bunge round trip and passive sense" {
    const e: Vec3 = .{ 1.1, 0.7, 2.9 };
    const q = fromBunge(e);
    try expectVec(e, toBunge(q), 1e-10);
    // phi1 alone: sample x expressed in a crystal frame turned by +phi1 about z
    const v = qapply(fromBunge(.{ 0.3, 0, 0 }), .{ 1, 0, 0 });
    try expectVec(v, .{ @cos(0.3), -@sin(0.3), 0 }, tol);
}

test "zyz composes z y z" {
    const q = fromZyz(0.4, 0.9, -1.2);
    const v: Vec3 = .{ 0.3, 0.1, -0.7 };
    const rz = struct {
        fn f(a: f64, p: Vec3) Vec3 {
            return .{ @cos(a) * p[0] - @sin(a) * p[1], @sin(a) * p[0] + @cos(a) * p[1], p[2] };
        }
    }.f;
    const ry = struct {
        fn f(a: f64, p: Vec3) Vec3 {
            return .{ @cos(a) * p[0] + @sin(a) * p[2], p[1], -@sin(a) * p[0] + @cos(a) * p[2] };
        }
    }.f;
    try expectVec(qapply(q, v), rz(0.4, ry(0.9, rz(-1.2, v))), tol);
}

test "so3 and se3 exp, log, jacobian" {
    const phi: Vec3 = .{ 0.3, -0.2, 0.5 };
    try expectVec(so3Log(so3Exp(phi)), phi, 1e-12);
    const xi: [6]f64 = .{ 0.4, -0.1, 0.25, 0.3, -0.2, 0.5 };
    const back = se3Log(se3Exp(xi));
    for (0..6) |i| try std.testing.expectApproxEqAbs(xi[i], back[i], 1e-12);
    // Exp(xi + d) ≈ Exp(J d) Exp(xi)
    const J = se3LeftJacobian(xi);
    const d: [6]f64 = .{ 1e-5, -2e-5, 3e-5, 2e-5, 1e-5, -1e-5 };
    var xd = xi;
    var Jd: [6]f64 = @splat(0);
    for (0..6) |i| {
        xd[i] += d[i];
        for (0..6) |j| Jd[i] += J[i * 6 + j] * d[j];
    }
    const lhs = se3Exp(xd);
    const rhs = se3Mul(se3Exp(Jd), se3Exp(xi));
    for (0..7) |i| try std.testing.expectApproxEqAbs(lhs[i], rhs[i], 1e-9);
    // the rotation block of the SE(3) Jacobian is the SO(3) one
    const Js = so3LeftJacobian(phi);
    for (0..3) |i| for (0..3) |j| try std.testing.expectApproxEqAbs(Js[i * 3 + j], J[(3 + i) * 6 + 3 + j], 1e-7);
}

test "cubochoric is an equal-volume map onto the homochoric ball" {
    // a face centre lands on the ball's surface, radius (3π/4)^(1/3)
    const top = cuToHo(.{ 0, 0, cubochoric_half_edge });
    try std.testing.expectApproxEqAbs(std.math.cbrt(@as(f64, 0.75 * std.math.pi)), top[2], 1e-12);
    // unit Jacobian determinant at interior points of each pyramid
    const pts = [_]Vec3{ .{ 0.21, -0.33, 0.87 }, .{ -0.9, 0.2, 0.4 }, .{ 0.1, 0.8, -0.35 }, .{ 0.05, 0.02, 0.01 } };
    const h = 1e-6;
    for (pts) |c| {
        var J: Mat3 = undefined;
        for (0..3) |j| {
            var cp = c;
            var cm = c;
            cp[j] += h;
            cm[j] -= h;
            const d = sub(cuToHo(cp), cuToHo(cm));
            for (0..3) |i| J[i * 3 + j] = d[i] / (2 * h);
        }
        try std.testing.expectApproxEqAbs(@as(f64, 1), det(J), 1e-6);
    }
}

test "homochoric is equal volume in angle" {
    // a homochoric vector of length r has angle θ with (3/4)(θ − sin θ) = r³
    const q = fromHomochoric(.{ 0.3, 0.4, 0.5 });
    const th = 2 * std.math.acos(q[0]);
    try std.testing.expectApproxEqAbs(0.75 * (th - @sin(th)), std.math.pow(f64, @sqrt(0.5), 3), 1e-12);
}
