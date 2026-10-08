//! Pose fit: the detector's pose and the fit patterns' orientations, together.
//!
//! The patterns of a lattice over the region (pcfit.fitCells; settings.pose_grid a side, more
//! than the vote's: a pattern costs this fit little, and the detector's rotation is only as
//! good as there are patterns to tell it from the crystals') are first indexed at the current
//! geometry, and those whose match indexing itself would not accept (settings.accept_nmi or
//! accept_z, and accept_step_deg) are left out: a pattern matched to the wrong orientation
//! pulls the pose toward whatever suits its mistake. Then the mean normalized cross-correlation between each pattern and its
//! simulation is maximized in the tangent space: over a left perturbation of the detector
//! pose, x ↦ Exp(δ) x with δ = (τ, φ) in se(3); over the scale σ between the beam's travel and
//! the detector, s_b ↦ e^σ s_b (with the pose, a similarity: Sim(3); it is the detector's
//! pixel size); and over a tangent-space update of every pattern's orientation,
//! q_b ↦ Exp(φ_b) q_b. 7 + 3N unknowns, by Levenberg–Marquardt on the Jacobian of the
//! normalized simulations.
//!
//! The device supplies, per pattern, sums over six directions d = (∂I/∂u, c × ∇I): how the
//! simulation changes when the pattern's ray origin shifts, and when the crystal turns
//! (detector.wgsl `sums`). A pose change moves pattern b's rays by τ + φ × (u + s_b), with s_b
//! its beam offset, and a scale change by −σ s_b, so its derivatives are a linear map J_b of d;
//! the normal equations have an arrowhead shape (a shared 7 × 7 block, one 3 × 3 block per
//! pattern) and are solved by its Schur complement. One readback per iteration.
//!
//! The pose's six unknowns are solved for in the detector's own axes, δ = diag(R, R) δ': there
//! the three of τ' are the pattern centre's coordinates (a turn of the detector about the
//! source leaves the pattern centre alone) and the three of φ' are the detector's swing, tilt
//! and twist (geometry.rot). Each of the seven can be held (settings.pose_x … pose_scale).
//!
//! What tells a turn of the detector from a common turn of the crystals is the beam's travel
//! between the patterns. With no pixel size there is none in the model, the two are exactly the
//! same thing, and the attitude and the scale are held whatever the settings say.
const std = @import("std");
const engine = @import("engine.zig");
const rot = @import("rot.zig");
const sym = @import("sym.zig");
const detector = @import("detector.zig");
const spherical = @import("spherical.zig");
const pcfit = @import("pcfit.zig");
const Geometry = @import("geometry.zig").Geometry;

const Engine = engine.Engine;
const NS = detector.NS;

pub const Result = extern struct {
    pc: [3]f64 = .{ 0, 0, 0 },
    /// the detector's rotation from its nominal attitude (rotation vector, radians, in its own
    /// axes: swing, tilt, twist)
    rot: [3]f64 = .{ 0, 0, 0 },
    start_pc: [3]f64 = .{ 0, 0, 0 },
    start_rot: [3]f64 = .{ 0, 0, 0 },
    /// mean NCC of the fit patterns after and before
    ncc: f64 = 0,
    start_ncc: f64 = 0,
    /// mean rotation the patterns' orientations moved by, radians
    mean_turn: f64 = 0,
    iterations: u32 = 0,
    accepted: u32 = 0,
    /// the lattice's patterns that took part: those whose starting match was accepted
    n_patterns: u32 = 0,
    converged: u32 = 0,
    /// the detector's pixel size after and before, micrometres (the same unless the scale is fitted)
    px_um: f64 = 0,
    start_px_um: f64 = 0,
};

const State = struct {
    R: rot.Mat3,
    /// binned pixels
    t: rot.Vec3,
    /// what the beam offsets are multiplied by (the pixel size is divided by it)
    k: f64 = 1,
    q: []rot.Quat,
};

/// The pose's unknowns (τ, φ, σ), and those of one pattern's system (the pose's, then φ_b).
const NP = 7;
const NL = NP + 3;

/// Per pattern: its NCC, gradient and Gauss–Newton matrix over (τ, φ, σ, φ_b), from the
/// device's sums.
const Local = struct {
    ncc: f64 = 0,
    g: [NL]f64 = @splat(0),
    H: [NL * NL]f64 = @splat(0),
};

fn local(sums: []const f32, s_b: rot.Vec3, q: rot.Quat) Local {
    var out: Local = .{};
    const n: f64 = @max(sums[0], 1);
    const mean = sums[1] / n;
    const var_n = sums[2] - n * mean * mean;
    if (!(var_n > 1e-12)) return out;
    const inv = 1.0 / @sqrt(var_n);
    out.ncc = sums[3] * inv;
    var sd: [6]f64 = undefined;
    var au: [6]f64 = undefined;
    var g6: [6]f64 = undefined;
    for (0..6) |i| {
        sd[i] = sums[4 + i];
        au[i] = (sums[10 + i] - mean * sd[i]) * inv;
        g6[i] = (sums[16 + i] - out.ncc * au[i]) * inv;
    }
    var G6: [36]f64 = undefined;
    var o: usize = 22;
    for (0..6) |i| for (i..6) |j| {
        const v = (sums[o] - sd[i] * sd[j] / n - au[i] * au[j]) * inv * inv;
        G6[i * 6 + j] = v;
        G6[j * 6 + i] = v;
        o += 1;
    };
    // J (NL × 6): rows τ = [I 0], φ = [[s]× Rqᵀ], σ = [−s 0], φ_b = [0 I]
    var J: [NL * 6]f64 = @splat(0);
    const Rq = rot.toMat(q);
    for (0..3) |i| {
        J[i * 6 + i] = 1;
        J[(NP + i) * 6 + 3 + i] = 1;
        for (0..3) |j| J[(3 + i) * 6 + 3 + j] = Rq[j * 3 + i];
        J[6 * 6 + i] = -s_b[i];
    }
    // [s]× w = s × w
    J[3 * 6 + 1] = -s_b[2];
    J[3 * 6 + 2] = s_b[1];
    J[4 * 6 + 0] = s_b[2];
    J[4 * 6 + 2] = -s_b[0];
    J[5 * 6 + 0] = -s_b[1];
    J[5 * 6 + 1] = s_b[0];
    var JG: [NL * 6]f64 = @splat(0);
    for (0..NL) |i| for (0..6) |j| {
        var acc: f64 = 0;
        for (0..6) |k| acc += J[i * 6 + k] * G6[k * 6 + j];
        JG[i * 6 + j] = acc;
    };
    for (0..NL) |i| {
        var acc: f64 = 0;
        for (0..6) |k| acc += J[i * 6 + k] * g6[k];
        out.g[i] = acc;
        for (0..NL) |j| {
            var h: f64 = 0;
            for (0..6) |k| h += JG[i * 6 + k] * J[j * 6 + k];
            out.H[i * NL + j] = h;
        }
    }
    return out;
}

/// Pᵀ A P for NP × NP matrices.
fn congruent(A: [NP * NP]f64, P: [NP * NP]f64) [NP * NP]f64 {
    var AP: [NP * NP]f64 = @splat(0);
    for (0..NP) |i| for (0..NP) |j| {
        for (0..NP) |k| AP[i * NP + j] += A[i * NP + k] * P[k * NP + j];
    };
    var out: [NP * NP]f64 = @splat(0);
    for (0..NP) |i| for (0..NP) |j| {
        for (0..NP) |k| out[i * NP + j] += P[k * NP + i] * AP[k * NP + j];
    };
    return out;
}

/// Solve A x = b for a small dense system by elimination with partial pivoting.
fn solve(comptime N: usize, A0: [N * N]f64, b0: [N]f64) ?[N]f64 {
    var A = A0;
    var b = b0;
    for (0..N) |c| {
        var piv = c;
        for (c + 1..N) |r| if (@abs(A[r * N + c]) > @abs(A[piv * N + c])) {
            piv = r;
        };
        if (@abs(A[piv * N + c]) < 1e-300) return null;
        if (piv != c) {
            for (0..N) |k| std.mem.swap(f64, &A[c * N + k], &A[piv * N + k]);
            std.mem.swap(f64, &b[c], &b[piv]);
        }
        for (c + 1..N) |r| {
            const f = A[r * N + c] / A[c * N + c];
            if (f == 0) continue;
            for (c..N) |k| A[r * N + k] -= f * A[c * N + k];
            b[r] -= f * b[c];
        }
    }
    var x: [N]f64 = undefined;
    var r: usize = N;
    while (r > 0) {
        r -= 1;
        var acc = b[r];
        for (r + 1..N) |k| acc -= A[r * N + k] * x[k];
        x[r] = acc / A[r * N + r];
    }
    return x;
}

const Fit = struct {
    e: *Engine,
    lv: engine.Level,
    cells: []const u64,
    phase: []const i32,
    offs: []const rot.Vec3,
    sums: []f32,
    locals: []Local,
    poses: []detector.Pose,
    /// which of the lattice's patterns take part
    use: []const bool,
    n_used: usize,

    /// Mean NCC at `st`, leaving every pattern's local system in `locals`.
    fn eval(self: *Fit, st: *const State) !f64 {
        const e = self.e;
        const g = &e.g;
        const n = self.cells.len;
        var fr = try e.frame(self.lv);
        fr.c0 = .{ @floatCast(st.R[0]), @floatCast(st.R[3]), @floatCast(st.R[6]) };
        fr.c1 = .{ @floatCast(st.R[1]), @floatCast(st.R[4]), @floatCast(st.R[7]) };
        fr.c2 = .{ @floatCast(st.R[2]), @floatCast(st.R[5]), @floatCast(st.R[8]) };
        for (0..n) |b| {
            const t = rot.sub(st.t, rot.scale(self.offs[b], st.k));
            self.poses[b] = .{
                .t = .{ @floatCast(t[0]), @floatCast(t[1]), @floatCast(t[2]) },
                .pat = @floatFromInt(b),
                .q = .{ @floatCast(st.q[b][0]), @floatCast(st.q[b][1]), @floatCast(st.q[b][2]), @floatCast(st.q[b][3]) },
            };
        }
        e.det.setPoses(self.poses);
        const n_ph = e.phases.items.len;
        for (e.phases.items, 0..) |*p, pi| {
            e.det.sumsOf(p.det[@intFromEnum(self.lv)], fr, e.pats[@intFromEnum(self.lv)], @intCast(n), false);
            g.read(e.det.sums, 0, std.mem.sliceAsBytes(self.sums[pi * n * NS ..][0 .. n * NS]));
        }
        try g.wait();
        var mean: f64 = 0;
        for (0..n) |b| {
            const p: usize = @intCast(@max(self.phase[b], 0));
            self.locals[b] = .{};
            if (p >= n_ph or !self.use[b]) continue;
            self.locals[b] = local(self.sums[(p * n + b) * NS ..][0..NS], rot.scale(self.offs[b], st.k), st.q[b]);
            mean += self.locals[b].ncc;
        }
        return mean / @as(f64, @floatFromInt(self.n_used));
    }

    /// The damped Gauss–Newton step from `locals`: the pose's (τ, φ), the scale's σ and each
    /// pattern's φ_b. `free` says which of the seven move, the pose's in the detector's axes
    /// (those of `R`): the position along its rows, columns and normal, the turns about them,
    /// then the scale.
    fn step(self: *Fit, lambda: f64, free: [NP]bool, R: rot.Mat3, dphi: []rot.Vec3) ?[NP]f64 {
        var S: [NP * NP]f64 = @splat(0);
        var rhs: [NP]f64 = @splat(0);
        for (self.locals) |*l| {
            for (0..NP) |i| {
                rhs[i] += l.g[i];
                for (0..NP) |j| S[i * NP + j] += l.H[i * NL + j];
            }
        }
        const S0 = S;
        // eliminate every pattern's orientation block
        for (self.locals) |*l| {
            var B: [9]f64 = undefined;
            for (0..3) |i| for (0..3) |j| {
                B[i * 3 + j] = l.H[(NP + i) * NL + NP + j];
            };
            const tr = B[0] + B[4] + B[8];
            for (0..3) |i| B[i * 3 + i] = B[i * 3 + i] * (1 + lambda) + 1e-9 * tr + 1e-30;
            // columns of B⁻¹ Cᵀ and B⁻¹ g_b
            var Y: [3 * NP]f64 = undefined;
            for (0..NP) |c| {
                const col = solve(3, B, .{ l.H[c * NL + NP], l.H[c * NL + NP + 1], l.H[c * NL + NP + 2] }) orelse return null;
                for (0..3) |i| Y[i * NP + c] = col[i];
            }
            const yg = solve(3, B, .{ l.g[NP], l.g[NP + 1], l.g[NP + 2] }) orelse return null;
            for (0..NP) |i| {
                var acc: f64 = 0;
                for (0..3) |k| acc += l.H[i * NL + NP + k] * yg[k];
                rhs[i] -= acc;
                for (0..NP) |j| {
                    var h: f64 = 0;
                    for (0..3) |k| h += l.H[i * NL + NP + k] * Y[k * NP + j];
                    S[i * NP + j] -= h;
                }
            }
        }
        // to the detector's axes, δ = P δ' with P = diag(R, R, 1), where an unknown can be held
        var P: [NP * NP]f64 = @splat(0);
        for (0..3) |i| for (0..3) |j| {
            P[i * NP + j] = R[i * 3 + j];
            P[(3 + i) * NP + 3 + j] = R[i * 3 + j];
        };
        P[6 * NP + 6] = 1;
        var Sd = congruent(S, P);
        const S0d = congruent(S0, P);
        var rd: [NP]f64 = @splat(0);
        for (0..NP) |i| for (0..NP) |k| {
            rd[i] += P[k * NP + i] * rhs[k];
        };
        for (0..NP) |i| Sd[i * NP + i] += lambda * S0d[i * NP + i] + 1e-12 * S0d[i * NP + i] + 1e-30;
        for (0..NP) |i| if (!free[i]) {
            for (0..NP) |j| {
                Sd[i * NP + j] = 0;
                Sd[j * NP + i] = 0;
            }
            Sd[i * NP + i] = 1;
            rd[i] = 0;
        };
        const dd = solve(NP, Sd, rd) orelse return null;
        var d: [NP]f64 = @splat(0);
        for (0..NP) |i| for (0..NP) |k| {
            d[i] += P[i * NP + k] * dd[k];
        };
        for (self.locals, 0..) |*l, b| {
            var B: [9]f64 = undefined;
            for (0..3) |i| for (0..3) |j| {
                B[i * 3 + j] = l.H[(NP + i) * NL + NP + j];
            };
            const tr = B[0] + B[4] + B[8];
            for (0..3) |i| B[i * 3 + i] = B[i * 3 + i] * (1 + lambda) + 1e-9 * tr + 1e-30;
            var r: rot.Vec3 = .{ l.g[NP], l.g[NP + 1], l.g[NP + 2] };
            for (0..3) |i| for (0..NP) |c| {
                r[i] -= l.H[c * NL + NP + i] * d[c];
            };
            dphi[b] = solve(3, B, r) orelse return null;
        }
        return d;
    }
};

pub fn run(e: *Engine) !Result {
    const gpa = e.gpa;
    const s = &e.set;
    e.cancel = false;
    // the pose fit is a refinement: it works where refinement does
    const lv = e.levelOf(.refine);
    const sz = try e.sizeOf(lv);
    const n_ph = e.phases.items.len;
    if (n_ph == 0) return error.NoPhases;

    // seeds: the fit patterns indexed at the current geometry (this also leaves them, filtered,
    // on the device)
    // as many patterns as the lattice asks for, or as fit in one buffer on this device
    var grid = s.pose_grid;
    const room = @min(e.g.lim.max_storage_binding, e.g.lim.max_buffer) / 2;
    while (grid > 1 and @as(u64, grid) * grid * @as(u64, @intFromFloat(@ceil(sz[0] * sz[1]))) * 4 > room) grid -= 1;
    if (grid < s.pose_grid) e.events.log("pose fit: {d} × {d} patterns of this size are what fits in one buffer here (pose_grid asks for {d} a side)", .{ grid, grid, s.pose_grid });
    const cells = try pcfit.fitCells(e, grid);
    errdefer gpa.free(cells);
    const n = cells.len;
    const recs = try gpa.alloc(engine.Record, n_ph * n);
    defer gpa.free(recs);
    try e.indexCells(cells, .{ .bandwidth = s.fit_bandwidth, .dict_grid = s.dict_grid }, recs);
    const phase = try gpa.alloc(i32, n);
    errdefer gpa.free(phase);
    var st: State = .{ .R = e.geo.rotation(), .t = e.geo.translation(sz[0], sz[1]), .q = try gpa.alloc(rot.Quat, n) };
    defer gpa.free(st.q);
    var trial: State = .{ .R = st.R, .t = st.t, .q = try gpa.alloc(rot.Quat, n) };
    defer gpa.free(trial.q);
    const q0 = try gpa.alloc(rot.Quat, n);
    defer gpa.free(q0);
    const offs = try gpa.alloc(rot.Vec3, n);
    defer gpa.free(offs);
    // which patterns take part: those whose match indexing would accept
    const use = try gpa.alloc(bool, n);
    defer gpa.free(use);
    const seed_ncc = try gpa.alloc(f32, n);
    defer gpa.free(seed_ncc);
    const settle: f32 = @floatCast(std.math.degreesToRadians(s.accept_step_deg));
    var n_used: usize = 0;
    for (0..n) |b| {
        const p = e.choosePhase(recs, n, b);
        phase[b] = @intCast(p);
        const r = recs[p * n + b];
        seed_ncc[b] = r.ncc;
        use[b] = (if (s.accept_nmi > 0) r.nmi >= s.accept_nmi else r.z >= s.accept_z) and (!s.refine or r.step <= settle);
        if (use[b]) n_used += 1;
        st.q[b] = rot.qstd(.{ r.q[0], r.q[1], r.q[2], r.q[3] });
        q0[b] = st.q[b];
        offs[b] = e.offsetOf(&e.geo, cells[b], lv);
    }

    if (n_used < 3) {
        // nothing to choose among: a start this far off, or a threshold this high, leaves the fit all of them
        e.events.log("pose fit: {d} of the {d} fit patterns have a match that would be accepted here; fitting to all of them", .{ n_used, n });
        @memset(use, true);
        n_used = n;
    } else if (n_used < n) {
        e.events.log("pose fit: {d} of the {d} fit patterns left out, their matches not accepted", .{ n - n_used, n });
    }
    var fit: Fit = .{
        .use = use,
        .n_used = n_used,
        .e = e,
        .lv = lv,
        .cells = cells,
        .phase = phase,
        .offs = offs,
        .sums = try gpa.alloc(f32, n_ph * n * NS),
        .locals = try gpa.alloc(Local, n),
        .poses = try gpa.alloc(detector.Pose, n),
    };
    defer gpa.free(fit.sums);
    defer gpa.free(fit.locals);
    defer gpa.free(fit.poses);
    const keep = try gpa.alloc(Local, n);
    defer gpa.free(keep);
    const dphi = try gpa.alloc(rot.Vec3, n);
    defer gpa.free(dphi);

    var res: Result = .{ .start_pc = e.geo.pc, .start_rot = e.geo.rot, .n_patterns = @intCast(n_used), .px_um = e.geo.px_um, .start_px_um = e.geo.px_um };
    // the detector's axes: rows (pattern centre y; swing about it), columns (x; tilt), normal (z; twist)
    // then the scale, which needs a pixel size to be a multiple of
    var free: [NP]bool = .{ s.pose_y, s.pose_x, s.pose_z, s.pose_swing, s.pose_tilt, s.pose_twist, s.pose_scale };
    if (!(e.geo.px_um > 0)) {
        // Without a pixel size every pattern comes from the one point, and a turn of the detector
        // about it is then exactly a turn of every crystal the other way: the correlation cannot
        // move for it, in any pattern or all of them. That is not a weak direction but no
        // direction at all, so the attitude stays as set rather than drift.
        for (3..NP) |i| free[i] = false;
        e.events.log("pose fit: the detector's pixel size is not set, so every pattern is taken to come from the one point of the sample. A turn of the detector about that point and the same turn of every crystal are then one and the same, so the detector's rotation is left as set and its position alone is fitted. Set the pixel size (Geometry) to fit the rotation.", .{});
    }
    var ncc = try fit.eval(&st);
    res.start_ncc = ncc;
    @memcpy(keep, fit.locals);
    var lambda: f64 = 1e-3;
    var small: u32 = 0;
    var it: u32 = 0;
    while (it < s.pose_iters) : (it += 1) {
        if (e.cancel) break;
        res.iterations = it + 1;
        // the step is always taken from the best state's system
        @memcpy(fit.locals, keep);
        const d = fit.step(lambda, free, st.R, dphi) orelse {
            lambda *= 10;
            continue;
        };
        // the step taken in the detector's axes, where it is Exp(δ) to first order and a held
        // unknown stays exactly where it is: the source's place on the detector moves by τ',
        // and the detector turns about the source by φ'
        const Rt = rot.transpose(st.R);
        const p = rot.add(rot.matVec(Rt, st.t), rot.matVec(Rt, .{ d[0], d[1], d[2] }));
        trial.R = rot.matMul(st.R, rot.toMat(rot.so3Exp(rot.matVec(Rt, .{ d[3], d[4], d[5] }))));
        trial.t = rot.matVec(trial.R, p);
        trial.k = st.k * @exp(std.math.clamp(d[6], -0.5, 0.5));
        for (0..n) |b| trial.q[b] = rot.qstd(rot.qmul(rot.so3Exp(dphi[b]), st.q[b]));
        const ncc_t = try fit.eval(&trial);
        if (ncc_t > ncc) {
            const gain = ncc_t - ncc;
            st.R = trial.R;
            st.t = trial.t;
            st.k = trial.k;
            @memcpy(st.q, trial.q);
            ncc = ncc_t;
            @memcpy(keep, fit.locals);
            lambda = @max(lambda / 3, 1e-4);
            res.accepted += 1;
            const pc = Geometry.pcOf(st.R, st.t, sz[0], sz[1]);
            e.events.emit(.{ .kind = "pose", .iter = it, .ncc = ncc, .pc = pc });
            // settled when two accepted steps in a row gain next to nothing
            small = if (gain < 1e-6 * @max(@abs(ncc), 1e-6)) small + 1 else 0;
            if (small >= 2) {
                res.converged = 1;
                break;
            }
        } else {
            lambda *= 5;
            if (lambda > 1e8) {
                res.converged = 1;
                break;
            }
        }
    }

    // the geometry takes the fitted pose
    e.geo.pc = Geometry.pcOf(st.R, st.t, sz[0], sz[1]);
    e.geo.rot = rot.so3Log(rot.fromMat(rot.matMul(rot.transpose(e.geo.nominal()), st.R)));
    if (free[6]) e.geo.px_um /= st.k;
    if (free[6] and st.k < 0.1) e.events.log("pose fit: the fitted scale leaves {d:.1} % of the beam's travel the pixel size and the steps give. A scale that runs away like this says the patterns do not move the way the scan says they should: the scan directions (scan_x_sign, scan_y_sign) are probably the wrong way round.", .{100 * st.k});
    res.px_um = e.geo.px_um;
    res.pc = e.geo.pc;
    res.rot = e.geo.rot;
    res.ncc = ncc;
    var turn: f64 = 0;
    for (0..n) |b| {
        if (use[b]) turn += rot.angleBetween(st.q[b], q0[b]);
    }
    res.mean_turn = turn / @as(f64, @floatFromInt(n_used));

    gpa.free(e.fit_cells);
    gpa.free(e.fit_quats);
    gpa.free(e.fit_phase);
    gpa.free(e.fit_ncc);
    e.fit_cells = cells;
    e.fit_phase = phase;
    e.fit_quats = &.{};
    e.fit_ncc = &.{};
    e.fit_quats = try gpa.alloc(f32, 4 * n);
    e.fit_ncc = try gpa.alloc(f32, n);
    for (0..n) |b| {
        const p: usize = @intCast(phase[b]);
        const q = sym.toFz(st.q[b], e.phases.items[p].master.rotation);
        for (0..4) |k| e.fit_quats[4 * b + k] = @floatCast(q[k]);
        e.fit_ncc[b] = if (use[b]) @floatCast(keep[b].ncc) else seed_ncc[b];
    }
    const deg = 180.0 / std.math.pi;
    e.events.log("pose fit: pattern centre {d:.5} {d:.5} {d:.5}  swing, tilt, twist {d:.3}° {d:.3}° {d:.3}°  pixel {d:.2} µm  mean NCC {d:.4} (was {d:.4}) in {d} iterations", .{ res.pc[0], res.pc[1], res.pc[2], res.rot[0] * deg, res.rot[1] * deg, res.rot[2] * deg, res.px_um, res.ncc, res.start_ncc, res.iterations });
    return res;
}

test "small dense solve" {
    const x = solve(3, .{ 2, 1, 0, 1, 3, 1, 0, 1, 4 }, .{ 3, 5, 5 }).?;
    for (x) |v| try std.testing.expectApproxEqAbs(@as(f64, 1), v, 1e-12);
    try std.testing.expect(solve(2, .{ 1, 2, 2, 4 }, .{ 1, 1 }) == null);
}
