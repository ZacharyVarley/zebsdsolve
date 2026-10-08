// Shared by every module (shaders.zig puts this text first).
//
// A quaternion is a vec4 holding (w, x, y, z) in (.x, .y, .z, .w), as the host stores it, and
// rotates a vector as q v q*. An orientation maps sample directions to crystal directions.

const PI: f32 = 3.14159265358979;
const TAU: f32 = 6.28318530717959;

fn qmul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        a.x * b.x - a.y * b.y - a.z * b.z - a.w * b.w,
        a.x * b.y + a.y * b.x + a.z * b.w - a.w * b.z,
        a.x * b.z - a.y * b.w + a.z * b.x + a.w * b.y,
        a.x * b.w + a.y * b.z - a.z * b.y + a.w * b.x);
}

fn qconj(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(q.x, -q.y, -q.z, -q.w);
}

fn qrot(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let u = q.yzw;
    let t = 2.0 * cross(u, v);
    return v + q.x * t + cross(u, t);
}

fn qnormalize(q: vec4<f32>) -> vec4<f32> {
    let n = length(q);
    if (n > 0.0) { return q / n; }
    return vec4<f32>(1.0, 0.0, 0.0, 0.0);
}

// exp of a rotation vector
fn qexp(phi: vec3<f32>) -> vec4<f32> {
    let th2 = dot(phi, phi);
    let th = sqrt(th2);
    var k = 0.5 - th2 / 48.0;
    if (th > 1e-3) { k = sin(0.5 * th) / th; }
    return vec4<f32>(cos(0.5 * th), k * phi);
}

// Unit vector → equal-area square coordinates in [−1, 1]² of its own hemisphere (lambert.zig).
// 1 − |z| is taken as (x² + y²) / (1 + |z|), which keeps its digits near the pole.
fn sphere_to_square(v: vec3<f32>) -> vec2<f32> {
    let ax = abs(v.x);
    let ay = abs(v.y);
    let xn = sqrt(max(PI * 0.5 * (ax * ax + ay * ay) / (1.0 + abs(v.z)), 0.0));
    var u = xn;
    var w = xn;
    if (ax >= ay) {
        w = (4.0 * xn / PI) * atan2(ay, max(ax, 1e-30));
    } else {
        u = (4.0 * xn / PI) * atan2(ax, max(ay, 1e-30));
    }
    let sc = 0.79788456080286536; // 1 / sqrt(π / 2)
    return vec2<f32>(u * sc * select(1.0, -1.0, v.x < 0.0), w * sc * select(1.0, -1.0, v.y < 0.0));
}

// kernels over more elements than one dispatch dimension holds index gid.x + gid.y · grid_x
fn flat_id(wid: vec3<u32>, lid: u32, grid_x: u32, wg: u32) -> u32 {
    return (wid.y * grid_x + wid.x) * wg + lid;
}
