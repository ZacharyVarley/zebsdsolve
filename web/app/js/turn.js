// Turning a view with the mouse: rotations as 9 numbers, row-major, each taking view directions
// (x right, y up, z toward the viewer) into the frame of what is shown. A drag turns the view so
// that the point under the pointer stays under it (an arcball).

export const IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0, 1];

/** R for a viewer at direction `d` (the shown frame), `up` pointing up on the screen (or, when
 *  the viewer looks along it, the frame's y, else its x). */
export function lookFrom(d, up = [0, 0, 1]) {
  const n = Math.hypot(...d);
  const zv = d.map((c) => c / n);
  const un = Math.hypot(...up);
  up = up.map((c) => c / un);
  if (Math.abs(up[0] * zv[0] + up[1] * zv[1] + up[2] * zv[2]) > 0.98) up = Math.abs(zv[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
  const k = up[0] * zv[0] + up[1] * zv[1] + up[2] * zv[2];
  let yv = up.map((c, i) => c - k * zv[i]);
  const yn = Math.hypot(...yv);
  yv = yv.map((c) => c / yn);
  const xv = [yv[1] * zv[2] - yv[2] * zv[1], yv[2] * zv[0] - yv[0] * zv[2], yv[0] * zv[1] - yv[1] * zv[0]];
  // columns: the view's axes in the shown frame
  return [xv[0], yv[0], zv[0], xv[1], yv[1], zv[1], xv[2], yv[2], zv[2]];
}

export const mul = (a, b) => {
  const o = new Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) o[3 * i + j] = a[3 * i] * b[j] + a[3 * i + 1] * b[3 + j] + a[3 * i + 2] * b[6 + j];
  return o;
};

/** R v, and Rᵀ v (a shown direction into the view's frame). */
export const apply = (R, v) => [R[0] * v[0] + R[1] * v[1] + R[2] * v[2], R[3] * v[0] + R[4] * v[1] + R[5] * v[2], R[6] * v[0] + R[7] * v[1] + R[8] * v[2]];
export const applyT = (R, v) => [R[0] * v[0] + R[3] * v[1] + R[6] * v[2], R[1] * v[0] + R[4] * v[1] + R[7] * v[2], R[2] * v[0] + R[5] * v[1] + R[8] * v[2]];

/** The rotation that spins a view by angle t about its own z (the axis toward the viewer: in a
 *  flat projection, its north pole), applied on the right: R' = R · spinZ(t). */
export const spinZ = (t) => axisAngle([0, 0, 1], t);

/** Rotation by angle t about the unit axis u. */
function axisAngle(u, t) {
  const [x, y, z] = u;
  const c = Math.cos(t), s = Math.sin(t), C = 1 - c;
  return [c + x * x * C, x * y * C - z * s, x * z * C + y * s, y * x * C + z * s, c + y * y * C, y * z * C - x * s, z * x * C - y * s, z * y * C + x * s, c + z * z * C];
}

/** The rotation that takes unit vector a onto unit vector b (the short way). */
export function between(a, b) {
  const ax = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const s = Math.hypot(...ax);
  const c = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  if (s < 1e-9) return IDENTITY.slice();
  return axisAngle(ax.map((v) => v / s), Math.atan2(s, c));
}

/** Back onto the rotations (Gram–Schmidt on the rows): a long drag accumulates rounding. */
export function orthonormal(m) {
  const r0 = [m[0], m[1], m[2]];
  let n = Math.hypot(...r0);
  r0.forEach((v, i) => { r0[i] = v / n; });
  const d = m[3] * r0[0] + m[4] * r0[1] + m[5] * r0[2];
  const r1 = [m[3] - d * r0[0], m[4] - d * r0[1], m[5] - d * r0[2]];
  n = Math.hypot(...r1);
  r1.forEach((v, i) => { r1[i] = v / n; });
  const r2 = [r0[1] * r1[2] - r0[2] * r1[1], r0[2] * r1[0] - r0[0] * r1[2], r0[0] * r1[1] - r0[1] * r1[0]];
  return [...r0, ...r1, ...r2];
}

/** A point of the arcball under (x, y), in units of the ball's radius from its centre (y up):
 *  on the ball, or on the hyperbolic sheet past it. */
export function arcball(x, y) {
  const r2 = x * x + y * y;
  const z = r2 <= 0.5 ? Math.sqrt(1 - r2) : 0.5 / Math.sqrt(r2);
  const n = Math.hypot(x, y, z);
  return [x / n, y / n, z / n];
}

/** Frames for a picture the engine renders: at most one on its way, only the latest wish kept;
 *  rough (half the pixels each way) while the mouse moves, sharp once it rests. `draw(rough)`
 *  renders and shows one frame (a promise). */
export class Frames {
  constructor(draw) {
    this.draw = draw;
    this.busy = false;
    this.dirty = false;
    this.fine = 0;
    this.rough = false;
  }

  request(rough = false) {
    this.rough = rough;
    if (this.busy) { this.dirty = true; return; }
    this.run();
    if (rough) {
      clearTimeout(this.fine);
      this.fine = setTimeout(() => this.request(false), 140);
    }
  }

  async run() {
    this.busy = true;
    try {
      await this.draw(this.rough);
    } catch (e) {
      console.warn(e);
    } finally {
      this.busy = false;
      if (this.dirty) { this.dirty = false; this.run(); }
    }
  }
}

/** The canvas size for a well: its pixels (half each way when rough), the longer side at most 1600. */
export function pictureSize(w, h, rough) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const k = Math.min(dpr * (rough ? 0.5 : 1), 1600 / Math.max(w, h, 1));
  return [Math.max(32, Math.round(w * k)), Math.max(32, Math.round(h * k))];
}

/** Drag to turn, scroll to zoom, control-scroll to spin about the view's axis, double-click to
 *  reset, on `el`: `at(ev)` gives the unit direction under the pointer in the view's frame (or
 *  null: no drag), `turn(R)` takes the rotation to apply on the right (R' = R · turn), `zoom(f)` a
 *  zoom factor, `reset()` the first view. */
export function wireTurning(el, { at, turn, zoom, reset, dragging, pan = null }) {
  let from = null;
  let last = null;
  // a drag moves the picture (with `pan`) when shift is down, or where `at` has no direction to
  // turn by (a locked view, the space beside a projection)
  el.addEventListener("pointerdown", (ev) => {
    if (ev.button !== 0) return;
    const p = ev.shiftKey && pan ? null : at(ev);
    if (!p && !pan) return;
    try { el.setPointerCapture(ev.pointerId); } catch { /* a pointer the browser does not track */ }
    from = p;
    last = [ev.clientX, ev.clientY];
    dragging(true, !p);
  });
  el.addEventListener("pointermove", (ev) => {
    if (!last) return;
    if (!from) {
      pan(ev.clientX - last[0], ev.clientY - last[1]);
      last = [ev.clientX, ev.clientY];
      return;
    }
    const p = at(ev);
    if (!p) return;
    // the point that was under the last pointer is under this one: R' p = R p₀
    turn(between(p, from));
    from = p;
  });
  const end = () => { if (last) { from = null; last = null; dragging(false, false); } };
  el.addEventListener("pointerup", end);
  el.addEventListener("pointercancel", end);
  el.addEventListener("wheel", (ev) => {
    ev.preventDefault();
    const dy = ev.deltaY * (ev.deltaMode === 1 ? 33 : 1);
    if (ev.ctrlKey) turn(spinZ(dy * 0.0006));
    else zoom(Math.exp(-dy * 0.0015), ev);
  }, { passive: false });
  el.addEventListener("dblclick", reset);
}
