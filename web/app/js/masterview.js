// A master pattern in a picture: the engine renders the pixels (mastersph.wgsl, sphere_view) in
// one of four projections; this holds each projection's rotation and zoom, turns the mouse into
// them, and draws the crystal's axes and its equator over the picture.
//
//   globe          the sphere seen from outside (orthographic), lit
//   lambert        the view's upper and lower hemispheres as equal-area squares
//   stereographic  … as stereographic discs (angles kept, areas not)
//   joint          the whole sphere in one square: the upper Lambert square in the middle (a
//                  diamond), the lower one's four triangles folded out into the corners
//
// The rotation R takes view directions (x right, y up, z toward the viewer) into the crystal's
// frame. The flat projections start with R = 1: the upper hemisphere is the crystal's, x to the
// right and y up, the lower one seen from above (through the sphere), as the master's sheets are.

import { IDENTITY, lookFrom, mul, applyT, orthonormal, arcball, Frames, pictureSize, wireTurning, spinZ } from "./turn.js";

const NS = "http://www.w3.org/2000/svg";
const AXES = [["x", "#e0705e"], ["y", "#7fbf6e"], ["z", "#6a9ec8"]];
export const PROJECTIONS = ["globe", "lambert", "stereographic", "joint"];
const FIRST = {
  globe: { rot: lookFrom([1.0, 0.55, 0.85]), zoom: 0.86 },
  lambert: { rot: IDENTITY, zoom: 0.86 },
  stereographic: { rot: IDENTITY, zoom: 0.86 },
  joint: { rot: IDENTITY, zoom: 0.9 },
};
const SQ = Math.sqrt(Math.PI / 2);
// a CSS pixel's width on a 96 dpi screen, mm
const MM_PER_PX = 25.4 / 96;

// ── the projections' maps (the kernel's, for the pointer and the marks over the picture) ──

/** A unit direction → its Lambert square coordinates in [−1, 1]² (by |z|). */
function toSquare(v) {
  const ax = Math.abs(v[0]), ay = Math.abs(v[1]);
  const xn = Math.sqrt(Math.max((Math.PI / 2) * (ax * ax + ay * ay) / (1 + Math.abs(v[2])), 0));
  let u = xn, w = xn;
  if (ax >= ay) w = (4 * xn / Math.PI) * Math.atan2(ay, Math.max(ax, 1e-30));
  else u = (4 * xn / Math.PI) * Math.atan2(ax, Math.max(ay, 1e-30));
  return [(u / SQ) * (v[0] < 0 ? -1 : 1), (w / SQ) * (v[1] < 0 ? -1 : 1)];
}

/** Lambert square coordinates → the direction on the upper (or lower) hemisphere. */
function fromSquare([px, py], lower) {
  const xa = Math.abs(px) * SQ, ya = Math.abs(py) * SQ;
  const xn = Math.max(xa, ya), yn = Math.min(xa, ya);
  let v = [0, 0, 1];
  if (xn > 0) {
    const r = (2 * xn / Math.PI) * Math.sqrt(Math.max(Math.PI - xn * xn, 0));
    const ang = (Math.PI * yn) / (4 * xn);
    let hx = r * Math.cos(ang), hy = r * Math.sin(ang);
    if (xa < ya) [hx, hy] = [hy, hx];
    v = [hx * (px < 0 ? -1 : 1), hy * (py < 0 ? -1 : 1), 1 - (2 * xn * xn) / Math.PI];
  }
  if (lower) v[2] = -v[2];
  const n = Math.hypot(...v);
  return v.map((c) => c / n);
}

/** Where each panel of a projection lies in a w × h picture: [{ x, y, half, lower }] (pixels, the
 *  upper hemisphere's first), as the kernel lays them out: the layout scales about the picture's
 *  middle by `zoom`, then moves by `pan` (shorter sides). `shown` (the flat projections): 0 both
 *  hemispheres, 1 the upper, 2 the lower. */
function panels(projection, w, h, zoom, pan = [0, 0], shown = 0) {
  const m = Math.min(w, h);
  const cx = w / 2 + pan[0] * m, cy = h / 2 + pan[1] * m;
  if ((projection === "lambert" || projection === "stereographic") && shown === 0) {
    const wide = w >= h;
    const half = 0.5 * (wide ? Math.min(w / 2, h) : Math.min(w, h / 2)) * zoom;
    const dx = wide ? (w / 4) * zoom : 0, dy = wide ? 0 : (h / 4) * zoom;
    return [{ x: cx - dx, y: cy - dy, half, lower: false }, { x: cx + dx, y: cy + dy, half, lower: true }];
  }
  return [{ x: cx, y: cy, half: 0.5 * m * zoom, lower: shown === 2 && projection !== "globe" && projection !== "joint" }];
}

/** A view direction → its place in the panels: { panel, q (panel coordinates, y up), near } —
 *  near: on the globe's visible side. */
function forward(projection, v) {
  switch (projection) {
    case "globe": return { panel: 0, q: [v[0], v[1]], near: v[2] >= 0 };
    case "lambert": return { panel: v[2] < 0 ? 1 : 0, q: toSquare(v), near: true };
    case "stereographic": { const k = 1 / (1 + Math.abs(v[2])); return { panel: v[2] < 0 ? 1 : 0, q: [v[0] * k, v[1] * k], near: true }; }
    default: {
      let [u, w] = toSquare(v);
      if (v[2] < 0) {
        // the lower square's triangle, folded out across the upper square's edge
        if (u >= Math.abs(w)) u = 2 - u;
        else if (u <= -Math.abs(w)) u = -2 - u;
        else if (w >= Math.abs(u)) w = 2 - w;
        else w = -2 - w;
      }
      return { panel: 0, q: [(u - w) / 2, (u + w) / 2], near: true };
    }
  }
}

/** Panel coordinates (y up) → the view direction there, or null outside the projection. */
function inverse(projection, lower, q) {
  const r = Math.hypot(q[0], q[1]);
  switch (projection) {
    case "globe": {
      // past the rim: the arcball's sheet, so that a drag can start beside the globe
      return arcball(q[0], q[1]);
    }
    case "lambert":
      if (Math.max(Math.abs(q[0]), Math.abs(q[1])) > 1) return null;
      return fromSquare(q, lower);
    case "stereographic": {
      if (r > 1) return null;
      const r2 = r * r;
      return [(2 * q[0]) / (1 + r2), (2 * q[1]) / (1 + r2), ((1 - r2) / (1 + r2)) * (lower ? -1 : 1)];
    }
    default: {
      if (Math.max(Math.abs(q[0]), Math.abs(q[1])) > 1) return null;
      const u = q[0] + q[1], w = q[1] - q[0];
      if (Math.abs(u) <= 1 && Math.abs(w) <= 1) return fromSquare([u, w], false);
      let s = [u, -2 - w];
      if (u > 1) s = [2 - u, w];
      else if (u < -1) s = [-2 - u, w];
      else if (w > 1) s = [u, 2 - w];
      return fromSquare(s.map((c) => Math.max(-1, Math.min(1, c))), true);
    }
  }
}

/** A crystal direction in words: an axis when it is one, else its components. */
function dirName(c) {
  for (let k = 0; k < 3; k++) {
    if (Math.abs(Math.abs(c[k]) - 1) < 2e-4) return `${c[k] < 0 ? "−" : "+"}${AXES[k][0]}`;
  }
  return `[${c.map((v) => v.toFixed(2)).join(" ")}]`;
}

export class MasterView {
  /** `well`: the element the canvas and its overlay sit in; `render({ projection, rot, zoom,
   *  width, height })` returns the engine's RGBA pixels (or null when there is nothing to show);
   *  `onTurn(rot)` hears of every turn the user makes. */
  constructor(well, render, onTurn = null) {
    this.well = well;
    this.render = render;
    this.onTurn = onTurn;
    // the flat projections' poles stay where they are (a drag does not turn them; control-
    // scroll still spins them about the north pole)
    this.locked = true;
    this.canvas = document.createElement("canvas");
    this.canvas.className = "globe-canvas";
    this.over = document.createElementNS(NS, "svg");
    this.over.setAttribute("class", "over globe-over");
    well.append(this.canvas, this.over);
    this.projection = "globe";
    this.state = Object.fromEntries(PROJECTIONS.map((p) => [p, { rot: FIRST[p].rot.slice(), zoom: FIRST[p].zoom, pan: [0, 0] }]));
    // what is shown of the sphere: the hemispheres (flat projections: 0 both, 1 upper, 2 lower)
    // and whether it is I(d) − I(−d)
    this.shown = 0;
    this.difference = false;
    // the globe as a stereogram: the eyes' separation and the distance to the screen (mm)
    this.stereo = { on: false, dist: 500, ipd: 60, swap: false };
    this.alive = false;
    this.frames = new Frames((rough) => this.frame(rough));
    wireTurning(this.canvas, {
      at: (ev) => (this.isLocked() ? null : this.dirAt(ev)),
      turn: (T) => { const s = this.state[this.projection]; s.rot = orthonormal(mul(s.rot, T)); this.frames.request(true); this.told(); },
      // zoom about the pointer: the point under it stays under it
      zoom: (f, ev) => {
        const s = this.state[this.projection];
        const z = Math.max(0.3, Math.min(this.isStereo() ? 1 : 40, s.zoom * f));
        const k = z / s.zoom;
        // the globe stays where it is, its middle the pane's: zoom is about the middle
        if (this.projection === "globe") { s.zoom = z; this.frames.request(true); return; }
        const r = this.canvas.getBoundingClientRect();
        const m = Math.min(r.width, r.height);
        const px = ev ? (ev.clientX - r.left - r.width / 2) / m : 0, py = ev ? (ev.clientY - r.top - r.height / 2) / m : 0;
        s.pan = [px - (px - s.pan[0]) * k, py - (py - s.pan[1]) * k];
        s.zoom = z;
        this.frames.request(true);
      },
      pan: (dx, dy) => {
        if (this.projection === "globe") return;
        const s = this.state[this.projection];
        const r = this.canvas.getBoundingClientRect();
        const m = Math.min(r.width, r.height);
        s.pan = [s.pan[0] + dx / m, s.pan[1] + dy / m];
        this.frames.request(true);
      },
      reset: () => this.reset(),
      dragging: (on, panning) => { if (on) { this.well.dataset.drag = "1"; this.well.dataset.pan = panning ? "1" : "0"; } else { delete this.well.dataset.drag; delete this.well.dataset.pan; this.frames.request(false); } },
    });
    new ResizeObserver(() => this.refresh()).observe(well);
  }

  /** Draw again (the master changed, or the well did). */
  refresh() { this.frames.request(false); }

  isLocked() { return this.locked && this.projection !== "globe"; }

  setLocked(on) {
    this.locked = !!on;
    this.well.dataset.locked = this.isLocked() ? "1" : "0";
  }

  /** The rotation shown now. */
  get rot() { return this.state[this.projection].rot; }

  /** Take a rotation from elsewhere (a linked view): not in a locked flat projection. */
  setRotation(rot) {
    if (this.isLocked()) return;
    this.state[this.projection].rot = rot.slice();
    this.frames.request(true);
  }

  told() { if (this.onTurn) this.onTurn(this.state[this.projection].rot); }

  /** The hemispheres shown (flat projections: 0 both, 1 upper, 2 lower) and whether the picture
   *  is the upper hemisphere less the lower one's inversion. */
  setShown(shown, difference) {
    this.shown = shown;
    this.difference = !!difference;
    this.refresh();
  }

  /** The globe as a stereogram (or not): two views side by side, as far apart as the eyes
   *  (a screen pixel taken to be 0.265 mm), each turned from the other by the angle the eyes
   *  have on the sphere at `dist` mm. */
  setStereo(on, dist, ipd, swap) {
    this.stereo = { on: !!on, dist, ipd, swap: !!swap };
    this.refresh();
  }

  /** In a stereogram: the two views' places, [{ x, y, half }] (left eye first, as drawn), in
   *  the pane's pixels, and each one's turn about the screen's vertical. */
  stereoLayout(w, h, zoom) {
    const { dist, ipd, swap } = this.stereo;
    const sep = ipd / MM_PER_PX;
    // each view as wide as the separation, no wider (they would overlap); room to be smaller
    const half = Math.min(0.5 * sep * 0.98, 0.5 * Math.min(w / 2, h)) * Math.min(zoom, 1);
    const phi = ipd / dist;
    const L = { x: w / 2 - sep / 2, y: h / 2, half, turn: swap ? phi / 2 : -phi / 2 };
    const Rr = { x: w / 2 + sep / 2, y: h / 2, half, turn: swap ? -phi / 2 : phi / 2 };
    return [L, Rr];
  }

  isStereo() { return this.stereo.on && this.projection === "globe"; }

  setProjection(p) {
    if (!PROJECTIONS.includes(p)) return;
    this.projection = p;
    this.well.dataset.projection = p;
    this.well.dataset.locked = this.isLocked() ? "1" : "0";
    this.refresh();
  }

  /** A new master: back to the whole picture (zoom and pan; the turn is kept). */
  resetLayout() {
    for (const k of PROJECTIONS) { this.state[k].zoom = FIRST[k].zoom; this.state[k].pan = [0, 0]; }
  }

  /** The projection's first view. */
  reset() {
    this.state[this.projection] = { rot: FIRST[this.projection].rot.slice(), zoom: FIRST[this.projection].zoom, pan: [0, 0] };
    this.refresh();
    this.told();
  }

  /** Look along a crystal direction (e.g. [0, 0, 1]: down onto the north pole; in a flat
   *  projection, that direction at the upper panel's centre). */
  lookAlong(d) {
    this.state[this.projection].rot = lookFrom(d);
    this.refresh();
    this.told();
  }

  size() {
    const r = this.well.getBoundingClientRect();
    return { w: r.width, h: r.height };
  }

  /** The view direction under the pointer (or null). */
  dirAt(ev) {
    const r = this.canvas.getBoundingClientRect();
    const x = ev.clientX - r.left, y = ev.clientY - r.top;
    const s = this.state[this.projection];
    // (a stereogram: the nearer of the two globes)
    if (this.isStereo()) {
      const [a, b] = this.stereoLayout(r.width, r.height, s.zoom);
      const p = Math.abs(x - a.x) <= Math.abs(x - b.x) ? a : b;
      return inverse("globe", false, [(x - p.x) / p.half, (p.y - y) / p.half]);
    }
    const ps = panels(this.projection, r.width, r.height, s.zoom, s.pan, this.shown);
    // the upper panel's where the pointer is in it, else the lower's
    let p = ps[0];
    if (ps.length > 1) {
      const q0 = [(x - ps[0].x) / ps[0].half, (ps[0].y - y) / ps[0].half];
      const inside = this.projection === "lambert" ? Math.max(Math.abs(q0[0]), Math.abs(q0[1])) <= 1 : Math.hypot(...q0) <= 1;
      if (!inside) p = ps[1];
    }
    return inverse(this.projection, p.lower, [(x - p.x) / p.half, (p.y - y) / p.half]);
  }

  async frame(rough) {
    const { w, h } = this.size();
    if (!this.well.offsetParent || Math.min(w, h) < 24) return;
    const [pw, ph] = pictureSize(w, h, rough);
    const projection = this.projection;
    const { rot, zoom, pan } = this.state[projection];
    const shown = this.shown, difference = this.difference;
    if (this.isStereo()) return this.frameStereo(w, h, rough, rot, zoom);
    const px = await this.render({ projection, rot: rot.slice(), zoom, pan: pan.slice(), shown, difference, width: pw, height: ph });
    if (px) {
      this.alive = true;
      this.canvas.width = pw;
      this.canvas.height = ph;
      this.canvas.getContext("2d").putImageData(new ImageData(px, pw, ph), 0, 0);
      this.canvas.hidden = false;
      this.drawOverlay(projection, rot, zoom, pan, shown);
    } else {
      this.alive = false;
      this.canvas.hidden = true;
      this.over.replaceChildren();
    }
  }

  /** The globe for two eyes: each eye's view rendered in its own square (as wide as the eyes
   *  are apart), put side by side on one canvas. */
  async frameStereo(w, h, rough, rot, zoom) {
    const [a, b] = this.stereoLayout(w, h, zoom);
    const dpr = Math.min(window.devicePixelRatio || 1, 2) * (rough ? 0.5 : 1);
    // each view's square: its globe's diameter and a little more (the sphere fills 1 / 0.98)
    const side = Math.max(32, Math.round(2 * a.half / 0.98 * dpr));
    const eyes = [a, b];
    const pics = [];
    for (const eye of eyes) {
      const R = mul(rot, [Math.cos(eye.turn), 0, -Math.sin(eye.turn), 0, 1, 0, Math.sin(eye.turn), 0, Math.cos(eye.turn)]);
      pics.push(await this.render({ projection: "globe", rot: R, zoom: 0.98, pan: [0, 0], shown: 0, difference: this.difference, width: side, height: side }));
    }
    if (!pics[0] || !pics[1]) { this.canvas.hidden = true; this.over.replaceChildren(); return; }
    const cw = Math.round(w * dpr), ch = Math.round(h * dpr);
    this.canvas.width = cw;
    this.canvas.height = ch;
    const ctx = this.canvas.getContext("2d");
    ctx.fillStyle = "#0a0a0b";
    ctx.fillRect(0, 0, cw, ch);
    pics.forEach((px, k) => {
      const tmp = new OffscreenCanvas(side, side);
      tmp.getContext("2d").putImageData(new ImageData(px, side, side), 0, 0);
      const e = eyes[k];
      const d = (2 * e.half / 0.98) * dpr;
      ctx.drawImage(tmp, (e.x * dpr) - d / 2, (e.y * dpr) - d / 2, d, d);
    });
    this.canvas.hidden = false;
    const els = [];
    eyes.forEach((e, k) => {
      const R = mul(rot, [Math.cos(e.turn), 0, -Math.sin(e.turn), 0, 1, 0, Math.sin(e.turn), 0, Math.cos(e.turn)]);
      els.push(...this.drawOverlay("globe", R, zoom, [0, 0], 0, e));
    });
    this.over.setAttribute("viewBox", `0 0 ${w} ${h}`);
    this.over.replaceChildren(...els);
  }

  /** The crystal's x, y and z (and, on the flat projections, −x, −y and −z), and its equator. */
  drawOverlay(projection, rot, zoom, pan, shown, eye = null) {
    const { w, h } = this.size();
    if (!eye) this.over.setAttribute("viewBox", `0 0 ${w} ${h}`);
    const ps = eye ? [{ x: eye.x, y: eye.y, half: eye.half, lower: false }] : panels(projection, w, h, zoom, pan, shown);
    // (the one hemisphere on show: a pole on the other falls on this panel's lower, so not drawn)
    const single = ps.length === 1 && ps[0].lower !== undefined && projection !== "globe" && projection !== "joint";
    const els = [];
    const el = (tag, attrs, text) => {
      const e = document.createElementNS(NS, tag);
      // (a label stays inside the well)
      if (tag === "text") { attrs.x = Math.max(10, Math.min(w - 10, attrs.x)); attrs.y = Math.max(12, Math.min(h - 5, attrs.y)); }
      for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, typeof v === "number" ? v.toFixed(1) : v);
      if (text !== undefined) e.textContent = text;
      els.push(e);
      return e;
    };
    const at = (f, s = 1) => { const p = ps[Math.min(f.panel, ps.length - 1)]; return [p.x + f.q[0] * p.half * s, p.y - f.q[1] * p.half * s]; };
    const onShow = (f) => !single || (f.panel === 1) === ps[0].lower;
    // the equator: crystal (cos t, sin t, 0) in the view, in runs that do not jump (between
    // panels, or across a fold) and, on the globe, stay on one side
    const runs = [];
    let run = null, last = null;
    for (let i = 0; i <= 180; i++) {
      const t = (i / 180) * 2 * Math.PI;
      const f = forward(projection, applyT(rot, [Math.cos(t), Math.sin(t), 0]));
      if (!onShow(f)) { run = null; last = null; continue; }
      const xy = at(f);
      const jump = last && (last.f.panel !== f.panel || last.f.near !== f.near || Math.hypot(xy[0] - last.xy[0], xy[1] - last.xy[1]) > 0.3 * ps[Math.min(f.panel, ps.length - 1)].half);
      if (!run || jump) {
        if (run && last && last.f.near !== f.near && last.f.panel === f.panel) run.pts.push(xy);
        run = { near: f.near, pts: [] };
        runs.push(run);
      }
      run.pts.push(xy);
      last = { f, xy };
    }
    for (const r of runs) if (r.pts.length > 1) el("polyline", { points: r.pts.map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" "), class: r.near ? "eq near" : "eq far" });
    for (let k = 0; k < 3; k++) {
      // axis k in the view: row k of R
      const v = [rot[3 * k], rot[3 * k + 1], rot[3 * k + 2]];
      const [name, colour] = AXES[k];
      if (projection === "globe") {
        const p = ps[0];
        const far = v[2] < 0 ? " far" : "";
        if (Math.hypot(v[0], v[1]) < 0.08) {
          // an axis that points at the viewer (or away) shows as a dot at the globe's centre
          el("circle", { cx: p.x, cy: p.y, r: 3, class: `axis-dot${far}`, style: `fill:${colour}` });
          el("text", { x: p.x + 8, y: p.y - 6, class: `axis-label${far}`, style: `fill:${colour}` }, name);
        } else {
          el("line", { x1: p.x + v[0] * p.half, y1: p.y - v[1] * p.half, x2: p.x + v[0] * p.half * 1.22, y2: p.y - v[1] * p.half * 1.22, class: `axis${far}`, style: `stroke:${colour}` });
          el("text", { x: p.x + v[0] * p.half * 1.33, y: p.y - v[1] * p.half * 1.33 + 4, class: `axis-label${far}`, style: `fill:${colour}` }, name);
        }
        continue;
      }
      // the flat projections: each pole where it falls, the negative one hollow
      for (const sgn of [1, -1]) {
        const f = forward(projection, v.map((c) => sgn * c));
        // (on a fold's edge a pole falls in two places, or four: the corners of the joint square)
        if (!onShow(f)) continue;
        const [x, y] = at(f);
        el("circle", { cx: x, cy: y, r: 3.2, class: sgn > 0 ? "pole" : "pole neg", style: sgn > 0 ? `fill:${colour}` : `stroke:${colour}` });
        el("text", { x: x + 6, y: y - 5, class: `axis-label${sgn < 0 ? " neg" : ""}`, style: `fill:${colour}` }, sgn > 0 ? name : `−${name}`);
      }
    }
    // what is at each panel's centre
    if (projection !== "globe") {
      const c = [rot[2], rot[5], rot[8]];
      ps.forEach((p, i) => {
        const low = ps.length === 1 ? p.lower : i === 1;
        const text = projection === "joint" ? `${dirName(c)} at the centre · ${dirName(c.map((v) => -v))} at the corners` : `${low ? "lower" : "upper"} · ${dirName(low ? c.map((v) => -v) : c)} at the centre`;
        el("text", { x: p.x, y: Math.min(h - 6, p.y + p.half + 15), class: "panel-label" }, text);
      });
    }
    if (eye) return els;
    this.over.replaceChildren(...els);
  }
}
