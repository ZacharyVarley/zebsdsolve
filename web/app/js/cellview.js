// The crystal's unit cell in a picture: the engine renders it (cellview.wgsl); this turns the
// mouse into the view's rotation and zoom and names what is drawn: the lattice vectors a, b and
// c along the edges from the cell's origin, the angles between them, and the elements.

import { lookFrom, mul, applyT, orthonormal, arcball, Frames, pictureSize, wireTurning } from "./turn.js";

const NS = "http://www.w3.org/2000/svg";
// a, b, c in the colours of the master view's x, y, z (the rods from the origin are drawn so)
const EDGES = [["a", "#e0705e"], ["b", "#7fbf6e"], ["c", "#6a9ec8"]];
// each angle: its name, the two lattice vectors it is between, its index in the cell
const ANGLES = [["α", 1, 2, "alpha"], ["β", 0, 2, "beta"], ["γ", 0, 1, "gamma"]];
const ZOOM = 0.92;
const fmt = (v, d) => (Math.abs(v - Math.round(v)) < 1e-9 ? String(Math.round(v)) : v.toFixed(d));

export class CellView {
  /** `well`: the element the picture sits in; `render({ rot, zoom, highlight, width, height })`
   *  returns the engine's RGBA pixels (or null when there is no crystal); `onTurn(rot)` hears of
   *  every turn the user makes. */
  constructor(well, render, onTurn = null) {
    this.well = well;
    this.render = render;
    this.onTurn = onTurn;
    this.canvas = document.createElement("canvas");
    this.canvas.className = "globe-canvas";
    this.over = document.createElementNS(NS, "svg");
    this.over.setAttribute("class", "over cell-over");
    this.legend = document.createElement("div");
    this.legend.className = "cell-legend";
    well.append(this.canvas, this.over, this.legend);
    this.info = null;
    this.cells = 1;   // cells along each lattice vector: 1, or 3 for a 3 × 3 × 3 supercell
    this.cell = null;
    this.rot = null;
    this.zoom = ZOOM;
    this.highlight = null;
    this.frames = new Frames((rough) => this.frame(rough));
    wireTurning(this.canvas, {
      at: (ev) => {
        const r = this.canvas.getBoundingClientRect();
        const rad = 0.5 * Math.min(r.width, r.height) * this.zoom;
        return arcball((ev.clientX - r.left - r.width / 2) / rad, (r.top + r.height / 2 - ev.clientY) / rad);
      },
      turn: (T) => { if (this.rot) { this.rot = orthonormal(mul(this.rot, T)); this.frames.request(true); this.told(); } },
      zoom: (f) => { this.zoom = Math.max(0.3, Math.min(10, this.zoom * f)); this.frames.request(true); },
      reset: () => this.reset(),
      dragging: (on) => { if (on) this.well.dataset.drag = "1"; else { delete this.well.dataset.drag; this.frames.request(false); } },
    });
    new ResizeObserver(() => this.refresh()).observe(well);
  }

  /** A new crystal: `info` the engine's account of its cell as drawn (cellInfo), `cell` its
   *  lattice parameters (Å, degrees), or nulls. */
  show(info, cell) {
    this.info = info;
    this.cell = cell;
    if (info && !this.rot) this.reset();
    else this.refresh();
    this.legend.replaceChildren();
    if (!info) return;
    // each element once
    const seen = new Map();
    for (const s of info.sites) if (!seen.has(s.symbol)) seen.set(s.symbol, s.colour);
    for (const [sym, col] of seen) {
      const chip = document.createElement("span");
      chip.innerHTML = `<i style="background:${col}"></i>`;
      chip.append(sym);
      this.legend.append(chip);
    }
  }

  refresh() { this.frames.request(false); }

  told() { if (this.onTurn && this.rot) this.onTurn(this.rot); }

  /** Take a rotation from elsewhere (a linked view). */
  setRotation(rot) {
    if (!this.info) return;
    this.rot = rot.slice();
    this.frames.request(true);
  }

  /** The first view: from in front, a little to the right and above, c up. */
  reset() {
    this.zoom = ZOOM;
    this.rot = this.info ? this.lookFromCell([2.6, 1.1, 0.95]) : null;
    this.refresh();
    this.told();
  }

  /** R for a viewer at a direction given in lattice components, c up on the screen. */
  lookFromCell(f) {
    const B = this.info.basis;
    const d = [0, 1, 2].map((i) => f[0] * B[0][i] + f[1] * B[1][i] + f[2] * B[2][i]);
    return lookFrom(d, B[2]);
  }

  /** Look along lattice vector k (0 a, 1 b, 2 c), toward the cell from its tip. */
  lookAlong(k) {
    if (!this.info) return;
    // (looking along c, b points up)
    const B = this.info.basis;
    this.rot = lookFrom(B[k], k === 2 ? B[1] : B[2]);
    this.zoom = ZOOM;
    this.refresh();
    this.told();
  }

  /** Pick out a lattice parameter ("a" … "gamma"), a site (its index), or nothing (null). */
  pick(what) {
    if (this.highlight === what) return;
    this.highlight = what;
    this.refresh();
  }

  async frame(rough) {
    const r = this.well.getBoundingClientRect();
    if (!this.well.offsetParent || Math.min(r.width, r.height) < 24) return;
    if (!this.info || !this.rot) {
      this.canvas.hidden = true;
      this.over.replaceChildren();
      return;
    }
    const [pw, ph] = pictureSize(r.width, r.height, rough);
    const rot = this.rot.slice(), zoom = this.zoom, highlight = this.highlight;
    const px = await this.render({ rot, zoom, highlight, cells: this.cells, width: pw, height: ph });
    if (!px) { this.canvas.hidden = true; this.over.replaceChildren(); return; }
    this.canvas.width = pw;
    this.canvas.height = ph;
    this.canvas.getContext("2d").putImageData(new ImageData(px, pw, ph), 0, 0);
    this.canvas.hidden = false;
    this.drawOverlay(r.width, r.height, rot, zoom, highlight);
  }

  /** a, b and c at the tips of the edges from the origin, and the angles between them. */
  drawOverlay(w, h, rot, zoom, highlight) {
    const { basis: B, centre: C, radius } = this.info;
    const scale = (0.5 * Math.min(w, h) * zoom) / radius;
    // a point of the cell (Å, Cartesian) on the screen, as the engine draws it
    const at = (p) => { const v = applyT(rot, [p[0] - C[0], p[1] - C[1], p[2] - C[2]]); return [w / 2 + v[0] * scale, h / 2 - v[1] * scale]; };
    const o = at([0, 0, 0]);
    const els = [];
    const el = (tag, attrs, text) => {
      const e = document.createElementNS(NS, tag);
      // (a label stays inside the well)
      if (tag === "text") { attrs.x = Math.max(16, Math.min(w - 16, attrs.x)); attrs.y = Math.max(13, Math.min(h - 5, attrs.y)); }
      for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, typeof v === "number" ? v.toFixed(1) : v);
      if (text !== undefined) e.textContent = text;
      els.push(e);
      return e;
    };
    const len = (v) => Math.hypot(...v);
    EDGES.forEach(([name, colour], k) => {
      const tip = at(B[k]);
      const dx = tip[0] - o[0], dy = tip[1] - o[1];
      const n = Math.max(Math.hypot(dx, dy), 1e-6);
      // past the tip, along the edge
      const x = tip[0] + (dx / n) * 14, y = tip[1] + (dy / n) * 14 + 4;
      const on = highlight === name;
      el("text", { x, y, class: `edge-label${on ? " on" : ""}`, style: on ? "" : `fill:${colour}` }, on && this.cell ? `${name} = ${fmt(this.cell[k], 4)} Å` : name);
    });
    for (const [name, i, j, key] of ANGLES) {
      const u = B[i], v = B[j];
      const lu = len(u), lv = len(v);
      const uh = u.map((c) => c / lu);
      const dot = (v[0] * uh[0] + v[1] * uh[1] + v[2] * uh[2]);
      const wv = v.map((c, m) => c - dot * uh[m]);
      const lw = len(wv);
      if (lw < 1e-9) continue;
      const wh = wv.map((c) => c / lw);
      const theta = Math.atan2(lw, dot);
      const r = 0.24 * Math.min(lu, lv);
      const pts = [];
      for (let s = 0; s <= 16; s++) {
        const t = (s / 16) * theta;
        pts.push(at([0, 1, 2].map((m) => r * (Math.cos(t) * uh[m] + Math.sin(t) * wh[m]))));
      }
      const on = highlight === key;
      el("polyline", { points: pts.map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" "), class: `angle${on ? " on" : ""}` });
      const mid = at([0, 1, 2].map((m) => 1.45 * r * (Math.cos(theta / 2) * uh[m] + Math.sin(theta / 2) * wh[m])));
      el("text", { x: mid[0], y: mid[1] + 4, class: `angle-label${on ? " on" : ""}` }, on && this.cell ? `${name} = ${fmt(this.cell[3 + ANGLES.findIndex((a) => a[3] === key)], 2)}°` : name);
    }
    this.over.setAttribute("viewBox", `0 0 ${w} ${h}`);
    this.over.replaceChildren(...els);
  }
}
