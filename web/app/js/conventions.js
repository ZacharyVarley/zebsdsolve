// The conventions, drawn: for each, the lab (the beam, the tilted sample with its axes, the
// detector and its pattern centre) and the pattern as stored (where its pattern centre is
// measured from, and against what), with its terms beside them. One drawing for every
// convention, from what it is (the engine's axes it takes, the way its angles rotate, its
// crystal axes, its pattern centre); the scan's own, as identified, among them.
//
// What is written here of each maker is what was measured on its files (docs/conventions.md),
// or how the engine's reader takes its pattern centre; what was not measured says so.

const SVGNS = "http://www.w3.org/2000/svg";
const el = (tag, attrs = {}, parent = null, text = null) => {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text != null) e.textContent = text;
  if (parent) parent.append(e);
  return e;
};
const add = (a, b) => a.map((v, i) => v + b[i]);
const mul = (a, s) => a.map((v) => v * s);

// how each convention measures the pattern centre on the pattern as stored (row 0 at the top,
// column 0 at the left): from which corner, which way, as a fraction of what
const PC = {
  bruker: { origin: "tl", x: "from left ÷ width", y: "from top ÷ height", z: "÷ height" },
  bruker_clipped: { origin: "tl", x: "from left ÷ width", y: "from top ÷ height before clipping", z: "÷ stored height" },
  oxford: { origin: "bl", x: "from left ÷ width", y: "from bottom ÷ width", z: "÷ width" },
  edax: { origin: "bl", x: "from left ÷ width", y: "from bottom ÷ height", z: "÷ shorter side" },
  emsoft: { origin: "c", x: "px from centre, leftward", y: "px from centre, up", z: "L (µm) ÷ pixel size" },
};

const AXES = "xyz";
/** "X +y · Y −x · Z +z": a convention's sample axes as the engine's. */
const axesText = (axes) => axes.map((a, i) => `${"XYZ"[i]} ${a < 0 ? "−" : "+"}${AXES[Math.abs(a) - 1]}`).join(" · ");
const CRYSTAL = { x_a_z_cstar: "X ∥ a, Z ∥ c*", x_astar_z_c: "X ∥ a*, Z ∥ c", y_b_z_cstar: "Y ∥ b, Z ∥ c*", y_bstar_z_c: "Y ∥ b*, Z ∥ c" };

/** The conventions known by measurement: the engine's, and three makers' as their files show. */
export const CONVENTIONS = [
  {
    key: "engine", name: "kikuchipy", sub: "the engine's frame",
    axes: [1, 2, 3], active: false, hexagonal: "x_a_z_cstar", pc: PC.bruker,
    detector: "sample and detector tilts", beam: "x and y, by the scan's steps",
    measured: "kikuchipy's refined nickel: median 0.32°, misorientation 0.08°",
  },
  {
    key: "oxford", name: "Oxford AZtec", sub: ".h5oina, .ctf",
    axes: [2, -1, 3], active: false, hexagonal: "x_astar_z_c", pc: PC.oxford,
    detector: "Detector Orientation Euler, from (0, 90°, 0)", beam: "x and y (stored per pattern)",
    measured: "Zr .h5oina: median 0.20°, misorientation 0.10° (3.4° without its detector orientation)",
  },
  {
    key: "bruker", name: "Bruker Esprit", sub: "h5ebsd",
    axes: [2, -1, 3], active: false, hexagonal: null, pc: PC.bruker_clipped,
    detector: "sample and camera tilts", beam: "x and y (stored per pattern)",
    measured: "ferrite h5ebsd: median 1.16°, misorientation 1.02°",
  },
  {
    key: "edax", name: "EDAX OIM", sub: "h5ebsd, .ang, .up1/.up2",
    axes: [1, 2, 3], active: false, hexagonal: null, pc: PC.edax,
    detector: "sample tilt and camera elevation", beam: "x only: y* and z* held as the beam moves down the rows",
    measured: "superalloy h5ebsd: median 0.76°, misorientation 0.34°, changing 1.96° from the first row to the last",
  },
  {
    key: "emsoft", name: "EMsoft", sub: "simulated patterns",
    axes: null, active: null, hexagonal: "x_a_z_cstar", pc: PC.emsoft,
    detector: "sample and camera tilts", beam: "—", measured: "—",
  },
];

/** The pattern-centre convention the engine reads a scan's format in. */
function pcOfFormat(format) {
  if (/^Oxford/.test(format)) return PC.oxford;
  if (/^EDAX/.test(format)) return PC.edax;
  if (/^EMsoft/.test(format)) return PC.emsoft;
  if (/^Bruker/.test(format)) return PC.bruker_clipped;
  if (/^kikuchipy/.test(format)) return PC.bruker;
  return null;
}

/** The lab: the engine's sample axes and the detector's (geometry.zig), the sample tilted
 *  about the lab's y toward the detector on the right. */
function lab(sampleTilt, detectorTilt) {
  const th = (sampleTilt * Math.PI) / 180;
  const xs = [Math.cos(th), 0, -Math.sin(th)], ys = [0, 1, 0], zs = [Math.sin(th), 0, Math.cos(th)];
  const toLab = (v) => add(add(mul(xs, v[0]), mul(ys, v[1])), mul(zs, v[2]));
  const a = ((detectorTilt - sampleTilt) * Math.PI) / 180, s = Math.sin(a), c = Math.cos(a);
  // the columns of R0: the detector's row direction, column direction and normal
  return { xs, ys, zs, e1: toLab([-s, 0, -c]), e2: toLab([0, 1, 0]), e3: toLab([c, 0, -s]) };
}

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const unit = (a) => mul(a, 1 / Math.hypot(...a));

function drawLab(svg, conv, g) {
  svg.replaceChildren();
  const W = 420, H = 300, pad = 26;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const L = lab(g.sampleTilt, g.detectorTilt);
  // seen from in front of the sample's surface, to the side and above: orthographic
  const c = unit([0.42, -0.82, 0.4]);
  const up = unit(add([0, 0, 1], mul(c, -c[2])));
  const right = cross(up, c);
  const proj = (v) => [dot(v, right), -dot(v, up)];
  const O = [0, 0, 0], hx = 1.1, hy = 1.35;
  const corner = (u, w) => add(add(O, mul(L.xs, u * hx)), mul(L.ys, w * hy));
  const pc = g.pc, Hd = 1.9, Wd = Hd * g.aspect, DD = 2.6;
  const PCp = add(O, mul(L.e3, DD));
  const c00 = add(add(PCp, mul(L.e1, -pc[1] * Hd)), mul(L.e2, -pc[0] * Wd));
  const dc = (r, q) => add(add(c00, mul(L.e1, r * Hd)), mul(L.e2, q * Wd));
  const B = [0, 0, 2.3];
  // the map's first cell, as the engine reads the scan (its signs), and the sample's axes at
  // another corner (the up-slope near one, unless the map starts there)
  const m0uw = g.signs ? [-g.signs[1], -g.signs[0]] : null;
  const [au, aw] = [[-1, -1], [-1, 1], [1, -1]].find(([u, w]) => !m0uw || u !== m0uw[0] || w !== m0uw[1]);
  const A = corner(au, aw), len = 0.9;
  const engine = [L.xs, L.ys, L.zs];
  // fitted to the frame
  const ext = [corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1), dc(0, 0), dc(0, 1), dc(1, 0), dc(1, 1), B, ...engine.flatMap((e) => [add(A, mul(e, len + 0.35)), add(A, mul(e, -0.2))])].map(proj);
  const x0 = Math.min(...ext.map((p) => p[0])), x1 = Math.max(...ext.map((p) => p[0]));
  const y0 = Math.min(...ext.map((p) => p[1])), y1 = Math.max(...ext.map((p) => p[1]));
  const s = Math.min((W - 2 * pad) / (x1 - x0), (H - 2 * pad) / (y1 - y0));
  const ox = (W - s * (x1 - x0)) / 2 - s * x0, oy = (H - s * (y1 - y0)) / 2 - s * y0;
  const P = (v) => { const [x, y] = proj(v); return [ox + s * x, oy + s * y]; };
  const f = (n) => n.toFixed(1);
  const pts = (...vs) => vs.map((v) => P(v).map(f).join(",")).join(" ");
  // every line drawn, on the page, for the labels to keep off
  const segs = [];
  const line = (a, b, cls, extra = {}) => {
    const [xa, ya] = P(a), [xb, yb] = P(b);
    segs.push([xa, ya, xb, yb]);
    return el("line", { x1: f(xa), y1: f(ya), x2: f(xb), y2: f(yb), class: cls, ...extra }, svg);
  };
  const polygon = (cls, ...vs) => {
    const q = vs.map(P);
    q.forEach((a, i) => { const b = q[(i + 1) % q.length]; segs.push([a[0], a[1], b[0], b[1], 0.25]); });
    el("polygon", { points: pts(...vs), class: cls }, svg);
  };
  const labels = [];
  // A label near a point: of the places around it (beyond the arrow it names first, when it
  // names one), the first that covers no line, no other label and nothing outside the frame;
  // else the one that covers least.
  const label = (at, text, cls, { from = null, gap = 10 } = {}) => {
    const [x, y] = P(at);
    const size = cls === "cv-note" ? 5.6 : /\bmap\b/.test(cls) ? 6.8 : 7.4;
    const w = text.length * size + 4, h = 13;
    const dirs = [];
    if (from) {
      const [xa, ya] = P(from);
      const d = Math.hypot(x - xa, y - ya) || 1;
      dirs.push([(x - xa) / d, (y - ya) / d]);
    }
    for (let k = 0; k < 16; k++) dirs.push([Math.cos((k * Math.PI) / 8), Math.sin((k * Math.PI) / 8)]);
    let best = null;
    for (const r of [gap, gap + 7, gap + 15, gap + 26]) for (const [dx, dy] of dirs) {
      // the box's centre where its edge, not its centre, keeps the distance
      const cx = x + dx * (r + Math.abs(dx) * w / 2), cy = y + dy * (r + Math.abs(dy) * h / 2);
      const box = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2];
      let cost = 0;
      if (box[0] < 2 || box[1] < 2 || box[2] > W - 2 || box[3] > H - 2) cost += 5;
      for (const b of labels) if (box[0] < b[2] && box[2] > b[0] && box[1] < b[3] && box[3] > b[1]) cost += 3;
      for (const s of segs) if (crosses(s, box)) cost += s[4] ?? 1;
      cost += r / 200;
      if (!best || cost < best.cost - 1e-9) best = { cost, cx, cy, box };
      if (cost < 0.2) break;
    }
    labels.push(best.box);
    el("text", { x: f(best.cx), y: f(best.cy + 4), "text-anchor": "middle", class: cls }, svg, text);
  };
  const defs = el("defs", {}, svg);
  for (const [id, cls] of [["cv-arrow", "cv-origin"], ["cv-arrow-map", "cv-pc"], ["cv-arrow-x", "cv-label x"], ["cv-arrow-y", "cv-label y"], ["cv-arrow-z", "cv-label z"]]) {
    const m = el("marker", { id, viewBox: "0 0 10 10", refX: 8, refY: 5, markerWidth: 6, markerHeight: 6, orient: "auto-start-reverse" }, defs);
    el("path", { d: "M0 1 L9 5 L0 9 z", class: cls, style: "stroke:none" }, m);
  }

  // everything drawn first, the labels after, each kept off what is there
  // the detector (behind, as seen from here), the sample, the beam
  polygon("cv-detector", dc(0, 0), dc(0, 1), dc(1, 1), dc(1, 0));
  polygon("cv-sample", corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1));
  line(B, O, "cv-beam");
  // the distance to the pattern centre, and the pattern's first pixel, rows and columns
  line(O, PCp, "cv-dd");
  const [px, py] = P(PCp);
  el("circle", { cx: f(px), cy: f(py), r: 3.4, class: "cv-pc" }, svg);
  const [fx, fy] = P(dc(0, 0));
  el("circle", { cx: f(fx), cy: f(fy), r: 2.6, class: "cv-origin" }, svg);
  line(dc(0, 0), dc(0.28, 0), "cv-measure", { "marker-end": "url(#cv-arrow)" });
  line(dc(0, 0), dc(0, 0.28), "cv-measure", { "marker-end": "url(#cv-arrow)" });
  // the map's rows and columns on the sample, as the engine reads the scan (its signs)
  let mapRow = null, mapCol = null;
  if (g.signs) {
    const [sx, sy] = g.signs;
    const m0 = corner(...m0uw);
    mapRow = [m0, add(m0, mul(L.xs, sy * 0.7))];
    mapCol = [m0, add(m0, mul(L.ys, sx * 0.8))];
    line(...mapRow, "cv-map", { "marker-end": "url(#cv-arrow-map)" });
    line(...mapCol, "cv-map", { "marker-end": "url(#cv-arrow-map)" });
  }
  // the convention's sample axes at the sample's corner, over the engine's (dashed, lower case)
  for (let i = 0; i < 3; i++) line(A, add(A, mul(engine[i], len)), "cv-ref");
  const axes = conv.axes ? conv.axes.map((code) => mul(engine[Math.abs(code) - 1], Math.sign(code))) : [];
  axes.forEach((v, i) => line(A, add(A, mul(v, len)), `cv-axis ${AXES[i]}`, { "marker-end": `url(#cv-arrow-${AXES[i]})` }));
  // the labels, the convention's axes first
  axes.forEach((v, i) => label(add(A, mul(v, len)), "XYZ"[i], `cv-label ${AXES[i]}`, { from: A, gap: 8 }));
  // (the engine's own name only where no axis of the convention lies along it)
  for (let i = 0; i < 3; i++) if (!conv.axes?.includes(i + 1)) label(add(A, mul(engine[i], len)), AXES[i], "cv-label ref", { from: A, gap: 5 });
  if (mapRow) {
    label(mapRow[1], "map row", "cv-label map", { from: mapRow[0], gap: 8 });
    label(mapCol[1], "map col", "cv-label map", { from: mapCol[0], gap: 8 });
  }
  label(PCp, "PC", "cv-note", { gap: 6 });
  label(B, "beam", "cv-note", { gap: 6 });
  label(dc(0.28, 0), "row", "cv-label ref", { from: dc(0, 0), gap: 6 });
  label(dc(0, 0.28), "col", "cv-label ref", { from: dc(0, 0), gap: 6 });
}

/** Whether segment s ([x1, y1, x2, y2]) touches box b ([x0, y0, x1, y1]). */
function crosses(s, b) {
  const [x1, y1, x2, y2] = s;
  if (Math.max(x1, x2) < b[0] || Math.min(x1, x2) > b[2] || Math.max(y1, y2) < b[1] || Math.min(y1, y2) > b[3]) return false;
  const inside = (x, y) => x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3];
  if (inside(x1, y1) || inside(x2, y2)) return true;
  // the box's corners on both sides of the segment's line: it passes through
  const side = (x, y) => Math.sign((x2 - x1) * (y - y1) - (y2 - y1) * (x - x1));
  const s0 = [side(b[0], b[1]), side(b[2], b[1]), side(b[0], b[3]), side(b[2], b[3])];
  return Math.min(...s0) < 0 && Math.max(...s0) > 0;
}

function drawPattern(svg, conv, g) {
  svg.replaceChildren();
  const pad = 26, h = 150, w = Math.round(h * g.aspect);
  const W = w + 2 * pad, H = h + 2 * pad + 8;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  el("rect", { x: pad, y: pad, width: w, height: h, class: "cv-detector" }, svg);
  el("text", { x: pad, y: pad - 6, class: "cv-note" }, svg, "row 0, col 0");
  const pcx = pad + g.pc[0] * w, pcy = pad + g.pc[1] * h;
  const p = conv.pc;
  const defs = el("defs", {}, svg);
  const m = el("marker", { id: "cv-parrow", viewBox: "0 0 10 10", refX: 8, refY: 5, markerWidth: 6, markerHeight: 6, orient: "auto-start-reverse" }, defs);
  el("path", { d: "M0 1 L9 5 L0 9 z", class: "cv-origin" }, m);
  const arrow = (x1, y1, x2, y2) => el("line", { x1, y1, x2, y2, class: "cv-measure", "marker-end": "url(#cv-parrow)" }, svg);
  const text = (x, y, s, anchor = "middle") => el("text", { x, y, "text-anchor": anchor, class: "cv-label ref" }, svg, s);
  if (p) {
    if (p.origin === "c") {
      const cx = pad + w / 2, cy = pad + h / 2;
      el("circle", { cx, cy, r: 3, class: "cv-origin" }, svg);
      arrow(cx, cy, cx - 0.32 * w, cy);
      text(cx - 0.16 * w, cy - 6, "x");
      arrow(cx, cy, cx, cy - 0.32 * h);
      text(cx + 8, cy - 0.16 * h, "y", "start");
    } else {
      const ox = pad, oy = p.origin === "tl" ? pad : pad + h;
      el("circle", { cx: ox, cy: oy, r: 3.2, class: "cv-origin" }, svg);
      const ay = p.origin === "tl" ? pad + 10 : pad + h - 10;
      arrow(ox, ay, pcx, ay);
      text((ox + pcx) / 2, p.origin === "tl" ? ay + 14 : ay - 5, "x");
      const ax = pad + 10;
      arrow(ax, oy, ax, pcy);
      text(ax + 6, (oy + pcy) / 2 + 4, "y", "start");
      el("line", { x1: ax, y1: pcy, x2: pcx, y2: pcy, class: "cv-ref" }, svg);
      el("line", { x1: pcx, y1: ay, x2: pcx, y2: pcy, class: "cv-ref" }, svg);
    }
  }
  el("circle", { cx: pcx, cy: pcy, r: 3.6, class: "cv-pc" }, svg);
  el("text", { x: pcx + 7, y: pcy + 4, class: "cv-note" }, svg, "PC · z: distance");
}

/**
 * The viewer: `current()` gives what the scan in hand says ({ format, found, sampleTilt,
 * detectorTilt, pc, detector, signs, shape: [height, width] }), or null with none open.
 */
export function createConventions({ dialog, list, lab: labSvg, pattern, kv, title, current }) {
  let picked = "engine";

  function entries() {
    const out = CONVENTIONS.map((c) => ({ ...c }));
    const cur = current();
    if (cur) {
      const f = cur.found?.frame || null;
      const m = cur.found ? (180 / Math.PI) * Math.hypot(...cur.found.misorientation) : null;
      const mr = cur.found ? (180 / Math.PI) * Math.hypot(...cur.found.misorientation_rows) : null;
      out.unshift({
        key: "scan", name: "this scan", sub: cur.format,
        axes: f ? f.axes : null, active: f ? f.active : null, hexagonal: f ? f.crystal : null,
        pc: pcOfFormat(cur.format), detector: cur.detector, beam: "x and y, by the scan's steps",
        measured: cur.found ? `median ${cur.found.median_deg.toFixed(2)}°, misorientation ${m.toFixed(2)}°, changing ${mr.toFixed(2)}° from the first row to the last; next convention ${cur.found.runner_up_deg.toFixed(1)}°` : "not identified: index, with a file that stores orientations",
        scan: cur,
      });
    }
    return out;
  }

  function show() {
    const all = entries();
    if (!all.some((e) => e.key === picked)) picked = all[0].key;
    list.replaceChildren(...all.map((e) => {
      const b = document.createElement("button");
      b.type = "button";
      b.setAttribute("role", "option");
      b.setAttribute("aria-selected", String(e.key === picked));
      b.dataset.key = e.key;
      b.innerHTML = "<span></span><small></small>";
      b.firstChild.textContent = e.name;
      b.lastChild.textContent = e.sub;
      b.addEventListener("click", () => { picked = e.key; show(); });
      return b;
    }));
    const c = all.find((e) => e.key === picked);
    const s = c.scan;
    const g = {
      sampleTilt: s ? s.sampleTilt : 70,
      detectorTilt: s ? s.detectorTilt : 0,
      pc: s ? s.pc : [0.5, 0.3, 0.6],
      aspect: s?.shape ? s.shape[1] / s.shape[0] : 4 / 3,
      signs: s ? s.signs : c.key === "engine" ? [-1, -1] : null,
    };
    drawLab(labSvg, c, g);
    drawPattern(pattern, c, g);
    title.textContent = `${c.name} · ${c.sub}`;
    const rows = [
      ["sample axes", c.axes ? `${axesText(c.axes)}  (of the engine's: x toward the detector, y the tilt axis, z out of the surface)` : "not measured"],
      ["angles", c.active == null ? "not measured" : c.active ? "active: crystal to sample" : "passive (Bunge): sample to crystal"],
      ["hexagonal", c.hexagonal ? CRYSTAL[c.hexagonal] : "not measured"],
      ["PC x", c.pc ? c.pc.x : "not stored"],
      ["PC y", c.pc ? c.pc.y : "not stored"],
      ["PC z", c.pc ? c.pc.z : "not stored"],
      ["detector", c.detector || "—"],
      ["PC follows beam", c.beam || "—"],
      [c.scan ? "against the file" : "measured", c.measured || "—"],
      ["tilts", `sample ${g.sampleTilt.toFixed(1)}°, detector ${g.detectorTilt.toFixed(1)}°${s ? "" : " (as drawn)"}`],
    ];
    kv.replaceChildren(...rows.flatMap(([k, v]) => {
      const dt = document.createElement("dt"); dt.textContent = k;
      const dd = document.createElement("dd"); dd.textContent = v;
      if (/^(not |—)/.test(v)) dd.className = "none";
      return [dt, dd];
    }));
  }

  return {
    // (the scan's own first, when there is one)
    open() { if (current()) picked = "scan"; show(); if (!dialog.open) dialog.showModal(); },
    refresh() { if (dialog.open) show(); },
  };
}
