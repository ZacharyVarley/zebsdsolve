// The pattern-centre search, laid open. Every grid it scored is a small volume of mean votes
// over (x, y, z); the view cuts it three ways through the cross-hair, or lays every slice along
// one axis side by side, with what led the search (each grid's winner and the next grid's
// extent), what the fit patterns each preferred on their own, and the profile of the vote along
// each axis. The cross-hair starts on the pattern centre and returns to it whenever the pattern
// centre changes (a fit, a typed value), so the cuts always go through the geometry in use
// until the cross-hair is moved by hand.
import { fit, svg, viridis } from "./draw.js";

const $ = (id) => document.getElementById(id);
const NAMES = ["x", "y", "z"];
// the three planes: [canvas id, horizontal axis, vertical axis, the axis held fixed]
const PLANES = [["votes", 0, 1, 2], ["votes-xz", 0, 2, 1], ["votes-yz", 1, 2, 0]];
// every slice along an axis: [horizontal axis, vertical axis, the axis the slices step along]
const ALONG = { z: [0, 1, 2], y: [0, 2, 1], x: [1, 2, 0] };

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
// value → position in cell units along an axis of candidates (cell centres at k + 0.5)
const pos = (v, a) => (a.length > 1 ? ((v - a[0]) / (a[a.length - 1] - a[0])) * (a.length - 1) + 0.5 : 0.5);
// the candidate nearest a value
const nearest = (v, a) => clamp(Math.round(pos(v, a) - 0.5), 0, a.length - 1);
// how far a value lies beyond an axis's candidates (their own cells included); 0 within
const beyond = (v, a) => {
  if (a.length < 2) return 0;
  const half = (a[1] - a[0]) / 2;
  return v < a[0] - half ? v - (a[0] - half) : v > a[a.length - 1] + half ? v - (a[a.length - 1] + half) : 0;
};

export function createVotes({ geometry, setPc }) {
  const view = { levels: null, level: 0, at: [0, 0, 0], start: null, file: null, mode: "planes", what: "ballots", following: true, tiles: null };

  const axesOf = (l) => [l.x, l.y, l.z];
  const sizes = (l) => [l.x.length, l.y.length, l.z.length];
  const scoreAt = (l, i) => { const [nx, ny] = sizes(l); return l.scores[i[0] + nx * (i[1] + ny * i[2])]; };
  // the volume in view: the ballots' density where ballots decided and are asked for, else the mean scores
  const shown = (l) => (view.what === "ballots" && l.density ? l.density : l.scores);
  const shownAt = (l, i) => { const [nx, ny] = sizes(l); return shown(l)[i[0] + nx * (i[1] + ny * i[2])]; };
  const split = (l, c) => { const [nx, ny] = sizes(l); return [c % nx, Math.floor(c / nx) % ny, Math.floor(c / (nx * ny))]; };
  const range = (v) => { let lo = Infinity, hi = -Infinity; for (const x of v) { if (x < lo) lo = x; if (x > hi) hi = x; } return [lo, hi]; };

  /** The finest grid that holds a pattern centre; failing that, the one it lies least outside. */
  function locate(pc) {
    let best = 0, least = Infinity;
    for (let i = view.levels.length - 1; i >= 0; i--) {
      const out = Math.max(...axesOf(view.levels[i]).map((a, k) => Math.abs(beyond(pc[k], a))));
      if (out === 0) return i;
      if (out < least) { least = out; best = i; }
    }
    return best;
  }

  /** Cut through the pattern centre as it is now. */
  function follow() {
    if (!view.levels) return draw();
    const pc = geometry().pc;
    view.level = locate(pc);
    view.at = axesOf(view.levels[view.level]).map((a, k) => nearest(pc[k], a));
    view.following = true;
    draw();
  }

  /** Another grid, cut where this one is (or through the pattern centre while following it). */
  function select(level) {
    const here = view.following ? geometry().pc : view.at.map((i, k) => axesOf(view.levels[view.level])[k][i]);
    view.level = clamp(level, 0, view.levels.length - 1);
    view.at = axesOf(view.levels[view.level]).map((a, k) => nearest(here[k], a));
    draw();
  }

  /** Move the cross-hair along some axes: { axis: index }. From then on it is the user's. */
  function move(to) {
    if (!view.levels) return;
    const n = sizes(view.levels[view.level]);
    for (const [k, i] of Object.entries(to)) view.at[k] = clamp(Math.round(i), 0, n[k] - 1);
    view.following = false;
    draw();
  }

  /** Paint cells [h0, h0 + nh) × [v0, v0 + nv) of an image from a slice of the grid. */
  function paint(img, W, l, h, v, f, k, ox, oy, lo, hi) {
    const n = sizes(l);
    const i = [0, 0, 0];
    i[f] = k;
    const s = 1 / Math.max(hi - lo, 1e-20);
    for (let b = 0; b < n[v]; b++) for (let a = 0; a < n[h]; a++) {
      i[h] = a; i[v] = b;
      const c = viridis((shownAt(l, i) - lo) * s);
      const p = 4 * ((oy + b) * W + ox + a);
      img.data[p] = c[0]; img.data[p + 1] = c[1]; img.data[p + 2] = c[2]; img.data[p + 3] = 255;
    }
  }

  /** What is marked on a slice k along f, drawn at (ox, oy) in cells; `flat`: one slice stands
   *  for the whole depth (a plane), so everything is projected onto it. u: a screen pixel. */
  function marks(over, l, next, h, v, f, k, ox, oy, u, flat) {
    const ax = axesOf(l), n = sizes(l);
    const win = split(l, l.winner);
    const g = geometry();
    // where the next grid looked
    if (next) {
      const na = axesOf(next);
      const half = (a) => (a.length > 1 ? (a[1] - a[0]) / 2 : 0);
      const within = flat || (ax[f][k] >= na[f][0] - half(na[f]) - 1e-12 && ax[f][k] <= na[f][na[f].length - 1] + half(na[f]) + 1e-12);
      const x0 = clamp(pos(na[h][0] - half(na[h]), ax[h]), 0, n[h]), x1 = clamp(pos(na[h][na[h].length - 1] + half(na[h]), ax[h]), 0, n[h]);
      const y0 = clamp(pos(na[v][0] - half(na[v]), ax[v]), 0, n[v]), y1 = clamp(pos(na[v][na[v].length - 1] + half(na[v]), ax[v]), 0, n[v]);
      if (within && x1 > x0 && y1 > y0) over.append(svg("rect", { x: ox + x0, y: oy + y0, width: x1 - x0, height: y1 - y0, fill: "none", stroke: "#fff", "stroke-opacity": 0.7, "stroke-width": u, "stroke-dasharray": `${4 * u} ${3 * u}` }));
    }
    // each fit pattern's own best candidate: a disc per candidate, sized by how many chose it
    if ($("votes-pats").checked) {
      const count = new Map();
      for (const c of l.patBest) {
        const i = split(l, c);
        if (!flat && i[f] !== k) continue;
        const key = i[h] + n[h] * i[v];
        count.set(key, (count.get(key) || 0) + 1);
      }
      for (const [key, m] of count) {
        over.append(svg("circle", { cx: ox + (key % n[h]) + 0.5, cy: oy + Math.floor(key / n[h]) + 0.5, r: Math.min(0.46, 0.12 + 0.34 * Math.sqrt(m / l.patBest.length)), fill: "rgba(10,10,11,.55)", stroke: "#fff", "stroke-width": u }));
      }
    }
    // the winner (faint on a plane it does not lie in)
    if (flat || win[f] === k) over.append(svg("rect", { x: ox + win[h] + 0.08, y: oy + win[v] + 0.08, width: 0.84, height: 0.84, fill: "none", stroke: "#fff", "stroke-width": 2 * u, "stroke-opacity": win[f] === k ? 1 : 0.4 }));
    // the cross-hair
    if (flat || view.at[f] === k) {
      const cx = ox + view.at[h] + 0.5, cy = oy + view.at[v] + 0.5;
      over.append(svg("path", { d: `M${cx} ${oy}V${oy + n[v]}M${ox} ${cy}H${ox + n[h]}`, stroke: "#fff", "stroke-opacity": 0.4, "stroke-width": u, fill: "none" }));
    }
    // the pattern centres worth comparing with; one outside the slice points the way from its edge
    const mark = (pc, color, shape) => {
      if (!pc || (!flat && nearest(pc[f], ax[f]) !== k) || (!flat && beyond(pc[f], ax[f]) !== 0)) return;
      const px = pos(pc[h], ax[h]), py = pos(pc[v], ax[v]);
      const x = ox + clamp(px, 0, n[h]), y = oy + clamp(py, 0, n[v]);
      const r = 5 * u;
      let d;
      if (px < 0 || py < 0 || px > n[h] || py > n[v]) {
        // an arrowhead on the border, pointing outward
        const dx = px < 0 ? -1 : px > n[h] ? 1 : 0, dy = py < 0 ? -1 : py > n[v] ? 1 : 0;
        const m = Math.hypot(dx, dy), nx = dx / m, ny = dy / m;
        const bx = x - nx * 2.2 * r, by = y - ny * 2.2 * r;
        d = `M${x} ${y}L${bx - ny * r} ${by + nx * r}L${bx + ny * r} ${by - nx * r}Z`;
        over.append(svg("path", { d, fill: color, stroke: "#0a0a0b", "stroke-width": u }));
        return;
      }
      d = shape === "plus" ? `M${x - r} ${y}H${x + r}M${x} ${y - r}V${y + r}` : `M${x - r} ${y - r}L${x + r} ${y + r}M${x - r} ${y + r}L${x + r} ${y - r}`;
      over.append(svg("path", { d, stroke: "#0a0a0b", "stroke-width": 4 * u, fill: "none" }));
      over.append(svg("path", { d, stroke: color, "stroke-width": 1.8 * u, fill: "none" }));
    };
    mark(view.start, "#9d9da4", "x");
    mark(view.file, "#6a9ec8", "x");
    mark(g.pc, "#e8c77e", "plus");
  }

  function slider(id, n, at, name) {
    const s = $(id);
    s.max = String(Math.max(0, n - 1));
    s.value = String(at);
    s.disabled = n < 2;
    s.title = `Slide through the ${n} slices along ${name} (or scroll over the slice)`;
  }

  function drawPlanes(l, next, lo, hi) {
    const ax = axesOf(l), n = sizes(l);
    for (const [id, h, v, f] of PLANES) {
      const canvas = $(id), over = $(`${id}-over`);
      canvas.width = n[h];
      canvas.height = n[v];
      const ctx = canvas.getContext("2d");
      const img = ctx.createImageData(n[h], n[v]);
      paint(img, n[h], l, h, v, f, view.at[f], 0, 0, lo, hi);
      ctx.putImageData(img, 0, 0);
      over.replaceChildren();
      fit(canvas, over);
      const u = n[h] / Math.max(parseFloat(canvas.style.width) || 1, 1);
      marks(over, l, next, h, v, f, view.at[f], 0, 0, u, true);
      $(`${id}-cap`).textContent = `${NAMES[h]} →  ${NAMES[v]} ↓   ${NAMES[f]} = ${ax[f][view.at[f]].toFixed(4)}  (${view.at[f] + 1}/${n[f]})`;
      slider(`${id}-slide`, n[f], view.at[f], NAMES[f]);
    }
  }

  /** Every slice along one axis as tiles of one image, as many across as fills the well best. */
  function drawEvery(l, next, lo, hi) {
    const [h, v, f] = ALONG[view.mode];
    const ax = axesOf(l), n = sizes(l);
    const canvas = $("votes-all"), over = $("votes-all-over");
    const well = canvas.parentElement;
    const W = Math.max(well.clientWidth, 1), H = Math.max(well.clientHeight, 1);
    let cols = 1, best = 0;
    for (let c = 1; c <= n[f]; c++) {
      const rows = Math.ceil(n[f] / c);
      const s = Math.min(W / (c * (n[h] + 1) - 1), H / (rows * (n[v] + 1) - 1));
      if (s > best) { best = s; cols = c; }
    }
    const rows = Math.ceil(n[f] / cols);
    const tw = n[h] + 1, th = n[v] + 1;
    canvas.width = cols * tw - 1;
    canvas.height = rows * th - 1;
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(canvas.width, canvas.height);
    for (let k = 0; k < n[f]; k++) paint(img, canvas.width, l, h, v, f, k, (k % cols) * tw, Math.floor(k / cols) * th, lo, hi);
    ctx.putImageData(img, 0, 0);
    over.replaceChildren();
    fit(canvas, over);
    const u = canvas.width / Math.max(parseFloat(canvas.style.width) || 1, 1);
    for (let k = 0; k < n[f]; k++) {
      const ox = (k % cols) * tw, oy = Math.floor(k / cols) * th;
      marks(over, l, next, h, v, f, k, ox, oy, u, false);
      if (k === view.at[f]) over.append(svg("rect", { x: ox, y: oy, width: n[h], height: n[v], fill: "none", stroke: "#e6e6e8", "stroke-width": 1.5 * u }));
    }
    view.tiles = { h, v, f, cols, tw, th };
    $("votes-all-cap").textContent = `${NAMES[h]} →  ${NAMES[v]} ↓  in each;  ${NAMES[f]} from ${ax[f][0].toFixed(4)} to ${ax[f][n[f] - 1].toFixed(4)},  outlined: ${ax[f][view.at[f]].toFixed(4)}  (${view.at[f] + 1}/${n[f]})`;
    slider("votes-all-slide", n[f], view.at[f], NAMES[f]);
  }

  /** The mean vote along each axis through the cross-hair. */
  function drawProfiles(l, lo, hi) {
    const canvas = $("votes-profiles");
    const box = canvas.parentElement;
    const dpr = window.devicePixelRatio || 1;
    const W = Math.max(60, box.clientWidth), H = Math.max(60, box.clientHeight);
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    ctx.font = "10.5px ui-monospace, Consolas, monospace";
    ctx.textBaseline = "middle";
    const ax = axesOf(l), n = sizes(l);
    const pc = geometry().pc;
    const rowH = H / 3, padL = 14, padR = 4;
    view.profiles = { W, rowH, padL, padR };
    for (let k = 0; k < 3; k++) {
      const top = k * rowH + 4, bot = (k + 1) * rowH - 5;
      const x = (c) => (n[k] > 1 ? padL + (pos(c, ax[k]) / n[k]) * (W - padL - padR) : (padL + W - padR) / 2);
      const y = (s) => bot - ((s - lo) / Math.max(hi - lo, 1e-20)) * (bot - top);
      ctx.fillStyle = "#67676e";
      ctx.textAlign = "left";
      ctx.fillText(NAMES[k], 2, (top + bot) / 2);
      ctx.strokeStyle = "#222226";
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(padL, Math.round(bot) + 0.5); ctx.lineTo(W - padR, Math.round(bot) + 0.5); ctx.stroke();
      // the cross-hair, and the pattern centre now where it lies within this grid
      ctx.strokeStyle = "rgba(255,255,255,.45)";
      const cx = Math.round(x(ax[k][view.at[k]])) + 0.5;
      ctx.beginPath(); ctx.moveTo(cx, top - 2); ctx.lineTo(cx, bot); ctx.stroke();
      if (beyond(pc[k], ax[k]) === 0 && n[k] > 1) {
        ctx.strokeStyle = "#e8c77e";
        const px = Math.round(x(pc[k])) + 0.5;
        ctx.beginPath(); ctx.moveTo(px, bot - 5); ctx.lineTo(px, bot + 3); ctx.stroke();
      }
      const i = view.at.slice();
      const pts = [];
      for (let c = 0; c < n[k]; c++) { i[k] = c; const s = shownAt(l, i); pts.push([x(ax[k][c]), y(s), s]); }
      ctx.strokeStyle = "#9d9da4";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      pts.forEach(([px, py], c) => (c ? ctx.lineTo(px, py) : ctx.moveTo(px, py)));
      ctx.stroke();
      for (const [px, py, s] of pts) {
        const c = viridis((s - lo) / Math.max(hi - lo, 1e-20));
        ctx.fillStyle = `rgb(${c[0] | 0},${c[1] | 0},${c[2] | 0})`;
        ctx.beginPath(); ctx.arc(px, py, 2.4, 0, 2 * Math.PI); ctx.fill();
      }
    }
  }

  function draw() {
    const L = view.levels;
    const empty = !L || !L.length;
    $("votes-body").hidden = empty;
    $("votes-hint").hidden = !empty;
    $("votes-meta").textContent = "";
    if (empty || $("dock-votes").hidden) return;
    const l = L[view.level], next = L[view.level + 1];
    const ax = axesOf(l);
    const win = split(l, l.winner);
    const pc = geometry().pc;
    const [lo, hi] = range(shown(l));
    const [slo, shi] = range(l.scores);

    for (const b of $("votes-what").children) { b.setAttribute("aria-pressed", String(b.dataset.what === (l.density ? view.what : "mean"))); b.disabled = !l.density && b.dataset.what === "ballots"; }
    for (const b of $("votes-levels").children) b.setAttribute("aria-pressed", String(+b.dataset.level === view.level));
    for (const b of $("votes-mode").children) b.setAttribute("aria-pressed", String(b.dataset.mode === view.mode));
    $("votes-grid").dataset.mode = view.mode;
    $("votes-follow").disabled = view.following;
    if (view.mode === "planes") drawPlanes(l, next, lo, hi); else drawEvery(l, next, lo, hi);
    drawProfiles(l, lo, hi);

    const here = view.at.map((i, k) => ax[k][i]);
    const agree = l.patBest.filter((c) => c === l.winner).length;
    const f4 = (v) => v.map((x) => x.toFixed(4)).join(" ");
    const out = ax.map((a, k) => beyond(pc[k], a));
    const outside = out.some((d) => d !== 0);
    const near = ax.map((a, k) => nearest(pc[k], a));
    $("votes-readout").textContent = [
      "             x      y      z     NCC",
      `cross-hair ${f4(here)} ${scoreAt(l, view.at).toFixed(4)}`,
      `winner     ${f4(win.map((i, k) => ax[k][i]))} ${l.scores[l.winner].toFixed(4)}`,
      `centre now ${f4(pc)} ${outside ? "outside" : scoreAt(l, near).toFixed(4)}`,
      ...(outside ? [`  beyond this grid by ${out.map((d, k) => (d ? `${NAMES[k]} ${d > 0 ? "+" : "−"}${Math.abs(d).toFixed(4)}` : "")).filter(Boolean).join(", ")}`] : []),
      ...(l.walked ? ["  the winner is on the border: the grid moved"] : []),
      `this grid  ${slo.toFixed(4)} … ${shi.toFixed(4)} over ${l.scores.length}`,
      l.density ? `${l.inliers} of ${l.ballots} ballots lie by the winner` : `${agree} of ${l.patBest.length} patterns prefer the winner`,
    ].join("\n");
    const span = (a) => (a.length > 1 ? `±${((a[a.length - 1] - a[0]) / 2).toFixed(4)}` : "fixed");
    $("votes-meta").textContent = `grid ${view.level + 1} of ${L.length}: x ${span(l.x)}, y ${span(l.y)}, z ${span(l.z)}${view.following ? "  ·  cut through the pattern centre" : ""}`;
  }

  // ── moving the cross-hair ──

  // a click or a drag on a plane moves it in the plane; the wheel, the slider and Page Up /
  // Page Down move it through the slices; the arrow keys step it in the plane
  const wheelStep = (() => { let acc = 0; return (e) => { acc += e.deltaMode === 1 ? e.deltaY * 30 : e.deltaY; if (Math.abs(acc) < 30) return 0; const s = Math.sign(acc); acc = 0; return s; }; })();
  const keys = (well, h, v, f) => well.addEventListener("keydown", (e) => {
    const d = { ArrowLeft: [h(), -1], ArrowRight: [h(), 1], ArrowUp: [v(), -1], ArrowDown: [v(), 1], PageUp: [f(), 1], PageDown: [f(), -1] }[e.key];
    if (!d || !view.levels) return;
    e.preventDefault();
    move({ [d[0]]: view.at[d[0]] + d[1] });
  });
  for (const [id, h, v, f] of PLANES) {
    const canvas = $(id);
    const to = (e) => {
      if (!view.levels) return;
      const r = canvas.getBoundingClientRect();
      const n = sizes(view.levels[view.level]);
      move({ [h]: Math.floor(((e.clientX - r.left) / r.width) * n[h]), [v]: Math.floor(((e.clientY - r.top) / r.height) * n[v]) });
    };
    canvas.addEventListener("click", to);
    canvas.addEventListener("mousemove", (e) => { if (e.buttons === 1) to(e); });
    canvas.parentElement.addEventListener("wheel", (e) => { e.preventDefault(); const s = wheelStep(e); if (s && view.levels) move({ [f]: view.at[f] + s }); }, { passive: false });
    $(`${id}-slide`).addEventListener("input", (e) => move({ [f]: +e.target.value }));
    keys(canvas.parentElement, () => h, () => v, () => f);
  }
  {
    const canvas = $("votes-all");
    const to = (e) => {
      if (!view.levels || !view.tiles) return;
      const { h, v, f, cols, tw, th } = view.tiles;
      const r = canvas.getBoundingClientRect();
      const n = sizes(view.levels[view.level]);
      const x = ((e.clientX - r.left) / r.width) * canvas.width, y = ((e.clientY - r.top) / r.height) * canvas.height;
      const col = Math.floor(x / tw), row = Math.floor(y / th);
      const k = row * cols + col;
      if (col >= cols || k >= n[f] || x - col * tw >= n[h] || y - row * th >= n[v]) return;
      move({ [f]: k, [h]: Math.floor(x - col * tw), [v]: Math.floor(y - row * th) });
    };
    canvas.addEventListener("click", to);
    canvas.addEventListener("mousemove", (e) => { if (e.buttons === 1) to(e); });
    const along = () => ALONG[view.mode] || ALONG.z;
    canvas.parentElement.addEventListener("wheel", (e) => { e.preventDefault(); const s = wheelStep(e); if (s && view.levels) move({ [along()[2]]: view.at[along()[2]] + s }); }, { passive: false });
    $("votes-all-slide").addEventListener("input", (e) => move({ [along()[2]]: +e.target.value }));
    keys(canvas.parentElement, () => along()[0], () => along()[1], () => along()[2]);
  }
  // a click on a profile moves the cross-hair along that profile's axis
  $("votes-profiles").addEventListener("click", (e) => {
    if (!view.levels || !view.profiles) return;
    const r = $("votes-profiles").getBoundingClientRect();
    const { W, rowH, padL, padR } = view.profiles;
    const k = clamp(Math.floor((e.clientY - r.top) / rowH), 0, 2);
    const n = sizes(view.levels[view.level]);
    move({ [k]: Math.floor(((e.clientX - r.left - padL) / (W - padL - padR)) * n[k]) });
  });
  $("votes-levels").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) select(+b.dataset.level); });
  $("votes-mode").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) { view.mode = b.dataset.mode; draw(); } });
  $("votes-pats").addEventListener("change", draw);
  $("votes-what").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b && !b.disabled) { view.what = b.dataset.what; draw(); } });
  $("votes-follow").addEventListener("click", follow);
  $("votes-take").addEventListener("click", () => {
    if (!view.levels) return;
    const l = view.levels[view.level];
    // (the pattern centre changing brings the cuts back onto it: here, where they are)
    setPc(view.at.map((i, a) => axesOf(l)[a][i]));
  });

  return {
    /** Show a search's grids (null: none yet); `start` and `file` are pattern centres to mark. */
    show(levels, { start = null, file = null } = {}) {
      view.levels = levels && levels.length ? levels : null;
      view.start = start;
      view.file = file;
      $("votes-levels").replaceChildren(...(view.levels || []).map((l, i) => {
        const b = document.createElement("button");
        b.type = "button";
        b.dataset.level = String(i);
        b.textContent = `${i + 1}${l.walked ? " →" : ""}`;
        b.title = l.walked ? "The winner was on this grid's border: the next grid is as wide, moved onto it." : "";
        return b;
      }));
      follow();
    },
    /** The pattern centre changed: cut through it. */
    follow,
    draw,
  };
}
