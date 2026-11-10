// The energy model's account (the engine's energyModel): how the backscattered electrons are
// spread over the energy they leave with and the depth they leave from. Above, the share in
// each energy bin, from the beam's energy at the left down to nothing at the right; below, on
// the same axis, their spread over energy and depth together (depth downward, into the
// sample), with the depth profile of them all beside it. A bin's bar picks its master out.
// Every number drawn is the engine's; this only places them.

const NS = "http://www.w3.org/2000/svg";
// the right margin holds the depth profile; both charts keep it, so their energy axes agree
const MARGIN = { left: 36, right: 26, top: 18, bottom: 18 };
// a perceptual ramp, dark to light (after matplotlib's inferno)
const RAMP = [[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164]];

function ramp(t) {
  const x = Math.max(0, Math.min(1, t)) * (RAMP.length - 1);
  const i = Math.min(RAMP.length - 2, Math.floor(x)), f = x - i;
  return RAMP[i].map((v, k) => v + f * (RAMP[i + 1][k] - v));
}

function el(parent, tag, attrs, text) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, typeof v === "number" ? v.toFixed(1) : v);
  if (text !== undefined) e.textContent = text;
  parent.append(e);
  return e;
}

/** A chart's frame in `svg`: its plot box (pixels). */
function frame(svg, title, unit = "") {
  const r = svg.getBoundingClientRect();
  const W = Math.max(140, r.width), H = Math.max(70, r.height);
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.replaceChildren();
  const t = el(svg, "text", { x: 8, y: 12, class: "chart-title" }, title);
  if (unit) el(t, "tspan", { class: "chart-unit" }, `  ${unit}`);
  return { x0: MARGIN.left, x1: W - MARGIN.right, y0: MARGIN.top, y1: H - MARGIN.bottom, W, H };
}

/** Round ticks over [lo, hi], about n of them. */
function ticks(lo, hi, n) {
  const raw = Math.max(hi - lo, 1e-9) / n;
  const p = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * p).find((s) => s >= raw) || raw;
  const out = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(6));
  return out;
}

/** Draw the model into `eSvg` (the energy bins) and `zSvg` (energy and depth together).
 *  `solved`: how many bins (from the least loss) the last simulation under these settings
 *  solved, or null; `picked`: the bin whose master is shown alone, or null; `pick(i)`: called
 *  with a bin's index when its bar is clicked. */
export function drawEnergy(eSvg, zSvg, m, { solved = null, picked = null, pick = null } = {}) {
  if (!m || !m.bins.length) { eSvg.replaceChildren(); zSvg.replaceChildren(); return; }
  const nb = m.bins.length;
  const E0 = m.beam_kv, dE = m.bin_kev;
  const on = (i) => solved === null || i < solved;
  // the energy axis: the beam's at the left, nothing at the right
  const axis = (b) => (kv) => b.x0 + ((E0 - kv) / E0) * (b.x1 - b.x0);
  const energyTicks = (svg, b, X) => {
    for (const t of ticks(0, E0, 5)) el(svg, "text", { x: X(t), y: b.y1 + 13, class: "chart-tick", "text-anchor": "middle" }, String(t));
    el(svg, "text", { x: b.x1 + 8, y: b.y1 + 13, class: "chart-tick" }, "kV");
  };
  // ── the share of the electrons in each energy bin ──
  {
    const b = frame(eSvg, "Energy distribution");
    const X = axis(b);
    let top = 0;
    for (const x of m.bins) top = Math.max(top, x.weight);
    const Y = (w) => b.y1 - (w / top) * (b.y1 - b.y0);
    el(eSvg, "line", { x1: b.x0, y1: b.y1, x2: b.x1, y2: b.y1, class: "chart-axis" });
    m.bins.forEach((x, i) => {
      // bin i: from its voltage down by a bin's width
      const xa = X(x.kv), xb = X(Math.max(0, x.kv - dE));
      const gap = Math.min(1, 0.12 * (xb - xa));
      const r = el(eSvg, "rect", { x: xa + gap, y: Y(x.weight), width: Math.max(1, xb - xa - 2 * gap), height: Math.max(0.5, b.y1 - Y(x.weight)),
        class: `bar${on(i) ? " on" : ""}${picked === i ? " picked" : ""}${pick && solved !== null && i < solved ? " pickable" : ""}`, "data-bin": String(i) });
      el(r, "title", {}, `${x.kv.toFixed(2)} kV: ${(100 * x.weight).toFixed(2)} % of the electrons, from a mean depth of ${(1 / x.beta).toFixed(1)} nm${on(i) ? (solved !== null ? ". Click to see this bin's master alone" : "") : "; not solved (the bins stopped before it)"}`);
    });
    energyTicks(eSvg, b, X);
    el(eSvg, "text", { x: b.x0 - 5, y: b.y0 + 8, class: "chart-tick", "text-anchor": "end" }, `${(100 * top).toFixed(0)}%`);
    if (solved !== null && solved > 0 && solved < nb) {
      // where the solve stopped
      const xs = X(m.bins[solved - 1].kv - dE);
      el(eSvg, "line", { x1: xs, y1: b.y0, x2: xs, y2: b.y1, class: "chart-mark" });
      el(eSvg, "text", { x: xs - 3, y: b.y0 + 8, class: "chart-tick", "text-anchor": "end" }, "← solved");
    }
    eSvg.onclick = pick ? (ev) => { const t = ev.target.closest("[data-bin]"); if (t && t.classList.contains("pickable")) pick(+t.dataset.bin); } : null;
  }
  // ── energy across, depth down: the joint distribution, and the depth profile beside it ──
  {
    const b = frame(zSvg, "Energy–depth distribution", "depth in nm");
    const X = axis(b);
    const z = m.depth_nm, zmax = z[z.length - 1];
    const Y = (d) => b.y0 + (d / zmax) * (b.y1 - b.y0);
    // one column a bin (the beam's at the left), one row a depth sampled; drawn smooth
    const rows = z.length;
    const cv = document.createElement("canvas");
    cv.width = nb;
    cv.height = rows;
    const img = cv.getContext("2d").createImageData(nb, rows);
    let top = 0;
    for (const col of m.depth_by_bin) for (const v of col) top = Math.max(top, v);
    for (let i = 0; i < nb; i++) {
      for (let r = 0; r < rows; r++) {
        const c = ramp((m.depth_by_bin[i][r] / top) ** 0.6);
        const k = 4 * (r * nb + i);
        img.data[k] = c[0]; img.data[k + 1] = c[1]; img.data[k + 2] = c[2]; img.data[k + 3] = 255;
      }
    }
    cv.getContext("2d").putImageData(img, 0, 0);
    el(zSvg, "rect", { x: b.x0, y: b.y0, width: b.x1 - b.x0, height: b.y1 - b.y0, fill: "rgb(0,0,4)" });
    // (each column at its bin's middle: the image spans from the first bin's middle to the last's)
    const xa = X(m.bins[0].kv - dE / 2), xb = X(m.bins[nb - 1].kv - dE / 2);
    const half = (xb - xa) / Math.max(nb - 1, 1) / 2;
    el(zSvg, "image", { href: cv.toDataURL(), x: xa - half, y: b.y0, width: xb - xa + 2 * half, height: b.y1 - b.y0, preserveAspectRatio: "none", class: "heat" });
    if (picked !== null && picked < nb) {
      const pa = X(m.bins[picked].kv), pb = X(m.bins[picked].kv - dE);
      el(zSvg, "rect", { x: pa, y: b.y0, width: pb - pa, height: b.y1 - b.y0, class: "picked-band" });
    }
    // the mean depth of all the electrons, and the depth above which 90 % leave, over the map
    for (const [v, name] of [[m.mean_depth_nm, "mean depth"], [m.depth90_nm, "90 % leave above"]]) {
      if (!(v < zmax)) continue;
      el(zSvg, "line", { x1: b.x0, y1: Y(v), x2: b.x1, y2: Y(v), class: "chart-mark on-map" });
      el(zSvg, "text", { x: b.x1 - 4, y: Y(v) - 3, class: "chart-tick on-map", "text-anchor": "end" }, `${name} ${v.toFixed(1)} nm`);
    }
    for (const t of ticks(0, zmax, 4)) el(zSvg, "text", { x: b.x0 - 5, y: Y(t) + 3.5, class: "chart-tick", "text-anchor": "end" }, String(t));
    energyTicks(zSvg, b, X);
  }
}
