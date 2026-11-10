// The plane of source points, laid open as Winkelmann et al. (2020) and kikuchipy draw it: where
// each fit pattern's own fit put its source (a dot a pattern), and where the fitted plane has it
// (gold), in the three pairs of pattern-centre coordinates. A fourth panel is the map: how far
// each pattern's own point lies from the plane.
import { viridis } from "./draw.js";

const $ = (id) => document.getElementById(id);
const FONT = "10.5px ui-monospace, Consolas, monospace";
// the pairs drawn: [across, up], indices into (PCx, PCy, PCz)
const PAIRS = [[0, 1], [2, 1], [0, 2]];
const NAME = ["PCx", "PCy", "PCz"];

export function createPlaneView() {
  const view = { r: null, pts: null, scan: null };

  function draw() {
    const empty = !view.r;
    $("plane-body").hidden = empty;
    $("plane-hint").hidden = !empty;
    if (empty || $("dock-plane").hidden) return;
    const canvas = $("plane-canvas");
    const box = canvas.parentElement;
    const dpr = window.devicePixelRatio || 1;
    const W = Math.max(60, box.clientWidth), H = Math.max(60, box.clientHeight);
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    canvas.style.width = `${W}px`; canvas.style.height = `${H}px`;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    ctx.font = FONT; ctx.textBaseline = "middle";
    const { pts } = view;
    const n = pts.cells.length;
    const padL = 46, padB = 26, padT = 6, gap = 14;
    const side = Math.max(30, Math.min((W - 4 * padL - 3 * gap) / 4, H - padT - padB));
    // each coordinate's range: the plane's own, with what the used points add
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let b = 0; b < n; b++) for (let k = 0; k < 3; k++) {
      const f = pts.fit[3 * b + k];
      if (f < lo[k]) lo[k] = f;
      if (f > hi[k]) hi[k] = f;
    }
    // (the points scatter far wider than the plane spans on a small map: the window is the
    // plane's span and three of the points' own standard deviations about it, whichever is more)
    for (let k = 0; k < 3; k++) {
      const pad = Math.max(0.6 * (hi[k] - lo[k]), 3 * view.r.rmsd[k], 1e-5);
      lo[k] -= pad; hi[k] += pad;
    }
    PAIRS.forEach(([a, c], p) => {
      const x0 = padL + p * (side + padL + gap), y0 = padT;
      const X = (v) => x0 + ((v - lo[a]) / (hi[a] - lo[a])) * side, Y = (v) => y0 + side - ((v - lo[c]) / (hi[c] - lo[c])) * side;
      ctx.fillStyle = "#0a0a0b"; ctx.fillRect(x0, y0, side, side);
      ctx.save(); ctx.beginPath(); ctx.rect(x0, y0, side, side); ctx.clip();
      // each pattern's own point: light where it took part, red where it was left out
      const s = n > 4000 ? 1 : n > 600 ? 1.6 : 2.4;
      for (let pass = 0; pass < 2; pass++) {
        ctx.fillStyle = pass ? "rgba(230,230,232,.5)" : "rgba(212,97,95,.55)";
        for (let b = 0; b < n; b++) {
          if (!!pts.used[b] !== !!pass) continue;
          ctx.fillRect(X(pts.pc[3 * b + a]) - s / 2, Y(pts.pc[3 * b + c]) - s / 2, s, s);
        }
      }
      // where the plane has them
      ctx.fillStyle = "#e8c77e";
      for (let b = 0; b < n; b++) ctx.fillRect(X(pts.fit[3 * b + a]) - 0.8, Y(pts.fit[3 * b + c]) - 0.8, 1.6, 1.6);
      ctx.restore();
      ctx.strokeStyle = "#2b2b30"; ctx.lineWidth = 1; ctx.strokeRect(x0 + 0.5, y0 + 0.5, side - 1, side - 1);
      ctx.fillStyle = "#9d9da4"; ctx.textAlign = "center";
      ctx.fillText(lo[a].toFixed(4), x0 + 18, y0 + side + 8); ctx.fillText(hi[a].toFixed(4), x0 + side - 18, y0 + side + 8);
      ctx.fillStyle = "#e6e6e8"; ctx.fillText(NAME[a], x0 + side / 2, y0 + side + 18);
      ctx.textAlign = "right"; ctx.fillStyle = "#9d9da4";
      ctx.fillText(hi[c].toFixed(4), x0 - 4, y0 + 6); ctx.fillText(lo[c].toFixed(4), x0 - 4, y0 + side - 6);
      ctx.fillStyle = "#e6e6e8"; ctx.fillText(NAME[c], x0 - 4, y0 + side / 2);
    });
    // the map: each fit pattern's distance from the plane, in the pattern's height
    {
      const x0 = padL + 3 * (side + padL + gap), y0 = padT;
      const { rows, cols } = view.scan;
      const cell = Math.min(side / cols, side / rows);
      const ox = x0 + (side - cell * cols) / 2, oy = y0 + (side - cell * rows) / 2;
      ctx.fillStyle = "#0a0a0b"; ctx.fillRect(x0, y0, side, side);
      let top = 0;
      const far = new Float64Array(n);
      for (let b = 0; b < n; b++) {
        let q = 0;
        for (let k = 0; k < 3; k++) { const d = pts.pc[3 * b + k] - pts.fit[3 * b + k]; q += d * d; }
        far[b] = Math.sqrt(q);
        if (pts.used[b] && far[b] > top) top = far[b];
      }
      // (a fit pattern stands for the cells about it: a square as wide as the lattice's stride)
      const stride = Math.max(1, Math.round(Math.sqrt((rows * cols) / Math.max(n, 1))));
      const w = Math.max(1, cell * stride);
      for (let b = 0; b < n; b++) {
        const r = Math.floor(pts.cells[b] / cols), c = pts.cells[b] % cols;
        const rgb = pts.used[b] ? viridis(far[b] / Math.max(top, 1e-12)) : [212, 97, 95];
        ctx.fillStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
        ctx.fillRect(ox + c * cell - (w - cell) / 2, oy + r * cell - (w - cell) / 2, w, w);
      }
      ctx.strokeStyle = "#2b2b30"; ctx.strokeRect(x0 + 0.5, y0 + 0.5, side - 1, side - 1);
      ctx.fillStyle = "#9d9da4"; ctx.textAlign = "right";
      ctx.fillText(`0 … ${top.toFixed(4)}`, x0 + side, y0 + side + 8);
      ctx.fillStyle = "#e6e6e8";
      ctx.fillText("off the plane", x0 + side, y0 + side + 18);
    }
    const r = view.r;
    $("plane-cap").textContent = `${r.kind === 2 ? "projective" : "affine"} plane through ${r.nUsed.toLocaleString()} of ${r.nPatterns.toLocaleString()} fit patterns' own source points  ·  light: a pattern's own point, red: left out, gold: the plane  ·  about the plane they scatter by ${r.rmsd.map((v) => v.toFixed(5)).join(", ")}`;
  }

  return {
    /** Show a plane fit: its result, its measured points, and the scan's shape (null: none). */
    show(r, pts, scan) {
      view.r = r; view.pts = pts; view.scan = scan;
      draw();
    },
    draw,
  };
}
