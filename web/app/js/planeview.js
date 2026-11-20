// How the source moves over the scan, as the fit has it and as the fit patterns measured it.
// Each pattern-centre coordinate is drawn against each direction of the map: down the rows
// (the tilted surface, where the sample's tilt shows) and across the columns. A pattern's own
// value is shown with the fit's change along the other direction taken out (a partial
// residual: as if the pattern stood on the map's middle row, or column), so that each panel is
// the fit's trend along its direction and the patterns' scatter about it. The trend is read
// from the medians of bins along the direction (with their quartiles) and the fit's line; the
// window holds the bulk of the points, and those beyond it are marked on its edge. Beside the
// six panels, the map: how far each pattern's own point lies from the fit.
import { viridis } from "./draw.js";

const $ = (id) => document.getElementById(id);
const FONT = "10.5px ui-monospace, Consolas, monospace";
const NAME = ["PCx", "PCy", "PCz"];
const MODELS = ["one centre", "tilt line", "affine plane", "projective plane"];

/** The plane's source for a map cell: P · (column, row, 1) = w · (PCx, PCy, PCz, 1). */
const at = (P, row, col) => {
  const w = P[9] * col + P[10] * row + P[11];
  return [(P[0] * col + P[1] * row + P[2]) / w, (P[3] * col + P[4] * row + P[5]) / w, (P[6] * col + P[7] * row + P[8]) / w];
};

const quantile = (sorted, q) => {
  if (!sorted.length) return NaN;
  const i = Math.min(sorted.length - 1, Math.max(0, q * (sorted.length - 1)));
  const lo = Math.floor(i), hi = Math.ceil(i);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo);
};

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];

/** A point of the unit ball as a colour: OKLab about the gray L = 0.725, radius 0.12 (the largest ball
 *  about the gray axis that sRGB holds whole), x → a, y → b, z → L; a point beyond the rim is drawn on it. */
const ball = (x, y, z) => {
  const n = Math.hypot(x, y, z);
  if (n > 1) { x /= n; y /= n; z /= n; }
  const L = 0.725 + 0.12 * z, A = 0.12 * x, B = 0.12 * y;
  const l = (L + 0.3963377774 * A + 0.2158037573 * B) ** 3;
  const m = (L - 0.1055613458 * A - 0.0638541728 * B) ** 3;
  const s = (L - 0.0894841775 * A - 1.291485548 * B) ** 3;
  const enc = (v) => {
    v = Math.min(1, Math.max(0, v));
    return Math.round(255 * (v <= 0.0031308 ? 12.92 * v : 1.055 * v ** (1 / 2.4) - 0.055));
  };
  return [enc(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s),
    enc(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s),
    enc(-0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s)];
};


/** The first of `texts` (longest first) that fits in `room` pixels, drawn at (x, y) as the
 *  context's alignment has it; none when none fits. */
function fitText(ctx, texts, x, y, room) {
  const t = texts.find((s) => ctx.measureText(s).width <= room);
  if (t) ctx.fillText(t, x, y);
}

export function createPlaneView() {
  const view = { r: null, pts: null, scan: null, mode: "trends" };

  /** The three coordinates as one colour, a point in OKLab (a colour space whose distances are close to how
   *  different two colours look): the change in x and y is the hue and its strength (a colour wheel: which
   *  way and how far the centre moves on the screen), z the lightness. One scale for the three, so an equal
   *  change in any direction looks about equally different; gray is no change, the key's rim the map's
   *  scale. x is counted in pattern heights (× the pattern's width over its height), like y and z. Three
   *  maps: the fit over every cell; the patterns' own values, each the median of the 5 × 5 fit patterns
   *  about it (one pattern's scatter is larger than the fit's whole span); and the difference of the two. */
  function colours(ctx, W, H) {
    const { pts, scan, r } = view;
    const { rows, cols } = scan;
    const asp = scan.patW / scan.patH;
    const P = r.plane;
    const n = pts.cells.length;
    const mid = at(P, (rows - 1) / 2, (cols - 1) / 2);
    const norm = (d) => Math.hypot(asp * d[0], d[1], d[2]);
    // the fit patterns on their lattice, and each one's 5 × 5 median of the used ones about it
    const stride = Math.max(1, Math.round(Math.sqrt((rows * cols) / Math.max(n, 1))));
    const where = new Map();
    pts.cells.forEach((c, b) => where.set(c, b));
    const smooth = new Float64Array(3 * n).fill(NaN);
    for (let b = 0; b < n; b++) {
      if (!pts.used[b]) continue;
      const rr = Math.floor(pts.cells[b] / cols), cc = pts.cells[b] % cols;
      const near = [[], [], []];
      for (let i = -2; i <= 2; i++) for (let j = -2; j <= 2; j++) {
        const r2 = rr + i * stride, c2 = cc + j * stride;
        if (r2 < 0 || r2 >= rows || c2 < 0 || c2 >= cols) continue;
        const o = where.get(r2 * cols + c2);
        if (o === undefined || !pts.used[o]) continue;
        for (let k = 0; k < 3; k++) near[k].push(pts.pc[3 * o + k]);
      }
      for (let k = 0; k < 3; k++) { near[k].sort((x, y) => x - y); smooth[3 * b + k] = quantile(near[k], 0.5); }
    }
    const ok = (b) => pts.used[b] && !Number.isNaN(smooth[3 * b]);
    const own = (b) => [smooth[3 * b], smooth[3 * b + 1], smooth[3 * b + 2]];
    const fit = (b) => [pts.fit[3 * b], pts.fit[3 * b + 1], pts.fit[3 * b + 2]];
    // each map's scale (the rim): the fit's farthest corner from mid-map; the 98th percentile of the
    // medians' distance from their own median, and of their distance from the fit
    let sFit = 1e-9;
    for (const [rr, cc] of [[0, 0], [0, cols - 1], [rows - 1, 0], [rows - 1, cols - 1]]) sFit = Math.max(sFit, norm(sub(at(P, rr, cc), mid)));
    const comp = [[], [], []];
    for (let b = 0; b < n; b++) if (ok(b)) for (let k = 0; k < 3; k++) comp[k].push(smooth[3 * b + k]);
    const centre = comp.map((v) => { v.sort((x, y) => x - y); return quantile(v, 0.5); });
    const dOwn = [], dRes = [];
    for (let b = 0; b < n; b++) if (ok(b)) { dOwn.push(norm(sub(own(b), centre))); dRes.push(norm(sub(own(b), fit(b)))); }
    const p98 = (v) => { v.sort((x, y) => x - y); return Math.max(quantile(v, 0.98), 1e-9); };
    const scales = [sFit, p98(dOwn), p98(dRes)];
    const tone = (d, sc) => ball((asp * d[0]) / sc, d[1] / sc, d[2] / sc);
    // the three maps side by side, and the key beside them or under them: whichever leaves the maps larger
    const padT = 22, padB = 8, gap = 22, padX = 14, keyW = 210, keyH = 128;
    const side = Math.min((W - 2 * padX - keyW - 3 * gap) / 3 / cols, (H - padT - padB) / rows);
    const under = Math.min((W - 2 * padX - 2 * gap) / 3 / cols, (H - padT - padB - keyH) / rows);
    const below = under > side;
    const cell = Math.max(0.5, below ? under : side);
    const pw = cell * cols, ph = cell * rows;
    const titles = ["PC fit", "PC measured, 5 × 5 median", "measured − fit"];
    const grays = ["mid-map", "median", "on the fit"];
    for (let p = 0; p < 3; p++) {
      const sc = scales[p];
      const img = new ImageData(cols, rows);
      const put = (rr, cc, c) => { const q = 4 * (rr * cols + cc); img.data[q] = c[0]; img.data[q + 1] = c[1]; img.data[q + 2] = c[2]; img.data[q + 3] = 255; };
      if (p === 0) {
        for (let rr = 0; rr < rows; rr++) for (let cc = 0; cc < cols; cc++) put(rr, cc, tone(sub(at(P, rr, cc), mid), sc));
      } else {
        for (let b = 0; b < n; b++) {
          const rr = Math.floor(pts.cells[b] / cols), cc = pts.cells[b] % cols;
          const c = !ok(b) ? [28, 28, 32] : tone(sub(own(b), p === 1 ? centre : fit(b)), sc);
          // (a fit pattern stands for the cells of its lattice square)
          for (let i = 0; i < stride; i++) for (let j = 0; j < stride; j++) if (rr + i < rows && cc + j < cols) put(rr + i, cc + j, c);
        }
      }
      const x0 = padX + p * (pw + gap), y0 = padT;
      const tmp = new OffscreenCanvas(cols, rows);
      tmp.getContext("2d").putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(tmp, x0, y0, pw, ph);
      ctx.textBaseline = "alphabetic";
      ctx.fillStyle = "#e6e6e8"; ctx.textAlign = "left"; ctx.fillText(titles[p], x0, y0 - 7);
      ctx.fillStyle = "#9d9da4"; ctx.textAlign = "right";
      fitText(ctx, [`gray: ${grays[p]} · rim ${sc.toFixed(4)}`, `rim ${sc.toFixed(4)}`], x0 + pw, y0 - 7, pw - ctx.measureText(titles[p]).width - 10);
    }
    if (below) key(ctx, padX, padT + ph + 16, W - 2 * padX, keyH - 16, true);
    else key(ctx, padX + 3 * pw + 3 * gap, padT, keyW - gap, ph, false);
    $("plane-cap").textContent = `${MODELS[r.model ?? (r.kind === 2 ? 3 : 2)]} through ${r.nUsed.toLocaleString()} of ${r.nPatterns.toLocaleString()} patterns  ·  each cell's PC change as one colour: x, y the hue, z the lightness`;
    $("plane-cap").title = "Each cell's change in (PCx, PCy, PCz) is one point in OKLab, a colour space in which distance is close to how different two colours look: the change in x and y sets the hue and its strength, the change in z the lightness. One scale for the three, so an equal change in any direction looks about equally different. x is counted in pattern heights, like y and z. The key's rim stands for each map's own scale, written above the map.";
  }

  /** The key: the colour wheel for x and y at the middle lightness, the lightness bar for z, and what they
   *  mean; in a column beside the maps, or in a row under them (`row`). */
  function key(ctx, x0, y0, w, h, row) {
    const R = Math.round(Math.max(18, Math.min(row ? (h - 34) / 2 : (w - 96) / 2, 56)));
    const cx = x0 + 22 + R, cy = y0 + 16 + R;
    const n = 2 * R + 1;
    const img = new ImageData(n, n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const dx = (j - R) / R, dy = (i - R) / R;
      const d2 = dx * dx + dy * dy;
      if (d2 > 1) continue;
      const c = ball(dx, dy, 0), q = 4 * (i * n + j);
      img.data[q] = c[0]; img.data[q + 1] = c[1]; img.data[q + 2] = c[2];
      img.data[q + 3] = Math.round(255 * Math.min(1, (1 - Math.sqrt(d2)) * R + 0.5));
    }
    const tmp = new OffscreenCanvas(n, n);
    tmp.getContext("2d").putImageData(img, 0, 0);
    ctx.drawImage(tmp, cx - R, cy - R);
    ctx.strokeStyle = "rgba(10,10,11,.45)"; ctx.lineWidth = 1; ctx.beginPath();
    ctx.moveTo(cx - R, cy + 0.5); ctx.lineTo(cx + R, cy + 0.5); ctx.moveTo(cx + 0.5, cy - R); ctx.lineTo(cx + 0.5, cy + R); ctx.stroke();
    ctx.fillStyle = "#e6e6e8"; ctx.textBaseline = "middle";
    ctx.textAlign = "left"; ctx.fillText("+x", cx + R + 5, cy);
    ctx.textAlign = "right"; ctx.fillText("−x", cx - R - 5, cy);
    ctx.textAlign = "center"; ctx.fillText("+y", cx, cy + R + 9); ctx.fillText("−y", cx, cy - R - 9);
    // the lightness bar: z from the rim below the centre to the rim above
    const bx = cx + R + 36, bw = 12;
    for (let i = 0; i <= 2 * R; i++) {
      const c = ball(0, 0, 1 - i / R);
      ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`; ctx.fillRect(bx, cy - R + i, bw, 1);
    }
    ctx.fillStyle = "#e6e6e8";
    ctx.fillText("+z", bx + bw / 2, cy - R - 9); ctx.fillText("−z", bx + bw / 2, cy + R + 9);
    // what they mean: beside the bar in a row, under the wheel in a column
    const lines = [["x, y", "hue: which way"], ["", "strength: how far"], ["z", "lightness"], ["gray", "no change"], ["rim", row ? "the map's scale (above it)" : "the map's scale"], ["+y", "down the screen"], ["dark", "left out"]];
    ctx.textAlign = "left"; ctx.textBaseline = "top";
    const tx = row ? bx + bw + 40 : x0 + 8, ty = row ? cy - R - 6 : cy + R + 26;
    // (in a row, two columns of them)
    lines.forEach(([k, t], i) => {
      const col = row && i >= 4 ? 1 : 0, y = ty + 15 * (i - 4 * col), x = tx + 210 * col;
      ctx.fillStyle = "#e6e6e8"; ctx.fillText(k, x, y);
      ctx.fillStyle = "#9d9da4"; ctx.fillText(t, x + 44, y);
    });
  }

  /** One panel: coordinate k against the map's rows (dir 0) or columns (dir 1). */
  function panel(ctx, box, k, dir) {
    const { pts, scan, r } = view;
    const { rows, cols } = scan;
    const P = r.plane;
    const n = pts.cells.length;
    const mid = [(rows - 1) / 2, (cols - 1) / 2];
    const span = dir === 0 ? rows : cols;
    // each pattern: where it stands along the direction, and its own value moved to the map's middle along the other
    const xs = new Float64Array(n), ys = new Float64Array(n);
    const used = [];
    for (let b = 0; b < n; b++) {
      const row = Math.floor(pts.cells[b] / cols), col = pts.cells[b] % cols;
      const here = at(P, row, col)[k];
      const moved = dir === 0 ? at(P, row, mid[1])[k] : at(P, mid[0], col)[k];
      xs[b] = dir === 0 ? row : col;
      ys[b] = pts.pc[3 * b + k] - here + moved;
      if (pts.used[b]) used.push(ys[b]);
    }
    const line = (t) => (dir === 0 ? at(P, t, mid[1]) : at(P, mid[0], t))[k];
    // the window: the bulk of the points (2nd to 98th percentile) and the whole of the fit's line
    const sorted = used.slice().sort((a, b) => a - b);
    let lo = quantile(sorted, 0.02), hi = quantile(sorted, 0.98);
    for (const t of [0, span - 1]) { lo = Math.min(lo, line(t)); hi = Math.max(hi, line(t)); }
    if (!(hi > lo)) { lo -= 1e-4; hi += 1e-4; }
    const pad = 0.08 * (hi - lo);
    lo -= pad; hi += pad;
    const { x0, y0, w, h } = box;
    const X = (v) => x0 + ((v + 0.5) / span) * w, Y = (v) => y0 + h - ((v - lo) / (hi - lo)) * h;
    ctx.fillStyle = "#0a0a0b"; ctx.fillRect(x0, y0, w, h);
    ctx.save(); ctx.beginPath(); ctx.rect(x0, y0, w, h); ctx.clip();
    // the points: faint where they took part, red where they were left out; those beyond the window on its edge
    const s = n > 3000 ? 1.4 : n > 600 ? 1.8 : 2.6;
    for (let pass = 0; pass < 2; pass++) {
      for (let b = 0; b < n; b++) {
        if (!!pts.used[b] !== !!pass) continue;
        const y = ys[b];
        const out = y < lo || y > hi;
        ctx.fillStyle = !pass ? "rgba(212,97,95,.6)" : out ? "rgba(230,230,232,.55)" : "rgba(200,200,206,.28)";
        const py = out ? (y < lo ? y0 + h - 2 : y0 + 2) : Y(y);
        if (out) ctx.fillRect(X(xs[b]) - 1, py - 2, 2, 4);
        else ctx.fillRect(X(xs[b]) - s / 2, py - s / 2, s, s);
      }
    }
    // the trend the points make: medians of bins along the direction, with their quartiles
    const nb = Math.max(4, Math.min(16, Math.round(Math.sqrt(used.length / 3))));
    const bins = Array.from({ length: nb }, () => []);
    for (let b = 0; b < n; b++) if (pts.used[b]) bins[Math.min(nb - 1, Math.floor((xs[b] / span) * nb))].push(ys[b]);
    ctx.strokeStyle = "rgba(240,240,242,.9)"; ctx.fillStyle = "#f0f0f2"; ctx.lineWidth = 1.2;
    bins.forEach((v, i) => {
      if (v.length < 3) return;
      v.sort((a, b) => a - b);
      const cx = x0 + ((i + 0.5) / nb) * w;
      const q1 = Y(quantile(v, 0.25)), q3 = Y(quantile(v, 0.75)), m = Y(quantile(v, 0.5));
      ctx.beginPath(); ctx.moveTo(cx, q1); ctx.lineTo(cx, q3); ctx.stroke();
      ctx.beginPath(); ctx.arc(cx, m, 2.6, 0, 2 * Math.PI); ctx.fill();
    });
    // the fit's line
    ctx.strokeStyle = "#e8c77e"; ctx.lineWidth = 1.6;
    ctx.beginPath();
    for (let i = 0; i <= 24; i++) {
      const t = -0.5 + (i / 24) * span;
      const px = X(t), py = Y(line(t));
      if (i) ctx.lineTo(px, py); else ctx.moveTo(px, py);
    }
    ctx.stroke();
    ctx.restore();
    ctx.strokeStyle = "#2b2b30"; ctx.lineWidth = 1; ctx.strokeRect(x0 + 0.5, y0 + 0.5, w - 1, h - 1);
    // what it says: the fit's change over the map, and the patterns' scatter about the fit
    const res = [];
    for (let b = 0; b < n; b++) if (pts.used[b]) res.push(Math.abs(ys[b] - line(xs[b])));
    res.sort((a, b) => a - b);
    const sd = 1.4826 * quantile(res, 0.5);
    const change = line(span - 1) - line(0);
    ctx.textBaseline = "alphabetic";
    ctx.fillStyle = "#e6e6e8"; ctx.textAlign = "left";
    const title = `${NAME[k]} along the ${dir === 0 ? "rows" : "columns"}`;
    ctx.fillText(title, x0, y0 - 6);
    ctx.fillStyle = "#9d9da4"; ctx.textAlign = "right";
    // the change over the map and the scatter, shortened to the room beside the title
    const ch = `${change >= 0 ? "+" : "−"}${Math.abs(change).toFixed(4)}`;
    fitText(ctx, [`${ch} over the map · ±${sd.toFixed(4)}`, `${ch} · ±${sd.toFixed(4)}`, ch], x0 + w, y0 - 6, w - ctx.measureText(title).width - 10);
    ctx.textBaseline = "middle";
    ctx.fillText(hi.toFixed(4), x0 - 4, y0 + 6); ctx.fillText(lo.toFixed(4), x0 - 4, y0 + h - 6);
    ctx.textAlign = "center";
    ctx.fillText("0", X(0), y0 + h + 9); ctx.fillText(String(span - 1), X(span - 1), y0 + h + 9);
    ctx.fillText(dir === 0 ? "row" : "column", x0 + w / 2, y0 + h + 9);
  }

  /** The map: how far each pattern's own point lies from the fit, on a scale the bulk of them sets. */
  function map(ctx, box) {
    const { pts, scan } = view;
    const { rows, cols } = scan;
    const n = pts.cells.length;
    const { x0, y0, w, h } = box;
    const cell = Math.min(w / cols, h / rows);
    const ox = x0 + (w - cell * cols) / 2, oy = y0 + (h - cell * rows) / 2;
    ctx.fillStyle = "#0a0a0b"; ctx.fillRect(x0, y0, w, h);
    const far = new Float64Array(n);
    const used = [];
    for (let b = 0; b < n; b++) {
      let q = 0;
      for (let k = 0; k < 3; k++) { const d = pts.pc[3 * b + k] - pts.fit[3 * b + k]; q += d * d; }
      far[b] = Math.sqrt(q);
      if (pts.used[b]) used.push(far[b]);
    }
    used.sort((a, b) => a - b);
    const top = Math.max(quantile(used, 0.95), 1e-12);
    // (a fit pattern stands for the cells about it: a square as wide as the lattice's stride)
    const stride = Math.max(1, Math.round(Math.sqrt((rows * cols) / Math.max(n, 1))));
    const wd = Math.max(1, cell * stride);
    // (the squares at the edges kept to the map)
    ctx.save();
    ctx.beginPath(); ctx.rect(ox, oy, cell * cols, cell * rows); ctx.clip();
    for (let b = 0; b < n; b++) {
      const r = Math.floor(pts.cells[b] / cols), c = pts.cells[b] % cols;
      const rgb = pts.used[b] ? viridis(Math.min(1, far[b] / top)) : [212, 97, 95];
      ctx.fillStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
      ctx.fillRect(ox + c * cell - (wd - cell) / 2, oy + r * cell - (wd - cell) / 2, wd, wd);
    }
    ctx.restore();
    ctx.strokeStyle = "#2b2b30"; ctx.lineWidth = 1; ctx.strokeRect(x0 + 0.5, y0 + 0.5, w - 1, h - 1);
    // the scale, under the map: viridis from 0 to the 95th percentile of the patterns' distances
    const by = Math.min(y0 + h - 22, oy + cell * rows + 14), bw = Math.min(w - 24, 180), bx = x0 + (w - bw) / 2;
    for (let i = 0; i < bw; i++) {
      const c = viridis(i / (bw - 1));
      ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`;
      ctx.fillRect(bx + i, by, 1, 7);
    }
    ctx.fillStyle = "#9d9da4"; ctx.textBaseline = "top";
    ctx.textAlign = "left"; ctx.fillText("0", bx, by + 10);
    ctx.textAlign = "right"; ctx.fillText(`${top.toFixed(4)} and more`, bx + bw, by + 10);
    ctx.textAlign = "center"; ctx.fillStyle = "rgb(212,97,95)"; ctx.fillText("red: left out", bx + bw / 2, by + 24);
    ctx.textBaseline = "alphabetic";
    ctx.fillStyle = "#e6e6e8"; ctx.textAlign = "left";
    ctx.fillText("distance from the fit", x0, y0 - 6);
    ctx.textBaseline = "middle";
  }

  function draw() {
    const empty = !view.r;
    $("plane-body").hidden = empty;
    $("plane-hint").hidden = !empty;
    if (empty || $("dock-plane").hidden) return;
    // (the caption first, one line: the canvas takes what is left of the view's height)
    const r = view.r;
    $("plane-cap").textContent = `${MODELS[r.model ?? (r.kind === 2 ? 3 : 2)]} through ${r.nUsed.toLocaleString()} of ${r.nPatterns.toLocaleString()} patterns  ·  white: bin medians and quartiles  ·  gold: the fit  ·  red: left out`;
    $("plane-cap").title = "Each panel: a pattern-centre coordinate along the map's rows or columns. Each fit pattern's own value is moved to the map's middle along the other direction (the fit's change that way taken out), so the panel shows the fit's trend along its direction and the patterns' scatter about it. The window holds the bulk of the points; those beyond it are marked on its edge. Above each panel: the fit's change over the map, and the patterns' scatter about it (1.4826 × the median absolute deviation).";
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
    for (const b of $("plane-mode").children) b.setAttribute("aria-pressed", String(b.dataset.mode === view.mode));
    if (view.mode === "colour") { colours(ctx, W, H); return; }
    // three columns of panels (PCx, PCy, PCz), along the rows above and the columns below; the map beside them
    const padL = 52, padT = 20, padB = 20, gapX = 16, gapY = 34, padR = 14;
    const mapW = Math.max(120, Math.min(0.24 * W, H - padT - padB));
    const pw = Math.max(40, (W - padL - 3 * gapX - mapW - padR - 2 * (padL - gapX)) / 3);
    const ph = Math.max(30, (H - padT - padB - gapY) / 2);
    for (let dir = 0; dir < 2; dir++) {
      for (let k = 0; k < 3; k++) {
        panel(ctx, { x0: padL + k * (pw + padL), y0: padT + dir * (ph + gapY), w: pw, h: ph }, k, dir);
      }
    }
    const mx = padL + 3 * (pw + padL) - padL + gapX;
    map(ctx, { x0: mx, y0: padT, w: Math.max(40, W - mx - padR), h: H - padT - padB });
  }

  $("plane-mode").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) { view.mode = b.dataset.mode; draw(); } });

  return {
    /** Show a plane fit: its result, its measured points, and the scan's shape (null: none). */
    show(r, pts, scan) {
      view.r = r; view.pts = pts; view.scan = scan;
      draw();
    },
    draw,
  };
}
