// Drawing helpers: float images to canvases, a perceptual colour map, SVG overlays.

/** Draw a Float32Array image (w × h) as gray between its 0.5 % and 99.5 % quantiles (or lo, hi). */
export function drawGray(canvas, data, w, h, { lo = null, hi = null, mask = null } = {}) {
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(w, h);
  if (lo === null || hi === null) [lo, hi] = quantiles(data, 0.005, 0.995);
  const k = 255 / Math.max(hi - lo, 1e-20);
  for (let i = 0; i < w * h; i++) {
    const v = Number.isFinite(data[i]) ? Math.max(0, Math.min(255, (data[i] - lo) * k)) : 0;
    const a = mask && !mask[i] ? 0.25 : 1;
    img.data[4 * i] = img.data[4 * i + 1] = img.data[4 * i + 2] = v * a;
    img.data[4 * i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

/** Draw a signed image (w × h) blue–black–orange about zero, scaled by its 99 % magnitude. */
export function drawSigned(canvas, data, w, h) {
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(w, h);
  const mags = Float32Array.from(data, Math.abs);
  const s = 1 / Math.max(quantiles(mags, 0, 0.99)[1], 1e-20);
  for (let i = 0; i < w * h; i++) {
    const t = Math.max(-1, Math.min(1, data[i] * s));
    const m = Math.abs(t);
    img.data[4 * i] = 255 * m * (t > 0 ? 0.95 : 0.25);
    img.data[4 * i + 1] = 255 * m * (t > 0 ? 0.62 : 0.55);
    img.data[4 * i + 2] = 255 * m * (t > 0 ? 0.22 : 0.95);
    img.data[4 * i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

export function quantiles(data, qlo, qhi) {
  // a sample is enough for display ranges
  const step = Math.max(1, Math.floor(data.length / 20000));
  const s = [];
  for (let i = 0; i < data.length; i += step) if (Number.isFinite(data[i])) s.push(data[i]);
  if (!s.length) return [0, 1];
  s.sort((a, b) => a - b);
  return [s[Math.floor(qlo * (s.length - 1))], s[Math.floor(qhi * (s.length - 1))]];
}

// viridis, sampled
const VIRIDIS = [[68, 1, 84], [72, 35, 116], [64, 67, 135], [52, 94, 141], [41, 120, 142], [32, 144, 140], [34, 167, 132], [68, 190, 112], [121, 209, 81], [189, 222, 38], [253, 231, 36]];

export function viridis(t) {
  const x = Math.max(0, Math.min(1, t)) * (VIRIDIS.length - 1);
  const i = Math.min(Math.floor(x), VIRIDIS.length - 2);
  const f = x - i;
  return VIRIDIS[i].map((c, k) => c + (VIRIDIS[i + 1][k] - c) * f);
}

/** Draw a w × h scalar field with the colour map between its extremes (or lo, hi). */
export function drawHeat(canvas, data, w, h, lo = null, hi = null) {
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(w, h);
  if (lo === null || hi === null) {
    lo = Infinity;
    hi = -Infinity;
    for (const v of data) { if (v < lo) lo = v; if (v > hi) hi = v; }
  }
  const k = 1 / Math.max(hi - lo, 1e-20);
  for (let i = 0; i < w * h; i++) {
    const c = viridis((data[i] - lo) * k);
    img.data[4 * i] = c[0];
    img.data[4 * i + 1] = c[1];
    img.data[4 * i + 2] = c[2];
    img.data[4 * i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return [lo, hi];
}

/** Size a canvas's box to the largest multiple of its pixels that fits its well, and lay an
 *  SVG overlay exactly over it with the canvas's pixel coordinates. Returns the scale. */
export function fit(canvas, over = null) {
  const well = canvas.parentElement;
  const W = well.clientWidth, H = well.clientHeight;
  const s = Math.max(0.01, Math.min(W / canvas.width, H / canvas.height));
  const w = Math.max(1, Math.floor(canvas.width * s)), h = Math.max(1, Math.floor(canvas.height * s));
  canvas.style.width = `${w}px`;
  canvas.style.height = `${h}px`;
  if (over) {
    over.style.width = `${w}px`;
    over.style.height = `${h}px`;
    over.style.left = `${(W - w) / 2}px`;
    over.style.top = `${(H - h) / 2}px`;
    over.setAttribute("viewBox", `0 0 ${canvas.width} ${canvas.height}`);
  }
  return s;
}

const NS = "http://www.w3.org/2000/svg";
export function svg(tag, attrs) {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
  return el;
}

export function download(name, bytes, type = "application/octet-stream") {
  const url = URL.createObjectURL(new Blob([bytes], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
