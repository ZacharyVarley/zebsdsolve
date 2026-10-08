// The zebsdsolve app: a thin page over the engine (zebsdsolve.wasm through zebsdsolve.js). It
// passes files and settings in and draws what comes out; every number it shows is the engine's.
import { Zebsdsolve } from "../zebsdsolve/zebsdsolve.js";
import { GEOMETRY, buildFields, groupOf } from "./fields.js";
import { drawGray, drawSigned, drawHeat, fit, svg, download, quantiles } from "./draw.js";
import { createVotes } from "./votes.js";

const $ = (id) => document.getElementById(id);
const app = $("app");
const PHASE_COLORS = ["#3b82f6", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#14b8a6", "#ec4899", "#84cc16"];
const RESULT_KINDS = ["ipf_z", "ipf_x", "ipf_y", "ncc", "nmi", "z", "phase", "sweep"];
const TABS = ["data", "patterns", "geometry", "index"];
// settings outside the pattern's own groups that change what the pattern view shows of a
// pattern indexed on the spot
const LIVE_KEYS = new Set(["fit_bandwidth", "refine", "refine_iters", "refine_step", "nmi_bins", "phase_by", "scan_x_sign", "scan_y_sign"]);
// what Tune speed tries, in the engine's order (Tune.moves in src/engine.zig)
const TUNE_MOVES = ["pixels per invocation", "the batch with them", "refinement's workgroup", "the correlation's workgroup"];

const st = {
  ze: null,
  scan: null,          // engine scan info, or null
  scanName: "",
  settings: {},
  cell: -1,            // selected map cell
  mapKind: "overview",
  overview: null,      // Float32Array(rows · cols): pattern contrast, NaN where not read yet
  overviewRun: 0,
  results: null,       // engine results after indexing
  fit: null,           // fit patterns { cells, phases, quats, ncc }
  votes: null,         // every grid of the last pattern-centre search
  pcStart: null,       // where that search started, and the pattern centre the file gave
  pcFile: null,
  geoOpened: null,     // the geometry as the scan was opened
  geoBefore: null,     // pattern centre and rotation before the last fit (or after it, once put back)
  geoIsBefore: false,
  phase: 0,            // selected phase (master view)
  phaseFiles: [],      // the master pattern files, as added: to add the others again when one is removed
  stage: "filtered",   // how far along its preparation the pattern is shown
  live: new Map(),     // cell → what indexing it on the spot found, under the settings as they are
  keys: new Map(),     // Laue class → its inverse pole figure key
  legendFolded: false,
  chance: null,        // what NMI chance gives on this scan (the last indexing run's calibration)
  truth: null,         // a toy scan's own answer: { quaternions, phase } per cell
  tuned: "",
  busy: 0,
  stop: false,
  showRun: 0,
  roiDrag: null,
  // the map's view: zoom (1: all of it), the cell at the well's centre, screen pixels per cell
  view: { k: 1, cx: 0, cy: 0, s: 1 },
  drag: null,
};

// ── status, log, busy ──

function status(text, err = false) {
  const el = $("status");
  el.textContent = text;
  el.classList.toggle("err", err);
}

function log(text, err = false) {
  const el = $("log");
  const t = new Date().toLocaleTimeString([], { hour12: false });
  const line = document.createElement("div");
  if (err) line.className = "err";
  line.innerHTML = `<span class="t">${t}</span>  `;
  line.append(text);
  el.append(line);
  el.scrollTop = el.scrollHeight;
}

function progress(frac) {
  $("progress").hidden = frac === null;
  if (frac !== null) $("progress-bar").style.width = `${(100 * Math.max(0, Math.min(1, frac))).toFixed(1)}%`;
}

/** Run one engine job: the actions and the settings are off and Stop is on while it runs. The
 *  selected pattern is shown again when it is done (a job changes what there is to show). */
async function busy(label, fn) {
  if (st.busy) return;
  st.busy = 1;
  st.stop = false;
  app.dataset.busy = "1";
  $("btn-stop").hidden = false;
  status(label);
  const t0 = performance.now();
  try {
    await fn();
    status(`${label.replace(/…$/, "")}: done in ${((performance.now() - t0) / 1000).toFixed(2)} s`);
  } catch (e) {
    const msg = String(e.message || e);
    if (/Cancelled/.test(msg)) status("stopped"); else { status(msg, true); log(msg, true); console.error(e); }
  } finally {
    st.busy = 0;
    app.dataset.busy = "0";
    $("btn-stop").hidden = true;
    progress(null);
    st.live.clear();
    showCell();
  }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const factRows = (facts) => facts.flatMap(([k, v]) => { const dt = document.createElement("dt"); dt.textContent = k; const dd = document.createElement("dd"); dd.textContent = v; dd.title = v; return [dt, dd]; });

// ── the steps ──

function selectTab(name) {
  for (const b of $("tabs").children) {
    const on = b.dataset.tab === name;
    b.setAttribute("aria-selected", String(on));
    b.tabIndex = on ? 0 : -1;
  }
  for (const p of $("panels").children) p.hidden = p.dataset.tab !== name;
  $("panels").scrollTop = 0;
}

const done = (name, on = true) => { $(`tab-${name}`).dataset.done = on ? "1" : "0"; };

// ── settings and geometry ──

let fields = null;
let votes = null;
const METHOD_NAMES = { spherical: "spherical", dictionary: "dictionary", pca: "dictionary components" };
const sweepLabel = (p) => (p.method === "spherical" ? `spherical ${p.bandwidth}` : `${METHOD_NAMES[p.method]} ${p.dictGrid}`);

async function applySetting(key, value) {
  try {
    st.ze.configure({ [key]: value });
  } catch (e) {
    status(String(e.message || e), true);
  }
  afterSettings([key]);
}

/** Show the engine's settings again and redraw what the changed ones touch. */
function afterSettings(keys) {
  st.settings = st.ze.settings;
  fields.show(st.settings);
  showResolution();
  showSweeps();
  showStage();
  const groups = new Set(keys.map(groupOf));
  if (["nlpar", "filters", "master", "resolution"].some((g) => groups.has(g)) || keys.some((k) => LIVE_KEYS.has(k))) {
    st.live.clear();
    showAngle();
    showCell();
    showMaster();
  }
  if (groups.has("roi") || keys.includes("fit_grid")) { st.fit = null; drawMapOverlay(); }
  if (keys.includes("accept_z") || keys.includes("accept_nmi")) drawZ();
  if (keys.some((k) => k.startsWith("ipf_"))) { st.keys.clear(); drawMap(); }
  drawLegend();
}

function buildGeometry() {
  const box = $("geometry-fields");
  for (const [key, label, hint, step, per] of GEOMETRY) {
    const triple = key === "pc" || key === "rot";
    const el = document.createElement("label");
    el.className = "field" + (triple ? " wide" : "");
    if (hint) el.title = hint;
    const span = document.createElement("span");
    span.textContent = label;
    el.append(span);
    const mk = (i) => {
      const input = document.createElement("input");
      input.type = "number";
      input.step = String(step);
      input.dataset.geo = key;
      if (i !== null) input.dataset.i = String(i);
      input.addEventListener("change", () => {
        const g = st.ze.geometry;
        const v = +input.value / per;
        if (!Number.isFinite(v)) return;
        if (i === null) g[key] = v; else g[key][i] = v;
        st.ze.setGeometry({ [key]: g[key] });
        geometryChanged();
      });
      return input;
    };
    if (triple) {
      const tri = document.createElement("div");
      tri.className = "triple";
      tri.append(mk(0), mk(1), mk(2));
      el.append(tri);
    } else el.append(mk(null));
    box.append(el);
  }
}

function showGeometry() {
  const g = st.ze.geometry;
  const per = Object.fromEntries(GEOMETRY.map((f) => [f[0], f[4]]));
  for (const input of $("geometry-fields").querySelectorAll("input")) {
    const v = input.dataset.i !== undefined ? g[input.dataset.geo][+input.dataset.i] : g[input.dataset.geo];
    input.value = String(+(v * per[input.dataset.geo]).toFixed(6));
  }
  $("btn-geo-undo").disabled = !st.geoBefore;
  $("btn-geo-undo").textContent = st.geoIsBefore ? "after the last fit" : "before the last fit";
  $("btn-geo-file").disabled = !st.geoOpened;
  showAngle();
  drawPc();
}

/** The geometry changed (a fit, a typed value, a candidate taken): everything drawn from it
 *  follows, the search's slices included, which then cut through the pattern centre as it is. */
function geometryChanged(byFit = false) {
  if (!byFit && st.fit) st.fit = { ...st.fit, quats: new Float32Array(0) };
  st.live.clear();
  showGeometry();
  if (votes) votes.follow();
  if (!st.busy) showCell();
  showMaster();
}

/** Remember the pattern centre and rotation a fit is about to change. */
function beforeFit() {
  const g = st.ze.geometry;
  st.geoBefore = { pc: g.pc, rot: g.rot };
  st.geoIsBefore = false;
}

/** The sizes patterns can be reduced to, and what the current reduction gives. */
function showResolution() {
  const s = st.scan;
  const b = st.settings.binning;
  $("pat-full-box").hidden = b === 1;
  if (!s) return;
  const sel = $("bin-choices");
  const whole = [];
  for (let k = 1; k <= 32; k++) if (s.patH % k === 0 && s.patW % k === 0 && s.patH / k >= 8 && s.patW / k >= 8) whole.push(k);
  sel.replaceChildren(...whole.map((k) => new Option(k === 1 ? `${s.patH} × ${s.patW}  as stored` : `${s.patH / k} × ${s.patW / k}  binned by ${k}`, String(k))));
  if (!whole.includes(b)) sel.append(new Option(`reduced by ${b}`, String(b)));
  sel.value = String(b);
  sel.disabled = false;
  let text = "";
  try {
    const [h, w] = st.ze.shape(false);
    if (b !== 1) text = `${s.patH} × ${s.patW} stored → ${h} × ${w}, ${whole.includes(b) ? "binned" : "resampled by area"} as it is read`;
  } catch (e) { text = String(e.message || e); }
  $("res-readout").textContent = text;
}

/** The angular width the master filters take a detector pixel to have. */
function showAngle() {
  if (!st.ze) return;
  const a = st.ze.pixelAngle;
  $("angle-readout").textContent = a > 0 && st.scan
    ? `a stored pixel spans ${a.toFixed(3)}° (${st.settings.pixel_angle_deg > 0 ? "as set" : "from the pattern centre"}); the pattern's height about ${(a * st.scan.patH).toFixed(0)}°`
    : "";
}

/** What each sweep will search with. */
function showSweeps() {
  if (!st.ze) return;
  const s = st.settings;
  const plan = st.ze.sweepPlan;
  $("sweep-readout").textContent = s.sweeps > 1
    ? `${plan.map(sweepLabel).join("  →  ")}\neach sweep takes what the ones before did not accept`
    : `one pass, ${sweepLabel(plan[0])}; settled matches with ${acceptText()} are marked accepted`;
  if (st.chance) $("sweep-readout").textContent += `\nchance on this scan: NMI ${st.chance.median.toFixed(4)} ± ${st.chance.spread.toFixed(4)}${s.accept_nmi > 0 ? "" : `; z ${s.accept_z} is NMI ${(st.chance.median + s.accept_z * st.chance.spread).toFixed(4)}`}`;
}

/** The acceptance threshold in words: an absolute NMI, or the calibration's z. */
function acceptText() {
  const s = st.settings;
  return s.accept_nmi > 0 ? `NMI ≥ ${s.accept_nmi}` : `z ≥ ${s.accept_z}`;
}

/** The stages a pattern can be shown at: the averaged one only while patterns are averaged. */
function showStage() {
  if (st.stage === "averaged" && !st.settings.nlpar) st.stage = "filtered";
  for (const b of $("pat-stage").children) {
    b.setAttribute("aria-pressed", String(b.dataset.stage === st.stage));
    if (b.dataset.stage === "averaged") b.disabled = !st.settings.nlpar;
  }
  if (!st.settings.nlpar) $("nlpar-readout").textContent = "";
}

// ── scan ──

function sortScanFiles(files) {
  // the pattern file first, side files after
  const rank = (f) => (/\.(h5|hdf5|h5ebsd|h5oina|up1|up2|ebsp|dat)$/i.test(f.name) ? 0 : 1);
  return [...files].sort((a, b) => rank(a) - rank(b));
}

async function openScan(files) {
  files = sortScanFiles(files);
  if (!files.length) return;
  await busy(`opening ${files[0].name}…`, async () => {
    await st.ze.openScan(files);
    st.scanName = files[0].name;
    afterOpen();
  });
}

function afterOpen() {
  const s = st.ze.scan;
  st.scan = s;
  st.truth = null;
  st.results = null;
  st.fit = null;
  st.votes = null;
  st.pcStart = null;
  st.pcFile = s.hasPc ? st.ze.geometry.pc : null;
  st.geoOpened = st.ze.geometry;
  st.geoBefore = null;
  st.geoIsBefore = false;
  st.chance = null;
  st.live.clear();
  st.overview = new Float32Array(s.rows * s.cols).fill(NaN);
  st.view = { k: 1, cx: s.cols / 2, cy: s.rows / 2, s: 1 };
  app.dataset.scan = "1";
  done("geometry", false);
  done("index", false);
  const gb = s.bytes / 2 ** 30;
  $("scan-meta").textContent = `${st.scanName} · ${s.format} · ${s.rows} × ${s.cols} patterns of ${s.patH} × ${s.patW} ${s.dtype}`;
  const facts = [
    ["format", s.format], ["map", `${s.rows} × ${s.cols} (${(s.rows * s.cols).toLocaleString()} patterns)`],
    ["pattern", `${s.patH} × ${s.patW} ${s.dtype}`], ["size", gb >= 1 ? `${gb.toFixed(2)} GiB` : `${(s.bytes / 2 ** 20).toFixed(1)} MiB`],
    ["static background", s.hasBackground ? "one in the file" : "none in the file"], ["pattern centre", s.hasPc ? "from the file" : "not in the file"],
  ];
  if (s.voltageKv) facts.push(["voltage", `${s.voltageKv} kV`]);
  $("scan-facts").replaceChildren(...factRows(facts));
  st.settings = st.ze.settings;
  fields.show(st.settings);
  showResolution();
  showStage();
  showGeometry();
  setMapKind("overview");
  drawVotes();
  drawZ();
  selectCell(Math.floor(s.rows / 2) * s.cols + Math.floor(s.cols / 2));
  runOverview();
  updateEnabled();
}

/** The overview: every pattern's contrast, on a lattice of patterns when the map is large (at
 *  most about 40 000 patterns or 1.5 GiB of them). Whole lattice rows are read, each once (a
 *  file reads neighbours fastest), every fourth first and then the rows between: the whole map
 *  shows at once and sharpens. It yields to the engine's jobs between pieces. */
async function runOverview() {
  const run = ++st.overviewRun;
  const { rows, cols, patH, patW, pixel } = st.scan;
  const budget = Math.max(1000, Math.min(40000, Math.floor((1.5 * 2 ** 30) / (patH * patW * [1, 2, 4][pixel]))));
  const stride = Math.max(1, Math.ceil(Math.sqrt((rows * cols) / budget)));
  const nr = Math.ceil(rows / stride);
  // passes over the lattice rows at steps 2^top … 1, the first of about a dozen rows
  let top = 0;
  while (nr >> (top + 1) >= 12) top++;
  const t0 = performance.now();
  let read = 0, lastDraw = 0;
  for (let p = top; p >= 0; p--) {
    const step = 1 << p;
    const todo = [];
    for (let j = 0; j < nr; j += step) if (p === top || j % (2 * step) !== 0) todo.push(j);
    for (let i = 0; i < todo.length;) {
      // as many lattice rows as make a batch of about 256 patterns
      const part = [];
      do {
        const r = todo[i++] * stride;
        for (let c = 0; c < cols; c += stride) part.push(r * cols + c);
      } while (i < todo.length && part.length < 256);
      while (st.busy) await sleep(150);
      if (run !== st.overviewRun) return;
      let res;
      try { res = await st.ze.patternStats(part); } catch (e) { log(String(e.message || e), true); return; }
      if (run !== st.overviewRun) return;
      read += part.length;
      // each value fills its lattice cell, and the rows below it no finer pass has read yet
      for (let k = 0; k < part.length; k++) {
        const r0 = Math.floor(part[k] / cols), c0 = part[k] % cols;
        const r1 = Math.min(rows, r0 + stride * step), c1 = Math.min(cols, c0 + stride);
        for (let r = r0; r < r1; r++) st.overview.fill(res.std[k], r * cols + c0, r * cols + c1);
      }
      if (performance.now() - lastDraw > 100 && (st.mapKind === "overview" || !st.results)) { drawMap(); lastDraw = performance.now(); }
    }
  }
  if (st.mapKind === "overview" || !st.results) drawMap();
  const io = st.ze.stats();
  log(`overview: ${read.toLocaleString()} patterns${stride > 1 ? ` (every ${stride}${stride === 2 ? "nd" : stride === 3 ? "rd" : "th"} each way)` : ""} in ${((performance.now() - t0) / 1000).toFixed(2)} s, ${(io.fileBytes / 2 ** 20).toFixed(0)} MiB in ${io.fileReads.toLocaleString()} reads`);
}

// ── phases ──

/** What a change of the candidate phases leaves standing: nothing found with the old ones. */
function phasesChanged() {
  st.results = null;
  st.fit = null;
  st.keys.clear();
  done("index", false);
  showPhases();
  if (st.mapKind !== "overview") setMapKind("overview"); else drawMap();
  showMaster();
}

async function addPhases(files) {
  await busy("reading master pattern…", async () => {
    for (const f of files) { await st.ze.addPhase(f); st.phaseFiles.push(f); }
    phasesChanged();
  });
}

/** Remove one phase: the engine drops them all and takes the others again. */
async function removePhase(i) {
  await busy("removing the phase…", async () => {
    const keep = st.phaseFiles.filter((_, k) => k !== i);
    st.ze.clearPhases();
    st.phaseFiles = [];
    for (const f of keep) { await st.ze.addPhase(f); st.phaseFiles.push(f); }
    phasesChanged();
  });
}

function showPhases() {
  const list = st.ze.phases;
  if (st.phase >= list.length) st.phase = 0;
  done("data", !!st.scan && list.length > 0);
  $("btn-toy").disabled = list.length === 0;
  $("btn-toy").title = list.length ? "Make the scan from the master patterns above, in place of any scan open" : "Needs at least one master pattern";
  $("phases").replaceChildren(...list.map((p, i) => {
    const li = document.createElement("li");
    li.setAttribute("aria-selected", String(i === st.phase));
    li.title = `${p.name}: point group ${p.pointGroupName}, Laue class ${p.laue}, ${p.side} × ${p.side} per hemisphere${p.voltageKv ? `, ${p.voltageKv} kV` : ""}`;
    li.innerHTML = `<span class="dot" style="background:${PHASE_COLORS[i % PHASE_COLORS.length]}"></span><span class="nm"></span><span class="pg"></span><button type="button" class="ghost" title="Remove this phase"><svg><use href="#i-x"/></svg></button>`;
    li.querySelector(".nm").textContent = p.name;
    li.querySelector(".pg").textContent = p.pointGroupName;
    li.addEventListener("click", () => { st.phase = i; showPhases(); showMaster(); });
    li.querySelector("button").addEventListener("click", (e) => { e.stopPropagation(); removePhase(i); });
    return li;
  }));
  updateEnabled();
}

async function showMaster() {
  if (!st.ze.phases.length || $("dock-master").hidden) return;
  try {
    const { side, data } = await st.ze.masterImage(st.phase);
    // the two hemispheres side by side
    const both = new Float32Array(2 * side * side);
    for (let r = 0; r < side; r++) {
      both.set(data.subarray(r * side, (r + 1) * side), r * 2 * side);
      both.set(data.subarray(side * side + r * side, side * side + (r + 1) * side), r * 2 * side + side);
    }
    drawGray($("master"), both, 2 * side, side);
    $("dock-meta").textContent = `${st.ze.phases[st.phase].name} · ${side} × ${side} per hemisphere`;
  } catch (e) { log(String(e.message || e), true); }
}

// ── the map ──

function setMapKind(kind) {
  st.mapKind = kind;
  for (const b of $("map-kind").children) b.setAttribute("aria-pressed", String(b.dataset.kind === kind));
  drawMap();
}

function drawMap() {
  if (!st.scan) return;
  const { rows, cols } = st.scan;
  const canvas = $("map");
  const dim = st.mapKind !== "overview" && st.results;
  drawGray(canvas, st.overview, cols, rows);
  if (dim) {
    const ctx = canvas.getContext("2d");
    // results over a dimmed overview, so cells not indexed yet still show the microstructure
    ctx.fillStyle = "rgba(10,10,11,0.72)";
    ctx.fillRect(0, 0, cols, rows);
    const img = new ImageData(st.ze.map(st.mapKind), cols, rows);
    const tmp = new OffscreenCanvas(cols, rows);
    tmp.getContext("2d").putImageData(img, 0, 0);
    ctx.drawImage(tmp, 0, 0);
  }
  layoutMap();
  drawMapOverlay();
  drawLegend();
}

/** What the map's colours mean, in its corner. */
function drawLegend() {
  const box = $("map-legend");
  box.replaceChildren();
  const kind = st.mapKind;
  const R = st.results;
  box.dataset.folded = st.legendFolded ? "1" : "0";
  const text = (t) => { const s = document.createElement("span"); s.textContent = t; return s; };
  const chip = (color, t) => { const s = document.createElement("span"); s.className = "chip"; const i = document.createElement("i"); i.style.background = color; s.append(i, t); return s; };
  const bar = () => { const b = document.createElement("span"); b.className = "bar"; return b; };
  if (!st.scan) { box.hidden = true; return; }
  if (kind === "overview" || !R) {
    box.append(text("pattern contrast"), bar());
  } else if (kind === "ncc" || kind === "nmi" || kind === "z") {
    // gray between the indexed cells' extremes, as the engine draws it
    let lo = Infinity, hi = -Infinity;
    const v = R[kind];
    for (let i = 0; i < v.length; i++) if (R.phase[i] >= 0) { if (v[i] < lo) lo = v[i]; if (v[i] > hi) hi = v[i]; }
    const f = (x) => (kind === "z" ? x.toFixed(1) : x.toFixed(3));
    box.append(text(`${kind === "z" ? "z" : kind.toUpperCase()}  ${Number.isFinite(lo) ? f(lo) : ""}`), bar(), text(Number.isFinite(hi) ? f(hi) : ""));
  } else if (kind === "phase") {
    st.ze.phases.forEach((p, i) => box.append(chip(PHASE_COLORS[i % PHASE_COLORS.length], p.name)));
  } else if (kind === "sweep") {
    st.ze.sweepPlan.forEach((p, i) => box.append(chip(PHASE_COLORS[i % PHASE_COLORS.length], `${i + 1}: ${sweepLabel(p)}`)));
    box.append(chip("#3a3a3e", "none accepted"));
  } else {
    // one key per Laue class among the phases
    const seen = new Set();
    st.ze.phases.forEach((p, i) => {
      // (two point groups of one Laue class can differ in their rotations, and so in their key)
      const id = `${p.laue}/${p.pointGroup}`;
      if (seen.has(id)) return;
      seen.add(id);
      box.append(ipfKey(i, p.laue, id));
    });
    box.append(text(`of sample ${kind.slice(-1)}`));
  }
  box.hidden = false;
}

/** The inverse pole figure key of a phase: the engine's image of the directions that stand for
 *  all the others, cropped to them, with the directions at its corners named. A key that reaches
 *  into the lower hemisphere comes as two, the upper half and the lower. */
function ipfKey(phase, laue, id) {
  let key = st.keys.get(id);
  if (!key) {
    key = [st.ze.ipfKey(phase, 72), st.ze.ipfKey(phase, 72, true)].filter((k) => k.box[2] > k.box[0] || k.box[3] > k.box[1]);
    st.keys.set(id, key);
  }
  const wrap = document.createElement("span");
  wrap.className = "keys";
  key.forEach((k, half) => wrap.append(ipfKeyHalf(k, laue, half === 1, key.length > 1)));
  return wrap;
}

function ipfKeyHalf(key, laue, lower, both) {
  const { rgba, size, box: [x0, y0, x1, y1], wedge } = key;
  const s = (size - 1) / Math.max(x1 - x0, y1 - y0);
  const w = Math.min(size, Math.ceil((x1 - x0) * s) + 1), h = Math.min(size, Math.ceil((y1 - y0) * s) + 1);
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const full = new OffscreenCanvas(size, size);
  full.getContext("2d").putImageData(new ImageData(rgba, size, size), 0, 0);
  canvas.getContext("2d").drawImage(full, 0, size - h, w, h, 0, 0, w, h);
  const wrap = document.createElement("span");
  wrap.className = "key";
  wrap.style.margin = "11px 14px 11px 12px";
  const scheme = { classic: "classic", hue: "white-centred", perfect: "Nolze and Hielscher, by direction", perfect_rotation: "Nolze and Hielscher, by orientation" }[st.settings.ipf_colors];
  wrap.title = `Inverse pole figure colours (${scheme}), Laue class ${laue}: the crystal direction along the sample axis, in stereographic projection${both ? (lower ? "; the lower hemisphere, seen from above" : "; the upper hemisphere") : ""}`;
  wrap.append(canvas);
  // a name at a point of the projection, set off from it by (dx, dy) in ems
  const name = (t, x, y, dx, dy) => {
    const el = document.createElement("span");
    el.textContent = t;
    el.style.left = `${(x - x0) * s}px`;
    el.style.top = `${h - 1 - (y - y0) * s}px`;
    el.style.transform = `translate(calc(-50% + ${dx}em), calc(-50% + ${dy}em))`;
    wrap.append(el);
  };
  const inside = (x, y) => x >= x0 - 1e-3 && x <= x1 + 1e-3 && y >= y0 - 1e-3 && y <= y1 + 1e-3;
  if (!wedge) {
    // the cubic classes: the triangle, or two or four of it
    const t8 = Math.SQRT2 - 1, h3 = (Math.sqrt(3) - 1) / 2;
    const kite = x1 > 0.9;
    for (const [t, x, y, dx, dy] of [["001", 0, 0, -0.9, kite ? 0 : 0.9], ["101", t8, 0, kite ? 0 : 0.3, kite ? 0 : 0.9], ["111", h3, h3, 0.3, -0.9], ["011", 0, t8, -0.9, -0.3], ["100", 1, 0, 0.9, 0], ["1\u030511", h3, -h3, 0.3, 0.9]]) {
      if (inside(x, y) && !(kite && t === "101")) name(t, x, y, dx, dy);
    }
  } else {
    // a wedge about the c axis: its axis, the crystal's x where the wedge holds it, and its ends
    const [a0, a1] = wedge;
    name(lower ? "001\u0305" : "001", 0, 0, -0.3, y0 < -0.01 ? 0 : 0.9);
    const rad = Math.PI / 180;
    if (a1 - a0 < 359) {
      if (a0 <= 0 && a1 >= 0 && a0 > -1) name("100", 1, 0, 0.3, 0.9);
      else if (a0 <= 0 && a1 >= 0) name("100", 1, 0, 0.9, 0);
      if (Math.abs(a0) > 1) name(`${Math.round(a0)}°`, Math.cos(a0 * rad), Math.sin(a0 * rad), 0.9, a0 < 0 ? 0.9 : 0);
      name(`${Math.round(a1)}°`, Math.cos(a1 * rad), Math.sin(a1 * rad), a1 > 90 ? -0.3 : 0.3, -0.9);
    } else name("100", 1, 0, 0.9, 0);
  }
  if (both) {
    const cap = document.createElement("span");
    cap.textContent = lower ? "lower" : "upper";
    cap.style.left = "50%";
    cap.style.top = "100%";
    cap.style.transform = "translate(-50%, 0.2em)";
    cap.style.opacity = "0.7";
    wrap.append(cap);
  }
  return wrap;
}

/** Place the map and its overlay in the well for the current view. At zoom 1 the whole map
 *  shows, as large as fits; zooming never goes below that, and the map cannot be dragged off. */
function layoutMap() {
  const canvas = $("map"), over = $("map-over"), well = $("map-well");
  if (!st.scan || canvas.width < 2) return;
  const W = well.clientWidth, H = well.clientHeight;
  const s0 = Math.max(1e-6, Math.min(W / canvas.width, H / canvas.height));
  const v = st.view;
  // in to 48 screen pixels per cell
  v.k = Math.max(1, Math.min(v.k, Math.max(1, 48 / s0)));
  const s = s0 * v.k;
  const w = canvas.width * s, h = canvas.height * s;
  const hx = Math.min(W, w) / (2 * s), hy = Math.min(H, h) / (2 * s);
  v.cx = Math.max(hx, Math.min(canvas.width - hx, v.cx));
  v.cy = Math.max(hy, Math.min(canvas.height - hy, v.cy));
  v.s = s;
  for (const el of [canvas, over]) {
    el.style.left = `${W / 2 - v.cx * s}px`;
    el.style.top = `${H / 2 - v.cy * s}px`;
    el.style.width = `${w}px`;
    el.style.height = `${h}px`;
  }
  over.setAttribute("viewBox", `0 0 ${canvas.width} ${canvas.height}`);
  const zoomed = v.k > 1.0001;
  $("btn-zoom").disabled = !zoomed;
  $("btn-zoom").textContent = zoomed ? `whole map  ×${v.k.toFixed(1)}` : "whole map";
}

/** Zoom by `factor` keeping the map point under well position (px, py) where it is. */
function zoomMap(factor, px, py) {
  const well = $("map-well");
  const W = well.clientWidth, H = well.clientHeight;
  const v = st.view;
  const ux = v.cx + (px - W / 2) / v.s, uy = v.cy + (py - H / 2) / v.s;
  v.k *= factor;
  layoutMap();
  v.cx = ux - (px - W / 2) / v.s;
  v.cy = uy - (py - H / 2) / v.s;
  layoutMap();
  drawMapOverlay();
}

function resetZoom() {
  if (!st.scan) return;
  st.view.k = 1;
  layoutMap();
  drawMapOverlay();
}

function drawMapOverlay() {
  const over = $("map-over");
  over.replaceChildren();
  if (!st.scan) return;
  const { rows, cols } = st.scan;
  const s = st.settings;
  // one screen pixel in map cells: marks keep their size on screen at any zoom
  const u = 1 / st.view.s;
  if (s.roi_w > 0 && s.roi_h > 0) {
    over.append(svg("rect", { x: s.roi_x, y: s.roi_y, width: Math.min(s.roi_w, cols - s.roi_x), height: Math.min(s.roi_h, rows - s.roi_y), fill: "none", stroke: "#d6b064", "stroke-width": 1.5 * u, "stroke-dasharray": `${5 * u} ${4 * u}` }));
  }
  if (st.roiDrag) {
    const d = st.roiDrag;
    over.append(svg("rect", { x: Math.min(d.c0, d.c1), y: Math.min(d.r0, d.r1), width: Math.abs(d.c1 - d.c0) + 1, height: Math.abs(d.r1 - d.r0) + 1, fill: "rgba(214,176,100,.15)", stroke: "#e8c77e", "stroke-width": u }));
  }
  if ($("show-fit").checked) {
    const cells = st.fit ? st.fit.cells : fitLattice();
    for (const c of cells) {
      over.append(svg("circle", { cx: (c % cols) + 0.5, cy: Math.floor(c / cols) + 0.5, r: 3.5 * u, fill: st.fit ? "#7fbf6e" : "none", stroke: st.fit ? "#0a0a0b" : "#9d9da4", "stroke-width": u }));
    }
  }
  if (st.cell >= 0) {
    const x = st.cell % cols, y = Math.floor(st.cell / cols);
    over.append(svg("rect", { x: x - 2 * u, y: y - 2 * u, width: 1 + 4 * u, height: 1 + 4 * u, fill: "none", stroke: "#e8c77e", "stroke-width": 1.5 * u }));
  }
}

/** The lattice the fits will use (pcfit.zig fitCells), to show before a fit has run. */
function fitLattice() {
  const { rows, cols } = st.scan;
  const s = st.settings;
  let x0 = 0, y0 = 0, w = cols, h = rows;
  if (s.roi_w > 0 && s.roi_h > 0) { x0 = Math.min(s.roi_x, cols - 1); y0 = Math.min(s.roi_y, rows - 1); w = Math.min(s.roi_w, cols - x0); h = Math.min(s.roi_h, rows - y0); }
  const gy = Math.min(s.fit_grid, h), gx = Math.min(s.fit_grid, w);
  const out = [];
  for (let r = 0; r < gy; r++) for (let c = 0; c < gx; c++) out.push((y0 + Math.floor(((2 * r + 1) * h) / (2 * gy))) * cols + x0 + Math.floor(((2 * c + 1) * w) / (2 * gx)));
  return out;
}

function mapCellAt(ev) {
  const rect = $("map").getBoundingClientRect();
  const { rows, cols } = st.scan;
  const c = Math.floor(((ev.clientX - rect.left) / rect.width) * cols);
  const r = Math.floor(((ev.clientY - rect.top) / rect.height) * rows);
  if (c < 0 || r < 0 || c >= cols || r >= rows) return null;
  return { r, c, cell: r * cols + c };
}

function hoverText(p) {
  const parts = [`row ${p.r}  col ${p.c}`];
  const R = st.results;
  if (R && R.phase[p.cell] >= 0) parts.push(`NCC ${R.ncc[p.cell].toFixed(3)}`, `NMI ${R.nmi[p.cell].toFixed(3)}`, `z ${R.z[p.cell].toFixed(1)}`);
  return parts.join("   ");
}

// ── the pattern ──

function selectCell(cell) {
  st.cell = cell;
  drawMapOverlay();
  showCell();
}

/** The orientation the engine has for a cell: from indexing, else from the last fit. */
function orientationOf(cell) {
  const R = st.results;
  if (R && R.phase[cell] >= 0) {
    return { phase: R.phase[cell], q: Array.from(R.quaternions.subarray(4 * cell, 4 * cell + 4)), ncc: R.ncc[cell], nmi: R.nmi[cell], z: R.z[cell], sweep: R.sweep[cell],
      euler: Array.from(R.euler.subarray(3 * cell, 3 * cell + 3)), from: "indexed" };
  }
  if (st.fit) {
    const k = st.fit.cells.indexOf(cell);
    if (k >= 0 && st.fit.quats.length >= 4 * (k + 1)) return { phase: st.fit.phases[k], q: Array.from(st.fit.quats.subarray(4 * k, 4 * k + 4)), ncc: st.fit.ncc[k], nmi: null, euler: null, from: "fit" };
  }
  return null;
}

/** Index one cell on the spot, as the fits index their patterns (the spherical correlation at
 *  the fit's bandwidth, then the refinement), under the settings and geometry as they are. */
async function indexNow(cell) {
  const hit = st.live.get(cell);
  if (hit) return hit;
  const recs = await st.ze.indexCells([cell], { method: "spherical", bandwidth: st.settings.fit_bandwidth });
  // the phase as the engine chooses it (the phase_by setting)
  const by = st.settings.phase_by === "ncc" ? 4 : 5;
  let p = 0;
  for (let k = 1; k < recs.length / 8; k++) if (recs[8 * k + by] > recs[8 * p + by]) p = k;
  const r = recs.subarray(8 * p, 8 * p + 8);
  const o = { phase: p, q: Array.from(r.subarray(0, 4)), ncc: r[4], nmi: r[5], z: r[7], sweep: null, euler: null, from: `found now, L ${st.settings.fit_bandwidth}` };
  if (st.live.size > 4096) st.live.clear();
  st.live.set(cell, o);
  return o;
}

function nccOf(a, b, mask) {
  let n = 0, sa = 0, sb = 0;
  for (let i = 0; i < a.length; i++) if (mask[i]) { n++; sa += a[i]; sb += b[i]; }
  const ma = sa / n, mb = sb / n;
  let ab = 0, aa = 0, bb = 0;
  for (let i = 0; i < a.length; i++) if (mask[i]) { const x = a[i] - ma, y = b[i] - mb; ab += x * y; aa += x * x; bb += y * y; }
  return ab / Math.sqrt(Math.max(aa * bb, 1e-30));
}

/** What the averaging did to the pattern shown: in words in the rail, and as the little map of
 *  its neighbours' shares in the pattern's corner. */
function showNlpar(a) {
  const inset = $("pat-shares");
  inset.hidden = !a;
  if (!a) return;
  const shares = Float32Array.from(a.shares, (v) => (Number.isFinite(v) ? v : 0));
  let top = 0;
  for (const v of shares) if (v > top) top = v;
  drawHeat(inset, shares, a.side, a.side, 0, Math.max(top, 1e-9));
  // (cells outside the map: dark)
  const ctx = inset.getContext("2d");
  ctx.fillStyle = "#0a0a0b";
  for (let i = 0; i < a.shares.length; i++) if (!Number.isFinite(a.shares[i])) ctx.fillRect(i % a.side, Math.floor(i / a.side), 1, 1);
  const px = Math.max(4, Math.min(9, Math.floor(72 / a.side)));
  inset.style.width = `${a.side * px}px`;
  inset.style.height = `${a.side * px}px`;
  const more = a.shares.filter((v) => v > 0.01).length - 1;
  $("nlpar-readout").textContent = `the selected pattern\nnoise ${a.noise.toFixed(a.noise < 10 ? 2 : 1)} a pixel as stored, ${(100 * a.left).toFixed(0)} % of it left\n${(100 * a.own).toFixed(0)} % of its average is the pattern itself\n${Math.max(more, 0)} of its ${a.side * a.side - 1} neighbours hold over 1 % of it`;
}

async function showCell() {
  if (!st.scan || st.cell < 0) return;
  const run = ++st.showRun;
  const cell = st.cell;
  const { cols } = st.scan;
  const s = st.settings;
  try {
    // at the reduced size, or (when patterns are reduced and it is asked for) the stored one
    const full = $("pat-full").checked && s.binning !== 1;
    const [h, w] = st.ze.shape(full);
    const stage = st.stage;
    const exp = await st.ze.pattern(cell, stage, full);
    if (run !== st.showRun) return;
    // the mask the engine scores under (settings mask, mask_radius)
    const mask = new Uint8Array(h * w).fill(1);
    if (s.mask) {
      const rad = s.mask_radius * 0.5 * Math.min(h, w);
      for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) mask[r * w + c] = (r - (h - 1) / 2) ** 2 + (c - (w - 1) / 2) ** 2 <= rad * rad ? 1 : 0;
    }
    drawGray($("pat-exp"), exp, w, h, stage === "filtered" ? { mask } : {});
    drawPc();
    showNlpar(exp.nlpar);
    const win = `${2 * s.nlpar_radius + 1} × ${2 * s.nlpar_radius + 1}${s.nlpar_iterations > 1 ? `, ${s.nlpar_iterations} passes` : ""}`;
    $("pat-exp-cap").textContent = `experimental · ${stage === "stored" ? "as stored" : [s.nlpar ? `averaged ${win}` : "", stage === "filtered" ? "filtered" : ""].filter(Boolean).join(", ")}`;
    const facts = [];
    $("pat-cell").textContent = `row ${Math.floor(cell / cols)}, col ${cell % cols}`;
    $("pat-cell").title = `map cell ${cell}; shown at ${h} × ${w}`;
    let o = orientationOf(cell);
    const nPh = st.ze.phases.length;
    if (!o && nPh && $("pat-live").checked && !st.busy) {
      o = await indexNow(cell);
      if (run !== st.showRun) return;
    }
    if (o && nPh) {
      const sim = await st.ze.simulate(o.phase, [cell], o.q, full);
      if (run !== st.showRun) return;
      drawGray($("pat-sim"), sim, w, h, { mask });
      // difference of the two, each zero-mean and unit-norm under the mask (what the NCC compares)
      const proc = stage === "filtered" ? exp : await st.ze.pattern(cell, "filtered", full);
      if (run !== st.showRun) return;
      const norm = (a) => { let n = 0, t = 0; for (let i = 0; i < a.length; i++) if (mask[i]) { n++; t += a[i]; } const m = t / n; let ss = 0; const out = new Float32Array(a.length); for (let i = 0; i < a.length; i++) if (mask[i]) { out[i] = a[i] - m; ss += out[i] * out[i]; } const k = 1 / Math.sqrt(Math.max(ss, 1e-30)); for (let i = 0; i < a.length; i++) out[i] *= k; return out; };
      const a = norm(proc), b = norm(sim);
      drawSigned($("pat-diff"), a.map((v, i) => v - b[i]), w, h);
      const ph = st.ze.phases[o.phase];
      $("pat-sim-cap").textContent = `simulated · ${ph.name} · ${o.from}`;
      if (nPh > 1) facts.push(["phase", ph.name]);
      facts.push(["NCC", nccOf(proc, sim, mask).toFixed(4)]);
      if (o.nmi !== null) {
        facts.push(["NMI", `${o.nmi.toFixed(4)},  z ${o.z.toFixed(1)} above chance`]);
        if (o.sweep !== null) facts.push(["accepted", o.sweep >= 0 ? (s.sweeps > 1 ? `in sweep ${o.sweep + 1}` : "yes") : ((s.accept_nmi > 0 ? o.nmi < s.accept_nmi : o.z < s.accept_z) ? `no (below ${acceptText()})` : "no (refinement not settled)")]);
      }
      // (as the results store an orientation: its representative in the fundamental zone)
      const red = st.ze.reduced(o.phase, o.q);
      facts.push(["Bunge (°)", red.euler.map((v) => ((v * 180) / Math.PI).toFixed(2)).join("  ")]);
      facts.push(["quaternion", red.q.map((v) => v.toFixed(4)).join("  ")]);
    } else {
      for (const id of ["pat-sim", "pat-diff"]) { const c = $(id); c.width = w; c.height = h; c.getContext("2d").clearRect(0, 0, w, h); }
      $("pat-sim-cap").textContent = !nPh ? "simulated · add a phase" : st.busy ? "simulated · when the job is done" : "simulated · fit or index to see it";
    }
    if (exp.nlpar) facts.push(["averaging", `${(100 * exp.nlpar.left).toFixed(0)} % of the noise left`]);
    $("pat-facts").replaceChildren(...factRows(facts));
  } catch (e) {
    if (run === st.showRun) { status(String(e.message || e), true); log(String(e.message || e), true); }
  }
}

/** The pattern centre on the experimental pattern (Bruker convention: fractions from the left and top). */
function drawPc() {
  const over = $("pat-over");
  const canvas = $("pat-exp");
  over.replaceChildren();
  if (!st.scan || canvas.width < 2) return;
  fit(canvas, over);
  const pc = st.ze.geometry.pc;
  // the fractions are of the stored pattern; a reduction that dropped a remainder shows less of it
  const full = $("pat-full").checked && st.settings.binning !== 1;
  const f = full ? 1 : st.settings.binning;
  const x = (pc[0] * st.scan.patW) / f - 0.5, y = (pc[1] * st.scan.patH) / f - 0.5;
  const r = canvas.width / 30;
  over.append(svg("path", { d: `M${x - r} ${y}H${x + r}M${x} ${y - r}V${y + r}`, stroke: "#e8c77e", "stroke-width": canvas.width / 220, fill: "none" }));
  over.append(svg("circle", { cx: x, cy: y, r: r * 0.55, stroke: "#e8c77e", "stroke-width": canvas.width / 300, fill: "none" }));
}

// ── fits ──

function drawVotes() {
  votes.show(st.votes, { start: st.pcStart, file: st.pcFile });
}

async function fitPc() {
  await busy("fitting the pattern centre…", async () => {
    beforeFit();
    const r = await st.ze.fitPc();
    st.fit = st.ze.fitPatterns();
    st.votes = st.ze.pcLevels();
    st.pcStart = r.start;
    $("fit-readout").textContent = `pattern centre  ${r.pc.map((v) => v.toFixed(5)).join("  ")}\nmean NCC ${r.score.toFixed(4)} (start ${r.startScore.toFixed(4)})  ·  bandwidth ${st.settings.fit_bandwidth}\n${r.nCandidates.toLocaleString()} candidates × ${r.nPatterns} patterns${r.atBorder ? "\nthe best candidate is at the search's border: widen it, or fit again from here" : ""}`;
    $("fit-readout").textContent += againstTrueGeometry();
    done("geometry");
    selectDock("votes");
    drawVotes();
    geometryChanged(true);
    drawMapOverlay();
  });
}

async function fitPose() {
  await busy("fitting the pose…", async () => {
    beforeFit();
    const r = await st.ze.fitPose();
    st.fit = st.ze.fitPatterns();
    const deg = (v) => ((v * 180) / Math.PI).toFixed(3);
    $("fit-readout").textContent = `pattern centre  ${r.pc.map((v) => v.toFixed(5)).join("  ")}\nswing, tilt, twist (°)  ${r.rot.map(deg).join("  ")}${r.pxUm > 0 ? `  ·  pixel ${r.pxUm.toFixed(1)} µm` : "\nno pixel size: every pattern is taken from one point, where the detector's rotation cannot be told from the crystals', so it is left as set"}\nmean NCC ${r.ncc.toFixed(4)} (start ${r.startNcc.toFixed(4)})  ·  ${r.iterations} iterations${r.converged ? "" : ", not converged"}  ·  ${r.nPatterns} patterns`;
    $("fit-readout").textContent += againstTrueGeometry();
    done("geometry");
    geometryChanged(true);
    drawMapOverlay();
  });
}

// ── indexing ──

async function runIndex() {
  await busy("indexing…", async () => {
    await st.ze.indexBegin();
    if (st.mapKind === "overview") setMapKind("ipf_z");
    const t0 = performance.now();
    let last = 0, visited = 0, s = null;
    for (;;) {
      const before = st.ze.indexStatus;
      s = await st.ze.indexStep();
      visited += Math.min(before.total - before.done, st.settings.batch);
      const dt = (performance.now() - t0) / 1000;
      progress(s.done / Math.max(s.total, 1));
      const head = s.sweeps > 1 ? `sweep ${s.sweep + 1} of ${s.sweeps}, ${sweepLabel(s)}\n` : "";
      $("index-readout").textContent = `${head}${s.done.toLocaleString()} of ${s.total.toLocaleString()} patterns  ·  ${(visited / Math.max(dt, 1e-6)).toFixed(0)} patterns/s\n${s.accepted.toLocaleString()} of ${s.region.toLocaleString()} accepted (${acceptText()})`;
      if (performance.now() - last > 250 || s.left === 0) { st.results = st.ze.results(); drawMap(); last = performance.now(); await sleep(0); }
      if (s.left === 0 || st.stop) break;
    }
    st.results = st.ze.results();
    $("index-readout").textContent += againstTruth();
    // what chance gives on this scan: the scale z is read on, measured once for the run
    try { st.chance = await st.ze.chance(0); } catch { st.chance = null; }
    showSweeps();
    done("index");
    drawMap();
    // (with no search to show, the dock shows how the matches stand)
    if (!st.votes) selectDock("z"); else drawZ();
    updateEnabled();
    const R = st.results;
    let n = 0, sum = 0;
    for (let i = 0; i < R.ncc.length; i++) if (R.phase[i] >= 0) { n++; sum += R.ncc[i]; }
    log(`indexed ${n.toLocaleString()} patterns in ${((performance.now() - t0) / 1000).toFixed(2)} s${s.sweeps > 1 ? ` (${visited.toLocaleString()} pattern visits over ${s.sweep + 1} sweeps)` : ""}: mean NCC ${(sum / Math.max(n, 1)).toFixed(4)}, ${s.accepted.toLocaleString()} accepted`);
  });
}

/** Fit the batch size and the cutting-up of the GPU's work to this machine and scan. */
async function runTune() {
  await busy("tuning…", async () => {
    st.tuned = "";
    await st.ze.tune(() => !st.stop);
    st.settings = st.ze.settings;
    fields.show(st.settings);
    // the engine's own summary, a setting to a line
    $("index-readout").textContent = st.tuned.replace(/, (?=batch |refine_pixels |wg_)/g, "\n").replace("; ", "\n");
  });
}

/** How the indexed cells' matches stand above chance: a histogram of z with the threshold. */
function drawZ() {
  const canvas = $("zhist");
  const R = st.results;
  if ($("dock-z").hidden) return;
  const well = canvas.parentElement;
  const dpr = window.devicePixelRatio || 1;
  const W = Math.max(50, well.clientWidth), H = Math.max(40, well.clientHeight);
  canvas.width = Math.round(W * dpr);
  canvas.height = Math.round(H * dpr);
  canvas.style.width = `${W}px`;
  canvas.style.height = `${H}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const zs = [];
  if (R) for (let i = 0; i < R.z.length; i++) if (R.phase[i] >= 0) zs.push(R.z[i]);
  if (!zs.length) { $("z-hint").textContent = "Index to see how the matches stand above chance."; return; }
  // the threshold on z's scale: the calibration's own, or the absolute NMI's place on it
  const t = st.settings.accept_nmi > 0 && st.chance ? (st.settings.accept_nmi - st.chance.median) / st.chance.spread : st.settings.accept_z;
  const [q0, q1] = quantiles(Float32Array.from(zs), 0.002, 0.998);
  const lo = Math.min(q0, 0, t) - 1, hi = Math.max(q1, t) + 1;
  const nb = Math.max(20, Math.min(120, Math.floor(W / 6)));
  const counts = new Float64Array(nb);
  let accepted = 0;
  for (const z of zs) {
    if (z >= t) accepted++;
    counts[Math.max(0, Math.min(nb - 1, Math.floor(((z - lo) / (hi - lo)) * nb)))]++;
  }
  const top = Math.max(...counts);
  const pad = { l: 8, r: 8, t: 8, b: 18 };
  const x = (z) => pad.l + ((z - lo) / (hi - lo)) * (W - pad.l - pad.r);
  const bw = (W - pad.l - pad.r) / nb;
  for (let b = 0; b < nb; b++) {
    const zc = lo + ((b + 0.5) / nb) * (hi - lo);
    const h = (counts[b] / top) * (H - pad.t - pad.b);
    ctx.fillStyle = zc >= t ? "#d6b064" : "#55555c";
    ctx.fillRect(pad.l + b * bw + 0.5, H - pad.b - h, Math.max(1, bw - 1), h);
  }
  ctx.strokeStyle = "#e6e6e8";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(Math.round(x(t)) + 0.5, pad.t);
  ctx.lineTo(Math.round(x(t)) + 0.5, H - pad.b);
  ctx.stroke();
  ctx.fillStyle = "#9d9da4";
  ctx.font = "11px ui-monospace, Consolas, monospace";
  ctx.textBaseline = "top";
  const step = [1, 2, 5, 10, 20, 50, 100, 200, 500].find((d) => (hi - lo) / d <= 10) || 1000;
  for (let z = Math.ceil(lo / step) * step; z <= hi; z += step) {
    ctx.textAlign = "center";
    ctx.fillText(String(z), x(z), H - pad.b + 4);
  }
  const sorted = zs.slice().sort((a, b) => a - b);
  $("z-hint").textContent = `z of ${zs.length.toLocaleString()} indexed cells: median ${sorted[sorted.length >> 1].toFixed(1)}; ${accepted.toLocaleString()} (${((100 * accepted) / zs.length).toFixed(1)} %) at or above the threshold ${st.settings.accept_nmi > 0 ? `NMI ${st.settings.accept_nmi} (z ${t.toFixed(1)})` : t}`;
}

// ── docks, enabling, files ──

function selectDock(kind) {
  for (const b of $("dock-kind").children) b.setAttribute("aria-pressed", String(b.dataset.dock === kind));
  for (const k of ["votes", "master", "z", "log"]) $(`dock-${k}`).hidden = k !== kind;
  $("dock-meta").textContent = "";
  $("votes-meta").hidden = kind !== "votes";
  if (kind === "master") showMaster();
  if (kind === "votes") votes.draw();
  if (kind === "z") drawZ();
  if (kind === "log") $("log").scrollTop = $("log").scrollHeight;
}

function updateEnabled() {
  const ready = !!st.scan && st.ze.phases.length > 0;
  for (const id of ["btn-fit-pc", "btn-fit-pose", "btn-index", "btn-tune"]) $(id).disabled = !ready;
  for (const b of document.querySelectorAll(".needs-results")) b.disabled = !st.results;
  for (const b of $("map-kind").children) if (RESULT_KINDS.includes(b.dataset.kind)) b.disabled = !st.results;
  $("btn-json").disabled = !st.scan;
  done("data", ready);
  done("patterns", ready);
}

function isMaster(f) {
  return /\.npz$/i.test(f.name) || (/master/i.test(f.name) && /\.(h5|hdf5)$/i.test(f.name));
}

/** Files picked or dropped: a project (.json, as the Index step saves one), a scan with its
 *  side files, master patterns. The project's settings go in before the scan opens (they may
 *  say how to read it) and its geometry after (over what the scan's file gives). */
async function takeFiles(files) {
  const projects = files.filter((f) => /\.json$/i.test(f.name));
  const masters = files.filter((f) => !projects.includes(f) && isMaster(f));
  const scans = files.filter((f) => !projects.includes(f) && !masters.includes(f));
  let project = null;
  if (projects.length) {
    try {
      project = JSON.parse(await projects[0].text());
      if (!project || typeof project !== "object" || (!project.settings && !project.geometry)) throw new Error("no settings or geometry in it");
    } catch (e) {
      status(`${projects[0].name}: not a zebsdsolve project (${e.message || e})`, true);
      return;
    }
    if (project.settings) {
      try { st.ze.configure(project.settings); } catch (e) { status(String(e.message || e), true); log(String(e.message || e), true); return; }
      afterSettings(Object.keys(project.settings));
    }
  }
  if (scans.length) await openScan(scans);
  if (masters.length) await addPhases(masters);
  if (project) {
    if (project.geometry) { beforeFit(); st.ze.setGeometry(project.geometry); geometryChanged(); }
    log(`project ${projects[0].name}: ${[project.settings && "settings", project.geometry && "geometry"].filter(Boolean).join(" and ")} taken`);
    if (!st.busy) status(`${projects[0].name}: ${[project.settings && "settings", project.geometry && "geometry"].filter(Boolean).join(" and ")} taken`);
  }
}

/** A toy scan from the phases loaded: the engine makes it and hands back its answer. */
async function makeToy() {
  await busy("making a toy scan…", async () => {
    const truth = await st.ze.synthesize();
    st.scanName = `toy scan, seed ${st.ze.settings.toy_seed}`;
    afterOpen();
    st.truth = truth;
  });
}

/** How the indexed orientations and phases stand against a toy scan's own answer. */
function againstTruth() {
  const R = st.results, T = st.truth;
  if (!R || !T || T.phase.length !== R.phase.length) return "";
  const off = [];
  let crystals = 0, right = 0, blank = 0, blankAccepted = 0;
  for (let i = 0; i < R.phase.length; i++) {
    if (R.phase[i] < 0) continue;
    if (T.phase[i] < 0) { blank++; if (R.sweep[i] >= 0) blankAccepted++; continue; }
    crystals++;
    if (R.phase[i] !== T.phase[i]) continue;
    right++;
    off.push((st.ze.disorientation(R.phase[i], R.quaternions.subarray(4 * i, 4 * i + 4), T.quaternions.subarray(4 * i, 4 * i + 4)) * 180) / Math.PI);
  }
  if (!crystals && !blank) return "";
  off.sort((a, b) => a - b);
  const within = off.filter((d) => d < 1).length;
  let text = "\nagainst the toy scan's answer: ";
  if (crystals) text += `${st.ze.phases.length > 1 ? `phase right for ${((100 * right) / crystals).toFixed(1)} %; ` : ""}orientations a median ${off.length ? off[off.length >> 1].toFixed(3) : "–"}° off, ${((100 * within) / crystals).toFixed(1)} % within 1°`;
  if (blank) text += `${crystals ? "; " : ""}${blankAccepted} of ${blank} blank cells accepted`;
  return text;
}

/** How a fitted geometry stands against the one a toy scan was made under. */
function againstTrueGeometry() {
  const T = st.truth && st.truth.geometry;
  if (!T) return "";
  const g = st.ze.geometry;
  const pc = Math.max(...g.pc.map((v, i) => Math.abs(v - T.pc[i])));
  const rot = Math.max(...g.rot.map((v, i) => Math.abs(v - T.rot[i]))) * 180 / Math.PI;
  return `\nagainst the toy scan's geometry: pattern centre within ${pc.toFixed(5)}, rotation within ${rot.toFixed(3)}°${T.pxUm > 0 && g.pxUm > 0 ? `, pixel ${(100 * (g.pxUm / T.pxUm - 1)).toFixed(2)} % off` : ""}`;
}

async function loadExample() {
  await busy("loading the example…", async () => {
    const get = async (name) => {
      const r = await fetch(`./presets/nickel/${name}`);
      if (!r.ok) throw new Error(`example file ${name} is not on this server`);
      return new File([await r.blob()], name);
    };
    const [scan, master] = await Promise.all([get("patterns_v2.h5"), get("Ni_master_20kV.npz")]);
    await st.ze.openScan([scan]);
    st.scanName = scan.name;
    st.ze.clearPhases();
    await st.ze.addPhase(master);
    st.phaseFiles = [master];
    afterOpen();
    phasesChanged();
  });
}

function projectJson() {
  return JSON.stringify({ zebsdsolve: st.ze.version, scan: st.scanName, geometry: st.ze.geometry, settings: st.ze.settings, phases: st.ze.phases.map((p) => p.name) }, null, 2);
}

// ── start ──

async function main() {
  buildGeometry();
  fields = buildFields(document, applySetting);
  votes = createVotes({
    geometry: () => st.ze.geometry,
    // the candidate under the cross-hair becomes the pattern centre, to fit again from there
    setPc: (pc) => { beforeFit(); st.ze.setGeometry({ pc }); geometryChanged(); },
  });
  updateWiring();
  try {
    st.ze = await Zebsdsolve.create({ wasm: new URL("../zebsdsolve/zebsdsolve.wasm", import.meta.url) });
  } catch (e) {
    status(String(e.message || e), true);
    $("map-empty").innerHTML = `<p><b>${String(e.message || e)}</b></p><p class="hint">zebsdsolve needs WebGPU and WebAssembly JavaScript Promise Integration: Chrome or Edge 137 or later.</p>`;
    return;
  }
  const ze = st.ze;
  window.ze = ze; // for the console and the browser test
  $("adapter").textContent = ze.adapter;
  st.settings = ze.settings;
  fields.show(st.settings);
  showGeometry();
  showSweeps();
  showStage();
  ze.onCompile(({ done: n, total }) => {
    $("compile").hidden = n >= total;
    $("compile").textContent = `compiling shaders ${n}/${total}`;
  });
  ze.onEvent((ev) => {
    if (ev.kind === "log") { log(ev.text); if (ev.text.startsWith("tuned on")) st.tuned = ev.text; }
    else if (ev.kind === "progress" && ev.stage === "pca") { progress(ev.done / Math.max(ev.total, 1)); status(`learning the dictionary's components: step ${ev.done} of ${ev.total}`); }
    else if (ev.kind === "progress" && ev.stage === "tune") { progress(ev.done / Math.max(ev.total, 1)); if (ev.done < ev.total) status(`tuning: ${TUNE_MOVES[ev.done]}…`); }
    else if (ev.kind === "progress") {
      // indexing draws its own; work outside a job (the background's average) clears its bar when done
      if (ev.stage !== "index") progress(!st.busy && ev.done >= ev.total ? null : ev.done / Math.max(ev.total, 1));
    }
    else if (ev.kind === "pc") status(`pattern centre, level ${ev.level + 1}: ${ev.pc.map((v) => v.toFixed(4)).join("  ")}   mean NCC ${ev.score.toFixed(4)}${ev.at_border ? "  (at the border: moving the grid)" : ""}`);
    else if (ev.kind === "pose") status(`pose fit, iteration ${ev.iter + 1}: mean NCC ${ev.ncc.toFixed(5)}   pattern centre ${ev.pc.map((v) => v.toFixed(5)).join("  ")}`);
  });
  updateEnabled();
  // the example scan is served by the site and by a checkout, not carried in the wheels
  fetch("./presets/nickel/Ni_master_20kV.npz", { method: "HEAD" }).then((r) => { $("btn-example").hidden = !r.ok; }, () => { $("btn-example").hidden = true; });
  status(`ready · zebsdsolve ${ze.version}`);
  log(`zebsdsolve ${ze.version} on ${ze.adapter}`);
}

/** The two bars between the panes: drag to give a pane more of the window, double-click for
 *  the default. The sizes are kept for the next visit. */
function wireSplits() {
  const stage = $("stage");
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem("zebsdsolve.layout") || "{}") || {}; } catch { /* no storage: the defaults */ }
  const apply = () => {
    for (const [prop, v] of [["--dock-h", saved.dock], ["--pat-w", saved.pat]]) {
      if (v > 0) stage.style.setProperty(prop, `${v}px`); else stage.style.removeProperty(prop);
    }
  };
  const keep = () => { try { localStorage.setItem("zebsdsolve.layout", JSON.stringify(saved)); } catch { /* not kept */ } };
  apply();
  const wire = (el, key, axis) => {
    el.addEventListener("mousedown", (e) => {
      if (e.button !== 0) return;
      e.preventDefault();
      document.body.dataset.resize = axis;
      el.dataset.drag = "1";
      const move = (ev) => {
        const r = stage.getBoundingClientRect();
        saved[key] = Math.round(axis === "row" ? Math.max(110, Math.min(r.height - 150, r.bottom - ev.clientY)) : Math.max(200, Math.min(r.width - 290, r.right - ev.clientX)));
        apply();
      };
      const up = () => {
        window.removeEventListener("mousemove", move);
        window.removeEventListener("mouseup", up);
        delete document.body.dataset.resize;
        el.dataset.drag = "0";
        keep();
      };
      window.addEventListener("mousemove", move);
      window.addEventListener("mouseup", up);
    });
    el.addEventListener("dblclick", () => { delete saved[key]; apply(); keep(); });
  };
  wire($("split-dock"), "dock", "row");
  wire($("split-pat"), "pat", "col");
}

function updateWiring() {
  $("tabs").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) selectTab(b.dataset.tab); });
  $("tabs").addEventListener("keydown", (e) => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    const at = TABS.indexOf($("tabs").querySelector('[aria-selected="true"]').dataset.tab);
    const next = TABS[(at + (e.key === "ArrowRight" ? 1 : TABS.length - 1)) % TABS.length];
    selectTab(next);
    $(`tab-${next}`).focus();
  });
  selectTab("data");
  wireSplits();
  $("btn-open").addEventListener("click", () => $("file-scan").click());
  $("file-scan").addEventListener("change", (e) => { const f = [...e.target.files]; e.target.value = ""; takeFiles(f); });
  $("btn-example").addEventListener("click", loadExample);
  $("btn-toy").addEventListener("click", makeToy);
  $("bin-choices").addEventListener("change", (e) => applySetting("binning", +e.target.value));
  $("btn-phase").addEventListener("click", () => $("file-phase").click());
  $("file-phase").addEventListener("change", (e) => { const f = [...e.target.files]; e.target.value = ""; addPhases(f); });
  $("btn-fit-pc").addEventListener("click", fitPc);
  $("btn-fit-pose").addEventListener("click", fitPose);
  $("btn-geo-undo").addEventListener("click", () => {
    if (!st.geoBefore) return;
    // back to before the fit, or forward to after it: the two change places
    const g = st.ze.geometry;
    const other = { pc: g.pc, rot: g.rot };
    st.ze.setGeometry(st.geoBefore);
    st.geoBefore = other;
    st.geoIsBefore = !st.geoIsBefore;
    geometryChanged();
  });
  $("btn-geo-file").addEventListener("click", () => {
    if (!st.geoOpened) return;
    beforeFit();
    st.ze.setGeometry(st.geoOpened);
    geometryChanged();
  });
  $("btn-index").addEventListener("click", runIndex);
  $("btn-tune").addEventListener("click", runTune);
  $("btn-stop").addEventListener("click", () => { st.stop = true; st.ze.cancel(); status("stopping…"); });
  $("btn-ang").addEventListener("click", () => download(st.scanName.replace(/\.[^.]+$/, "") + ".ang", st.ze.exportText("ang"), "text/plain"));
  $("btn-ctf").addEventListener("click", () => download(st.scanName.replace(/\.[^.]+$/, "") + ".ctf", st.ze.exportText("ctf"), "text/plain"));
  $("btn-json").addEventListener("click", () => download(st.scanName.replace(/\.[^.]+$/, "") + ".zebsdsolve.json", projectJson(), "application/json"));
  $("map-kind").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b && !b.disabled) setMapKind(b.dataset.kind); });
  $("dock-kind").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) selectDock(b.dataset.dock); });
  $("pat-stage").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b && !b.disabled) { st.stage = b.dataset.stage; showStage(); showCell(); } });
  $("show-fit").addEventListener("change", drawMapOverlay);
  $("map-legend").addEventListener("click", () => { st.legendFolded = !st.legendFolded; drawLegend(); });
  $("pat-live").addEventListener("change", showCell);
  $("pat-full").addEventListener("change", showCell);
  $("btn-zoom").addEventListener("click", resetZoom);

  // the map: scroll zooms about the pointer, a drag moves a zoomed map (and scrubs the selection
  // over one that shows whole), a click selects, shift-drag sets the region of interest, and
  // the arrow keys step the selection
  const map = $("map");
  const well = $("map-well");
  well.addEventListener("wheel", (e) => {
    if (!st.scan) return;
    e.preventDefault();
    const rect = well.getBoundingClientRect();
    zoomMap(Math.exp(-e.deltaY * (e.deltaMode === 1 ? 0.05 : 0.0015)), e.clientX - rect.left, e.clientY - rect.top);
  }, { passive: false });
  well.addEventListener("keydown", (e) => {
    const d = { ArrowLeft: [0, -1], ArrowRight: [0, 1], ArrowUp: [-1, 0], ArrowDown: [1, 0] }[e.key];
    if (!d || !st.scan || st.cell < 0) return;
    e.preventDefault();
    const { rows, cols } = st.scan;
    const k = e.shiftKey ? 10 : 1;
    const r = Math.max(0, Math.min(rows - 1, Math.floor(st.cell / cols) + k * d[0])), c = Math.max(0, Math.min(cols - 1, (st.cell % cols) + k * d[1]));
    if (r * cols + c !== st.cell) selectCell(r * cols + c);
  });
  map.addEventListener("dblclick", resetZoom);
  map.addEventListener("mousedown", (e) => {
    if (!st.scan || e.button !== 0) return;
    well.focus({ preventScroll: true });
    const p = mapCellAt(e);
    if (e.shiftKey) {
      if (p) { st.roiDrag = { r0: p.r, c0: p.c, r1: p.r, c1: p.c }; drawMapOverlay(); }
      e.preventDefault();
      return;
    }
    st.drag = { x: e.clientX, y: e.clientY, cx: st.view.cx, cy: st.view.cy, moved: false, pan: st.view.k > 1.0001 };
    if (!st.drag.pan && p) selectCell(p.cell);
    e.preventDefault();
  });
  map.addEventListener("mousemove", (e) => {
    if (!st.scan) return;
    const p = mapCellAt(e);
    $("map-hover").textContent = p ? hoverText(p) : "";
  });
  window.addEventListener("mousemove", (e) => {
    if (!st.scan) return;
    const d = st.drag;
    if (d) {
      const dx = e.clientX - d.x, dy = e.clientY - d.y;
      if (!d.moved && Math.hypot(dx, dy) > 4) { d.moved = true; if (d.pan) well.dataset.pan = "1"; }
      if (d.pan && d.moved) {
        st.view.cx = d.cx - dx / st.view.s;
        st.view.cy = d.cy - dy / st.view.s;
        layoutMap();
      } else if (!d.pan) {
        const p = mapCellAt(e);
        if (p && p.cell !== st.cell) selectCell(p.cell);
      }
    }
    if (st.roiDrag) {
      const p = mapCellAt(e);
      if (p) { st.roiDrag.r1 = p.r; st.roiDrag.c1 = p.c; drawMapOverlay(); }
    }
  });
  window.addEventListener("mouseup", (e) => {
    const d = st.drag;
    st.drag = null;
    well.dataset.pan = "0";
    if (d && d.pan && !d.moved) { const p = mapCellAt(e); if (p) selectCell(p.cell); }
    const roi = st.roiDrag;
    if (!roi) return;
    st.roiDrag = null;
    const x = Math.min(roi.c0, roi.c1), y = Math.min(roi.r0, roi.r1), w = Math.abs(roi.c1 - roi.c0) + 1, h = Math.abs(roi.r1 - roi.r0) + 1;
    // a click without a drag clears the region
    const set = w < 2 && h < 2 ? { roi_x: 0, roi_y: 0, roi_w: 0, roi_h: 0 } : { roi_x: x, roi_y: y, roi_w: w, roi_h: h };
    try { st.ze.configure(set); } catch (err) { status(String(err.message || err), true); }
    st.fit = null;
    afterSettings(Object.keys(set));
  });
  map.addEventListener("mouseleave", () => { $("map-hover").textContent = ""; });
  // whatever is laid out by hand follows its pane
  new ResizeObserver(() => { if (st.scan) { layoutMap(); drawMapOverlay(); } }).observe(well);
  new ResizeObserver(() => { if (st.scan) drawPc(); }).observe($("pats"));
  new ResizeObserver(() => { if (!$("dock-votes").hidden) votes.draw(); drawZ(); }).observe(document.querySelector(".dock"));

  // drop files anywhere
  let depth = 0;
  window.addEventListener("dragenter", (e) => { if (e.dataTransfer?.types.includes("Files")) { depth++; $("dropveil").hidden = false; e.preventDefault(); } });
  window.addEventListener("dragleave", () => { depth = Math.max(0, depth - 1); if (!depth) $("dropveil").hidden = true; });
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => { e.preventDefault(); depth = 0; $("dropveil").hidden = true; if (e.dataTransfer?.files.length) takeFiles([...e.dataTransfer.files]); });
}

main();
