// The zebsdsolve app: a thin page over the engine (zebsdsolve.wasm through zebsdsolve.js). It
// passes files and settings in and draws what comes out; every number it shows is the engine's.
import { Zebsdsolve } from "../zebsdsolve/zebsdsolve.js";
import { GEOMETRY, buildFields, groupOf } from "./fields.js";
import { drawGray, drawSigned, drawHeat, fit, svg, download, quantiles } from "./draw.js";
import { createVotes } from "./votes.js";
import { createPlaneView } from "./planeview.js";
import { MasterView, PROJECTIONS } from "./masterview.js";
import { CellView } from "./cellview.js";
import { drawEnergy } from "./energy.js";

const $ = (id) => document.getElementById(id);
const app = $("app");
const PHASE_COLORS = ["#3b82f6", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#14b8a6", "#ec4899", "#84cc16"];
const RESULT_KINDS = ["ipf_z", "ipf_x", "ipf_y", "ncc", "nmi", "z", "phase", "sweep"];
const TABS = ["simulate", "data", "patterns", "geometry", "index"];
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
  resultsGeo: "",      // the geometry they were indexed under
  fit: null,           // fit patterns { cells, phases, quats, ncc }
  votes: null,         // every grid of the last pattern-centre search
  cmp: { mode: "diff", at: 0.5, exp: null, sim: null, w: 0, h: 0 },  // the third image: difference, or split by a bar
  pcStart: null,       // where that search started, and the pattern centre the file gave
  pcFile: null,
  geoOpened: null,     // the geometry as the scan was opened
  geoBefore: null,     // pattern centre and rotation before the last fit (or after it, once put back)
  geoIsBefore: false,
  phase: 0,            // selected phase (in the list under Data)
  phaseFiles: [],      // the master pattern files, as added ({ file, on, info }): the engine holds those that are on, and is given them again when one is removed or switched
  masterSrc: 0,        // the master the dock shows: a phase, or −1 the last simulation
  masterStage: "stored", // a phase's master as it was loaded, or filtered as patterns are matched with it
  simBin: null,        // the simulation's master shown of one energy bin alone, and of one site alone
  simSite: null,
  crystal: null,       // the crystal to simulate (the engine's account of it), its file's name,
  crystalName: "",
  cellInfo: null,      // its cell as drawn
  half: null,          // the hemispheres of a master shown (null: by its symmetry)
  sim: null,           // and the last simulation: its summary, its crystal's name, the settings it ran under
  roiTool: false,      // the next drag on the map draws the region of interest
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
  el.title = text;
  el.classList.toggle("err", err);
}

function log(text, err = false) {
  const el = $("log");
  const t = new Date().toLocaleTimeString([], { hour12: false });
  const line = document.createElement("div");
  if (err) line.className = "err";
  line.innerHTML = `<span class="t">${t} </span>`;
  const msg = document.createElement("span");
  msg.append(text);
  line.append(msg);
  el.append(line);
  el.scrollTop = el.scrollHeight;
  // an error while the log is folded: counted on the banner until it is opened
  if (err && $("console").dataset.open !== "1") {
    st.unseen = (st.unseen || 0) + 1;
    $("log-badge").textContent = String(st.unseen);
    $("log-badge").hidden = false;
    $("console").dataset.err = "1";
  }
}

/** Open or fold the log above the status banner. */
function openLog(open = $("console").dataset.open !== "1") {
  $("console").dataset.open = open ? "1" : "0";
  $("console-bar").setAttribute("aria-expanded", String(open));
  $("console-bar").title = open ? "Fold the log away" : "The log: click to open it";
  if (open) {
    st.unseen = 0;
    $("log-badge").hidden = true;
    $("console").dataset.err = "0";
    requestAnimationFrame(() => { $("log").scrollTop = $("log").scrollHeight; });
  }
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
  // the Geometry step is a console for the fits: the dock takes most of the stage there, and
  // shows a fit (the pattern-centre search, unless it is already on the plane)
  app.dataset.tab = name;
  if (name === "geometry" && votes && !["votes", "plane"].includes($("dock-kind").querySelector('[aria-pressed="true"]')?.dataset.dock)) selectDock(st.plane && !st.votes ? "plane" : "votes");
  // the Simulate step: the crystal beside the master, the simulation's when there is one
  if (name === "simulate" && votes) {
    if (st.sim) st.masterSrc = -1;
    selectDock("master");
    showEnergy();
  }
}

const done = (name, on = true) => { $(`tab-${name}`).dataset.done = on ? "1" : "0"; };

// ── settings and geometry ──

let fields = null;
let votes = null;
let planeView = null;
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
  if (["nlpar", "bgstatic", "bgdynamic", "clahe", "bandpass", "mask", "master", "resolution"].some((g) => groups.has(g)) || keys.some((k) => LIVE_KEYS.has(k))) {
    st.live.clear();
    showAngle();
    showCell();
    showMaster();
  }
  if (groups.has("roi") || keys.includes("fit_grid")) { st.fit = null; drawMapOverlay(); showRoi(); }
  if (keys.some((k) => ["sim_voltage_kv", "sim_tilt_deg", "sim_bin_kev"].includes(k))) showEnergy();
  if (keys.includes("accept_z") || keys.includes("accept_nmi")) drawZ();
  if (keys.some((k) => k.startsWith("ipf_"))) { st.keys.clear(); drawMap(); }
  drawLegend();
}

function buildGeometry() {
  const box = $("geometry-fields");
  for (const [key, label, hint, step, per, width] of GEOMETRY) {
    const triple = key === "pc" || key === "rot";
    const el = document.createElement("label");
    el.className = "field" + (triple ? " wide" : "") + (width ? ` ${width}` : "");
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
        // (a pixel size cleared, or 0, goes back to the scan's own: the engine puts it there)
        const v = +input.value / per;
        if (!Number.isFinite(v) || (key === "pxUm" && v < 0)) return;
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
  // (with a plane, it says where the beam stood for each map cell: the steps and the pixel size do not)
  $("geo-readout").textContent = st.scan ? (g.planeKind ? `beam positions: the fitted ${g.planeKind === 2 ? "projective" : "affine"} plane\n(the pattern centre is the plane's at the map's middle;\n the steps and the pixel size are not used)` : `pixel ${pixelText()}`) : "";
  $("btn-geo-plane").disabled = !g.planeKind;
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
  st.geoBefore = { pc: g.pc, rot: g.rot, pxUm: g.pxUm, plane: g.plane, planeKind: g.planeKind };
  st.geoIsBefore = false;
}

/** The detector's pixel size and where it is from: typed in, the scan's file, or the detector's
 *  width over the pattern's pixels across (an estimate). */
function pixelText() {
  const px = st.ze.geometry.pxUm;
  if (!(px > 0)) return "not known";
  const own = st.geoOpened ? st.geoOpened.pxUm : px;
  if (Math.abs(px - own) > 1e-9 * own) return `${px.toFixed(1)} µm, typed in`;
  if (st.scan && !st.scan.hasPx) return `${px.toFixed(1)} µm: an estimate, ${st.settings.detector_width_mm} mm across ${st.scan.patW} px`;
  return `${px.toFixed(1)} µm, the file's`;
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

/** The stages a pattern can be shown at: the averaged one only while patterns are averaged, the
 *  background only while one is subtracted. */
function showStage() {
  const s = st.settings;
  const sel = $("pat-stage");
  const bg = sel.querySelector('[value="background"]'), avg = sel.querySelector('[value="averaged"]');
  bg.disabled = s.static_background === "none";
  bg.textContent = s.static_background === "none" ? "background (none)" : "background";
  avg.disabled = !s.nlpar;
  avg.textContent = s.nlpar ? "averaged" : "averaged (off)";
  if ((st.stage === "averaged" && !s.nlpar) || (st.stage === "background" && s.static_background === "none")) st.stage = "filtered";
  sel.value = st.stage;
  if (!s.nlpar) $("nlpar-readout").textContent = "";
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
  st.plane = null;
  if (planeView) planeView.show(null, null, null);
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
  $("scan-hint").hidden = true;
  done("geometry", false);
  done("index", false);
  const gb = s.bytes / 2 ** 30;
  $("scan-meta").textContent = `${st.scanName} · ${s.format} · ${s.rows} × ${s.cols} patterns of ${s.patH} × ${s.patW} ${s.dtype}`;
  $("scan-meta").title = $("scan-meta").textContent;
  const facts = [
    ["format", s.format], ["map", `${s.rows} × ${s.cols} (${(s.rows * s.cols).toLocaleString()} patterns)`],
    ["pattern", `${s.patH} × ${s.patW} ${s.dtype}`], ["size", gb >= 1 ? `${gb.toFixed(2)} GiB` : `${(s.bytes / 2 ** 20).toFixed(1)} MiB`],
    ["static background", s.hasBackground ? "one in the file" : "none in the file"], ["pattern centre", s.hasPc ? "from the file" : "not in the file"],
    ["detector pixel", s.hasPx ? `${st.ze.geometry.pxUm.toFixed(1)} µm, from the file` : "not in the file"],
  ];
  if (s.voltageKv) facts.push(["voltage", `${s.voltageKv} kV`]);
  $("scan-facts").replaceChildren(...factRows(facts));
  st.settings = st.ze.settings;
  fields.show(st.settings);
  showResolution();
  showStage();
  showGeometry();
  setMapKind("overview");
  showRoi();
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
    for (const f of files) { await st.ze.addPhase(f); st.phaseFiles.push({ file: f, on: true, info: null }); }
    // the one just added is looked at (with no scan open, in the master view that takes the window)
    st.phase = st.masterSrc = st.ze.phases.length - 1;
    st.half = null;
    phasesChanged();
    if (!st.scan) selectTab("simulate");
    selectDock("master");
  });
}

/** Remove one phase: the engine drops them all and takes the others again. */
async function removePhase(i) {
  await busy("removing the phase…", async () => {
    st.phaseFiles = st.phaseFiles.filter((_, k) => k !== i);
    await rebuildPhases();
    phasesChanged();
  });
}

/** The engine's phases are the master patterns that are on, in order: give it them again. */
async function rebuildPhases() {
  st.ze.clearPhases();
  for (const p of st.phaseFiles) if (p.on) await st.ze.addPhase(p.file);
}

/** Switch a master pattern on or off for the fits and the indexing, without removing it. */
async function togglePhase(i) {
  await busy("switching the phase…", async () => {
    st.phaseFiles[i].on = !st.phaseFiles[i].on;
    await rebuildPhases();
    st.masterSrc = Math.max(0, Math.min(st.masterSrc, st.ze.phases.length - 1));
    st.phase = Math.min(st.phase, Math.max(0, st.ze.phases.length - 1));
    phasesChanged();
    log(`phase ${st.phaseFiles[i].info?.name ?? ""}: ${st.phaseFiles[i].on ? "on" : "off"}`);
  });
}

function showPhases() {
  const list = st.ze.phases;
  if (st.phase >= list.length) st.phase = 0;
  // what the engine says of each master that is on (a master that is off keeps what it said)
  let e = 0;
  for (const p of st.phaseFiles) if (p.on && list[e]) p.info = list[e++];
  done("data", !!st.scan && list.length > 0);
  $("btn-toy").disabled = list.length === 0;
  $("btn-toy").title = list.length ? "Make the scan from the master patterns above, in place of any scan open" : "Needs at least one master pattern";
  $("phases-hint").hidden = st.phaseFiles.length > 0;
  e = 0;
  $("phases").replaceChildren(...st.phaseFiles.map((entry, k) => {
    const idx = entry.on ? e++ : -1;
    const p = entry.info || { name: entry.file.name.replace(/\.[^.]+$/, ""), pointGroupName: "" };
    const li = document.createElement("li");
    li.setAttribute("aria-selected", String(idx >= 0 && idx === st.phase));
    li.dataset.on = entry.on ? "1" : "0";
    li.title = entry.info ? `${p.name}: point group ${p.pointGroupName}, Laue class ${p.laue}, ${p.side} × ${p.side} per hemisphere${p.voltageKv ? `, ${p.voltageKv} kV` : ""}${entry.on ? "" : ". Off: not fitted or indexed with, nor shown; switch it on with the box"}` : p.name;
    li.innerHTML = `<input type="checkbox" class="on" title="On: a candidate phase for the fits and the indexing. Off: kept here, set aside"><span class="dot"></span><span class="nm"></span><span class="pg"></span><button type="button" class="ghost" title="Remove this phase"><svg><use href="#i-x"/></svg></button>`;
    li.querySelector(".on").checked = entry.on;
    li.querySelector(".dot").style.background = entry.on ? PHASE_COLORS[idx % PHASE_COLORS.length] : "#3a3a41";
    li.querySelector(".nm").textContent = p.name;
    li.querySelector(".pg").textContent = p.pointGroupName;
    li.querySelector(".on").addEventListener("click", (ev) => { ev.stopPropagation(); });
    li.querySelector(".on").addEventListener("change", () => togglePhase(k));
    li.addEventListener("click", () => { if (idx < 0) return; st.phase = idx; st.masterSrc = idx; showPhases(); if ($("dock-master").hidden) selectDock("master"); else showMaster(); });
    li.querySelector("button").addEventListener("click", (ev) => { ev.stopPropagation(); removePhase(k); });
    return li;
  }));
  updateEnabled();
}

/** The master the dock shows: a phase (its index), −1 the last simulation, or null for none.
 *  The one asked for, else the simulation, else the first phase. */
function masterSource() {
  const n = st.ze ? st.ze.phases.length : 0;
  if (st.masterSrc === -1 && st.sim) return -1;
  if (st.masterSrc >= 0 && st.masterSrc < n) return st.masterSrc;
  return st.sim ? -1 : n ? 0 : null;
}

/** The master view: which masters there are to choose from, and the chosen one drawn. A
 *  phase's as stored or filtered; the simulation's every energy bin and site together, or one
 *  bin's and one site's alone. */
function showMaster() {
  if (!st.ze) return;
  const phases = st.ze.phases;
  const src = masterSource();
  const sel = $("master-src");
  const opts = phases.map((p, i) => new Option(`${p.name}${phases.length > 1 ? `  (phase ${i + 1})` : ""}`, String(i)));
  if (st.sim) opts.push(new Option(`${st.sim.name}, simulated${st.sim.added ? " (added)" : ""}`, "-1"));
  if (!opts.length) opts.push(new Option("no master pattern yet", ""));
  // (and a way to open another: a master to look at needs no scan, nor a simulation)
  opts.push(new Option("Open a master pattern…", "open"));
  sel.replaceChildren(...opts);
  sel.value = src === null ? "" : String(src);
  $("master-stage").disabled = src === null || src < 0;
  $("master-stage").value = st.masterStage;
  // the simulation's parts: its energy bins (the chart's bars; the arrow keys) and its sites
  const sim = st.sim;
  const bins = sim ? sim.binKv : [];
  const sites = sim ? sim.sites : [];
  const siteSeg = $("master-site");
  siteSeg.hidden = src !== -1 || sites.length < 2;
  if (!siteSeg.hidden) {
    siteSeg.replaceChildren(...[["all sites", null], ...sites.map((q, i) => [`${q.symbol}${sites.filter((s) => s.symbol === q.symbol).length > 1 ? ` ${i + 1}` : ""}`, i])].map(([label, i]) => {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = label;
      b.title = i === null ? "The master of every site together, weighted by their shares" : `The master of site ${i + 1} (${sites[i].symbol}) alone`;
      b.setAttribute("aria-pressed", String(st.simSite === i));
      b.addEventListener("click", () => { st.simSite = i; showMaster(); showSiteMarks(); });
      return b;
    }));
  }
  showHalf(src);
  $("dock-master").dataset.empty = src === null ? "1" : "0";
  if (!$("dock-master").hidden) {
    const meta = src === null ? "" : src >= 0
      ? `${phases[src].side} × ${phases[src].side} a hemisphere${phases[src].voltageKv ? ` · ${phases[src].voltageKv} kV` : ""}`
      : `${2 * sim.halfw + 1} × ${2 * sim.halfw + 1} a hemisphere · ${st.simBin === null ? `${sim.voltageKv} kV beam` : `${bins[st.simBin]} kV bin`}${st.simSite === null ? "" : ` · site ${st.simSite + 1} alone`}`;
    $("dock-meta").textContent = meta;
    sel.title = meta ? `${meta}. Which master pattern: a candidate phase's, or the last simulated one` : "Which master pattern: a candidate phase's, or the last simulated one";
    st.masterView.refresh();
  }
}

/** Whether the master shown has a centre of symmetry (its lower hemisphere is the upper one
 *  inverted), when that is known. */
function masterCentro(src) {
  if (src === null) return false;
  if (src === -1) return !!st.sim?.crystal?.centrosymmetric;
  return CENTRO_PG.has(st.ze.phases[src].pointGroup);
}

/** The hemispheres shown: what was chosen, else (for a crystal with a centre of symmetry, whose
 *  two hemispheres are one) the upper alone, else both. */
function showHalf(src = masterSource()) {
  const centro = masterCentro(src);
  const half = st.half ?? (centro ? "north" : "both");
  for (const b of $("view-half").children) {
    b.setAttribute("aria-pressed", String(b.dataset.half === half));
    // (two hemispheres that are one have no difference; the globe shows all of the sphere)
    b.disabled = (b.dataset.half === "diff" && centro) || (st.projection === "globe" && b.dataset.half !== "diff");
  }
  $("view-half").dataset.off = st.projection === "globe" && half !== "diff" ? "1" : "0";
  const diff = half === "diff" && !centro;
  st.masterView.setShown(half === "south" ? 2 : half === "both" ? 0 : 1, diff);
}

/** Show the simulation's master of one energy bin (or site) alone, or all of them again. */
function pickSimPart(kind, i) {
  if (!st.sim) return;
  if (kind === "bin") st.simBin = st.simBin === i ? null : i;
  else st.simSite = st.simSite === i ? null : i;
  st.masterSrc = -1;
  if ($("dock-master").hidden) selectDock("master"); else showMaster();
  showEnergy();
  showSiteMarks();
}

// ── the Simulate step: a master pattern from a crystal ──

const fixed = (v, d) => (Math.abs(v - Math.round(v)) < 1e-9 ? String(Math.round(v)) : v.toFixed(d));

/** Change the crystal (its free lattice parameters, a site's occupancy or B): the engine ties
 *  what the crystal's family ties, and keeps how it was read. */
function editCrystal(edit, said) {
  try {
    st.crystal = st.ze.editCrystal(edit);
    log(`crystal ${st.crystalName}: ${said}`);
    status(`${st.crystalName}: ${said}`);
  } catch (e) {
    const msg = String(e.message || e).replace(/^sim_edit_crystal: (sim_edit_crystal: )?/, "");
    status(msg, true);
    log(msg, true);
  }
  showCrystal();
}

/** The crystal the engine has: its name and group, its cell (the free parameters editable) and
 *  sites (occupancy and B editable; each pointable, the cell view picking it out, and each a
 *  click from its own master), its cell drawn, and how the electrons spread in it. */
function showCrystal() {
  const c = st.crystal;
  app.dataset.crystal = c ? "1" : "0";
  $("btn-simulate").disabled = !c;
  $("btn-simulate").title = c ? `Simulate ${st.crystalName}'s master pattern under the settings below` : "Needs a crystal: open a CIF or pick a preset";
  $("crystal-none").hidden = !!c;
  $("crystal").hidden = !c;
  st.cellInfo = null;
  if (c) {
    try { st.cellInfo = st.ze.cellInfo(st.cellView.cells); } catch (e) { log(String(e.message || e), true); }
  }
  st.cellView.show(st.cellInfo, c ? c.cell : null);
  $("xtal-meta").textContent = c ? `${st.crystalName} · ${c.name} (${c.space_group})` : "";
  $("xtal-meta").title = c && st.cellInfo ? `${st.cellInfo.atoms} atoms drawn: each site${st.cellInfo.cells > 1 ? " in every cell" : ""} with its images on the faces, edges and corners of the cell${st.cellInfo.cells > 1 ? "s" : ""}` : "";
  showEnergy();
  if (!c) return;
  $("crystal-name").textContent = st.crystalName;
  $("crystal-group").textContent = `${c.name}  (${c.space_group})`;
  const pick = (el, what) => {
    el.addEventListener("mouseenter", () => st.cellView.pick(what));
    el.addEventListener("mouseleave", () => st.cellView.pick(null));
  };
  const number = (value, step, min, title, apply) => {
    const input = document.createElement("input");
    input.type = "number";
    input.step = String(step);
    input.min = String(min);
    input.value = String(value);
    input.title = title;
    input.addEventListener("change", () => { const v = +input.value; if (Number.isFinite(v)) apply(v); else input.value = String(value); });
    return input;
  };
  // the cell: the free parameters to type in, the tied ones following them
  const names = [["a", "a", "Å", 4], ["b", "b", "Å", 4], ["c", "c", "Å", 4], ["α", "alpha", "°", 3], ["β", "beta", "°", 3], ["γ", "gamma", "°", 3]];
  const tie = ["", "= a", "= a", "", "", ""];
  $("cellpars").replaceChildren(...names.map(([sym, key, unit, d], i) => {
    const el = document.createElement("label");
    el.className = "par";
    const name = document.createElement("i");
    name.textContent = sym;
    el.append(name);
    if (c.free[i]) {
      el.append(number(+c.cell[i].toFixed(d + 2), unit === "Å" ? 0.001 : 0.01, 0, `${sym} (${unit}): type a value; what the crystal's family ties follows`, (v) => {
        const cell = c.cell.slice();
        cell[i] = v;
        editCrystal({ cell }, `${sym} = ${v} ${unit}`);
      }));
    } else {
      const v = document.createElement("span");
      v.textContent = fixed(c.cell[i], d);
      v.title = `${sym} = ${fixed(c.cell[i], d)} ${unit}: tied by the crystal's family${i < 3 && tie[i] && Math.abs(c.cell[i] - c.cell[0]) < 1e-9 ? ` (${sym} ${tie[i]})` : ""}`;
      el.append(v);
    }
    pick(el, key);
    return el;
  }));
  // the sites: occupancy and the Debye–Waller B to type in
  const head = document.createElement("div");
  head.className = "site head";
  head.innerHTML = "<span></span><span>site</span><span>x y z</span><span>occ.</span><span>B (Å²)</span>";
  const edits = () => c.sites.map((q) => ({ occupancy: q.occupancy, b_iso: q.b_iso }));
  $("crystal-sites").replaceChildren(head, ...c.sites.map((q, i) => {
    const row = document.createElement("div");
    row.className = "site";
    row.dataset.site = String(i);
    const dot = document.createElement("span");
    dot.className = "dot";
    dot.style.background = st.cellInfo?.sites[i]?.colour || "#888";
    const sym = document.createElement("span");
    sym.className = "sym";
    sym.textContent = `${q.symbol} ×${q.multiplicity}`;
    const at = document.createElement("span");
    at.className = "at";
    at.textContent = q.xyz.map((v) => fixed(v, 4)).join(" ");
    at.title = `${q.symbol} at ${q.xyz.map((v) => fixed(v, 4)).join(", ")}: ${q.multiplicity} in the cell`;
    const occ = number(q.occupancy, 0.01, 0, "The site's occupancy: the share of its places the element holds", (v) => { const s = edits(); s[i].occupancy = v; editCrystal({ sites: s }, `site ${i + 1} (${q.symbol}) occupancy ${v}`); });
    const b = number(+q.b_iso.toFixed(4), 0.05, 0, "The site's isotropic Debye–Waller factor B, Å² (B = 8π²⟨u²⟩): how far its atoms' thermal motion damps the reflections. ebsdsim writes it in nm² (Å² / 100)", (v) => { const s = edits(); s[i].b_iso = v; editCrystal({ sites: s }, `site ${i + 1} (${q.symbol}) B ${v} Å²`); });
    row.append(dot, sym, at, occ, b);
    row.title = "Point to see its atoms on the cell; click its name to see its master alone (after a simulation)";
    sym.addEventListener("click", () => { if (st.sim && st.sim.sites.length === c.sites.length && c.sites.length > 1) pickSimPart("site", i); });
    pick(row, i);
    return row;
  }));
  showSiteMarks();
  $("crystal-facts").textContent = `${c.point_group}${c.centrosymmetric ? " (centrosymmetric)" : ""} · ${c.density.toFixed(3)} g/cm³ · ${(c.volume_nm3 * 1000).toFixed(2)} Å³`;
}

/** The site whose master is shown alone, marked in the list. */
function showSiteMarks() {
  for (const row of $("crystal-sites").querySelectorAll(".site[data-site]")) row.dataset.picked = st.simSite !== null && +row.dataset.site === st.simSite ? "1" : "0";
}

/** How the energy model spreads the electrons for the crystal under the settings, drawn under
 *  its cell; the bins the last simulation solved picked out, when it ran under the same, each
 *  a click from its own master. */
function showEnergy() {
  if (!st.ze || $("xtal-pane").offsetParent === null) return;
  let m = null;
  if (st.crystal) {
    try { m = st.ze.energyModel(); } catch (e) { log(String(e.message || e), true); }
  }
  const s = st.settings;
  const same = st.sim && st.sim.crystal === st.crystal && st.sim.energyKey === `${s.sim_voltage_kv}/${s.sim_tilt_deg}/${s.sim_bin_kev}`;
  drawEnergy($("energy-e"), $("energy-z"), m, { solved: same ? st.sim.nBinsRun : null, picked: same ? st.simBin : null, pick: same ? (i) => pickSimPart("bin", i) : null });
  st.energySolved = same ? st.sim.nBinsRun : 0;
  $("energy-all").disabled = !same;
  $("energy-all").setAttribute("aria-pressed", String(!same || st.simBin === null));
  $("energy-readout").textContent = m ? `${m.bins.length} bins of ${m.bin_kev} keV${same ? `, ${st.sim.nBinsRun} solved` : ""} · mean loss ${m.mean_loss_kev.toFixed(2)} keV · mean depth ${m.mean_depth_nm.toFixed(1)} nm, 90 % within ${m.depth90_nm.toFixed(1)} nm` : "";
}

/** A CIF's text to the engine, which reads it (or says why not). */
function loadCif(text, name) {
  try {
    st.crystal = st.ze.readCif(text);
    st.crystalName = name.replace(/\.cif$/i, "");
    $("sim-name").value = st.crystalName;
    const c = st.crystal, r = c.reading;
    log(`crystal ${name}: space group ${c.space_group} (${c.name}), ${c.sites.length} site(s)`);
    // how the file was read
    if (r) {
      log(`  symmetry: ${r.symmetry_provenance}`);
      log(`  setting: ${r.setting}${r.setting_note ? ` (${r.setting_note})` : ""}`);
      if (r.cell_as_read.some((v, i) => Math.abs(v - c.cell[i]) > 1e-6)) log(`  cell as read: ${r.cell_as_read.slice(0, 3).map((v) => fixed(v, 4)).join(" ")}  ${r.cell_as_read.slice(3).map((v) => fixed(v, 2)).join(" ")}`);
      if (r.sites_as_read !== c.sites.length) log(`  ${r.sites_as_read} sites as read, ${c.sites.length} in the asymmetric unit`);
      if (r.sites_given_default_b) log(`  B = 0.5 Å² given to ${r.sites_given_default_b} of ${r.sites_as_read} sites (${r.thermal_parameters_in_file ? "zero in the file" : "none in the file"})`);
    }
    status(`${name}: space group ${c.space_group} (${c.name})`);
  } catch (e) {
    const msg = String(e.message || e).replace(/^sim_read_cif: (sim_read_cif: )?/, "");
    status(`${name}: ${msg}`, true);
    log(`${name}: ${msg}`, true);
  }
  showCrystal();
}

async function loadPreset(key) {
  if (!key) return;
  try {
    const r = await fetch(`./cif/${key}.cif`);
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    loadCif(await r.text(), `${key}.cif`);
  } catch (e) { status(`preset ${key}: ${e.message || e}`, true); }
  $("cif-preset").value = "";
}

/** What a simulation did, for the log. */
function simReadout(r) {
  const side = 2 * r.halfw + 1;
  return [
    `${st.sim.name}: ${r.nK.toLocaleString()} directions (the fundamental sector of ${side} × ${side} a hemisphere), ${r.nG.toLocaleString()} reflections, at most ${r.maxBeams} beams${r.maxWeak ? ` and ${r.maxWeak} weak` : ""} a direction`,
    `energy bins: ${r.nBinsRun} of ${r.nBins}${r.stopped ? `, stopped when one changed the pattern by ${(100 * r.lastChange).toFixed(2)} %` : ""}`,
    r.failed ? `${r.failed.toLocaleString()} directions failed to solve (left dark)` : "",
  ].filter(Boolean);
}

async function runSimulate() {
  if (!st.crystal) return;
  const name = st.crystalName;
  const s = st.settings;
  await busy(`simulating ${name}'s master pattern…`, async () => {
    const r = await st.ze.simulateMaster();
    // (the bins it solved are the energy model's first, from the least loss)
    const model = st.ze.energyModel();
    st.sim = { ...r, name, voltageKv: s.sim_voltage_kv, crystal: st.crystal, energyKey: `${s.sim_voltage_kv}/${s.sim_tilt_deg}/${s.sim_bin_kev}`, added: false,
      binKv: model.bins.slice(0, r.nBinsRun).map((b) => b.kv), sites: st.crystal.sites.map((q) => ({ symbol: q.symbol })) };
    st.simBin = null;
    st.simSite = null;
    showSiteMarks();
    for (const line of simReadout(r)) log(line);
    $("sim-result").hidden = false;
    if (!$("sim-name").value.trim()) $("sim-name").value = name;
    st.masterSrc = -1;
    st.half = null;
    st.masterView.resetLayout();
    selectDock("master");
    showEnergy();
  });
  $("btn-sim-phase").disabled = $("btn-sim-save").disabled = !st.sim;
}

const simName = () => ($("sim-name").value.trim() || st.sim?.name || "simulated").replace(/[\\/:*?"<>|]/g, "_");

/** The last simulation as a candidate phase; its .npz is kept, as an added file's would be, so
 *  that removing another phase can add it again. */
async function addSimPhase() {
  if (!st.sim) return;
  const name = simName();
  await busy("adding the simulated phase…", async () => {
    st.ze.addSimulatedPhase(name);
    st.phaseFiles.push({ file: new File([st.ze.simulationNpz()], `${name}.npz`, { type: "application/octet-stream" }), on: true, info: null });
    st.phase = st.ze.phases.length - 1;
    st.sim.added = true;
    phasesChanged();
    log(`phase ${name} added from the simulation`);
  });
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

/** The orientation the engine has for a cell: from indexing (while the geometry is the one the
 *  map was indexed under: under another, that orientation is not this geometry's match), else
 *  from the last fit. */
function orientationOf(cell) {
  const R = st.results;
  if (R && R.phase[cell] >= 0 && st.resultsGeo === JSON.stringify(st.ze.geometry)) {
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
  const moved = st.results && st.results.phase[cell] >= 0 ? "; the map was indexed under another geometry" : "";
  const o = { phase: p, q: Array.from(r.subarray(0, 4)), ncc: r[4], nmi: r[5], z: r[7], sweep: null, euler: null, from: `found now, L ${st.settings.fit_bandwidth}${moved}` };
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
    // (what the stage's name does not say: the averaging's window)
    $("pat-exp-cap").textContent = stage === "averaged" ? win : stage === "filtered" && s.nlpar ? `averaged ${win}` : stage === "background" ? (s.static_background === "file" ? "the file's" : `the average of ${s.static_patterns}`) : "";
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
      st.cmp.diff = a.map((v, i) => v - b[i]);
      st.cmp.exp = proc; st.cmp.sim = sim; st.cmp.w = w; st.cmp.h = h; st.cmp.mask = mask;
      drawCompare();
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
      st.cmp.exp = null;
      $("pat-wipe").hidden = true;
      $("pat-sim-cap").textContent = !nPh ? "simulated · add a phase" : st.busy ? "simulated · when the job is done" : "simulated · fit or index to see it";
    }
    if (exp.nlpar) facts.push(["averaging", `${(100 * exp.nlpar.left).toFixed(0)} % of the noise left`]);
    $("pat-facts").replaceChildren(...factRows(facts));
  } catch (e) {
    if (run === st.showRun) { status(String(e.message || e), true); log(String(e.message || e), true); }
  }
}

/** The third image: the difference of the pattern and its simulation, or the two in one frame,
 *  the experimental left of a bar and the simulation right of it. */
function drawCompare() {
  const c = st.cmp;
  const canvas = $("pat-diff"), bar = $("pat-wipe"), well = $("pat-cmp-well");
  const split = c.mode === "split";
  $("pat-cmp").value = c.mode;
  well.dataset.split = split && c.exp ? "1" : "0";
  $("pat-cmp").title = split ? "The experimental pattern left of the bar, its simulation right of it. Drag the bar across." : "Experimental minus simulated, each zero-mean and unit-norm under the mask: what the NCC compares";
  bar.hidden = !(split && c.exp);
  if (!c.exp) return;
  if (!split) { drawSigned(canvas, c.diff, c.w, c.h); return; }
  const { w, h } = c;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(w, h);
  const cut = Math.round(c.at * w);
  const scale = (a) => { const [lo, hi] = quantiles(a, 0.005, 0.995); return [lo, 255 / Math.max(hi - lo, 1e-20)]; };
  const [lo0, k0] = scale(c.exp), [lo1, k1] = scale(c.sim);
  for (let i = 0; i < w * h; i++) {
    const left = i % w < cut;
    const v = Math.max(0, Math.min(255, left ? (c.exp[i] - lo0) * k0 : (c.sim[i] - lo1) * k1)) * (c.mask && !c.mask[i] ? 0.25 : 1);
    img.data[4 * i] = img.data[4 * i + 1] = img.data[4 * i + 2] = v;
    img.data[4 * i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  // the bar, where the cut falls in the image as it is shown (the canvas is fitted into its well)
  const r = imageBox(canvas);
  bar.style.left = `${r.left + (cut / w) * r.width - 1}px`;
  bar.style.top = `${r.top}px`;
  bar.style.bottom = "auto";
  bar.style.height = `${r.height}px`;
}

/** Where a canvas's image lies in its well (object-fit: contain), in the well's pixels. */
function imageBox(canvas) {
  const W = canvas.clientWidth, H = canvas.clientHeight;
  const s = Math.min(W / canvas.width, H / canvas.height);
  const width = canvas.width * s, height = canvas.height * s;
  return { left: canvas.offsetLeft + (W - width) / 2, top: canvas.offsetTop + (H - height) / 2, width, height };
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
    $("fit-readout").textContent = `centre  ${r.pc.map((v) => v.toFixed(5)).join("  ")}\nNCC     ${r.score.toFixed(4)}  (start ${r.startScore.toFixed(4)})\n${r.nCandidates.toLocaleString()} candidates × ${r.nPatterns} patterns, L ${st.settings.fit_bandwidth}${r.atBorder ? "\nthe best candidate is at the search's border: widen it, or fit again from here" : ""}`;
    $("fit-readout").textContent += againstTrueGeometry();
    done("geometry");
    selectDock("votes");
    drawVotes();
    geometryChanged(true);
    drawMapOverlay();
  });
}

// What a plane fit found: the plane's source at the map's middle, how its points lie about it, and
// what it says of the tilt, the scan and the steps, each with its standard error.
function planeReadout(r) {
  const deg = (v) => (v * 180) / Math.PI;
  const signed = (v, d) => `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(d)}`;
  const g = st.ze.geometry;
  const lines = [
    `plane   ${r.kind === 2 ? "projective" : "affine"}, ${r.nUsed.toLocaleString()} of ${r.nPatterns.toLocaleString()} patterns`,
    `centre  ${r.pc.map((v) => v.toFixed(5)).join("  ")}`,
    `scatter ${r.rmsd.map((v) => v.toFixed(5)).join("  ")}`,
    `tilt    ${signed(deg(r.saidRot[1]), 2)}° ± ${deg(r.saidRotSe[1]).toFixed(2)}°  (sample ${(g.sampleTiltDeg - deg(r.saidRot[1])).toFixed(2)}°)`,
    `swing   ${signed(deg(r.saidRot[0]), 2)}° ± ${deg(r.saidRotSe[0]).toFixed(2)}°`,
    `scan    turned ${signed(deg(r.saidRot[2]), 2)}° ± ${deg(r.saidRotSe[2]).toFixed(2)}°`,
    `        ${signed(deg(r.skew), 2)}° ± ${deg(r.skewSe).toFixed(2)}° off square`,
  ];
  if (r.step[0] > 0) lines.push(`steps   row ${r.step[0].toFixed(3)} ± ${r.stepSe[0].toFixed(3)}`, `        column ${r.step[1].toFixed(3)} ± ${r.stepSe[1].toFixed(3)}`);
  lines.push(`frame   ${r.frame ? "follows the plane" : "as the tilts say"}`, `NCC     ${r.ncc.toFixed(4)}  (start ${r.startNcc.toFixed(4)})`);
  return lines.join("\n");
}

/** A plane fit is done (by itself, or after the vote): its numbers, its points, and everything drawn from the
 *  geometry. `vote`: a line for the vote that came before it. */
function afterPlane(r, vote = "") {
  st.fit = st.ze.fitPatterns();
  st.plane = { r, pts: st.ze.planePoints() };
  $("fit-readout").textContent = vote + planeReadout(r) + againstTrueGeometry();
  done("geometry");
  selectDock("plane");
  planeView.show(r, st.plane.pts, st.scan);
  geometryChanged(true);
  drawMapOverlay();
}

async function fitPlane() {
  await busy("fitting the plane…", async () => {
    beforeFit();
    afterPlane(await st.ze.fitPlane());
  });
}

async function fitGeometry() {
  await busy("fitting the geometry…", async () => {
    beforeFit();
    const r = await st.ze.fitGeometry();
    st.votes = st.ze.pcLevels();
    // (the vote's first grid is centred on where it started)
    const mid = (a) => (a[0] + a[a.length - 1]) / 2;
    st.pcStart = st.votes.length ? [mid(st.votes[0].x), mid(st.votes[0].y), mid(st.votes[0].z)] : null;
    drawVotes();
    afterPlane(r, `vote    ${r.startPc.map((v) => v.toFixed(5)).join("  ")}\n`);
  });
}

// ── indexing ──

async function runIndex() {
  await busy("indexing…", async () => {
    await st.ze.indexBegin();
    st.resultsGeo = JSON.stringify(st.ze.geometry);
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
  for (const k of ["votes", "plane", "master", "z"]) $(`dock-${k}`).hidden = k !== kind;
  $("dock-meta").textContent = "";
  $("votes-meta").hidden = kind !== "votes";
  if (kind === "master") showMaster();
  if (kind === "votes") votes.draw();
  if (kind === "plane") planeView.draw();
  if (kind === "z") drawZ();
}

/** How the master view draws the sphere: a globe, the Lambert squares, the stereographic discs,
 *  or the joint Lambert square (kept for the next visit). */
const CENTRO_PG = new Set([2, 5, 8, 11, 15, 17, 20, 23, 27, 29, 32]);

function setProjection(p) {
  if (!PROJECTIONS.includes(p)) p = "globe";
  st.projection = p;
  try { localStorage.setItem("zebsdsolve.projection", p); } catch { /* private window */ }
  for (const b of $("view-mode").children) b.setAttribute("aria-pressed", String(b.dataset.view === p));
  st.masterView.setProjection(p);
  showViewLock();
  showHalf();
}

/** The lock applies to the flat projections only (dimmed on the globe); while it holds, the
 *  buttons that would turn the sphere are dimmed too (the first view is still there). */
function showViewLock() {
  const flat = st.projection !== "globe";
  $("view-lock").disabled = !flat;
  $("view-lock-box").dataset.off = flat ? "0" : "1";
  st.masterView.setLocked($("view-lock").checked);
  for (const b of $("view-look").children) b.disabled = !!b.dataset.look && st.masterView.isLocked();
  showStereo();
}

/** The globe as a stereogram: two views for the two eyes, the distance to the screen and the
 *  eyes' separation set at the foot of the picture. */
function showStereo() {
  const globe = st.projection === "globe";
  const on = globe && $("view-stereo").checked;
  $("view-stereo").disabled = !globe;
  $("view-stereo-box").dataset.off = globe ? "0" : "1";
  $("stereo-bar").hidden = !on;
  const dist = +$("stereo-dist").value, ipd = +$("stereo-ipd").value;
  $("stereo-dist-out").textContent = `${dist} mm`;
  $("stereo-ipd-out").textContent = `${ipd} mm`;
  st.masterView.setStereo(on, dist, ipd, $("stereo-swap").checked);
}

/** Step the master's energy bin with the arrow keys: right toward less energy, left toward
 *  more, past the first back to all of them. */
function stepBin(dir) {
  const solved = st.energySolved;
  if (!solved) return false;
  const at = st.simBin === null ? -1 : st.simBin;
  const next = Math.max(-1, Math.min(solved - 1, at + dir));
  if (next === at) return true;
  st.simBin = next < 0 ? null : next;
  st.masterSrc = -1;
  if ($("dock-master").hidden) selectDock("master"); else showMaster();
  showEnergy();
  return true;
}

/** The region of interest on the map's bar: its size, and the cross that clears it. */
function showRoi() {
  const s = st.settings;
  const on = s.roi_w > 0 && s.roi_h > 0;
  $("btn-roi").setAttribute("aria-pressed", String(st.roiTool));
  $("btn-roi").dataset.set = on ? "1" : "0";
  $("roi-text").textContent = on ? `region ${s.roi_w} × ${s.roi_h}` : "region";
  $("btn-roi").title = `${on ? `Region of interest: ${s.roi_w} × ${s.roi_h} cells from column ${s.roi_x}, row ${s.roi_y}; the fits and indexing keep to it. ` : "The region of interest, which the fits and indexing keep to (now the whole map). "}Press, then drag a box on the map (shift-drag does it any time).`;
  $("btn-roi-clear").hidden = !on;
  $("map-well").dataset.tool = st.roiTool ? "roi" : "";
}

function setRoi(set) {
  try { st.ze.configure(set); } catch (err) { status(String(err.message || err), true); }
  st.fit = null;
  afterSettings(Object.keys(set));
}

function updateEnabled() {
  const ready = !!st.scan && st.ze.phases.length > 0;
  for (const id of ["btn-fit-auto", "btn-fit-pc", "btn-fit-plane", "btn-index", "btn-tune"]) $(id).disabled = !ready;
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
  const cifs = files.filter((f) => /\.cif$/i.test(f.name));
  if (cifs.length) {
    loadCif(await cifs[0].text(), cifs[0].name);
    selectTab("simulate");
    files = files.filter((f) => !cifs.includes(f));
    if (!files.length) return;
  }
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
      // (a project saved by another version may hold settings this one does not have)
      const have = st.ze.settings;
      const gone = Object.keys(project.settings).filter((k) => !(k in have));
      const take = Object.fromEntries(Object.entries(project.settings).filter(([k]) => k in have));
      if (gone.length) log(`project ${projects[0].name}: settings this version does not have, left out: ${gone.join(", ")}`);
      try { st.ze.configure(take); } catch (e) { status(String(e.message || e), true); log(String(e.message || e), true); return; }
      afterSettings(Object.keys(take));
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
  return `\nagainst the toy scan's geometry: pattern centre within ${pc.toFixed(5)}, rotation within ${rot.toFixed(3)}°${Math.abs(g.pxUm - T.pxUm) > 1e-9 * T.pxUm ? `; the pixel size is ${g.pxUm.toFixed(1)} µm, the scan was made with ${T.pxUm.toFixed(1)}` : ""}`;
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
    st.phaseFiles = [{ file: master, on: true, info: null }];
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
  planeView = createPlaneView();
  // the master view (a phase's master or the last simulation's) and the crystal's cell, both
  // rendered by the engine (nothing until there is one to show)
  st.masterView = new MasterView($("master-well"), (o) => {
    const source = st.ze ? masterSource() : null;
    if (source === null) return null;
    return st.ze.sphereView({ ...o, source, stored: source >= 0 && st.masterStage === "stored", bin: source < 0 ? st.simBin : null, site: source < 0 ? st.simSite : null });
  }, (rot) => { if (st.link) st.cellView.setRotation(rot); });
  // (the cell and the master share the crystal's Cartesian frame: linked, one rotation turns both)
  st.cellView = new CellView($("cell-well"), (o) => (st.ze && st.crystal ? st.ze.cellView(o) : null), (rot) => { if (st.link) st.masterView.setRotation(rot); });
  let projection = "globe";
  try { projection = localStorage.getItem("zebsdsolve.projection") || "globe"; } catch { /* private window */ }
  setProjection(projection);
  app.dataset.crystal = "0";
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
  });
  updateEnabled();
  showMaster();
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
    for (const [prop, v] of [["--dock-h", saved.dock], ["--pat-w", saved.pat], ["--xtal-w", saved.xtal]]) {
      if (v > 0) stage.style.setProperty(prop, `${v}px`); else stage.style.removeProperty(prop);
    }
  };
  const keep = () => { try { localStorage.setItem("zebsdsolve.layout", JSON.stringify(saved)); } catch { /* not kept */ } };
  apply();
  const wire = (el, which, axis) => {
    // (the bar beside the pattern is the bar beside the crystal on the Simulate step)
    const keyOf = () => (which === "pat" && app.dataset.tab === "simulate" ? "xtal" : which);
    el.addEventListener("mousedown", (e) => {
      if (e.button !== 0) return;
      e.preventDefault();
      const key = keyOf();
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
    el.addEventListener("dblclick", () => { delete saved[keyOf()]; apply(); keep(); });
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
  $("btn-cif").addEventListener("click", () => $("file-cif").click());
  $("file-cif").addEventListener("change", async (e) => { const f = e.target.files[0]; e.target.value = ""; if (f) loadCif(await f.text(), f.name); });
  $("cif-preset").addEventListener("change", (e) => loadPreset(e.target.value));
  $("btn-simulate").addEventListener("click", runSimulate);
  $("btn-sim-phase").addEventListener("click", addSimPhase);
  $("btn-sim-save").addEventListener("click", () => { if (st.sim) download(`${simName()}.npz`, st.ze.simulationNpz(), "application/octet-stream"); });
  $("file-phase").addEventListener("change", (e) => { const f = [...e.target.files]; e.target.value = ""; addPhases(f); });
  $("btn-fit-auto").addEventListener("click", fitGeometry);
  $("btn-fit-pc").addEventListener("click", fitPc);
  $("btn-fit-plane").addEventListener("click", fitPlane);
  $("btn-geo-plane").addEventListener("click", () => {
    beforeFit();
    st.ze.setGeometry({ planeKind: 0 });
    geometryChanged();
  });
  $("btn-geo-undo").addEventListener("click", () => {
    if (!st.geoBefore) return;
    // back to before the fit, or forward to after it: the two change places
    const g = st.ze.geometry;
    const other = { pc: g.pc, rot: g.rot, pxUm: g.pxUm, plane: g.plane, planeKind: g.planeKind };
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
  $("view-mode").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) setProjection(b.dataset.view); });
  $("view-look").addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    if (b.dataset.look) st.masterView.lookAlong(b.dataset.look.split(",").map(Number)); else st.masterView.reset();
  });
  $("master-src").addEventListener("change", (e) => {
    if (e.target.value === "") return;
    if (e.target.value === "open") { showMaster(); $("file-phase").click(); return; }
    st.masterSrc = +e.target.value;
    st.half = null;
    st.masterView.resetLayout();
    if (st.masterSrc >= 0) { st.phase = st.masterSrc; showPhases(); }
    showMaster();
  });
  $("master-stage").addEventListener("change", (e) => { st.masterStage = e.target.value; showMaster(); });
  $("view-half").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b && !b.disabled) { st.half = b.dataset.half; showHalf(); } });
  for (const id of ["view-stereo", "stereo-dist", "stereo-ipd", "stereo-swap"]) $(id).addEventListener("input", showStereo);
  $("view-lock").addEventListener("change", showViewLock);
  $("cell-super").addEventListener("change", (e) => {
    st.cellView.cells = e.target.checked ? 3 : 1;
    if (st.crystal) showCrystal();
  });
  $("cell-link").addEventListener("change", (e) => {
    st.link = e.target.checked;
    // linked: the master turns to the cell as it is
    if (st.link && st.cellView.rot) st.masterView.setRotation(st.cellView.rot);
  });
  $("energy-all").addEventListener("click", () => { if (st.simBin !== null) pickSimPart("bin", st.simBin); });
  $("cell-look").addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    if (b.dataset.look) st.cellView.lookAlong(+b.dataset.look); else st.cellView.reset();
  });
  $("btn-roi").addEventListener("click", () => { st.roiTool = !st.roiTool; showRoi(); if (st.roiTool) status("drag a box on the map for the region of interest (Escape: never mind)"); });
  $("btn-roi-clear").addEventListener("click", () => setRoi({ roi_x: 0, roi_y: 0, roi_w: 0, roi_h: 0 }));
  // the log: the banner opens it (and folds it again); Escape folds it
  $("console-bar").addEventListener("click", (e) => { if (!e.target.closest("#btn-stop")) openLog(); });
  $("console-bar").addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openLog(); } });
  $("btn-log-close").addEventListener("click", () => openLog(false));
  $("btn-log-copy").addEventListener("click", () => navigator.clipboard?.writeText($("log").innerText).then(() => status("the log is on the clipboard")));
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    if ($("console").dataset.open === "1") openLog(false);
    if (st.roiTool || st.roiDrag) { st.roiTool = false; st.roiDrag = null; showRoi(); drawMapOverlay(); }
  });
  $("pat-stage").addEventListener("change", (e) => { st.stage = e.target.value; showStage(); showCell(); });
  $("show-fit").addEventListener("change", drawMapOverlay);
  $("map-legend").addEventListener("click", () => { st.legendFolded = !st.legendFolded; drawLegend(); });
  $("pat-live").addEventListener("change", showCell);
  $("pat-cmp").addEventListener("change", (e) => { st.cmp.mode = e.target.value; drawCompare(); });
  // the bar between the pattern and its simulation: press or drag anywhere on the image
  {
    const well = $("pat-cmp-well");
    const to = (e) => {
      if (st.cmp.mode !== "split" || !st.cmp.exp) return;
      const r = imageBox($("pat-diff"));
      const box = well.getBoundingClientRect();
      st.cmp.at = Math.max(0, Math.min(1, (e.clientX - box.left - r.left) / Math.max(r.width, 1)));
      drawCompare();
    };
    well.addEventListener("mousedown", (e) => {
      if (e.button !== 0 || st.cmp.mode !== "split") return;
      e.preventDefault();
      to(e);
      const up = () => { window.removeEventListener("mousemove", to); window.removeEventListener("mouseup", up); };
      window.addEventListener("mousemove", to);
      window.addEventListener("mouseup", up);
    });
  }
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
  // the arrow keys, wherever the focus is but in a field or a view with keys of its own: on the
  // Simulate step they step the master's energy bin, elsewhere the selected pattern on the map
  document.addEventListener("keydown", (e) => {
    const d = { ArrowLeft: [0, -1], ArrowRight: [0, 1], ArrowUp: [-1, 0], ArrowDown: [1, 0] }[e.key];
    if (!d || e.altKey || e.ctrlKey || e.metaKey) return;
    const t = e.target;
    if (t.closest && t.closest("input, select, textarea, [contenteditable], #tabs, .votes-grid")) return;
    if (app.dataset.tab === "simulate") {
      if (d[0] === 0 && stepBin(d[1])) e.preventDefault();
      return;
    }
    if (!st.scan || st.cell < 0) return;
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
    if (e.shiftKey || st.roiTool) {
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
    st.roiTool = false;
    const x = Math.min(roi.c0, roi.c1), y = Math.min(roi.r0, roi.r1), w = Math.abs(roi.c1 - roi.c0) + 1, h = Math.abs(roi.r1 - roi.r0) + 1;
    // a click without a drag clears the region
    const whole = w < 2 && h < 2;
    setRoi(whole ? { roi_x: 0, roi_y: 0, roi_w: 0, roi_h: 0 } : { roi_x: x, roi_y: y, roi_w: w, roi_h: h });
    status(whole ? "region of interest: the whole map" : `region of interest: ${w} × ${h} cells from column ${x}, row ${y}`);
  });
  map.addEventListener("mouseleave", () => { $("map-hover").textContent = ""; });
  // whatever is laid out by hand follows its pane
  new ResizeObserver(() => { if (st.scan) { layoutMap(); drawMapOverlay(); } }).observe(well);
  new ResizeObserver(() => { if (st.scan) { drawPc(); if (st.cmp.mode === "split") drawCompare(); } }).observe($("pats"));
  new ResizeObserver(() => { if (!$("dock-votes").hidden) votes.draw(); if (!$("dock-plane").hidden) planeView.draw(); drawZ(); }).observe(document.querySelector(".dock"));
  new ResizeObserver(() => showEnergy()).observe($("energy"));

  // drop files anywhere
  let depth = 0;
  window.addEventListener("dragenter", (e) => { if (e.dataTransfer?.types.includes("Files")) { depth++; $("dropveil").hidden = false; e.preventDefault(); } });
  window.addEventListener("dragleave", () => { depth = Math.max(0, depth - 1); if (!depth) $("dropveil").hidden = true; });
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => { e.preventDefault(); depth = 0; $("dropveil").hidden = true; if (e.dataTransfer?.files.length) takeFiles([...e.dataTransfer.files]); });
}

main();
