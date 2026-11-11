#!/usr/bin/env node
// The web app end to end in headless Chrome (WebGPU), driven over the DevTools protocol with no
// npm packages (Node 22+: the built-in WebSocket).
//
//   node tests/test_web.mjs                        the repo's app (zig build wasm first)
//   node tests/test_web.mjs --dir dist/web         a built site
//   CHROME=/path/to/chrome node tests/test_web.mjs
//   node tests/test_web.mjs --dir dist/web --save-pipelines dist/web/zebsdsolve/pipelines.json
//                                                  … and record the pipelines it compiled: the
//                                                  site's first-visit warm-up (web/zebsdsolve.js)
//
// The example (the nickel scan, read by the wasm module's own HDF5 reader) from a wrong pattern
// centre → Fit pattern centre → Fit plane → Index, then the fitted centre and the orientations
// against what kikuchipy refined for the scan, the same again in two sweeps and on binned
// patterns, the export, and the page's own moving parts: the steps' tabs, the pattern indexed
// on the spot, the search's volume (its slices, and that they follow the pattern centre), the
// averaging's preview, the map's key and zoom. Fails on any error the page logs.
import { spawn } from "node:child_process";
import { existsSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);
const dir = args.includes("--dir") ? args[args.indexOf("--dir") + 1] : null;
const savePipelines = args.includes("--save-pipelines") ? args[args.indexOf("--save-pipelines") + 1] : null;
const port = 4300 + Math.floor(Math.random() * 500);
const appUrl = `http://127.0.0.1:${port}/${dir ? "" : "web/app/"}`;

function chromePath() {
  if (process.env.CHROME) return process.env.CHROME;
  const c = {
    darwin: ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "/Applications/Chromium.app/Contents/MacOS/Chromium"],
    win32: ["C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe", "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe"],
    linux: ["/usr/bin/google-chrome", "/usr/bin/google-chrome-stable", "/usr/bin/chromium", "/usr/bin/chromium-browser"],
  }[process.platform] || [];
  const found = c.find((p) => existsSync(p));
  if (!found) throw new Error("no Chrome found; set CHROME");
  return found;
}

const failures = [];
const check = (name, ok, info = "") => {
  console.log(`${ok ? "ok  " : "FAIL"} ${name}${info ? "  " + info : ""}`);
  if (!ok) failures.push(name);
};

// ── the static server and the browser ──
const server = spawn(process.execPath, [join(ROOT, "scripts", "serve.mjs"), "--port", String(port), ...(dir ? ["--dir", dir] : [])], { stdio: "ignore" });
const profile = mkdtempSync(join(tmpdir(), "zebsdsolve-chrome-"));
const flags = [
  "--headless=new", "--remote-debugging-port=0", `--user-data-dir=${profile}`, "--no-first-run", "--no-default-browser-check",
  "--enable-unsafe-webgpu", "--ignore-gpu-blocklist", "--enable-features=Vulkan", "--window-size=1500,900",
];
const chrome = spawn(chromePath(), [...flags, "about:blank"], { stdio: ["ignore", "ignore", "pipe"] });
const cleanup = () => {
  try { chrome.kill(); } catch { /* gone */ }
  try { server.kill(); } catch { /* gone */ }
  try { rmSync(profile, { recursive: true, force: true }); } catch { /* locked */ }
};
process.on("exit", cleanup);

const wsUrl = await new Promise((res, rej) => {
  let err = "";
  const t = setTimeout(() => rej(new Error("Chrome did not start: " + err.slice(-500))), 30000);
  chrome.stderr.on("data", (d) => {
    err += d;
    const m = /DevTools listening on (ws:\/\/\S+)/.exec(err);
    if (m) { clearTimeout(t); res(m[1]); }
  });
});

// ── a minimal DevTools protocol client ──
const ws = new WebSocket(wsUrl);
await new Promise((r, j) => { ws.onopen = r; ws.onerror = j; });
let seq = 0;
const pending = new Map();
// the page's console errors and uncaught exceptions, with their stacks: what an error in the
// app's log came from (printed by "no errors in the log")
const pageErrors = [];
const stackOf = (st) => (st?.callFrames || []).slice(0, 6).map((f) => `${f.functionName || "(anonymous)"} ${f.url.split("/").pop()}:${f.lineNumber + 1}:${f.columnNumber + 1}`).join(" < ");
ws.onmessage = (ev) => {
  const m = JSON.parse(ev.data);
  if (m.method === "Runtime.consoleAPICalled" && m.params.type === "error") {
    pageErrors.push(`${m.params.args.map((a) => a.description || a.value || "").join(" ")} [${stackOf(m.params.stackTrace)}]`);
  } else if (m.method === "Runtime.exceptionThrown") {
    const d = m.params.exceptionDetails;
    pageErrors.push(`${d.exception?.description || d.text} [${stackOf(d.stackTrace)}]`);
  }
  if (m.id && pending.has(m.id)) {
    const { res, rej } = pending.get(m.id);
    pending.delete(m.id);
    if (m.error) rej(new Error(m.error.message)); else res(m.result);
  }
};
const send = (method, params = {}, sessionId) => new Promise((res, rej) => {
  const id = ++seq;
  pending.set(id, { res, rej });
  ws.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }));
});
const { targetId } = await send("Target.createTarget", { url: "about:blank" });
const { sessionId } = await send("Target.attachToTarget", { targetId, flatten: true });
const evaluate = async (expr, timeout = 300000) => {
  const r = await send("Runtime.evaluate", { expression: `(async () => { ${expr} })()`, awaitPromise: true, returnByValue: true, timeout }, sessionId);
  if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || r.exceptionDetails.text);
  return r.result.value;
};
await send("Runtime.enable", {}, sessionId);
await send("Page.navigate", { url: appUrl }, sessionId);

// Set a control of the rail by its label, as a user would.
const setField = (label, value) => `{
  const el = [...document.querySelectorAll(".field")].find((f) => f.querySelector("span")?.textContent === ${JSON.stringify(label)});
  const input = el.querySelector("input, select");
  if (input.type === "checkbox") input.checked = ${JSON.stringify(value === true)}; else input.value = ${JSON.stringify(String(value))};
  input.dispatchEvent(new Event("change"));
  await new Promise((r) => setTimeout(r, 60));
}`;

// Click a button and wait for the app's job to finish.
const job = (id) => `
  document.getElementById("${id}").click();
  for (let i = 0; i < 3000 && (document.getElementById("app").dataset.busy === "1" || i < 3); i++) await new Promise((r) => setTimeout(r, 100));`;

try {
  const ready = await evaluate(`
    // the page may not have loaded yet
    for (let i = 0; i < 600 && !/^ready/.test(document.getElementById("status")?.textContent || "") && !document.getElementById("status")?.classList.contains("err"); i++) await new Promise((r) => setTimeout(r, 100));
    return { status: document.getElementById("status").textContent, adapter: document.getElementById("adapter").textContent };`);
  check("the app starts on WebGPU", /^ready/.test(ready.status), `${ready.status} on ${ready.adapter}`);
  if (!/^ready/.test(ready.status)) throw new Error("no WebGPU: " + ready.status);

  // the example: the module reads the HDF5 scan and the master pattern itself
  const opened = await evaluate(`${job("btn-example")}
    const s = window.ze.scan, g = window.ze.geometry;
    return { s, pc: g.pc, phases: window.ze.phases, status: document.getElementById("status").textContent };`);
  check("the example opens", opened.s.rows === 55 && opened.s.cols === 75 && opened.s.patH === 60 && opened.s.format === "kikuchipy h5ebsd" && opened.phases.length === 1 && opened.phases[0].laue === "m-3m",
    `${opened.s.format}, ${opened.s.rows} × ${opened.s.cols}, phase ${opened.phases[0]?.name} (${opened.phases[0]?.pointGroupName})`);
  const pcFile = opened.pc;

  // the overview fills in by itself; the steps are tabs
  const seen = await evaluate(`
    for (let i = 0; i < 300 && !/overview:/.test(document.getElementById("log").textContent); i++) await new Promise((r) => setTimeout(r, 100));
    const shown = () => [...document.querySelectorAll("#panels .panel")].filter((p) => !p.hidden).map((p) => p.dataset.tab).join();
    const before = shown();
    document.getElementById("tab-index").click();
    return { overview: (/overview: [^\\n]*/.exec(document.getElementById("log").textContent) || [""])[0], before, after: shown(), done: document.getElementById("tab-data").dataset.done };`);
  check("the overview reads every pattern", /overview: 4,125 patterns/.test(seen.overview), seen.overview);
  check("the steps are tabs, one panel at a time", seen.before === "data" && seen.after === "index" && seen.done === "1", `${seen.before} → ${seen.after}`);

  // before anything is fitted or indexed, the selected pattern is indexed on the spot
  const now = await evaluate(`
    for (let i = 0; i < 200 && !/found now/.test(document.getElementById("pat-sim-cap").textContent); i++) await new Promise((r) => setTimeout(r, 100));
    return { cap: document.getElementById("pat-sim-cap").textContent, facts: document.getElementById("pat-facts").textContent, sim: document.getElementById("pat-sim").width };`);
  check("a pattern nothing has indexed is indexed on the spot", /found now/.test(now.cap) && /NCC0\.\d+/.test(now.facts) && now.sim === 60, `${now.cap}; ${now.facts.slice(0, 40)}`);

  // the fits, from a wrong pattern centre
  const fits = await evaluate(`
    window.ze.setGeometry({ pc: [0.50, 0.30, 0.55] });
    let t0 = performance.now();
    ${job("btn-fit-pc")}
    const vote = { pc: window.ze.geometry.pc, s: (performance.now() - t0) / 1000, votes: window.ze.pcScores().shape };
    // the votes view: every grid of the search, three slices each; a click moves the cross-hair
    const grids = [...document.querySelectorAll("#votes-levels button")];
    grids[0].click();
    const xy = document.getElementById("votes");
    const r = xy.getBoundingClientRect();
    const before = document.getElementById("votes-readout").textContent;
    xy.dispatchEvent(new MouseEvent("click", { clientX: r.left + r.width * 0.1, clientY: r.top + r.height * 0.9, bubbles: true }));
    // the slices show the ballots' density, or the mean correlation
    const shows = () => document.getElementById("votes").getContext("2d").getImageData(0, 0, 9, 9).data.join();
    const asBallots = shows();
    document.querySelector('#votes-what [data-what="mean"]').click();
    vote.what = { differ: asBallots !== shows(), said: document.getElementById("votes-readout").textContent.split("\\n").pop() };
    document.querySelector('#votes-what [data-what="ballots"]').click();
    // the patterns' own favourites are an overlay to ask for: off as the app opens
    const pats = document.getElementById("votes-pats");
    const marksOff = document.querySelectorAll("#votes-over circle").length + (pats.checked ? 1000 : 0);
    pats.click();
    vote.view = { marksOff, grids: grids.length, slices: ["votes", "votes-xz", "votes-yz"].map((id) => document.getElementById(id).width), moved: before !== document.getElementById("votes-readout").textContent,
      marks: document.querySelectorAll("#votes-over circle").length, readout: document.getElementById("votes-readout").textContent.split("\\n")[1] };
    // through the volume: the slider under a slice steps along its third axis; every slice
    // along an axis shows side by side
    pats.click();
    const cap = () => document.getElementById("votes-cap").textContent;
    const slide = document.getElementById("votes-slide");
    const at0 = cap();
    slide.value = String((+slide.value + 1) % (+slide.max + 1));
    slide.dispatchEvent(new Event("input"));
    const at1 = cap();
    document.getElementById("votes").parentElement.dispatchEvent(new WheelEvent("wheel", { deltaY: 120, bubbles: true, cancelable: true }));
    const at2 = cap();
    document.querySelector('#votes-mode [data-mode="z"]').click();
    vote.volume = { slid: at0 !== at1, scrolled: at1 !== at2, steps: +slide.max + 1, every: document.getElementById("votes-all").width, shown: document.getElementById("votes-all").getBoundingClientRect().width > 50, cap: at1 };
    document.querySelector('#votes-mode [data-mode="planes"]').click();
    grids[grids.length - 1].click();
    t0 = performance.now();
    ${job("btn-fit-plane")}
    // the plane of source points: each fit pattern's own, and the plane through them, in a view of their own
    const plane = { pc: window.ze.geometry.pc, s: (performance.now() - t0) / 1000, readout: document.getElementById("fit-readout").textContent, kind: window.ze.geometry.planeKind,
      said: document.getElementById("geo-readout").textContent, shown: !document.getElementById("dock-plane").hidden && document.getElementById("plane-canvas").width > 40,
      points: window.ze.planePoints().cells.length, cap: document.getElementById("plane-cap").textContent };
    document.querySelector('#dock-kind [data-dock="votes"]').click();
    // the plane fit moved the pattern centre: the slices cut through it again
    const cross = /cross-hair\\s+([\\d.]+) ([\\d.]+) ([\\d.]+)/.exec(document.getElementById("votes-readout").textContent).slice(1).map(Number);
    const follow = { cross, meta: document.getElementById("votes-meta").textContent, grid: document.querySelector("#votes-levels [aria-pressed=true]").textContent };
    // the whole fit by itself, from a pattern centre put well off
    window.ze.setGeometry({ pc: [plane.pc[0] + 0.03, plane.pc[1] - 0.03, plane.pc[2] + 0.02] });
    t0 = performance.now();
    ${job("btn-fit-auto")}
    const auto = { pc: window.ze.geometry.pc, rot: window.ze.geometry.rot, px: window.ze.geometry.pxUm, s: (performance.now() - t0) / 1000, readout: document.getElementById("fit-readout").textContent,
      grids: document.querySelectorAll("#votes-levels button").length, kind: window.ze.geometry.planeKind,
      // the plane's view came up by itself
      shown: !document.getElementById("dock-plane").hidden && document.getElementById("plane-canvas").width > 40 };
    document.querySelector('#dock-kind [data-dock="votes"]').click();
    await new Promise((r) => setTimeout(r, 50));
    const space = document.getElementById("votes-3d");
    const ink = () => { const d = space.getContext("2d").getImageData(0, 0, space.width, space.height).data; let n = 0, h = 0; for (let i = 0; i < d.length; i += 4) { if (d[i] + d[i + 1] + d[i + 2] > 60) n++; h = (h * 31 + d[i + 1]) | 0; } return [n, h]; };
    const still = ink();
    const box = space.getBoundingClientRect();
    space.dispatchEvent(new MouseEvent("mousedown", { button: 0, clientX: box.left + 40, clientY: box.top + 40, bubbles: true }));
    window.dispatchEvent(new MouseEvent("mousemove", { clientX: box.left + 110, clientY: box.top + 70 }));
    window.dispatchEvent(new MouseEvent("mouseup", { clientX: box.left + 110, clientY: box.top + 70 }));
    auto.volume = { lit: still[0], turned: ink()[1] !== still[1] };
    // the pattern and its simulation in one frame, either side of a bar that drags
    const cmpSel = document.getElementById("pat-cmp");
    cmpSel.value = "split";
    cmpSel.dispatchEvent(new Event("change"));
    // (the bar shows once the pattern's simulation is drawn: on a slow GPU, well after the change)
    const wipe = document.getElementById("pat-wipe"), cmp = document.getElementById("pat-cmp-well");
    for (let t = 0; t < 100 && wipe.hidden; t++) await new Promise((r) => setTimeout(r, 100));
    await new Promise((r) => setTimeout(r, 100));
    const wipe0 = wipe.getBoundingClientRect().left, cb = cmp.getBoundingClientRect();
    cmp.dispatchEvent(new MouseEvent("mousedown", { button: 0, clientX: cb.left + cb.width * 0.3, clientY: cb.top + cb.height / 2, bubbles: true }));
    window.dispatchEvent(new MouseEvent("mouseup", {}));
    auto.split = { shown: !wipe.hidden, moved: Math.abs(wipe.getBoundingClientRect().left - wipe0) > 5, cap: cmpSel.value };
    cmpSel.value = "diff";
    cmpSel.dispatchEvent(new Event("change"));
    // the static background, shown where the pattern is: the file's, the same for every cell
    const stageSel = document.getElementById("pat-stage");
    const pexp = document.getElementById("pat-exp");
    const inkOf = (c) => { const d = c.getContext("2d").getImageData(0, 0, c.width, c.height).data; let h = 0; for (let i = 0; i < d.length; i += 4) h = (h * 31 + d[i]) | 0; return h; };
    const sb = document.querySelector('.fields[data-group="bgstatic"] select');
    const sb0 = sb.value;
    sb.value = "file";
    sb.dispatchEvent(new Event("change"));
    await new Promise((r) => setTimeout(r, 100));
    stageSel.value = "stored";
    stageSel.dispatchEvent(new Event("change"));
    await new Promise((r) => setTimeout(r, 500));
    const storedInk = inkOf(pexp);
    stageSel.value = "background";
    stageSel.dispatchEvent(new Event("change"));
    await new Promise((r) => setTimeout(r, 500));
    auto.background = { other: inkOf(pexp) !== storedInk, cap: document.getElementById("pat-exp-cap").textContent, offWhenNone: false };
    sb.value = "none";
    sb.dispatchEvent(new Event("change"));
    await new Promise((r) => setTimeout(r, 300));
    auto.background.offWhenNone = stageSel.querySelector('[value="background"]').disabled && stageSel.value === "filtered";
    sb.value = sb0;
    sb.dispatchEvent(new Event("change"));
    await new Promise((r) => setTimeout(r, 100));
    // the region of interest, drawn on the map with its tool, and cleared
    document.getElementById("btn-roi").click();
    const mapc = document.getElementById("map"), mr = mapc.getBoundingClientRect();
    mapc.dispatchEvent(new MouseEvent("mousedown", { button: 0, clientX: mr.left + mr.width * 0.2, clientY: mr.top + mr.height * 0.2, bubbles: true }));
    window.dispatchEvent(new MouseEvent("mousemove", { clientX: mr.left + mr.width * 0.6, clientY: mr.top + mr.height * 0.6 }));
    window.dispatchEvent(new MouseEvent("mouseup", { clientX: mr.left + mr.width * 0.6, clientY: mr.top + mr.height * 0.6 }));
    await new Promise((r) => setTimeout(r, 100));
    auto.roi = { w: window.ze.settings.roi_w, h: window.ze.settings.roi_h, said: document.getElementById("roi-text").textContent, toolOff: document.getElementById("btn-roi").getAttribute("aria-pressed") === "false" };
    document.getElementById("btn-roi-clear").click();
    await new Promise((r) => setTimeout(r, 100));
    auto.roi.cleared = window.ze.settings.roi_w === 0 && document.getElementById("btn-roi-clear").hidden;
    // back to one pattern centre and the scan's steps
    document.getElementById("btn-geo-plane").click();
    await new Promise((r) => setTimeout(r, 150));
    auto.dropped = window.ze.geometry.planeKind;
    // the detector's pixel size: typed in, and cleared back to the scan's own
    const px = document.querySelector('#geometry-fields input[data-geo="pxUm"]');
    px.value = "500";
    px.dispatchEvent(new Event("change"));
    await new Promise((r) => setTimeout(r, 150));
    auto.pixel = { typed: window.ze.geometry.pxUm, said: document.getElementById("geo-readout").textContent };
    px.value = "";
    px.dispatchEvent(new Event("change"));
    await new Promise((r) => setTimeout(r, 150));
    auto.pixel.cleared = window.ze.geometry.pxUm;
    auto.pixel.shown = +px.value;
    // What follows compares the indexed orientations with kikuchipy's, which were refined with
    // the detector at its nominal attitude: the plane's sample frame is the nominal one unless
    // it is asked to follow the plane.
    auto.turned = window.ze.geometry.rot.map((v) => (v * 180) / Math.PI);
    return { vote, plane, auto, readout: document.getElementById("fit-readout").textContent, follow };`);
  const err = (pc) => Math.max(...pc.map((v, i) => Math.abs(v - pcFile[i])));
  check("Fit pattern centre votes near the scan's centre", err(fits.vote.pc) < 0.008, `${fits.vote.pc.map((v) => v.toFixed(4)).join(" ")} (${err(fits.vote.pc).toFixed(4)} off) in ${fits.vote.s.toFixed(1)} s`);
  check("the votes view shows how the search went", fits.vote.view.grids >= 3 && fits.vote.view.slices.join() === "9,9,9" && fits.vote.view.moved && fits.vote.view.marks > 0 && fits.vote.view.marksOff === 0,
    `${fits.vote.view.grids} grids, slices ${fits.vote.view.slices.join(" / ")} candidates wide, ${fits.vote.view.marks} candidates some pattern prefers; ${fits.vote.view.readout}`);
  check("the volume shows the ballots that chose, or the mean correlation", fits.vote.what.differ && /ballots lie by the winner/.test(fits.vote.what.said), fits.vote.what.said);
  check("its volume can be gone through slice by slice", fits.vote.volume.slid && fits.vote.volume.scrolled && fits.vote.volume.steps === 5 && fits.vote.volume.every > 9 && fits.vote.volume.shown,
    `${fits.vote.volume.steps} slices along z by the slider and the wheel (${fits.vote.volume.cap.trim()}); every z side by side, ${fits.vote.volume.every} cells across`);
  const off = Math.max(...fits.follow.cross.map((v, i) => Math.abs(v - fits.plane.pc[i])));
  check("the slices follow the pattern centre a fit moves", /cut through the pattern centre/.test(fits.follow.meta) && off < 0.006,
    `after Fit plane: cross-hair ${fits.follow.cross.join(" ")} in grid ${fits.follow.grid}, ${off.toFixed(4)} from the pattern centre`);
  check("Fit plane fits the plane of source points and shows it", err(fits.plane.pc) < 0.004 && fits.plane.kind === 1 && fits.plane.shown && fits.plane.points > 100 && /^plane\s+affine, [\d,]+ of [\d,]+ patterns$/m.test(fits.plane.readout)
    && /^tilt\s+[+−][\d.]+° ± [\d.]+°/m.test(fits.plane.readout) && /the fitted affine plane/.test(fits.plane.said) && /affine plane through/.test(fits.plane.cap),
    `${fits.plane.pc.map((v) => v.toFixed(5)).join(" ")} (${err(fits.plane.pc).toFixed(5)} from kikuchipy's) in ${fits.plane.s.toFixed(2)} s; ${fits.plane.points} points; ${fits.plane.readout.split("\n").slice(0, 4).join("; ")}`);
  // (the scan's file gives the detector's pixel: 70 µm binned by 8)
  const lines = fits.auto.readout.split("\n");
  const line = (start) => lines.find((l) => l.startsWith(start)) || "";
  check("Auto fit geometry is the vote, then the plane", err(fits.auto.pc) < 0.004 && fits.auto.grids >= 3 && fits.auto.px === 560 && fits.auto.shown && fits.auto.kind === 1 && fits.auto.dropped === 0
    && /^vote\s+[\d.]+\s+[\d.]+\s+[\d.]+$/.test(lines[0]) && /^plane\s+affine/.test(line("plane")) && /^centre\s+[\d.]+\s+[\d.]+\s+[\d.]+$/.test(line("centre")) && fits.auto.turned.every((v) => v === 0),
    `from 0.03 off: ${fits.auto.pc.map((v) => v.toFixed(5)).join(" ")} (${err(fits.auto.pc).toFixed(5)} from kikuchipy's), in ${fits.auto.s.toFixed(2)} s; "${lines[0]}"; "${line("plane")}"`);
  check("the detector's pixel size can be typed in, and cleared", fits.auto.pixel.typed === 500 && /typed in/.test(fits.auto.pixel.said) && fits.auto.pixel.cleared === 560 && fits.auto.pixel.shown === 560,
    `${fits.auto.pixel.said}; cleared: ${fits.auto.pixel.cleared} µm`);
  check("the pattern centre's volume can be turned about", fits.auto.volume.lit > 200 && fits.auto.volume.turned, `${fits.auto.volume.lit} pixels lit; a drag redraws it`);
  check("the pattern and its simulation split by a bar", fits.auto.split.shown && fits.auto.split.moved && fits.auto.split.cap === "split", fits.auto.split.cap);
  check("the static background shows where the pattern does (and is off when none is subtracted)", fits.auto.background.other && /file/.test(fits.auto.background.cap) && fits.auto.background.offWhenNone, fits.auto.background.cap);
  check("the region of interest is drawn on the map, and cleared", fits.auto.roi.w > 5 && fits.auto.roi.h > 5 && fits.auto.roi.toolOff && /region \d+ × \d+/.test(fits.auto.roi.said) && fits.auto.roi.cleared, fits.auto.roi.said);

  // index, against the orientations kikuchipy refined
  const idx = await evaluate(`
    const t0 = performance.now();
    ${job("btn-index")}
    const secs = (performance.now() - t0) / 1000;
    const R = window.ze.results();
    const ref = new Float32Array(await (await fetch("./presets/nickel/reference_quaternions.f32")).arrayBuffer());
    const dis = [];
    let n = 0, ncc = 0;
    for (let i = 0; i < R.phase.length; i++) {
      if (R.phase[i] < 0) continue;
      n++; ncc += R.ncc[i];
      dis.push(window.ze.disorientation(0, R.quaternions.subarray(4 * i, 4 * i + 4), ref.subarray(4 * i, 4 * i + 4)) * 180 / Math.PI);
    }
    dis.sort((a, b) => a - b);
    const ang = new TextDecoder().decode(window.ze.exportText("ang")).split("\\n").filter((l) => l && !l.startsWith("#"));
    const map = window.ze.map("ipf_z");
    return { n, ncc: ncc / n, median: dis[dis.length >> 1], within3: dis.filter((d) => d < 3).length / dis.length, secs, ang: ang.length, map: map.length,
      kind: document.querySelector("#map-kind [aria-pressed=true]").dataset.kind, stats: window.ze.stats(),
      key: { w: document.querySelector("#map-legend canvas")?.width || 0, names: [...document.querySelectorAll("#map-legend .key span")].map((e) => e.textContent).join(" ") } };`);
  check("Index covers the map", idx.n === 4125 && idx.kind === "ipf_z" && idx.map === 4125 * 4, `${idx.n} patterns in ${idx.secs.toFixed(2)} s (${(idx.n / idx.secs).toFixed(0)} patterns/s)`);
  check("the orientations are the ones kikuchipy refined", idx.median < 0.6 && idx.within3 > 0.97 && idx.ncc > 0.45, `median ${idx.median.toFixed(3)}° apart, within 3°: ${(100 * idx.within3).toFixed(1)} %, mean NCC ${idx.ncc.toFixed(3)}`);
  check("the export has every cell", idx.ang === 4125, `${idx.ang} lines`);
  check("the map says what its colours mean", idx.key.w > 20 && idx.key.names === "001 101 111", `an inverse pole figure key ${idx.key.w} px wide, corners ${idx.key.names}`);
  // another colouring from the Index tab: the map and its key follow it
  const keyed = await evaluate(`
    const before = window.ze.map("ipf_z").join().length;
    const pick = async (v) => { const sel = [...document.querySelectorAll("select")].find((s) => [...s.options].some((o) => o.value === "perfect_rotation")); sel.value = v; sel.dispatchEvent(new Event("change")); await new Promise((r) => setTimeout(r, 300)); };
    const c0 = document.getElementById("map").getContext("2d").getImageData(0, 0, 75, 55).data.join();
    await pick("perfect_rotation");
    const c1 = document.getElementById("map").getContext("2d").getImageData(0, 0, 75, 55).data.join();
    const out = { changed: c0 !== c1, setting: window.ze.settings.ipf_colors, w: document.querySelector("#map-legend canvas")?.width || 0,
      names: [...document.querySelectorAll("#map-legend .key span")].map((e) => e.textContent).join(" "), dimmed: [...document.querySelectorAll(".field")].filter((f) => /λ/.test(f.textContent)).map((f) => f.dataset.off).join("") };
    await pick("classic");
    out.back = [...document.querySelectorAll("#map-legend .key span")].map((e) => e.textContent).join(" ");
    out.dimmedAfter = [...document.querySelectorAll(".field")].filter((f) => /λ/.test(f.textContent)).map((f) => f.dataset.off).join("");
    return out;`);
  check("the colours can be the paper's, by orientation", keyed.changed && keyed.setting === "perfect_rotation" && keyed.w > 20 && keyed.names === "001 101 111 011" && keyed.dimmed === "00" && keyed.back === "001 101 111" && keyed.dimmedAfter === "11",
    `the key's corners ${keyed.names}; its two gradients' controls in use, and dimmed again under the classic colours`);

  // a toy scan made from the master loaded: it replaces the scan, indexes, and is checked against its own answer
  const toy = await evaluate(`
    const geo = window.ze.geometry, set = { ...window.ze.settings };
    document.getElementById("tab-data").click();
    window.ze.configure({ toy_seed: 3, toy_rows: 24, toy_cols: 32, toy_grains: 9, toy_counts: 20 });
    const enabled = !document.getElementById("btn-toy").disabled;
    ${job("btn-toy")}
    const scan = window.ze.scan, meta = document.getElementById("scan-meta").textContent;
    document.getElementById("tab-index").click();
    ${job("btn-index")}
    const said = document.getElementById("index-readout").textContent.split("\\n").pop();
    const out = { enabled, rows: scan.rows, cols: scan.cols, meta, said, median: +(/median ([\\d.]+)°/.exec(said) || [0, 99])[1] };
    return out;`);
  check("a toy scan can be made from the masters and indexed against its answer", toy.enabled && toy.rows === 24 && toy.cols === 32 && /toy scan, seed 3/.test(toy.meta) && toy.median < 0.5, `${toy.meta}; ${toy.said}`);
  // back to the example for what follows
  await evaluate(`${job("btn-example")} return 1;`);
  await evaluate(`${job("btn-index")} return 1;`);

  // the same in two sweeps, then with the search on binned patterns and refinement at the stored size
  const again = () => evaluate(`
    const t0 = performance.now();
    ${job("btn-index")}
    const secs = (performance.now() - t0) / 1000;
    const R = window.ze.results();
    const ref = new Float32Array(await (await fetch("./presets/nickel/reference_quaternions.f32")).arrayBuffer());
    const dis = [];
    const bySweep = {};
    let ncc = 0;
    for (let i = 0; i < R.phase.length; i++) {
      ncc += R.ncc[i] / R.phase.length;
      bySweep[R.sweep[i]] = (bySweep[R.sweep[i]] || 0) + 1;
      dis.push(window.ze.disorientation(0, R.quaternions.subarray(4 * i, 4 * i + 4), ref.subarray(4 * i, 4 * i + 4)) * 180 / Math.PI);
    }
    dis.sort((a, b) => a - b);
    return { median: dis[dis.length >> 1], within3: dis.filter((d) => d < 3).length / dis.length, secs, bySweep, ncc, shape: window.ze.patternShape };`);
  await evaluate(`${setField("sweeps", 2)} return 1;`);
  const sw = await again();
  // (on a scan this small a pass is a third of a second, most of it the round trips of its 17
  // batches, which two sweeps make too: the sweeps' gain shows on large scans)
  check("two sweeps index it as one pass does", sw.median < 0.6 && sw.within3 > 0.965 && sw.bySweep[0] > 3300 && sw.bySweep[1] > 0 && sw.secs < 1.5 * idx.secs,
    `median ${sw.median.toFixed(3)}°, within 3°: ${(100 * sw.within3).toFixed(1)} %, accepted at bandwidth 32: ${sw.bySweep[0]}, at 64: ${sw.bySweep[1]}, never: ${sw.bySweep[-1] || 0}; ${sw.secs.toFixed(2)} s (one pass: ${idx.secs.toFixed(2)} s)`);
  // Tune speed: trials timed in the page; the settings it lands on show in the controls
  const tune = await evaluate(`
    const before = window.ze.settings.batch;
    ${job("btn-tune")}
    const said = (/tuned on .*?per second/.exec(document.getElementById("log").textContent) || [""])[0];
    const s = window.ze.settings;
    const shown = [...document.querySelectorAll(".field")].find((f) => f.querySelector("span")?.textContent === "patterns per step").querySelector("input").value;
    return { said, batch: s.batch, shown, before };`);
  check("Tune speed times a sample and sets the batch", tune.said.startsWith("tuned on") && String(tune.batch) === tune.shown, tune.said);
  await evaluate(`${setField("patterns per step", 256)} ${setField("pixels per invocation", 64)} ${setField("correlation memory (MiB)", 256)} ${setField("workgroup, correlation", 64)} ${setField("workgroup, refinement", 64)} return 1;`);

  // averaged with their neighbours first (NLPAR): the pattern view shows the average, and
  // shows another when the averaging's settings change
  const look = `
    const run = async () => {
      await new Promise((r) => setTimeout(r, 150));
      for (let i = 0; i < 100 && !/averaged/.test(document.getElementById("pat-exp-cap").textContent); i++) await new Promise((r) => setTimeout(r, 100));
      await new Promise((r) => setTimeout(r, 400));
      const c = document.getElementById("pat-exp");
      return { px: Array.from(c.getContext("2d").getImageData(0, 0, c.width, c.height).data.filter((_, i) => i % 4 === 0)), cap: document.getElementById("pat-exp-cap").textContent,
        said: document.getElementById("nlpar-readout").textContent, shares: document.getElementById("pat-shares").hidden ? 0 : document.getElementById("pat-shares").width };
    };`;
  const pre = await evaluate(`${setField("sweeps", 1)} ${look}
    const c = document.getElementById("pat-exp");
    const plain = Array.from(c.getContext("2d").getImageData(0, 0, c.width, c.height).data.filter((_, i) => i % 4 === 0));
    ${setField("average with neighbours (NLPAR)", true)}
    const on = await run();
    ${setField("lambda", 1.4)}
    const wide = await run();
    ${setField("lambda", 0.7)}
    ${setField("cells each way", 1)}
    const small = await run();
    ${setField("cells each way", 3)}
    const differ = (a, b) => a.reduce((n, v, i) => n + (v !== b[i]), 0) / a.length;
    return { on: differ(plain, on.px), lambda: differ(on.px, wide.px), radius: differ(on.px, small.px), cap: on.cap, said: on.said.split("\\n").slice(1).join("; "), saidWide: wide.said.split("\\n")[1], shares: [on.shares, small.shares] };`);
  check("the pattern view shows the averaging, and follows its settings", pre.on > 0.3 && pre.lambda > 0.3 && pre.radius > 0.3 && /averaged 7 × 7/.test(pre.cap) && pre.shares.join() === "7,3" && /of it left/.test(pre.said) && pre.said !== pre.saidWide,
    `${pre.cap}: ${(100 * pre.on).toFixed(0)} % of its pixels changed; λ 0.7 → 1.4 changed ${(100 * pre.lambda).toFixed(0)} %, 7 × 7 → 3 × 3 ${(100 * pre.radius).toFixed(0)} %; ${pre.said}`);
  const nl = await again();
  const said = await evaluate(`return (/neighbour averaging: .*?itself/.exec(document.getElementById("log").textContent) || [""])[0];`);
  check("patterns averaged with their neighbours index better", nl.median < 0.6 && nl.within3 > 0.965 && nl.ncc > idx.ncc + 0.03 && said.includes("% of it left"),
    `median ${nl.median.toFixed(3)}°, within 3°: ${(100 * nl.within3).toFixed(1)} %, mean NCC ${idx.ncc.toFixed(3)} → ${nl.ncc.toFixed(3)}; ${said}`);
  await evaluate(`${setField("average with neighbours (NLPAR)", false)} return 1;`);
  await evaluate(`${setField("sweeps", 1)} ${setField("reduce by", 2)} ${setField("at stored size", "refine")} return 1;`);
  const bin = await again();
  check("a search on binned patterns, refined at the stored size", bin.shape.join() === "30,30" && bin.median < 0.6 && bin.within3 > 0.965,
    `search at ${bin.shape.join(" × ")}: median ${bin.median.toFixed(3)}°, within 3°: ${(100 * bin.within3).toFixed(1)} %`);
  await evaluate(`${setField("reduce by", 1)} return 1;`);

  // a first sweep on the dictionary's components, the correlation for what it leaves
  await evaluate(`${setField("sweeps", 2)} ${setField("earlier sweeps by", "pca")} ${setField("first dictionary grid", 40)} return 1;`);
  const mix = await again();
  check("dictionary components first, the correlation for the rest", mix.median < 0.6 && mix.within3 > 0.965 && mix.bySweep[0] > 3300 && mix.bySweep[1] > 0,
    `median ${mix.median.toFixed(3)}°, within 3°: ${(100 * mix.within3).toFixed(1)} %, accepted by the components: ${mix.bySweep[0]}, by the correlation: ${mix.bySweep[1]}, never: ${mix.bySweep[-1] || 0}`);
  await evaluate(`${setField("sweeps", 1)} return 1;`);

  // the map zooms in about the pointer and never out past the whole map
  const zoom = await evaluate(`
    const well = document.getElementById("map-well"), map = document.getElementById("map");
    const r = well.getBoundingClientRect();
    const size = () => [map.getBoundingClientRect().width, map.getBoundingClientRect().height];
    const whole = size();
    const wheel = (dy) => well.dispatchEvent(new WheelEvent("wheel", { deltaY: dy, clientX: r.left + r.width / 3, clientY: r.top + r.height / 3, bubbles: true, cancelable: true }));
    for (let i = 0; i < 5; i++) wheel(-300);
    const zin = size();
    for (let i = 0; i < 40; i++) wheel(300);
    const out = size();
    for (let i = 0; i < 5; i++) wheel(-300);
    document.getElementById("btn-zoom").click();
    return { whole, zin, out, reset: size(), fits: whole[0] <= r.width + 1 && whole[1] <= r.height + 1 };`);
  check("the map zooms in, and out only to the whole map", zoom.fits && zoom.zin[0] > 3 * zoom.whole[0] && Math.abs(zoom.out[0] - zoom.whole[0]) < 1 && Math.abs(zoom.reset[0] - zoom.whole[0]) < 1,
    `whole ${zoom.whole.map((v) => v.toFixed(0)).join(" × ")}, zoomed ${zoom.zin.map((v) => v.toFixed(0)).join(" × ")}, back out ${zoom.out.map((v) => v.toFixed(0)).join(" × ")}`);
  // the arrow keys move the selected pattern, the map focused or not
  const keys = await evaluate(`
    document.body.focus();
    const was = document.getElementById("pat-cell").textContent;
    document.body.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowDown", bubbles: true, cancelable: true }));
    document.body.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowRight", bubbles: true, cancelable: true }));
    await new Promise((r) => setTimeout(r, 200));
    return { was, now: document.getElementById("pat-cell").textContent };`);
  const rc = (t) => (/row (\d+), col (\d+)/.exec(t) || []).slice(1).map(Number);
  check("the arrow keys move the selected pattern on the map", rc(keys.now)[0] === rc(keys.was)[0] + 1 && rc(keys.now)[1] === rc(keys.was)[1] + 1, `${keys.was} → ${keys.now}`);

  // the pattern view follows the map
  const view = await evaluate(`
    for (let i = 0; i < 100 && !/NCC/.test(document.getElementById("pat-facts").textContent); i++) await new Promise((r) => setTimeout(r, 100));
    return { facts: document.getElementById("pat-facts").textContent, sim: document.getElementById("pat-sim").width, cap: document.getElementById("pat-sim-cap").textContent };`);
  check("the selected pattern shows its simulation", /NCC/.test(view.facts) && view.sim === 60 && /indexed/.test(view.cap), view.cap);

  // a project file dropped on the window brings its settings and geometry back
  const proj = await evaluate(`
    const g = window.ze.geometry;
    const file = new File([JSON.stringify({ zebsdsolve: window.ze.version, geometry: { ...g, pc: [0.41, 0.22, 0.51] }, settings: { ...window.ze.settings, bandwidth: 48, accept_z: 12, pose_grid: 15 } })], "saved.zebsdsolve.json");
    const dt = new DataTransfer();
    dt.items.add(file);
    window.dispatchEvent(new DragEvent("drop", { dataTransfer: dt, bubbles: true, cancelable: true }));
    for (let i = 0; i < 100 && window.ze.settings.bandwidth !== 48; i++) await new Promise((r) => setTimeout(r, 50));
    await new Promise((r) => setTimeout(r, 300));
    const shown = (label) => [...document.querySelectorAll(".field")].find((f) => f.querySelector("span")?.textContent === label).querySelector("input").value;
    const out = { s: window.ze.settings, pc: window.ze.geometry.pc, bw: shown("bandwidth L"), pcShown: [...document.querySelectorAll('#geometry-fields input[data-geo="pc"]')].map((i) => +i.value) };
    window.ze.configure({ bandwidth: 64, accept_z: 15 });
    window.ze.setGeometry(g);
    return out;`);
  // (pose_grid: a setting of an earlier version, which this one leaves out and says so)
  check("a project file brings its settings and geometry back", proj.s.bandwidth === 48 && proj.s.accept_z === 12 && Math.abs(proj.pc[0] - 0.41) < 1e-12 && proj.pcShown.join() === "0.41,0.22,0.51",
    `bandwidth ${proj.s.bandwidth}, accept at z ≥ ${proj.s.accept_z}, pattern centre ${proj.pcShown.join(" ")}`);

  // the Simulate step: a preset crystal (GaN: no centre of symmetry, both hemispheres solved),
  // its master simulated and shown, then added as a candidate phase
  const sim = await evaluate(`
    document.getElementById("tab-simulate").click();
    const sel = document.getElementById("cif-preset");
    sel.value = "GaN";
    sel.dispatchEvent(new Event("change"));
    for (let i = 0; i < 100 && !document.getElementById("crystal-group").textContent; i++) await new Promise((r) => setTimeout(r, 50));
    ${setField("half-width (px)", 60)}
    const before = window.ze.phases.length;
    ${job("btn-simulate")}
    const readout = [...document.querySelectorAll("#log > div")].map((d) => d.textContent).filter((t) => /the fundamental sector of/.test(t)).pop() || "";
    // shown in the master view (the engine's picture), in each of its projections
    await new Promise((r) => setTimeout(r, 400));
    const mc = document.querySelector("#master-well canvas");
    const inkOf = (c) => { const d = c.getContext("2d").getImageData(0, 0, c.width, c.height).data; let h = 0, n = 0; for (let i = 0; i < d.length; i += 4) { h = (h * 31 + d[i]) | 0; if (d[i] > 40) n++; } return [h, n]; };
    const shown = !mc.hidden && mc.width > 100 && document.getElementById("master-src").value === "-1";
    const dock = document.querySelector("#dock-kind [aria-pressed=true]").dataset.dock;
    const views = {};
    for (const p of ["globe", "lambert", "stereographic", "joint"]) {
      document.querySelector('#view-mode [data-view="' + p + '"]').click();
      await new Promise((r) => setTimeout(r, 500));
      views[p] = inkOf(mc);
    }
    document.querySelector('#view-mode [data-view="globe"]').click();
    const distinct = new Set(Object.values(views).map((v) => v[0])).size;
    // the cell beside it, and how the electrons spread
    const cc = document.querySelector("#cell-well canvas");
    const cell = { drawn: !cc.hidden && cc.width > 100 && inkOf(cc)[1] > 500, legend: document.querySelector("#cell-well .cell-legend").textContent, labels: [...document.querySelectorAll("#cell-well .edge-label")].map((e) => e.textContent).join(""),
      energy: document.getElementById("energy-readout").textContent, bars: document.querySelectorAll("#energy-e .bar.on").length };
    ${job("btn-sim-phase")}
    const ph = window.ze.phases;
    return { group: document.getElementById("crystal-group").textContent, readout, shown, dock, distinct, cell, added: ph.length - before, last: ph[ph.length - 1], npz: window.ze.simulationNpz().length,
      sources: [...document.getElementById("master-src").options].map((o) => o.value).join(",") };`);
  check("the Simulate step simulates a crystal's master and adds it as a phase", /P 63 m c\s*\(186\)/.test(sim.group) && /directions/.test(sim.readout) && sim.shown && sim.dock === "master" && sim.added === 1 && sim.last.name === "GaN" && sim.last.side === 121 && sim.npz > 1000,
    `${sim.readout.split("\n")[0]}; phase ${sim.last?.name} ${sim.last?.pointGroupName} ${sim.last?.side} px a side`);
  check("the master view draws four projections, and lists every master", sim.distinct === 4 && sim.sources.split(",").includes("-1") && sim.sources.split(",").length >= 2, `${sim.distinct} distinct pictures; sources ${sim.sources}`);
  // the flat projections' lock, control-scroll's spin, the cell and the master turning together,
  // the arrow keys through the energy bins and the integrated button
  const views = await evaluate(`
    const mc = document.querySelector("#master-well canvas");
    const inkOf = (c) => { const d = c.getContext("2d").getImageData(0, 0, c.width, c.height).data; let h = 0; for (let i = 0; i < d.length; i += 4) h = (h * 31 + d[i]) | 0; return h; };
    const wait = (ms) => new Promise((r) => setTimeout(r, ms));
    document.querySelector('#view-mode [data-view="lambert"]').click();
    await wait(500);
    const r = mc.getBoundingClientRect();
    const drag = (el, x0, y0, x1, y1) => { const ev = (t, x, y) => el.dispatchEvent(new PointerEvent(t, { clientX: x, clientY: y, button: 0, pointerId: 1, bubbles: true })); ev("pointerdown", x0, y0); ev("pointermove", (x0 + x1) / 2, (y0 + y1) / 2); ev("pointermove", x1, y1); ev("pointerup", x1, y1); };
    const before = inkOf(mc);
    drag(mc, r.left + r.width * 0.3, r.top + r.height * 0.5, r.left + r.width * 0.35, r.top + r.height * 0.4);
    await wait(500);
    const panned = inkOf(mc) !== before;
    mc.dispatchEvent(new WheelEvent("wheel", { deltaY: -200, ctrlKey: true, bubbles: true, cancelable: true }));
    await wait(500);
    const spun = inkOf(mc) !== before;
    document.querySelector('#view-mode [data-view="globe"]').click();
    await wait(400);
    const link = document.getElementById("cell-link");
    link.checked = true;
    link.dispatchEvent(new Event("change"));
    await wait(400);
    const g0 = inkOf(mc);
    const cc = document.querySelector("#cell-well canvas");
    const cr = cc.getBoundingClientRect();
    drag(cc, cr.left + cr.width * 0.5, cr.top + cr.height * 0.5, cr.left + cr.width * 0.62, cr.top + cr.height * 0.45);
    await wait(600);
    const linked = inkOf(mc) !== g0;
    link.checked = false;
    link.dispatchEvent(new Event("change"));
    document.body.focus();
    const key = (k) => document.body.dispatchEvent(new KeyboardEvent("keydown", { key: k, bubbles: true, cancelable: true }));
    key("ArrowRight");
    await wait(300);
    const picked = () => { const b = document.querySelector("#energy-e .bar.picked"); return b ? b.dataset.bin : ""; };
    const right = picked();
    key("ArrowRight");
    await wait(300);
    const right2 = picked();
    const allOff = document.getElementById("energy-all").getAttribute("aria-pressed") === "false";
    document.getElementById("energy-all").click();
    await wait(300);
    return { panned, spun, linked, right, right2, allOff, back: picked() === "" && document.getElementById("energy-all").getAttribute("aria-pressed") === "true" };`);
  check("a locked flat projection pans on a drag and spins on control-scroll; linked, the cell turns the master", views.panned && views.spun && views.linked, JSON.stringify(views));
  check("the arrow keys step the energy bins, and integrated goes back to them all", views.right === "0" && views.right2 === "1" && views.allOff && views.back, `bins ${views.right}, ${views.right2}`);
  // the Simulate step's dock is the master's alone; a bin's bar and a site's name show their own
  // master; a site's B and the free lattice parameters are typed in, the tied ones following
  const parts = await evaluate(`
    const mc = document.querySelector("#master-well canvas");
    const inkOf = (c) => { const d = c.getContext("2d").getImageData(0, 0, c.width, c.height).data; let h = 0; for (let i = 0; i < d.length; i += 4) h = (h * 31 + d[i]) | 0; return h; };
    const docks = document.getElementById("dock-kind").offsetParent === null;
    document.querySelector('#master-src').value = "-1";
    document.querySelector('#master-src').dispatchEvent(new Event("change"));
    await new Promise((r) => setTimeout(r, 400));
    const all = inkOf(mc);
    const bar = document.querySelector("#energy-e .bar.pickable");
    bar.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    await new Promise((r) => setTimeout(r, 500));
    const bin = { value: (document.querySelector("#energy-e .bar.picked") || { dataset: { bin: "" } }).dataset.bin, other: inkOf(mc) !== all, marked: !!document.querySelector("#energy-e .bar.picked") };
    document.querySelector("#energy-e .bar.picked").dispatchEvent(new MouseEvent("click", { bubbles: true }));
    document.querySelector('#crystal-sites .site[data-site="1"] .sym').click();
    await new Promise((r) => setTimeout(r, 500));
    const site = { value: String([...document.getElementById("master-site").children].findIndex((b) => b.getAttribute("aria-pressed") === "true") - 1), binBack: !document.querySelector("#energy-e .bar.picked"), other: inkOf(mc) !== all };
    document.querySelector('#crystal-sites .site[data-site="1"] .sym').click();
    const b = document.querySelectorAll('#crystal-sites .site[data-site="0"] input')[1];
    b.value = "0.8";
    b.dispatchEvent(new Event("change"));
    const a = document.querySelector("#cellpars .par input");
    a.value = "3.2";
    a.dispatchEvent(new Event("change"));
    const c = window.ze.crystal();
    return { docks, bin, site, b: c.sites[0].b_iso, cell: c.cell, free: c.free.join(), reading: !!c.reading };`);
  check("on the Simulate step the dock is the master's, and a bin's bar or a site shows its own", parts.docks && parts.bin.value === "0" && parts.bin.other && parts.bin.marked && parts.site.value === "1" && parts.site.binBack && parts.site.other,
    `bin ${parts.bin.value}, site ${parts.site.value}`);
  // a master switched off stays listed and leaves the engine; the globe zooms about its middle; the
  // cell as 3 × 3 × 3 cells
  const more = await evaluate(`
    const wait = (ms) => new Promise((r) => setTimeout(r, ms));
    const n0 = window.ze.phases.length;
    const items = () => [...document.querySelectorAll("#phases li")];
    const last = items().length - 1;
    items()[last].querySelector("input.on").click();
    await wait(1200);
    const off = { engine: window.ze.phases.length, listed: items().length, dimmed: items()[last].dataset.on === "0" };
    items()[last].querySelector("input.on").click();
    await wait(1200);
    const on = { engine: window.ze.phases.length, back: items()[last].dataset.on === "1" };
    // switched while the pattern view is drawing (which, on a slow GPU, it still is when the
    // engine's phases are emptied and filled again): no error, and the view names a phase
    // that is there
    // (each phase taken back 400 ms late, as a slow GPU takes it: the view's drawing goes on
    // while the list is empty or half full)
    const errsBefore = document.querySelectorAll("#log .err").length;
    const stage = document.getElementById("pat-stage");
    const addPhase = window.ze.addPhase;
    window.ze.addPhase = async function (f) { await wait(400); return addPhase.call(this, f); };
    try {
      for (let t = 0; t < 4; t++) {
        const k = t % 2 ? last : 0;
        stage.dispatchEvent(new Event("change"));
        await wait(t * 40);
        items()[k].querySelector("input.on").click();
        await wait(1500);
        stage.dispatchEvent(new Event("change"));
        items()[k].querySelector("input.on").click();
        await wait(1500);
      }
    } finally {
      window.ze.addPhase = addPhase;
    }
    on.raced = { errors: document.querySelectorAll("#log .err").length - errsBefore, engine: window.ze.phases.length, said: [...document.querySelectorAll("#log .err")].slice(errsBefore).map((e) => e.textContent).join(" | ") };
    document.querySelector("#tab-simulate").click();
    document.querySelector('#view-mode [data-view="globe"]').click();
    await wait(500);
    const gc = document.querySelector("#master-well canvas");
    const inkOf = (c) => { const d = c.getContext("2d").getImageData(0, 0, c.width, c.height).data; let h = 0; for (let i = 0; i < d.length; i += 4) h = (h * 31 + d[i]) | 0; return h; };
    const r = gc.getBoundingClientRect();
    const zoomAt = async (fx) => { gc.dispatchEvent(new WheelEvent("wheel", { deltaY: -300, clientX: r.left + r.width * fx, clientY: r.top + r.height * 0.5, bubbles: true, cancelable: true })); await wait(500); return inkOf(gc); };
    const left = await zoomAt(0.1);
    gc.dispatchEvent(new WheelEvent("wheel", { deltaY: 300, bubbles: true, cancelable: true }));
    await wait(500);
    const right = await zoomAt(0.9);
    gc.dispatchEvent(new WheelEvent("wheel", { deltaY: 300, bubbles: true, cancelable: true }));
    await wait(300);
    const cell = document.querySelector("#cell-well canvas");
    const c1 = inkOf(cell);
    const sup = document.getElementById("cell-super");
    sup.checked = true;
    sup.dispatchEvent(new Event("change"));
    await wait(700);
    const c3 = inkOf(cell);
    const atoms3 = window.ze.cellInfo(3).atoms, atoms1 = window.ze.cellInfo(1).atoms;
    sup.checked = false;
    sup.dispatchEvent(new Event("change"));
    await wait(300);
    return { n0, off, on, sameZoom: left === right, cellChanged: c3 !== c1, atoms1, atoms3 };`);
  check("a master pattern can be switched off and on without removing it", more.off.engine === more.n0 - 1 && more.off.listed === more.n0 && more.off.dimmed && more.on.engine === more.n0 && more.on.back
    && more.on.raced.errors === 0 && more.on.raced.engine === more.n0, JSON.stringify(more));
  check("the globe zooms about its middle, wherever the pointer is", more.sameZoom, "");
  check("the cell can be drawn as 3 × 3 × 3 cells", more.cellChanged && more.atoms3 > 20 * more.atoms1 * 0.5 && more.atoms3 > more.atoms1, `${more.atoms1} atoms, ${more.atoms3} in 27 cells`);

  // the globe as a stereogram (two views as far apart as the eyes), and the Simulate panel's order
  const ster = await evaluate(`
    const wait = (ms) => new Promise((r) => setTimeout(r, ms));
    document.querySelector('#view-mode [data-view="globe"]').click();
    await wait(400);
    const mc = document.querySelector("#master-well canvas");
    const inkOf = (c) => { const d = c.getContext("2d").getImageData(0, 0, c.width, c.height).data; let h = 0; for (let i = 0; i < d.length; i += 4) h = (h * 31 + d[i]) | 0; return h; };
    const before = inkOf(mc);
    const st = document.getElementById("view-stereo");
    st.checked = true;
    st.dispatchEvent(new Event("input"));
    await wait(700);
    const on = { changed: inkOf(mc) !== before, bar: !document.getElementById("stereo-bar").hidden, labels: document.getElementById("stereo-dist-out").textContent + "/" + document.getElementById("stereo-ipd-out").textContent, labelsOver: document.querySelectorAll("#master-well .axis-label").length };
    const ipd = document.getElementById("stereo-ipd");
    ipd.value = "70";
    ipd.dispatchEvent(new Event("input"));
    await wait(600);
    const wider = inkOf(mc) !== before;
    st.checked = false;
    st.dispatchEvent(new Event("input"));
    await wait(500);
    ipd.value = "60";
    ipd.dispatchEvent(new Event("input"));
    const panel = document.querySelector('.panel[data-tab="simulate"]');
    const order = ["btn-simulate", "sim-result"].map((id) => document.getElementById(id).getBoundingClientRect().top);
    const beam = panel.querySelector('[data-group="simbeam"]').getBoundingClientRect().top;
    return { on, wider, simAbove: order[0] < beam && order[1] < beam, bar: document.querySelector(".dock-pane > .pane-bar").offsetParent === null };`);
  check("the globe can be a stereogram, the Simulate button and its result sit above the beam settings, the dock's bar is gone there", ster.on.changed && ster.on.bar && ster.on.labels === "500 mm/60 mm" && ster.on.labelsOver >= 2 && ster.wider && ster.simAbove && ster.bar, JSON.stringify(ster));

  // one hemisphere for a crystal with a centre of symmetry, both (and their difference) without;
  // zoom about the pointer and pan on the flat projections
  const halves = await evaluate(`
    const mc = document.querySelector("#master-well canvas");
    const inkOf = (c) => { const d = c.getContext("2d").getImageData(0, 0, c.width, c.height).data; let h = 0; for (let i = 0; i < d.length; i += 4) h = (h * 31 + d[i] + 7 * d[i + 2]) | 0; return h; };
    const wait = (ms) => new Promise((r) => setTimeout(r, ms));
    const pressed = () => [...document.getElementById("view-half").children].filter((b) => b.getAttribute("aria-pressed") === "true").map((b) => b.dataset.half).join();
    const src = document.getElementById("master-src");
    document.querySelector('#view-mode [data-view="lambert"]').click();
    src.value = "0";
    src.dispatchEvent(new Event("change"));
    await wait(500);
    const centro = { pressed: pressed(), diffOff: document.querySelector('#view-half [data-half="diff"]').disabled };
    src.value = "-1";
    src.dispatchEvent(new Event("change"));
    await wait(500);
    const non = { pressed: pressed(), diffOn: !document.querySelector('#view-half [data-half="diff"]').disabled };
    const both = inkOf(mc);
    document.querySelector('#view-half [data-half="north"]').click();
    await wait(400);
    const north = inkOf(mc);
    document.querySelector('#view-half [data-half="diff"]').click();
    await wait(400);
    const diff = inkOf(mc);
    document.querySelector('#view-half [data-half="both"]').click();
    await wait(400);
    const r = mc.getBoundingClientRect();
    mc.dispatchEvent(new WheelEvent("wheel", { deltaY: -500, clientX: r.left + r.width * 0.3, clientY: r.top + r.height * 0.5, bubbles: true, cancelable: true }));
    await wait(500);
    const zoomed = inkOf(mc);
    return { centro, non, distinct: new Set([both, north, diff]).size, zoomed: zoomed !== both };`);
  check("a crystal with a centre of symmetry shows one hemisphere, one without both and their difference; the projections zoom", halves.centro.pressed === "north" && halves.centro.diffOff && halves.non.pressed === "both" && halves.non.diffOn && halves.distinct === 3 && halves.zoomed, JSON.stringify(halves));
  check("a site's B and the free lattice parameters can be typed in, the tied ones following", Math.abs(parts.b - 0.8) < 1e-9 && parts.cell[0] === 3.2 && parts.cell[1] === 3.2 && parts.cell[5] === 120 && parts.free === "true,false,true,false,false,false" && parts.reading,
    `B ${parts.b} Å², cell ${parts.cell.map((v) => +v.toFixed(4)).join(" ")}`);
  check("the crystal's cell is drawn and named, with its electrons' energy and depth", sim.cell.drawn && /Ga/.test(sim.cell.legend) && /N/.test(sim.cell.legend) && sim.cell.labels === "abc" && /mean depth/.test(sim.cell.energy) && sim.cell.bars > 0,
    `legend ${sim.cell.legend}; ${sim.cell.energy}`);

  const errs = await evaluate(`return [...document.querySelectorAll("#log .err")].map((e) => e.textContent).slice(0, 5);`);
  check("no errors in the log", errs.length === 0, [...errs, ...(errs.length ? pageErrors.slice(0, 5).map((e) => `console: ${e}`) : [])].join(" | "));

  // a CIF the reader cannot take: refused, with the reason (and an error in the log, rightly)
  const refused = await evaluate(`
    const dt = new DataTransfer();
    dt.items.add(new File(["data_x\\n_cell_length_a 3\\n"], "broken.cif"));
    window.dispatchEvent(new DragEvent("drop", { dataTransfer: dt, bubbles: true, cancelable: true }));
    await new Promise((r) => setTimeout(r, 300));
    return document.getElementById("status").textContent;`);
  check("a CIF the reader cannot take is refused, saying why", /broken\.cif: the CIF is missing one or more cell parameters/.test(refused), refused);

  if (savePipelines) {
    const rec = await evaluate(`return window.ze.warmList();`);
    check("the pipelines it compiled", rec?.list?.length > 0, `${rec?.list?.length} pipelines → ${savePipelines}`);
    if (rec?.list?.length) writeFileSync(savePipelines, JSON.stringify(rec));
  }
} catch (e) {
  check("the browser run", false, String(e.stack || e));
} finally {
  ws.close();
  cleanup();
}
console.log("web", failures.length ? `FAILED: ${failures.join(", ")}` : "OK");
process.exit(failures.length ? 1 : 0);
