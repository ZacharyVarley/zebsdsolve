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
// centre → Fit pattern centre → Fit pose → Index, then the fitted centre and the orientations
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
ws.onmessage = (ev) => {
  const m = JSON.parse(ev.data);
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
    ${job("btn-fit-pose")}
    // the pose fit moved the pattern centre: the slices cut through it again
    const cross = /cross-hair\\s+([\\d.]+) ([\\d.]+) ([\\d.]+)/.exec(document.getElementById("votes-readout").textContent).slice(1).map(Number);
    return { vote, pose: { pc: window.ze.geometry.pc, s: (performance.now() - t0) / 1000 }, readout: document.getElementById("fit-readout").textContent,
      follow: { cross, meta: document.getElementById("votes-meta").textContent, grid: document.querySelector("#votes-levels [aria-pressed=true]").textContent } };`);
  const err = (pc) => Math.max(...pc.map((v, i) => Math.abs(v - pcFile[i])));
  check("Fit pattern centre votes near the scan's centre", err(fits.vote.pc) < 0.008, `${fits.vote.pc.map((v) => v.toFixed(4)).join(" ")} (${err(fits.vote.pc).toFixed(4)} off) in ${fits.vote.s.toFixed(1)} s`);
  check("the votes view shows how the search went", fits.vote.view.grids >= 3 && fits.vote.view.slices.join() === "9,9,9" && fits.vote.view.moved && fits.vote.view.marks > 0 && fits.vote.view.marksOff === 0,
    `${fits.vote.view.grids} grids, slices ${fits.vote.view.slices.join(" / ")} candidates wide, ${fits.vote.view.marks} candidates some pattern prefers; ${fits.vote.view.readout}`);
  check("the volume shows the ballots that chose, or the mean correlation", fits.vote.what.differ && /ballots lie by the winner/.test(fits.vote.what.said), fits.vote.what.said);
  check("its volume can be gone through slice by slice", fits.vote.volume.slid && fits.vote.volume.scrolled && fits.vote.volume.steps === 5 && fits.vote.volume.every > 9 && fits.vote.volume.shown,
    `${fits.vote.volume.steps} slices along z by the slider and the wheel (${fits.vote.volume.cap.trim()}); every z side by side, ${fits.vote.volume.every} cells across`);
  const off = Math.max(...fits.follow.cross.map((v, i) => Math.abs(v - fits.pose.pc[i])));
  check("the slices follow the pattern centre a fit moves", /cut through the pattern centre/.test(fits.follow.meta) && off < 0.006,
    `after Fit pose: cross-hair ${fits.follow.cross.join(" ")} in grid ${fits.follow.grid}, ${off.toFixed(4)} from the pattern centre`);
  check("Fit pose lands on it", err(fits.pose.pc) < 0.004, `${fits.pose.pc.map((v) => v.toFixed(5)).join(" ")} (${err(fits.pose.pc).toFixed(5)} from kikuchipy's) in ${fits.pose.s.toFixed(2)} s`);

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

  // the pattern view follows the map
  const view = await evaluate(`
    for (let i = 0; i < 100 && !/NCC/.test(document.getElementById("pat-facts").textContent); i++) await new Promise((r) => setTimeout(r, 100));
    return { facts: document.getElementById("pat-facts").textContent, sim: document.getElementById("pat-sim").width, cap: document.getElementById("pat-sim-cap").textContent };`);
  check("the selected pattern shows its simulation", /NCC/.test(view.facts) && view.sim === 60 && /indexed/.test(view.cap), view.cap);

  // a project file dropped on the window brings its settings and geometry back
  const proj = await evaluate(`
    const g = window.ze.geometry;
    const file = new File([JSON.stringify({ zebsdsolve: window.ze.version, geometry: { ...g, pc: [0.41, 0.22, 0.51] }, settings: { ...window.ze.settings, bandwidth: 48, accept_z: 12 } })], "saved.zebsdsolve.json");
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
  check("a project file brings its settings and geometry back", proj.s.bandwidth === 48 && proj.s.accept_z === 12 && Math.abs(proj.pc[0] - 0.41) < 1e-12 && proj.pcShown.join() === "0.41,0.22,0.51",
    `bandwidth ${proj.s.bandwidth}, accept at z ≥ ${proj.s.accept_z}, pattern centre ${proj.pcShown.join(" ")}`);

  const errs = await evaluate(`return [...document.querySelectorAll("#log .err")].map((e) => e.textContent).slice(0, 5);`);
  check("no errors in the log", errs.length === 0, errs.join(" | "));

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
