/**
 * zebsdsolve in the browser: loads zebsdsolve.wasm (Zig) and gives it WebGPU and files.
 *
 * The module schedules every GPU operation and every file read itself; this adapter only carries
 * out its requests (buffers, pipelines, dispatches, copies, submits through the browser's WebGPU
 * API; byte ranges of Blobs) and resolves them asynchronously. No indexing logic lives here.
 *
 *   import { Zebsdsolve } from "./zebsdsolve.js";
 *   const ze = await Zebsdsolve.create({ wasm: "zebsdsolve.wasm" });
 *   await ze.openScan([scanFile]);               // a File of any size: it is read in ranges
 *   await ze.addPhase(masterFile);
 *   await ze.fitPc(); await ze.fitPose();
 *   await ze.index(({ done, total }) => …);
 *   const { quaternions, ncc, phase } = ze.results();
 *
 * The module's blocking code suspends on GPU readbacks and file reads through WebAssembly
 * JavaScript Promise Integration (Chrome / Edge 137+). Calls are serialized; only cancel() may
 * run while one is suspended.
 */

const LIMIT_KEYS = ["maxStorageBufferBindingSize", "maxBufferSize", "maxComputeWorkgroupStorageSize", "maxStorageBuffersPerShaderStage"];
const utf8 = new TextDecoder();
const newStats = () => ({ dispatches: 0, submits: 0, reads: 0, waits: 0, waitMs: 0, compileWaitMs: 0, pipelines: 0, pipelineMs: 0 });
const utf8enc = new TextEncoder();
// ── pipeline warm-up across page loads ──
// Browsers compile a WGSL pipeline the first time a page asks for it, which costs 20–1000 ms
// each. The adapter remembers (in localStorage) which pipelines this origin created, and at
// the next load starts compiling all of them in parallel while the page sets up, so the first
// fit or indexing step finds them ready. The list belongs to one build of zebsdsolve.wasm (its
// fingerprint), so a new build starts a new list. `warm: false` in Zebsdsolve.create turns this off.
// A first visit starts from a list shipped with the site (`pipelines.json` beside the wasm,
// recorded by tests/test_web.mjs --save-pipelines) when it belongs to the same build.
const WARM_KEY = "zebsdsolve.pipelines.v1";
const WARM_MAX = 400;

function hashCode(str) {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) h = Math.imul(h ^ str.charCodeAt(i), 0x01000193);
  return (h >>> 0).toString(36) + "." + str.length.toString(36);
}

function hashBytes(bytes) {
  const u = new Uint8Array(bytes);
  let h = 0x811c9dc5;
  for (let i = 0; i < u.length; i++) h = Math.imul(h ^ u[i], 0x01000193);
  return (h >>> 0).toString(36) + "." + u.length.toString(36);
}

function loadWarm() {
  try { return JSON.parse(localStorage.getItem(WARM_KEY)); } catch { return null; }
}

async function fetchWarm(url) {
  try {
    const r = await fetch(url);
    return r.ok ? await r.json() : null;
  } catch { return null; }
}

function warmRecord(w) {
  const list = w.list.slice(-WARM_MAX);
  const codes = {};
  for (const [h] of list) codes[h] = w.codes.get(h);
  return { fp: w.fp, codes, list };
}

function saveWarm(state) {
  const w = state.warm;
  if (!w || w.timer) return;
  w.timer = setTimeout(() => {
    w.timer = null;
    try { localStorage.setItem(WARM_KEY, JSON.stringify(warmRecord(w))); } catch { /* quota: skip */ }
  }, 1500);
}

function gpuImports(state) {
  const { device } = state;
  const bufs = [null];
  const freeIds = [];
  // pipelines: { pipeline, layout } once compiled (pipeline null while compiling or failed)
  const pipes = [null];
  const modules = new Map();
  // (code hash, entry) → compile promise, shared by the warm-up and createPipeline
  const compiled = new Map();
  let enc = null;
  let pass = null;
  let recorded = [];
  // Pipelines compile asynchronously and in parallel (createComputePipelineAsync). While a
  // pipeline that recorded work uses is still compiling, queue operations are held back in
  // order (`backlog`) and replayed once it is ready, so the module keeps recording ahead and
  // only a wait (a readback) blocks on the compiler.
  let backlog = null;
  let blocking = new Set();
  let drained = null;
  const mem = () => state.memory.buffer;
  const bytes = (ptr, len) => new Uint8Array(mem(), ptr, len);
  const endPass = () => { if (pass) { pass.end(); pass = null; } };
  const encoder = () => (enc ||= device.createCommandEncoder());
  // counters for ze.stats(): what an operation cost in GPU round trips
  const st = state.stats;
  device.addEventListener?.("uncapturederror", (e) => state.errors.push(String(e.error?.message || e.error)));

  const defer = (op) => {
    if (!backlog) {
      backlog = [];
      let resolve;
      drained = { promise: new Promise((r) => (resolve = r)), resolve };
    }
    backlog.push(op);
  };
  const replay = () => {
    const ops = backlog;
    backlog = null;
    for (const op of ops) op();
    drained.resolve();
    drained = null;
  };

  // ── operations, issued directly (or replayed from the backlog) ──
  // indirect: [buffer, byte offset] of the workgroup counts (3 × u32) instead of x, y, z
  const doDispatch = (p, x, y, z, entries, indirect) => {
    if (!p.pipeline) return; // failed to compile: reported through takeError
    if (!pass) pass = encoder().beginComputePass();
    pass.setPipeline(p.pipeline);
    pass.setBindGroup(0, device.createBindGroup({ layout: p.layout, entries }));
    if (indirect) pass.dispatchWorkgroupsIndirect(indirect[0], indirect[1]);
    else pass.dispatchWorkgroups(Math.max(1, x), Math.max(1, y), Math.max(1, z));
  };
  const bindEntries = (bindsPtr, n) => {
    const v = new DataView(mem(), bindsPtr, n * 24);
    const entries = [];
    for (let i = 0; i < n; i++) {
      const o = i * 24;
      const size = Number(v.getBigUint64(o + 16, true));
      const resource = { buffer: bufs[v.getUint32(o + 4, true)], offset: Number(v.getBigUint64(o + 8, true)) };
      if (size) resource.size = size;
      entries.push({ binding: v.getUint32(o, true), resource });
    }
    return entries;
  };
  const doSubmit = () => {
    endPass();
    if (enc) { st.submits++; device.queue.submit([enc.finish()]); enc = null; }
    for (const b of recorded) b.destroy();
    recorded = [];
    for (const r of state.reads) {
      if (r.promise || !r.issued) continue;
      r.promise = r.staging.mapAsync(GPUMapMode.READ).then(() => {
        // wasm memory may have grown since the read was requested: take a fresh view.
        new Uint8Array(mem(), r.dst, r.size).set(new Uint8Array(r.staging.getMappedRange(), 0, r.size));
        r.staging.unmap();
        r.staging.destroy();
      });
    }
  };
  const doDestroy = (b) => { if (enc) recorded.push(b); else b.destroy(); }; // never under an open batch

  const moduleOf = (code) => {
    let module = modules.get(code);
    if (!module) { module = device.createShaderModule({ code }); modules.set(code, module); }
    return module;
  };
  // Compile progress for onCompile listeners: counts restart once everything started has finished.
  const cp = state.compiling;
  const told = () => { for (const fn of cp.listeners) try { fn({ done: cp.done, total: cp.total }); } catch (e) { console.error(e); } };
  const compile = (code, h, entryPoint) => {
    const key = h + "/" + entryPoint;
    let c = compiled.get(key);
    if (!c) {
      if (cp.done === cp.total) cp.done = cp.total = 0;
      cp.total++;
      told();
      c = device.createComputePipelineAsync({ label: entryPoint, layout: "auto", compute: { module: moduleOf(code), entryPoint } });
      c.finally(() => { cp.done++; told(); }).catch(() => {});
      compiled.set(key, c);
    }
    return c;
  };
  // Start compiling what this origin used before. Entries from an older build may no longer
  // compile: their errors stay inside this error scope (not reported against a later call) and
  // they leave the list.
  const w = state.warm;
  if (w?.saved) {
    w.list = w.saved.list.slice();
    device.pushErrorScope("validation");
    for (const [h, entry] of w.saved.list) {
      const code = w.saved.codes[h];
      if (!code) continue;
      w.codes.set(h, code);
      compile(code, h, entry).catch(() => {
        const i = w.list.findIndex(([a, b]) => a === h && b === entry);
        if (i >= 0) w.list.splice(i, 1);
        saveWarm(state);
      });
    }
    device.popErrorScope().catch(() => {});
    w.saved = null;
  }

  return {
    limits(out) {
      const v = new DataView(mem(), out, 40);
      const L = device.limits;
      v.setBigUint64(0, BigInt(L.maxStorageBufferBindingSize), true);
      v.setBigUint64(8, BigInt(L.maxBufferSize), true);
      v.setUint32(16, L.maxComputeWorkgroupStorageSize, true);
      v.setUint32(20, L.maxStorageBuffersPerShaderStage, true);
      v.setUint32(24, L.minUniformBufferOffsetAlignment, true);
      v.setUint32(28, L.minStorageBufferOffsetAlignment, true);
      v.setUint32(32, device.features.has("shader-f16") ? 1 : 0, true);
      v.setUint32(36, navigator.gpu?.wgslLanguageFeatures?.has?.("packed_4x8_integer_dot_product") ? 1 : 0, true);
    },
    info(ptr, cap) {
      const b = utf8enc.encode(state.info).subarray(0, cap);
      bytes(ptr, b.length).set(b);
      return b.length;
    },
    createBuffer(size, usage) {
      const u = usage === 1
        ? GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        : GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT;
      const b = device.createBuffer({ size, usage: u });
      const id = freeIds.length ? freeIds.pop() : bufs.length;
      bufs[id] = b;
      return id;
    },
    destroyBuffer(id) {
      const b = bufs[id];
      if (!b) return;
      bufs[id] = null;
      freeIds.push(id);
      if (backlog) defer(() => doDestroy(b)); else doDestroy(b);
    },
    writeBuffer(id, offset, ptr, len) {
      const b = bufs[id];
      if (backlog) { const data = bytes(ptr, len).slice(); defer(() => device.queue.writeBuffer(b, offset, data)); }
      else device.queue.writeBuffer(b, offset, mem(), ptr, len);
    },
    // Starts compiling and returns at once; the first dispatch that needs it holds the queue
    // until it is ready.
    createPipeline(codePtr, codeLen, entryPtr, entryLen) {
      const t0 = performance.now();
      st.pipelines++;
      const code = utf8.decode(bytes(codePtr, codeLen));
      const entryPoint = utf8.decode(bytes(entryPtr, entryLen));
      const h = hashCode(code);
      if (w) {
        w.codes.set(h, code);
        const i = w.list.findIndex(([a, b]) => a === h && b === entryPoint);
        if (i >= 0) w.list.splice(i, 1);
        w.list.push([h, entryPoint]); // most recent last
        saveWarm(state);
      }
      const p = { pipeline: null, layout: null, ready: false };
      p.promise = compile(code, h, entryPoint)
        .then((pl) => { p.pipeline = pl; p.layout = pl.getBindGroupLayout(0); })
        .catch((e) => state.errors.push(`pipeline ${entryPoint}: ${e.message || e}`))
        .finally(() => {
          p.ready = true;
          if (blocking.delete(p) && backlog && blocking.size === 0) replay();
        });
      pipes.push(p);
      st.pipelineMs += performance.now() - t0;
      return pipes.length - 1;
    },
    dispatch(pipe, x, y, z, bindsPtr, n) {
      st.dispatches++;
      const p = pipes[pipe];
      const entries = bindEntries(bindsPtr, n);
      if (!p.ready) blocking.add(p);
      if (backlog || !p.ready) defer(() => doDispatch(p, x, y, z, entries));
      else doDispatch(p, x, y, z, entries);
    },
    dispatchIndirect(pipe, buf, offset, bindsPtr, n) {
      st.dispatches++;
      const p = pipes[pipe];
      const entries = bindEntries(bindsPtr, n);
      const ind = [bufs[buf], offset];
      if (!p.ready) blocking.add(p);
      if (backlog || !p.ready) defer(() => doDispatch(p, 0, 0, 0, entries, ind));
      else doDispatch(p, 0, 0, 0, entries, ind);
    },
    clearBuffer(id, offset, size) {
      const b = bufs[id];
      const op = () => { endPass(); encoder().clearBuffer(b, offset, size); };
      if (backlog) defer(op); else op();
    },
    copyBuffer(src, soff, dst, doff, size) {
      const a = bufs[src];
      const b = bufs[dst];
      const op = () => { endPass(); encoder().copyBufferToBuffer(a, soff, b, doff, size); };
      if (backlog) defer(op); else op();
    },
    readBuffer(src, offset, size, dst) {
      st.reads++;
      const a = bufs[src];
      const padded = Math.ceil(size / 4) * 4;
      const staging = device.createBuffer({ size: padded, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const r = { staging, size, dst, promise: null, issued: false };
      state.reads.push(r);
      const op = () => { endPass(); encoder().copyBufferToBuffer(a, offset, staging, 0, padded); r.issued = true; };
      if (backlog) defer(op); else op();
    },
    submit() {
      if (backlog) defer(doSubmit); else doSubmit();
    },
    takeError(ptr, cap) {
      if (!state.errors.length) return 0;
      const b = utf8enc.encode(state.errors.shift()).subarray(0, cap);
      bytes(ptr, b.length).set(b);
      return b.length;
    },
    // Suspending import (JS Promise Integration): the module's stack waits here while pipelines
    // still compiling and the submitted reads resolve, then resumes as if the call had blocked.
    wait: new WebAssembly.Suspending(async () => {
      const t0 = performance.now();
      st.waits++;
      try {
        if (drained) { await drained.promise; st.compileWaitMs += performance.now() - t0; }
        const reads = state.reads;
        state.reads = [];
        await Promise.all(reads.map((r) => r.promise));
        return 0;
      } catch (e) { state.errors.push(String(e)); return 1; } finally { st.waitMs += performance.now() - t0; }
    }),
  };
}

// ── files ──
// The module reads scans and master patterns itself, in ranges (src/io/file.zig). Here a file is
// a Blob (a File the user picked, or a fetched example): `open` resolves the token `addFile`
// gave it, `read` slices it. `read` is a suspending import: the module's stack waits while the
// slice loads, so a scan far larger than memory is read in place. `readv` loads many slices in
// one suspension: the module asks for scattered patterns (a preview lattice, the chunks of an
// HDF5 dataset) together, because each suspension costs about as much as half a megabyte read.
function ioImports(state) {
  const files = state.files;
  // Load ranges of a file into wasm memory back to back; 0 on success. Slices load in
  // parallel, a large range as several pieces (Chrome reads four pieces of a range together
  // about three times as fast as the range whole), a few hundred at a time.
  const PIECE = 4 << 20;
  const load = async (id, off, len, dst) => {
    try {
      const t0 = performance.now();
      const file = files[id];
      const pieces = [];
      for (let i = 0; i < off.length; i++) {
        for (let a = 0; a < len[i]; a += PIECE) pieces.push([off[i] + a, Math.min(PIECE, len[i] - a)]);
      }
      let at = dst;
      for (let i0 = 0; i0 < pieces.length; i0 += 512) {
        const bufs = await Promise.all(pieces.slice(i0, i0 + 512).map(([o, n]) => file.slice(o, o + n).arrayBuffer()));
        // wasm memory may have grown while the slices loaded: take a fresh view
        const mem = new Uint8Array(state.memory.buffer);
        for (let k = 0; k < bufs.length; k++) {
          if (bufs[k].byteLength !== pieces[i0 + k][1]) return 1;
          mem.set(new Uint8Array(bufs[k]), at);
          at += bufs[k].byteLength;
        }
      }
      state.io.reads++;
      state.io.slices += pieces.length;
      state.io.bytes += at - dst;
      state.io.ms += performance.now() - t0;
      return 0;
    } catch (e) { state.errors.push(`file read: ${e.message || e}`); return 1; }
  };
  return {
    open(ptr, len) {
      const name = utf8.decode(new Uint8Array(state.memory.buffer, ptr, len));
      const m = /^@(\d+):/.exec(name);
      const id = m ? +m[1] : 0;
      return files[id] ? id : 0;
    },
    size(id) { return files[id] ? files[id].size : 0; },
    read: new WebAssembly.Suspending((id, offset, dst, len) => load(id, [offset], [len], dst)),
    readv: new WebAssembly.Suspending((id, n, offs, lens, dst) =>
      // the ranges, copied out: wasm memory may grow while the slices load
      load(id, new Float64Array(state.memory.buffer, offs, n).slice(), new Uint32Array(state.memory.buffer, lens, n).slice(), dst)),
    // the engine closes a file when it is done with it; the Blob stays for a later open
    close() {},
  };
}

// Exports that never wait (plain calls); every other function export may suspend (on the GPU
// or on a file read) and is called through WebAssembly.promising.
const SYNC_EXPORTS = new Set(["ze_alloc", "ze_free", "ze_version", "ze_create", "ze_destroy", "ze_adapter", "ze_error",
  "ze_configure", "ze_settings", "ze_events", "ze_cancel", "ze_close_scan", "ze_scan_info", "ze_scan_format",
  "ze_get_geometry", "ze_set_geometry", "ze_clear_phases", "ze_phase_count", "ze_phase_info", "ze_pattern_shape", "ze_nlpar_view",
  "ze_pc_scores", "ze_fit_patterns", "ze_results", "ze_map", "ze_ipf_key", "ze_export", "ze_reduce", "ze_disorientation", "ze_shader",
  "ze_index_status", "ze_tune_begin", "ze_tune_report", "ze_pixel_angle", "ze_lattice", "ze_sweep", "ze_pc_level_count", "ze_pc_level", "ze_pc_level_ballots",
  "ze_gpu_counts", "ze_profile", "ze_profile_report", "ze_compile_wgsl"]);

// Struct layouts (src/engine.zig, geometry.zig, pcfit.zig, posefit.zig): [name, type, byte offset].
const SCAN = [["rows", "u32", 0], ["cols", "u32", 4], ["patH", "u32", 8], ["patW", "u32", 12], ["pixel", "u32", 16],
  ["hasBackground", "bool", 20], ["hasPc", "bool", 24], ["formatId", "u32", 28], ["voltageKv", "f64", 32], ["bytes", "f64", 40]];
const SCAN_SIZE = 48;
const PHASE = [["side", "u32", 0], ["pointGroup", "u32", 4], ["spaceGroup", "u32", 8], ["laueId", "u32", 12],
  ["nFold", "u32", 16], ["mirror", "bool", 20], ["inversion", "bool", 24], ["voltageKv", "f64", 32]];
const PHASE_SIZE = 104;
const GEOMETRY = [["pc", "f64x3", 0], ["sampleTiltDeg", "f64", 24], ["detectorTiltDeg", "f64", 32], ["rot", "f64x3", 40],
  ["pxUm", "f64", 64], ["stepXUm", "f64", 72], ["stepYUm", "f64", 80]];
const GEOMETRY_SIZE = 88;
const PC_RESULT = [["pc", "f64x3", 0], ["start", "f64x3", 24], ["score", "f64", 48], ["startScore", "f64", 56],
  ["nPatterns", "u32", 64], ["nCandidates", "u32", 68], ["levels", "u32", 72], ["atBorder", "bool", 76]];
const PC_RESULT_SIZE = 80;
const POSE_RESULT = [["pc", "f64x3", 0], ["rot", "f64x3", 24], ["startPc", "f64x3", 48], ["startRot", "f64x3", 72],
  ["ncc", "f64", 96], ["startNcc", "f64", 104], ["meanTurn", "f64", 112],
  ["iterations", "u32", 120], ["accepted", "u32", 124], ["nPatterns", "u32", 128], ["converged", "bool", 132], ["pxUm", "f64", 136], ["startPxUm", "f64", 144]];
const POSE_RESULT_SIZE = 152;
const INDEX_STATUS = [["sweep", "u32", 0], ["sweeps", "u32", 4], ["bandwidth", "u32", 8], ["dictGrid", "u32", 12],
  ["done", "u64", 16], ["total", "u64", 24], ["region", "u64", 32], ["accepted", "u64", 40], ["methodId", "u32", 48]];
const INDEX_STATUS_SIZE = 56;
export const METHODS = ["spherical", "dictionary", "pca"];

export const LAUE = [null, "-1", "2/m", "mmm", "4/m", "4/mmm", "-3", "-3m", "6/m", "6/mmm", "m-3", "m-3m"];
export const POINT_GROUPS = [null, "1", "-1", "2", "m", "2/m", "222", "mm2", "mmm", "4", "-4", "4/m", "422", "4mm", "-42m", "4/mmm",
  "3", "-3", "32", "3m", "-3m", "6", "-6", "6/m", "622", "6mm", "-6m2", "6/mmm", "23", "m-3", "432", "-43m", "m-3m"];
export const MAPS = { ipf_x: 0, ipf_y: 1, ipf_z: 2, ncc: 3, nmi: 4, phase: 5, z: 6, sweep: 7 };

export class Zebsdsolve {
  /** Load the module and open a high-performance device with the limits the engine needs. */
  static async create({ wasm = new URL("../zig-out/web/zebsdsolve.wasm", import.meta.url), device = null, warm = true, onCompile = null } = {}) {
    let info = "";
    if (!device) {
      const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
      if (!adapter) throw new Error("WebGPU is not available");
      const requiredLimits = {};
      for (const k of LIMIT_KEYS) requiredLimits[k] = adapter.limits[k];
      const requiredFeatures = adapter.features.has("shader-f16") ? ["shader-f16"] : [];
      device = await adapter.requestDevice({ requiredLimits, requiredFeatures });
      const ai = adapter.info || {};
      info = `${ai.vendor || "gpu"} ${ai.architecture || ai.device || ""}`.trim();
    }
    const compiling = { done: 0, total: 0, listeners: onCompile ? [onCompile] : [] };
    const state = { device, memory: null, reads: [], errors: [], info, listeners: [], stats: newStats(), compiling,
      files: [null], io: { reads: 0, slices: 0, bytes: 0, ms: 0 } };
    const warmOn = warm && typeof localStorage !== "undefined";
    // This origin's own list when it belongs to this build; else the shipped one, downloaded
    // alongside the wasm on a first visit (or after the wasm changed).
    const own = warmOn ? loadWarm() : null;
    const shippedUrl = new URL("pipelines.json", new URL(wasm, location.href));
    const shipped = warmOn && !own ? fetchWarm(shippedUrl) : null;
    const bytes = await (await fetch(wasm)).arrayBuffer();
    if (warmOn) {
      const fp = hashBytes(bytes);
      let saved = own?.fp === fp ? own : null;
      if (!saved) {
        const ship = await (shipped || fetchWarm(shippedUrl));
        saved = ship?.fp === fp ? ship : null;
      }
      state.warm = { fp, saved, codes: new Map(), list: [], timer: null };
    }
    const env = {
      ze_log(ptr, len) { console.warn("zebsdsolve:", utf8.decode(new Uint8Array(state.memory.buffer, ptr, len))); },
      // progress events from long operations (fits, indexing): one JSON object each
      ze_event(ptr, len) {
        const ev = JSON.parse(utf8.decode(new Uint8Array(state.memory.buffer, ptr, len)));
        for (const fn of state.listeners) try { fn(ev); } catch (e) { console.error(e); }
      },
    };
    if (typeof WebAssembly.Suspending !== "function") throw new Error("this browser lacks WebAssembly JS Promise Integration (Chrome / Edge 137+)");
    const { instance } = await WebAssembly.instantiate(bytes, { zegpu: gpuImports(state), zeio: ioImports(state), env });
    state.memory = instance.exports.memory;
    const x = {};
    for (const [k, v] of Object.entries(instance.exports)) {
      x[k] = typeof v === "function" && !SYNC_EXPORTS.has(k) ? WebAssembly.promising(v) : v;
    }
    return new Zebsdsolve(x, state);
  }

  constructor(x, state) {
    this.x = x;
    this.state = state;
    this._queue = Promise.resolve();
    this.e = x.ze_create();
    if (!this.e) throw new Error("zebsdsolve: engine creation failed");
  }

  get device() { return this.state.device; }
  get mem() { return this.state.memory.buffer; }

  get version() {
    const p = this.x.ze_version();
    const u = new Uint8Array(this.mem, p, 32);
    return utf8.decode(u.subarray(0, u.indexOf(0)));
  }

  /** Call fn({ done, total }) as pipelines compile (total counts the current burst). */
  onCompile(fn) {
    this.state.compiling.listeners.push(fn);
    fn({ done: this.state.compiling.done, total: this.state.compiling.total });
  }

  /** The pipelines this page created, as the warm-up list (pipelines.json) records them. */
  warmList() { return this.state.warm ? warmRecord(this.state.warm) : null; }

  /** GPU and file traffic since the last call, then reset (fileReads: round trips; fileSlices: ranges read). */
  stats() {
    const s = { ...this.state.stats, fileReads: this.state.io.reads, fileSlices: this.state.io.slices, fileBytes: this.state.io.bytes, fileMs: this.state.io.ms };
    Object.assign(this.state.stats, newStats());
    this.state.io = { reads: 0, slices: 0, bytes: 0, ms: 0 };
    return s;
  }

  /** Listen to progress events ({kind: "log" | "progress" | "pc" | "pose", …}); returns an unsubscribe. */
  onEvent(fn) {
    this.state.listeners.push(fn);
    return () => { this.state.listeners = this.state.listeners.filter((f) => f !== fn); };
  }

  // ── plumbing ──

  /** Temporary wasm allocations for one call, freed afterwards (each at least 8 bytes, so 8-aligned). */
  _frame(fn) {
    const held = [];
    const f = {
      alloc: (n) => { const p = this.x.ze_alloc(Math.max(n, 8)); if (!p) throw new Error("zebsdsolve: out of memory"); held.push([p, Math.max(n, 8)]); return p; },
      f32: (arr) => { const p = f.alloc(arr.length * 4); new Float32Array(this.mem, p, arr.length).set(arr); return p; },
      u64: (arr) => { const p = f.alloc(arr.length * 8); const v = new BigUint64Array(this.mem, p, arr.length); for (let i = 0; i < arr.length; i++) v[i] = BigInt(arr[i]); return p; },
      str: (s) => { const b = utf8enc.encode(s); const p = f.alloc(b.length); new Uint8Array(this.mem, p, b.length).set(b); return [p, b.length]; },
    };
    const done = () => { for (const [p, n] of held) this.x.ze_free(p, n); };
    let out;
    try { out = fn(f); } catch (e) { done(); throw e; }
    if (out instanceof Promise) return out.finally(done);
    done();
    return out;
  }

  _read(layout, p) {
    const v = new DataView(this.mem);
    const o = {};
    for (const [k, t, off] of layout) {
      const a = p + off;
      o[k] = t === "u32" ? v.getUint32(a, true) : t === "bool" ? !!v.getUint32(a, true) : t === "u64" ? Number(v.getBigUint64(a, true))
        : t === "f64x3" ? [v.getFloat64(a, true), v.getFloat64(a + 8, true), v.getFloat64(a + 16, true)] : v.getFloat64(a, true);
    }
    return o;
  }

  _error() {
    return this._frame((f) => {
      const p = f.alloc(1024);
      const n = this.x.ze_error(this.e, p, 1024);
      return utf8.decode(new Uint8Array(this.mem, p, n));
    });
  }

  _check(rc, what) {
    const err = this._error();
    if (Number(rc) < 0 || err) throw new Error(`${what}: ${err || rc}`);
    return Number(rc);
  }

  /** Calls are serialized: one operation on the module at a time (only cancel() may interleave). */
  _serial(fn) {
    const run = this._queue.then(fn, fn);
    this._queue = run.catch(() => {});
    return run;
  }

  _text(call, cap = 256) {
    return this._frame((f) => {
      const p = f.alloc(cap);
      return utf8.decode(new Uint8Array(this.mem, p, call(p, cap)));
    });
  }

  get adapter() { return this._text((p, cap) => this.x.ze_adapter(this.e, p, cap)); }

  /** Ask the running fit or indexing to stop at its next step (it keeps what it has). */
  cancel() { this.x.ze_cancel(this.e); }

  // ── settings and geometry ──

  /** Apply settings overrides (see src/settings.zig for the keys). */
  configure(obj) {
    return this._frame((f) => {
      const [p, n] = f.str(JSON.stringify(obj));
      this._check(this.x.ze_configure(this.e, p, n), "configure");
    });
  }

  get settings() {
    return this._frame((f) => {
      let cap = 8192, p = f.alloc(cap), n = this.x.ze_settings(this.e, p, cap);
      if (n > cap) { cap = n; p = f.alloc(cap); n = this.x.ze_settings(this.e, p, cap); }
      return JSON.parse(utf8.decode(new Uint8Array(this.mem, p, n)));
    });
  }

  /** The detector geometry: { pc[3], sampleTiltDeg, detectorTiltDeg, rot[3], pxUm, stepXUm, stepYUm }. */
  get geometry() {
    return this._frame((f) => {
      const p = f.alloc(GEOMETRY_SIZE);
      this.x.ze_get_geometry(this.e, p);
      return this._read(GEOMETRY, p);
    });
  }

  /** Set any of the geometry's fields. */
  setGeometry(fields) {
    this._frame((f) => {
      const p = f.alloc(GEOMETRY_SIZE);
      this.x.ze_get_geometry(this.e, p);
      const v = new DataView(this.mem);
      for (const [k, t, off] of GEOMETRY) {
        if (!(k in fields)) continue;
        if (t === "f64x3") for (let i = 0; i < 3; i++) v.setFloat64(p + off + 8 * i, +fields[k][i], true);
        else v.setFloat64(p + off, +fields[k], true);
      }
      this.x.ze_set_geometry(this.e, p);
    });
  }

  // ── scan ──

  /** Register a Blob (a picked File, or fetched bytes) the module may open; returns its token. */
  addFile(blob, name = blob.name || "file") {
    this.state.files.push(blob);
    return `@${this.state.files.length - 1}:${name}`;
  }

  /** Open a scan from Blobs: the pattern file first, then side files (NORDIF's Setting.txt). */
  openScan(blobs) {
    return this._serial(() => this._frame(async (f) => {
      const tokens = [].concat(blobs).map((b) => this.addFile(b));
      const [p, n] = f.str(tokens.join("\n"));
      this._check(await this.x.ze_open_scan(this.e, p, n), "open");
      return this.scan;
    }));
  }

  /** Make a toy scan from the phases loaded (settings toy_*), in place of any scan open: grains
   *  with random orientations and phases, simulated under the present geometry (its true one),
   *  laid on a background and counted as electrons. Resolves to its answer: { quaternions (4 per
   *  cell), phase (−1: a blank cell), grain, geometry }. */
  synthesize() {
    return this._serial(() => this._frame(async (f) => {
      const s = this.settings, n = s.toy_rows * s.toy_cols;
      const qp = f.alloc(16 * n), pp = f.alloc(4 * n), gp = f.alloc(4 * n);
      const geometry = this.geometry;
      this._check(await this.x.ze_synthesize(this.e, qp, pp, gp, n), "synthesize");
      return { quaternions: new Float32Array(this.mem, qp, 4 * n).slice(), phase: new Int32Array(this.mem, pp, n).slice(), grain: new Int32Array(this.mem, gp, n).slice(), geometry };
    }));
  }

  /** Take rows · cols patterns of h × w from a Uint8Array, Uint16Array or Float32Array (row-major). */
  setPatterns(data, rows, cols, h, w) {
    return this._serial(() => this._frame(async (f) => {
      const pixel = data instanceof Uint8Array ? 0 : data instanceof Uint16Array ? 1 : 2;
      const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
      const p = f.alloc(bytes.length);
      new Uint8Array(this.mem, p, bytes.length).set(bytes);
      this._check(await this.x.ze_set_patterns(this.e, p, bytes.length, pixel, rows, cols, h, w), "set_patterns");
    }));
  }

  closeScan() { this.x.ze_close_scan(this.e); }

  /** What the scan is: { rows, cols, patH, patW, pixel, format, hasBackground, hasPc, voltageKv, bytes }. */
  get scan() {
    const s = this._frame((f) => {
      const p = f.alloc(SCAN_SIZE);
      this.x.ze_scan_info(this.e, p);
      return this._read(SCAN, p);
    });
    s.format = this._text((p, cap) => this.x.ze_scan_format(this.e, p, cap), 64);
    s.dtype = ["uint8", "uint16", "float32"][s.pixel];
    return s;
  }

  /** [height, width] of a pattern: reduced by the binning setting, or (full) as stored. */
  shape(full = false) {
    return this._frame((f) => {
      const p = f.alloc(8);
      this._check(this.x.ze_pattern_shape(this.e, full ? 1 : 0, p), "pattern_shape");
      return Array.from(new Uint32Array(this.mem, p, 2));
    });
  }

  /** [height, width] of a pattern as the search sees it (reduced by the binning setting). */
  get patternShape() { return this.shape(false); }

  /** One pattern (map cell, row-major) as Float32Array(h·w), at the reduced size or (full) the
   *  stored one. `stage`: "filtered" (or true), as it is indexed: averaged with its neighbours
   *  when the nlpar setting is on, then filtered; "stored" (or false), as it is stored;
   *  "averaged", averaged with its neighbours and not filtered. */
  pattern(cell, stage = true, full = false) {
    return this._serial(() => this._frame(async (f) => {
      const [h, w] = this.shape(full);
      const p = f.alloc(h * w * 4);
      const k = stage === "averaged" ? 2 : stage === "stored" || !stage ? 0 : 1;
      this._check(await this.x.ze_pattern(this.e, BigInt(cell), k, full ? 1 : 0, p, h * w), "pattern");
      const out = new Float32Array(this.mem, p, h * w).slice();
      // what the averaging did to it, kept with it (another pattern() replaces the engine's)
      out.nlpar = this._nlparView();
      return out;
    }));
  }

  /** What averaging with its neighbours did to the pattern last read: { noise (as stored,
   *  standard deviation a pixel), left (the share of it the average keeps), own (the share of
   *  the average that is the pattern itself), side, shares (side², the pattern in the middle,
   *  NaN outside the map) }, or null when it was not averaged. */
  _nlparView() {
    return this._frame((f) => {
      const op = f.alloc(12);
      const side = this.x.ze_nlpar_view(this.e, op, 0, 0);
      if (!side) return null;
      const sp = f.alloc(side * side * 4);
      this.x.ze_nlpar_view(this.e, op, sp, side * side);
      const o = new Float32Array(this.mem, op, 3);
      return { noise: o[0], left: o[1], own: o[2], side, shares: new Float32Array(this.mem, sp, side * side).slice() };
    });
  }

  /** About n map cells on a lattice over the whole map (every cell when it has no more). */
  lattice(n) {
    return this._frame((f) => {
      const count = this._check(this.x.ze_lattice(this.e, n, 0, 0), "lattice");
      const p = f.alloc(count * 8);
      this._check(this.x.ze_lattice(this.e, n, p, count), "lattice");
      return Array.from(new BigUint64Array(this.mem, p, count), Number);
    });
  }

  /** Angular width of one stored detector pixel as the master filters take it, degrees. */
  get pixelAngle() { return this.x.ze_pixel_angle(this.e); }

  /** Mean and standard deviation of the stored patterns of the given cells: { mean, std }. */
  patternStats(cells) {
    return this._serial(() => this._frame(async (f) => {
      const n = cells.length;
      const out = f.alloc(n * 8);
      this._check(await this.x.ze_pattern_stats(this.e, f.u64(cells), n, out, 2 * n), "pattern_stats");
      const v = new Float32Array(this.mem, out, 2 * n);
      const mean = new Float32Array(n), std = new Float32Array(n);
      for (let i = 0; i < n; i++) { mean[i] = v[2 * i]; std[i] = v[2 * i + 1]; }
      return { mean, std };
    }));
  }

  // ── phases ──

  /** Add a candidate phase from a master pattern Blob (ebsdsim .npz or EMsoft .h5); returns its index. */
  addPhase(blob) {
    return this._serial(() => this._frame(async (f) => {
      const [p, n] = f.str(this.addFile(blob));
      return this._check(await this.x.ze_add_phase(this.e, p, n), "add_phase");
    }));
  }

  clearPhases() { this.x.ze_clear_phases(this.e); }

  /** The candidate phases: [{ name, side, pointGroup, spaceGroup, laue, nFold, mirror, inversion, voltageKv }]. */
  get phases() {
    return this._frame((f) => {
      const out = [];
      const p = f.alloc(PHASE_SIZE);
      const n = this.x.ze_phase_count(this.e);
      for (let i = 0; i < n; i++) {
        this._check(this.x.ze_phase_info(this.e, i, p), "phase_info");
        const o = this._read(PHASE, p);
        const name = new Uint8Array(this.mem, p + 40, 64);
        // the token's "@id:" prefix is the adapter's, not the file's name
        o.name = utf8.decode(name.subarray(0, name.indexOf(0))).replace(/^@\d+:/, "");
        o.laue = LAUE[o.laueId] || "?";
        o.pointGroupName = POINT_GROUPS[o.pointGroup] || "?";
        out.push(o);
      }
      return out;
    });
  }

  /** The phase's master as the search compares patterns with it (smoothed, and filtered as the
   *  patterns are): { side, data: Float32Array(2·side²) }, north then south. */
  masterImage(phase = 0) {
    return this._serial(() => this._frame(async (f) => {
      const side = this.phases[phase].side;
      const n = 2 * side * side;
      const p = f.alloc(n * 4);
      this._check(await this.x.ze_master_image(this.e, phase, p, n), "master_image");
      return { side, data: new Float32Array(this.mem, p, n).slice() };
    }));
  }

  /** Simulated patterns of a phase: cells[k] with orientation quats[4k..4k+4] (w, x, y, z), at
   *  the reduced size or (full) the stored one; Float32Array(n·h·w). */
  simulate(phase, cells, quats, full = false) {
    return this._serial(() => this._frame(async (f) => {
      const [h, w] = this.shape(full);
      const n = cells.length;
      const p = f.alloc(n * h * w * 4);
      this._check(await this.x.ze_simulate(this.e, phase, f.u64(cells), f.f32(quats), n, full ? 1 : 0, p, n * h * w), "simulate");
      return new Float32Array(this.mem, p, n * h * w).slice();
    }));
  }

  // ── fits ──

  /** Fit the pattern centre by the voting search; the geometry takes the result. */
  fitPc() {
    return this._serial(() => this._frame(async (f) => {
      const p = f.alloc(PC_RESULT_SIZE);
      this._check(await this.x.ze_fit_pc(this.e, p), "fit_pc");
      return this._read(PC_RESULT, p);
    }));
  }

  /** The last search level's votes: { shape: [nz, ny, nx], x, y, z, scores } (scores z-major). */
  pcScores() {
    return this._frame((f) => {
      const sp = f.alloc(12);
      const n = this.x.ze_pc_scores(this.e, sp, 0, 0, 0, 0);
      const [nz, ny, nx] = new Uint32Array(this.mem, sp, 3);
      if (n <= 0) return null;
      const ap = f.alloc((nx + ny + nz) * 8), cp = f.alloc(n * 4);
      this.x.ze_pc_scores(this.e, sp, ap, nx + ny + nz, cp, n);
      const axes = new Float64Array(this.mem, ap, nx + ny + nz);
      return { shape: [nz, ny, nx], x: Array.from(axes.subarray(0, nx)), y: Array.from(axes.subarray(nx, nx + ny)), z: Array.from(axes.subarray(nx + ny)),
        scores: new Float32Array(this.mem, cp, n).slice() };
    });
  }

  /** Every grid the last pattern-centre search scored, in order: [{ shape: [nz, ny, nx], x, y,
   *  z, scores (the fit patterns' mean score per candidate, z-major), winner (index into scores:
   *  the candidate nearest the grid's choice), walked (the grid then moved onto a winner at its
   *  border), patBest (per fit pattern, the candidate it scored highest under), patNcc, and,
   *  when ballots decided (the pc_vote setting), density (of the ballots, per candidate), peak
   *  (the choice in candidate steps along x, y, z), ballots and inliers (counts) }]. */
  pcLevels() {
    return this._frame((f) => {
      const out = [];
      const count = this.x.ze_pc_level_count(this.e);
      const nPat = Math.max(1, this.x.ze_fit_patterns(this.e, 0, 0, 0, 0, 0));
      const sp = f.alloc(12), ip = f.alloc(8);
      for (let i = 0; i < count; i++) {
        const n = this.x.ze_pc_level(this.e, i, sp, ip, 0, 0, 0, 0, 0, 0, 0);
        const [nz, ny, nx] = new Uint32Array(this.mem, sp, 3);
        const ap = f.alloc((nx + ny + nz) * 8), cp = f.alloc(n * 4), bp = f.alloc(nPat * 4), np = f.alloc(nPat * 4);
        this.x.ze_pc_level(this.e, i, sp, ip, ap, nx + ny + nz, cp, n, bp, np, nPat);
        const axes = new Float64Array(this.mem, ap, nx + ny + nz);
        const info = new Uint32Array(this.mem, ip, 2);
        const fp = f.alloc(40), dn = this.x.ze_pc_level_ballots(this.e, i, 0, 0, fp);
        const dp = f.alloc(Math.max(dn, 1) * 4);
        this.x.ze_pc_level_ballots(this.e, i, dp, dn, fp);
        const bal = new Float64Array(this.mem, fp, 5);
        out.push({ shape: [nz, ny, nx], x: Array.from(axes.subarray(0, nx)), y: Array.from(axes.subarray(nx, nx + ny)), z: Array.from(axes.subarray(nx + ny)),
          density: dn > 0 ? new Float32Array(this.mem, dp, dn).slice() : null, peak: [bal[0], bal[1], bal[2]], ballots: bal[3], inliers: bal[4],
          scores: new Float32Array(this.mem, cp, n).slice(), winner: info[0], walked: !!info[1],
          patBest: Array.from(new Uint32Array(this.mem, bp, nPat)), patNcc: new Float32Array(this.mem, np, nPat).slice() });
      }
      return out;
    });
  }

  /** Fit the detector pose and the fit patterns' orientations together; the geometry takes the result. */
  fitPose() {
    return this._serial(() => this._frame(async (f) => {
      const p = f.alloc(POSE_RESULT_SIZE);
      this._check(await this.x.ze_fit_pose(this.e, p), "fit_pose");
      return this._read(POSE_RESULT, p);
    }));
  }

  /** The patterns of the last fit: { cells, phases, quats (4 per pattern), ncc }. */
  fitPatterns() {
    return this._frame((f) => {
      const n = this.x.ze_fit_patterns(this.e, 0, 0, 0, 0, 0);
      if (n <= 0) return { cells: [], phases: new Int32Array(0), quats: new Float32Array(0), ncc: new Float32Array(0) };
      const cp = f.alloc(n * 8), pp = f.alloc(n * 4), qp = f.alloc(n * 16), sp = f.alloc(n * 4);
      this.x.ze_fit_patterns(this.e, cp, pp, qp, sp, n);
      return { cells: Array.from(new BigUint64Array(this.mem, cp, n), Number), phases: new Int32Array(this.mem, pp, n).slice(),
        quats: new Float32Array(this.mem, qp, 4 * n).slice(), ncc: new Float32Array(this.mem, sp, n).slice() };
    });
  }

  // ── indexing ──

  /** What normalized mutual information chance gives on this scan for a phase: { median,
   *  spread }, measured once on a sample of the region's patterns against simulations at
   *  random orientations, the same for every pattern. z in the results is the NMI on that
   *  scale; the accept_nmi setting, above 0, is an absolute threshold in its place. */
  chance(phase = 0) {
    return this._serial(() => this._frame(async (f) => {
      const p = f.alloc(8);
      this._check(await this.x.ze_chance(this.e, phase, p), "chance");
      const v = new Float32Array(this.mem, p, 2);
      return { median: v[0], spread: v[1] };
    }));
  }

  /** Start indexing the region of interest (settings roi_*; the whole map by default). */
  indexBegin() {
    return this._serial(async () => { this._check(await this.x.ze_index_begin(this.e), "index"); });
  }

  /** Where indexing stands: { sweep, sweeps, bandwidth, dictGrid, done, total (patterns of this
   *  sweep), region (cells of the region), accepted }. */
  get indexStatus() {
    return this._frame((f) => {
      const p = f.alloc(INDEX_STATUS_SIZE);
      this.x.ze_index_status(this.e, p);
      const s = this._read(INDEX_STATUS, p);
      s.method = METHODS[s.methodId];
      return s;
    });
  }

  /** What each sweep searches with, in order: [{ method, bandwidth, dictGrid }]. */
  get sweepPlan() {
    return this._frame((f) => {
      const p = f.alloc(12);
      const out = [];
      for (let i = 0; i < this.settings.sweeps; i++) {
        this.x.ze_sweep(this.e, i, p);
        const v = new Uint32Array(this.mem, p, 3);
        out.push({ method: METHODS[v[2]], bandwidth: v[0], dictGrid: v[1] });
      }
      return out;
    });
  }

  /** Index the next batch of the sweep in progress: the status, with `left` cells still to visit (0: done). */
  indexStep() {
    return this._serial(async () => {
      const left = this._check(await this.x.ze_index_step(this.e), "index");
      return { left, ...this.indexStatus };
    });
  }

  /** Index the region, calling progress(status) after each batch (return false to stop). */
  async index(progress = null) {
    await this.indexBegin();
    for (;;) {
      const st = await this.indexStep();
      if ((progress && progress(st) === false) || st.left === 0) return st;
    }
  }

  /** Fit how the work is cut up (settings batch, refine_pixels, wg_refine, wg_fft) to this
   *  machine and this scan: a sample of the region, sized by the bandwidth, is indexed and
   *  timed here, and each setting in turn climbs from its value (doubled while that is faster,
   *  else halved while that is). About ten trials. Nothing stored is touched.
   *  progress(callsDone) may return false to stop (the best so far stays). Resolves to
   *  { before, after } of the four settings. */
  tune(progress = null) {
    return this._serial(async () => {
      const pick = (s) => ({ batch: s.batch, refine_pixels: s.refine_pixels, wg_refine: s.wg_refine, wg_fft: s.wg_fft });
      const before = pick(this.settings);
      this._check(this.x.ze_tune_begin(this.e), "tune");
      for (let done = 1; ; done++) {
        const t0 = performance.now();
        const rc = await this.x.ze_tune_trial(this.e);
        let seconds = (performance.now() - t0) / 1000;
        if (rc !== 0) {
          // a value this machine cannot run just loses
          const message = this._error();
          if (/Cancelled/.test(message)) throw new Error(`tune: ${message}`);
          seconds = -1;
        }
        const more = this.x.ze_tune_report(this.e, seconds);
        if (!more || (progress && progress(done) === false)) break;
      }
      return { before, after: pick(this.settings) };
    });
  }

  /** Index given cells without storing: per phase and cell, 8 floats (w, x, y, z, NCC, NMI, the
   *  last improving refinement step in radians, and the NMI's distance above chance in standard
   *  deviations). The search is the settings' unless `search` gives one:
   *  { method, bandwidth, dictGrid }, each defaulting to its setting. */
  indexCells(cells, search = null) {
    return this._serial(() => this._frame(async (f) => {
      const n = cells.length, nPh = this.x.ze_phase_count(this.e);
      const p = f.alloc(nPh * n * 32);
      if (search) {
        const s = this.settings;
        this._check(await this.x.ze_index_cells_by(this.e, f.u64(cells), n, METHODS.indexOf(search.method ?? s.method), search.bandwidth ?? s.bandwidth, search.dictGrid ?? s.dict_grid, p, nPh * n * 8), "index_cells");
      } else this._check(await this.x.ze_index_cells(this.e, f.u64(cells), n, p, nPh * n * 8), "index_cells");
      return new Float32Array(this.mem, p, nPh * n * 8).slice();
    }));
  }

  /** The stored results per map cell: { quaternions (4 per cell), euler (3), ncc, nmi, phase
   *  (Int32, −1: not indexed), z (the NMI's distance above chance), sweep (Int32: the sweep that
   *  accepted the cell, −1: none) }. */
  results() {
    return this._frame((f) => {
      const s = this.scan, n = s.rows * s.cols;
      const get = (kind, k, T) => {
        const p = f.alloc(n * k * 4);
        this._check(this.x.ze_results(this.e, kind, p, n * k * 4), "results");
        return new T(this.mem, p, n * k).slice();
      };
      return { quaternions: get(0, 4, Float32Array), euler: get(4, 3, Float32Array), ncc: get(1, 1, Float32Array), nmi: get(2, 1, Float32Array), phase: get(3, 1, Int32Array),
        z: get(5, 1, Float32Array), sweep: get(6, 1, Int32Array) };
    });
  }

  /** A result map as RGBA (rows · cols · 4), for an ImageData: "ipf_x" | "ipf_y" | "ipf_z" | "ncc" | "nmi" | "phase" | "z" | "sweep". */
  map(kind = "ipf_z") {
    return this._frame((f) => {
      const s = this.scan, n = s.rows * s.cols * 4;
      const p = f.alloc(n);
      this._check(this.x.ze_map(this.e, MAPS[kind], p, n), "map");
      return new Uint8ClampedArray(this.mem, p, n).slice();
    });
  }

  /** The key to a phase's inverse pole figure colours, as the settings have them (ipf_colors,
   *  ipf_lightness, ipf_gray): { rgba (size × size × 4, for an ImageData), size, box, wedge }.
   *  The crystal directions that stand for all the others, in stereographic projection with the
   *  crystal's x to the right ([001] at the origin), transparent outside them; `box` is their
   *  bounding box in the projection [x0, y0, x1, y1], drawn from the image's lower left corner at
   *  (size − 1) / its longer side pixels per unit, and `wedge` the two azimuths (degrees) of a
   *  wedge about the c axis, null for the cubic classes' polygons. `lower` asks for the part in
   *  the lower hemisphere, which only "perfect_rotation" has for the classes of a single axis:
   *  elsewhere the box is all zero. */
  ipfKey(phase = 0, size = 128, lower = false) {
    return this._frame((f) => {
      const p = f.alloc(size * size * 4), bp = f.alloc(24);
      this._check(this.x.ze_ipf_key(this.e, phase, size, lower ? 1 : 0, p, size * size * 4, bp), "ipf_key");
      const b = Array.from(new Float32Array(this.mem, bp, 6));
      return { rgba: new Uint8ClampedArray(this.mem, p, size * size * 4).slice(), size, box: b.slice(0, 4), wedge: Number.isNaN(b[4]) ? null : [b[4], b[5]] };
    });
  }

  /** The results as a text file: "ang" (EDAX) or "ctf" (HKL); a Uint8Array. */
  exportText(kind = "ang") {
    return this._frame((f) => {
      const k = kind === "ctf" ? 1 : 0;
      const n = this._check(this.x.ze_export(this.e, k, 0, 0), "export");
      const p = f.alloc(n);
      this._check(this.x.ze_export(this.e, k, p, n), "export");
      return new Uint8Array(this.mem, p, n).slice();
    });
  }

  /** An orientation as the results store one: { q, euler }, its representative in the phase's
   *  fundamental zone and that one's Bunge angles (radians). */
  reduced(phase, q) {
    return this._frame((f) => {
      const qp = f.f32(q), ep = f.alloc(24);
      this.x.ze_reduce(this.e, phase, qp, ep);
      return { q: Array.from(new Float32Array(this.mem, qp, 4)), euler: Array.from(new Float64Array(this.mem, ep, 3)) };
    });
  }

  /** Smallest rotation (radians) between two quaternions under a phase's symmetry. */
  disorientation(phase, a, b) {
    return this._frame((f) => this.x.ze_disorientation(this.e, phase, f.f32(a), f.f32(b)));
  }
}
