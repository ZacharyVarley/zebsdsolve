// The actual adapter queue with suspended IO: no browser or GPU required.
import assert from "node:assert/strict";
import { Zebsdsolve } from "../web/zebsdsolve.js";
const delay = ms => new Promise(r => setTimeout(r, ms));
const until = async fn => { for (let i = 0; i < 300; i++) { if (fn()) return; await delay(2); } throw Error("queue timeout"); };
const memory = new WebAssembly.Memory({initial: 2}), calls = [];
let next = 4096, active = 0, source = 0, visited = 0, destroyed = false;
const operation = async (name, fn) => {
  assert.equal(active, 0, "native operations must not overlap");
  assert.equal(destroyed, false);
  active++; calls.push(name + ":start"); await delay(15); fn?.();
  active--; calls.push(name + ":end");
};
const x = {
  ze_create: () => 1, ze_error: () => 0,
  ze_alloc(n) { const p = next; next += Math.ceil(n / 8) * 8; return p; }, ze_free() { calls.push("free"); },
  ze_pattern_shape(e, full, p) { new Uint32Array(memory.buffer, p, 2).set([2, 2]); return 0; }, ze_nlpar_view: () => 0,
  async ze_open_scan() { await operation("open", () => { source++; visited = 0; }); return 0; },
  async ze_pattern(e, cell, k, full, p) { await operation("pattern", () => new Float32Array(memory.buffer, p, 4).fill(source)); return 0; },
  async ze_prepare_index_step() { await operation("warm", () => visited++); return visited >= 3 ? 1 : 0; },
  ze_close_scan() { assert.equal(active, 0); calls.push("close"); },
  ze_destroy() { assert.equal(active, 0); destroyed = true; calls.push("destroy"); },
  ze_pixel_angle() { assert.equal(destroyed, false); return 1; }, ze_clear_phases() { assert.equal(destroyed, false); },
};
const z = new Zebsdsolve(x, {memory, listeners: [], files: []});
Object.defineProperty(z, "scan", {get: () => ({source})});
try {
  for (const n of [-1, 0, 1.5, 2 ** 32, Infinity]) assert.throws(() => z.prepareIndexStep(n), /uint32/);
  // A preview queued immediately after open belongs to the newly opened scan.
  const opening = z.openScan(new Blob(["scan"])), first = z.pattern(0, false);
  await opening; assert.equal((await first)[0], 1); assert.equal(visited, 0);
  await until(() => calls.includes("warm:start"));
  assert.equal((await z.pattern(1, false))[0], 1);
  assert.ok(visited < 3, "foreground preview must not wait for the complete lookup");
  await z.closeScan(); const stopped = visited; await delay(35); assert.equal(visited, stopped);
  const reopening = z.openScan(new Blob(["next"])), preview = z.pattern(0, false);
  await reopening; assert.equal((await preview)[0], 2);
  await until(() => visited > 0);
} finally { await z.dispose(); }
assert.equal(calls.at(-1), "destroy");
assert.ok(calls.lastIndexOf("free") < calls.indexOf("destroy"));
assert.throws(() => z.pixelAngle, /disposed/); assert.throws(() => z.clearPhases(), /disposed/);
await assert.rejects(z.pattern(0, false), /disposed/);
const stopped = calls.length; await delay(35); assert.equal(calls.length, stopped);
console.log("Chunk queue passed: queued open/preview, foreground priority, close and disposal.");
