// Native WASM conversion and browser adapter, without GPU or browser dependencies.
// node web/tests/fitted-pose.mjs
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { Zebsdsolve } from "../zebsdsolve.js";

const module = await WebAssembly.compile(await readFile(new URL("../../zig-out/web/zebsdsolve.wasm", import.meta.url)));
const imports = {};
for (const entry of WebAssembly.Module.imports(module)) {
  assert.equal(entry.kind, "function");
  (imports[entry.module] ||= {})[entry.name] = () => { throw new Error(`unexpected import: ${entry.module}.${entry.name}`); };
}
const { exports } = await WebAssembly.instantiate(module, imports);
const adapter = Object.create(Zebsdsolve.prototype);
adapter.state = { memory: exports.memory };
adapter.x = { ze_alloc: exports.ze_alloc, ze_free: exports.ze_free, ze_bunge: exports.ze_bunge };
// A half turn about Z would reduce to identity under generic C2(Z). The inspector
// must retain that pose and show its own Bunge angles instead.
for (const q of [[0, 0, 0, 1], [0, 0, 0, -1], [0, 0, 0, 2]]) {
  const before = q.slice();
  const euler = adapter.bunge(q);
  assert(Math.abs(euler[0] - Math.PI) < 1e-12);
  assert.equal(euler[1], 0);
  assert.equal(euler[2], 0);
  assert.deepEqual(q, before);
}
assert(adapter.bunge([1, 0, 0, 0]).every((v) => Math.abs(v) < 1e-12));
assert(adapter.bunge(new Float32Array([1, 0, 0, 0])).every((v) => Math.abs(v) < 1e-12));
for (const invalid of [null, "1234", [], [1, 0, 0], [1, 0, 0, 0, 0], new Float32Array(3)])
  assert.throws(() => adapter.bunge(invalid), /four quaternion components/);
console.log("fitted pose browser/native WASM accessor OK");
