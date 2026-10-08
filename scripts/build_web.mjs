#!/usr/bin/env node
// Static web build: dist/web/ is the deployable site (any static host; WebGPU needs HTTPS or
// localhost). No bundler and no npm dependencies: the app is plain ES modules.
//   zig build wasm -Doptimize=ReleaseSmall && node scripts/build_web.mjs
//
// Layout (the same paths the app uses in the repo through scripts/serve.mjs aliases):
//   index.html css/ js/             web/app
//   zebsdsolve/zebsdsolve.wasm, .js zig-out/web, web/zebsdsolve.js (the engine; its WGSL is compiled in)
//   presets/nickel/                 testdata/nickel (the example scan and its master pattern)
import { cpSync, existsSync, mkdirSync, rmSync } from "node:fs";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(fileURLToPath(new URL("..", import.meta.url)));
const OUT = join(ROOT, "dist", "web");
const wasm = join(ROOT, "zig-out", "web", "zebsdsolve.wasm");
if (!existsSync(wasm)) {
  console.error("zig-out/web/zebsdsolve.wasm missing: run `zig build wasm -Doptimize=ReleaseSmall` first");
  process.exit(1);
}
rmSync(OUT, { recursive: true, force: true });
mkdirSync(join(OUT, "zebsdsolve"), { recursive: true });
for (const d of ["index.html", "css", "js"]) cpSync(join(ROOT, "web", "app", d), join(OUT, d), { recursive: true });
cpSync(wasm, join(OUT, "zebsdsolve", "zebsdsolve.wasm"));
cpSync(join(ROOT, "web", "zebsdsolve.js"), join(OUT, "zebsdsolve", "zebsdsolve.js"));
cpSync(join(ROOT, "testdata", "nickel"), join(OUT, "presets", "nickel"), { recursive: true });
console.log(`built ${OUT}`);
