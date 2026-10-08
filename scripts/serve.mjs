#!/usr/bin/env node
// Static dev server for the repo root (web/app, with the engine from web/zebsdsolve.js and
// zig-out/web/). WebGPU needs a secure context; http://localhost and 127.0.0.1 count as one.
//   node scripts/serve.mjs [--port 4190]
//   node scripts/serve.mjs --dir dist/web --port 4191     (a built site as is, e.g. to check a deploy)
// Range requests are honoured, so the app's example (and the browser test) read scans in pieces
// exactly as they read a picked file.
import { createServer } from "node:http";
import { createReadStream, existsSync, statSync } from "node:fs";
import { extname, join, normalize, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const REPO = resolve(fileURLToPath(new URL("..", import.meta.url)));
const TYPES = {
  ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".mjs": "text/javascript",
  ".css": "text/css", ".json": "application/json", ".wasm": "application/wasm",
  ".wgsl": "text/plain; charset=utf-8", ".png": "image/png", ".jpg": "image/jpeg", ".svg": "image/svg+xml",
};
const args = process.argv.slice(2);
const port = +(args[args.indexOf("--port") + 1] || process.env.PORT || 4190);
const dir = args.includes("--dir") ? resolve(REPO, args[args.indexOf("--dir") + 1]) : null;
const ROOT = dir || REPO;

// Paths the pages fetch next to themselves, mapped onto the repo (scripts/build_web.mjs lays
// the deployed site out the same).
const ALIASES = dir ? [] : [
  // the app loads zebsdsolve/zebsdsolve.js and zebsdsolve/zebsdsolve.wasm next to its page
  ["/web/app/zebsdsolve/zebsdsolve.js", "/web/zebsdsolve.js"], ["/web/app/zebsdsolve/", "/zig-out/web/"],
  // the example scan and its master pattern
  ["/web/app/presets/nickel/", "/testdata/nickel/"],
];

createServer((req, res) => {
  let path = decodeURIComponent(req.url.split("?")[0]);
  if (path === "/" && !dir) { res.writeHead(302, { Location: "/web/app/" }).end(); return; }
  for (const [from, to] of ALIASES) if (path.startsWith(from)) { path = to + path.slice(from.length); break; }
  if (path.endsWith("/")) path += "index.html";
  const file = normalize(join(ROOT, path));
  if (!file.startsWith(ROOT + sep) || !existsSync(file) || !statSync(file).isFile()) {
    res.writeHead(404).end("not found");
    return;
  }
  const size = statSync(file).size;
  const head = { "Content-Type": TYPES[extname(file)] || "application/octet-stream", "Cache-Control": "no-store", "Accept-Ranges": "bytes" };
  const m = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range || "");
  if (m && (m[1] || m[2])) {
    const start = m[1] ? +m[1] : Math.max(0, size - +m[2]);
    const end = m[1] && m[2] ? Math.min(+m[2], size - 1) : size - 1;
    if (start > end || start >= size) { res.writeHead(416, { "Content-Range": `bytes */${size}` }).end(); return; }
    res.writeHead(206, { ...head, "Content-Range": `bytes ${start}-${end}/${size}`, "Content-Length": end - start + 1 });
    createReadStream(file, { start, end }).pipe(res);
    return;
  }
  res.writeHead(200, { ...head, "Content-Length": size });
  createReadStream(file).pipe(res);
}).listen(port, "127.0.0.1", () => console.log(`zebsdsolve dev server: http://127.0.0.1:${port}/`));
