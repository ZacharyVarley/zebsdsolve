#!/usr/bin/env python3
"""The `zebsdsolve` app launcher serves every file the web app loads, with the right types.

    python tests/test_app.py        (installed wheel, or a checkout after `zig build wasm`)

No GPU and no browser: this only checks the local server.
"""
import sys
import urllib.request
from pathlib import Path

try:
    import zebsdsolve.app as app
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
    import zebsdsolve.app as app

url = app.serve(port=0, open_browser=False, block=False)
failures = []


def get(path):
    with urllib.request.urlopen(url + path.lstrip("/"), timeout=20) as r:
        return r.status, r.headers.get("Content-Type", ""), r.read()


def check(path, ctype, min_size, prefix=None):
    """GET path: status 200, Content-Type starting with ctype, at least min_size bytes."""
    try:
        status, ct, body = get(path)
        ok = status == 200 and ct.startswith(ctype) and len(body) >= min_size and (prefix is None or body.startswith(prefix))
        info = f"{status} {ct} {len(body)} B"
    except Exception as e:
        ok, info, body = False, repr(e), b""
    print(f"{'ok  ' if ok else 'FAIL'} {path}  {info}")
    if not ok:
        failures.append(path)
    return body


TYPES = {".html": "text/html", ".css": "text/css", ".js": "text/javascript", ".wasm": "application/wasm"}
check("/", "text/html", 1000)
root, extra = app.site()
paths = {"/" + p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} | set(extra)
for p in ["/zebsdsolve/zebsdsolve.js", "/zebsdsolve/zebsdsolve.wasm", "/js/app.js", "/css/app.css"]:
    if p not in paths:
        print(f"FAIL {p}  missing from the app")
        failures.append(p)
for p in sorted(paths):
    ext = p[p.rfind("."):]
    check(p, TYPES.get(ext, ""), 1, prefix=b"\0asm" if ext == ".wasm" else None)
print("app", "OK" if not failures else f"FAILED: {failures}")
sys.exit(1 if failures else 0)
