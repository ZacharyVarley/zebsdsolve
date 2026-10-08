"""Serve the zebsdsolve web app locally and open it in the browser.

    zebsdsolve                        # or: python -m zebsdsolve
    zebsdsolve --port 8000 --no-browser

The app runs entirely in the browser (zebsdsolve.wasm on WebGPU): this only serves its static files on
127.0.0.1, which browsers treat as a secure context, as WebGPU requires. Needs a browser with
WebGPU and JavaScript Promise Integration (Chrome / Edge 137+). Scans are read in place by the
browser, in pieces, and never leave the machine.

The files come from the installed package (``zebsdsolve/web``, put there when the wheel is built) or,
in a source checkout, from the repository (``web/app``, ``web/zebsdsolve.js`` and
``zig-out/web/zebsdsolve.wasm`` after ``zig build wasm``).
"""
import argparse
import functools
import http.server
import socket
import sys
import threading
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

TYPES = {
    ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".mjs": "text/javascript",
    ".css": "text/css", ".json": "application/json", ".wasm": "application/wasm",
    ".png": "image/png", ".jpg": "image/jpeg", ".svg": "image/svg+xml",
}


def site():
    """(root directory, {url path: file}) of the app: the packaged copy, else the source checkout."""
    packaged = HERE / "web"
    if (packaged / "index.html").is_file():
        return packaged, {}
    app = REPO / "web" / "app"
    wasm = REPO / "zig-out" / "web" / "zebsdsolve.wasm"
    if not (app / "index.html").is_file():
        raise SystemExit("zebsdsolve: the web app is not in this installation (a wheel built by scripts/build_wheels.py includes it)")
    if not wasm.is_file():
        raise SystemExit(f"zebsdsolve: {wasm} missing; build it with `zig build wasm`")
    files = {"/zebsdsolve/zebsdsolve.js": REPO / "web" / "zebsdsolve.js", "/zebsdsolve/zebsdsolve.wasm": wasm}
    # the example scan (the site carries it; the wheels do not)
    for p in sorted((REPO / "testdata" / "nickel").glob("*")):
        if p.is_file():
            files[f"/presets/nickel/{p.name}"] = p
    return app, files


class _Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, **TYPES}
    # HTTP/1.1: the connection stays open after a response, for the browser to close. Under
    # http.server's default, HTTP/1.0, the server closes it as soon as the last byte is handed to
    # the system, and on at least one Windows machine files then arrived cut short or not at all
    # over loopback (15 MB: none of 6 whole; with the connection kept open, 6 of 6). A client
    # that asks for "Connection: close" still meets that close.
    protocol_version = "HTTP/1.1"

    def __init__(self, *args, files=None, **kwargs):
        self._files = files or {}
        super().__init__(*args, **kwargs)

    def translate_path(self, path):
        p = path.split("?", 1)[0].split("#", 1)[0]
        if p in self._files:
            return str(self._files[p])
        return super().translate_path(path)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass


def _free_port(preferred):
    for port in (preferred, 0):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
            except OSError:
                continue
    raise SystemExit("zebsdsolve: no free port")


def serve(port=4190, open_browser=True, block=True):
    """Serve the app on http://127.0.0.1:<port>/ (the next free port if taken); returns the URL.
    With block=False the server runs on a daemon thread (e.g. from a notebook)."""
    root, files = site()
    port = _free_port(port)
    handler = functools.partial(_Handler, directory=str(root), files=files)
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"zebsdsolve app: {url}  (Ctrl+C to stop)" if block else f"zebsdsolve app: {url}")
    if open_browser:
        threading.Timer(0.3, webbrowser.open, (url,)).start()
    if not block:
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        return url
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return url


def main(argv=None):
    ap = argparse.ArgumentParser(prog="zebsdsolve", description="Serve the zebsdsolve web app locally and open it in the browser.")
    ap.add_argument("--port", type=int, default=4190, help="port on 127.0.0.1 (default 4190; the next free one if taken)")
    ap.add_argument("--no-browser", action="store_true", help="only print the URL")
    a = ap.parse_args(argv)
    serve(a.port, open_browser=not a.no_browser)


if __name__ == "__main__":
    sys.exit(main())
