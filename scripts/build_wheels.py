#!/usr/bin/env python3
"""Build zebsdsolve wheels (stdlib only on Python 3.11+). Every platform is cross-compiled from this machine by Zig;
the package metadata comes from pyproject.toml's [project] table.

    python scripts/fetch_deps.py --all          # Zig + wgpu-native for every platform
    python scripts/build_wheels.py              # all platforms → dist/*.whl
    python scripts/build_wheels.py win_amd64    # just one
    python -m twine upload dist/*.whl           # publish

A wheel holds the Python package, the zebsdsolve shared library for its platform, that
platform's wgpu-native, and the web app (zebsdsolve/web: the `zebsdsolve` command serves it on
localhost). The package calls the library through ctypes (no CPython ABI), so one wheel per
platform serves every Python 3 version: tag py3-none-<platform>.
"""
import base64
import hashlib
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_deps import DEPS, WGPU_ASSETS, WGPU_VERSION, ZIG_VERSION, fetch_wgpu  # noqa: E402

def _version():
    """The package version (python/zebsdsolve/__init__.py), which the engine's must equal."""
    py = re.search(r'^__version__ = "([^"]+)"', (ROOT / "python" / "zebsdsolve" / "__init__.py").read_text(encoding="utf-8"), re.M)[1]
    zig = re.search(r'^pub const version = "([^"]+)";', (ROOT / "src" / "engine.zig").read_text(encoding="utf-8"), re.M)[1]
    if py != zig:
        raise SystemExit(f"version mismatch: python/zebsdsolve/__init__.py {py}, src/engine.zig {zig}")
    return py


VERSION = _version()
# wheel platform tag → (zig target, zebsdsolve library, wgpu-native library)
TARGETS = {
    "win_amd64": ("x86_64-windows", "bin/zebsdsolve.dll", "wgpu_native.dll"),
    "win_arm64": ("aarch64-windows", "bin/zebsdsolve.dll", "wgpu_native.dll"),
    "manylinux_2_28_x86_64": ("x86_64-linux-gnu.2.28", "lib/libzebsdsolve.so", "libwgpu_native.so"),
    "manylinux_2_28_aarch64": ("aarch64-linux-gnu.2.28", "lib/libzebsdsolve.so", "libwgpu_native.so"),
    "macosx_11_0_x86_64": ("x86_64-macos.11.0", "lib/libzebsdsolve.dylib", "libwgpu_native.dylib"),
    "macosx_11_0_arm64": ("aarch64-macos.11.0", "lib/libzebsdsolve.dylib", "libwgpu_native.dylib"),
}
assert set(TARGETS) == set(WGPU_ASSETS)

def _pypi_readme():
    """The README with its relative links made absolute (PyPI has no repository to resolve them in)."""
    repo = "https://github.com/ZacharyVarley/zebsdsolve"
    return re.sub(r"(!?\[[^\]]*\])\((?!https?://|#|mailto:)([^)\s]+)\)",
                  lambda m: f"{m[1]}({repo}/{'raw' if m[1][0] == '!' else 'blob'}/main/{m[2]})",
                  (ROOT / "README.md").read_text(encoding="utf-8"))


def _project():
    """pyproject.toml's [project] table: the package metadata, in one place."""
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        import tomli as tomllib
    with open(ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]


PROJECT = _project()


def _metadata(p):
    """Core metadata 2.4 (METADATA / PKG-INFO) from the [project] table."""
    out = [("Metadata-Version", "2.4"), ("Name", p["name"]), ("Version", VERSION), ("Summary", p["description"])]
    out += [("Author", a["name"]) for a in p.get("authors", [])]
    out += [("License-Expression", p["license"])]
    out += [("License-File", Path(f).name) for f in p.get("license-files", [])]
    out += [("Project-URL", f"{k}, {v}") for k, v in p.get("urls", {}).items()]
    out += [("Keywords", ",".join(p.get("keywords", [])))]
    out += [("Classifier", c) for c in p.get("classifiers", [])]
    out += [("Requires-Python", p["requires-python"])]
    out += [("Requires-Dist", d) for d in p.get("dependencies", [])]
    for extra, deps in p.get("optional-dependencies", {}).items():
        out += [("Provides-Extra", extra)] + [("Requires-Dist", f'{d}; extra == "{extra}"') for d in deps]
    out += [("Description-Content-Type", "text/markdown")]
    return "".join(f"{k}: {v}\n" for k, v in out) + "\n" + _pypi_readme()


METADATA = _metadata(PROJECT)
ENTRY_POINTS = "[console_scripts]\n" + "".join(f"{k} = {v}\n" for k, v in PROJECT.get("scripts", {}).items())


def zig():
    """$ZIG if set (e.g. CI's setup-zig), else the fetched .deps/zig-<version>."""
    if os.environ.get("ZIG"):
        return Path(os.environ["ZIG"])
    exe = DEPS / f"zig-{ZIG_VERSION}" / ("zig.exe" if sys.platform == "win32" else "zig")
    if not exe.exists():
        raise SystemExit("Zig missing: python scripts/fetch_deps.py")
    return exe


def build_lib(tag):
    triple, rel, _ = TARGETS[tag]
    wgpu = fetch_wgpu(tag)
    prefix = ROOT / "zig-out" / "wheel" / tag
    cmd = [str(zig()), "build", "-Doptimize=ReleaseFast", f"-Dtarget={triple}", f"-Dwgpu={wgpu}", "--prefix", str(prefix)]
    print(" ", " ".join(cmd[1:4]))
    subprocess.run(cmd, cwd=ROOT, check=True)
    return prefix / rel, wgpu


_web = None


def web_files(wgpu):
    """The web app as the wheel carries it (zebsdsolve/web/...), built once: zebsdsolve.wasm and the
    static files, laid out as scripts/build_web.mjs lays out dist/web."""
    global _web
    if _web is None:
        print("  wasm")
        subprocess.run([str(zig()), "build", "wasm", f"-Dwgpu={wgpu}"], cwd=ROOT, check=True)
        app = ROOT / "web" / "app"
        _web = {f"zebsdsolve/web/{p.relative_to(app).as_posix()}": p.read_bytes()
                for p in sorted(app.rglob("*")) if p.is_file()}
        _web["zebsdsolve/web/zebsdsolve/zebsdsolve.js"] = (ROOT / "web" / "zebsdsolve.js").read_bytes()
        _web["zebsdsolve/web/zebsdsolve/zebsdsolve.wasm"] = (ROOT / "zig-out" / "web" / "zebsdsolve.wasm").read_bytes()
    return _web


def record_line(arc, data):
    h = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return f"{arc},sha256={h},{len(data)}"


def build_wheel(tag):
    triple, _, wgpu_name = TARGETS[tag]
    lib, wgpu = build_lib(tag)
    files = {}
    for p in sorted((ROOT / "python" / "zebsdsolve").glob("*.py")):
        files[f"zebsdsolve/{p.name}"] = p.read_bytes()
    files.update(web_files(wgpu))
    files[f"zebsdsolve/_lib/{lib.name}"] = lib.read_bytes()
    files[f"zebsdsolve/_lib/{wgpu_name}"] = (wgpu / "lib" / wgpu_name).read_bytes()
    # the release archives carry no license; wgpu-native is MIT OR Apache-2.0 (scripts/licenses/)
    files["zebsdsolve/_lib/wgpu-native-LICENSE.MIT"] = (ROOT / "scripts" / "licenses" / "wgpu-native-LICENSE.MIT").read_bytes()
    di = f"zebsdsolve-{VERSION}.dist-info"
    files[f"{di}/METADATA"] = METADATA.encode()
    files[f"{di}/entry_points.txt"] = ENTRY_POINTS.encode()
    for f in PROJECT.get("license-files", []):
        files[f"{di}/licenses/{Path(f).name}"] = (ROOT / f).read_bytes()
    files[f"{di}/WHEEL"] = f"Wheel-Version: 1.0\nGenerator: zebsdsolve build_wheels.py\nRoot-Is-Purelib: false\nTag: py3-none-{tag}\n".encode()
    record = [record_line(a, d) for a, d in files.items()] + [f"{di}/RECORD,,"]
    files[f"{di}/RECORD"] = ("\n".join(record) + "\n").encode()
    out = ROOT / "dist" / f"zebsdsolve-{VERSION}-py3-none-{tag}.whl"
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for arc, data in files.items():
            z.writestr(arc, data)
    print(f"{out.name}  {out.stat().st_size / 1e6:.1f} MB  ({triple})")
    return out


def main():
    tags = sys.argv[1:] or list(TARGETS)
    for t in tags:
        if t not in TARGETS:
            raise SystemExit(f"unknown platform {t}; one of {', '.join(TARGETS)}")
        build_wheel(t)


if __name__ == "__main__":
    main()
