#!/usr/bin/env python3
"""Fetch the pinned build dependencies into .deps/ (stdlib only).

    python scripts/fetch_deps.py            # Zig for this machine + wgpu-native for this machine
    python scripts/fetch_deps.py --all      # + wgpu-native for every wheel platform (release builds)

Zig is verified against the checksum in ziglang.org's release index. wgpu-native release
archives are recorded in scripts/wgpu_native.sha256 on first download and verified after.
Nothing here is needed at run time: a wheel carries its own wgpu-native library.
"""
import argparse
import hashlib
import json
import platform
import shutil
import sys
import urllib.request
import zipfile
import tarfile
from pathlib import Path

ZIG_VERSION = "0.16.0"
WGPU_VERSION = "v29.0.1.1"

ROOT = Path(__file__).resolve().parent.parent
DEPS = ROOT / ".deps"
SUMS = Path(__file__).resolve().parent / "wgpu_native.sha256"

# wheel platform tag → wgpu-native release asset
WGPU_ASSETS = {
    "win_amd64": "wgpu-windows-x86_64-msvc-release.zip",
    "win_arm64": "wgpu-windows-aarch64-msvc-release.zip",
    "manylinux_2_28_x86_64": "wgpu-linux-x86_64-release.zip",
    "manylinux_2_28_aarch64": "wgpu-linux-aarch64-release.zip",
    "macosx_11_0_x86_64": "wgpu-macos-x86_64-release.zip",
    "macosx_11_0_arm64": "wgpu-macos-aarch64-release.zip",
}


def host_zig_key():
    m = platform.machine().lower()
    arch = {"amd64": "x86_64", "x86_64": "x86_64", "arm64": "aarch64", "aarch64": "aarch64"}.get(m, m)
    osn = {"win32": "windows", "darwin": "macos"}.get(sys.platform, "linux")
    return f"{arch}-{osn}"


def host_wheel_tag():
    m = platform.machine().lower()
    arm = m in ("arm64", "aarch64")
    if sys.platform == "win32":
        return "win_arm64" if arm else "win_amd64"
    if sys.platform == "darwin":
        return "macosx_11_0_arm64" if arm else "macosx_11_0_x86_64"
    return "manylinux_2_28_aarch64" if arm else "manylinux_2_28_x86_64"


def download(url, dst):
    print("  get", url, flush=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.replace(dst)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract(archive, dst):
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dst)
    else:
        with tarfile.open(archive) as t:
            t.extractall(dst)


def fetch_zig():
    key = host_zig_key()
    out = DEPS / f"zig-{ZIG_VERSION}"
    exe = out / ("zig.exe" if sys.platform == "win32" else "zig")
    if exe.exists():
        print(f"zig {ZIG_VERSION}: present")
        return exe
    with urllib.request.urlopen("https://ziglang.org/download/index.json") as r:
        index = json.load(r)
    rel = index[ZIG_VERSION][key]
    archive = DEPS / "dl" / Path(rel["tarball"]).name
    if not archive.exists():
        download(rel["tarball"], archive)
    got = sha256(archive)
    if got != rel["shasum"]:
        archive.unlink()
        raise SystemExit(f"zig checksum mismatch: {got} != {rel['shasum']}")
    tmp = DEPS / "dl" / "zig-extract"
    shutil.rmtree(tmp, ignore_errors=True)
    extract(archive, tmp)
    (inner,) = list(tmp.iterdir())
    shutil.rmtree(out, ignore_errors=True)
    inner.rename(out)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"zig {ZIG_VERSION}: {exe}")
    return exe


def load_sums():
    if not SUMS.exists():
        return {}
    out = {}
    for line in SUMS.read_text().splitlines():
        if line.strip():
            h, name = line.split()
            out[name] = h
    return out


def fetch_wgpu(tag):
    asset = WGPU_ASSETS[tag]
    out = DEPS / "wgpu-native" / tag
    if (out / "include").exists():
        print(f"wgpu-native {WGPU_VERSION} {tag}: present")
        return out
    url = f"https://github.com/gfx-rs/wgpu-native/releases/download/{WGPU_VERSION}/{asset}"
    archive = DEPS / "dl" / f"{WGPU_VERSION}-{asset}"
    if not archive.exists():
        download(url, archive)
    sums = load_sums()
    name = f"{WGPU_VERSION}/{asset}"
    got = sha256(archive)
    if name in sums and sums[name] != got:
        archive.unlink()
        raise SystemExit(f"wgpu-native checksum mismatch for {name}")
    if name not in sums:
        sums[name] = got
        SUMS.write_text("".join(f"{h} {n}\n" for n, h in sorted(sums.items())))
        print(f"  recorded sha256 for {name}")
    shutil.rmtree(out, ignore_errors=True)
    extract(archive, out)
    print(f"wgpu-native {WGPU_VERSION} {tag}: {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="wgpu-native for every wheel platform")
    ap.add_argument("--no-zig", action="store_true")
    a = ap.parse_args()
    if not a.no_zig:
        fetch_zig()
    for tag in (WGPU_ASSETS if a.all else [host_wheel_tag()]):
        fetch_wgpu(tag)


if __name__ == "__main__":
    main()
