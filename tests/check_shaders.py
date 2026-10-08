"""Build every compute pipeline of the engine's shader modules on this GPU and report the ones
its compiler rejects, with the message. Each kernel builds in a process of its own: after one
internal compiler error, some drivers (Metal on GitHub's M1 runners) fail every later build in
the same process. Also counts the storage buffers each kernel binds: WebGPU guarantees only 8
per shader stage. Exit status 1 if any kernel fails either check.

    python tests/check_shaders.py
"""
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))


def entries(code):
    return re.findall(r"@compute\s+@workgroup_size\([^)]*\)\s*fn\s+(\w+)", code)


def storage_buffers(code):
    """{entry point: the storage buffers it uses, itself or through the functions it calls}."""
    storage = set(re.findall(r"var<storage[^>]*>\s*(\w+)\s*:", code))
    bodies = {}
    for m in re.finditer(r"\bfn\s+(\w+)\s*\(", code):
        i = code.index("{", m.end())
        depth, j = 1, i + 1
        while depth:
            depth += {"{": 1, "}": -1}.get(code[j], 0)
            j += 1
        bodies[m.group(1)] = set(re.findall(r"\b\w+\b", code[i:j]))

    def used(fn, seen):
        seen.add(fn)
        out = bodies[fn] & storage
        for callee in bodies[fn] & bodies.keys():
            if callee not in seen:
                out |= used(callee, seen)
        return out

    return {e: used(e, set()) for e in entries(code)}


def one(module, entry):
    """Build one kernel in this process; print its compiler message if it fails."""
    import zebsdsolve

    s = zebsdsolve.Solver()
    msg = s._compile_wgsl(zebsdsolve.shader_modules()[module], entry)
    if msg is not None:
        lines = [l.strip() for l in msg.strip().splitlines() if l.strip()]
        print(" | ".join(lines[:6]) if lines else msg)
        sys.exit(1)


def main():
    import zebsdsolve

    s = zebsdsolve.Solver()
    print("adapter:", s.adapter, flush=True)
    s.close()
    modules = zebsdsolve.shader_modules()
    jobs = [(name, entry) for name, code in modules.items() for entry in entries(code)]

    def run(job):
        r = subprocess.run([sys.executable, __file__, "--one", *job], capture_output=True, text=True, timeout=300)
        return job, r.returncode, (r.stdout.strip().splitlines() or [r.stderr.strip()[-600:]])[-1]

    with ThreadPoolExecutor(4) as ex:
        results = list(ex.map(run, jobs))
    bad = 0
    for (name, entry), rc, msg in results:
        if rc != 0:
            bad += 1
            print(f"FAIL {name}/{entry}: {msg}")
    most = 0
    for name, code in modules.items():
        for entry, bufs in storage_buffers(code).items():
            most = max(most, len(bufs))
            if len(bufs) > 8:
                bad += 1
                print(f"FAIL {name}/{entry}: binds {len(bufs)} storage buffers (WebGPU guarantees 8): {', '.join(sorted(bufs))}")
    for name in modules:
        n = sum(1 for (m, _), rc, _ in results if m == name)
        print(f"{name}: {n} kernels checked")
    print(f"most storage buffers in one kernel: {most}")
    print("shaders OK" if bad == 0 else f"{bad} kernels failed")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--one":
        one(sys.argv[2], sys.argv[3])
    else:
        main()
