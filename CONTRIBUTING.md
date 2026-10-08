# Contributing to zebsdsolve

Issues and pull requests are welcome. For anything beyond a small fix, open an issue first so we can agree on the approach.

## How it fits together

- **WGSL** (`shaders/`) does the math on patterns and spheres.
- **Zig** (`src/`) owns the schedule: which kernels run, in what order, on which buffers, and what is read back. It also reads the files.
- **Two layers connect that schedule to each environment**, each one interface over two backends:
  - the GPU layer (`src/gpu/`): natively `webgpu.h` through [wgpu-native](https://github.com/gfx-rs/wgpu-native) (Vulkan, Metal, Direct3D 12), loaded at run time; in the browser, imports that `web/zebsdsolve.js` implements with WebGPU;
  - the file layer (`src/io/file.zig`): natively positioned reads from the operating system; in the browser, byte ranges of the `File` the user picked.
- **The front ends** only pass files and settings in and take results out:
  - the web app (`web/app/`) runs `zebsdsolve.wasm`;
  - the Python package (`python/zebsdsolve/`) calls the native library through `ctypes`.

A fix to the engine therefore lands in both front ends at once.

```
shaders/            WGSL, the single copy (compiled into the library and the wasm)
  common.wgsl         quaternions, the equal-area map, flat indexing (prefixed to every module)
  fft.wgsl            radix-2 line FFT, one text for every power-of-two length
  prep.wgsl           pattern filters: load, background, blur, CLAHE, normalize
  sphere.wgsl         patterns onto the sphere and into harmonics
  so3.wgsl            normalized cross-correlation over SO(3), its peak
  detector.wgsl       master on the detector: simulation, fit sums, refinement, NMI and what chance
                      gives it, dictionary
  mastersph.wgsl      master filters on the sphere: smoothing, symmetrization; its gradient field
  pca.wgsl            a dictionary's leading components (Oja's rule) and matching in them
  nlpar.wgsl          patterns averaged with the neighbours in the map that resemble them
src/gpu/            GPU layer: gpu.zig (policy), native.zig (webgpu.h), web.zig (browser imports)
src/io/             file.zig (ranged and vectored reads, both backends, block cache), hdf5.zig (the
                    HDF5 reader), npz.zig (ZIP + NPY), codec.zig (inflate, LZF, shuffle)
src/scan.zig        every scan format behind `read(first, n)` and `readCells(cells)`
src/reduce.zig      patterns reduced on the host as they are read (binning, area resampling)
src/master.zig      master patterns (ebsdsim, EMsoft), their measured symmetry, their harmonics
src/rot.zig         quaternions, Bunge angles, so(3) / se(3), cubochoric grid
src/sym.zig         Laue groups, fundamental zones, disorientation, inverse pole figure colour keys
src/sh.zig          Gauss–Legendre grids, Legendre functions, Wigner d(π/2), host transforms
src/lambert.zig     the equal-area square ↔ hemisphere map
src/geometry.zig    the detector geometry: one pose (pattern centre ↔ pose) and the beam's positions
src/prep.zig        schedule of the pattern filters and the spherical master filters
src/spherical.zig   schedule of spherical indexing: tables, coefficients, correlation, denominators
src/detector.zig    schedule of the detector kernels
src/pca.zig         schedule of the dictionary's components: learning, coefficients, matching
src/nlpar.zig       schedule of the averaging with neighbours (NLPAR): its passes over a window
src/pcfit.zig       the pattern-centre vote
src/posefit.zig     the pose fit: SE(3) or Sim(3) with the fit patterns' orientations, in the
                    tangent space (Levenberg–Marquardt, Schur complement)
src/engine.zig      the engine both front ends call: scan, phases, the two pattern resolutions,
                    batches, sweeps, results
src/output.zig      maps as images, .ang and .ctf
src/settings.zig    every option with the app's defaults (JSON in, JSON out)
src/exports.zig     the C ABI; native_api.zig and web_api.zig are its two roots
python/zebsdsolve/  __init__.py (Solver), api.py (the plain functions), _native.py (ctypes), app.py (the `zebsdsolve` command)
web/zebsdsolve.js   the browser adapter: WebGPU and files for the wasm module
web/app/            the app (js/app.js, js/fields.js: one control per setting, js/votes.js: the
                    pattern-centre search laid open, js/draw.js)
tests/              probe.zig, check_shaders.py, test_api.py, test_app.py, test_web.mjs
examples/           the tutorials (jupytext percent scripts)
scripts/            fetch_deps.py, build_wheels.py, build_backend.py, build_web.mjs, serve.mjs,
                    run_tutorials.py, check_formats.py
testdata/           the nickel scan and master patterns the tests and the app's example use
```

## Build

Everything is fetched into `.deps/` (git-ignored, checksummed). Nothing is installed system-wide. You need Python 3, and Node 22+ for the dev server, the site build and the browser test.

```bash
python scripts/fetch_deps.py                        # Zig 0.16.0 + wgpu-native for this machine
.deps/zig-0.16.0/zig build -Doptimize=ReleaseFast   # native library → zig-out/bin (Windows) or zig-out/lib
.deps/zig-0.16.0/zig build wasm                     # zig-out/web/zebsdsolve.wasm
.deps/zig-0.16.0/zig build probe                    # open the GPU, run one kernel
.deps/zig-0.16.0/zig build test                     # host-side unit tests (math, file readers; no GPU)
```

`pip install .` builds and installs the wheel for this machine (`pyproject.toml` hands the build to `scripts/build_backend.py`, which runs `build_wheels.py` for the host platform). There is no editable install: reinstall after changing Zig or WGSL, or run from the checkout with `PYTHONPATH=python`.

From a checkout, `node scripts/serve.mjs` serves `web/app` with the freshly built `zebsdsolve.wasm` on <http://127.0.0.1:4190/web/app/>. `python -m zebsdsolve`, run from `python/`, does the same.

**Other builds:**
- **The site.** `node scripts/build_web.mjs` writes `dist/web/`: the app, `zebsdsolve.js`, `zebsdsolve.wasm` and the example scan. There is no bundler and there are no npm dependencies; any static server can host it, as long as it uses HTTPS or localhost (WebGPU requires it).
- **Wheels.** `python scripts/fetch_deps.py --all`, then `python scripts/build_wheels.py [platform …]`. Zig cross-compiles all six platforms from any host (Windows x64 / ARM64, manylinux 2.28 x64 / ARM64, macOS 11 x64 / ARM64). Each wheel is `py3-none-<platform>` (one wheel serves every Python 3) and bundles the library, that platform's wgpu-native and the web app, without the example scan.

## Test

```bash
.deps/zig-0.16.0/zig build test               # sh, rot, sym, lambert, master, scan, HDF5, NPZ, files, geometry, settings, reduce
python tests/check_shaders.py                 # every shader kernel builds on this GPU (the compiler's message if not)
python tests/test_api.py                      # the Python API on the GPU, against ground truth
python tests/test_app.py                      # the `zebsdsolve` launcher serves every file of the app
node tests/test_web.mjs                       # the app in headless Chrome (WebGPU); --dir dist/web for a built site
python scripts/run_tutorials.py out/          # examples/*.py executed as notebooks → out/*.ipynb, out/*.html
python scripts/check_formats.py FILE …        # a scan reader against kikuchipy, on files you have
```

- **`zig build test`** pins the conventions by brute force: harmonics rotate by the Wigner D-matrix, the correlation's Fourier formula equals the integral it stands for, the HDF5 and NPZ readers reproduce what h5py and NumPy read from the test files, and every inverse pole figure key gives symmetry-equivalent directions one colour, without a jump, for all eleven Laue classes.
- **`test_api`** simulates scans with known geometry, orientations and phases and checks that indexing (at the stored size, reduced, in sweeps), the vote and the pose fit recover them (a detector turned in a way the fit is not told, and a pixel size given 30 % off, included), and that cells of pure noise are not accepted; then indexes the nickel scan against the orientations and the pattern centre kikuchipy refined for it, and checks the filters against their definitions (CLAHE against the same algorithm written out in NumPy).
- **`test_web`** runs the example in the app from a wrong pattern centre: fit, fit, index, and the same comparison with kikuchipy; then again in two sweeps and on binned patterns, and the page's own moving parts: the tabs, the pattern indexed on the spot, the search's volume (its slices, and that they follow the pattern centre a fit moves), a toy scan made from the master and indexed against its own answer, the averaging's preview following its settings, the map's key (and that it follows the colours chosen) and zoom, a project file dropped on the window.
- **The tutorials** end with checks against the same truths, so they fail like tests.

Changes to a fit, to the indexing or to a default should keep these passing, and should come with numbers: before and after, on the nickel scan and on a synthetic one. Report accuracy against ground truth, not only against a previous run. Several defaults here were chosen that way (the README lists them), and some plausible ones lost.

**CI** (`.github/workflows/ci.yml`) runs on every push: on Linux (no GPU) the library, the wasm, the site and wheels, and the launcher; on an Apple M1 runner (a real GPU through Metal) `check_shaders`, `test_api` and the tutorials on the installed wheel, and `test_web` on the built site.

## Conventions

- **Orientation.** A quaternion (w, x, y, z) maps sample directions to crystal directions: `v_crystal = q v_sample q*`. Crystal symmetry multiplies on the left, and the symmetry of an orientation is the proper rotations of the phase's point group (`Master.rotation`), which are the Laue class's (`Master.laue`) only for a crystal with a centre of symmetry; the Laue class is for colouring directions. Bunge angles are radians.
- **Sample frame.** Origin at the source point of the map's centre cell; x down the tilted surface toward the detector, y along the detector's columns, z along the surface normal (kikuchipy's and EDAX's frame).
- **Geometry.** One rigid pose of the detector, and for each pattern the point of the sample the beam stood on. `x = R (row, col, 0) + t` takes detector pixel coordinates into the sample frame; `R = R0 · Exp(rot)`, the nominal attitude (sample tilt, detector tilt) turned by `rot`, a rotation vector in the detector's own axes (swing about its rows, tilt about its columns, twist about its normal). The pattern centre is in the Bruker convention and, with R, gives t (`src/geometry.zig`). A pattern recorded with the beam at `s` sees the detector at `t − s`. Nothing in the code, the interfaces or the documentation gives a pattern a pattern centre of its own: the pattern centre is the one pose's position, written for the map's centre.
- **Resolutions.** Patterns exist at two levels: reduced by `binning` on the host as they are read (`src/reduce.zig`), and as stored. Each stage (search, refinement, final scores) works at the level `full_resolution` gives it (`Engine.levelOf`), and everything a level needs is kept per level: filtered patterns, static background, each phase's filtered master. Pixels in the geometry are a level's pixels; the pattern centre's fractions are of the stored pattern, so a reduction that drops a remainder leaves it where it is on the detector. Filter sizes given in pixels (the band pass) are stored pixels.
- **Files.** A round trip to the file costs the browser about 0.8 ms and each slice about 0.25 ms, while large slices read at a gigabyte per second. Anything that reads many patterns asks for them together (`File.readv`, `File.readSpans`, `Scan.readCells`), and nothing reads a pattern at a time in a loop. Indexing reads the next batch between submitting a batch and waiting for it (`Engine.readAhead`), into a buffer the next upload takes over whole, so the file and the device work at the same time and no pattern is copied on the way. Only when patterns are averaged with their neighbours do they pass through a cache by map cell (`Engine.cache`), which holds what consecutive batches share.
- **Averaging with neighbours** (`shaders/nlpar.wgsl`, `src/nlpar.zig`) happens in `Engine.uploadCells`, before the filters, so the fits, indexing and the pattern view all see the same patterns. Its result for a cell must not depend on which batch the cell came in: the window read around a batch reaches as far as the passes do, plus one cell for the noise estimates (`Engine.nlparWindow`), and a test compares batches of different sizes.
- **Scan direction.** `scan_x_sign`, `scan_y_sign` say which way the beam moves in the sample frame as the map index grows; −1, −1 reproduces kikuchipy's `extrapolate_pc`.
- **Harmonics.** `Y_lm = P̄_l^m(cos θ) e^{imφ}`, orthonormal with the Condon–Shortley phase; coefficients stored for m ≥ 0. `src/sh.zig` states the rest and tests it.
- **Settings.** Every option lives in `src/settings.zig` with the app's default and is checked there. A new option goes there first; both front ends pick it up by name, and the app gets a control in `web/app/js/fields.js`. No control stands for a value the engine does not have, and the engine applies no constant behind a control. A control that does not apply with the settings as they are is dimmed, never hidden: what is not in view is taken not to exist. A default is a value, not a missing control: when a simpler variant of a method is the default, the fuller one's parameters stay in view.
- **Result structs** cross the C ABI in three places: the Zig `extern struct`, the byte offsets in `web/zebsdsolve.js`, and the `ctypes.Structure` in `python/zebsdsolve/_native.py`. Append fields at the end and update all three together.
- **GPU work:**
  - Buffers are allocated once and grown; patterns and masters stay on the GPU between calls.
  - Dispatches are batched into few submits; readbacks are few and batched (each costs about 3 ms in a browser). Indexing reads back once per batch (the records and chance's scores together); the pose fit once per iteration; orientation refinement runs wholly on the device.
  - The spherical work buffers are one set for every bandwidth (`Engine.sp`); a bandwidth keeps only its tables and each phase's weights (`Band`), so sweeps and fits can alternate between bandwidths.
  - Every distinct shader text is a compile on the user's machine. The FFT is one text for every length (lengths are powers of two); nothing is generated per size.
  - Lay a table out for the invocations that read it together: the index neighbouring invocations differ in goes innermost, so that one step of a workgroup reads neighbouring words (the correlation's weight tables with k innermost, the Legendre table with the pair innermost, a kernel's partial sums with the chunk innermost). Scattered reads cost several times the arithmetic they feed; this layout is what a new kernel should start from.
  - Sums that a loop adds to stay in scalars or vectors. An array indexed by a loop variable lives in memory, and a kernel that accumulated into one ran at half the speed.
  - Divisions by a number the compiler does not know (a uniform) cost more than a butterfly: sizes that are powers of two are shifted and masked, and a row and column are counted along a loop rather than divided out per pixel.
  - What may be fitted to a machine is a setting (`batch`, `refine_pixels`, `wg_refine`, `wg_fft`) that `Engine.tuneBegin` / `tuneTrial` / `tuneReport` move on a sample while the front end holds the clock (the engine has none). Tuning is a few moves, each climbing by doubling or halving from the value held (`Tune.moves`), on a sample sized by the bandwidth (`Tune.sampleOf`), never a search of every value: a new knob of that kind joins a move there, with the settings it trades against.
  - Nothing is chosen from the GPU's name: defaults are workgroups of 64 or 256 invocations within the WebGPU default limits. The two workgroup sizes that tuning may change are not written in the shader texts; the engine sets each in a line before the text it compiles (`Gpu.pipelineWith`, `shaders.zig`).
  - Kernels bind at most 8 storage buffers (the WebGPU default limit), and no dispatch exceeds 65535 workgroups per dimension.
  - The GPU computes in f32. Masters are normalized to zero mean and unit variance so sums stay well scaled; compare scores with a tolerance, never for equality.
- **Master filters** act on the sphere (`shaders/mastersph.wgsl`), not on the stored square images, so they keep the crystal's symmetry. They are not settings of their own: the master gets the patterns' CLAHE and band pass, at the sizes those have on the sphere (`Engine.bakeKeyOf`).
- **Browser calls** are serialized through the adapter. Only `cancel()` may run while an operation waits on the GPU or on a file.
- **Profiling.** Browser: `ze.stats()`. Native: `Solver.profile()`, then `Solver.profile_report()`, which times every dispatch with GPU timestamps.

## Releasing

1. **Version.** Set the version in both `python/zebsdsolve/__init__.py` (`__version__`) and `src/engine.zig` (`version`). The wheel build refuses a mismatch.
2. **Tag.** Push a tag `vX.Y.Z`. `.github/workflows/release.yml` then builds the six wheels and the site; tests the wheels on Linux, Windows and macOS (on the M1, on its GPU, in headless Chrome, and by running the tutorials); publishes to PyPI by trusted publishing; attaches the wheels, the site and the executed tutorials to a GitHub Release; and deploys the site to GitHub Pages with `zebsdsolve/pipelines.json`, the pipelines the M1's browser test compiled (`test_web.mjs --save-pipelines`), which a first visit starts compiling as the page loads. Run by hand (Actions → release → Run workflow), it only builds and tests.
3. **One-time setup.** On PyPI, add a trusted publisher for this repository: workflow `release.yml`, environment `pypi`. In the repository settings, create the environment `pypi`, set Pages' source to **GitHub Actions**, and under Environments → `github-pages` → Deployment branches and tags add a tag rule `v*`.
