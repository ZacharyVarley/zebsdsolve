# zebsdsolve

EBSD pattern-centre fitting and indexing on the GPU, in the browser or from Python.

**[Open the web app](https://zacharyvarley.github.io/zebsdsolve/)**: it runs on your GPU inside the page (Chrome or Edge 137+). Scans are read in place, in pieces, so a 100 GB file is neither uploaded nor loaded into memory. Click **Example** for a small nickel scan, or **Make toy scan** to simulate one from a master pattern.

## What it does

- **Indexes** by spherical cross-correlation over all orientations, refined on the detector: about 23 000 patterns per second on a desktop GPU, 67 000 in two sweeps. Dictionary indexing is there too.
- **Fits the geometry**: the pattern centre by a vote of a few patterns; then the plane of source points: each fit pattern's own source point, and the affine or projective transformation of the scan through them (after Winkelmann et al. 2020 and kikuchipy's `fit_pc`), from the scan's own patterns and asking nothing of the step, the pixel size or the tilt, which it then reports with their standard errors. The app shows the search's volume in 3D and every pattern's source point against the plane.
- **Prepares patterns**: binning, neighbour averaging (NLPAR), static and dynamic background, CLAHE, band pass, mask.
- **Simulates master patterns** from a CIF (dynamical Bloch waves, depth-integrated by a low-rank Lyapunov solve corrected by its dual-weighted residual: [ebsdsim](https://github.com/ZacharyVarley/ebsdsim)'s method made 3–6× faster and several times more accurate), in the app's Simulate tab or from Python, to index with at once or save.
- **Handles several phases**, and writes `.ang` and `.ctf`.
- **Reads** h5ebsd (kikuchipy, Bruker, EDAX, Oxford `.h5oina`), EDAX `.up1` / `.up2`, Oxford `.ebsp`, NORDIF `Pattern.dat`. Master patterns: [ebsdsim](https://github.com/ZacharyVarley/ebsdsim) `.npz` and EMsoft `.h5`.

One engine (Zig and WGSL) runs natively through wgpu-native and as WebAssembly on WebGPU, so the app and the Python package do the same thing.

## Python

```bash
pip install zebsdsolve
python -m zebsdsolve          # the same app, served locally
```

```python
import zebsdsolve as ze

res = ze.index("scan.h5", ["Ni.npz"], fit="geometry")   # fit the geometry, then index
res.quaternions, res.ncc, res.phase                  # per map cell
res.geometry                                         # what the fits found
```

Every option is a keyword ([src/settings.zig](src/settings.zig)); `ze.Solver` drives the steps one at a time. Two tutorials are in [examples/](examples/).

## More

- [How it works, and what was measured](docs/how-it-works.md): each method, the numbers behind the defaults, and what is not verified.
- [CONTRIBUTING.md](CONTRIBUTING.md): building, testing, conventions.

MIT ([LICENSE](LICENSE)). The wheels bundle [wgpu-native](https://github.com/gfx-rs/wgpu-native) (MIT / Apache-2.0); the example scan is kikuchipy's `nickel_ebsd_large`, CC BY 4.0 ([sources](testdata/README.md)).
