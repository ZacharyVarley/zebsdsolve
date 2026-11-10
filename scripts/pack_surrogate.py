"""Pack the energy model's weights (TinyHistNet, a small network trained on Monte Carlo
simulations of the backscattered electrons; its weights as an .npz of named arrays, e.g.
ebsdsim's data/lightweight_hist_surrogate.npz) into src/sim/surrogate.bin, which the engine
embeds. A development aid; its output is in the repository.

    python scripts/pack_surrogate.py <weights.npz>

surrogate.bin, little-endian: magic "ZSUR", u32 version 1, u32 layer count, then per layer a
u32 kind (0 linear, 1 batch norm folded to a scale and a shift, 2 SiLU) and its arrays (linear:
u32 out, u32 in, f32 weight[out × in] row-major, f32 bias[out]; batch norm: u32 n, f32 scale[n],
f32 shift[n]); then the two heads (each linear 1 × hidden), f64 x_mean[6], f64 x_std[6], f64
the energy bin (keV), u32 1 when the outputs are logarithms (else softplus), u32 1 when the
model's energy axis runs from the beam's energy down.
"""
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main():
    blob = np.load(sys.argv[1], allow_pickle=False)
    body = {}
    for key in blob.files:
        if key.startswith("body."):
            _, idx, name = key.split(".", 2)
            body.setdefault(int(idx), {})[name] = blob[key]
    out = bytearray(b"ZSUR")
    layers = []
    for i in range(max(body) + 1):
        p = body.get(i)
        if p is None:
            layers.append(("silu",))
        elif p["weight"].ndim == 2:
            layers.append(("linear", p["weight"].astype(np.float32), p["bias"].astype(np.float32)))
        else:
            scale = p["weight"].astype(np.float64) / np.sqrt(np.maximum(p["running_var"].astype(np.float64), 0) + 1e-5)
            shift = p["bias"].astype(np.float64) - p["running_mean"].astype(np.float64) * scale
            layers.append(("bn", scale.astype(np.float32), shift.astype(np.float32)))
    layers.append(("silu",))
    out += struct.pack("<II", 1, len(layers))
    for l in layers:
        if l[0] == "linear":
            o, i = l[1].shape
            out += struct.pack("<III", 0, o, i) + l[1].tobytes() + l[2].tobytes()
        elif l[0] == "bn":
            out += struct.pack("<II", 1, l[1].size) + l[1].tobytes() + l[2].tobytes()
        else:
            out += struct.pack("<I", 2)
    for head in ("amp_head", "rate_head"):
        w = blob[f"{head}.weight"].astype(np.float32)
        out += struct.pack("<II", w.shape[0], w.shape[1]) + w.tobytes() + blob[f"{head}.bias"].astype(np.float32).tobytes()
    out += blob["x_mean"].astype(np.float64).tobytes() + blob["x_std"].astype(np.float64).tobytes()
    out += struct.pack("<d", float(blob["energy_bin_size_kev"]))
    param = str(blob["output_parameterization"]) if "output_parameterization" in blob.files else "log_amplitude_decay_weighted"
    axis = str(blob["energy_axis_kind"]) if "energy_axis_kind" in blob.files else "delta_e_ascending"
    out += struct.pack("<II", int(param.startswith("log_")), int(axis in ("delta_e_descending", "loss_descending", "legacy_descending")))
    (ROOT / "src" / "sim" / "surrogate.bin").write_bytes(bytes(out))
    print(f"{len(layers)} layers, {len(out)} bytes ({param}, {axis})")


if __name__ == "__main__":
    main()
