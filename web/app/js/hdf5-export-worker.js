// Load HDF5 only when exporting. WORKERFS reads slices from the local File.
import * as h from "./vendor/h5wasm/hdf5_hl.js";
import { writeOxford } from "./hdf5-export.js";

self.onmessage = async ({ data: { source, snapshot } }) => {
  try {
    await h.ready;
    h.FS.mkdir("/acquisition");
    h.FS.mount(h.FS.filesystems.WORKERFS, {
      blobs: [{ name: "source.h5oina", data: source }],
    }, "/acquisition");
    const input = new h.File("/acquisition/source.h5oina", "r");
    try {
      const output = new h.File("/results.h5oina", "w");
      try {
        writeOxford(input, output, snapshot, h);
        output.flush();
      } finally {
        output.close();
      }
    } finally {
      input.close();
      h.FS.unmount("/acquisition");
    }
    const bytes = h.FS.readFile("/results.h5oina");
    self.postMessage({ bytes }, [bytes.buffer]);
  } catch (error) {
    self.postMessage({ error: error?.message || String(error) });
  } finally {
    self.close();
  }
};
