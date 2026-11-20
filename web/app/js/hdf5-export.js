// HDF5 serialization of the native Oxford CS1 snapshot. All orientation and
// detector frame conversions are performed by the engine.
import { copyOxfordMetadata } from "./hdf5-metadata-copy.js";

const MICROMETRES = ["um", "µm", "μm", "micrometres", "micrometers"];
const PC_FIELDS = ["Pattern Center X", "Pattern Center Y", "Detector Distance"];
const INDEX_METRICS = ["Mean Angular Deviation", "MAD", "Mean Angular Error", "Error", "Bands",
  "Band Count", "Number of Bands", "Bands Detected", "Bands Indexed", "Confidence Index", "Fit", "Pattern Fit"];

function put(group, name, data, shape) {
  return group.create_dataset({ name, data, shape });
}

function scalar(group, name) {
  const value = group.get(name)?.value;
  return Number(value?.[0] ?? value);
}

function unit(dataset) {
  const value = dataset?.attrs?.Unit?.value;
  return String(typeof value === "string" ? value : value?.[0] ?? "").trim().toLowerCase();
}

function floats(values) {
  return values instanceof Float32Array || values instanceof Float64Array;
}

function write(dataset, values) {
  dataset.write_slice(dataset.shape.map((n) => [0, n]), values);
}

function matchingPhase(group, phase) {
  if (phase.space_group <= 0 || scalar(group, "Space Group") !== phase.space_group) return false;
  const lengths = group.get("Lattice Dimensions"), angles = group.get("Lattice Angles");
  const factor = { angstrom: 1, angstroms: 1, "å": 1, nm: 10 }[unit(lengths)];
  if (!factor || !["rad", "radian", "radians"].includes(unit(angles)))
    throw new Error("Oxford source phase has unsupported lattice units.");
  const lattice = [...Array.from(lengths.value, (v) => v * factor), ...Array.from(angles.value, (v) => v * 180 / Math.PI)];
  if (lattice.length !== 6) throw new Error("Oxford source phase has an unsupported lattice layout.");
  return lattice.every((value, j) => Math.abs(value - phase.lattice[j]) <= 1e-4 + 1e-5 * Math.abs(phase.lattice[j]));
}

function mapPhases(group, phases, maxID) {
  const ids = group.keys().map(Number);
  if (ids.some((id) => !Number.isSafeInteger(id) || id < 0)) throw new Error("Invalid Oxford source phase IDs.");
  let next = Math.max(0, ...ids) + 1;
  const additions = [];
  const map = phases.map((phase) => {
    const matches = ids.filter((id) => id > 0 && matchingPhase(group.get(String(id)), phase));
    if (matches.length > 1) throw new Error("Ambiguous Oxford phase identities.");
    if (matches.length) return matches[0];
    const id = next++;
    additions.push([id, phase]);
    return id;
  });
  if (new Set(map).size !== map.length) throw new Error("Each indexed phase must map to a distinct Oxford phase.");
  if (map.some((id) => id > maxID)) throw new Error("New Oxford phase IDs exceed the original Phase datatype.");
  for (const [id, phase] of additions) {
    const g = group.create_group(String(id));
    put(g, "Phase Name", phase.name);
    put(g, "Reference", "zebsdsolve master pattern: " + phase.name);
    put(g, "Space Group", new Int32Array([phase.space_group]));
    put(g, "Laue Group", new Int32Array([phase.laue])).create_attribute("Symbol", phase.laue_symbol);
    put(g, "Lattice Dimensions", Float64Array.from(phase.lattice.slice(0, 3)), [1, 3]).create_attribute("Unit", "angstrom");
    put(g, "Lattice Angles", Float64Array.from(phase.lattice.slice(3), (v) => v * Math.PI / 180), [1, 3]).create_attribute("Unit", "rad");
  }
  return map;
}

function verifyRaster(header, data, snapshot) {
  const { rows, cols, pat_h, pat_w } = snapshot.scan;
  const n = rows * cols;
  if (scalar(header, "X Cells") !== cols || scalar(header, "Y Cells") !== rows)
    throw new Error("Oxford source grid differs from the indexed map.");
  for (const [name, size] of [["Pattern Width", pat_w], ["Pattern Height", pat_h]]) {
    if (header.get(name) && scalar(header, name) !== size)
      throw new Error("Oxford source detector dimensions differ from the indexed patterns.");
  }
  for (const [axis, axisIndex] of [["X", 0], ["Y", 1]]) {
    const positions = data.get(axis);
    const step = scalar(header, axis + " Step");
    if (positions?.shape?.join() !== String(n) || !MICROMETRES.includes(unit(positions)) ||
        !MICROMETRES.includes(unit(header.get(axis + " Step"))))
      throw new Error("Oxford positions and raster steps must declare micrometre units.");
    if (!Number.isFinite(step) || step <= 0) throw new Error("Invalid Oxford raster step.");
    const indexedStep = snapshot.geometry[`step_${axis.toLowerCase()}_um`];
    if (Math.abs(step - indexedStep) > 1e-7 + Math.abs(indexedStep) * 1e-5)
      throw new Error("Oxford source and indexed map steps differ.");
    const values = positions.value;
    const tolerance = Math.max(1e-6, step * 1e-3);
    for (let i = 0; i < n; i++) {
      const cell = axisIndex === 0 ? i % cols : Math.floor(i / cols);
      if (!Number.isFinite(values[i]) || !(Math.abs(values[i] - (values[0] + cell * step)) <= tolerance))
        throw new Error("Oxford source positions do not match engine row-major pixel IDs.");
    }
  }
}

function writeCalibration(header, provenance, calibration, h) {
  const original = provenance.create_group("OriginalCalibration");
  for (const [name, field, count] of [["Detector Orientation Euler", "effective_detector_euler_radians", 3],
    ["Tilt Angle", "effective_tilt_radians", 1]]) {
    const d = header.get(name), previous = d?.value;
    const value = calibration[field], values = Array.isArray(value) ? value : [value];
    if (values.length !== count || !values.every(Number.isFinite))
      throw new Error("Invalid native Oxford calibration: " + field);
    if (d?.metadata?.type !== h.Module.H5T_class_t.H5T_FLOAT.value ||
        (previous.length ?? 1) !== count || !["rad", "radians"].includes(unit(d)))
      throw new Error("Unsupported Oxford calibration layout or units: " + name);
    const copy = original.create_dataset({ name, data: previous, shape: d.shape, dtype: d.dtype });
    for (const [key, attr] of Object.entries(d.attrs))
      copy.create_attribute(key, attr.value, attr.shape, attr.dtype);
    write(d, values);
  }
}

export function writeOxford(input, output, snapshot, h) {
  const { geometry, scan, results, phases, calibration, oxford_frame: frame, pattern_centres: centres } = snapshot;
  const n = scan.rows * scan.cols;
  if (snapshot.schema_version !== 1 || frame?.convention_version !== 2 || frame?.automatic !== true ||
      scan.format !== "Oxford h5oina" || !Number.isSafeInteger(n) || n <= 0 ||
      results.phase.length !== n || results.euler.length !== n * 3 ||
      centres.length !== n * 3 || !centres.every(Number.isFinite))
    throw new Error("Incomplete native Oxford export snapshot.");

  const { scanPath, omitted } = copyOxfordMetadata(input, output, h);
  const header = output.get(scanPath + "/EBSD/Header"), data = output.get(scanPath + "/EBSD/Data");
  verifyRaster(header, data, snapshot);
  const euler = data.get("Euler"), phase = data.get("Phase");
  const eulerValues = euler?.value, phaseValues = phase?.value;
  const phaseSizes = new Map([[Uint8Array, 255], [Int8Array, 127], [Uint16Array, 65535], [Int16Array, 32767],
    [Uint32Array, 4294967295], [Int32Array, 2147483647], [BigUint64Array, 4294967295], [BigInt64Array, 4294967295]]);
  const maxID = phaseSizes.get(phaseValues?.constructor);
  if (euler?.shape?.join() !== `${n},3` || phase?.shape?.join() !== String(n) || !floats(eulerValues) || maxID === undefined)
    throw new Error("Unsupported Oxford Euler or Phase layout.");
  if (!["rad", "radian", "radians"].includes(unit(euler))) throw new Error("Oxford Euler angles must declare radians.");

  const map = mapPhases(header.get("Phases"), phases, maxID), indexed = new Uint8Array(n);
  const bigPhase = phaseValues instanceof BigInt64Array || phaseValues instanceof BigUint64Array;
  let count = 0;
  for (let i = 0; i < n; i++) {
    const phaseIndex = results.phase[i];
    const start = i * 3;
    const valid = Number.isInteger(phaseIndex) && phaseIndex >= 0 && phaseIndex < map.length &&
      Number.isFinite(results.euler[start]) && Number.isFinite(results.euler[start + 1]) && Number.isFinite(results.euler[start + 2]);
    const id = valid ? map[phaseIndex] : 0;
    phaseValues[i] = bigPhase ? BigInt(id) : id;
    indexed[i] = valid ? 1 : 0;
    count += indexed[i];
    for (let j = start; j < start + 3; j++) eulerValues[j] = valid ? results.euler[j] : 0;
  }
  write(euler, eulerValues);
  write(phase, phaseValues);

  const provenance = output.create_group("zebsdsolve");
  put(provenance, "Version", snapshot.version);
  put(provenance, "GeometryJSON", JSON.stringify(geometry));
  put(provenance, "ProvenanceJSON", JSON.stringify({ frame, calibration, omitted_signal_paths: omitted }));
  put(provenance, "PatternCentres", Float64Array.from(centres), [n, 3])
    .create_attribute("Convention", "Bruker: x from left / width; y from top / height; z / height");
  put(provenance, "IndexedMask", indexed);
  put(provenance, "PhaseIDMap", Uint32Array.from(map));
  writeCalibration(header, provenance, calibration, h);

  const hitRate = header.get("Hit Rate");
  if (hitRate) {
    const previous = hitRate.value;
    if (!ArrayBuffer.isView(previous) && typeof previous !== "number" && typeof previous !== "bigint")
      throw new Error("Unsupported Oxford Hit Rate layout.");
    const rate = 100 * count / n, big = typeof previous === "bigint" || previous instanceof BigInt64Array || previous instanceof BigUint64Array;
    write(hitRate, new Array(previous.length ?? 1).fill(big ? BigInt(Math.trunc(rate)) : rate));
  }
  for (const name of INDEX_METRICS) {
    const d = data.get(name);
    if (!d) continue;
    const values = d.value;
    if (d.shape?.[0] !== n || !ArrayBuffer.isView(values)) throw new Error("Unsupported Oxford indexing metric: " + name);
    const width = values.length / n, big = values instanceof BigInt64Array || values instanceof BigUint64Array;
    if (name === "Error") {
      // Oxford NotAnalyzed (0) and Success (1).
      for (let i = 0; i < n; i++) values.fill(big ? BigInt(indexed[i]) : indexed[i], i * width, (i + 1) * width);
    } else {
      values.fill(floats(values) ? NaN : big ? 0n : 0);
    }
    write(d, values);
  }

  if (!frame.preserve_source_pattern_centres) {
    const ratio = scan.pat_h / scan.pat_w;
    for (const [j, name] of PC_FIELDS.entries()) {
      const d = data.get(name);
      const values = d ? d.value : new Float64Array(n);
      if (d && (d.shape?.join() !== String(n) || !floats(values))) throw new Error("Unsupported Oxford pattern-centre layout: " + name);
      for (let i = 0; i < n; i++) {
        const value = centres[i * 3 + j];
        values[i] = indexed[i] ? (j === 0 ? value : (j === 1 ? 1 - value : value) * ratio) : NaN;
      }
      if (d) write(d, values);
      else put(data, name, values).create_attribute("Unit", "fraction");
    }
  }
  for (const name of ["ncc", "nmi", "z", "sweep"]) {
    if (results[name]) put(provenance, name.toUpperCase(), Float64Array.from(results[name], (v, i) => indexed[i] ? v ?? NaN : NaN));
  }
}
