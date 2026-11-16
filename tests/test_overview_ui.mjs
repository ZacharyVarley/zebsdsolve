// Focused async lifecycle checks against the app's actual overview routines; no GPU or browser.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

const source = readFileSync(new URL("../web/app/js/app.js", import.meta.url), "utf8");
const controls = new Map();
const control = (id) => {
  if (!controls.has(id)) controls.set(id, {
    options: [], disabled: false, textContent: "", value: "",
    replaceChildren() { this.options = []; }, add(option) { this.options.push(option); },
  });
  return controls.get(id);
};
let contrastCalls = 0, previewCalls = 0, draws = 0, stored = null;
const st = {
  scan: { rows: 2, cols: 3, patH: 8, patW: 10, pixel: 0 },
  overviewRun: 0, overviewLoading: true, busy: false, mapKind: "overview",
  storedOverview: null, calculatedOverview: null, overviewSource: null,
  ze: {
    storedOverview: async () => stored,
    patternStats: async (cells) => {
      contrastCalls++;
      return { std: Float32Array.from(cells, (c) => c + 10) };
    },
  },
};
const context = vm.createContext({
  st, $: control, Float32Array, Option: function(text, value) { Object.assign(this, {text, value}); },
  performance, log() {}, requestAnimationFrame: (f) => setImmediate(f),
  sleep: async () => {}, drawMap: () => draws++, showCell: () => previewCalls++,
  setMapKind: (kind) => { st.mapKind = kind; },
});
vm.runInContext(source.slice(source.indexOf("function invalidateOverview()"), source.indexOf("// ── phases ──")), context);
const run = (code) => vm.runInContext(code, context);

// No stored field leaves a neutral map; opening never computes pattern statistics.
await run("loadStoredOverview()");
assert.equal(contrastCalls, 0);
assert.equal(st.overviewSource, null);
assert.equal(st.overviewLoading, false);
assert.equal(previewCalls, 1);
assert.equal(control("btn-overview").disabled, false);

// Stored acquisition quality is drawn before the selected pattern can be shown.
stored = {label: "Band contrast", values: Float32Array.from([5, 1, 8, 3, 6, 2])};
st.overviewLoading = true;
context.showCell = () => { assert.ok(draws > 0); previewCalls++; };
await run("loadStoredOverview()");
assert.equal(contrastCalls, 0);
assert.equal(st.overviewSource, "stored");
assert.equal(st.overview, stored.values);
assert.equal(control("overview-source").value, "stored");

// Calculation only begins when explicitly requested; a switch to stored survives it.
let finish;
st.ze.patternStats = async () => { contrastCalls++; return new Promise((r) => { finish = r; }); };
const calculating = run("runOverview()");
assert.equal(contrastCalls, 1);
assert.equal(st.overviewCalculating, true);
st.overviewSource = "stored";
st.overview = stored.values;
finish({ std: Float32Array.from([10, 11, 12, 13, 14, 15]) });
await calculating;
assert.equal(st.overviewSource, "stored");
assert.equal(st.overview, stored.values);
assert.deepEqual(Array.from(st.calculatedOverview.values), [10, 11, 12, 13, 14, 15]);
assert.equal(st.overviewCalculating, false);

// Stop and scan replacement discard late callbacks without touching the next overview.
const pending = run("runOverview()");
const old = st.calculatedOverview.values;
run("invalidateOverview()");
finish({std: new Float32Array(6).fill(99)});
await pending;
assert.ok(Array.from(old).every(Number.isNaN));
assert.equal(st.overviewCalculating, false);
let deliver;
st.ze.storedOverview = () => new Promise((r) => { deliver = r; });
st.overviewLoading = true;
const loading = run("loadStoredOverview()");
run("invalidateOverview()");
st.overviewSource = null;
st.storedOverview = null;
deliver(stored);
await loading;
assert.equal(st.overviewSource, null);
assert.equal(st.storedOverview, null);
console.log("Overview UI checks passed: neutral fallback, stored-first preview, explicit contrast, source switching, stop and stale results.");
