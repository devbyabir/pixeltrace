// PixelTrace showcase: fills the page from results.json and runs the ONNX model in the browser.
import { preprocess } from "./preprocess.js";
import { footer } from "./footer.js";

const $ = (s) => document.querySelector(s);
const fmt = (v, d = 3) => (typeof v === "number" ? v.toFixed(d) : "–");
const pct = (v) => (typeof v === "number" ? (100 * v).toFixed(1) + " %" : "–");
const ci = (a) => (Array.isArray(a) ? `${fmt(a[0])}–${fmt(a[1])}` : "–");
const get = (o, path) => path.split(".").reduce((x, k) => (x == null ? x : x[k]), o);

let RESULTS = null;
let session = null;

async function loadResults() {
  try {
    const r = await fetch("assets/results.json", { cache: "no-cache" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    RESULTS = await r.json();
  } catch (e) {
    $("#results-status").textContent = "Results are not available (results.json missing).";
    $("#results-status").classList.add("error");
    return;
  }
  document.querySelectorAll("[data-k]").forEach((el) => {
    const v = get(RESULTS, el.dataset.k);
    if (v == null) return;
    el.textContent = typeof v === "number" && v < 1 && v > 0 ? pct(v) : v.toLocaleString("en-US");
  });
  const dep = RESULTS.models.find((m) => m.run === RESULTS.deployment.run);
  RESULTS.headline = { acc: dep.accuracy, auc: dep.roc_auc.toFixed(3) };
  document.querySelectorAll('[data-k^="headline."]').forEach((el) => {
    const v = RESULTS.headline[el.dataset.k.split(".")[1]];
    el.textContent = typeof v === "number" ? pct(v) : v;
  });
  renderTable();
  renderChart();
  const env = RESULTS.environment;
  if (env) $("#env-line").textContent = `Trained with PyTorch ${env.torch} / timm ${env.timm} on ${env.device}. ` +
    `Verification: ${RESULTS.verification_passed ? "all checks passed" : "see repository"}.`;
}

function renderTable() {
  const tb = $("#results-table tbody");
  for (const m of RESULTS.models) {
    const tr = document.createElement("tr");
    if (m.run === RESULTS.deployment.run) tr.className = "deploy";
    const cells = [m.label, fmt(m.params_m, 1), fmt(m.best_val_roc_auc), `${fmt(m.accuracy)} (${ci(m.accuracy_95ci)})`,
      fmt(m.macro_f1), `${fmt(m.ai_precision)} / ${fmt(m.ai_recall)}`, `${fmt(m.roc_auc)} (${ci(m.roc_auc_95ci)})`,
      fmt(m.roc_auc_jpeg50)];
    for (const c of cells) { const td = document.createElement("td"); td.textContent = c; tr.appendChild(td); }
    tb.appendChild(tr);
  }
  $("#results-table").hidden = false;
  $("#results-status").textContent = `Deployment model (chosen by validation ROC-AUC): ${RESULTS.deployment.label}.`;
  if (RESULTS.robustness_note) { $("#robust-note").textContent = RESULTS.robustness_note; $("#robust-note").hidden = false; }
}

function renderChart() {
  const svg = $("#chart-svg"), ms = RESULTS.models, W = 640, H = 240, pad = { l: 44, r: 12, t: 12, b: 36 };
  const lo = 0.5, hi = 1.0, bw = (W - pad.l - pad.r) / ms.length;
  const y = (v) => pad.t + (H - pad.t - pad.b) * (1 - (v - lo) / (hi - lo));
  const css = getComputedStyle(document.documentElement);
  const c1 = css.getPropertyValue("--accent").trim(), c2 = css.getPropertyValue("--muted").trim(), fg = css.getPropertyValue("--fg").trim();
  let s = `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg">`;
  for (const t of [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) {
    s += `<line x1="${pad.l}" x2="${W - pad.r}" y1="${y(t)}" y2="${y(t)}" stroke="${c2}" stroke-opacity="0.25"/>`;
    s += `<text x="${pad.l - 6}" y="${y(t) + 4}" font-size="11" text-anchor="end" fill="${c2}">${t.toFixed(1)}</text>`;
  }
  ms.forEach((m, i) => {
    const x0 = pad.l + i * bw + bw * 0.2, w = bw * 0.28;
    for (const [k, v, col] of [[0, m.best_val_roc_auc, c2], [1, m.roc_auc, c1]]) {
      const x = x0 + k * (w + 4);
      s += `<rect x="${x}" y="${y(v)}" width="${w}" height="${y(lo) - y(v)}" fill="${col}" rx="2"><title>${m.label} ${k ? "test" : "val"} ROC-AUC ${fmt(v)}</title></rect>`;
    }
    if (Array.isArray(m.roc_auc_95ci)) {
      const x = x0 + w + 4 + w / 2;
      s += `<line x1="${x}" x2="${x}" y1="${y(m.roc_auc_95ci[0])}" y2="${y(Math.min(hi, m.roc_auc_95ci[1]))}" stroke="${fg}" stroke-width="1.5"/>`;
    }
    s += `<text x="${pad.l + i * bw + bw / 2}" y="${H - 14}" font-size="12" text-anchor="middle" fill="${fg}">${m.label}</text>`;
  });
  s += `<rect x="${W - 170}" y="${pad.t}" width="10" height="10" fill="${c2}"/><text x="${W - 155}" y="${pad.t + 9}" font-size="11" fill="${fg}">validation</text>`;
  s += `<rect x="${W - 90}" y="${pad.t}" width="10" height="10" fill="${c1}"/><text x="${W - 75}" y="${pad.t + 9}" font-size="11" fill="${fg}">test (95% CI)</text>`;
  svg.outerHTML = s + "</svg>";
  $("#chart").hidden = false;
}

// ------------------------------------------------------------------ demo
function setStatus(msg, error = false) {
  const el = $("#demo-status");
  el.textContent = msg;
  el.classList.toggle("error", error);
}

async function getSession() {
  if (session) return session;
  if (!window.ort) throw new Error("ONNX Runtime Web failed to load (blocked or offline).");
  if (!RESULTS || !RESULTS.model) throw new Error("No deployed model is published with this page.");
  setStatus(`Loading model (${(RESULTS.model.bytes / 1e6).toFixed(1)} MB)…`);
  ort.env.wasm.numThreads = 1;
  session = await ort.InferenceSession.create(RESULTS.model.file, { executionProviders: ["wasm"] });
  return session;
}

async function classify(blob, label) {
  try {
    setStatus("Preprocessing…");
    $("#score").hidden = true;
    const size = RESULTS?.model?.input_size || 224;
    const { tensor, smallRGB } = await preprocess(blob, size);
    drawPreview(smallRGB, size);
    const sess = await getSession();
    setStatus("Running the model…");
    const out = await sess.run({ pixels: new ort.Tensor("float32", tensor, [1, 3, size, size]) });
    const pAI = out.scores.data[1];
    $("#meter-fill").style.left = `${(100 * pAI).toFixed(1)}%`;
    $("#score-text").innerHTML = `Model score: <strong>${(100 * pAI).toFixed(1)} %</strong> AI-generated` +
      ` <span class="small">(${pAI >= 0.5 ? "above" : "below"} the 0.5 threshold)</span>`;
    $("#model-name").textContent = RESULTS.deployment.label;
    $("#score").dataset.pAi = String(pAI);           // exact score, used by automated parity checks
    $("#score").hidden = false;
    setStatus(label ? `Input: ${label}` : "Done.");
  } catch (e) {
    console.error(e);
    setStatus(`Could not analyse this image: ${e.message}`, true);
  }
}

function drawPreview(rgb, size) {
  const c = $("#preview"), ctx = c.getContext("2d"), img = ctx.createImageData(size, size);
  for (let i = 0; i < size * size; i++) {
    img.data[i * 4] = rgb[i * 3]; img.data[i * 4 + 1] = rgb[i * 3 + 1]; img.data[i * 4 + 2] = rgb[i * 3 + 2]; img.data[i * 4 + 3] = 255;
  }
  c.width = size; c.height = size; ctx.putImageData(img, 0, 0);
}

function setupDemo() {
  const file = $("#file"), drop = $("#drop");
  file.addEventListener("change", () => file.files[0] && classify(file.files[0], file.files[0].name));
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault(); drop.classList.remove("over");
    const f = e.dataTransfer.files[0];
    if (f) classify(f, f.name);
  });
  for (const name of (RESULTS?.samples || [])) {
    const b = document.createElement("button");
    b.textContent = name.replace(/^pattern_|\.\w+$/g, "").replace(/_/g, " ");
    b.addEventListener("click", async () => {
      try {
        const r = await fetch(`samples/${name}`);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        classify(await r.blob(), `${name} (synthetic test pattern, not a photo)`);
      } catch (e) { setStatus(`Could not load sample: ${e.message}`, true); }
    });
    $("#samples").appendChild(b);
  }
  if (!RESULTS?.model) {
    setStatus("The in-browser demo is not available in this build.", true);
    file.disabled = true;
  }
}

footer(document.querySelector("#footer"));
await loadResults();
setupDemo();
