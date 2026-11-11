import { lineChart, intervalChart, barChart, table, legend } from "./charts.js";
import { footer } from "./footer.js";

const $ = (s) => document.querySelector(s);
const f3 = (v) => v.toFixed(3);
const pct = (v) => `${(100 * v).toFixed(1)} %`;
const COLORS = { efficientnet_b0: "--s1", resnet50: "--s2", simple_cnn: "--s3" };
const LEVEL_NAMES = { jpeg: "JPEG quality", downscale: "down-scale factor", blur: "blur radius (px)" };

let R;
try {
  R = await (await fetch("assets/results.json", { cache: "no-cache" })).json();
} catch {
  document.querySelector("main").innerHTML = `<p class="status error">Results could not be loaded.</p>`;
  throw new Error("results.json missing");
}
footer($("#footer"));
const M = R.models;
const dep = M.find((m) => m.run === R.deployment.run);

// ---- headline tiles
$("#tiles").innerHTML = [
  ["Deployed model", R.deployment.label, `chosen by validation ROC-AUC ${f3(dep.best_val_roc_auc)}`],
  ["Test accuracy", pct(dep.accuracy), `95 % CI ${pct(dep.accuracy_95ci[0])}–${pct(dep.accuracy_95ci[1])}`],
  ["Test ROC-AUC", f3(dep.roc_auc), `95 % CI ${f3(dep.roc_auc_95ci[0])}–${f3(dep.roc_auc_95ci[1])}`],
  ["Legacy notebooks reported", `${pct(R.baselines.legacy_reported_accuracy.min)}–${pct(R.baselines.legacy_reported_accuracy.max)}`, "leaky split, duplicates, test-set selection"],
].map(([k, v, s]) => `<div class="tile"><div class="tile-k">${k}</div><div class="tile-v">${v}</div><div class="tile-s">${s}</div></div>`).join("");
$("#split-note").textContent = `Test split: ${R.split.test} images (${R.split.test / 2} AI / ${R.split.test / 2} real), never used for training or model selection. ` +
  `Trained on ${R.environment.device}, PyTorch ${R.environment.torch}.`;

// ---- context bars
const ctx = [
  { label: "Legacy notebooks (best reported)", value: R.baselines.legacy_reported_accuracy.max, color: "--muted-bar", tip: "Reported by the original notebooks; inflated by a leaky split and duplicates." },
  { label: "Metadata only, raw files (no pixels)", value: R.baselines.metadata_raw.test_accuracy, color: "--muted-bar", tip: "Logistic regression on aspect ratio, resolution, PNG/JPEG, EXIF, file size. Shows shortcut risk." },
  { label: "Metadata only, cleaned files", value: R.baselines.filesize_clean.test_accuracy, color: "--muted-bar", tip: "After cleaning only JPEG file size remains: close to chance." },
  { label: `PixelTrace ${R.deployment.label} (images)`, value: dep.accuracy, color: "--s1", tip: `Rebuilt pipeline, test split. 95 % CI ${pct(dep.accuracy_95ci[0])}–${pct(dep.accuracy_95ci[1])}.` },
];
barChart($("#c-context"), { rows: ctx, max: 1, ticks: [0, 0.25, 0.5, 0.75, 1], fmt: pct });
table($("#c-context"), ["What", "Test accuracy"], ctx.map((r) => [r.label, pct(r.value)]));

// ---- CI charts
const rows = (k, ci) => M.map((m) => ({ label: m.label, color: COLORS[m.run], value: m[k], lo: m[ci][0], hi: m[ci][1],
  marker: k === "roc_auc" ? { value: m.best_val_roc_auc, label: "Validation ROC-AUC" } : null }));
intervalChart($("#c-auc"), { rows: rows("roc_auc", "roc_auc_95ci"), domain: [0.7, 1], ticks: [0.7, 0.8, 0.9, 1.0], fmt: f3, xLabel: "Test ROC-AUC", valueName: "Test ROC-AUC" });
intervalChart($("#c-acc"), { rows: rows("accuracy", "accuracy_95ci"), domain: [0.5, 1], ticks: [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fmt: pct,
  xLabel: "Test accuracy", valueName: "Test accuracy", refLines: [{ value: 0.5, label: "chance" }] });
table($("#c-acc"), ["Model", "Params (M)", "Val ROC-AUC", "Test accuracy (95 % CI)", "Macro-F1", "AI precision", "AI recall", "Test ROC-AUC (95 % CI)"],
  M.map((m) => [m.label, m.params_m.toFixed(1), f3(m.best_val_roc_auc), `${f3(m.accuracy)} (${f3(m.accuracy_95ci[0])}–${f3(m.accuracy_95ci[1])})`,
    f3(m.macro_f1), f3(m.ai_precision), f3(m.ai_recall), `${f3(m.roc_auc)} (${f3(m.roc_auc_95ci[0])}–${f3(m.roc_auc_95ci[1])})`]));

// ---- ROC curves
lineChart($("#c-roc"), {
  size: { width: 640, height: 380, margin: { t: 16, r: 120, b: 44, l: 52 } },
  series: M.map((m, i) => ({ name: `${m.label} (AUC ${f3(m.roc_auc)})`, short: m.label, color: COLORS[m.run], points: R.dashboard[m.run].roc, labelDy: [-10, 4, 18][i] })),
  xDomain: [0, 1], yDomain: [0, 1], xTicks: [0, 0.25, 0.5, 0.75, 1], yTicks: [0, 0.25, 0.5, 0.75, 1], xFmt: (v) => v.toFixed(2), yFmt: (v) => v.toFixed(2),
  xLabel: "False-positive rate (real photos flagged)", yLabel: "True-positive rate", diagonal: true,
  tipX: (v) => `FPR ≈ ${Math.max(0, v).toFixed(2)}`, tipFmt: (p) => `TPR ${p[1].toFixed(3)}`,
});

// ---- robustness small multiples (shared y)
for (const fam of ["jpeg", "downscale", "blur"]) {
  const lv = R.dashboard[dep.run].robustness.filter((r) => r.corruption === fam).map((r) => r.level);
  const cats = ["clean", ...lv];
  const series = M.map((m) => {
    const rb = R.dashboard[m.run].robustness;
    const clean = rb.find((r) => r.corruption === "clean").roc_auc;
    return { name: m.label, color: COLORS[m.run], points: [[0, clean], ...rb.filter((r) => r.corruption === fam).map((r, i) => [i + 1, r.roc_auc])] };
  });
  const box = $(`#c-rob-${fam}`);
  box.insertAdjacentHTML("beforebegin", "");
  lineChart(box, {
    size: { width: 320, height: 240, margin: { t: 24, r: 12, b: 40, l: 44 } }, series, markers: true, noLegend: true,
    xDomain: [-0.3, cats.length - 0.7], yDomain: [0.7, 0.95], xCats: cats, yTicks: [0.7, 0.75, 0.8, 0.85, 0.9, 0.95], yFmt: (v) => v.toFixed(2),
    xLabel: LEVEL_NAMES[fam], tipX: (v) => cats[Math.max(0, Math.min(cats.length - 1, Math.round(v)))], tipFmt: (p) => `ROC-AUC ${p[1].toFixed(3)}`,
  });
  box.insertAdjacentHTML("afterbegin", `<div class="chart-title">${LEVEL_NAMES[fam]}</div>`);
}
const robGrid = $("#c-rob-jpeg").parentElement;
const robLegend = document.createElement("div"), robTable = document.createElement("div");
robGrid.before(robLegend); robGrid.after(robTable);
legend(robLegend, M.map((m) => ({ name: m.label, color: COLORS[m.run] })));
table(robTable, ["Model", "Corruption", "Level", "ROC-AUC", "Accuracy", "AI recall", "Real recall"],
  M.flatMap((m) => R.dashboard[m.run].robustness.map((r) => [m.label, r.corruption, r.level, f3(r.roc_auc), f3(r.accuracy), f3(r.ai_recall), f3(r.real_recall)])));

// ---- recall split for the deployed model
const rb = R.dashboard[dep.run].robustness;
const conds = rb.map((r) => (r.corruption === "clean" ? "clean" : `${r.corruption} ${r.level}`));
const withGaps = (key) => rb.flatMap((r, i) => (i > 0 && r.corruption !== rb[i - 1].corruption ? [[i - 0.5, null], [i, r[key]]] : [[i, r[key]]]));
$("#recall-note").textContent = R.robustness_note;
lineChart($("#c-recall"), {
  size: { width: 640, height: 300, margin: { t: 16, r: 110, b: 64, l: 52 } }, markers: true, directLabels: true,
  series: [
    { name: "Real-image recall", short: "real recall", color: "--s2", points: withGaps("real_recall") },
    { name: "AI-image recall", short: "AI recall", color: "--s1", points: withGaps("ai_recall") },
  ],
  xDomain: [-0.4, rb.length - 0.6], yDomain: [0.5, 1], xCats: conds.map((c) => c.replace("downscale", "scale")), yTicks: [0.5, 0.6, 0.7, 0.8, 0.9, 1],
  yFmt: (v) => v.toFixed(1), xLabel: `${R.deployment.label}, test split`, tipX: (v) => conds[Math.max(0, Math.min(conds.length - 1, Math.round(v)))],
  tipFmt: (p) => pct(p[1]),
});
document.querySelectorAll("#c-recall text.tick").forEach((t) => { if (+t.getAttribute("y") > 250) { t.setAttribute("transform", `rotate(-35 ${t.getAttribute("x")} ${t.getAttribute("y")})`); t.setAttribute("text-anchor", "end"); } });

// ---- training curves
lineChart($("#c-train"), {
  size: { width: 640, height: 320, margin: { t: 16, r: 110, b: 44, l: 52 } }, directLabels: true,
  series: M.map((m, i) => ({ name: m.label, short: m.label, color: COLORS[m.run], labelDy: [-6, 8, 0][i],
    points: R.dashboard[m.run].history.epoch.map((e, j) => [e, R.dashboard[m.run].history.val_roc_auc[j]]) })),
  xDomain: [1, 60], yDomain: [0.6, 1], xTicks: [1, 10, 20, 30, 40, 50, 60], yTicks: [0.6, 0.7, 0.8, 0.9, 1], yFmt: (v) => v.toFixed(1),
  xLabel: "Epoch", yLabel: "Validation ROC-AUC", tipX: (v) => `epoch ${Math.round(v)}`, tipFmt: (p) => `${p[1].toFixed(3)} (epoch ${p[0]})`,
});

// ---- confusion matrix
const cm = dep.confusion || R.dashboard[dep.run].confusion;
const n = cm.tn + cm.fp + cm.fn + cm.tp;
const cell = (v, ok) => `<td class="cm ${ok ? "ok" : "bad"}"><b>${v}</b><span>${pct(v / (n / 2))} of row</span></td>`;
$("#c-cm").innerHTML = `<table class="cm-table"><thead><tr><th></th><th>Predicted real</th><th>Predicted AI</th></tr></thead><tbody>
<tr><th>Actually real (150)</th>${cell(cm.tn, true)}${cell(cm.fp, false)}</tr>
<tr><th>Actually AI (150)</th>${cell(cm.fn, false)}${cell(cm.tp, true)}</tr></tbody></table>
<p class="note">${cm.fp} real photos were flagged as AI and ${cm.fn} AI images were passed as real, out of ${n} test images.</p>`;
