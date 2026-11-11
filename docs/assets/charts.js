// Minimal SVG charts for the PixelTrace dashboard (no dependencies).
// Thin marks, recessive grid, legends + direct labels, hover tooltips, colours from CSS custom properties.

const NS = "http://www.w3.org/2000/svg";
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const el = (tag, attrs = {}, parent) => {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
};

let tip;
function tooltip() {
  if (!tip) {
    tip = document.createElement("div");
    tip.className = "viz-tip";
    tip.setAttribute("role", "status");
    document.body.appendChild(tip);
  }
  return tip;
}
function showTip(html, evt) {
  const t = tooltip();
  t.innerHTML = html;
  t.style.display = "block";
  const pad = 12, r = t.getBoundingClientRect();
  let x = evt.clientX + pad, y = evt.clientY + pad;
  if (x + r.width > innerWidth - 8) x = evt.clientX - r.width - pad;
  if (y + r.height > innerHeight - 8) y = evt.clientY - r.height - pad;
  t.style.left = `${x}px`; t.style.top = `${y}px`;
}
const hideTip = () => tip && (tip.style.display = "none");

function frame(container, { width = 640, height = 300, margin = { t: 16, r: 16, b: 40, l: 48 } } = {}) {
  container.innerHTML = "";
  const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", class: "viz" }, container);
  return { svg, width, height, m: margin, iw: width - margin.l - margin.r, ih: height - margin.t - margin.b };
}

function axes(f, { x, y, xTicks, yTicks, xFmt = String, yFmt = String, xLabel, yLabel, xCats }) {
  const g = el("g", {}, f.svg);
  const grid = css("--grid"), muted = css("--muted");
  for (const t of yTicks) {
    el("line", { x1: f.m.l, x2: f.m.l + f.iw, y1: y(t), y2: y(t), stroke: grid, "stroke-width": 1 }, g);
    el("text", { x: f.m.l - 6, y: y(t) + 4, "text-anchor": "end", class: "tick" }, g).textContent = yFmt(t);
  }
  const xs = xCats ? xCats.map((c, i) => [i, c]) : xTicks.map((t) => [t, xFmt(t)]);
  for (const [t, lab] of xs) {
    el("text", { x: x(t), y: f.m.t + f.ih + 16, "text-anchor": "middle", class: "tick" }, g).textContent = lab;
  }
  el("line", { x1: f.m.l, x2: f.m.l + f.iw, y1: f.m.t + f.ih, y2: f.m.t + f.ih, stroke: muted, "stroke-width": 1 }, g);
  if (xLabel) el("text", { x: f.m.l + f.iw / 2, y: f.height - 6, "text-anchor": "middle", class: "axis-label" }, g).textContent = xLabel;
  if (yLabel) el("text", { x: 12, y: f.m.t + f.ih / 2, transform: `rotate(-90 12 ${f.m.t + f.ih / 2})`, "text-anchor": "middle", class: "axis-label" }, g).textContent = yLabel;
}

export function legend(container, series) {
  const l = document.createElement("div");
  l.className = "viz-legend";
  for (const s of series) {
    const i = document.createElement("span");
    i.innerHTML = `<i style="background:${css(s.color)}"></i>${s.name}`;
    l.appendChild(i);
  }
  container.prepend(l);
}

/** Line chart. series: [{name, color: "--s1", points: [[x, y], ...]}]. Nearest-x crosshair tooltip. */
export function lineChart(container, o) {
  const f = frame(container, o.size);
  const [x0, x1] = o.xDomain, [y0, y1] = o.yDomain;
  const x = (v) => f.m.l + ((v - x0) / (x1 - x0)) * f.iw;
  const y = (v) => f.m.t + (1 - (v - y0) / (y1 - y0)) * f.ih;
  axes(f, { ...o, x, y });
  if (o.diagonal) el("line", { x1: x(x0), y1: y(y0), x2: x(x1), y2: y(y1), stroke: css("--muted"), "stroke-dasharray": "4 4", "stroke-width": 1 }, f.svg);
  for (const r of o.refLines || []) {
    el("line", { x1: f.m.l, x2: f.m.l + f.iw, y1: y(r.value), y2: y(r.value), stroke: css("--muted"), "stroke-dasharray": "3 3" }, f.svg);
    el("text", { x: f.m.l + f.iw - 4, y: y(r.value) - 4, "text-anchor": "end", class: "tick" }, f.svg).textContent = r.label;
  }
  const surf = css("--card");
  for (const s of o.series) {
    let pen = "M";
    const d = s.points.map((p) => {
      if (p[1] == null) { pen = "M"; return ""; }
      const seg = `${pen}${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`; pen = "L"; return seg;
    }).join("");
    el("path", { d, fill: "none", stroke: css(s.color), "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, f.svg);
    if (o.markers) for (const p of s.points.filter((q) => q[1] != null)) el("circle", { cx: x(p[0]), cy: y(p[1]), r: 4, fill: css(s.color), stroke: surf, "stroke-width": 2 }, f.svg);
    if (o.directLabels) {
      const p = s.points.filter((q) => q[1] != null).slice(-1)[0];
      el("text", { x: x(p[0]) + 6, y: y(p[1]) + (s.labelDy || 0) + 4, class: "dlabel" }, f.svg).textContent = s.short || s.name;
    }
  }
  // hover layer
  const cross = el("line", { y1: f.m.t, y2: f.m.t + f.ih, stroke: css("--muted"), "stroke-width": 1, visibility: "hidden" }, f.svg);
  const dots = o.series.map((s) => el("circle", { r: 5, fill: css(s.color), stroke: surf, "stroke-width": 2, visibility: "hidden" }, f.svg));
  const hit = el("rect", { x: f.m.l, y: f.m.t, width: f.iw, height: f.ih, fill: "transparent" }, f.svg);
  hit.addEventListener("pointermove", (e) => {
    const pt = f.svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
    const loc = pt.matrixTransform(f.svg.getScreenCTM().inverse());
    const xv = x0 + ((loc.x - f.m.l) / f.iw) * (x1 - x0);
    let html = `<b>${o.tipX ? o.tipX(xv, o) : (o.xFmt || String)(xv)}</b>`;
    o.series.forEach((s, i) => {
      const pts = s.points.filter((q) => q[1] != null);
      let best = pts[0];
      for (const p of pts) if (Math.abs(p[0] - xv) < Math.abs(best[0] - xv)) best = p;
      dots[i].setAttribute("cx", x(best[0])); dots[i].setAttribute("cy", y(best[1])); dots[i].setAttribute("visibility", "visible");
      html += `<br><i style="background:${css(s.color)}"></i>${s.name}: ${o.tipFmt ? o.tipFmt(best, s) : best[1].toFixed(3)}`;
    });
    cross.setAttribute("x1", loc.x); cross.setAttribute("x2", loc.x); cross.setAttribute("visibility", "visible");
    showTip(html, e);
  });
  hit.addEventListener("pointerleave", () => { hideTip(); cross.setAttribute("visibility", "hidden"); dots.forEach((d) => d.setAttribute("visibility", "hidden")); });
  if (o.series.length > 1 && !o.noLegend) legend(container, o.series);
}

/** Dot + 95 % CI interval rows. rows: [{label, color, value, lo, hi, marker?: {value, label}}] */
export function intervalChart(container, o) {
  const rowH = 46, f = frame(container, { width: 640, height: o.rows.length * rowH + 50, margin: { t: 10, r: 24, b: 40, l: 150 } });
  const [d0, d1] = o.domain;
  const x = (v) => f.m.l + ((v - d0) / (d1 - d0)) * f.iw;
  const y = (i) => f.m.t + i * rowH + rowH / 2;
  const grid = css("--grid"), muted = css("--muted"), surf = css("--card");
  for (const t of o.ticks) {
    el("line", { x1: x(t), x2: x(t), y1: f.m.t, y2: f.m.t + f.ih, stroke: grid }, f.svg);
    el("text", { x: x(t), y: f.m.t + f.ih + 16, "text-anchor": "middle", class: "tick" }, f.svg).textContent = o.fmt(t);
  }
  el("text", { x: f.m.l + f.iw / 2, y: f.height - 6, "text-anchor": "middle", class: "axis-label" }, f.svg).textContent = o.xLabel;
  for (const r of o.refLines || []) {
    el("line", { x1: x(r.value), x2: x(r.value), y1: f.m.t, y2: f.m.t + f.ih, stroke: muted, "stroke-dasharray": "3 3" }, f.svg);
    el("text", { x: x(r.value) - 4, y: f.m.t + 10, "text-anchor": "end", class: "tick" }, f.svg).textContent = r.label;
  }
  o.rows.forEach((r, i) => {
    const c = css(r.color);
    el("text", { x: f.m.l - 10, y: y(i) + 4, "text-anchor": "end", class: "rowlabel" }, f.svg).textContent = r.label;
    el("line", { x1: x(r.lo), x2: x(r.hi), y1: y(i), y2: y(i), stroke: c, "stroke-width": 2, "stroke-linecap": "round" }, f.svg);
    for (const v of [r.lo, r.hi]) el("line", { x1: x(v), x2: x(v), y1: y(i) - 6, y2: y(i) + 6, stroke: c, "stroke-width": 2, "stroke-linecap": "round" }, f.svg);
    if (r.marker) el("circle", { cx: x(r.marker.value), cy: y(i), r: 5, fill: surf, stroke: c, "stroke-width": 2 }, f.svg);
    el("circle", { cx: x(r.value), cy: y(i), r: 6, fill: c, stroke: surf, "stroke-width": 2 }, f.svg);
    el("text", { x: x(Math.max(r.hi, r.marker ? r.marker.value : -Infinity)) + 10, y: y(i) + 4, class: "dlabel" }, f.svg).textContent = o.fmt(r.value);
    const hit = el("rect", { x: f.m.l, y: y(i) - rowH / 2, width: f.iw, height: rowH, fill: "transparent" }, f.svg);
    hit.addEventListener("pointermove", (e) => showTip(`<b>${r.label}</b><br>${o.valueName}: ${o.fmt(r.value)}<br>95 % CI: ${o.fmt(r.lo)}–${o.fmt(r.hi)}` +
      (r.marker ? `<br>${r.marker.label}: ${o.fmt(r.marker.value)}` : ""), e));
    hit.addEventListener("pointerleave", hideTip);
  });
}

/** Horizontal bars (one value per row) with labels at the bar end. */
export function barChart(container, o) {
  const rowH = 40, f = frame(container, { width: 640, height: o.rows.length * rowH + 44, margin: { t: 8, r: 64, b: 36, l: 230 } });
  const x = (v) => f.m.l + (v / o.max) * f.iw;
  const grid = css("--grid");
  for (const t of o.ticks) {
    el("line", { x1: x(t), x2: x(t), y1: f.m.t, y2: f.m.t + f.ih, stroke: grid }, f.svg);
    el("text", { x: x(t), y: f.m.t + f.ih + 16, "text-anchor": "middle", class: "tick" }, f.svg).textContent = o.fmt(t);
  }
  o.rows.forEach((r, i) => {
    const yy = f.m.t + i * rowH + 8, h = rowH - 16;
    el("text", { x: f.m.l - 10, y: yy + h / 2 + 4, "text-anchor": "end", class: "rowlabel" }, f.svg).textContent = r.label;
    const w = Math.max(2, x(r.value) - f.m.l);
    el("path", { d: `M${f.m.l},${yy}h${w - 4}a4,4 0 0 1 4,4v${h - 8}a4,4 0 0 1 -4,4h${-(w - 4)}z`, fill: css(r.color) }, f.svg);
    el("text", { x: f.m.l + w + 6, y: yy + h / 2 + 4, class: "dlabel" }, f.svg).textContent = r.text || o.fmt(r.value);
    const hit = el("rect", { x: f.m.l, y: yy - 6, width: f.iw, height: rowH, fill: "transparent" }, f.svg);
    hit.addEventListener("pointermove", (e) => showTip(`<b>${r.label}</b><br>${r.tip || o.fmt(r.value)}`, e));
    hit.addEventListener("pointerleave", hideTip);
  });
}

/** Plain HTML table for the "view as table" disclosure. */
export function table(container, head, rows, caption) {
  const d = document.createElement("details");
  d.className = "viz-table";
  d.innerHTML = `<summary>View as table</summary>`;
  const t = document.createElement("table");
  if (caption) t.createCaption().textContent = caption;
  t.innerHTML += `<thead><tr>${head.map((h) => `<th>${h}</th>`).join("")}</tr></thead>` +
    `<tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody>`;
  d.appendChild(t);
  container.appendChild(d);
}
