// Bit-exactness test of the JS Pillow-resampler port against Pillow reference vectors.
//   node tools/parity/test_resample.mjs dist/parity
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { resampleRGB, fitBox } from "../../docs/assets/preprocess.js";

const dir = process.argv[2] || "dist/parity";
const index = JSON.parse(readFileSync(join(dir, "index.json"), "utf8"));
const load = (n, k) => new Uint8Array(readFileSync(join(dir, `${n}.${k}.bin`)));

function compare(a, b) {
  let maxd = 0, diff = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d) { diff++; if (d > maxd) maxd = d; }
  }
  return { maxd, diff, n: a.length };
}

let fail = 0;
for (const s of index) {
  const rgb = load(s.name, "rgb");
  const sq = resampleRGB(rgb, s.w, s.h, 512, 512, "lanczos", fitBox(s.w, s.h, 512));
  const c1 = compare(sq.data, load(s.name, "sq"));
  const small = resampleRGB(load(s.name, "jpg"), 512, 512, 224, 224, "bilinear");
  const c2 = compare(small.data, load(s.name, "small"));
  const ok = c1.diff === 0 && c2.diff === 0;
  if (!ok) fail++;
  console.log(`${ok ? "PASS" : "FAIL"} ${s.name.padEnd(28)} ${s.w}x${s.h}  ` +
    `lanczos-fit: ${c1.diff} px-values differ (max ${c1.maxd})  bilinear-224: ${c2.diff} differ (max ${c2.maxd})`);
}
console.log(fail ? `\n${fail} sample(s) FAILED` : "\nALL BIT-EXACT");
process.exit(fail ? 1 : 0);
