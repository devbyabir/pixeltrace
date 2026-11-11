// PixelTrace browser preprocessing: a port of pixeltrace.preprocess.normalise + the eval resize.
//
// Python pipeline (src/pixeltrace/preprocess.py + export_onnx.pixel_tensor):
//   EXIF transpose -> flatten alpha on white -> ImageOps.fit(512, LANCZOS) -> JPEG q95 round trip
//   -> resize(224, BILINEAR) -> float32 CHW in [0, 1]
//
// The two resizes are a line-by-line port of Pillow's fixed-point resampler (libImaging/Resample.c,
// 8-bit path), so for identical input pixels the output is bit-identical (tested in tools/parity).
// The JPEG round trip uses the browser's encoder/decoder, which may differ slightly from libjpeg in
// Pillow; tools/parity measures that difference.

const PRECISION_BITS = 32 - 8 - 2;

function sinc(x) {
  if (x === 0.0) return 1.0;
  x *= Math.PI;
  return Math.sin(x) / x;
}
const FILTERS = {
  bilinear: { support: 1.0, fn: (x) => { x = Math.abs(x); return x < 1.0 ? 1.0 - x : 0.0; } },
  lanczos: { support: 3.0, fn: (x) => (x >= -3.0 && x < 3.0 ? sinc(x) * sinc(x / 3.0) : 0.0) },
};

// Pillow precompute_coeffs + normalize_coeffs_8bpc
function coeffs(inSize, in0, in1, outSize, filter) {
  const scale = (in1 - in0) / outSize;
  const filterscale = Math.max(scale, 1.0);
  const support = filter.support * filterscale;
  const ksize = Math.ceil(support) * 2 + 1;
  const kk = new Int32Array(outSize * ksize);
  const bounds = new Int32Array(outSize * 2);
  const k = new Float64Array(ksize);
  const ss = 1.0 / filterscale;
  for (let xx = 0; xx < outSize; xx++) {
    const center = in0 + (xx + 0.5) * scale;
    let xmin = Math.trunc(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    let xmax = Math.trunc(center + support + 0.5);
    if (xmax > inSize) xmax = inSize;
    xmax -= xmin;
    let ww = 0.0;
    for (let x = 0; x < xmax; x++) {
      const w = filter.fn((x + xmin - center + 0.5) * ss);
      k[x] = w;
      ww += w;
    }
    for (let x = 0; x < ksize; x++) {
      let v = x < xmax && ww !== 0.0 ? k[x] / ww : x < xmax ? k[x] : 0.0;
      // (int) cast in C truncates toward zero
      kk[xx * ksize + x] = v < 0 ? Math.trunc(-0.5 + v * (1 << PRECISION_BITS))
                                 : Math.trunc(0.5 + v * (1 << PRECISION_BITS));
    }
    bounds[xx * 2] = xmin;
    bounds[xx * 2 + 1] = xmax;
  }
  return { ksize, kk, bounds };
}

function clip8(v) {
  // Pillow: if (in >= (1 << PRECISION_BITS << 8)) return 255; if (in <= 0) return 0; return in >> PRECISION_BITS;
  if (v >= 2 ** (PRECISION_BITS + 8)) return 255;
  if (v <= 0) return 0;
  return Math.floor(v / 2 ** PRECISION_BITS);
}

/**
 * Resize an RGB image (Uint8Array, length w*h*3) like PIL.Image.resize(size, filter, box).
 * box = [x0, y0, x1, y1] in source pixels (floats allowed). Returns {data, width, height}.
 */
export function resampleRGB(src, w, h, outW, outH, filterName, box = [0, 0, w, h]) {
  const filter = FILTERS[filterName];
  const needH = outW !== w || box[0] !== 0 || box[2] !== outW;
  const needV = outH !== h || box[1] !== 0 || box[3] !== outH;
  const H = coeffs(w, box[0], box[2], outW, filter);
  const V = coeffs(h, box[1], box[3], outH, filter);

  // Pillow 13 (Resample.c, ImagingResampleInner): "If height is being scaled down more than twice
  // the amount the width is, run the vertical pass first". In that mode the vertical pass covers the
  // full input width and the horizontal bounds are not shifted.
  const horizontalFirst = !((h - outH) > 0 && (h - outH) > (w - outW) * 2);
  if (!horizontalFirst) {
    return verticalFirst(src, w, h, outW, outH, H, V, needH, needV);
  }

  let cur = src, curW = w, curH = h, yOffset = 0;
  const yFirst = V.bounds[0];
  const yLast = V.bounds[outH * 2 - 2] + V.bounds[outH * 2 - 1];

  if (needH) {
    const rows = yLast - yFirst;
    const tmp = new Uint8Array(outW * rows * 3);
    const half = 1 << (PRECISION_BITS - 1);
    for (let yy = 0; yy < rows; yy++) {
      const srow = (yy + yFirst) * w * 3;
      for (let xx = 0; xx < outW; xx++) {
        const xmin = H.bounds[xx * 2], xmax = H.bounds[xx * 2 + 1], base = xx * H.ksize;
        let s0 = half, s1 = half, s2 = half;
        for (let x = 0; x < xmax; x++) {
          const c = H.kk[base + x], p = srow + (x + xmin) * 3;
          s0 += cur[p] * c; s1 += cur[p + 1] * c; s2 += cur[p + 2] * c;
        }
        const o = (yy * outW + xx) * 3;
        tmp[o] = clip8(s0); tmp[o + 1] = clip8(s1); tmp[o + 2] = clip8(s2);
      }
    }
    cur = tmp; curW = outW; curH = rows; yOffset = yFirst;
  }
  if (needV) {
    const out = new Uint8Array(curW * outH * 3);
    const half = 1 << (PRECISION_BITS - 1);
    for (let yy = 0; yy < outH; yy++) {
      const ymin = V.bounds[yy * 2] - (needH ? yOffset : 0), ymax = V.bounds[yy * 2 + 1], base = yy * V.ksize;
      for (let xx = 0; xx < curW; xx++) {
        let s0 = half, s1 = half, s2 = half;
        for (let y = 0; y < ymax; y++) {
          const c = V.kk[base + y], p = ((y + ymin) * curW + xx) * 3;
          s0 += cur[p] * c; s1 += cur[p + 1] * c; s2 += cur[p + 2] * c;
        }
        const o = (yy * curW + xx) * 3;
        out[o] = clip8(s0); out[o + 1] = clip8(s1); out[o + 2] = clip8(s2);
      }
    }
    return { data: out, width: curW, height: outH };
  }
  if (!needH) return { data: src.slice(), width: w, height: h };
  return { data: cur, width: curW, height: curH };
}

function verticalFirst(src, w, h, outW, outH, H, V, needH, needV) {
  const half = 1 << (PRECISION_BITS - 1);
  let cur = src, curH = h;
  if (needV) {
    const tmp = new Uint8Array(w * outH * 3);
    for (let yy = 0; yy < outH; yy++) {
      const ymin = V.bounds[yy * 2], ymax = V.bounds[yy * 2 + 1], base = yy * V.ksize;
      for (let xx = 0; xx < w; xx++) {
        let s0 = half, s1 = half, s2 = half;
        for (let y = 0; y < ymax; y++) {
          const c = V.kk[base + y], p = ((y + ymin) * w + xx) * 3;
          s0 += src[p] * c; s1 += src[p + 1] * c; s2 += src[p + 2] * c;
        }
        const o = (yy * w + xx) * 3;
        tmp[o] = clip8(s0); tmp[o + 1] = clip8(s1); tmp[o + 2] = clip8(s2);
      }
    }
    cur = tmp; curH = outH;
  }
  if (!needH) return { data: needV ? cur : src.slice(), width: w, height: curH };
  const out = new Uint8Array(outW * curH * 3);
  for (let yy = 0; yy < curH; yy++) {
    for (let xx = 0; xx < outW; xx++) {
      const xmin = H.bounds[xx * 2], xmax = H.bounds[xx * 2 + 1], base = xx * H.ksize;
      let s0 = half, s1 = half, s2 = half;
      for (let x = 0; x < xmax; x++) {
        const c = H.kk[base + x], p = (yy * w + x + xmin) * 3;
        s0 += cur[p] * c; s1 += cur[p + 1] * c; s2 += cur[p + 2] * c;
      }
      const o = (yy * outW + xx) * 3;
      out[o] = clip8(s0); out[o + 1] = clip8(s1); out[o + 2] = clip8(s2);
    }
  }
  return { data: out, width: outW, height: curH };
}

/** PIL ImageOps.fit crop box for a square output (centering 0.5, no bleed). */
export function fitBox(w, h, size = 512) {
  const ratio = w / h, out = size / size;
  let cw, ch;
  if (ratio === out) { cw = w; ch = h; }
  else if (ratio >= out) { cw = out * h; ch = h; }
  else { cw = w; ch = w / out; }
  const left = (w - cw) * 0.5, top = (h - ch) * 0.5;
  return [left, top, left + cw, top + ch];
}

/** RGBA (from canvas) -> RGB flattened on white, using Pillow's paste-with-mask rounding. */
export function flattenOnWhite(rgba, w, h) {
  const out = new Uint8Array(w * h * 3);
  let anyAlpha = false;
  for (let i = 0; i < w * h; i++) if (rgba[i * 4 + 3] !== 255) { anyAlpha = true; break; }
  for (let i = 0, j = 0; i < w * h; i++, j += 3) {
    const a = rgba[i * 4 + 3];
    for (let c = 0; c < 3; c++) {
      const v = rgba[i * 4 + c];
      if (!anyAlpha || a === 255) out[j + c] = v;
      else {
        // DIV255(bg*(255-a) + v*a) with bg = 255
        const t = 255 * (255 - a) + v * a + 128;
        out[j + c] = ((t >> 8) + t) >> 8;
      }
    }
  }
  return out;
}

/** Pixel array (RGB uint8, 224x224) -> Float32Array CHW in [0,1]. */
export function toCHW(rgb, w, h) {
  const n = w * h, out = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    out[i] = rgb[i * 3] / 255;
    out[n + i] = rgb[i * 3 + 1] / 255;
    out[2 * n + i] = rgb[i * 3 + 2] / 255;
  }
  return out;
}

// ----------------------------------------------------------------- browser-only helpers
function rgbToImageData(rgb, w, h) {
  const d = new ImageData(w, h);
  for (let i = 0, j = 0; i < w * h; i++, j += 3) {
    d.data[i * 4] = rgb[j]; d.data[i * 4 + 1] = rgb[j + 1]; d.data[i * 4 + 2] = rgb[j + 2]; d.data[i * 4 + 3] = 255;
  }
  return d;
}

async function decodeToRGBA(blob) {
  // imageOrientation 'from-image' applies EXIF rotation (like ImageOps.exif_transpose);
  // colorSpaceConversion 'none' and premultiplyAlpha 'none' keep the stored pixel values (Pillow ignores ICC).
  const bmp = await createImageBitmap(blob, { imageOrientation: "from-image", premultiplyAlpha: "none",
                                              colorSpaceConversion: "none" });
  const canvas = new OffscreenCanvas(bmp.width, bmp.height);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bmp, 0, 0);
  const { data } = ctx.getImageData(0, 0, bmp.width, bmp.height);
  const out = { data, width: bmp.width, height: bmp.height };
  bmp.close();
  return out;
}

async function jpegRoundTrip(rgb, w, h, quality = 0.95) {
  const canvas = new OffscreenCanvas(w, h);
  canvas.getContext("2d").putImageData(rgbToImageData(rgb, w, h), 0, 0);
  const blob = await canvas.convertToBlob({ type: "image/jpeg", quality });
  const dec = await decodeToRGBA(blob);
  const out = new Uint8Array(w * h * 3);
  for (let i = 0; i < w * h; i++) { out[i * 3] = dec.data[i * 4]; out[i * 3 + 1] = dec.data[i * 4 + 1]; out[i * 3 + 2] = dec.data[i * 4 + 2]; }
  return out;
}

export const MAX_PIXELS = 60e6; // refuse very large images instead of exhausting browser memory

/** File/Blob -> { tensor: Float32Array(3*size*size), preview: ImageData(512x512) } */
export async function preprocess(blob, size = 224, square = 512) {
  const img = await decodeToRGBA(blob);
  if (img.width * img.height > MAX_PIXELS) {
    throw new Error(`Image is ${(img.width * img.height / 1e6).toFixed(0)} MP; the browser demo accepts up to ${MAX_PIXELS / 1e6} MP.`);
  }
  const rgb = flattenOnWhite(img.data, img.width, img.height);
  const sq = resampleRGB(rgb, img.width, img.height, square, square, "lanczos", fitBox(img.width, img.height, square));
  const jpg = await jpegRoundTrip(sq.data, square, square, 0.95);
  const small = resampleRGB(jpg, square, square, size, size, "bilinear");
  return { tensor: toCHW(small.data, size, size), squarePreJpeg: sq.data, squareRGB: jpg, smallRGB: small.data };
}
