// Shared footer: ownership notice. Honest about what "all rights reserved" can and cannot do for a public web model.
export function footer(container) {
  container.innerHTML = `
  <p><strong>© 2026 Abir Ahmmed (devbyabir). All rights reserved.</strong> The PixelTrace code and the trained model are not
  released under an open-source licence; no permission to reuse, redistribute or build on them is granted.</p>
  <p>Because the demo runs in your browser, the model file is necessarily downloaded to your device and can technically be
  copied. This notice states the author's rights; it does not and cannot technically prevent that. The training images are
  not part of this site or repository (their source and licence are unknown), and no image you try here leaves your device.</p>
  <p>Built with PyTorch, timm and ONNX Runtime Web. <a href="https://github.com/devbyabir/pixeltrace">Source code</a>.</p>`;
}
