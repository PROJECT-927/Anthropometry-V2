/* Binary silhouette preprocessing shared by browser capture and regression tests. */
(function (root) {
  "use strict";

  function resampleProbability(values, maskWidth, maskHeight, width, height, paddedWidth = width) {
    // Match half-pixel bilinear resizing before thresholding, then remove the
    // right-hand alignment padding. Nearest-neighbour thresholded resizing can
    // introduce gaps that break row-wise torso tracing.
    const mask = new Uint8Array(width * height);
    for (let y = 0; y < height; y++) {
      const fy = Math.max(0, Math.min(maskHeight - 1, (y + 0.5) * maskHeight / height - 0.5));
      const y0 = Math.floor(fy), y1 = Math.min(y0 + 1, maskHeight - 1), ty = fy - y0;
      for (let x = 0; x < width; x++) {
        const fx = Math.max(0, Math.min(maskWidth - 1, (x + 0.5) * maskWidth / paddedWidth - 0.5));
        const x0 = Math.floor(fx), x1 = Math.min(x0 + 1, maskWidth - 1), tx = fx - x0;
        const top = values[y0 * maskWidth + x0] * (1 - tx) + values[y0 * maskWidth + x1] * tx;
        const bottom = values[y1 * maskWidth + x0] * (1 - tx) + values[y1 * maskWidth + x1] * tx;
        mask[y * width + x] = top * (1 - ty) + bottom * ty > 0.5 ? 1 : 0;
      }
    }
    return mask;
  }

  function cleanSilhouette(input, width, height) {
    const size = width * height;
    if (input.length !== size || size === 0) throw new Error("Invalid silhouette dimensions.");
    const mask = new Uint8Array(input);
    const labels = new Int32Array(size);
    const queue = new Int32Array(size);

    function flood(start, value, label) {
      let head = 0, tail = 1, border = false;
      queue[0] = start;
      labels[start] = label;
      while (head < tail) {
        const index = queue[head++], y = Math.floor(index / width), x = index - y * width;
        if (x === 0 || x === width - 1 || y === 0 || y === height - 1) border = true;
        for (let yy = Math.max(0, y - 1); yy <= Math.min(height - 1, y + 1); yy++) {
          for (let xx = Math.max(0, x - 1); xx <= Math.min(width - 1, x + 1); xx++) {
            const next = yy * width + xx;
            if (!labels[next] && mask[next] === value) {
              labels[next] = label;
              queue[tail++] = next;
            }
          }
        }
      }
      return { area: tail, border };
    }

    let label = 0, largest = 0, foreground = 0;
    for (let i = 0; i < size; i++) {
      if (mask[i] && !labels[i]) {
        const component = flood(i, 1, ++label);
        if (component.area > foreground) {
          foreground = component.area;
          largest = label;
        }
      }
    }
    if (!foreground) throw new Error("The silhouette contains no person.");
    for (let i = 0; i < size; i++) mask[i] = labels[i] === largest ? 1 : 0;

    // Fill small enclosed holes only. Preserve exterior gaps, including the
    // gap between legs, exactly as the original Python preprocessing does.
    labels.fill(0);
    for (let i = 0; i < size; i++) {
      if (!mask[i] && !labels[i]) {
        const component = flood(i, 0, 1);
        if (!component.border && component.area < 0.02 * foreground) {
          for (let j = 0; j < component.area; j++) mask[queue[j]] = 1;
        }
      }
    }
    let left = width, right = -1, top = height, bottom = -1;
    for (let i = 0; i < size; i++) {
      if (!mask[i]) continue;
      const y = Math.floor(i / width), x = i - y * width;
      left = Math.min(left, x); right = Math.max(right, x);
      top = Math.min(top, y); bottom = Math.max(bottom, y);
    }
    const marginX = Math.floor((right - left) * 0.15);
    const marginY = Math.floor((bottom - top) * 0.15);
    left = Math.max(0, left - marginX); right = Math.min(width - 1, right + marginX);
    top = Math.max(0, top - marginY); bottom = Math.min(height - 1, bottom + marginY);
    return { mask, crop: { x: left, y: top, width: right - left + 1, height: bottom - top + 1 } };
  }

  root.AnthropometryPreprocessing = Object.freeze({ resampleProbability, cleanSilhouette });
})(globalThis);
