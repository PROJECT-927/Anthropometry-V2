"""Memory bounds shared by voxel reconstruction and mesh export."""

import math
import os

import numpy as np


class GridMemoryError(ValueError):
    """A grid would exceed the configured cell budget."""


def check_grid_size(shape, stage="voxel grid"):
    cells = math.prod(int(size) for size in shape)
    limit = int(os.environ.get("MAX_GRID_VOXELS", "400000000"))
    if limit <= 0:
        raise ValueError("MAX_GRID_VOXELS must be positive.")
    if cells > limit:
        raise GridMemoryError(
            f"{stage} needs {cells:,} cells (limit {limit:,}). "
            "Check silhouette framing and height, or increase MAX_GRID_VOXELS "
            "only after measuring available server memory. Resolution was not changed."
        )
    print(f"[grid] {stage}: {tuple(shape)}, {cells:,} cells, "
          f"{cells / 2**20:.1f} MiB per boolean grid", flush=True)


def component_counts(labels, n_labels, chunk_cells=1_000_000):
    """Count labels without a full-volume int64/float64 bincount conversion."""
    flat = labels.reshape(-1)
    counts = np.zeros(n_labels + 1, dtype=np.int64)
    for start in range(0, flat.size, chunk_cells):
        counts += np.bincount(flat[start:start + chunk_cells], minlength=n_labels + 1)
    counts[0] = 0
    return counts
