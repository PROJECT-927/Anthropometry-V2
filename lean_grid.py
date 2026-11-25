"""
Compact stand-ins for the dense voxel grids. Same voxels, far less memory.

RectGrid:     the two-silhouette hull is, at every height, (front runs) x (side runs):
              a short list of axis-aligned rectangles. No 3D array is needed.
IntervalGrid: the profile carve gives every (x, y) column exactly one Z interval.

Both convert to the full boolean array with np.asarray() and hand out 2D
slices with the usual indexing, so code that reads a slice is unchanged.
"""
import cv2
import numpy as np
from scipy import ndimage


class _UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        p = self.p
        while p[a] != a:
            p[a] = p[p[a]]
            a = p[a]
        return a

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)


def runs(row):
    d = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1).tolist(), (np.flatnonzero(d == -1) - 1).tolist()))


class RectGrid:
    """rows[iy] = list of (x0, x1, z0, z1), inclusive, in scan order."""

    def __init__(self, shape, rows, edited=None):
        self.shape, self.rows, self.edited = tuple(shape), rows, edited or {}
        self.size = int(np.prod(self.shape, dtype=np.int64))
        self.dtype = np.dtype(bool)

    @classmethod
    def from_tables(cls, occ_f, occ_s):
        nx, ny = occ_f.shape
        rows = []
        for iy in range(ny):
            rx, rz = runs(occ_f[:, iy]), runs(occ_s[iy])
            rows.append([(a, b, c, d) for a, b in rx for c, d in rz])
        return cls((nx, ny, occ_s.shape[1]), rows)

    def row(self, iy):
        if iy in self.edited:
            return self.edited[iy].copy()
        out = np.zeros((self.shape[0], self.shape[2]), dtype=bool)
        for a, b, c, d in self.rows[iy]:
            out[a:b + 1, c:d + 1] = True
        return out

    def __getitem__(self, key):
        if (isinstance(key, tuple) and len(key) == 3 and key[0] == slice(None)
                and isinstance(key[1], (int, np.integer)) and key[2] == slice(None)):
            return self.row(int(key[1]))
        return np.asarray(self)[key]

    def __setitem__(self, key, value):
        sx, iy, iz = key
        iy = int(iy)
        if iy not in self.edited:
            self.edited[iy] = self.row(iy)
        self.edited[iy][sx, iz] = value

    def copy(self):
        return RectGrid(self.shape, list(self.rows), {k: v.copy() for k, v in self.edited.items()})

    def sum(self):
        n = 0
        for iy, rects in enumerate(self.rows):
            n += int(self.edited[iy].sum()) if iy in self.edited else sum(
                (b - a + 1) * (d - c + 1) for a, b, c, d in rects)
        return n

    def any(self, axis=None):
        if axis is None:
            return any(self.rows)
        if axis == 1:
            out = np.zeros((self.shape[0], self.shape[2]), bool)
            for iy in range(self.shape[1]):
                out |= self.row(iy)
            return out
        lines = [self.row(iy).any(axis=1 if axis == 2 else 0) for iy in range(self.shape[1])]
        return np.stack(lines, axis=1 if axis == 2 else 0)

    def __array__(self, dtype=None, copy=None):
        out = np.zeros(self.shape, dtype=bool)
        for iy in range(self.shape[1]):
            out[:, iy, :] = self.row(iy)
        return out if dtype is None else out.astype(dtype)

    def keep_largest(self):
        """Largest 26-connected component: rectangles in adjacent rows touch iff
        their X and Z ranges come within one voxel of each other."""
        assert not self.edited, "keep_largest() runs before any row edits"
        start, total = [], 0
        for r in self.rows:
            start.append(total)
            total += len(r)
        uf = _UnionFind(total)
        for iy in range(1, len(self.rows)):
            for i, (a, b, c, d) in enumerate(self.rows[iy - 1]):
                for j, (e, f, g, h) in enumerate(self.rows[iy]):
                    if a <= f + 1 and e <= b + 1 and c <= h + 1 and g <= d + 1:
                        uf.union(start[iy - 1] + i, start[iy] + j)
        size = {}
        for iy, r in enumerate(self.rows):
            for i, (a, b, c, d) in enumerate(r):
                k = uf.find(start[iy] + i)
                size[k] = size.get(k, 0) + (b - a + 1) * (d - c + 1)
        if len(size) <= 1:
            return self
        best = max(size, key=lambda k: (size[k], -k))
        return RectGrid(self.shape, [[q for i, q in enumerate(r) if uf.find(start[iy] + i) == best]
                                     for iy, r in enumerate(self.rows)])

    def row_blobs(self, iy, x_mm, z_mm):
        blobs = []
        for lab, (a, b, c, d) in enumerate(self.rows[iy], 1):
            xs, zs = x_mm[a:b + 1], z_mm[c:d + 1]
            # repeat/tile reproduce the voxel order of np.where, so the pairwise
            # float summation (and hence every later comparison) is bit-identical.
            blobs.append({"label": lab,
                          "cx_mm": float(np.repeat(xs, len(zs)).mean()),
                          "cz_mm": float(np.tile(zs, len(xs)).mean()),
                          "x_lo": float(xs.min()), "x_hi": float(xs.max()),
                          "z_lo": float(zs.min()), "z_hi": float(zs.max())})
        return blobs

    def shrink(self, iy, lab, x_mm, z_mm, x_lo, x_hi, z_lo, z_hi):
        a, b, c, d = self.rows[iy][lab - 1]
        xi = a + np.flatnonzero((x_mm[a:b + 1] >= x_lo) & (x_mm[a:b + 1] <= x_hi))
        zi = c + np.flatnonzero((z_mm[c:d + 1] >= z_lo) & (z_mm[c:d + 1] <= z_hi))
        if len(xi) == 0 or len(zi) == 0:
            return None
        return int(xi[0]), int(xi[-1]), int(zi[0]), int(zi[-1])


class IntervalGrid:
    def __init__(self, occ_f, lo_ij, hi_ij, zs):
        self.f, self.zs = occ_f, zs
        # first/one-past-last Z index satisfying zs >= lo and zs <= hi
        self.i0 = np.searchsorted(zs, lo_ij, side="left").astype(np.int32)
        self.i1 = np.searchsorted(zs, hi_ij, side="right").astype(np.int32)
        self.i1 = np.where(occ_f, np.maximum(self.i1, self.i0), self.i0)
        self.iz = np.arange(len(zs), dtype=np.int32)
        self.shape = (occ_f.shape[0], occ_f.shape[1], len(zs))
        self.size = int(np.prod(self.shape, dtype=np.int64))
        self.dtype = np.dtype(bool)

    def sum(self):
        return int((self.i1 - self.i0).sum())

    def block(self, sx, sy):
        return (self.iz >= self.i0[sx, sy][..., None]) & (self.iz < self.i1[sx, sy][..., None])

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),) * (3 - len(key))
        return self.block(key[0], key[1])[..., key[2]]

    def __array__(self, dtype=None, copy=None):
        out = np.zeros(self.shape, dtype=bool)
        for i in range(0, self.shape[0], 64):
            out[i:i + 64] = self.block(slice(i, i + 64), slice(None))
        return out if dtype is None else out.astype(dtype)


def component_sizes(vol, chunk=64):
    """Sizes of the 6-connected components, labelling one X slab at a time."""
    sizes, pairs, offset, prev = [], [], 0, None
    for i in range(0, vol.shape[0], chunk):
        lab, n = ndimage.label(vol[i:i + chunk])
        sizes.append(np.bincount(lab[lab > 0], minlength=n + 1)[1:])
        if prev is not None:
            m = (prev > 0) & (lab[0] > 0)
            if m.any():
                pairs.append(np.unique(np.stack([prev[m], lab[0][m] + offset], 1), axis=0))
        prev = np.where(lab[-1] > 0, lab[-1] + offset, 0)
        offset += n
    sizes = np.concatenate(sizes) if sizes else np.zeros(0, np.int64)
    uf = _UnionFind(offset + 1)
    for p in pairs:
        for a, b in p.tolist():
            uf.union(a, b)
    out = {}
    for k in range(1, offset + 1):
        r = uf.find(k)
        out[r] = out.get(r, 0) + int(sizes[k - 1])
    return np.array(list(out.values()), dtype=np.int64)


def _filt3(v, op, w):
    """Separable cube max/min filter, in place, constant-1 border."""
    kv, kh = np.ones((w, 1), np.uint8), np.ones((1, w), np.uint8)
    nx, ny, nz = v.shape
    for i in range(nx):
        s = op(v[i], kv, borderType=cv2.BORDER_CONSTANT, borderValue=1)
        v[i] = op(s, kh, borderType=cv2.BORDER_CONSTANT, borderValue=1)
    for j in range(0, ny, 32):
        sl = np.ascontiguousarray(v[:, j:j + 32].reshape(nx, -1))
        v[:, j:j + 32] = op(sl, kv, borderType=cv2.BORDER_CONSTANT, borderValue=1).reshape(nx, -1, nz)


def close_cube(vol_u8, radius_px):
    """
    Binary closing by a cube of the given radius.

    The constant-1 border is not decoration: everything outside the array
    counts as solid in both passes, so surfaces flush against an array face
    (the crown, the soles) are not eaten by the erosion.
    """
    w = 2 * radius_px + 1
    v = vol_u8.copy()
    _filt3(v, cv2.dilate, w)
    if v.all():
        # No background left: keep the old distance-transform code's empty result.
        return np.zeros_like(v)
    _filt3(v, cv2.erode, w)
    return v


def surface_indices(occ, chunk=64):
    """np.flatnonzero(occ & ~binary_erosion(occ)) without full-size temporaries."""
    nx, ny, nz = occ.shape
    out = []
    for i in range(0, nx, chunk):
        a, b = max(i - 1, 0), min(i + chunk + 1, nx)
        o = occ[a:b]
        e = o.copy()
        e[1:] &= o[:-1]; e[:-1] &= o[1:]
        e[:, 1:] &= o[:, :-1]; e[:, :-1] &= o[:, 1:]
        e[:, :, 1:] &= o[:, :, :-1]; e[:, :, :-1] &= o[:, :, 1:]
        if a == 0: e[0] = False
        if b == nx: e[-1] = False
        e[:, 0] = False; e[:, -1] = False; e[:, :, 0] = False; e[:, :, -1] = False
        s = (o & ~e)[i - a:min(i + chunk, nx) - a]
        out.append(np.flatnonzero(s) + i * ny * nz)
    return np.concatenate(out)
