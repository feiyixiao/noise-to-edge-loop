"""Controlled mask corruptions with known ground truth.

Every function takes a boolean 3D mask and a severity in {1, 2, 3} and returns a
corrupted boolean mask. Severity is in voxels (or voxel-sized units), so results
are comparable across datasets.
"""
import numpy as np
from scipy import ndimage as ndi


def erode(mask, sev, rng):
    return ndi.binary_erosion(mask, iterations=sev)


def dilate(mask, sev, rng):
    return ndi.binary_dilation(mask, iterations=sev)


def shift(mask, sev, rng):
    axis = int(rng.integers(0, 3))
    sign = int(rng.choice([-1, 1]))
    return np.roll(mask, sign * sev, axis=axis)


def drop_slab(mask, sev, rng):
    """Remove a slab 'sev' voxels thick from one end of the structure along a random axis."""
    axis = int(rng.integers(0, 3))
    idx = np.where(mask.any(axis=tuple(a for a in range(3) if a != axis)))[0]
    out = mask.copy()
    sl = [slice(None)] * 3
    sl[axis] = slice(idx[0], idx[0] + 2 * sev)
    out[tuple(sl)] = False
    return out


def spurious_blob(mask, sev, rng):
    """Add a small separate island of size ~ (2*sev+1)^3 away from the structure."""
    out = mask.copy()
    far = ~ndi.binary_dilation(mask, iterations=4)
    cand = np.argwhere(far)
    z, y, x = cand[int(rng.integers(len(cand)))]
    r = sev + 1
    out[max(z - r, 0):z + r, max(y - r, 0):y + r, max(x - r, 0):x + r] = True
    return out


CORRUPTIONS = {
    "erode": erode,
    "dilate": dilate,
    "shift": shift,
    "drop_slab": drop_slab,
    "spurious_blob": spurious_blob,
}
