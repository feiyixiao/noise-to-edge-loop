"""Synthetic stand-in for (image, mask) pairs, so the audit pipeline can be tested
without any download. Each case is a small 3D volume with one bright ellipsoid
("structure") on a smooth noisy background; the clean mask is that ellipsoid.
"""
import numpy as np
from scipy import ndimage as ndi

SHAPE = (40, 40, 40)


def make_case(rng, shape=SHAPE):
    zz, yy, xx = np.indices(shape)
    centre = np.array(shape) / 2 + rng.uniform(-3, 3, 3)
    radii = rng.uniform(5, 9, 3)
    d = sum(((c - m) / r) ** 2 for c, m, r in zip((zz, yy, xx), centre, radii))
    mask = d <= 1.0

    background = ndi.gaussian_filter(rng.normal(0, 1, shape), 4)
    background = 0.3 * background / background.std()
    img = background + 0.8 * mask + rng.normal(0, 0.12, shape)
    img = ndi.gaussian_filter(img, 0.8)
    return img.astype(np.float32), mask
