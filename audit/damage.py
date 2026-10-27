"""Damage a mask so that Dice(clean, damaged) hits a target, for three kinds of error.

  size    threshold of the Gaussian-smoothed mask moved up (too small) or down (too big)
  shift   the smoothed mask moved by a sub-voxel/voxel amount along a random direction
  island  a smooth ball added away from the structure; radius chosen to hit the target

`clean` is always smooth_mask(original) = gaussian(sigma 0.7) > 0.5, so clean and damaged masks
have been smoothed the same way (blocky-edge artefacts cannot give the damage away).
"""
import numpy as np
from scipy import ndimage as ndi

SIGMA = 0.7


def field_of(mask):
    return ndi.gaussian_filter(mask.astype(np.float32), SIGMA)


def clean_mask(mask):
    return field_of(mask) > 0.5


def dice(a, b):
    s = a.sum() + b.sum()
    return 2.0 * (a & b).sum() / s if s else 1.0


def _bisect(make, target, lo, hi, increasing_hurts):
    """Find parameter p in [lo, hi] with dice(make(p)) ~ target. increasing_hurts: dice falls as p grows."""
    best = (None, None)
    for _ in range(30):
        mid = (lo + hi) / 2
        m = make(mid)
        d = dice(CLEAN[0], m)
        best = (m, d)
        if (d > target) == increasing_hurts:
            lo = mid
        else:
            hi = mid
    return best


CLEAN = [None]   # set per call (small helper to keep _bisect short)


def damage(mask, kind, target_dice, rng):
    """Return (damaged_mask, achieved_dice, info dict)."""
    field = field_of(mask)
    clean = field > 0.5
    CLEAN[0] = clean
    if kind == "size":
        if rng.random() < 0.5:   # too small: raise the threshold
            m, d = _bisect(lambda t: field > t, target_dice, 0.5, 0.999, increasing_hurts=True)
            direction = "smaller"
        else:                    # too big: lower the threshold
            m, d = _bisect(lambda t: field > t, target_dice, 0.001, 0.5, increasing_hurts=False)
            direction = "bigger"
        return m, d, dict(direction=direction)
    if kind == "shift":
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        m, d = _bisect(lambda s: ndi.shift(field, v * s, order=1, mode="constant") > 0.5,
                       target_dice, 0.0, 8.0, increasing_hurts=True)
        return m, d, dict(direction=np.round(v, 2).tolist())
    if kind == "island":
        dist_out = ndi.distance_transform_edt(~clean)
        cand = np.argwhere(dist_out >= np.percentile(dist_out[~clean], 80))
        c = cand[int(rng.integers(len(cand)))]
        zz, yy, xx = np.indices(mask.shape)
        r2 = (zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2

        def make(r):
            ball = ndi.gaussian_filter((r2 <= r * r).astype(np.float32), SIGMA) > 0.5
            return clean | ball

        # island grows -> dice falls
        m, d = _bisect(make, target_dice, 0.0, 15.0, increasing_hurts=True)
        return m, d, dict(centre=c.tolist())
    raise ValueError(kind)
