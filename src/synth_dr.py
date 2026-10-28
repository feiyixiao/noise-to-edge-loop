"""Domain-randomised synthetic volumes with perfect masks (Step 5b). One generator configuration, fixed seed."""
import csv

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial.transform import Rotation

from audit.damage import clean_mask

SYNTH_SEED = 0
N_VOLUMES = 500


def make_case(rng):
    shape = tuple(int(s) for s in rng.integers(32, 53, 3))
    centre = np.array(shape) / 2 + rng.uniform(-0.12, 0.12, 3) * np.array(shape)
    radii = rng.uniform(4, 11, 3)
    rot = Rotation.random(random_state=int(rng.integers(2**31))).as_matrix()
    zz, yy, xx = np.indices(shape)
    p = np.stack([zz - centre[0], yy - centre[1], xx - centre[2]], -1) @ rot
    mask = (((p / radii) ** 2).sum(-1)) <= 1.0
    contrast = float(rng.uniform(0.3, 1.2)) * float(rng.choice([-1, 1]))
    bg_sigma, bg_amp = float(rng.uniform(2, 6)), float(rng.uniform(0.1, 0.5))
    noise_sd, blur = float(rng.uniform(0.03, 0.3)), float(rng.uniform(0.4, 1.5))
    bg = ndi.gaussian_filter(rng.normal(0, 1, shape), bg_sigma)
    bg = bg_amp * bg / (bg.std() + 1e-9)
    img = bg + contrast * mask + rng.normal(0, noise_sd, shape)
    img = ndi.gaussian_filter(img, blur)
    lo, hi = np.percentile(img, [1, 99])
    img = np.clip((img - lo) / (hi - lo + 1e-9), 0, 1).astype(np.float32)
    params = dict(shape="x".join(map(str, shape)), radii=np.round(radii, 2).tolist(), contrast=round(contrast, 3),
                  bg_sigma=round(bg_sigma, 2), bg_amp=round(bg_amp, 3), noise_sd=round(noise_sd, 3), blur=round(blur, 2))
    return img, clean_mask(mask), params


def make_all(path_params=None):
    rng = np.random.default_rng(SYNTH_SEED)
    imgs, masks, rows = [], [], []
    for i in range(N_VOLUMES):
        im, m, p = make_case(rng)
        if not m.any():                    # essentially never; keep the count at 500 anyway
            im, m, p = make_case(rng)
        imgs.append(im)
        masks.append(m)
        rows.append(dict(idx=i, **p))
    if path_params:
        with open(path_params, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return imgs, masks


def cnr(img, mask):
    inside, outside = img[mask], img[~mask]
    return float(abs(inside.mean() - outside.mean()) / (outside.std() + 1e-9))
