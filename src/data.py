"""Load cases, split by case, make 2D axial slices (64x64)."""
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config as C  # noqa: E402
from audit.damage import clean_mask  # noqa: E402
from audit.load_msd import load_cases  # noqa: E402

AXIS = 2   # slice axis (last volume axis); volumes ~35 x 50 x 35 -> slices ~35 x 50


def load_split():
    cases, names = load_cases(seed=C.CASE_SEED)
    split = {"train": list(C.TRAIN), "val": list(C.VAL), "test": list(C.TEST)}
    ids = [set(names[i] for i in v) for v in split.values()]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2]), "case leakage"
    imgs = [c[0] for c in cases]
    clean = [clean_mask(c[1]) for c in cases]      # every mask is smoothed (see DESIGN.md)
    h = hashlib.sha1(",".join(f"{k}:{names[i]}" for k, v in split.items() for i in v).encode()).hexdigest()[:10]
    return imgs, clean, names, split, h


def _pad64(a):
    out = np.zeros((C.IMG_SIZE, C.IMG_SIZE), a.dtype)
    h, w = a.shape
    h2, w2 = min(h, C.IMG_SIZE), min(w, C.IMG_SIZE)
    sy, sx = (h - h2) // 2, (w - w2) // 2
    oy, ox = (C.IMG_SIZE - h2) // 2, (C.IMG_SIZE - w2) // 2
    out[oy:oy + h2, ox:ox + w2] = a[sy:sy + h2, sx:sx + w2]
    return out


def slices(idx, imgs, masks):
    """Axial slices with >=1 mask voxel for case indices idx. Returns X (N,1,64,64), Y, case_idx, slice_idx."""
    X, Y, ci, si = [], [], [], []
    for i in idx:
        im, m = np.moveaxis(imgs[i], AXIS, 0), np.moveaxis(masks[i], AXIS, 0)
        for s in range(m.shape[0]):
            if m[s].any():
                X.append(_pad64(im[s]))
                Y.append(_pad64(m[s].astype(np.float32)))
                ci.append(i)
                si.append(s)
    return (np.stack(X)[:, None].astype(np.float32), np.stack(Y)[:, None].astype(np.float32),
            np.array(ci), np.array(si))
