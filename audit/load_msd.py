"""Load Medical Segmentation Decathlon Task04 (hippocampus) as (image, mask) pairs.
Labels 1 (anterior) and 2 (posterior) are merged into one binary hippocampus mask.
Images are rescaled per case to robust [0, 1] so edge strength is comparable across cases.
"""
import json
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent / "data_raw" / "msd_hippocampus" / "Task04_Hippocampus"


def load_cases(n=None, seed=0):
    meta = json.loads((ROOT / "dataset.json").read_text())
    items = sorted(meta["training"], key=lambda d: d["image"])
    rng = np.random.default_rng(seed)
    rng.shuffle(items)
    if n:
        items = items[:n]
    cases, names = [], []
    for it in items:
        img = nib.load(ROOT / it["image"]).get_fdata().astype(np.float32)
        lab = nib.load(ROOT / it["label"]).get_fdata() > 0
        lo, hi = np.percentile(img, [1, 99])
        img = np.clip((img - lo) / (hi - lo + 1e-9), 0, 1)
        cases.append((img, lab))
        names.append(Path(it["image"]).name.replace(".nii.gz", ""))
    return cases, names
