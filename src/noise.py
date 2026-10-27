"""Damage a fixed, nested subset of TRAIN cases (3D), write the manifest."""
import csv

import numpy as np

import config as C
from audit.damage import damage, dice


def damaged_pool(imgs, clean, train_idx):
    """For the first round(0.4*160) cases of one seeded permutation: kind + damaged 3D mask.
    Same damaged mask is reused at every rate (nested sets)."""
    perm = np.random.default_rng(C.NOISE_SEED).permutation(list(train_idx))
    n_max = int(round(max(C.RATES) * len(train_idx)))
    pool = {}
    for rank, ci in enumerate(perm[:n_max]):
        rng = np.random.default_rng(C.NOISE_SEED * 100003 + int(ci))
        kind = C.KINDS[int(rng.integers(len(C.KINDS)))]
        m, d, _ = damage(clean[ci], kind, C.TARGET_DICE, rng)
        pool[int(ci)] = (rank, kind, m, float(d))
    return perm, pool


def masks_for_rate(rate, clean, train_idx, pool):
    """Return (dict case->mask used for training, set of damaged case ids)."""
    n = int(round(rate * len(train_idx)))
    damaged = {ci for ci, (rank, *_ ) in pool.items() if rank < n}
    return {i: (pool[i][2] if i in damaged else clean[i]) for i in train_idx}, damaged


def write_manifest(path, names, pool):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_idx", "case_id", "rank", "kind", "achieved_dice", "damaged_at_rate_0.2", "damaged_at_rate_0.4"])
        n = {r: int(round(r * len(C.TRAIN))) for r in C.RATES}
        for ci, (rank, kind, m, d) in sorted(pool.items(), key=lambda kv: kv[1][0]):
            w.writerow([ci, names[ci], rank, kind, f"{d:.4f}", int(rank < n[0.2]), int(rank < n[0.4])])
