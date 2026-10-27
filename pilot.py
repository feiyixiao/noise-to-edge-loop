"""Pilot: unet_scratch, rate 0, seed 0. Measures time/epoch and that val Dice > 0.7."""
import time
import config as C
from src.data import load_split, slices
from src.train import fit, predict, case_dice

imgs, clean, names, split, h = load_split()
Xtr, Ytr, ctr, _ = slices(split["train"], imgs, clean)
Xva, Yva, cva, _ = slices(split["val"], imgs, clean)
print("split hash", h, "train/val slices", len(Xtr), len(Xva))
m, best, t, ept, ne = fit("unet_scratch", 0, Xtr, Ytr, Xva, Yva, cva)
print(f"PILOT best val dice {best:.4f}  epochs {ne}  s/epoch {ept:.1f}  total {t:.0f}s")
