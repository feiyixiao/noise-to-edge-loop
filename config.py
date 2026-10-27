"""Single source of truth for seeds, split and hyper-parameters. Nothing else sets a seed."""
import os
import random

import numpy as np

# --- split (by CASE, after load_cases(seed=0) shuffling) ---
CASE_SEED = 0
TRAIN = range(0, 160)
VAL = range(160, 200)
TEST = range(200, 260)

# --- noise ---
NOISE_SEED = 0                 # one noise realisation (damaged-case choice, kind, direction)
RATES = (0.0, 0.2, 0.4)
KINDS = ("size", "shift", "island")
TARGET_DICE = 0.8

# --- data ---
IMG_SIZE = 64

# --- training (EPOCHS/PATIENCE may change ONLY after the pilot; log the change in DESIGN.md) ---
MODELS = ("unet_scratch", "unet_resnet18")
TRAIN_SEEDS = (0, 1, 2)
EPOCHS = 30
PATIENCE = 8
BATCH = 32
LR = 1e-3
BASE_CH = 16
N_THREADS = 4

# --- quantization / latency ---
N_CALIB = 100
WARMUP, RUNS = 20, 200
HARDWARE_LABEL = "4-core cloud CPU"   # never label these numbers as MacBook


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.set_num_threads(N_THREADS)
        torch.use_deterministic_algorithms(False)
    except ImportError:
        pass
