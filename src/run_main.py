"""Main run: 3 rates x 2 models x 3 seeds (unfiltered) + filtered runs at rates 0.2/0.4. Resumable."""
import csv
import json
import os
import time

import mlflow
import numpy as np

import config as C
from audit.detectors import Calibrated
from src.data import load_split, slices
from src.noise import damaged_pool, masks_for_rate, write_manifest
from src.train import case_dice, fit, predict

R = "results"
RUNS_CSV = f"{R}/runs.csv"
FIELDS = ["model", "rate", "filtered", "seed", "n_train_cases", "n_train_slices", "val_dice", "test_dice",
          "epochs", "train_s", "s_per_epoch", "split_hash"]


def filter_cases(imgs, clean, names, split, masks_by_case, rate):
    """Score every train case with Calibrated.combined (fit on CLEAN val masks); drop top round(rate*160)."""
    cal = Calibrated([(imgs[i], clean[i]) for i in split["val"]])
    score = {i: cal.combined(imgs[i], masks_by_case[i]) for i in split["train"]}
    k = int(round(rate * len(split["train"])))
    dropped = set(sorted(score, key=score.get, reverse=True)[:k])
    return dropped, score


def main():
    os.makedirs(R, exist_ok=True)
    imgs, clean, names, split, h = load_split()
    perm, pool = damaged_pool(imgs, clean, split["train"])
    write_manifest(f"{R}/noise_manifest.csv", names, pool)
    Xva, Yva, cva, _ = slices(split["val"], imgs, clean)
    Xte, Yte, cte, _ = slices(split["test"], imgs, clean)
    done = set()
    if os.path.exists(RUNS_CSV):
        done = {(r["model"], float(r["rate"]), r["filtered"], int(r["seed"])) for r in csv.DictReader(open(RUNS_CSV))}
    else:
        csv.writer(open(RUNS_CSV, "w")).writerow(FIELDS)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("noise_to_edge_loop")
    filt_rows = []
    for rate in C.RATES:
        masks, damaged = masks_for_rate(rate, clean, split["train"], pool)
        dropped, score = filter_cases(imgs, clean, names, split, masks, rate)
        tp = len(dropped & damaged)
        kinds = {k: [] for k in C.KINDS}
        for ci in damaged:
            kinds[pool[ci][1]].append(ci in dropped)
        filt_rows.append(dict(rate=rate, n_damaged=len(damaged), n_dropped=len(dropped), tp=tp,
                              precision=(tp / len(dropped)) if dropped else "", recall=(tp / len(damaged)) if damaged else "",
                              **{f"recall_{k}": (np.mean(v) if v else "") for k, v in kinds.items()}))
        for filtered in ("n", "y"):
            if filtered == "y" and rate == 0.0:
                continue
            train_idx = [i for i in split["train"] if not (filtered == "y" and i in dropped)]
            Xtr, Ytr, ctr, _ = slices(train_idx, imgs, [masks.get(i) if i in masks else None for i in range(len(imgs))])
            for name in C.MODELS:
                for seed in C.TRAIN_SEEDS:
                    if (name, rate, filtered, seed) in done:
                        continue
                    print(f"== {name} rate={rate} filtered={filtered} seed={seed} slices={len(Xtr)}", flush=True)
                    with mlflow.start_run(run_name=f"{name}_r{rate}_f{filtered}_s{seed}"):
                        m, vd, t, ept, ne = fit(name, seed, Xtr, Ytr, Xva, Yva, cva,
                                                log=lambda s: print(s, flush=True))
                        td = case_dice(predict(m, Xte), Yte, cte)
                        mlflow.log_params(dict(model=name, rate=rate, filtered=filtered, seed=seed, split_hash=h,
                                               n_train_cases=len(train_idx)))
                        mlflow.log_metrics(dict(val_dice=vd, test_dice=td, train_s=t, epochs=ne))
                    row = [name, rate, filtered, seed, len(train_idx), len(Xtr), f"{vd:.4f}", f"{td:.4f}", ne,
                           f"{t:.1f}", f"{ept:.1f}", h]
                    csv.writer(open(RUNS_CSV, "a")).writerow(row)
                    if rate == 0.0 and filtered == "n" and seed == 0:
                        import torch
                        torch.save(m.state_dict(), f"{R}/{name}_r0_s0.pt")
    with open(f"{R}/filter_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filt_rows[0].keys()))
        w.writeheader()
        w.writerows(filt_rows)


if __name__ == "__main__":
    main()
