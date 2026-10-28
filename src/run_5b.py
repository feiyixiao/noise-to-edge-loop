"""Step 5b: real-only (i) vs synthetic pretraining + fine-tune (ii) vs synthetic only (iii). Model A, resumable."""
import csv
import os

import mlflow
import numpy as np
import torch
from scipy.stats import wasserstein_distance

import config as C
from src.data import load_split, slices
from src.synth_dr import cnr, make_all
from src.train import case_dice, fit, predict

R = "results"
OUT = f"{R}/runs_5b.csv"
FIELDS = ["condition", "seed", "n_real_train_cases", "n_train_slices", "val_dice", "test_dice", "epochs", "train_s"]
N_REAL = 20


def realism(imgs, clean, train_idx, s_imgs, s_masks):
    rng = np.random.default_rng(0)
    c_real = [cnr(imgs[i], clean[i]) for i in train_idx]
    c_syn = [cnr(a, b) for a, b in zip(s_imgs, s_masks)]
    pick = lambda L: np.concatenate([x.ravel() for x in L])
    r, s = pick([imgs[i] for i in train_idx]), pick(s_imgs)
    d = wasserstein_distance(rng.choice(r, 200000, replace=False), rng.choice(s, 200000, replace=False))
    with open(f"{R}/synth_realism.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "real_train_mean", "real_train_sd", "synthetic_mean", "synthetic_sd"])
        w.writerow(["CNR_inside_vs_outside_mask", f"{np.mean(c_real):.4f}", f"{np.std(c_real):.4f}",
                    f"{np.mean(c_syn):.4f}", f"{np.std(c_syn):.4f}"])
        w.writerow(["intensity_wasserstein_distance_real_vs_synthetic", f"{d:.4f}", "", "", ""])


def main():
    imgs, clean, names, split, h = load_split()
    s_imgs, s_masks = make_all(f"{R}/synth_params.csv")
    if not os.path.exists(f"{R}/synth_realism.csv"):
        realism(imgs, clean, split["train"], s_imgs, s_masks)
    Xs, Ys, cs, _ = slices(range(len(s_imgs)), s_imgs, s_masks)
    X20, Y20, c20, _ = slices(split["train"][:N_REAL], imgs, clean)
    Xva, Yva, cva, _ = slices(split["val"], imgs, clean)
    Xte, Yte, cte, _ = slices(split["test"], imgs, clean)
    print("synthetic slices", len(Xs), "real-20 slices", len(X20), flush=True)
    done = set()
    if os.path.exists(OUT):
        done = {(r["condition"], int(r["seed"])) for r in csv.DictReader(open(OUT))}
    else:
        csv.writer(open(OUT, "w")).writerow(FIELDS)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("noise_to_edge_loop_5b")

    def record(cond, seed, nreal, nsl, vd, td, ne, t):
        csv.writer(open(OUT, "a")).writerow([cond, seed, nreal, nsl, f"{vd:.4f}", f"{td:.4f}", ne, f"{t:.1f}"])

    log = lambda s: print(s, flush=True)
    for seed in C.TRAIN_SEEDS:
        if ("i_real_only", seed) not in done:
            print(f"== (i) real only seed={seed}", flush=True)
            with mlflow.start_run(run_name=f"5b_i_s{seed}"):
                m, vd, t, _, ne = fit("unet_scratch", seed, X20, Y20, Xva, Yva, cva, log=log,
                                      ckpt=f"{R}/ckpt_5b_i_s{seed}.pt")
                td = case_dice(predict(m, Xte), Yte, cte)
                mlflow.log_params(dict(condition="i", seed=seed, n_real=N_REAL))
                mlflow.log_metrics(dict(val_dice=vd, test_dice=td))
            record("i_real_only", seed, N_REAL, len(X20), vd, td, ne, t)
            os.remove(f"{R}/ckpt_5b_i_s{seed}.pt")
        state_path = f"{R}/5b_pretrained_s{seed}.pt"
        if ("iii_synthetic_only", seed) not in done:
            print(f"== pretrain on synthetic (iii) seed={seed}", flush=True)
            with mlflow.start_run(run_name=f"5b_iii_s{seed}"):
                m, vd, t, _, ne = fit("unet_scratch", seed, Xs, Ys, Xva, Yva, cva, log=log,
                                      ckpt=f"{R}/ckpt_5b_iii_s{seed}.pt")
                td = case_dice(predict(m, Xte), Yte, cte)
                mlflow.log_params(dict(condition="iii", seed=seed, n_real=0))
                mlflow.log_metrics(dict(val_dice=vd, test_dice=td))
            torch.save(m.state_dict(), state_path)
            record("iii_synthetic_only", seed, 0, len(Xs), vd, td, ne, t)
            os.remove(f"{R}/ckpt_5b_iii_s{seed}.pt")
        if ("ii_pretrain_finetune", seed) not in done:
            print(f"== (ii) fine-tune seed={seed}", flush=True)
            with mlflow.start_run(run_name=f"5b_ii_s{seed}"):
                m, vd, t, _, ne = fit("unet_scratch", seed, X20, Y20, Xva, Yva, cva, log=log,
                                      ckpt=f"{R}/ckpt_5b_ii_s{seed}.pt", init_state=torch.load(state_path))
                td = case_dice(predict(m, Xte), Yte, cte)
                mlflow.log_params(dict(condition="ii", seed=seed, n_real=N_REAL))
                mlflow.log_metrics(dict(val_dice=vd, test_dice=td))
            record("ii_pretrain_finetune", seed, N_REAL, len(X20), vd, td, ne, t)
            os.remove(f"{R}/ckpt_5b_ii_s{seed}.pt")
    open(f"{R}/train_done_5b", "w").write("done\n")


if __name__ == "__main__":
    main()
