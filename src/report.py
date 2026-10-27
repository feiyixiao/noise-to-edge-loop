"""Aggregate results/runs.csv -> main_table.csv, figure, RESULTS_FOR_CV.md numbers. Every number comes from results/*.csv."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results"


def main():
    d = pd.read_csv(f"{R}/runs.csv")
    g = d.groupby(["model", "rate", "filtered"])
    t = g.agg(n_seeds=("seed", "count"), test_mean=("test_dice", "mean"), test_sd=("test_dice", "std"),
              val_mean=("val_dice", "mean"), epochs_mean=("epochs", "mean"), train_s_mean=("train_s", "mean"),
              n_train_cases=("n_train_cases", "first")).reset_index()
    t.round(4).to_csv(f"{R}/main_table.csv", index=False)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, name in zip(axs, ["unet_scratch", "unet_resnet18"]):
        for f, lab, ls in (("n", "no filter", "-"), ("y", "audit-filtered", "--")):
            s = t[(t.model == name) & (t.filtered == f)]
            if f == "y":   # rate 0: nothing is filtered, so the filtered point equals the unfiltered one
                s = pd.concat([t[(t.model == name) & (t.filtered == "n") & (t.rate == 0)], s])
            ax.errorbar(s.rate, s.test_mean, yerr=s.test_sd, marker="o", ls=ls, capsize=3, label=lab)
        ax.set_title(name)
        ax.set_xlabel("noise rate (fraction of train cases damaged)")
        ax.set_xticks([0, 0.2, 0.4])
    axs[0].set_ylabel("test Dice (clean masks), mean ± SD over seeds")
    axs[0].legend()
    fig.tight_layout()
    fig.savefig(f"{R}/dice_vs_noise.png", dpi=150)
    print(t.round(4).to_string())


if __name__ == "__main__":
    main()
