import copy
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import config as C
from src.models import build


def dice_loss(logit, y):
    p = torch.sigmoid(logit)
    inter = (p * y).sum((1, 2, 3))
    return (1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)).mean()


def case_dice(pred_bin, y, case_idx):
    """Mean over cases of the Dice pooled over that case's slices."""
    ds = []
    for c in np.unique(case_idx):
        s = case_idx == c
        a, b = pred_bin[s], y[s] > 0.5
        ds.append(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))
    return float(np.mean(ds))


@torch.no_grad()
def predict(model, X, bs=256):
    model.eval()
    out = [torch.sigmoid(model(torch.from_numpy(X[i:i + bs]))).numpy() > 0.5 for i in range(0, len(X), bs)]
    return np.concatenate(out)


def fit(name, seed, Xtr, Ytr, Xva, Yva, cva, log=print, ckpt=None):
    C.seed_everything(seed)
    model = build(name)
    opt = torch.optim.Adam(model.parameters(), lr=C.LR)
    n = len(Xtr)
    Xt, Yt = torch.from_numpy(Xtr), torch.from_numpy(Ytr)
    best, best_state, bad, t0, ep_times = -1, None, 0, time.time(), []
    g = torch.Generator().manual_seed(seed)
    start, t_prev = 0, 0.0
    if ckpt and os.path.exists(ckpt):          # resume after a container restart (same RNG stream)
        s = torch.load(ckpt, weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); g.set_state(s["g"])
        best, best_state, bad, ep_times, start, t_prev = s["best"], s["best_state"], s["bad"], s["ep_times"], s["ep"], s["t"]
        log(f"  resumed from epoch {start}")
    t0 = time.time() - t_prev
    for ep in range(start, C.EPOCHS):
        te = time.time()
        model.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, C.BATCH):
            b = perm[i:i + C.BATCH]
            x, y = Xt[b], Yt[b]
            if torch.rand(1, generator=g) < 0.5:
                x, y = x.flip(3), y.flip(3)
            if torch.rand(1, generator=g) < 0.5:
                x, y = x.flip(2), y.flip(2)
            lg = model(x)
            loss = F.binary_cross_entropy_with_logits(lg, y) + dice_loss(lg, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        vd = case_dice(predict(model, Xva), Yva, cva)
        ep_times.append(time.time() - te)
        log(f"  ep{ep:02d} loss {loss.item():.3f} val_dice {vd:.4f} {ep_times[-1]:.1f}s")
        if vd > best:
            best, best_state, bad = vd, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        if ckpt:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), g=g.get_state(), best=best,
                            best_state=best_state, bad=bad, ep_times=ep_times, ep=ep + 1, t=time.time() - t0), ckpt + ".tmp")
            os.replace(ckpt + ".tmp", ckpt)
        if bad >= C.PATIENCE:
            break
    model.load_state_dict(best_state)
    return model, best, time.time() - t0, float(np.mean(ep_times)), ep + 1
