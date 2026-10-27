"""Step 5: ONNX export, ORT post-training INT8 (static, fallback dynamic), FP32 vs INT8 on CLEAN test slices."""
import csv
import os
import platform
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType, quantize_dynamic,
                                      quantize_static)

import config as C
from src.data import load_split, slices
from src.models import build
from src.train import case_dice

R = "results"


class Reader(CalibrationDataReader):
    def __init__(self, X):
        self.it = iter([{"x": X[i:i + 1]} for i in range(len(X))])

    def get_next(self):
        return next(self.it, None)


def session(path):
    o = ort.SessionOptions()
    o.intra_op_num_threads, o.inter_op_num_threads = 1, 1
    return ort.InferenceSession(path, o, providers=["CPUExecutionProvider"])


def run_dice(sess, X, Y, ci):
    pred = np.concatenate([sess.run(None, {"x": X[i:i + 1]})[0] > 0 for i in range(len(X))])
    return case_dice(pred, Y, ci)


def latency(sess, X):
    for i in range(C.WARMUP):
        sess.run(None, {"x": X[i % len(X):i % len(X) + 1]})
    t = []
    for i in range(C.RUNS):
        x = X[i % len(X):i % len(X) + 1]
        s = time.perf_counter()
        sess.run(None, {"x": x})
        t.append((time.perf_counter() - s) * 1e3)
    return float(np.median(t)), float(np.percentile(t, 95))


def main():
    imgs, clean, names, split, h = load_split()
    Xtr, _, _, _ = slices(split["train"], imgs, clean)          # clean train slices, calibration only
    rng = np.random.default_rng(C.NOISE_SEED)
    Xcal = Xtr[rng.choice(len(Xtr), C.N_CALIB, replace=False)]
    Xte, Yte, cte, _ = slices(split["test"], imgs, clean)
    rows = []
    for name in C.MODELS:
        if not os.path.exists(f"{R}/{name}_r0_s0.pt"):
            print("skip (not trained yet):", name)
            continue
        m = build(name)
        m.load_state_dict(torch.load(f"{R}/{name}_r0_s0.pt"))
        m.eval()
        fp32 = f"{R}/{name}_fp32.onnx"
        torch.onnx.export(m, torch.from_numpy(Xte[:1]), fp32, opset_version=17, input_names=["x"],
                          output_names=["y"], dynamo=False)
        onnx.checker.check_model(onnx.load(fp32))
        with torch.no_grad():
            ref = m(torch.from_numpy(Xte[:64])).numpy()
        s32 = session(fp32)
        out = np.concatenate([s32.run(None, {"x": Xte[i:i + 1]})[0] for i in range(64)])
        maxdiff = float(np.abs(out - ref).max())
        int8 = f"{R}/{name}_int8.onnx"
        mode = "static_QDQ"
        try:
            quantize_static(fp32, int8, Reader(Xcal), quant_format=QuantFormat.QDQ, per_channel=True,
                            activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
        except Exception as e:                                   # report which was used
            print("static failed:", repr(e)[:200])
            mode = "dynamic"
            quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8)
        s8 = session(int8)
        for tag, path, sess in (("fp32", fp32, s32), ("int8", int8, s8)):
            med, p95 = latency(sess, Xte)
            rows.append(dict(model=name, precision=tag, quant_mode=mode if tag == "int8" else "",
                             test_dice=f"{run_dice(sess, Xte, Yte, cte):.4f}",
                             size_mb=f"{os.path.getsize(path) / 1e6:.3f}", median_ms=f"{med:.3f}", p95_ms=f"{p95:.3f}",
                             onnx_vs_torch_max_abs_diff=f"{maxdiff:.2e}" if tag == "fp32" else "",
                             hardware=C.HARDWARE_LABEL, cpu=platform.processor() or platform.machine(),
                             ort=ort.__version__, threads=1, n_calib=C.N_CALIB, n_test_slices=len(Xte)))
            print(rows[-1], flush=True)
    with open(f"{R}/quant.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
