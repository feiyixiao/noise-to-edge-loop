# RESULTS_FOR_CV (measured numbers only; source: results/*.csv)

Setup: MSD Task04 Hippocampus, 2D axial slices (64x64), split by case (train 0-159, val 160-199, test 200-259, clean test masks). 30 training runs = 2 models x 3 seeds (0,1,2) x 5 configurations (noise rate 0/0.2/0.4 unfiltered, rate 0.2/0.4 audit-filtered). Sum of per-run training time: 5.34 h (CPU). Wall-clock was longer: the cloud container restarted several times and training resumed from per-epoch checkpoints; `run_all.py` was not timed end to end.

## Test Dice, mean ± SD over 3 seeds (`results/main_table.csv`)
| model | rate | no filter | audit-filtered |
|---|---|---|---|
| U-Net from scratch | 0 | 0.8975 ± 0.0025 | – |
| | 0.2 | 0.8944 ± 0.0015 | 0.8920 ± 0.0020 |
| | 0.4 | 0.8923 ± 0.0018 | 0.8855 ± 0.0028 |
| U-Net, ImageNet ResNet18 encoder | 0 | 0.8960 ± 0.0042 | – |
| | 0.2 | 0.8908 ± 0.0027 | 0.8927 ± 0.0023 |
| | 0.4 | 0.8904 ± 0.0024 | 0.8885 ± 0.0008 |

- Noise effect (rate 0 -> 0.4, unfiltered): scratch -0.0052, ResNet18 -0.0056 Dice (about 0.5 Dice points).
- Filtering effect at rate 0.4: scratch -0.0068, ResNet18 -0.0019 (filtering did not recover Dice); at rate 0.2: scratch -0.0024, ResNet18 +0.0019.

## Audit filter (`results/filter_stats.csv`; drops top-k% by `Calibrated.combined`, k = nominal rate)
| rate | damaged | dropped | caught | precision = recall | recall size | recall shift | recall island |
|---|---|---|---|---|---|---|---|
| 0.2 | 32 | 32 | 24 | 0.75 | 1.00 | 0.53 | 1.00 |
| 0.4 | 64 | 64 | 53 | 0.83 | 0.89 | 0.61 | 1.00 |

## INT8 post-training quantization (`results/quant.csv`; ORT 1.29.0, static QDQ, 100 train calibration slices; 4-core cloud CPU, 1 thread, batch 1, 1501 test slices; models = rate 0, seed 0)
| model | precision | test Dice | size (MB) | latency median / p95 (ms) |
|---|---|---|---|---|
| scratch | FP32 | 0.8952 | 1.934 | 2.581 / 3.497 |
| scratch | INT8 | 0.8952 | 0.526 | 1.181 / 1.395 |
| ResNet18 | FP32 | 0.8912 | 57.281 | 6.071 / 7.015 |
| ResNet18 | INT8 | 0.8779 | 14.474 | 2.102 / 3.024 |

ONNX vs PyTorch max abs diff: 1.71e-05 (scratch), 3.43e-05 (ResNet18). INT8 changed Dice by 0.0000 (scratch) and -0.0133 (ResNet18); size 3.7x / 4.0x smaller; median latency 2.2x / 2.9x lower on this CPU.

## Step 5b: synthetic data arm (`results/runs_5b.csv`, `summary_5b.csv`, `synth_realism.csv`)
Model A (scratch U-Net), 3 seeds, tested on the same 60 clean real test cases. 500 domain-randomised ellipsoid volumes (7577 slices); (i) 20 real training cases only; (ii) synthetic pretraining then fine-tune on the same 20 real cases; (iii) synthetic only, zero real training cases.
| condition | test Dice mean ± SD (3 seeds) | per seed (0,1,2) |
|---|---|---|
| (i) real only, 20 cases | 0.8568 ± 0.0062 | 0.8507, 0.8567, 0.8630 |
| (ii) synthetic pretrain + fine-tune | 0.8673 ± 0.0059 | 0.8652, 0.8740, 0.8627 |
| (iii) synthetic only | 0.3429 ± 0.0094 | 0.3321, 0.3483, 0.3483 |
- (ii) minus (i): +0.0105 mean (per seed +0.0145, +0.0173, -0.0003). Mean gain is larger than 1 SD (0.0062) but not in every seed, and on clean real validation Dice the gap is only 0.0028 (0.8571 vs 0.8543).
- Realism, real train vs synthetic: contrast-to-noise inside/outside mask 0.55 ± 0.09 vs 2.41 ± 1.48; intensity-histogram (Wasserstein) distance 0.066. The synthetic structures are far higher-contrast than real hippocampus.
- Predictions: P6 held (iii far below i). P7 met by the pre-registered ">1 SD" criterion on the mean but weak (2 of 3 seeds).

## What did NOT work / was wrong
- Audit filtering did not recover Dice (it lowered it at rate 0.4 for both models); the injected noise (Dice 0.8, 32-64 cases) cost only about 0.5 Dice points here.
- Synthetic-only training (toy ellipsoid generator) reached only 0.34 test Dice on real data (sim-to-real gap).
- ResNet18 INT8 lost 0.0133 Dice (more than the 0.01 I predicted); pretrained encoder did not lose less to noise than scratch.

## Do not claim
Embedded deployment, QAT, vision foundation model, clinical validity, 3D results. Latency is a 4-core cloud CPU number, not a MacBook or embedded number.
