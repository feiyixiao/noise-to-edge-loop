# DESIGN (written BEFORE the main run)

Question: when training masks are damaged in known ways, how much does a small 2D segmentation model lose, how much of that does the audit check win back by filtering, and what does post-training INT8 cost in Dice vs CPU speed?

## Data and split (fixed)
- `audit.load_msd.load_cases(seed=0)` -> 260 cases, shuffled once. Split by CASE index: train 0-159, val 160-199, test 200-259. An assert checks no case id is in two splits.
- 2D samples: axial slices (last volume axis is NOT assumed; slice axis = the axis fixed in `src/data.py`) containing >= 1 mask voxel of the CLEAN mask, centre pad/crop to 64x64. Per slice we store case id and slice index.
- Every mask (train, val, test, rate 0) goes through `audit.damage.clean_mask` (Gaussian sigma 0.7 > 0.5). Reason: `damage()` works on the smoothed mask, so unsmoothed "clean" masks would differ from damaged ones by blockiness alone and leak the label. Val/test ground truth is the smoothed clean mask.

## Noise
- Rate r in {0, 0.2, 0.4} = fraction of the 160 train cases replaced by a damaged 3D mask (then sliced). 0 / 32 / 64 cases.
- Damaged cases come from one seeded permutation (`NOISE_SEED=0`), first round(r*160) cases, so noise sets are nested across rates and a damaged case gets the same damaged mask at 0.2 and 0.4.
- Kind per damaged case uniform over {size, shift, island}; target Dice 0.8. Val and test masks are clean.
- Manifest: `results/noise_manifest.csv` (case id, rate, kind, achieved Dice).
- Dice 0.9 stress test: only if time allows, not part of the main run.

## Models (same loss/epochs/aug/optimizer for both)
- A: U-Net from scratch, 4 levels, base 16 channels.
- B: U-Net with torchvision ResNet18 ImageNet encoder, conv1 weights averaged over RGB for 1 input channel.
- BCE + Dice loss, Adam lr 1e-3, batch 32, flip-only augmentation, max 30 epochs, early stopping (patience 8) on CLEAN val Dice. Seeds 0,1,2. 3 rates x 2 models x 3 seeds = 18 unfiltered runs, + 12 filtered runs (rates 0.2, 0.4; at rate 0 k=0 so nothing is filtered) = 30 runs.
- Pilot first: unet_scratch, rate 0, seed 0. Check time/epoch (<60 s, else shrink model/epochs and log here) and val Dice > 0.7. After the pilot the hyper-parameters are frozen; no tuning on test.

## Audit filter
- `Calibrated(ref_cases)` is fitted on the CLEAN masks of the 40 val cases (no manifest, no train-set info).
- Score every train case with `Calibrated.combined(img, mask)` on the (possibly damaged) 3D mask; drop the top round(k*160) cases, k = nominal rate. Retrain both models with the same seeds.
- Report precision/recall of removing damaged cases against the manifest, overall and per kind.

## Quantization (Step 5)
- Best FP32 per architecture = rate 0, seed 0. ONNX opset 17, check max abs diff vs PyTorch. ORT static INT8 with 100 train-slice calibration images; dynamic only if static fails (report which).
- Test slices: Dice, file size (MB), median/p95 latency, single thread, batch 1, 20 warm-up + 200 timed runs. Hardware label: "4-core cloud CPU" (not a MacBook). ORT version recorded.

## Predictions (fixed before the main run)
- P1: test Dice at rate 0.4 is lower than at rate 0 for both models, by more than 1 SD over the 3 seeds.
- P2: the pretrained-encoder model loses less Dice from noise than the scratch model (may be false).
- P3: filtering recovers part but not all of the loss; filter recall < 0.8 because shifts are missed.
- P4: INT8 loses < 0.01 Dice on both models.
- P5: INT8 file is about 3-4x smaller; latency gain uncertain. Expected direction: INT8 is NOT faster on this single-thread CPU for the small scratch U-Net (quant/dequant overhead), and about equal or slightly faster for the ResNet18 model.

## Decision rules
- If P1 fails for a model: report "this noise at this size did not matter for this model"; do not describe filtering as a recovery for it (still report the numbers).
- Null or negative results (filter no help, INT8 loses Dice or is slower) are reported as they are.
- No numbers in README/CV that are not in `results/*.csv`.

## Limits
One small public dataset (MSD Task04 Hippocampus), 2D slices ~50x35 px, method demo, no clinical claim. Damage is synthetic and Dice 0.8 is not matched to real annotator disagreement. 3 training seeds but ONE split and ONE noise realisation; SD is across training seeds only. Post-training INT8 on a cloud CPU, not an embedded target, no QAT. ResNet18 ImageNet encoder is a pretrained encoder, not a foundation model.

## Step 5b - Synthetic data arm (optional; runs only after Steps 1-6 are finished)
Fixed before any 5b run.
- Generator: extend `audit/synth.py` (do not modify the original; put the extension in `src/synth_dr.py`) with randomised pose, size, contrast, blur and noise level. 500 synthetic 3D volumes, fixed seed, sliced and padded exactly like the real data. Generator parameters saved to `results/synth_params.csv`.
- Conditions (Model A, `unet_scratch`, seeds 0,1,2, MLflow, same loss/epochs/early stopping on CLEAN real val Dice; all tested on the CLEAN real test cases 200-259):
  (i) real only: the first 20 cases of the train split (indices 0-19), clean masks, from scratch;
  (ii) pretrain on the 500 synthetic volumes, then fine-tune on the same 20 real cases;
  (iii) synthetic only, zero real training cases (checkpoint chosen on real val Dice).
- Realism numbers, real vs synthetic: contrast-to-noise ratio inside vs outside the mask, and intensity-histogram distance (definition fixed in code before the run).
- P6: (iii) is clearly worse than (i), because a hippocampus is not an ellipsoid.
- P7: (ii) beats (i) by more than 1 SD over the 3 seeds in the 20-case regime (uncertain, may be false).
- Decision rule: if (ii) does not beat (i), report "synthetic pretraining did not help at this realism level"; the generator is NOT tuned on the test cases (one generator configuration, one run).
- Limits: synthetic shapes are simple and far from real anatomy; this is a sim-to-real test of a toy generator, not realistic image simulation.

## Pilot result (unet_scratch, rate 0, seed 0)
See `results/pilot_log.txt`: ~23-28 s/epoch (limit 60 s) and val Dice > 0.7 from epoch 0, so no model/epoch change was needed; hyper-parameters in `config.py` are frozen as written above.

## Deviation log
- MLflow 3.17 refuses the `./mlruns` file store (maintenance mode), so runs are logged to a local SQLite store `mlflow.db` instead (same content: params, seed, split hash, val/test Dice, train time). No effect on results.
- Run-budget rule (agreed with owner, 2026-10-27): the full plan (18 unfiltered + 12 filtered = 30 runs) stays unchanged for now. Loop order is rate -> filtered -> model -> seed, so the first ResNet18 run starts after the three rate-0 scratch runs. The run budget is reviewed once its s/epoch is known. Any reduction (e.g. fewer seeds or epochs for some arm) is decided on TIME ONLY, never on Dice, needs the owner's approval, and is logged here with the date before it takes effect. Runs already finished are never deleted.
