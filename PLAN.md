# Label noise -> training -> audit filter -> INT8 quantization (minimal closed loop, 2D slices)

Hand-off plan for a separate Claude session. Owner: Feixiang Xiao. Everything here is for a CV project (ZEISS vision/edge-AI internship); the CV bullets are filled with measured numbers only after the runs.

## Question (one only)
When training masks are damaged in known ways, how much does a small 2D segmentation model lose, how much of that does the existing audit check win back by filtering, and what does post-training INT8 quantization cost in Dice versus CPU speed?

## Hard rules (read first)
- Scope is the minimum: 2D slices only, CPU only, small models. No 3D, no extra datasets, no GPU. Do not add steps.
- Write predictions and decision rules in DESIGN.md BEFORE the main run. Pilot first, main run once, no re-tuning on the test cases.
- Split by case, never by slice (same patient in train and test = leakage). Test cases always get CLEAN masks.
- Report what happened, including null results (e.g. filtering does not help, quantization loses Dice). Do not delete bad results.
- Never claim: embedded deployment, QAT, "vision foundation model" (a ResNet18 ImageNet encoder is a pretrained encoder), clinical validity, 3D results. This is post-training INT8 on a laptop CPU.
- Git: commit with her identity only. NO "Co-Authored-By" / Claude / "Generated with" lines in commits or PR text; scan before any push. Do not push anything without asking her.
- Do not invent numbers. Every number in README/CV comes from `results/*.csv`.

## Existing material to reuse (do not rewrite)
`audit/` is an unchanged copy of the original audit code; `data_raw/msd_hippocampus/Task04_Hippocampus/` holds the extracted MSD Task04 data (28 MB, already in the repo, no download needed). Do not rewrite the detectors. This runs in a CLOUD container (about 4 CPU cores, not her MacBook): install torch, torchvision, onnx, mlflow, onnxruntime with pip (ask her first); the Step 5 latency numbers are cloud-CPU numbers and must be labelled "4-core cloud CPU" in README and RESULTS_FOR_CV.md, never as MacBook numbers. Git identity: ask her for name and email before committing, or leave changes uncommitted for her.
- `audit/load_msd.py` `load_cases(n, seed)`: 260 cases, binary hippocampus mask, per-case robust [0,1] rescaling. Volumes are about 35 x 50 x 35 voxels (checked: mean 35.4 x 50.0 x 35.7, min 31 x 40 x 24, max 43 x 59 x 47).
- `audit/damage.py` `damage(mask, kind, target_dice, rng)`: kinds = wrong size, shift, extra island, set by target Dice via bisection. `clean_mask`, `dice`.
- `audit/detectors.py` `Calibrated(ref_cases)` with `.blind`, `.edge`, `.components`, `.rules`, `.combined`. Calibrate on clean masks only.
- Earlier result to cite: `1016_vialytics_CV_audit/README.md` (AUROC table; shift at Dice 0.9 is caught by nothing, island caught by components).
- Installed: torch, torchvision, onnxruntime 1.27. NOT installed: `onnx`, `mlflow` -> `pip install onnx mlflow` (ask her before installing).

## Steps

### Step 0 - Setup (30 min)
Create `DESIGN.md`, `README.md`, `results/`, `src/`, `run_all.py`. Pin seeds (numpy, torch, python) in one `config.py`.

### Step 1 - Data (1 h)
- `load_cases(seed=0)`, then split by CASE: train 0-159, val 160-199, test 200-259 (fixed, written in DESIGN.md).
- 2D slices: axial slices that contain at least 1 mask voxel (plus nothing else), pad/crop to 64 x 64 (images are ~50 x 35).
- Store per slice: case id, slice index. Check: no case id appears in two splits (assert).

### Step 2 - Noise injection (1 h)
- Noise rate r in {0, 0.2, 0.4} = fraction of TRAIN CASES whose masks are replaced by a damaged version (damage applied to the 3D mask, then sliced, so slices of one case stay consistent).
- Damage kind per damaged case: drawn uniformly from {wrong size, shift, island}, target Dice 0.8 (fixed; Dice 0.9 is nearly undetectable per the earlier audit, so it is a separate stress test only if time allows).
- Validation masks: clean (so model selection is not contaminated). Test masks: clean.
- Save a CSV `results/noise_manifest.csv`: case id, rate, kind, achieved Dice. Seeds fixed.

### Step 3 - Models and training (2-3 h, mostly waiting)
- Model A: small U-Net from scratch (about 4 levels, base 16 channels).
- Model B: U-Net with ImageNet-pretrained ResNet18 encoder (torchvision weights; adapt 1 input channel by averaging conv1 weights over RGB).
- Same loss (BCE + Dice), same epochs, same augmentation (flip only), Adam, fixed seeds. Early stopping on CLEAN val Dice.
- 3 noise rates x 2 models x 3 seeds = 18 runs. Log every run in MLflow (local `mlruns/`): params, seed, split hash, val/test Dice, train time.
- First run a PILOT: 1 model, rate 0, 1 seed, to measure time per epoch and that the model reaches a sensible Dice (> 0.7 on val). If one epoch takes more than 60 s, shrink the model or the epochs, and write the change in DESIGN.md.

### Step 4 - Audit filter (1 h)
- For each noisy train set: run `Calibrated.combined` on (image, damaged mask) of every train case; drop the top-k% most suspicious cases, with k = the nominal noise rate (so no information leak from the manifest). Retrain both models on the filtered set (same seeds).
- Report: filter precision/recall of removing damaged cases (from the manifest), and test Dice before vs after filtering at each rate.
- Note the earlier finding: shifts are mostly missed, so recall will be partial. That is expected and must be reported as it is.

### Step 5 - Quantization (1-2 h)
- Pick the best FP32 model per architecture (rate 0 run, seed 0), export to ONNX (opset 17), check ONNX output matches PyTorch (max abs diff).
- ONNX Runtime post-training static INT8 quantization with 100 calibration slices from the TRAIN split. Also dynamic quantization if static fails on an op; report which was used.
- On the test slices compare FP32 vs INT8: Dice, file size (MB), median and p95 latency per slice on CPU (single thread, batch 1, 200 runs after 20 warm-up runs). Record the machine (MacBook Air M4, CPU, ONNX Runtime version).
- If INT8 is slower than FP32 on this CPU, report that (it happens on small models).

### Step 5b - Synthetic data arm (optional, only after Steps 1-6 are done; about 2-3 h)
Goal: generate synthetic images with perfect masks and EVALUATE whether they help on real data.
- Generator: extend `audit/synth.py` (ellipsoid structure on smooth noisy background) with randomised pose, size, contrast, blur and noise level (domain randomisation). 500 synthetic 3D volumes, sliced the same way as the real data, fixed seed. Save the generator parameters.
- Three conditions, all tested on the same CLEAN real test cases (200-259), small model (Model A), 3 seeds, logged in MLflow:
  (i) real only: 20 real training cases, from scratch;
  (ii) synthetic pretraining on the 500 volumes, then fine-tune on the same 20 real cases;
  (iii) synthetic only, zero real cases (sim-to-real gap).
- Also report two simple realism numbers, real vs synthetic: contrast-to-noise ratio inside vs outside the mask, and intensity-histogram distance.
- Predictions (fix before running): P6: (iii) is clearly worse than (i) because a hippocampus is not an ellipsoid; P7: (ii) beats (i) by more than 1 SD in the 20-case regime (uncertain, may be false). Decision rule: if (ii) does not beat (i), report "synthetic pretraining did not help at this realism level" and do not tune the generator on the test cases.
- Limits to state: the synthetic shapes are simple and far from real anatomy; this is a sim-to-real test of a toy generator, not realistic image simulation.

### Step 6 - Report (1-2 h)
- `run_all.py` reproduces everything from raw data to `results/` and the figure. Commands in README.
- `results/main_table.csv`: rows = model x rate x filtered(y/n), mean +- SD over 3 seeds. One figure: test Dice vs noise rate, with and without filtering. One table: FP32 vs INT8.
- README sections: question, data, what was done, results, what it means, limits (copy the honest-limits list below), files, reproduce.

## Predictions to write in DESIGN.md before the main run (change the numbers only before running)
- P1: test Dice at rate 0.4 is lower than at rate 0 for both models (with 3 seeds, difference larger than 1 SD).
- P2: the pretrained-encoder model loses less from noise than the from-scratch model (this may be false; state it as a prediction).
- P3: filtering recovers part but not all of the loss; filter recall for damaged cases is below 0.8 because shifts are missed.
- P4: INT8 loses less than 0.01 Dice on both models.
- P5: INT8 is smaller (about 3-4x) but latency gain is uncertain on this CPU (state which direction you expect).
Decision rule: if P1 fails (noise does not hurt), say so and do not continue to interpret filtering as a "recovery"; report it as "this noise at this size did not matter for this model".

## Outputs the owner needs back (for CV / cover letter)
A short `RESULTS_FOR_CV.md` with only measured numbers (including Step 5b: Dice for conditions i/ii/iii if it was done): Dice at each rate with and without filtering for both models; filter recall/precision; FP32 vs INT8 Dice, size, latency; number of runs; seeds; time to run `run_all.py`. Plus one line on what did NOT work.

## Limits to state in the README
- One small public dataset (MSD Task04 Hippocampus, CC-BY-SA 4.0), 2D slices about 50 x 35 px, method demo only, no clinical claim.
- Damage is synthetic; its size (Dice 0.8) is not matched to real annotator disagreement.
- 3 seeds, one split; SD is across seeds, not across splits.
- Post-training INT8 on a laptop CPU, not on an embedded target; no QAT.
- Pretrained encoder is an ImageNet ResNet18, not a vision foundation model.

## Time budget
About 1-1.5 working days in total: setup/data/noise 2.5 h, training 2-3 h, filter 1 h, quantization 1-2 h, report 1-2 h.
