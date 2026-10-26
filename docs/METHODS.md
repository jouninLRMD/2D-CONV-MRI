# Methods in detail

This page describes what each script does, with the exact settings used for the paper. It follows the order in which `reproduce.py` runs the stages.

## Data

The study uses version 1.1.2 of the Stroke Outcome Optimization Project (SOOP), OpenNeuro accession ds004889, released under CC0 in BIDS format. SOOP has 1,715 MRI sessions of patients admitted with acute ischemic stroke to one comprehensive stroke center in South Carolina, with FLAIR and T1-weighted (T1w) volumes, the DWI trace image and the apparent diffusion coefficient (ADC) map, expert lesion masks drawn on the trace image, and clinical variables in `participants.tsv`, among them age, NIHSS at admission, prior stroke, body mass index (BMI) and the modified Rankin Scale (mRS) at discharge.

`scripts/download_soop.py` downloads every file it needs from the OpenNeuro S3 bucket (with the OpenNeuro API as a second mirror) and checks each one against the size and SHA-256 recorded in `data/soop_manifest.csv`. It downloads the DWI of all sessions, because the cohort definition needs to know which sessions have a usable DWI, and the FLAIR and T1w volumes only of the 718 patients of the study, listed in `data/labelled_participants.txt`. The download can be interrupted and restarted at any time.

## Shared preprocessing (same code as CALM-Stroke)

The steps below are shared with the companion study, [CALM-Stroke](https://github.com/jouninLRMD/CALM-Stroke), so both studies use the same patients, the same images in MNI space and the same five folds.

* `src/preprocess_dwi.py`: brain mask, relative ADC, affine registration of the trace image to the MNI152 2 mm template with ANTsPy, resampling of the expert lesion masks, lesion features and quality-control images.
* `src/cohort_splits.py`: the cohort table `results/cohort_all.csv`. A patient is part of the study ("labelled") when the discharge mRS, age and NIHSS are recorded and the DWI passed quality control: 718 patients. Poor outcome is mRS 3 to 6 (337 patients). The five folds are stratified by outcome and magnetic field strength with seed 2026 and stored in the column `fold`.
* `src/atlas_features.py` and `src/build_cache.py`: the arterial territory atlas on the image grid and a memory-mapped cache of the MNI-space DWI volumes and lesion masks (`<work>/cache/X.npy`, `M.npy`, `index.csv`). This study only uses the trace channel and the lesion mask of the cache.

## Slices (`src/thesis_slices.py`)

For every patient, 20 slices of 224 x 224 pixels are made for five inputs and stored as uint8 arrays of shape (718, 20, 224, 224) in `<work>/thesis`:

| Input | How the slices are taken |
|---|---|
| `flair_native`, `t1_native` | As in the thesis: the volume as stored, intensities clipped at the 1st and 99th percentiles and scaled to [0, 1], 20 equidistant slices along the third array axis over its full range, bilinear resize. The thesis treated these as axial slices, but the third axis is coronal for 164 FLAIR series and sagittal or coronal for 45 T1w series, and the outer slices hold little or no brain. |
| `flair_axial`, `t1_axial` | Corrected: the volume is reoriented to the closest RAS orientation, and 20 equidistant axial slices are taken between the first and last slice whose foreground (voxels above the 10th percentile of positive values) covers more than 2% of the slice. Same normalization and resize. |
| `dwi_axial` | The DWI trace image in MNI space from the cache, normalized the same way over the brain, 20 equidistant axial slices over the brain extent. |
| `dwi_mask` | The expert acute lesion mask on the same 20 DWI slices (nearest-neighbor resize), used only by the Grad-CAM analysis. |

`ids.csv` gives the row order of the arrays and `planes.csv` the anatomical plane of the stored third axis of every FLAIR and T1w volume.

## Networks (`src/thesis_models.py`)

The three architectures follow the original thesis code ([LauraDonado/TFE](https://github.com/LauraDonado/TFE)), re-implemented in PyTorch.

| Network | Image branch | Clinical branch | Fusion head |
|---|---|---|---|
| Basic CNN | three blocks of 3 x 3 convolution (32, 64, 128 filters), ReLU and 2 x 2 max pooling, then global average pooling; trained end to end | Dense(32) + ReLU | Dense(64) + ReLU + Dropout(0.3), Dense(32) + ReLU + Dropout(0.3), Dense(1) |
| VGG16 | ImageNet weights (torchvision `IMAGENET1K_V1`), frozen, global average pooling (512 features) | Dense(64) + ReLU + Dropout(0.3) | Dense(128) + ReLU + Dropout(0.4), Dense(1) |
| EfficientNet-B0 | ImageNet weights (torchvision `IMAGENET1K_V1`), frozen, global average pooling (1,280 features) | as VGG16 | as VGG16 |

For the pretrained networks, the grey-level slice is copied to three channels and normalized with the ImageNet mean and standard deviation. Because the backbones are frozen, their features are computed once per slice and cached in `<work>/thesis/feat_<arch>_<input>.npy`; the cache is rebuilt automatically when the slice array is newer.

Training: Adam with learning rate 1e-4, batches of 16 slices, binary cross-entropy (equivalent to the two-class softmax of the thesis), at most 50 epochs, early stopping on the validation loss with patience 8, and the weights of the best epoch restored. Every slice is a training sample that carries the label of its patient. Mixed precision (bfloat16) is used on the GPU. One seed (0) per configuration.

## The factors that are varied

| Option | Values | Meaning |
|---|---|---|
| `--input` | `mix_native`, `flair_axial`, `t1_axial`, `dwi_axial` (also `flair_native`, `t1_native`) | `mix_native` is the thesis input: the 20 FLAIR and the 20 T1w slices of a patient, both as stored, pooled as samples of the same patient. |
| `--clin` | `thesis`, `fold`, `none` | `thesis`: age, NIHSS, prior stroke and BMI with missing values set to zero (a BMI of zero cannot occur, so missingness becomes a value the network can learn). `fold`: missing values replaced by the median of the training patients, a missing-value indicator added, and standardization fitted on the training patients. `none`: image only, no clinical branch. |
| `--split` | `thesis`, `cv5` | `thesis`: the sorted list of patients split with two stratified `train_test_split` calls with `random_state=42` (15% test, then 17.65% of the rest for validation), which gives 502 training, 108 validation and 108 test patients, exactly as in the thesis. `cv5`: the five frozen folds; inside the training part of each fold a stratified 15% of patients (seed equal to the fold number) is the validation set. |
| `--gradcam` | flag | For DWI runs only: Grad-CAM on held-out slices with lesion (see below). |

Slices of one patient are never split across partitions in any design.

`reproduce.py` runs nine configurations for each of the three networks (27 runs):

| Step | Input | Clinical | Split |
|---|---|---|---|
| Thesis setting | FLAIR + T1w as stored | thesis | thesis |
| Cross-validation | FLAIR + T1w as stored | thesis | cv5 |
| In-fold clinical handling | FLAIR + T1w as stored | fold | cv5 |
| Reoriented FLAIR | FLAIR axial | fold, none | cv5 |
| Reoriented T1w | T1w axial | fold, none | cv5 |
| DWI | DWI axial | fold, none | cv5 |

## Evaluation (`src/thesis_eval.py`)

* **Slice level and patient level.** The thesis computed accuracy and AUC over test slices. Here both are reported. The patient probability is the mean of the slice probabilities (over both sequences for FLAIR + T1w). Accuracy uses a threshold of 0.5, as in the thesis.
* **Confidence intervals.** 2,000 bootstrap resamples of patients (seed 0). For slice-level AUC the resampling unit is also the patient.
* **Clinical logistic model.** A logistic regression with the same four variables (median imputation with missing indicators, standardization), fitted on the training and validation patients of each split and tested on the same patients as the networks. It needs no images, so `python src/thesis_eval.py` reproduces its AUCs exactly on any machine: 0.824 on the thesis test patients and 0.756 in cross-validation.
* **Paired difference.** Network minus clinical model on the same patients, with a paired bootstrap 95% CI.
* **Per fold.** The AUC of every fold, in `per_unit.csv`.
* **Variability of a single split.** The clinical model on 200 random stratified 85/15 splits (seeds 1000 to 1199), in `split_variability.csv`.
* **Grad-CAM.** For the DWI networks, Grad-CAM on the last convolutional feature map of every held-out slice whose expert acute lesion covers at least 20 pixels. The map is upsampled to the slice and restricted to the brain. It is compared with the lesion dilated by five pixels: the share of the map's energy inside the dilated lesion, the share of the brain that the dilated lesion occupies (what a map that ignored the lesion would give), and the pointing-game hit (the maximum of the map inside the dilated lesion).

## Differences from the original thesis code

The re-implementation keeps the architectures, the inputs of the thesis setting, the optimizer, the batch size, the early stopping and the split. It differs in the framework (PyTorch instead of the framework of the thesis code), in random initialization, and in the use of a single sigmoid output with binary cross-entropy instead of a two-class softmax, which is mathematically equivalent. On the thesis split the re-implementation gives slice-level AUCs 0.02 to 0.04 higher than the thesis reported (see [RESULTS.md](RESULTS.md)), close enough to show that it behaves like the original.

## Reproducibility notes

* Everything that does not involve training (cohort, folds, thesis split, clinical model, split variability) is deterministic and reproduces the paper exactly.
* Training was rerun from the slices onward on the same computer (one NVIDIA RTX 4000 Ada, PyTorch 2.11) on 26 October 2026, and the new predictions reproduced every table and number of the paper exactly. GPU training is not guaranteed to be bit-for-bit deterministic across GPUs, drivers or library versions, so on other hardware AUCs may differ slightly. The predictions in `results/thesis/oof` are the ones behind the paper's numbers.
* The original thesis used a 70/15/15 split of the same 718 patients. It described the cohort as 1,436 patients and about 28,700 slices, which counts each patient once per sequence (718 patients x 2 sequences = 1,436 series of 20 slices).
