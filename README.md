# 2D-CONV-MRI

Code and results for **"Do Two-Dimensional Convolutional Networks on Structural MRI Add to Clinical Variables for Predicting Discharge Outcome after Ischemic Stroke? A Patient-Level Re-Evaluation on a Public Dataset"** (L. S. Donado Pinto, L. R. Mercado-Diaz, H. F. Posada-Quintero, C. Barrera-Causil; submitted to *Brain Sciences*).

A master's thesis trained three 2D convolutional networks (a basic CNN, VGG16 and EfficientNet-B0) on FLAIR and T1-weighted slices plus four clinical variables to predict a poor discharge outcome (modified Rankin Scale 3 to 6) in the public [Stroke Outcome Optimization Project (SOOP)](https://openneuro.org/datasets/ds004889/versions/1.1.2), and reported AUCs close to 0.80 on one test split. This repository re-implements that pipeline and changes its evaluation one factor at a time: patient-level instead of slice-level metrics, five-fold cross-validation instead of one split, missing values handled inside each training fold, reoriented axial slices, and DWI instead of FLAIR. Every network is compared with a logistic model of the same clinical variables on the same patients, and Grad-CAM maps are scored against the expert lesion masks.

**Main findings** (718 patients; full tables in [docs/RESULTS.md](docs/RESULTS.md)):

* On the thesis split the re-implementation behaves like the thesis (slice-level AUCs 0.80 to 0.83), but the clinical model alone reaches 0.824 on the same 108 test patients. Only 6% of 200 random splits are that favorable.
* In five-fold cross-validation, FLAIR and T1w networks stay within 0.02 of the clinical model (0.749 to 0.771 against 0.756).
* DWI adds information: EfficientNet-B0 reaches 0.790 (+0.035 over the clinical model, 95% CI +0.018 to +0.052), and 0.787 from DWI alone.
* Patient-level scoring, the handling of missing values and reorientation each change the AUC by 0.01 or less; replacing the single split by cross-validation lowers every network by 0.04 to 0.07.

## What is in the repository

```
reproduce.py                 runs the whole study, stage by stage
scripts/download_soop.py     downloads SOOP from OpenNeuro and checks every file against data/soop_manifest.csv
scripts/fetch_atlas.py       downloads the arterial territory atlas (Liu et al., 2023), used by the image cache
src/common.py                paths, clinical-variable preparation and metrics (AUC, bootstrap CIs)
src/preprocess_dwi.py        DWI: brain mask, relative ADC, affine registration to MNI152 2 mm, lesion features, QC
src/cohort_splits.py         cohort table (718 patients) and the frozen five folds
src/atlas_features.py        arterial atlas on the image grid and lesion load per territory
src/build_cache.py           memory-mapped DWI volumes and lesion masks in MNI space
src/design_figure.py         Figure 1 (study design)
src/cohort_table.py          Table 1 (cohort characteristics by discharge outcome)
src/thesis_slices.py         2D slices: FLAIR and T1w as in the thesis and reoriented, DWI with lesion masks
src/thesis_models.py         the three networks of the thesis, trained under each combination of factors; Grad-CAM
src/thesis_eval.py           AUCs, confidence intervals, clinical model, split variability, Grad-CAM summary,
                             figures and the number macros of the manuscript
data/soop_manifest.csv       size and SHA-256 of every SOOP file
data/labelled_participants.txt   the 718 patients of the study
docs/METHODS.md              every step and setting in detail, and the differences from the thesis code
docs/RESULTS.md              all results: every run, every fold, split variability, Grad-CAM
docs/OUTPUTS.md              every output file, its columns, and the macro names used by the manuscript
results/                     cohort table, logs and the manuscript outputs (tables, figures, numbers)
```

The DWI preprocessing, the cohort and the folds are shared with the companion repository [CALM-Stroke](https://github.com/jouninLRMD/CALM-Stroke), so both studies use exactly the same patients and folds. If you already ran CALM-Stroke, point `CALM_DATA`, `CALM_WORK` and `CALM_RESULTS` to the same folders, run `python scripts/download_soop.py` to add the FLAIR and T1w volumes (CALM-Stroke downloads DWI only), and start at the `design` stage.

## Installation

Python 3.11 and a CUDA GPU with at least 8 GB of memory are needed for training (the frozen backbones are run once per slice, and only small heads are trained on top of them); the evaluation runs on a CPU. The paper used PyTorch 2.11 (CUDA 12.8) on one NVIDIA RTX 4000 Ada (20 GB).

```bash
git clone https://github.com/jouninLRMD/2D-CONV-MRI.git
cd 2D-CONV-MRI
conda env create -f environment.yml
conda activate 2d-conv-mri
# install the PyTorch build that matches your CUDA version, for example:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

The ImageNet weights of VGG16 (528 MB) and EfficientNet-B0 (21 MB) are downloaded by torchvision the first time they are used.

## Reproducing the paper

From the repository root:

```bash
python reproduce.py            # everything, from the download to the manuscript outputs
python reproduce.py --dry-run  # list the commands without running them
```

| Stage | What it does | Time (approximate) |
|---|---|---|
| `download` | SOOP metadata, lesion masks and DWI of all sessions; FLAIR and T1w of the 718 patients (31 GB) | depends on the connection |
| `atlas` | arterial territory atlas | seconds |
| `preprocess` | DWI preprocessing and registration of 1,715 sessions | 1 to 2 h with 6 CPU workers |
| `cohort`, `features`, `cache` | cohort and folds, atlas on the image grid, MNI image cache | 20 min |
| `design`, `table` | Figure 1 and Table 1 | seconds |
| `slices` | 2D slices of the five inputs for the 718 patients (about 4.3 GB) | 30 to 45 min |
| `smoke` | two-epoch test of the basic network on DWI, written to `results/thesis/smoke` | 2 min |
| `train` | the 27 network runs (3 networks x 9 configurations) | about 3 h |
| `evaluate` | every table, figure and number of the manuscript (`results/thesis/paper`) | 5 min |

Each stage can be run alone with `--only <stage>`, or the pipeline can be resumed with `--from <stage>`. Finished training runs are skipped, so the command can be interrupted and started again; `--force` retrains them, and `--arch effb0` limits training to one network. Every script can also be run directly, for example:

```bash
python src/thesis_models.py --arch effb0 --input dwi_axial --clin fold --split cv5 --gradcam
python src/thesis_models.py --help
```

If you rerun `slices`, the cached backbone features are rebuilt automatically, but existing predictions are kept; use `--force` (or delete `results/thesis/oof`) to retrain on the new slices.

### Without images or a GPU

The repository includes the predictions of every network in the paper, so all results can be recomputed on any computer, without images or a GPU:

* `results/thesis/oof`: the predicted probability of poor outcome for every held-out slice of the 27 network runs.
* `results/thesis/gradcam_*.csv` and `gradcam_examples_*.npz`: the Grad-CAM scores of every held-out DWI slice with lesion, and the examples shown in Figure 6.
* `results/thesis/planes.csv` and `slice_ids.csv`: the anatomical plane of every stored FLAIR and T1w volume, and the patient order of the slice arrays.

```bash
python reproduce.py --only design table evaluate
```

This rewrites Table 1, every figure except Figure 2 (which shows image slices), `all_runs.csv`, `per_unit.csv`, `split_variability.csv`, `gradcam_summary.csv`, `patient_predictions.csv` and `numbers_t.tex`, the file with every number quoted in the manuscript. The predictions come from a complete rerun of the pipeline from the slices onward (26 October 2026), which reproduced every table and number of the paper exactly. If `results/thesis/oof` is empty, `thesis_eval.py` computes only the clinical model and the split variability and writes them to `results/thesis/check_without_networks`, leaving the manuscript outputs untouched.

`results/thesis/paper` holds the outputs of the run reported in the paper: `tab_cohort.tex` (Table 1), `patient_predictions.csv` (patient-level probability of every network and of the clinical model, with outcome, fold and thesis partition), `all_runs.csv` (every configuration, slice-level and patient-level AUC with 95% CI, accuracy, difference from the clinical model), `per_unit.csv` (every fold), `split_variability.csv`, `gradcam_summary.csv`, the figures and `numbers_t.tex`.

## The configurations

| Factor | Thesis setting | Corrected setting |
|---|---|---|
| Slices | 20 slices along the third axis of the stored array, full range (often coronal or sagittal, outer slices empty) | volume reoriented to RAS, 20 axial slices over the head |
| Sequence | FLAIR and T1w pooled | FLAIR, T1w or DWI, each alone |
| Clinical variables | missing values set to zero | median imputation with missing indicator, fitted on the training patients; or no clinical variables |
| Split | one 70/15/15 split (502/108/108 patients, seed 42) | five-fold cross-validation over all 718 patients |
| Metric | over slices | per patient (mean of the slice probabilities), with bootstrap 95% CI |
| Reference | none | logistic model of the same four clinical variables on the same patients |

See [docs/METHODS.md](docs/METHODS.md) for every setting.

## Data and licenses

SOOP (OpenNeuro ds004889, version 1.1.2) is released under a CC0 license by Absher et al., *Scientific Data* 11, 839 (2024), doi:[10.1038/s41597-024-03667-5](https://doi.org/10.1038/s41597-024-03667-5). The arterial territory atlas is from Liu et al., *Scientific Data* 10, 74 (2023), doi:[10.1038/s41597-022-01923-0](https://doi.org/10.1038/s41597-022-01923-0), as distributed in the [SOOP demonstration repository](https://github.com/neurolabusc/StrokeOutcomeOptimizationProjectDemo) under a BSD 2-Clause license; it is downloaded, not redistributed here. The ImageNet weights are those distributed with torchvision. The original thesis code is at [LauraDonado/TFE](https://github.com/LauraDonado/TFE).

By default the data go to `data/` inside the repository. The environment variables `CALM_DATA` (download, preprocessed volumes, atlas), `CALM_WORK` (image cache, slices and backbone features) and `CALM_RESULTS` (tables and predictions) select other folders, for example:

```bash
export CALM_DATA=/data/soop CALM_WORK=/scratch/soop_work CALM_RESULTS=$PWD/results      # Linux, macOS
set CALM_DATA=D:\soop & set CALM_WORK=D:\soop_work                                     # Windows
```

## Troubleshooting

* **A download stops or a file fails its checksum.** Run `python scripts/download_soop.py` again; finished files are skipped and failed ones retried.
* **"not readable ... left empty" while making slices.** That FLAIR or T1w file is missing; run the download again and then `python reproduce.py --only slices --force`.
* **Out of GPU memory.** Only the basic network trains on full slices, in batches of 16; close other GPU jobs. The pretrained networks train small heads on cached features and need little memory.
* **The run was interrupted.** Start it again with `python reproduce.py --from train`; finished runs are skipped. If `slices` was interrupted, run it again before training, because a partly written slice array is not detected automatically.

## Citation

If you use this code, please cite the paper (the reference will be updated after publication):

```
L. S. Donado Pinto, L. R. Mercado-Diaz, H. F. Posada-Quintero and C. Barrera-Causil,
"Do two-dimensional convolutional networks on structural MRI add to clinical variables for predicting discharge
outcome after ischemic stroke? A patient-level re-evaluation on a public dataset," submitted to Brain Sciences, 2026.
```

## License

The code is released under the GNU General Public License v3.0 (see `LICENSE`).

## Contact

Luis R. Mercado-Diaz, Department of Biomedical Engineering, University of Connecticut.
Questions and issues are welcome through the GitHub issue tracker.
