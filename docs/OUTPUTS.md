# Files and formats

## Inputs kept in the repository

| File | Content |
|---|---|
| `data/soop_manifest.csv` | Every file of SOOP 1.1.2 with its size, SHA-256 and group (`meta`, `deriv`, `dwi`, `flair`, `t1`). |
| `data/labelled_participants.txt` | The 718 participant IDs of the study, one per line (used to limit the FLAIR and T1w download). |
| `results/cohort_all.csv` | One row per SOOP session: clinical variables, discharge mRS (`mrs`), poor outcome (`y_poor`), scanner, DWI quality control, lesion features, `labelled` (part of the study) and `fold` (0 to 4). |
| `results/folds_repeats.csv`, `results/cohort_summary.txt` | Repeated folds and cohort counts, shared with CALM-Stroke. |

## Large intermediate files (not in the repository)

Written under the work folder (`CALM_WORK`, by default `data/work`):

| File | Shape and type | Content |
|---|---|---|
| `cache/X.npy`, `cache/M.npy`, `cache/index.csv` | float16 (N, 2, 96, 112, 96), uint8 (N, 1, 96, 112, 96) | DWI trace and relative ADC in MNI space, expert lesion mask, row index |
| `thesis/<input>.npy` | uint8 (718, 20, 224, 224) | the slices of each input (`flair_native`, `t1_native`, `flair_axial`, `t1_axial`, `dwi_axial`) |
| `thesis/dwi_mask.npy` | uint8 (718, 20, 224, 224) | expert acute lesion on the DWI slices |
| `thesis/ids.csv`, `thesis/planes.csv` | | row order of the arrays; plane of the stored FLAIR and T1w volumes |
| `thesis/feat_<arch>_<input>.npy` | float16 (718, 20, 512 or 1280) | frozen VGG16 or EfficientNet-B0 features of every slice |

## Training outputs (`results/thesis`)

| File | Columns | Content |
|---|---|---|
| `oof/<arch>__<input>__<clin>__<split>__s0.csv` | `participant_id, unit, src, slice, p` | Predicted probability of poor outcome for every held-out slice. `unit` is `thesis_test` or `fold0` to `fold4`; `src` is the input the slice came from (`flair_native` or `t1_native` for the thesis input). |
| `gradcam_<tag>.csv` | `participant_id, slice, lesion_px, energy_in_lesion, lesion_area_frac, pointing_hit` | One row per held-out DWI slice with at least 20 lesion pixels. |
| `gradcam_examples_<arch>__<clin>.npz` | `img, mask, cam` | The first 12 slices of the Grad-CAM analysis, for Figure 6. |
| `logs/train_*.txt`, `logs/evaluate.txt` | | Console output of every run: training slices, stopping epoch and time per fold. |

## Manuscript outputs (`results/thesis/paper`)

| File | Content |
|---|---|
| `tab_cohort.tex` | Body of Table 1: cohort characteristics by discharge outcome, with Mann-Whitney U or chi-squared p values (`src/cohort_table.py`). |
| `all_runs.csv` | One row per configuration and for the clinical model: slice-level and patient-level AUC, bootstrap 95% CI (`lo`, `hi`), slice and patient accuracy, difference from the clinical model with its CI (`d_clin`, `d_lo`, `d_hi`). |
| `per_unit.csv` | AUC of every configuration in every fold (or on the thesis test set). |
| `patient_predictions.csv` | Patient-level probability of every configuration and of the clinical model, with outcome, mRS, fold and the thesis partition of each patient (written by `src/thesis_eval.py` when the predictions in `results/thesis/oof` are present). |
| `split_variability.csv` | Test AUC of the clinical model on 200 random 85/15 splits. |
| `gradcam_summary.csv` | Mean energy in the lesion, expected share by area and pointing hit rate per DWI network. |
| `numbers_t.tex` | Every number quoted in the manuscript, as LaTeX macros. |
| `figT0_design` to `figT5_splits` (`.pdf`, `.png`) | Figures 1 to 6 of the manuscript (T0 design, T1 inputs, T2 evaluation ladder, T3 sequences, T5 split variability, T4 Grad-CAM). |

## Macro names in `numbers_t.tex`

The manuscript uses `\NT{key}`. Keys are built from the configuration and the quantity:

* `<arch><input without underscores><clin><split><quantity>`, for example `effb0dwiaxialfoldcv5aucpatient` (EfficientNet-B0, DWI, in-fold clinical handling, five-fold CV, patient-level AUC). Quantities: `aucslice`, `aucpatient`, `lo`, `hi`, `accslice`, `accpatient`, `dclin`, `dlo`, `dhi`, and `foldmin`, `foldmax` for the range over folds.
* `clinthesis...` and `clincv5...` for the clinical model.
* `gc<arch><clin>energy`, `area`, `hit`, `n` for Grad-CAM.
* `svmed`, `svlo`, `svhi`, `svmin`, `svmax`, `svn` for the split variability.
* `npat`, `npoor`, and `plane<flair|t1><axial|sagittal|coronal>` for the cohort.
