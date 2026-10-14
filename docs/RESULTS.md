# Results

All numbers below were produced by `src/thesis_eval.py` from the predictions of the 27 network runs (one seed each) and are the ones reported in the paper. The files they come from are in `results/thesis/paper`. AUC is the area under the ROC curve for poor discharge outcome (mRS 3 to 6); 95% confidence intervals come from 2,000 bootstrap resamples of patients. "Difference" is the network minus the clinical logistic model on the same patients, with a paired bootstrap 95% CI.

## Cohort

718 patients with a recorded discharge mRS, age and NIHSS; 337 (46.9%) with a poor outcome. Thesis split: 502 training, 108 validation and 108 test patients. Five-fold cross-validation: folds of 144, 144, 144, 143 and 143 patients.

Anatomical plane of the third array axis of the stored volumes (the axis the thesis sliced along):

| Sequence | Axial | Sagittal | Coronal |
|---|---|---|---|
| FLAIR | 554 | 0 | 164 |
| T1w | 673 | 35 | 10 |

## 1. The thesis setting

FLAIR + T1w slices as stored, clinical variables with missing values set to zero, thesis split, metrics over slices.

| Network | Slice AUC (thesis) | Slice AUC (re-implementation) | Slice accuracy (thesis) | Slice accuracy (re-implementation) | Patient AUC (95% CI) |
|---|---|---|---|---|---|
| Basic CNN | 0.793 | 0.818 | 0.710 | 0.706 | 0.819 (0.734 to 0.892) |
| VGG16 | 0.756 | 0.797 | 0.719 | 0.699 | 0.801 (0.711 to 0.881) |
| EfficientNet-B0 | 0.797 | 0.825 | 0.708 | 0.723 | 0.835 (0.753 to 0.904) |
| Clinical logistic model | | | | | 0.824 (0.746 to 0.897) |

## 2. Every run

| Network | Input | Clinical variables | Split | Slice AUC | Patient AUC (95% CI) | Patient accuracy | Difference vs clinical model (95% CI) | Fold AUC range |
|---|---|---|---|---|---|---|---|---|
| Basic CNN | FLAIR + T1w, as stored | thesis handling | thesis split | 0.818 | 0.819 (0.734 to 0.892) | 0.694 | -0.005 (-0.022 to +0.012) |  |
| Basic CNN | FLAIR + T1w, as stored | thesis handling | five-fold CV | 0.753 | 0.754 (0.718 to 0.789) | 0.708 | -0.002 (-0.013 to +0.009) | 0.723 to 0.828 |
| Basic CNN | FLAIR + T1w, as stored | in-fold | five-fold CV | 0.752 | 0.752 (0.717 to 0.789) | 0.708 | -0.004 (-0.013 to +0.005) | 0.722 to 0.831 |
| Basic CNN | FLAIR, axial | in-fold | five-fold CV | 0.751 | 0.753 (0.718 to 0.789) | 0.708 | -0.003 (-0.015 to +0.010) | 0.732 to 0.842 |
| Basic CNN | FLAIR, axial | image only | five-fold CV | 0.601 | 0.634 (0.595 to 0.677) | 0.607 | -0.121 (-0.175 to -0.070) | 0.596 to 0.674 |
| Basic CNN | T1w, axial | in-fold | five-fold CV | 0.749 | 0.749 (0.713 to 0.785) | 0.706 | -0.007 (-0.018 to +0.005) | 0.721 to 0.833 |
| Basic CNN | T1w, axial | image only | five-fold CV | 0.583 | 0.608 (0.567 to 0.650) | 0.588 | -0.147 (-0.199 to -0.097) | 0.529 to 0.668 |
| Basic CNN | DWI, axial | in-fold | five-fold CV | 0.766 | 0.772 (0.737 to 0.807) | 0.721 | +0.016 (-0.003 to +0.036) | 0.752 to 0.823 |
| Basic CNN | DWI, axial | image only | five-fold CV | 0.687 | 0.708 (0.670 to 0.746) | 0.680 | -0.048 (-0.096 to +0.003) | 0.655 to 0.756 |
| VGG16 | FLAIR + T1w, as stored | thesis handling | thesis split | 0.797 | 0.801 (0.711 to 0.881) | 0.685 | -0.023 (-0.062 to +0.012) |  |
| VGG16 | FLAIR + T1w, as stored | thesis handling | five-fold CV | 0.750 | 0.757 (0.721 to 0.793) | 0.705 | +0.002 (-0.017 to +0.020) | 0.744 to 0.834 |
| VGG16 | FLAIR + T1w, as stored | in-fold | five-fold CV | 0.757 | 0.766 (0.730 to 0.802) | 0.705 | +0.010 (-0.003 to +0.023) | 0.732 to 0.835 |
| VGG16 | FLAIR, axial | in-fold | five-fold CV | 0.755 | 0.763 (0.727 to 0.797) | 0.712 | +0.007 (-0.008 to +0.021) | 0.718 to 0.837 |
| VGG16 | FLAIR, axial | image only | five-fold CV | 0.630 | 0.671 (0.632 to 0.711) | 0.623 | -0.084 (-0.131 to -0.037) | 0.646 to 0.721 |
| VGG16 | T1w, axial | in-fold | five-fold CV | 0.757 | 0.763 (0.728 to 0.798) | 0.705 | +0.008 (-0.006 to +0.021) | 0.722 to 0.847 |
| VGG16 | T1w, axial | image only | five-fold CV | 0.623 | 0.667 (0.628 to 0.706) | 0.606 | -0.089 (-0.136 to -0.042) | 0.641 to 0.697 |
| VGG16 | DWI, axial | in-fold | five-fold CV | 0.764 | 0.777 (0.742 to 0.812) | 0.740 | +0.022 (+0.007 to +0.036) | 0.751 to 0.844 |
| VGG16 | DWI, axial | image only | five-fold CV | 0.686 | 0.773 (0.739 to 0.808) | 0.716 | +0.018 (-0.023 to +0.059) | 0.759 to 0.806 |
| EfficientNet-B0 | FLAIR + T1w, as stored | thesis handling | thesis split | 0.825 | 0.835 (0.753 to 0.904) | 0.732 | +0.011 (-0.016 to +0.040) |  |
| EfficientNet-B0 | FLAIR + T1w, as stored | thesis handling | five-fold CV | 0.758 | 0.770 (0.734 to 0.804) | 0.708 | +0.014 (-0.002 to +0.029) | 0.735 to 0.840 |
| EfficientNet-B0 | FLAIR + T1w, as stored | in-fold | five-fold CV | 0.755 | 0.769 (0.734 to 0.804) | 0.706 | +0.013 (+0.000 to +0.025) | 0.725 to 0.840 |
| EfficientNet-B0 | FLAIR, axial | in-fold | five-fold CV | 0.756 | 0.771 (0.736 to 0.806) | 0.708 | +0.015 (-0.002 to +0.032) | 0.724 to 0.836 |
| EfficientNet-B0 | FLAIR, axial | image only | five-fold CV | 0.631 | 0.675 (0.636 to 0.714) | 0.617 | -0.081 (-0.129 to -0.035) | 0.630 to 0.733 |
| EfficientNet-B0 | T1w, axial | in-fold | five-fold CV | 0.757 | 0.770 (0.736 to 0.805) | 0.703 | +0.015 (-0.002 to +0.031) | 0.718 to 0.847 |
| EfficientNet-B0 | T1w, axial | image only | five-fold CV | 0.630 | 0.667 (0.625 to 0.707) | 0.620 | -0.089 (-0.137 to -0.043) | 0.624 to 0.703 |
| EfficientNet-B0 | DWI, axial | in-fold | five-fold CV | 0.769 | 0.790 (0.756 to 0.824) | 0.735 | +0.035 (+0.018 to +0.052) | 0.763 to 0.851 |
| EfficientNet-B0 | DWI, axial | image only | five-fold CV | 0.704 | 0.787 (0.753 to 0.821) | 0.733 | +0.032 (-0.008 to +0.072) | 0.753 to 0.816 |
| Clinical logistic model | none | in-fold | thesis split | | 0.824 (0.746 to 0.897) | | | |
| Clinical logistic model | none | in-fold | five-fold CV | | 0.756 (0.719 to 0.793) | | | |

## 3. Variability of a single split

The clinical logistic model was fitted and tested on 200 random stratified splits with 15% test patients (`split_variability.csv`). Test AUC: median 0.759, central 95% range 0.677 to 0.840, minimum 0.630, maximum 0.870. The thesis split gave 0.824; 12 of the 200 splits (6%) reached that value or more.

## 4. Grad-CAM against the expert lesion (DWI networks, five-fold CV)

Held-out slices whose expert acute lesion covers at least 20 pixels. "Energy in lesion" is the share of the Grad-CAM map inside the lesion dilated by five pixels; "Expected by area" is the share of the brain that the dilated lesion occupies, the value a map that ignores the lesion would give; "Pointing hits" is the share of slices whose maximum falls inside the dilated lesion.

| Network | Clinical variables | Slices | Energy in lesion | Expected by area | Ratio | Pointing hits |
|---|---|---|---|---|---|---|
| Basic CNN | in-fold | 4,353 | 0.020 | 0.102 | 0.19 | 0.008 |
| Basic CNN | image only | 4,431 | 0.154 | 0.101 | 1.52 | 0.238 |
| EfficientNet-B0 | in-fold | 4,415 | 0.149 | 0.101 | 1.47 | 0.217 |
| EfficientNet-B0 | image only | 4,421 | 0.153 | 0.101 | 1.51 | 0.238 |
| VGG16 | in-fold | 4,427 | 0.130 | 0.101 | 1.29 | 0.195 |
| VGG16 | image only | 4,429 | 0.138 | 0.101 | 1.36 | 0.209 |

## Reading the results

* On the thesis split, the re-implementation behaves like the thesis (slice AUCs 0.02 to 0.04 higher), but the clinical model alone does as well on the same 108 test patients.
* Moving from the single split to five-fold cross-validation lowers every network by 0.04 to 0.07. Scoring per patient, in-fold handling of missing values and reorientation each change the AUC by 0.01 or less.
* In cross-validation, FLAIR and T1w networks with clinical variables stay within 0.02 of the clinical model; from the image alone they are well below it.
* DWI networks are above the clinical model (EfficientNet-B0: +0.035, 95% CI +0.018 to +0.052), and EfficientNet-B0 on DWI alone reaches 0.787.
* The pretrained DWI networks put 1.3 to 1.5 times more Grad-CAM energy on the infarct than expected from its area; the basic network with clinical variables hardly looks at it.
