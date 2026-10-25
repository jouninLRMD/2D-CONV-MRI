"""
Table 1 of the manuscript: characteristics of the 718 patients by discharge outcome.

Reads results/cohort_all.csv and writes results/thesis/paper/tab_cohort.tex (the table body). Values are median
(interquartile range) or n (%); p values from the Mann-Whitney U test (continuous variables) or the chi-squared test
(categorical variables). Lesion rows use the patients with an expert acute lesion mask. No images are needed.
Usage: python src/cohort_table.py
"""
import pandas as pd
from scipy.stats import mannwhitneyu, chi2_contingency
from common import RES

OUT = RES / "thesis" / "paper"; OUT.mkdir(parents=True, exist_ok=True)
coh = pd.read_csv(RES / "cohort_all.csv")
L = coh[coh.labelled].reset_index(drop=True)
Lc = L.copy(); good, poor = Lc[Lc.y_poor == 0], Lc[Lc.y_poor == 1]


def med(c, d):
    v = d[c].dropna(); return f"{v.median():.0f} ({v.quantile(.25):.0f}, {v.quantile(.75):.0f})"


def medf(c, d):
    v = d[c].dropna(); return f"{v.median():.1f} ({v.quantile(.25):.1f}, {v.quantile(.75):.1f})"


def npct(msk, d):
    return f"{int(msk.sum())} ({100 * msk.mean():.1f})"


def pcont(c):
    return mannwhitneyu(good[c].dropna(), poor[c].dropna()).pvalue


def pcat(col):
    return chi2_contingency(pd.crosstab(Lc[col], Lc.y_poor))[1]


def pf(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


Lc["race_c"] = Lc.race.fillna("u"); Lc["toast_c"] = Lc.toast.fillna(0)
Lc["mask_c"] = Lc.has_mask.astype(int); Lc["field_c"] = (Lc.field >= 3).astype(int)
Lc["chronic_c"] = (Lc.chronic_ml_filled.fillna(0) >= 1).astype(int)
good, poor = Lc[Lc.y_poor == 0], Lc[Lc.y_poor == 1]
R = [f"Patients, n & {len(Lc)} & {len(good)} & {len(poor)} & \\\\",
     f"Age, years & {med('age', Lc)} & {med('age', good)} & {med('age', poor)} & {pf(pcont('age'))} \\\\",
     f"Female sex & {npct(Lc.sex == 'F', Lc)} & {npct(good.sex == 'F', good)} & {npct(poor.sex == 'F', poor)} & {pf(chi2_contingency(pd.crosstab(Lc.sex, Lc.y_poor))[1])} \\\\",
     f"Race & & & & {pf(pcat('race_c'))} \\\\",
     f"\\quad Black & {npct(Lc.race == 'b', Lc)} & {npct(good.race == 'b', good)} & {npct(poor.race == 'b', poor)} & \\\\",
     f"\\quad White & {npct(Lc.race == 'w', Lc)} & {npct(good.race == 'w', good)} & {npct(poor.race == 'w', poor)} & \\\\",
     f"\\quad other or unknown & {npct(Lc.race.isna(), Lc)} & {npct(good.race.isna(), good)} & {npct(poor.race.isna(), poor)} & \\\\",
     f"Admission NIHSS & {med('nihss', Lc)} & {med('nihss', good)} & {med('nihss', poor)} & {pf(pcont('nihss'))} \\\\",
     f"BMI, kg/m$^2$ & {med('bmi', Lc)} & {med('bmi', good)} & {med('bmi', poor)} & {pf(pcont('bmi'))} \\\\",
     f"Prior stroke & {npct(Lc.priorstroke == 1, Lc)} & {npct(good.priorstroke == 1, good)} & {npct(poor.priorstroke == 1, poor)} & {pf(pcat('priorstroke'))} \\\\"]
TO = {1: "large artery", 2: "cardioembolism", 3: "small-vessel disease", 4: "other cause", 5: "cryptogenic"}
R.append("TOAST subtype & & & & " + pf(pcat("toast_c")) + " \\\\")
for k, v in TO.items():
    R.append(f"\\quad {v} & {npct(Lc.toast == k, Lc)} & {npct(good.toast == k, good)} & {npct(poor.toast == k, poor)} & \\\\")
R += [f"Scanned at 3 T & {npct(Lc.field >= 3, Lc)} & {npct(good.field >= 3, good)} & {npct(poor.field >= 3, poor)} & {pf(pcat('field_c'))} \\\\",
      f"Expert lesion mask & {npct(Lc.has_mask.astype(bool), Lc)} & {npct(good.has_mask.astype(bool), good)} & {npct(poor.has_mask.astype(bool), poor)} & {pf(pcat('mask_c'))} \\\\"]
mk = Lc[Lc.has_mask.astype(bool)]; mg, mp = mk[mk.y_poor == 0], mk[mk.y_poor == 1]
R += [f"Acute lesion, mL$^{{\\dagger}}$ & {medf('acute_ml', mk)} & {medf('acute_ml', mg)} & {medf('acute_ml', mp)} & {pf(mannwhitneyu(mg.acute_ml.dropna(), mp.acute_ml.dropna()).pvalue)} \\\\",
      f"Chronic infarct$^{{\\dagger}}$ & {npct(mk.chronic_ml_filled.fillna(0) >= 1, mk)} & {npct(mg.chronic_ml_filled.fillna(0) >= 1, mg)} & {npct(mp.chronic_ml_filled.fillna(0) >= 1, mp)} & {pf(chi2_contingency(pd.crosstab(mk.chronic_c, mk.y_poor))[1])} \\\\",
      f"Lesion mean rADC$^{{\\dagger}}$ & {mk.acute_radc_mean.median():.2f} ({mk.acute_radc_mean.quantile(.25):.2f}, {mk.acute_radc_mean.quantile(.75):.2f}) & "
      f"{mg.acute_radc_mean.median():.2f} ({mg.acute_radc_mean.quantile(.25):.2f}, {mg.acute_radc_mean.quantile(.75):.2f}) & "
      f"{mp.acute_radc_mean.median():.2f} ({mp.acute_radc_mean.quantile(.25):.2f}, {mp.acute_radc_mean.quantile(.75):.2f}) & {pf(mannwhitneyu(mg.acute_radc_mean.dropna(), mp.acute_radc_mean.dropna()).pvalue)} \\\\"]
(OUT / "tab_cohort.tex").write_text("\n".join(R).replace("%", "\\%") + "\n")
print("written", OUT / "tab_cohort.tex")
