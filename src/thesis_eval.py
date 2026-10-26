"""
Step 3 of the study: evaluation, tables, figures and the number macros of the manuscript.

Reads results/thesis/oof/*.csv (one row per held-out slice) and results/thesis/gradcam_*.csv.
  * slice-level AUC (every slice counted as a sample, as in the thesis) and patient-level AUC (mean probability over
    a patient's slices), with patient-level bootstrap 95% CIs (2,000 resamples)
  * clinical logistic model on the same splits (thesis split and five folds) and the paired bootstrap difference
    between every network and that model
  * the evaluation ladder from the thesis setting to the corrected setting (Figure 3, Table 2)
  * sequence comparison: FLAIR, T1w and DWI, with and without clinical variables (Figure 4)
  * clinical model on 200 random 85/15 splits (Figure 5)
  * Grad-CAM: share of the map inside the expert lesion against the share expected from its area, and the
    pointing-game hit rate (Figure 6)
Outputs (results/thesis/paper): all_runs.csv, per_unit.csv, split_variability.csv, gradcam_summary.csv,
  patient_predictions.csv, figT1_inputs to figT5_splits (.pdf and .png) and numbers_t.tex
Usage: python src/thesis_eval.py
"""
import logging
import numpy as np, pandas as pd
from common import RES, WORK, auc, boot_auc_ci, paired_boot_delta, clinical_matrix
from thesis_models import splits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams.update({"font.family": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 7, "axes.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42})
T = RES / "thesis"; OUT = T / "paper"
if not list((T / "oof").glob("*.csv")):
    # no network predictions yet: keep the manuscript outputs untouched and write the clinical-only analyses elsewhere
    OUT = T / "check_without_networks"
    print(f"No predictions in {T / 'oof'}; writing the clinical model and split-variability results to {OUT}")
OUT.mkdir(parents=True, exist_ok=True)
coh = pd.read_csv(RES / "cohort_all.csv").set_index("participant_id")
# patient order of the slice arrays; without them (no images) the labelled cohort in the same order
ids = (pd.read_csv(WORK / "thesis" / "ids.csv").participant_id.values if (WORK / "thesis" / "ids.csv").exists()
       else coh.index[coh.labelled].values)
L = coh.loc[ids]
ARCH = {"basic": "Basic CNN", "vgg16": "VGG16", "effb0": "EfficientNet-B0"}
INP = {"mix_native": "FLAIR + T1w, acquisition axis", "flair_native": "FLAIR, acquisition axis", "t1_native": "T1w, acquisition axis",
       "flair_axial": "FLAIR, axial", "t1_axial": "T1w, axial", "dwi_axial": "DWI, axial"}
CL = {"thesis": "clinical (thesis handling)", "fold": "clinical (in-fold)", "none": "image only"}


def clinical_baseline(split):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    Ld = L.reset_index(); p = pd.Series(np.nan, index=Ld.participant_id)
    for unit, tr, va, te in splits(Ld, split):
        fit = Ld[Ld.participant_id.isin(tr | va)]; tst = Ld[Ld.participant_id.isin(te)]
        Xf = clinical_matrix(fit, fit); Xt = clinical_matrix(tst, fit); sc = StandardScaler().fit(Xf)
        m = LogisticRegression(max_iter=2000).fit(sc.transform(Xf), fit.y_poor.astype(int))
        p[tst.participant_id] = m.predict_proba(sc.transform(Xt))[:, 1]
    return p.dropna()


rows = []
res = {}
groups = {}
for f in sorted((T / "oof").glob("*.csv")):
    groups.setdefault("__".join(f.stem.split("__")[:4]), []).append(f)
unit_rows = []
for cfg, fs in groups.items():
    arch, inp, clin, split = cfg.split("__")
    # several seeds: average the slice probabilities (seed ensemble)
    d = pd.concat([pd.read_csv(f) for f in fs]).groupby(["participant_id", "unit", "src", "slice"], as_index=False).p.mean()
    for u, du in d.groupby("unit"):
        pu = du.groupby("participant_id").p.mean(); yu = L.loc[pu.index, "y_poor"].values.astype(int)
        if 0 < yu.sum() < len(yu):
            unit_rows.append(dict(tag=cfg, unit=u, n=len(pu), auc=auc(yu, pu.values)))
    y_s = L.loc[d.participant_id, "y_poor"].values.astype(int)
    pat = d.groupby("participant_id").p.mean(); y_p = L.loc[pat.index, "y_poor"].values.astype(int)
    lo, hi = boot_auc_ci(y_p, pat.values)
    r = dict(tag=cfg, n_seeds=len(fs), arch=arch, input=inp, clin=clin, split=split, n_patients=len(pat), n_slices=len(d),
             auc_slice=auc(y_s, d.p.values), auc_patient=auc(y_p, pat.values), lo=lo, hi=hi,
             acc_slice=float(((d.p.values >= 0.5) == y_s).mean()), acc_patient=float(((pat.values >= 0.5) == y_p).mean()))
    res[cfg] = pat; rows.append(r)
R = pd.DataFrame(rows)
base = {s: clinical_baseline(s) for s in ["thesis", "cv5"]}
for s, p in base.items():
    y = L.loc[p.index, "y_poor"].values.astype(int); lo, hi = boot_auc_ci(y, p.values)
    R = pd.concat([R, pd.DataFrame([dict(tag=f"clinical__{s}", arch="logistic", input="none", clin="fold", split=s,
                                         n_patients=len(p), auc_patient=auc(y, p.values), lo=lo, hi=hi)])], ignore_index=True)
# paired difference against the clinical baseline of the same split
for i, r in R.iterrows():
    if r.arch != "logistic" and r.tag in res:
        p = res[r.tag]; b = base[r.split].reindex(p.index)
        y = L.loc[p.index, "y_poor"].values.astype(int)
        d, (dl, dh), _ = paired_boot_delta(y, p.values, b.values)
        R.loc[i, ["d_clin", "d_lo", "d_hi"]] = [d, dl, dh]
R.round(4).to_csv(OUT / "all_runs.csv", index=False)
# patient-level predictions of every configuration and of the clinical model, for reuse without retraining
PP = L[["y_poor", "mrs", "fold"]].copy()
_, tr_, va_, te_ = next(splits(L.reset_index(), "thesis"))
PP["thesis_split"] = ["train" if i in tr_ else "validation" if i in va_ else "test" if i in te_ else "" for i in PP.index]
for s_, p_ in base.items():
    PP[f"clinical__{s_}"] = p_.reindex(PP.index)
for cfg, p_ in res.items():
    PP[cfg] = p_.reindex(PP.index)
PP.round(5).to_csv(OUT / "patient_predictions.csv")
U = pd.DataFrame(unit_rows); U.round(4).to_csv(OUT / "per_unit.csv", index=False)


def split_variability(n_rep=200):
    """Clinical logistic model evaluated on n_rep random 85/15 patient splits (the test fraction of the thesis): spread of the
    test AUC one would report from a single split."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    Ld = L.reset_index(); y = Ld.y_poor.values.astype(int); out = []
    for k in range(n_rep):
        tr, te = train_test_split(np.arange(len(Ld)), test_size=0.15, stratify=y, random_state=1000 + k)
        fit = Ld.iloc[tr]; tst = Ld.iloc[te]
        Xf = clinical_matrix(fit, fit); Xt = clinical_matrix(tst, fit); sc = StandardScaler().fit(Xf)
        m = LogisticRegression(max_iter=2000).fit(sc.transform(Xf), y[tr])
        out.append(auc(y[te], m.predict_proba(sc.transform(Xt))[:, 1]))
    return np.array(out)


SV = split_variability(); pd.DataFrame(dict(auc=SV)).to_csv(OUT / "split_variability.csv", index=False)
print(R.reindex(columns=["tag", "auc_slice", "auc_patient", "lo", "hi", "d_clin"]).round(3).to_string())

# ------------------------------------------------------------------------------------------- figures
_pl = [f for f in (WORK / "thesis" / "planes.csv", T / "planes.csv") if f.exists()]   # from thesis_slices.py, or the copy kept in results
pl = pd.read_csv(_pl[0]) if _pl else None
cols = {"basic": "#7570b3", "vgg16": "#1b9e77", "effb0": "#d95f02"}
# Fig T1: what the networks see. Native-axis slices (thesis) against axial slices after reorientation, for one patient
# whose FLAIR was stored axially and one whose FLAIR was stored sagittally, plus DWI; and acquisition planes per sequence
try:
    TH = WORK / "thesis"; Xn = np.load(TH / "flair_native.npy", mmap_mode="r"); Xa = np.load(TH / "flair_axial.npy", mmap_mode="r")
    Tn = np.load(TH / "t1_native.npy", mmap_mode="r"); Dw = np.load(TH / "dwi_axial.npy", mmap_mode="r")
    cand = []
    if pl is not None:
        for want in ["axial", "sagittal"]:
            hit = np.where(pl.flair_plane.values == want)[0]
            if len(hit):
                cand.append((hit[0], want))
    if not cand:
        cand = [(0, "")]
    show = [3, 7, 10, 13, 16]
    rows_ = []
    for i, pln in cand:
        rows_ += [(Xn[i], f"FLAIR as stored ({pln})"), (Xa[i], "FLAIR, axial")]
    rows_ += [(Tn[cand[0][0]], f"T1w as stored ({pl.t1_plane.values[cand[0][0]] if pl is not None else ''})"), (Dw[cand[0][0]], "DWI trace, axial")]
    fig = plt.figure(figsize=(7.16, 1.05 * len(rows_)))
    gs = fig.add_gridspec(len(rows_), len(show) + 2, width_ratios=[1] * len(show) + [0.6, 2.0], wspace=0.04, hspace=0.12)
    for r_, (vol, lab) in enumerate(rows_):
        for c_, z in enumerate(show):
            a_ = fig.add_subplot(gs[r_, c_]); a_.imshow(vol[z], cmap="gray", vmin=0, vmax=255); a_.set_xticks([]); a_.set_yticks([])
            for sp in a_.spines.values():
                sp.set_visible(False)
            if c_ == 0:
                a_.set_ylabel(lab, fontsize=5.5, rotation=0, ha="right", va="center")
            if r_ == 0:
                a_.set_title(f"slice {z + 1} of 20", fontsize=5.5)
    if pl is not None:
        a_ = fig.add_subplot(gs[:, -1])
        P_ = pd.DataFrame({m: pl[f"{m}_plane"].value_counts() for m in ["flair", "t1"]}).reindex(["axial", "sagittal", "coronal"]).fillna(0)
        P_.columns = ["FLAIR", "T1w"]
        P_.T.plot.barh(stacked=True, ax=a_, color=["#1b9e77", "#d95f02", "#7570b3"], width=0.6)
        a_.set_xlabel("Patients"); a_.set_title("Plane of the stored third axis", fontsize=6.5); a_.legend(fontsize=5.5, loc="lower right")
    fig.savefig(OUT / "figT1_inputs.pdf", bbox_inches="tight", pad_inches=0.02); fig.savefig(OUT / "figT1_inputs.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close(fig)
except FileNotFoundError as e:
    print("figT1 skipped:", e)

# Fig T2: evaluation ladder (thesis setting -> corrected), patient-level AUC with CI, per architecture
ladder = [("Thesis setting,\nslice-level", "mix_native", "thesis", "thesis", "slice"),
          ("Same models,\npatient-level", "mix_native", "thesis", "thesis", "patient"),
          ("Five-fold CV,\npatient-level", "mix_native", "thesis", "cv5", "patient"),
          ("+ in-fold clinical\nhandling", "mix_native", "fold", "cv5", "patient"),
          ("+ axial FLAIR\n(reoriented)", "flair_axial", "fold", "cv5", "patient"),
          ("DWI instead of\nFLAIR", "dwi_axial", "fold", "cv5", "patient")]
fig, ax = plt.subplots(figsize=(7.16, 2.4))
for j, a in enumerate(["basic", "vgg16", "effb0"]):
    for i, (lab, inp, clin, split, lvl) in enumerate(ladder):
        r = R[(R.arch == a) & (R.input == inp) & (R.clin == clin) & (R.split == split)]
        if len(r):
            r = r.iloc[0]; v = r.auc_slice if lvl == "slice" else r.auc_patient
            x = i + (j - 1) * 0.2
            ax.errorbar(x, v, yerr=None if lvl == "slice" else [[v - r.lo], [r.hi - v]], fmt="o", ms=3.5, color=cols[a], lw=0.8,
                        label=ARCH[a] if i == 0 else None)
for s, ls in [("thesis", ":"), ("cv5", "--")]:
    rr = R[R.tag == f"clinical__{s}"]
    if len(rr):
        ax.axhline(rr.auc_patient.iloc[0], color="#555555", lw=0.7, ls=ls, label=f"clinical logistic ({'thesis split' if s == 'thesis' else 'five-fold CV'})")
ax.set_xticks(range(len(ladder))); ax.set_xticklabels([l[0] for l in ladder], fontsize=6.2); ax.set_ylabel("AUC, poor outcome (95% CI)")
ax.legend(fontsize=6, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0)); ax.set_ylim(0.5, 0.95)
fig.savefig(OUT / "figT2_ladder.pdf", bbox_inches="tight", pad_inches=0.02); fig.savefig(OUT / "figT2_ladder.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close(fig)

# Fig T3: sequence comparison (cv5, in-fold clinical and image only)
fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.3), sharey=True, gridspec_kw=dict(wspace=0.08))
for k, clin in enumerate(["fold", "none"]):
    for j, a in enumerate(["basic", "vgg16", "effb0"]):
        for i, inp in enumerate(["flair_axial", "t1_axial", "dwi_axial"]):
            r = R[(R.arch == a) & (R.input == inp) & (R.clin == clin) & (R.split == "cv5")]
            if len(r):
                r = r.iloc[0]; x = i + (j - 1) * 0.22
                ax[k].errorbar(x, r.auc_patient, yerr=[[r.auc_patient - r.lo], [r.hi - r.auc_patient]], fmt="o", ms=3.5, color=cols[a], lw=0.8,
                               label=ARCH[a] if i == 0 else None)
    rr = R[R.tag == "clinical__cv5"]
    if len(rr):
        ax[k].axhline(rr.auc_patient.iloc[0], color="#555555", lw=0.7, ls="--", label="clinical logistic")
    ax[k].set_xticks(range(3)); ax[k].set_xticklabels(["FLAIR", "T1w", "DWI"]); ax[k].set_title("with clinical variables" if clin == "fold" else "image only", fontsize=7)
ax[0].set_ylabel("Patient-level AUC (95% CI)"); ax[0].legend(fontsize=6); ax[0].set_ylim(0.45, 0.9)
fig.savefig(OUT / "figT3_sequences.pdf", bbox_inches="tight", pad_inches=0.02); fig.savefig(OUT / "figT3_sequences.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close(fig)

# Fig T5: single-split variability of the clinical model
fig, ax = plt.subplots(figsize=(3.5, 2.0))
ax.hist(SV, bins=25, color="#bbbbbb", edgecolor="white", lw=0.4)
rr = R[R.tag == "clinical__thesis"]
if len(rr):
    ax.axvline(rr.auc_patient.iloc[0], color="#d95f02", lw=1, label="thesis split")
rr = R[R.tag == "clinical__cv5"]
if len(rr):
    ax.axvline(rr.auc_patient.iloc[0], color="#1b9e77", lw=1, ls="--", label="five-fold CV (pooled)")
ax.set_xlabel("Test AUC of the clinical model, one random 85/15 split"); ax.set_ylabel("Splits"); ax.legend(fontsize=6)
fig.savefig(OUT / "figT5_splits.pdf", bbox_inches="tight", pad_inches=0.02); fig.savefig(OUT / "figT5_splits.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close(fig)

# Grad-CAM
G = []
for f in sorted(T.glob("gradcam_*.csv")):
    g = pd.read_csv(f); g["tag"] = f.stem.replace("gradcam_", ""); G.append(g)
mac = ["% generated by src/thesis_eval.py",
       "\\providecommand{\\NT}[1]{\\ifcsname nt@#1\\endcsname\\csname nt@#1\\endcsname\\else\\textbf{\\textcolor{red}{??}}\\fi}"]
put = lambda k, v, f="{:.3f}": mac.append(f"\\expandafter\\def\\csname nt@{k}\\endcsname{{{f.format(v)}}}") if v == v else None
if G:
    G = pd.concat(G); GS = G.groupby("tag").agg(n=("slice", "size"), energy=("energy_in_lesion", "mean"), area=("lesion_area_frac", "mean"),
                                                  hit=("pointing_hit", "mean")).reset_index()
    GS.round(4).to_csv(OUT / "gradcam_summary.csv", index=False)
    for _, r in GS.iterrows():
        k = r.tag.split("__")[0] + r.tag.split("__")[2]
        put(f"gc{k}energy", r.energy); put(f"gc{k}area", r.area); put(f"gc{k}hit", r.hit); put(f"gc{k}n", r.n, "{:.0f}")
    ex = sorted(T.glob("gradcam_examples_*.npz"))
    if (T / "gradcam_examples_vgg16__none.npz").exists():      # the network shown in Figure 6
        ex = [T / "gradcam_examples_vgg16__none.npz"]
    if ex:
        z = np.load(ex[-1]); n = min(6, len(z["img"]))
        fig, axs = plt.subplots(2, n, figsize=(7.16, 2.6))
        for i in range(n):
            for rr, show_cam in enumerate([False, True]):
                a_ = axs[rr, i]; a_.imshow(z["img"][i], cmap="gray"); a_.axis("off")
                if show_cam:
                    c = z["cam"][i].astype(float); a_.imshow(np.ma.masked_where(c < 0.2 * c.max(), c), cmap="jet", alpha=0.45)
                a_.contour(z["mask"][i], levels=[0.5], colors="#00e5ff" if show_cam else "#ff3b30", linewidths=0.5)
        fig.savefig(OUT / "figT4_gradcam.pdf", bbox_inches="tight", pad_inches=0.02); fig.savefig(OUT / "figT4_gradcam.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close(fig)
key = lambda r: f"{r.arch}{r.input.replace('_', '')}{r.clin}{r.split}"
for _, r in R.iterrows():
    k = "clin" + r.split if r.arch == "logistic" else key(r)
    for c in ["auc_slice", "auc_patient", "lo", "hi", "d_clin", "d_lo", "d_hi", "acc_slice", "acc_patient"]:
        if c in r and pd.notna(r[c]):
            put(k + c.replace("_", ""), float(r[c]))
put("svmed", float(np.median(SV))); put("svlo", float(np.percentile(SV, 2.5))); put("svhi", float(np.percentile(SV, 97.5)))
put("svmin", float(SV.min())); put("svmax", float(SV.max())); put("svn", float(len(SV)), "{:.0f}")
for cfg, du in U.groupby("tag") if len(U) else []:
    if du.unit.str.startswith("fold").any():
        k = cfg.replace("_", "").replace("__", "")
        put(k.replace("__", "") + "foldmin", float(du.auc.min())); put(k.replace("__", "") + "foldmax", float(du.auc.max()))
put("npat", float(len(L)), "{:.0f}"); put("npoor", float(L.y_poor.sum()), "{:.0f}")
if pl is not None:
    for m in ["flair", "t1"]:
        for p_ in ["axial", "sagittal", "coronal"]:
            put(f"plane{m}{p_}", float((pl[f"{m}_plane"] == p_).sum()), "{:.0f}")
(OUT / "numbers_t.tex").write_text("\n".join(m for m in mac if m) + "\n")
print("written", OUT)
