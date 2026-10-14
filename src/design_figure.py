"""
Figure 1 of the manuscript (study design). Drawn with matplotlib; no data needed.
Output: results/thesis/paper/figT0_design.pdf and .png
Usage:  python src/design_figure.py
"""
from common import RES
OUT = RES / "thesis" / "paper"; OUT.mkdir(parents=True, exist_ok=True)
import logging, matplotlib; matplotlib.use("Agg"); logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({"font.family": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 6.5, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(7.16, 2.45)); ax.set_xlim(0, 100); ax.set_ylim(5.5, 37.5); ax.axis("off")
def box(x, y, w, h, title, lines, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec="#444444", lw=0.6))
    ax.text(x + w / 2, y + h - 1.4, title, ha="center", va="top", weight="bold", fontsize=6.6)
    ax.text(x + 0.9, y + h - 4.6, "\n".join(lines), ha="left", va="top", fontsize=5.6, linespacing=1.3)
def arrow(x0, y0, x1, y1):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#444444"))
col = ["#e8f1fa", "#eaf5ec", "#fdf1e3", "#f3ecf7", "#f6f6f6"]
Y, H = 10.5, 26.5
box(0.5, Y, 17.5, H, "a  Data", ["SOOP (ds004889)", "718 patients with", "discharge mRS, age, NIHSS", "poor: mRS 3 to 6", "", "thesis count of 1,436 =", "718 patients × 2 sequences", "", "FLAIR, T1w; DWI added;", "expert lesion masks"], col[0])
box(20.2, Y, 18, H, "b  Slices (20/patient)", ["Thesis: third axis of the", "stored array, full range,", "often sagittal or coronal", "", "Corrected: reoriented,", "axial slices over the", "head extent", "", "DWI: axial, MNI space"], col[1])
box(40.4, Y, 18, H, "c  Networks", ["Basic CNN (3 blocks)", "VGG16, ImageNet, frozen", "EfficientNet-B0,", "ImageNet, frozen", "", "clinical branch: age,", "NIHSS, prior stroke, BMI", "(or image only)", "", "one prediction per slice"], col[2])
box(60.6, Y, 18, H, "d  Evaluation ladder", ["1 thesis split, 502/108/108", "   patients, slice level", "2 same split, patient level", "3 five-fold CV, 718 patients", "4 in-fold clinical handling", "5 axial FLAIR", "6 DWI instead of FLAIR", "", "patient probability =", "mean over the slices"], col[3])
box(80.8, Y, 18.7, H, "e  Added analyses", ["clinical logistic model", "on the same patients", "", "FLAIR, T1w and DWI,", "with and without", "clinical variables", "", "200 random single splits", "", "Grad-CAM against the", "expert DWI lesion"], col[4])
for x0, x1 in [(18.3, 19.9), (38.5, 40.1), (58.7, 60.3), (78.9, 80.5)]:
    arrow(x0, Y + H / 2, x1, Y + H / 2)
ax.text(50, 7.2, "Models are fitted on training patients only; slice probabilities are averaged per patient, and AUCs carry patient-level bootstrap 95% CIs.", ha="center", fontsize=6.2, style="italic")
fig.savefig(OUT / "figT0_design.pdf", bbox_inches="tight", pad_inches=0.02); fig.savefig(OUT / "figT0_design.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
