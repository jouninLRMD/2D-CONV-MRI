"""
Reproduces every result of the study, from the public SOOP download to the tables, figures and numbers of the paper
"Do Two-Dimensional Convolutional Networks on Structural MRI Add to Clinical Variables for Predicting Discharge
Outcome after Ischemic Stroke? A Patient-Level Re-Evaluation on a Public Dataset".

Stages, in order:
  download    SOOP metadata, lesion masks and DWI of every session; FLAIR and T1w of the 718 study patients (31 GB)
  atlas       arterial territory atlas (Liu et al. 2023), used by the shared image cache
  preprocess  DWI: brain mask, relative ADC, affine registration to MNI152 2 mm, lesion features, QC images
  cohort      cohort table (718 patients with a discharge mRS) and the frozen five folds
  features    lesion load per arterial territory (atlas resampled to the image grid)
  cache       memory-mapped DWI volumes and lesion masks in MNI space
  design      Figure 1 (study design)
  table       Table 1 (cohort characteristics by outcome)
  slices      2D slice datasets: FLAIR and T1w as in the thesis and reoriented, DWI with lesion masks
  smoke       two-epoch test of the networks on a few batches (results/thesis/smoke); nothing else is touched
  train       the 27 network runs of the paper, seed 0 (GPU, about 3 hours)
  extra       single-factor ladder steps (pooled axial FLAIR + T1w; FLAIR alone as stored) and the thesis input
              preprocessing for the pretrained networks (GPU, about 1 hour)
  gradcam     Grad-CAM of the six DWI networks with the mirror-image control region (GPU, about 1.5 hours)
  seeds       seeds 1 to 4 of the 27 configurations, for training variability (GPU, about 11 hours; the pretrained
              networks run first)
  evaluate    AUCs, confidence intervals, clinical model, split variability, Grad-CAM, figures and number macros

Usage (from the repository root):
  python reproduce.py                          # everything
  python reproduce.py --dry-run                # print the commands only
  python reproduce.py --from slices            # start at a stage
  python reproduce.py --only evaluate          # recompute the paper's numbers from saved predictions
  python reproduce.py --only train --arch effb0     # one architecture
Finished training runs are skipped; use --force to run them again.
"""
import argparse, os, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent
SRC, SCRIPTS = REPO / "src", REPO / "scripts"
RES = Path(os.environ.get("CALM_RESULTS", str(REPO / "results")))
STAGES = ["download", "atlas", "preprocess", "cohort", "features", "cache", "design", "table", "slices", "smoke",
          "train", "extra", "gradcam", "seeds", "evaluate"]
ARCHS = ["basic", "vgg16", "effb0"]
SHORT = {"mix_native": "mix", "dwi_axial": "dwi"}           # log file names
SEEDS = [1, 2, 3, 4]
# single-factor ladder steps: reorientation alone (both sequences pooled), and FLAIR alone without reorientation
EXTRA = [("mix_axial", "fold", "cv5"), ("flair_native", "fold", "cv5")]
# the thesis setting with the thesis input preprocessing of the frozen backbones (pretrained networks only)
THESIS_PREP = [("mix_native", "thesis", "thesis"), ("mix_native", "thesis", "cv5")]

# (input, clinical handling, split, Grad-CAM): the 9 configurations run for each architecture, in the paper's order
CONFIGS = [
    ("mix_native", "thesis", "thesis", False),   # the thesis setting
    ("mix_native", "thesis", "cv5", False),      # + five-fold cross-validation
    ("mix_native", "fold", "cv5", False),        # + in-fold clinical handling
    ("flair_axial", "fold", "cv5", False),       # + reoriented axial FLAIR
    ("flair_axial", "none", "cv5", False),
    ("t1_axial", "fold", "cv5", False),
    ("t1_axial", "none", "cv5", False),
    ("dwi_axial", "fold", "cv5", True),          # DWI instead of FLAIR
    ("dwi_axial", "none", "cv5", True),
]


def run(cmd, log=None, dry=False):
    print("  $ " + " ".join(str(c) for c in cmd) + (f"  > {log}" if log else ""), flush=True)
    if dry:
        return
    t0 = time.time()
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "w") as fh:
            rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=REPO)
    else:
        rc = subprocess.call(cmd, cwd=REPO)
    if rc != 0:
        sys.exit(f"Command failed (exit code {rc})" + (f"; see {log}" if log else ""))
    print(f"    done in {(time.time() - t0) / 60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=STAGES)
    ap.add_argument("--from", dest="start", choices=STAGES)
    ap.add_argument("--arch", nargs="+", choices=ARCHS, help="limit the train stage to these architectures")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2), help="CPU workers")
    ap.add_argument("--force", action="store_true", help="retrain runs whose predictions already exist")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.only:
        stages = [s for s in STAGES if s in a.only]
    elif a.start:
        stages = STAGES[STAGES.index(a.start):]
    else:
        stages = STAGES
    py, L = sys.executable, RES / "thesis" / "logs"
    for st in stages:
        print(f"\n=== {st}", flush=True)
        if st == "download":
            run([py, SCRIPTS / "download_soop.py"], dry=a.dry_run)
        elif st == "atlas":
            run([py, SCRIPTS / "fetch_atlas.py"], dry=a.dry_run)
        elif st == "preprocess":
            run([py, SRC / "preprocess_dwi.py", "--workers", str(a.workers)], dry=a.dry_run)
        elif st == "cohort":
            run([py, SRC / "cohort_splits.py"], dry=a.dry_run)
        elif st == "features":
            run([py, SRC / "atlas_features.py"], dry=a.dry_run)
        elif st == "cache":
            run([py, SRC / "build_cache.py"], dry=a.dry_run)
        elif st == "design":
            run([py, SRC / "design_figure.py"], dry=a.dry_run)
        elif st == "table":
            run([py, SRC / "cohort_table.py"], dry=a.dry_run)
        elif st == "slices":
            run([py, SRC / "thesis_slices.py"], log=L / "slices.txt", dry=a.dry_run)
        elif st == "smoke":
            run([py, SRC / "thesis_models.py", "--arch", "basic", "--input", "dwi_axial", "--clin", "fold",
                 "--split", "cv5", "--gradcam", "--quick"], log=L / "smoke.txt", dry=a.dry_run)
        elif st == "train":
            for arch in ARCHS:
                if a.arch and arch not in a.arch:
                    continue
                for inp, clin, split, cam in CONFIGS:
                    tag = f"{arch}__{inp}__{clin}__{split}__s0"
                    if (RES / "thesis" / "oof" / f"{tag}.csv").exists() and not a.force:
                        print(f"  {tag}: predictions exist, skipped")
                        continue
                    cmd = [py, SRC / "thesis_models.py", "--arch", arch, "--input", inp, "--clin", clin, "--split", split]
                    if cam:
                        cmd.append("--gradcam")
                    run(cmd, log=L / f"train_{arch}_{SHORT.get(inp, inp)}_{clin}_{split}.txt", dry=a.dry_run)
        elif st == "extra":
            jobs = [(arch, inp, clin, split, []) for arch in ARCHS for inp, clin, split in EXTRA]
            jobs += [(arch, inp, clin, split, ["--prep", "thesis"]) for arch in ["vgg16", "effb0"]
                     for inp, clin, split in THESIS_PREP]
            for arch, inp, clin, split, extra in jobs:
                if a.arch and arch not in a.arch:
                    continue
                name = inp + ("-thesisprep" if extra else "")
                tag = f"{arch}__{name}__{clin}__{split}__s0"
                if (RES / "thesis" / "oof" / f"{tag}.csv").exists() and not a.force:
                    print(f"  {tag}: predictions exist, skipped")
                    continue
                cmd = [py, SRC / "thesis_models.py", "--arch", arch, "--input", inp, "--clin", clin, "--split", split] + extra
                run(cmd, log=L / f"train_{arch}_{SHORT.get(name, name)}_{clin}_{split}.txt", dry=a.dry_run)
        elif st == "gradcam":
            for arch in ARCHS:
                if a.arch and arch not in a.arch:
                    continue
                for clin in ["fold", "none"]:
                    tag = f"{arch}__dwi_axial__{clin}__cv5__s0"
                    g = RES / "thesis" / f"gradcam_{tag}.csv"
                    if g.exists() and "energy_in_mirror" in g.open().readline() and not a.force:
                        print(f"  {tag}: Grad-CAM with mirror control exists, skipped")
                        continue
                    cmd = [py, SRC / "thesis_models.py", "--arch", arch, "--input", "dwi_axial", "--clin", clin,
                           "--split", "cv5", "--gradcam"]
                    run(cmd, log=L / f"train_{arch}_dwi_{clin}_cv5.txt", dry=a.dry_run)
        elif st == "seeds":
            for seed in SEEDS:
                for arch in ["vgg16", "effb0", "basic"]:
                    if a.arch and arch not in a.arch:
                        continue
                    for inp, clin, split, _ in CONFIGS:
                        tag = f"{arch}__{inp}__{clin}__{split}__s{seed}"
                        if (RES / "thesis" / "oof" / f"{tag}.csv").exists() and not a.force:
                            print(f"  {tag}: predictions exist, skipped")
                            continue
                        cmd = [py, SRC / "thesis_models.py", "--arch", arch, "--input", inp, "--clin", clin,
                               "--split", split, "--seed", str(seed)]
                        run(cmd, log=L / "seeds" / f"train_{arch}_{SHORT.get(inp, inp)}_{clin}_{split}_s{seed}.txt",
                            dry=a.dry_run)
        elif st == "evaluate":
            run([py, SRC / "thesis_eval.py"], log=L / "evaluate.txt", dry=a.dry_run)
    print("\nFinished. Manuscript outputs: results/thesis/paper")


if __name__ == "__main__":
    main()
