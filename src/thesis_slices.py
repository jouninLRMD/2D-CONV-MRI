"""
Step 1 of the study: 2D slice datasets for the 718 patients with a recorded discharge outcome.

Five inputs, 20 slices of 224 x 224 per patient, intensities scaled to [0, 1] by the 1st and 99th percentiles of the
volume (as in the thesis) and stored as uint8:
  flair_native, t1_native   the thesis pipeline: 20 equidistant slices over the full third array axis of the stored
                            volume, that is, the acquisition axis, whatever its anatomical plane, including empty slices
  flair_axial, t1_axial     corrected: volume reoriented to RAS, 20 equidistant axial slices over the head extent
  dwi_axial                 DWI trace from the MNI-registered image cache (src/build_cache.py), 20 equidistant axial
                            slices over the brain extent, with the expert acute lesion mask on the same slices
                            (dwi_mask, used by the Grad-CAM analysis)
It also records the anatomical plane of each stored FLAIR and T1w volume.

Inputs:  data/ds004889 (FLAIR, T1w), results/cohort_all.csv, <work>/cache (X.npy, M.npy, index.csv)
Outputs: <work>/thesis/{flair_native,t1_native,flair_axial,t1_axial,dwi_axial,dwi_mask}.npy (about 4.3 GB),
         <work>/thesis/ids.csv (row order of the arrays) and <work>/thesis/planes.csv
Usage:   python src/thesis_slices.py              (all 718 patients, 30 to 45 minutes)
         python src/thesis_slices.py --limit 20   (quick check on 20 patients)
"""
import argparse
import numpy as np, pandas as pd, nibabel as nib
import torch, torch.nn.functional as F
from common import RES, WORK, BIDS

N_SL, SZ = 20, 224


def norm01(v):
    f = np.isfinite(v)
    lo, hi = np.percentile(v[f], [1, 99]) if f.any() else (0, 1)
    if hi <= lo:
        return np.zeros_like(v, np.float32)
    return ((np.clip(v, lo, hi) - lo) / (hi - lo)).astype(np.float32)


def resize(sl, mode="bilinear"):
    t = torch.from_numpy(np.ascontiguousarray(sl, dtype=np.float32))[None, None]
    return F.interpolate(t, size=(SZ, SZ), mode=mode, align_corners=False if mode == "bilinear" else None)[0, 0].numpy()


def extent(v, frac=0.02):
    """z range of slices whose foreground (above the 10th percentile of positive voxels) covers > frac of the slice."""
    pos = v[v > 0]
    thr = np.percentile(pos, 10) if pos.size else 0
    fg = (v > thr).mean((0, 1))
    zz = np.where(fg > frac)[0]
    return (zz.min(), zz.max()) if zz.size else (0, v.shape[2] - 1)


def plane(img):
    ax = nib.aff2axcodes(img.affine)
    third = ax[2]
    return {"S": "axial", "I": "axial", "A": "coronal", "P": "coronal", "L": "sagittal", "R": "sagittal"}[third]


def main():
    ap = argparse.ArgumentParser(description="2D slice datasets of the study")
    ap.add_argument("--limit", type=int, default=0, help="only the first N patients (quick check)")
    a = ap.parse_args()
    out = WORK / "thesis"; out.mkdir(parents=True, exist_ok=True)
    coh = pd.read_csv(RES / "cohort_all.csv"); L = coh[coh.labelled].reset_index(drop=True)
    if a.limit:
        L = L.head(a.limit)
    idx = pd.read_csv(WORK / "cache" / "index.csv").set_index("participant_id")
    X = np.load(WORK / "cache" / "X.npy", mmap_mode="r"); M = np.load(WORK / "cache" / "M.npy", mmap_mode="r")
    N = len(L)
    arrs = {k: np.lib.format.open_memmap(out / f"{k}.npy", mode="w+", dtype=np.uint8, shape=(N, N_SL, SZ, SZ))
            for k in ["flair_native", "t1_native", "flair_axial", "t1_axial", "dwi_axial"]}
    mk = np.lib.format.open_memmap(out / "dwi_mask.npy", mode="w+", dtype=np.uint8, shape=(N, N_SL, SZ, SZ))
    planes = []
    for i, r in L.iterrows():
        s = r.participant_id; pl = dict(participant_id=s)
        for mod, tag in [("FLAIR", "flair"), ("T1w", "t1")]:
            p = BIDS / s / "anat" / f"{s}_{mod}.nii.gz"
            try:
                img = nib.load(str(p)); v = np.asarray(img.dataobj, dtype=np.float32)
            except Exception as e:  # missing or not downloaded: leave zeros and record it
                print(f"  {s} {mod}: not readable ({type(e).__name__}); left empty", flush=True)
                pl[f"{tag}_plane"] = "missing"; pl[f"{tag}_shape"] = ""
                continue
            while v.ndim > 3:
                v = v[..., 0]
            pl[f"{tag}_plane"] = plane(img); pl[f"{tag}_shape"] = "x".join(map(str, v.shape))
            # thesis: stored array, third axis, full range
            vn = norm01(v); zi = np.linspace(0, v.shape[2] - 1, N_SL).astype(int)
            arrs[f"{tag}_native"][i] = np.stack([np.clip(resize(vn[:, :, z]), 0, 1) * 255 for z in zi]).astype(np.uint8)
            # corrected: RAS, axial, head extent
            c = nib.as_closest_canonical(img); vc = np.asarray(c.dataobj, dtype=np.float32)
            while vc.ndim > 3:
                vc = vc[..., 0]
            vcn = norm01(vc); z0, z1 = extent(vc); zi = np.linspace(z0, z1, N_SL).astype(int)
            arrs[f"{tag}_axial"][i] = np.stack([np.clip(resize(np.rot90(vcn[:, :, z])), 0, 1) * 255 for z in zi]).astype(np.uint8)
        row = int(idx.loc[s, "row"]); tr = np.asarray(X[row, 0], dtype=np.float32); m = np.asarray(M[row, 0], dtype=np.float32)
        tn = norm01(np.where(tr != 0, tr, np.nan)); tn = np.nan_to_num(tn, nan=0.0)
        z0, z1 = extent(np.abs(tr)); zi = np.linspace(z0, z1, N_SL).astype(int)
        arrs["dwi_axial"][i] = np.stack([np.clip(resize(np.rot90(tn[:, :, z])), 0, 1) * 255 for z in zi]).astype(np.uint8)
        mk[i] = np.stack([(resize(np.rot90(m[:, :, z]), "nearest") > 0.5) for z in zi]).astype(np.uint8)
        planes.append(pl)
        if i % 50 == 0:
            print(f"  {i + 1}/{N}", flush=True)
    for v in list(arrs.values()) + [mk]:
        v.flush()
    L[["participant_id"]].to_csv(out / "ids.csv", index=False)
    P = pd.DataFrame(planes); P.to_csv(out / "planes.csv", index=False)
    print("FLAIR planes:", P.flair_plane.value_counts().to_dict(), "| T1w planes:", P.t1_plane.value_counts().to_dict())


if __name__ == "__main__":
    main()
