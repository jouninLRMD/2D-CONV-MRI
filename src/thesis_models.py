"""
Step 2 of the study: the multimodal 2D networks of the thesis, re-implemented in PyTorch, under controlled variations.

Architectures (as in the original thesis code):
  basic   3 x [Conv 3x3 (32, 64, 128) + ReLU + MaxPool] + global average pooling; clinical branch Dense(32);
          fusion Dense(64) -> Dropout 0.3 -> Dense(32) -> Dropout 0.3 -> output; trained end to end
  vgg16   ImageNet VGG16, frozen, global average pooling (512); clinical branch Dense(64) + Dropout 0.3;
          fusion Dense(128) -> Dropout 0.4 -> output
  effb0   ImageNet EfficientNet-B0, frozen, global average pooling (1280); same heads as vgg16
Adam (learning rate 1e-4), batch 16, up to 50 epochs, early stopping on the validation loss (patience 8), each slice a
training sample carrying the patient's label. The frozen backbones are run once and their features cached in
<work>/thesis/feat_<arch>_<input>.npy.

Factors (one run = one combination):
  --input    mix_native (thesis: FLAIR and T1w acquisition-axis slices pooled) | flair_native | t1_native |
             flair_axial | t1_axial | dwi_axial
  --clin     thesis (raw values, missing BMI set to 0, as in the thesis) | fold (median imputation with a missing
             indicator and standardization fitted on the training patients) | none (image only)
  --split    thesis (the thesis' 70/15/15 patient split, random_state 42: 502/108/108 patients) |
             cv5 (the frozen five folds in results/cohort_all.csv, with a stratified 15% validation split of
             the training part of each fold)
  --gradcam  for dwi_axial: Grad-CAM on held-out slices that contain lesion, scored against the expert mask

Outputs: results/thesis/oof/<arch>__<input>__<clin>__<split>__s<seed>.csv (one row per held-out slice:
         participant_id, unit, src, slice, p), results/thesis/gradcam_<tag>.csv and
         results/thesis/gradcam_examples_<arch>__<clin>.npz. With --quick (two epochs on a few batches, a test of the
         installation) everything goes to results/thesis/smoke instead.
Usage:   python src/thesis_models.py --arch effb0 --input dwi_axial --clin fold --split cv5
"""
import argparse, time
try:
    from PIL import Image  # noqa: F401
except Exception:
    pass
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from common import RES, WORK

TH = WORK / "thesis"; OUT = RES / "thesis"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def backbone(arch):
    import torchvision
    if arch == "vgg16":
        m = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1); return m.features, 512
    m = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    return m.features, 1280


def to_rgb(x):                                       # (B,1,H,W) in [0,1] -> ImageNet-normalized 3 channels
    return (x.repeat(1, 3, 1, 1) - MEAN.to(x.device)) / STD.to(x.device)


def cache_features(arch, inp, dev):
    f = TH / f"feat_{arch}_{inp}.npy"
    X = np.load(TH / f"{inp}.npy", mmap_mode="r"); N, S = X.shape[:2]
    if f.exists() and f.stat().st_mtime > (TH / f"{inp}.npy").stat().st_mtime:
        c = np.load(f, mmap_mode="r")
        if c.shape[:2] == (N, S):
            return c
    bb, D = backbone(arch); bb = bb.to(dev).eval()
    out = np.lib.format.open_memmap(f, mode="w+", dtype=np.float16, shape=(N, S, D))
    with torch.no_grad():
        for i in range(N):
            x = torch.from_numpy(np.asarray(X[i], dtype=np.float32) / 255.0)[:, None].to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                h = bb(to_rgb(x)).float().mean((2, 3))
            out[i] = h.cpu().numpy().astype(np.float16)
    out.flush(); print(f"cached {f.name}", flush=True)
    return np.load(f, mmap_mode="r")


class Basic(nn.Module):
    def __init__(self, n_tab):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                  nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                  nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.tab = nn.Sequential(nn.Linear(n_tab, 32), nn.ReLU()) if n_tab else None
        self.head = nn.Sequential(nn.Linear(128 + (32 if n_tab else 0), 64), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1))

    def forward(self, x, t, return_maps=False):
        fm = self.conv(x); h = fm.mean((2, 3))
        if self.tab is not None:
            h = torch.cat([h, self.tab(t)], 1)
        o = self.head(h)[:, 0]
        return (o, fm) if return_maps else o


class Head(nn.Module):
    def __init__(self, D, n_tab):
        super().__init__()
        self.tab = nn.Sequential(nn.Linear(n_tab, 64), nn.ReLU(), nn.Dropout(0.3)) if n_tab else None
        self.head = nn.Sequential(nn.Linear(D + (64 if n_tab else 0), 128), nn.ReLU(), nn.Dropout(0.4), nn.Linear(128, 1))

    def forward(self, f, t):
        h = torch.cat([f, self.tab(t)], 1) if self.tab is not None else f
        return self.head(h)[:, 0]


def clin_matrix(df, tr_mask, mode):
    cols = ["age", "nihss", "priorstroke", "bmi"]
    if mode == "none":
        return np.zeros((len(df), 0), np.float32)
    if mode == "thesis":
        return df[cols].astype(float).fillna(0.0).values.astype(np.float32)
    tr = df[tr_mask]; X = []
    for c in cols:
        v = df[c].astype(float); med = tr[c].median()
        X.append(v.fillna(med).values)
        if tr[c].isna().any() or v.isna().any():
            X.append(v.isna().astype(float).values)
    X = np.vstack(X).T
    mu, sd = X[tr_mask].mean(0), X[tr_mask].std(0) + 1e-6
    return ((X - mu) / sd).astype(np.float32)


def splits(L, how):
    from sklearn.model_selection import train_test_split
    if how == "thesis":
        subj = np.array(sorted(L.participant_id)); lab = L.set_index("participant_id").loc[subj, "y_poor"].values.astype(int)
        tr, te = train_test_split(subj, test_size=0.15, stratify=lab, random_state=42)
        lab_tr = L.set_index("participant_id").loc[tr, "y_poor"].values.astype(int)
        tr, va = train_test_split(tr, test_size=0.1765, stratify=lab_tr, random_state=42)
        yield "thesis_test", set(tr), set(va), set(te)
        return
    for k in range(5):
        te = set(L.participant_id[L.fold == k]); pool = L[L.fold != k]
        tr, va = train_test_split(pool.participant_id.values, test_size=0.15, stratify=pool.y_poor.values, random_state=k)
        yield f"fold{k}", set(tr), set(va), te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["basic", "vgg16", "effb0"], required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--clin", choices=["thesis", "fold", "none"], default="fold")
    ap.add_argument("--split", choices=["thesis", "cv5"], default="cv5")
    ap.add_argument("--epochs", type=int, default=50); ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--gradcam", action="store_true")
    ap.add_argument("--quick", action="store_true", help="two epochs on a few batches; outputs to results/thesis/smoke")
    a = ap.parse_args()
    if a.quick:
        a.epochs, a.patience = 2, 1
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); amp = dev.type == "cuda"
    a.out = OUT / "smoke" if a.quick else OUT
    (a.out / ("." if a.quick else "oof")).mkdir(parents=True, exist_ok=True)
    ids = pd.read_csv(TH / "ids.csv").participant_id.values
    coh = pd.read_csv(RES / "cohort_all.csv").set_index("participant_id")
    L = coh.loc[ids].reset_index()
    inputs = ["flair_native", "t1_native"] if a.input == "mix_native" else [a.input]
    tag = f"{a.arch}__{a.input}__{a.clin}__{a.split}__s{a.seed}"
    rows, cams = [], []
    for unit, tr, va, te in splits(L, a.split):
        t0 = time.time()
        trm = L.participant_id.isin(tr).values
        T = clin_matrix(L, trm, a.clin); n_tab = T.shape[1]
        # sample index: (patient row, input id, slice)
        S = 20
        def samples(pset):
            pr = np.where(L.participant_id.isin(pset).values)[0]
            return np.array([(p, j, s) for p in pr for j in range(len(inputs)) for s in range(S)], dtype=np.int64)
        s_tr, s_va, s_te = samples(tr), samples(va), samples(te)
        if a.quick:
            s_tr, s_va, s_te = s_tr[:256], s_va[:64], s_te[:64]
        y = L.y_poor.values.astype(np.float32)
        if a.arch == "basic":
            Xs = [np.load(TH / f"{i}.npy", mmap_mode="r") for i in inputs]
            model = Basic(n_tab).to(dev)
            def batch(idx):
                x = np.stack([Xs[j][p, s] for p, j, s in idx]).astype(np.float32) / 255.0
                return torch.from_numpy(x)[:, None].to(dev), torch.from_numpy(T[idx[:, 0]]).to(dev), torch.from_numpy(y[idx[:, 0]]).to(dev)
            fwd = lambda xb, tb: model(xb, tb)
        else:
            Fs = [cache_features(a.arch, i, dev) for i in inputs]
            model = Head(Fs[0].shape[-1], n_tab).to(dev)
            def batch(idx):
                f = np.stack([Fs[j][p, s] for p, j, s in idx]).astype(np.float32)
                return torch.from_numpy(f).to(dev), torch.from_numpy(T[idx[:, 0]]).to(dev), torch.from_numpy(y[idx[:, 0]]).to(dev)
            fwd = lambda fb, tb: model(fb, tb)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        best, bad, state = 1e9, 0, None; rng = np.random.default_rng(a.seed)
        def run(sidx, train):
            model.train(train); tot = n = 0
            order = rng.permutation(len(sidx)) if train else np.arange(len(sidx))
            for b in range(0, len(sidx), 16):
                xb, tb, yb = batch(sidx[order[b:b + 16]])
                with torch.set_grad_enabled(train), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    lo = fwd(xb, tb).float()
                loss = F.binary_cross_entropy_with_logits(lo, yb)
                if train:
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                tot += loss.item() * len(yb); n += len(yb)
            return tot / max(n, 1)
        for ep in range(a.epochs):
            trl = run(s_tr, True); val = run(s_va, False)
            if val < best - 1e-4:
                best, bad, state = val, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= a.patience:
                    break
        model.load_state_dict(state); model.eval()
        with torch.no_grad():
            for b in range(0, len(s_te), 64):
                idx = s_te[b:b + 64]; xb, tb, _ = batch(idx)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    p = torch.sigmoid(fwd(xb, tb).float()).cpu().numpy()
                for (pi, j, s), pv in zip(idx, p):
                    rows.append(dict(participant_id=L.participant_id[pi], unit=unit, src=inputs[j], slice=int(s), p=float(pv)))
        if a.gradcam and a.input == "dwi_axial":
            cams += gradcam(a, model, L, s_te, T, dev, amp)
        print(f"[{tag}] {unit}: {len(s_tr)} train slices, stopped at epoch {ep}, {(time.time()-t0)/60:.1f} min", flush=True)
    pd.DataFrame(rows).to_csv((a.out if a.quick else a.out / "oof") / f"{tag}.csv", index=False)
    if cams:
        pd.DataFrame(cams).to_csv(a.out / f"gradcam_{tag}.csv", index=False)


def gradcam(a, model, L, s_te, T, dev, amp):
    """Grad-CAM for held-out DWI slices containing lesion; energy inside the (dilated) expert mask and pointing hit."""
    X = np.load(TH / "dwi_axial.npy", mmap_mode="r"); M = np.load(TH / "dwi_mask.npy", mmap_mode="r")
    if a.arch == "basic":
        net = model
    else:
        bb, _ = backbone(a.arch); bb = bb.to(dev).eval()
    out, examples = [], []
    for p, j, s in s_te:
        m = np.asarray(M[p, s], dtype=np.float32)
        if m.sum() < 20:
            continue
        x = torch.from_numpy(np.asarray(X[p, s], dtype=np.float32) / 255.0)[None, None].to(dev)
        t = torch.from_numpy(T[[p]]).to(dev)
        if a.arch == "basic":
            fm = model.conv(x); fm.retain_grad(); h = fm.mean((2, 3))
            if model.tab is not None:
                h = torch.cat([h, model.tab(t)], 1)
            logit = model.head(h)[0, 0]
        else:
            fm = bb(to_rgb(x)); fm.retain_grad(); logit = model(fm.mean((2, 3)), t)[0]
        model.zero_grad(); logit.backward()
        w = fm.grad.mean((2, 3), keepdim=True); cam = F.relu((w * fm).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=m.shape, mode="bilinear", align_corners=False)[0, 0].detach().cpu().numpy()
        brain = np.asarray(X[p, s]) > 0
        cam = cam * brain
        if cam.sum() <= 0:
            continue
        from scipy.ndimage import binary_dilation
        md = binary_dilation(m > 0.5, iterations=5)
        yx = np.unravel_index(np.argmax(cam), cam.shape)
        out.append(dict(participant_id=L.participant_id[p], slice=int(s), lesion_px=int(m.sum()),
                        energy_in_lesion=float(cam[md].sum() / cam.sum()), lesion_area_frac=float(md.sum() / max(brain.sum(), 1)),
                        pointing_hit=int(md[yx])))
        if len(examples) < 12:
            examples.append((np.asarray(X[p, s]), m, cam))
    if examples:
        np.savez_compressed(a.out / f"gradcam_examples_{a.arch}__{a.clin}.npz", img=np.stack([e[0] for e in examples]),
                            mask=np.stack([e[1] for e in examples]), cam=np.stack([e[2] for e in examples]).astype(np.float16))
    return out


if __name__ == "__main__":
    main()
