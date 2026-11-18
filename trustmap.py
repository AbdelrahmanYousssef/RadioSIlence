"""
Radio Silence - trust maps for a pretrained fastMRI brain E2E-VarNet.

For each annotated FLAIR slice:
  1. ground truth = RSS of IFFT of fully sampled k-space, center-cropped (fastMRI style)
  2. undersample with an equispaced mask (fastMRI "equispaced" as used for the
     Nov-2020 brain leaderboard model = today's EquispacedMaskFractionFunc)
  3. reconstruct with the pretrained VarNet (inference only, batch size 1)
  4. trust map = pixel-wise std over N rebuilds; each rebuild keeps the same mask but
     randomly drops ~5% of the acquired non-center lines. This only uses data a real
     fast scan would have measured.
  5. error map = |recon - gt|
  6. per fastMRI+ box: mean trust and mean error inside the box
  7. per-slice pixel correlation (trust vs error) and per-box correlation

All maps are stored/plotted in the fastMRI+ labelling orientation, i.e. the
320x320 image flipped up/down (img[::-1, :]), exactly as in
fastmri-plus/ExampleScripts/example.ipynb. Box (x, y, w, h) is then used as-is.
"""

import argparse
import sys
import time
import warnings
import xml.etree.ElementTree as etree
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "third_party" / "fastMRI"))
warnings.filterwarnings("ignore", message="Could not retrieve fastmri version")

import fastmri  # noqa: E402
from fastmri.data import transforms as T  # noqa: E402
from fastmri.data.mri_data import et_query  # noqa: E402
from fastmri.data.subsample import EquispacedMaskFractionFunc  # noqa: E402
from fastmri.models import VarNet  # noqa: E402

WEIGHTS_URL = (
    "https://dl.fbaipublicfiles.com/fastMRI/trained_models/varnet/"
    "brain_leaderboard_state_dict.pt"
)
# scans go in ./DATA (inside the repo); falls back to a DATA folder next to it
DEFAULT_DATA = ROOT / "DATA" if (ROOT / "DATA").is_dir() else ROOT.parent / "DATA"
DEFAULT_ANNOTATIONS = ROOT / "third_party" / "fastmri-plus" / "Annotations" / "brain.csv"
DEFAULT_WEIGHTS = ROOT / "models" / "brain_leaderboard_state_dict.pt"
# training config of varnet_brain_leaderboard.py: accelerations [4, 8], center fractions [0.08, 0.04]
CENTER_FRACTION = {4: 0.08, 8: 0.04}


# --------------------------------------------------------------------------- model
def load_model(weights: Path, device: torch.device) -> VarNet:
    if not weights.exists():
        import requests

        print(f"Downloading weights to {weights} ...")
        weights.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(WEIGHTS_URL, timeout=30, stream=True)
        r.raise_for_status()
        with open(weights, "wb") as fh:
            for chunk in r.iter_content(8 * 1024 * 1024):
                fh.write(chunk)
    # same constructor as fastmri_examples/varnet/run_pretrained_varnet_inference.py
    model = VarNet(num_cascades=12, pools=4, chans=18, sens_pools=4, sens_chans=8)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    return model.eval().to(device)


class Reconstructor:
    """Runs the VarNet in inference mode, batch size 1. Falls back to fp16 autocast on OOM."""

    def __init__(self, model, device, amp=False):
        self.model, self.device, self.amp = model, device, amp

    def __call__(self, kspace: torch.Tensor, line_mask: np.ndarray) -> torch.Tensor:
        """kspace: (coils, H, W, 2) on device. line_mask: (W,) 0/1. Returns (H, W) image on CPU."""
        mask = torch.from_numpy(line_mask.astype(bool)).view(1, 1, 1, -1, 1).to(self.device)
        masked = (kspace.unsqueeze(0) * mask) + 0.0  # same as fastmri apply_mask
        while True:
            try:
                with torch.inference_mode(), torch.autocast(
                    "cuda", dtype=torch.float16, enabled=self.amp and self.device.type == "cuda"
                ):
                    # num_low_frequencies not passed -> model detects the center block
                    # from the mask, exactly like the pretrained inference script
                    out = self.model(masked, mask)
                return out.float()[0].cpu()
            except torch.cuda.OutOfMemoryError:
                if self.amp:
                    raise
                print("CUDA OOM in fp32 -> retrying with fp16 autocast")
                torch.cuda.empty_cache()
                self.amp = True


# --------------------------------------------------------------------------- data
def read_metadata(hf: h5py.File) -> dict:
    """Same fields as fastmri.data.mri_data.SliceDataset._retrieve_metadata."""
    root = etree.fromstring(hf["ismrmrd_header"][()])
    enc = ["encoding", "encodedSpace", "matrixSize"]
    enc_y = int(et_query(root, enc + ["y"]))
    rec = ["encoding", "reconSpace", "matrixSize"]
    recon_size = (int(et_query(root, rec + ["x"])), int(et_query(root, rec + ["y"])))
    lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
    center = int(et_query(root, lims + ["center"]))
    maximum = int(et_query(root, lims + ["maximum"])) + 1
    padding_left = enc_y // 2 - center
    return dict(padding_left=padding_left, padding_right=padding_left + maximum, recon_size=recon_size)


def crop_size_for(hf: h5py.File, meta: dict, width: int) -> tuple:
    # Boxes were drawn on reconstruction_rss, so its shape defines the box frame.
    if "reconstruction_rss" in hf:
        return tuple(hf["reconstruction_rss"].shape[-2:])
    crop = meta["recon_size"]
    if width < crop[1]:  # "FLAIR 203" rule from run_pretrained_varnet_inference.py
        crop = (width, width)
    return crop


def ground_truth(kspace: torch.Tensor, crop) -> torch.Tensor:
    """RSS of the coil images from fully sampled k-space, center-cropped."""
    img = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace)), dim=0)
    return T.center_crop(img, crop)


# --------------------------------------------------------------------------- masks
def base_mask(width, accel, meta, offset=None, seed=None) -> np.ndarray:
    mf = EquispacedMaskFractionFunc([CENTER_FRACTION[accel]], [accel])
    mask, _ = mf((1, 1, width, 2), offset, seed)
    m = mask[0, 0, :, 0].numpy().copy()
    m[: meta["padding_left"]] = 0  # as VarNetDataTransform / apply_mask(padding=...)
    m[meta["padding_right"]:] = 0
    return m


def center_run(m: np.ndarray) -> tuple:
    """Contiguous block of acquired lines around W//2 (what VarNet treats as the ACS center)."""
    cent = len(m) // 2
    left = int(np.argmin(m[:cent][::-1]))
    right = int(np.argmin(m[cent:]))
    return cent - left, cent + right  # [start, stop)


def rebuild_masks(main: np.ndarray, n, frac, seed) -> tuple:
    """n copies of the main mask, each with a random ~frac of the acquired outer lines removed.
    The center block is never touched, so the model's calibration region stays the same."""
    s, e = center_run(main)
    outer = np.array([i for i in np.flatnonzero(main) if not (s <= i < e)])
    n_drop = max(1, int(round(frac * len(outer))))
    out = []
    for k in range(n):
        rng = np.random.default_rng([*seed, k])
        m = main.copy()
        m[rng.choice(outer, size=n_drop, replace=False)] = 0
        out.append(m)
    return out, n_drop, len(outer)


# --------------------------------------------------------------------------- metrics
def psnr(gt, x):
    return 10 * np.log10(gt.max() ** 2 / np.mean((gt - x) ** 2))


def ssim(gt, x):
    from skimage.metrics import structural_similarity

    return structural_similarity(gt, x, data_range=gt.max())


def corr(a, b):
    from scipy.stats import pearsonr, spearmanr

    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan, np.nan
    return float(pearsonr(a, b)[0]), float(spearmanr(a, b)[0])


def box_slices(box, shape):
    x, y, w, h = (int(box[k]) for k in ("x", "y", "width", "height"))
    y0, y1 = max(y, 0), min(y + h, shape[0])
    x0, x1 = max(x, 0), min(x + w, shape[1])
    return slice(y0, y1), slice(x0, x1)


# --------------------------------------------------------------------------- figure
def save_figure(path, gt, rec, err, trust, boxes, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    vmax = gt.max()
    panels = [
        (gt, "Ground truth", "gray", 0, vmax),
        (rec, "Reconstruction", "gray", 0, vmax),
        (err, "|recon - GT|", "inferno", 0, np.percentile(err, 99.5)),
        (trust, "Trust map (std over rebuilds)", "inferno", 0, np.percentile(trust, 99.5)),
    ]
    for ax, (img, t, cmap, lo, hi) in zip(axes, panels):
        im = ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(t, fontsize=10)
        ax.axis("off")
        if cmap != "gray":
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        for _, b in boxes.iterrows():
            ax.add_patch(Rectangle((b.x - 0.5, b.y - 0.5), b.width, b.height,
                                   fill=False, edgecolor="cyan", linewidth=1.2))
    fig.suptitle(title + "   (fastMRI+ orientation: flipped up/down)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- main loop
def compute_slice(hf, fname, sl, boxes, accel, recon, device,
                  n_rebuilds=10, drop_frac=0.05, fg_thresh=0.05, seed_extra=0) -> dict:
    """Everything for one slice. All 2D maps are returned in the fastMRI+ (flipped) frame."""
    meta = read_metadata(hf)
    kspace = T.to_tensor(hf["kspace"][sl]).to(device)  # (coils, H, W, 2)
    width = kspace.shape[-2]
    crop = crop_size_for(hf, meta, width)

    gt = ground_truth(kspace, crop).cpu().numpy()
    check = ""
    if "reconstruction_rss" in hf:
        ref = hf["reconstruction_rss"][sl]
        check = f"gt vs stored reconstruction_rss: max rel diff {np.abs(gt - ref).max() / ref.max():.2e}"

    seed = tuple(map(ord, fname))  # same per-volume seed as VarNetDataTransform(use_seed=True)
    main = base_mask(width, accel, meta, seed=seed)
    cc = lambda x: T.center_crop(x, crop).numpy()  # noqa: E731

    t0 = time.perf_counter()
    rec = cc(recon(kspace, main))
    zf = cc(fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(
        kspace * torch.from_numpy(main).to(device).view(1, 1, -1, 1))), dim=0).cpu())

    masks, n_drop, n_outer = rebuild_masks(main, n_rebuilds, drop_frac, (*seed, sl, seed_extra))
    stack = np.stack([cc(recon(kspace, m)) for m in masks])
    dt = time.perf_counter() - t0

    # flip up/down into the fastMRI+ labelling frame
    flip = lambda x: np.ascontiguousarray(x[..., ::-1, :])  # noqa: E731
    gt, rec, zf, stack = map(flip, (gt, rec, zf, stack))
    err = np.abs(rec - gt)
    trust = stack.std(0)

    fg = gt > fg_thresh * gt.max()
    p_all, s_all = corr(trust.ravel(), err.ravel())
    p_fg, s_fg = corr(trust[fg], err[fg])
    s, e = center_run(main)

    slice_row = dict(file=fname, slice=sl, accel=accel, n_boxes=len(boxes),
                     psnr=psnr(gt, rec), ssim=ssim(gt, rec), psnr_zero_filled=psnr(gt, zf),
                     pearson_all=p_all, spearman_all=s_all, pearson_fg=p_fg, spearman_fg=s_fg,
                     fp16=recon.amp)
    box_rows = []
    for _, b in boxes.iterrows():
        ys, xs = box_slices(b, gt.shape)
        box_rows.append(dict(
            file=fname, slice=sl, label=b.label, x=b.x, y=b.y, width=b.width, height=b.height,
            accel=accel,
            mean_trust=float(trust[ys, xs].mean()), mean_error=float(err[ys, xs].mean()),
            # slice-level brain means so values can be compared across slices
            fg_mean_trust=float(trust[fg].mean()), fg_mean_error=float(err[fg].mean()),
            gt_max=float(gt.max()),
        ))
    return dict(gt=gt, rec=rec, zf=zf, err=err, trust=trust, fg=fg,
                main_mask=main, masks=masks, center=(s, e), n_drop=n_drop, n_outer=n_outer,
                padding=(meta["padding_left"], meta["padding_right"]), width=width,
                kspace_shape=tuple(kspace.shape), crop=crop, check=check, seconds=dt,
                slice_row=slice_row, box_rows=box_rows)


def process_slice(hf, fname, sl, boxes, accel, recon, args, device, fig_dir):
    r = compute_slice(hf, fname, sl, boxes, accel, recon, device,
                      args.n_rebuilds, args.drop_frac, args.fg_thresh, args.seed)
    sr, (s, e) = r["slice_row"], r["center"]
    print(f"  {fname} slice {sl:2d} | kspace {r['kspace_shape']} crop {r['crop']} | "
          f"lines {int(r['main_mask'].sum())}/{r['width']} (acquired cols {r['padding'][0]}-{r['padding'][1] - 1}, "
          f"center {e - s}), rebuilds drop {r['n_drop']}/{r['n_outer']} outer lines | PSNR {sr['psnr']:.2f} dB "
          f"(zero-filled {sr['psnr_zero_filled']:.2f}), SSIM {sr['ssim']:.3f} | "
          f"{r['seconds']:.1f}s{' | fp16' if recon.amp else ''}")
    if r["check"]:
        print("   ", r["check"])
    if fig_dir is not None:
        save_figure(fig_dir / f"{fname}_s{sl:02d}_acc{accel}.png", r["gt"], r["rec"], r["err"],
                    r["trust"], boxes,
                    f"{fname}  slice {sl}  {accel}x  PSNR {sr['psnr']:.1f} dB")
    if args.save_npz:
        np.savez_compressed(args.out / f"acc{accel}" / "arrays" / f"{fname}_s{sl:02d}_acc{accel}.npz",
                            gt=r["gt"], rec=r["rec"], err=r["err"], trust=r["trust"])
    return sr, r["box_rows"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", type=Path, default=DEFAULT_DATA)
    p.add_argument("--annotations", type=Path,
                   default=ROOT / "third_party" / "fastmri-plus" / "Annotations" / "brain.csv")
    p.add_argument("--weights", type=Path, default=ROOT / "models" / "brain_leaderboard_state_dict.pt")
    p.add_argument("--out", type=Path, default=ROOT / "outputs")
    p.add_argument("--accel", type=int, nargs="+", default=[4], choices=[4, 8])
    p.add_argument("--file", type=str, default=None, help="only this file stem (e.g. file_brain_AXFLAIR_200_6002425)")
    p.add_argument("--slice", type=int, default=None, help="only this slice (use with --file)")
    p.add_argument("--n_rebuilds", type=int, default=10)
    p.add_argument("--drop_frac", type=float, default=0.05)
    p.add_argument("--fg_thresh", type=float, default=0.05, help="foreground = gt > thresh * max(gt)")
    p.add_argument("--max_figures", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fp16", action="store_true", help="force fp16 autocast from the start")
    p.add_argument("--save_npz", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ann = pd.read_csv(args.annotations)
    ann = ann[ann.study_level.str.strip() == "No"].copy()  # study-level rows have no box
    ann["slice"] = ann["slice"].astype(int)

    files = sorted(args.data_dir.glob("*.h5"))
    if args.file:
        files = [f for f in files if f.stem == args.file]
    files = [f for f in files if f.stem in set(ann.file)]
    if not files:
        sys.exit(f"No annotated .h5 files found in {args.data_dir}")

    device = torch.device(args.device)
    recon = Reconstructor(load_model(args.weights, device), device, amp=args.fp16)

    for accel in args.accel:
        out = args.out / f"acc{accel}"
        (out / "figures").mkdir(parents=True, exist_ok=True)
        if args.save_npz:
            (out / "arrays").mkdir(parents=True, exist_ok=True)
        slice_rows, box_rows, n_fig = [], [], 0
        print(f"=== {accel}x (center fraction {CENTER_FRACTION[accel]}) ===")
        for f in files:
            fa = ann[ann.file == f.stem]
            slices = sorted(fa.slice.unique())
            if args.slice is not None:
                slices = [s for s in slices if s == args.slice]
            with h5py.File(f, "r") as hf:
                for sl in slices:
                    fig_dir = out / "figures" if n_fig < args.max_figures else None
                    sr, br = process_slice(hf, f.stem, sl, fa[fa.slice == sl], accel, recon, args, device, fig_dir)
                    n_fig += fig_dir is not None
                    slice_rows.append(sr)
                    box_rows.extend(br)

        sdf, bdf = pd.DataFrame(slice_rows), pd.DataFrame(box_rows)
        sdf.to_csv(out / "slices.csv", index=False)
        bdf.to_csv(out / "boxes.csv", index=False)

        lines = [f"Acceleration {accel}x: {len(sdf)} slices, {len(bdf)} boxes, {sdf.file.nunique()} files",
                 f"Recon PSNR mean {sdf.psnr.mean():.2f} dB, SSIM mean {sdf.ssim.mean():.3f}",
                 "Pixel-level trust vs error, per slice (mean [min, max]):"]
        for c in [c for c in sdf.columns if c.startswith(("pearson", "spearman"))]:
            lines.append(f"  {c:16s} {sdf[c].mean():+.3f} [{sdf[c].min():+.3f}, {sdf[c].max():+.3f}]")
        lines.append("Per-box trust vs error (across all boxes):")
        pr, sr_ = corr(bdf.mean_trust.values, bdf.mean_error.values)
        # ratio to the slice brain mean removes slice-to-slice intensity scale
        pr2, sr2 = corr((bdf.mean_trust / bdf.fg_mean_trust).values, (bdf.mean_error / bdf.fg_mean_error).values)
        lines.append(f"  raw pearson {pr:+.3f} spearman {sr_:+.3f} | "
                     f"normalised-by-slice pearson {pr2:+.3f} spearman {sr2:+.3f}")
        ratio = bdf.mean_trust / bdf.fg_mean_trust
        lines.append(f"Box trust vs slice brain average: median {ratio.median():.2f}x, "
                     f"{(ratio > 1).mean() * 100:.0f}% of boxes above average")
        summary = "\n".join(lines)
        print(summary)
        (out / "summary.txt").write_text(summary + "\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
