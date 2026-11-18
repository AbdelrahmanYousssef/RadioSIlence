"""
Radio Silence - local web demo for the VarNet trust maps.

    python app.py            ->  open http://127.0.0.1:5000

Runs the same code as trustmap.py (compute_slice) live on the GPU, one slice at a time.
Results are cached in memory, so going back to a slice is instant.
Binds to 127.0.0.1 only: the fastMRI data must not be served to other machines.
"""

import argparse
import base64
import io
import threading
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from flask import Flask, abort, jsonify, request, send_from_directory
from matplotlib import colormaps
from PIL import Image

import trustmap as tm

WEB = Path(__file__).resolve().parent / "web"
app = Flask(__name__, static_folder=None)
STATE = {}
GPU_LOCK = threading.Lock()
CACHE: "OrderedDict[tuple, dict]" = OrderedDict()
CACHE_SIZE = 48


# --------------------------------------------------------------------------- helpers
def to_png(img: np.ndarray, vmax: float, cmap: str) -> str:
    x = np.clip(img / max(vmax, 1e-20), 0, 1)
    if cmap == "gray":
        rgb = (x * 255).astype(np.uint8)
        im = Image.fromarray(rgb)
    else:
        rgb = (colormaps[cmap](x)[..., :3] * 255).astype(np.uint8)
        im = Image.fromarray(rgb)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def f16(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f2").tobytes()).decode()


def clean(v):
    """JSON-safe floats (NaN -> None)."""
    if isinstance(v, (float, np.floating)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def file_path(stem: str) -> Path:
    p = STATE["data_dir"] / f"{stem}.h5"
    if stem not in STATE["files"] or not p.exists():
        abort(404, f"unknown file {stem}")
    return p


# --------------------------------------------------------------------------- routes
@app.get("/")
def index():
    return send_from_directory(WEB, "index.html")


@app.get("/api/files")
def files():
    ann = STATE["ann"]
    out = []
    for stem, n_slices in STATE["files"].items():
        a = ann[ann.file == stem]
        counts = a.groupby("slice").size().to_dict()
        out.append(dict(file=stem, n_slices=n_slices,
                        boxes_per_slice={int(k): int(v) for k, v in counts.items()},
                        n_boxes=int(len(a))))
    gpu = torch.cuda.get_device_name(0) if STATE["device"].type == "cuda" else "CPU"
    return jsonify(files=out, device=gpu)


@app.get("/api/slice")
def run_slice():
    stem = request.args.get("file", "")
    sl = int(request.args.get("slice", 0))
    accel = int(request.args.get("accel", 4))
    n = max(2, min(int(request.args.get("n", 10)), 30))
    drop = float(request.args.get("drop", 0.05))
    if accel not in tm.CENTER_FRACTION:
        abort(400, "accel must be 4 or 8")
    key = (stem, sl, accel, n, round(drop, 4))
    if key in CACHE:
        CACHE.move_to_end(key)
        return jsonify({**CACHE[key], "cached": True})

    path = file_path(stem)
    ann = STATE["ann"]
    boxes = ann[(ann.file == stem) & (ann.slice == sl)]
    with GPU_LOCK, h5py.File(path, "r") as hf:
        if not 0 <= sl < hf["kspace"].shape[0]:
            abort(400, "slice out of range")
        r = tm.compute_slice(hf, stem, sl, boxes, accel, STATE["recon"], STATE["device"],
                             n_rebuilds=n, drop_frac=drop)

    gt = r["gt"]
    gmax = float(np.percentile(gt, 99.8))
    emax = float(np.percentile(r["err"], 99.5))
    ta = float(np.percentile(r["trust_a"], 99.5))
    tb = float(np.percentile(r["trust_b"], 99.5))
    fg = gt > 0.05 * gt.max()  # same brain mask as trustmap.compute_slice
    fg_e, fg_a, fg_b = (float(r[k][fg].mean()) for k in ("err", "trust_a", "trust_b"))
    res = dict(
        file=stem, slice=sl, accel=accel, n=n, drop=drop,
        shape=list(gt.shape),
        images=dict(gt=to_png(gt, gmax, "gray"), rec=to_png(r["rec"], gmax, "gray"),
                    zf=to_png(r["zf"], gmax, "gray"), err=to_png(r["err"], emax, "inferno"),
                    trust_a=to_png(r["trust_a"], ta, "inferno"), trust_b=to_png(r["trust_b"], tb, "inferno")),
        scales=dict(err=emax, trust_a=ta, trust_b=tb),
        metrics={k: clean(v) for k, v in r["slice_row"].items()},
        boxes=[{k: clean(v) for k, v in b.items()} for b in r["box_rows"]],
        fg_means=dict(err=fg_e, trust_a=fg_a, trust_b=fg_b),
        # float16 maps relative to the slice brain average, for the hover readout
        pixels=dict(err=f16(r["err"] / fg_e), trust_a=f16(r["trust_a"] / fg_a),
                    trust_b=f16(r["trust_b"] / fg_b)),
        masks=dict(main=r["main_mask"].astype(int).tolist(),
                   a=[m.astype(int).tolist() for m in r["masks_a"]],
                   b=[m.astype(int).tolist() for m in r["masks_b"]],
                   center=list(r["center"]), padding=list(r["padding"]), width=r["width"]),
        n_drop=r["n_drop"], n_outer=r["n_outer"], seconds=round(r["seconds"], 2),
        unique_a=len({m.tobytes() for m in r["masks_a"]}), fp16=STATE["recon"].amp,
    )
    CACHE[key] = res
    while len(CACHE) > CACHE_SIZE:
        CACHE.popitem(last=False)
    return jsonify({**res, "cached": False})


@app.get("/api/study")
def study():
    """Results of the batch run (python trustmap.py --accel 4 8), if present."""
    boxes, slices = [], []
    for accel in (4, 8):
        d = STATE["out_dir"] / f"acc{accel}"
        if (d / "boxes.csv").exists():
            boxes.append(pd.read_csv(d / "boxes.csv"))
        if (d / "slices.csv").exists():
            slices.append(pd.read_csv(d / "slices.csv"))
    if not boxes:
        return jsonify(available=False)
    b = pd.concat(boxes, ignore_index=True)
    s = pd.concat(slices, ignore_index=True)
    summary = []
    for accel, g in b.groupby("accel"):
        sg = s[s.accel == accel]
        for opt in "AB":
            rel_t = g[f"mean_trust_{opt}"] / g[f"fg_mean_trust_{opt}"]
            rel_e = g.mean_error / g.fg_mean_error
            _, sp_raw = tm.corr(g[f"mean_trust_{opt}"].values, g.mean_error.values)
            _, sp_rel = tm.corr(rel_t.values, rel_e.values)
            summary.append(dict(
                accel=int(accel), option=opt, n_boxes=int(len(g)), n_slices=int(len(sg)),
                pixel_spearman_fg_mean=clean(sg[f"spearman_{opt}_fg"].mean()),
                pixel_spearman_fg_min=clean(sg[f"spearman_{opt}_fg"].min()),
                pixel_spearman_fg_max=clean(sg[f"spearman_{opt}_fg"].max()),
                box_spearman_raw=clean(sp_raw), box_spearman_rel=clean(sp_rel),
                psnr_mean=clean(sg.psnr.mean()), ssim_mean=clean(sg.ssim.mean()),
                psnr_zf_mean=clean(sg.psnr_zero_filled.mean()),
            ))
    rows = b[["file", "slice", "label", "accel", "x", "y", "width", "height",
              "mean_trust_A", "mean_trust_B", "mean_error",
              "fg_mean_trust_A", "fg_mean_trust_B", "fg_mean_error"]]
    rows = [{k: clean(v) for k, v in r.items()} for r in rows.to_dict("records")]
    return jsonify(available=True, summary=summary, boxes=rows)


# --------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=Path, default=tm.DEFAULT_DATA)
    p.add_argument("--annotations", type=Path, default=tm.DEFAULT_ANNOTATIONS)
    p.add_argument("--weights", type=Path, default=tm.DEFAULT_WEIGHTS)
    p.add_argument("--out_dir", type=Path, default=tm.ROOT / "outputs", help="batch results for the Study tab")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ann = pd.read_csv(args.annotations)
    ann = ann[ann.study_level.str.strip() == "No"].copy()
    ann["slice"] = ann["slice"].astype(int)
    files = {}
    for f in sorted(args.data_dir.glob("*.h5")):
        with h5py.File(f, "r") as hf:
            files[f.stem] = int(hf["kspace"].shape[0])
    if not files:
        raise SystemExit(f"no .h5 files in {args.data_dir}")

    device = torch.device(args.device)
    print(f"Loading VarNet on {device} ...")
    STATE.update(ann=ann, files=files, data_dir=args.data_dir, out_dir=args.out_dir, device=device,
                 recon=tm.Reconstructor(tm.load_model(args.weights, device), device, amp=args.fp16))
    print(f"{len(files)} scans in {args.data_dir}. Open http://127.0.0.1:{args.port}")
    app.run(host="127.0.0.1", port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
