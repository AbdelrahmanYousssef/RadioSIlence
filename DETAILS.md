# Radio Silence: technical details

The [README](README.md) covers how to run the project. This file is for the team: what the code does, what was checked, and what was assumed.

## Command-line script

```bash
python trustmap.py --file file_brain_AXFLAIR_200_6002581 --slice 3   # one slice
python trustmap.py --accel 4 8                                        # every annotated slice, 4x and 8x
```

Flags: `--data_dir`, `--n_rebuilds 10`, `--drop_frac 0.05`, `--max_figures 6`, `--save_npz` (saves the maps as arrays), and `--fp16` (forces fp16 autocast; the script also switches to it automatically on a CUDA out-of-memory error).
Only slices with at least one fastMRI+ box are processed. Rows with `study_level == "Yes"` are skipped.

`trustmap.py` imports `fastmri` from `third_party/fastMRI`, so it never runs `pip install fastmri`, which can replace your CUDA build of torch.

## Outputs (`outputs/acc{4,8}/`)

| file | content |
|---|---|
| `boxes.csv` | one row per box: `file, slice, label, x, y, width, height, accel, mean_trust_A, mean_trust_B, mean_error`, plus foreground means of trust/error for the slice and `gt_max` (for normalising across slices) |
| `slices.csv` | one row per slice: PSNR / SSIM of the reconstruction (and zero-filled PSNR), Pearson and Spearman correlation of trust vs error over **all** pixels and over **foreground** pixels (`gt > 0.05 * max`) for A and B |
| `summary.txt` | mean/min/max of the per-slice correlations, and per-box correlation (raw, and normalised by the slice foreground mean) |
| `figures/*.png` | first N slices: GT, reconstruction, error and trust map, with boxes drawn on all four panels. Row 1 is trust A, row 2 is trust B |

All maps and figures use the **fastMRI+ labelling orientation**: the 320×320 image flipped up/down.

## Method

1. **Ground truth.** RSS over coils of `ifft2c(kspace)`, center-cropped to the shape of the
   file's `reconstruction_rss` (320×320). The script prints the maximum relative difference
   to the stored `reconstruction_rss` for each slice as a check.
2. **Mask.** The fastMRI "equispaced" mask with center fraction 0.08 at 4x and 0.04 at 8x.
   The mask is zeroed outside `padding_left/right` from the ISMRMRD header, the same way
   `VarNetDataTransform` does it. The main mask offset comes from the per-file seed, as in the
   fastMRI validation transform.
3. **Reconstruction.** `VarNet(num_cascades=12, pools=4, chans=18, sens_pools=4, sens_chans=8)`,
   in `torch.inference_mode()` with batch size 1. It is called as `model(masked_kspace, mask)`,
   the same call the official inference script makes.
4. **Trust maps.** Each is the pixel-wise std over 10 rebuilds.
   - **A:** 10 equispaced masks at the same acceleration with different offsets. The offsets
     are evenly spaced in `[0, adjusted_accel)`; see note (b) below.
   - **B:** the main mask, where each rebuild randomly drops `round(0.05 × n)` of the *n*
     acquired lines outside the contiguous center block. That is 3 of 54 lines at 4x and
     1 of 27 at 8x for 320 columns.
5. **Error map.** `|recon − GT|`.
6. **Per box.** The mean over `img[y:y+h, x:x+w]` in the flipped frame.

## What was verified in the repos vs assumed

**Verified by reading the code or running it:**
- **Model architecture.** The constructor in `run_pretrained_varnet_inference.py` and the
  defaults in `varnet_reproduce_20201111/varnet_brain_leaderboard.py` both give 12 cascades,
  18 chans, 4 pools, 8 sens chans and 4 sens pools. Loading `brain_leaderboard_state_dict.pt`
  with `strict=True` matches every key.
- **No dropout.** The training script uses equispaced masks with accelerations [4, 8] and
  center fractions [0.08, 0.04]. `VarNet` builds its U-Nets with `drop_prob=0.0`.
- **(a) Mask class.** The repo has since renamed its mask functions. At the 2020-11-11 commit
  (`cc42c7a`), `"equispaced"` was the version that adjusts the line spacing for the center
  lines, which is today's `EquispacedMaskFractionFunc`. Today's `EquiSpacedMaskFunc` is a
  different mask, so the script uses `EquispacedMaskFractionFunc`. It gives exactly 80 of
  320 lines at 4x and 40 of 320 at 8x.
- **Center detection.** The 2020 model found the center block from the contiguous run of
  ones in the mask, and the current code does the same when `num_low_frequencies` is not
  passed. Option B only drops lines outside that run, so the model's center (ACS) block
  does not change.
- **Box orientation.** fastmri-plus `example.ipynb` reads `reconstruction_rss`, flips it with
  `[:, ::-1, :]`, and draws `(x, y)` to `(x+w, y+h)` directly on the flipped image. The DICOM
  script's default is `flip_up_down=True` and `flip_left_right=False`.
- **Real-data orientation check.** On the three scans in `Project/DATA`, every one of the 12 small
  "lesion" boxes is brighter than the ring of pixels around it when the image is flipped up/down
  (median ratio 1.11; WM lesions are bright on FLAIR). Without the flip only 4 of 12 are (median
  0.99), and with an extra left/right flip it is 8 of 12. The flip convention holds.
- **Acquired columns.** On these scans the k-space has data only in columns 48 to 271 (224 of 320;
  the header's `padding_left/right`). The mask is zeroed outside them, the same as `VarNetDataTransform`.
  "4x" therefore keeps 62 lines of the 224 actually acquired, about 3.6x. That is the repo's own
  behaviour, not a bug.
- **Synthetic end-to-end test.** I ran the full pipeline on a synthetic 16-coil scan with a
  planted lesion. The box drawn in the flipped frame lands on the lesion, the computed GT
  matches the stored RSS (relative difference about 1e-7), and the fp16 path runs.
  Peak GPU memory is about 0.95 GB in fp32, and a slice (21 reconstructions) takes about 6 s.

**Assumed or chosen (please sanity-check):**
- **(b) Option A offsets.** With 320 columns, integer offsets only give round(5.44) = 5
  distinct masks at 4x (11 at 8x). To get 10 different rebuilds I use fractional start
  offsets `k·adj/10`. All masks have the same spacing and line count, but some are only a
  one-line shift of each other. The printed "A unique masks" count shows how many were
  distinct.
- **Main mask offset.** The main reconstruction uses its own offset (from the file seed), and
  the A rebuilds are not built around it.
- **Option B drop count.** "~5%" is rounded to a whole number of lines, with at least 1. At 8x
  that is a single line, so trust B is a weak perturbation there.
- **Foreground threshold.** `gt > 0.05·max(gt)` is my choice. Correlations over all pixels are
  inflated by the near-zero background, so both versions are reported.
- **Box pixels.** I take a box as covering `w × h` pixels. PIL's rectangle in the notebook
  draws its outline one pixel wider.
- **Missing `reconstruction_rss`.** If a file has no `reconstruction_rss`, the crop falls back
  to `reconSpace` (and the "FLAIR 203" rule from the inference script). Boxes are then only
  trustworthy if that crop is still 320×320.
- **Brain mask in the app.** The app's per-pixel readout and box ratios divide by the mean over
  the same brain mask (`gt > 0.05 · max`).
