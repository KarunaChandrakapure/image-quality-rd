#!/usr/bin/env python3
"""
deblur_compare.py - apply several classical (non-AI) deblurring / sharpening
methods to one image and stack the results into a single comparison sheet.

Usage:
    python deblur_compare.py my_blurry_image.jpg
    python deblur_compare.py my_blurry_image.jpg --config deblur_config.yaml
    python deblur_compare.py my_blurry_image.jpg --out results
    python deblur_compare.py --batch my_folder --method enhanced     # whole folder

Install:
    pip install opencv-python scikit-image pyyaml pillow numpy
"""

import argparse
import csv
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageFilter
from skimage import restoration

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
INTERP = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sharpness_score(img):
    """Variance of the Laplacian. Higher = more edge contrast.
    Note: noise and halos also raise it, so always judge visually too."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ----------------------------------------------------------------------
# PSF (blur kernel) builders for the deconvolution methods
# ----------------------------------------------------------------------
def gaussian_psf(size, sigma):
    size = int(size)
    if size % 2 == 0:
        size += 1
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return (k / k.sum()).astype(np.float32)


def motion_psf(length, angle):
    length = max(int(length), 3)
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0
    center = (length / 2 - 0.5, length / 2 - 0.5)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    k = cv2.warpAffine(k, M, (length, length))
    s = k.sum()
    return k / s if s > 0 else k


def build_psf(psf_cfg):
    if psf_cfg["type"] == "motion":
        return motion_psf(psf_cfg["motion_length"], psf_cfg["motion_angle"])
    return gaussian_psf(psf_cfg["gaussian_size"], psf_cfg["gaussian_sigma"])


def apply_per_channel(img, func, luminance_only, pad=0):
    """Run a float [0,1] single-channel function on the image.
    luminance_only=True -> only the Y channel (faster, avoids colour fringes).
    pad > 0 -> mirror-pad the image first and crop afterwards. Deconvolution works
    in the frequency domain and otherwise creates ringing/frames at the borders."""
    h, w = img.shape[:2]
    pad = int(min(pad, h - 1, w - 1))
    if pad > 0:
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)

    def crop(a):
        return a[pad:pad + h, pad:pad + w] if pad > 0 else a

    if luminance_only:
        ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(np.float32) / 255.0
        ycc[..., 0] = np.clip(func(ycc[..., 0]), 0, 1)
        out = crop((ycc * 255).clip(0, 255).astype(np.uint8))
        return cv2.cvtColor(out, cv2.COLOR_YCrCb2BGR)
    f = img.astype(np.float32) / 255.0
    chans = [np.clip(func(f[..., c]), 0, 1) for c in range(3)]
    return crop((np.dstack(chans) * 255).clip(0, 255).astype(np.uint8))


# ----------------------------------------------------------------------
# Methods  (each: uint8 BGR image, params dict -> uint8 BGR image)
# ----------------------------------------------------------------------
def m_unsharp_mask(img, p, **_):
    blurred = cv2.GaussianBlur(img, (0, 0), p["sigma"])
    sharp = cv2.addWeighted(img, 1 + p["amount"], blurred, -p["amount"], 0)
    if p.get("threshold", 0) > 0:
        diff = np.abs(img.astype(np.int16) - blurred.astype(np.int16))
        mask = diff.max(axis=2) < p["threshold"]
        sharp[mask] = img[mask]
    return sharp


def m_pillow_unsharp(img, p, **_):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pil = pil.filter(ImageFilter.UnsharpMask(
        radius=p["radius"], percent=p["percent"], threshold=p["threshold"]))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def m_richardson_lucy(img, p, psf=None, **_):
    def rl(ch):
        try:
            return restoration.richardson_lucy(
                ch, psf, num_iter=p["num_iter"], clip=p.get("clip", True))
        except TypeError:  # older scikit-image uses "iterations"
            return restoration.richardson_lucy(
                ch, psf, iterations=p["num_iter"], clip=p.get("clip", True))
    return apply_per_channel(img, rl, p.get("luminance_only", True), pad=3 * max(psf.shape))


def m_wiener(img, p, psf=None, **_):
    def wn(ch):
        return restoration.wiener(ch, psf, p["balance"])
    return apply_per_channel(img, wn, p.get("luminance_only", True), pad=3 * max(psf.shape))


def step_denoise(img, p):
    return cv2.fastNlMeansDenoisingColored(
        img, None, p["h"], p["h_color"], p["template_window"], p["search_window"])


def step_upscale(img, p):
    s = p["scale"]
    return cv2.resize(img, None, fx=s, fy=s, interpolation=INTERP[p["interpolation"]])


def step_clahe(img, p):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    t = int(p["tile_grid"])
    l = cv2.createCLAHE(clipLimit=p["clip_limit"], tileGridSize=(t, t)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


METHODS = {
    "unsharp_mask": m_unsharp_mask,
    "pillow_unsharp": m_pillow_unsharp,
    "richardson_lucy": m_richardson_lucy,
    "wiener": m_wiener,
}


def m_full_pipeline(img, p, psf=None, all_methods_cfg=None, **_):
    if p["denoise"]["enabled"]:
        img = step_denoise(img, p["denoise"])
    if p["upscale"]["enabled"]:
        img = step_upscale(img, p["upscale"])
    if p["sharpen"]["enabled"]:
        name = p["sharpen"]["method"]
        if name not in METHODS:
            sys.exit(f"[config error] full_pipeline.sharpen.method '{name}' is not valid. "
                     f"Choose from: {', '.join(METHODS)}")
        params = {**all_methods_cfg[name], **(p["sharpen"].get("params") or {})}
        img = METHODS[name](img, params, psf=psf)
    if p["clahe"]["enabled"]:
        img = step_clahe(img, p["clahe"])
    return img



# ----------------------------------------------------------------------
# ENHANCE pipeline: sharpness + visibility + fg/bg separation + tone/colour
# NEVER resizes the output (same width x height as the input).
# ----------------------------------------------------------------------
def unsharp_luma(img, sigma, amount, threshold=0):
    """Unsharp mask on brightness only -> no coloured halos."""
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y = ycc[..., 0].astype(np.float32)
    d = y - cv2.GaussianBlur(y, (0, 0), sigma)
    if threshold > 0:
        d = np.where(np.abs(d) < threshold, 0, d)
    ycc[..., 0] = np.clip(y + amount * d, 0, 255).astype(np.uint8)
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


def step_white_balance(img, p):
    """Gray-world white balance, blended by 'strength' (0 = off, 1 = full)."""
    f = img.astype(np.float32)
    means = f.reshape(-1, 3).mean(axis=0)
    gain = means.mean() / np.maximum(means, 1e-6)
    gain = 1 + (gain - 1) * p["strength"]
    return np.clip(f * gain, 0, 255).astype(np.uint8)


def step_tone(img, p):
    """Exposure: auto levels, auto gamma, shadow lift, highlight recovery (on L channel)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[..., 0].astype(np.float32) / 255.0

    if p["auto_levels"]:
        lo, hi = np.percentile(L, [p["clip_low"], 100 - p["clip_high"]])
        if hi - lo > 1e-3:
            L = np.clip((L - lo) / (hi - lo), 0, 1)

    if p["auto_gamma"]:
        m = float(np.clip(L.mean(), 0.05, 0.95))
        gamma = np.log(p["target_mean"]) / np.log(m)
        gamma = float(np.clip(gamma, p["gamma_min"], p["gamma_max"]))
    else:
        gamma = p["gamma"]  # <1 brightens, >1 darkens
    L = np.power(np.clip(L, 0, 1), gamma)

    base = cv2.GaussianBlur(L, (0, 0), p["tonal_radius"])  # smooth "how bright is this area"
    L = L + p["shadows"] * (1 - base) ** 2 * (1 - L)       # lift dark areas
    L = L - p["highlights"] * base ** 2 * L                # tame bright areas

    lab[..., 0] = np.clip(L * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def step_local_contrast(img, p):
    """CLAHE + 'clarity' (large-radius unsharp) on L -> elements stand out more."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[..., 0]
    t = int(p["tile_grid"])
    Lc = cv2.createCLAHE(clipLimit=p["clahe_clip"], tileGridSize=(t, t)).apply(L)
    Lf = cv2.addWeighted(L.astype(np.float32), 1 - p["clahe_strength"],
                         Lc.astype(np.float32), p["clahe_strength"], 0)
    if p["clarity_amount"] > 0:
        Lf = Lf + p["clarity_amount"] * (Lf - cv2.GaussianBlur(Lf, (0, 0), p["clarity_radius"]))
    lab[..., 0] = np.clip(Lf, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def step_sharpen(img, p, psf, mcfg):
    name = p["method"]
    if name == "none":
        return img
    if name not in METHODS:
        sys.exit(f"[config error] enhanced.sharpen.method '{name}' invalid. "
                 f"Choose from: none, {', '.join(METHODS)}")
    params = {**mcfg[name], **(p.get("params") or {})}
    if name == "unsharp_mask" and p.get("luminance_only", True):
        return unsharp_luma(img, params["sigma"], params["amount"], params.get("threshold", 0))
    return METHODS[name](img, params, psf=psf)


def make_fg_mask(img, p):
    """Soft foreground mask in [0,1], same size as img. Classical methods only."""
    h, w = img.shape[:2]
    mode = p["mask"]
    if mode == "grabcut":
        # GrabCut runs on a small internal copy for speed; the mask is scaled back up.
        scale = min(1.0, p["grabcut_work_size"] / max(h, w))
        small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img
        sh, sw = small.shape[:2]
        m = p["grabcut_margin"]
        rect = (int(sw * m), int(sh * m), max(int(sw * (1 - 2 * m)), 1), max(int(sh * (1 - 2 * m)), 1))
        gc = np.zeros((sh, sw), np.uint8)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(small, gc, rect, bgd, fgd, int(p["grabcut_iter"]), cv2.GC_INIT_WITH_RECT)
        fg = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.float32)
        fg = cv2.resize(fg, (w, h), interpolation=cv2.INTER_LINEAR)
    elif mode == "focus":
        # sharper (in-focus) regions are treated as foreground
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        lap = np.abs(cv2.Laplacian(cv2.GaussianBlur(gray, (0, 0), 1.0), cv2.CV_32F))
        energy = cv2.GaussianBlur(lap, (0, 0), p["focus_window"])
        lo, hi = np.percentile(energy, [5, 95])
        fg = np.clip((energy - lo) / (hi - lo + 1e-6), 0, 1)
    elif mode == "center":
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        d = ((xx - w / 2) / (w * p["center_radius"])) ** 2 + ((yy - h / 2) / (h * p["center_radius"])) ** 2
        fg = np.exp(-d).astype(np.float32)
    else:
        sys.exit(f"[config error] separation.mask '{mode}' invalid. Use grabcut | focus | center")
    if p["feather"] > 0:
        fg = cv2.GaussianBlur(fg, (0, 0), p["feather"])
    return np.clip(fg, 0, 1).astype(np.float32)


def _hsv_scale(img, sat, val):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * sat, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * val, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)


def step_separation(img, p, out_dir=None, stem=""):
    """Make the subject pop: slightly boost foreground, calm down the background."""
    mask = make_fg_mask(img, p)
    if out_dir is not None:
        cv2.imwrite(str(Path(out_dir) / f"{stem}_fg_mask.png"), (mask * 255).astype(np.uint8))
    fg = _hsv_scale(img, p["fg_saturation"], p["fg_brightness"])
    bg = _hsv_scale(img, p["bg_saturation"], p["bg_brightness"])
    if p["bg_blur"] > 0:
        bg = cv2.GaussianBlur(bg, (0, 0), p["bg_blur"])
    m = mask[..., None]
    return np.clip(fg * m + bg * (1 - m), 0, 255).astype(np.uint8)


def step_color(img, p):
    """Saturation + vibrance (vibrance protects already-vivid colours and greys)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    s = hsv[..., 1] / 255.0
    s = s * p["saturation"]
    s = s + p["vibrance"] * 2.0 * s * (1 - s)
    hsv[..., 1] = np.clip(s, 0, 1) * 255
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def m_enhance(img, p, psf=None, all_methods_cfg=None, out_dir=None, stem="", save_steps=False, **_):
    out = img.copy()
    counter = [1]

    def rec(name):
        if save_steps and out_dir is not None:
            cv2.imwrite(str(Path(out_dir) / f"{stem}_enh_{counter[0]:02d}_{name}.png"), out)
            counter[0] += 1

    if p["denoise"]["enabled"]:
        out = step_denoise(out, p["denoise"]); rec("denoise")
    if p["white_balance"]["enabled"]:
        out = step_white_balance(out, p["white_balance"]); rec("white_balance")
    if p["tone"]["enabled"]:
        out = step_tone(out, p["tone"]); rec("tone")
    if p["local_contrast"]["enabled"]:
        out = step_local_contrast(out, p["local_contrast"]); rec("local_contrast")
    if p["sharpen"]["enabled"]:
        out = step_sharpen(out, p["sharpen"], psf, all_methods_cfg); rec("sharpen")
    if p["separation"]["enabled"]:
        out = step_separation(out, p["separation"], out_dir, stem); rec("separation")
    if p["color"]["enabled"]:
        out = step_color(out, p["color"]); rec("color")

    assert out.shape == img.shape, "enhance pipeline must not change image size"
    return out


# ----------------------------------------------------------------------
# Comparison sheet
# ----------------------------------------------------------------------
def make_cell(img, title, orig_shape, comp_cfg):
    """Resize result to the original's size, optionally crop+zoom, add a title band."""
    oh, ow = orig_shape[:2]
    cell = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_AREA
                      if img.shape[0] > oh else cv2.INTER_LANCZOS4)

    crop = comp_cfg["crop"]
    if crop["enabled"]:
        x, y = int(crop["x"]), int(crop["y"])
        w, h = int(crop["width"]), int(crop["height"])
        cell = cell[y:y + h, x:x + w]
        z = crop.get("zoom", 1)
        if z != 1:
            cell = cv2.resize(cell, None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST)

    if comp_cfg.get("cell_height"):
        ch = int(comp_cfg["cell_height"])
        scale = ch / cell.shape[0]
        cell = cv2.resize(cell, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    band = np.full((comp_cfg["label_height"], cell.shape[1], 3), 30, np.uint8)
    fs = comp_cfg["font_scale"]
    while fs > 0.25 and cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)[0][0] > cell.shape[1] - 16:
        fs -= 0.05  # shrink text until the title fits the cell width
    cv2.putText(band, title, (8, int(comp_cfg["label_height"] * 0.68)),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2 if fs > 0.45 else 1, cv2.LINE_AA)
    return np.vstack([band, cell])


def stack_grid(cells, columns):
    h = max(c.shape[0] for c in cells)
    w = max(c.shape[1] for c in cells)
    padded = []
    for c in cells:
        canvas = np.full((h, w, 3), 30, np.uint8)
        canvas[:c.shape[0], :c.shape[1]] = c
        padded.append(canvas)
    while len(padded) % columns:
        padded.append(np.full((h, w, 3), 30, np.uint8))
    rows = [np.hstack(padded[i:i + columns]) for i in range(0, len(padded), columns)]
    return np.vstack(rows)



# ----------------------------------------------------------------------
# Sweep: try many PSFs x settings at once to FIND the right deconvolution values
# ----------------------------------------------------------------------
def run_sweep(img, kind, cfg, out_dir, stem):
    sw, mcfg, comp = cfg["sweep"], cfg["methods"], cfg["comparison"]
    psfs = []
    for sg in sw["gaussian_sigmas"]:
        size = max(3, int(np.ceil(6 * sg)) | 1)
        psfs.append((f"gauss s={sg}", gaussian_psf(size, sg)))
    for ln in sw["motion_lengths"]:
        psfs.append((f"motion L={ln} a={sw['motion_angle']}", motion_psf(ln, sw["motion_angle"])))
    values = sw["wiener_balances"] if kind == "wiener" else sw["rl_iters"]
    tag = "bal" if kind == "wiener" else "it"

    cells = []
    for pname, psf in psfs:
        for v in values:
            if kind == "wiener":
                res = m_wiener(img, {**mcfg["wiener"], "balance": v}, psf=psf)
            else:
                res = m_richardson_lucy(img, {**mcfg["richardson_lucy"], "num_iter": v}, psf=psf)
            cells.append(make_cell(res, f"{pname} {tag}={v}", img.shape, comp))
        print(f"  done: {pname}")
    sheet = stack_grid(cells, len(values))
    path = out_dir / f"{stem}_sweep_{kind}.png"
    cv2.imwrite(str(path), sheet)
    print(f"\nSweep sheet: {path}")
    print(f"Rows = blur kernels ({len(psfs)}), columns = {tag} values {values}")
    print("Pick the cell that looks sharpest WITHOUT ringing/noise, then copy its values into the config.")


# ----------------------------------------------------------------------
# BATCH: process a whole folder with ONE method (output size never changes)
# ----------------------------------------------------------------------
BATCH_METHODS = ["unsharp_mask", "pillow_unsharp", "richardson_lucy", "wiener",
                 "full_pipeline", "enhanced"]


def apply_method(key, img, cfg, psf, out_dir=None, stem="", save_steps=False):
    mcfg = cfg["methods"]
    if key in METHODS:
        return METHODS[key](img, mcfg[key], psf=psf)
    if key == "full_pipeline":
        return m_full_pipeline(img, mcfg["full_pipeline"], psf=psf, all_methods_cfg=mcfg)
    if key == "enhanced":
        return m_enhance(img, mcfg["enhanced"], psf=psf, all_methods_cfg=mcfg,
                         out_dir=out_dir, stem=stem, save_steps=save_steps)
    sys.exit(f"Unknown method '{key}'. Choose from: {', '.join(BATCH_METHODS)}")


def _init_worker():
    cv2.setNumThreads(1)  # avoid CPU oversubscription when several workers run


def _write_image(path, img, bcfg):
    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        return cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, int(bcfg["jpg_quality"])])
    return cv2.imwrite(str(path), img)


def _batch_worker(task):
    src, rel, cfg, key, dst_root = task
    bcfg = cfg["batch"]
    t0 = time.time()
    row = {"file": rel.as_posix(), "width": "", "height": "",
           "sharpness_before": "", "sharpness_after": "", "seconds": "", "status": ""}
    try:
        save_result = bcfg.get("save_result", True)
        dst = dst_root / rel
        fmt = bcfg["output_format"]
        if fmt in ("png", "jpg"):
            dst = dst.with_suffix("." + fmt)
        sub = bcfg.get("comparison_dir", "_comparisons").replace("{method}", key)
        cdir = (dst_root / sub / rel.parent) if sub else (dst_root / rel.parent)
        cmp_path = cdir / f"{rel.stem}_compare.png"

        # resume logic: skip if the file(s) this run would create already exist
        check = dst if save_result else cmp_path
        if check.exists() and not bcfg["overwrite"]:
            row["status"] = "skipped (exists)"
            return row

        img = cv2.imread(str(src))
        if img is None:
            row["status"] = "ERROR: unreadable image"
            return row
        res = apply_method(key, img, cfg, build_psf(cfg["methods"]["psf"]))
        if res.shape != img.shape:
            raise RuntimeError("method changed the image size")

        if save_result:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not _write_image(dst, res, bcfg):
                raise RuntimeError("could not write output")
        if bcfg["save_comparison"]:
            comp = {**cfg["comparison"], "crop": {**cfg["comparison"]["crop"], "enabled": False}}
            if bcfg.get("comparison_labels", True):
                cells = [make_cell(img, "Original", img.shape, comp),
                         make_cell(res, key, img.shape, comp)]
            else:
                cells = [img, res]
            cols = 1 if bcfg.get("comparison_layout", "side_by_side") == "stacked" else 2
            cdir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(cmp_path), stack_grid(cells, cols))
        row.update(width=img.shape[1], height=img.shape[0],
                   sharpness_before=round(sharpness_score(img), 1),
                   sharpness_after=round(sharpness_score(res), 1), status="ok")
    except Exception as e:  # keep going; one bad file must not stop the batch
        row["status"] = f"ERROR: {e}"
    row["seconds"] = round(time.time() - t0, 2)
    return row


def run_batch(args, cfg):
    bcfg = cfg["batch"]
    in_dir = Path(args.batch).resolve()
    if not in_dir.is_dir():
        sys.exit(f"Not a folder: {args.batch}")
    out_dir = Path(args.out or cfg["general"]["output_dir"]).resolve()
    if out_dir == in_dir:
        sys.exit("Output folder must be different from the input folder "
                 "(otherwise your originals could be overwritten).")
    key = args.method or bcfg["method"]
    if key not in BATCH_METHODS:
        sys.exit(f"Unknown method '{key}'. Choose from: {', '.join(BATCH_METHODS)}")

    if not bcfg.get("save_result", True) and not bcfg["save_comparison"]:
        sys.exit("Nothing to save: batch.save_result and batch.save_comparison are both false.")
    exts = {e.lower() for e in bcfg["extensions"]}
    it = in_dir.rglob("*") if bcfg["recursive"] else in_dir.glob("*")
    files = sorted(p for p in it if p.is_file() and p.suffix.lower() in exts
                   and out_dir not in p.resolve().parents)
    if args.limit:
        files = files[:args.limit]
    if not files:
        sys.exit(f"No images with extensions {sorted(exts)} found in {in_dir}")

    if cfg["general"].get("max_side"):
        print("Note: general.max_side is ignored in batch mode (outputs keep their original size).")
    workers = int(bcfg["workers"]) or (os.cpu_count() or 1)
    workers = max(1, min(workers, len(files)))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Batch: {len(files)} image(s) | method: {key} | workers: {workers}")
    print(f"  in : {in_dir}\n  out: {out_dir}\n")

    tasks = [(str(p), p.relative_to(in_dir), cfg, key, out_dir) for p in files]
    rows, t0 = [], time.time()
    if workers > 1:
        with mp.Pool(workers, initializer=_init_worker) as pool:
            iterator = pool.imap_unordered(_batch_worker, tasks, chunksize=1)
            for i, row in enumerate(iterator, 1):
                rows.append(row)
                print(f"[{i}/{len(tasks)}] {row['file']}  {row['status']}")
    else:
        for i, task in enumerate(tasks, 1):
            row = _batch_worker(task)
            rows.append(row)
            print(f"[{i}/{len(tasks)}] {row['file']}  {row['status']}")

    rows.sort(key=lambda r: r["file"])
    if bcfg["save_report"]:
        rpt = out_dir / "_report.csv"
        if rpt.exists():  # resumed run: keep earlier results for files that were skipped now
            with open(rpt, newline="", encoding="utf-8") as f:
                prev = {r["file"]: r for r in csv.DictReader(f)}
            rows = [prev[r["file"]] if r["status"].startswith("skipped")
                    and prev.get(r["file"], {}).get("status") == "ok" else r for r in rows]
        with open(rpt, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nReport: {rpt}")

    ok = [r for r in rows if r["status"] == "ok"]
    skipped = [r for r in rows if r["status"].startswith("skipped")]
    failed = [r for r in rows if r["status"].startswith("ERROR")]
    print(f"Done in {time.time() - t0:.1f}s: {len(ok)} processed, {len(skipped)} skipped, {len(failed)} failed")
    if ok:
        b = np.mean([r["sharpness_before"] for r in ok])
        a = np.mean([r["sharpness_after"] for r in ok])
        print(f"Average sharpness score: {b:.1f} -> {a:.1f} (noise also raises it; check images visually)")
    for r in failed:
        print(f"  FAILED: {r['file']} -> {r['status']}")

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Compare classical deblurring methods.")
    ap.add_argument("image", nargs="?", help="path to the blurry input image (omit when using --batch)")
    ap.add_argument("--config", default=str(Path(__file__).with_name("deblur_config.yaml")),
                    help="path to YAML config (default: deblur_config.yaml next to this script)")
    ap.add_argument("--out", default=None, help="output folder (overrides config)")
    ap.add_argument("--only", default=None,
                    help="comma-separated methods to run, e.g. --only enhanced "
                         "(original is always included). Overrides the 'enabled' block.")
    ap.add_argument("--sweep", choices=["wiener", "rl"], default=None,
                    help="try many blur kernels / settings in one sheet to find good "
                         "deconvolution values (see 'sweep' in the config)")
    ap.add_argument("--batch", metavar="FOLDER", default=None,
                    help="process every image in FOLDER with one method (see 'batch' in the config)")
    ap.add_argument("--method", default=None, choices=BATCH_METHODS,
                    help="method to use in --batch mode (overrides batch.method in the config)")
    ap.add_argument("--limit", type=int, default=None,
                    help="--batch only: process just the first N images (good for a quick test)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.batch:
        run_batch(args, cfg)
        return
    if not args.image:
        ap.error("provide an image path, or use --batch FOLDER")
    gen, enabled, comp = cfg["general"], cfg["enabled"], cfg["comparison"]
    mcfg = cfg["methods"]
    out_dir = Path(args.out or gen["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"Could not read image: {args.image}")

    if gen.get("max_side"):
        m = int(gen["max_side"])
        h, w = img.shape[:2]
        if max(h, w) > m:
            s = m / max(h, w)
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    print(f"Input: {args.image}  ({img.shape[1]}x{img.shape[0]})")

    psf = build_psf(mcfg["psf"])
    stem = Path(args.image).stem
    if args.sweep:
        run_sweep(img, args.sweep, cfg, out_dir, stem)
        return
    only = {k.strip() for k in args.only.split(",")} if args.only else None

    jobs = [
        ("original", "Original", lambda: img.copy()),
        ("unsharp_mask", "OpenCV Unsharp Mask", lambda: m_unsharp_mask(img, mcfg["unsharp_mask"])),
        ("pillow_unsharp", "Pillow Unsharp Mask", lambda: m_pillow_unsharp(img, mcfg["pillow_unsharp"])),
        ("richardson_lucy", "Richardson-Lucy", lambda: m_richardson_lucy(img, mcfg["richardson_lucy"], psf=psf)),
        ("wiener", "Wiener", lambda: m_wiener(img, mcfg["wiener"], psf=psf)),
        ("full_pipeline", "Full Pipeline", lambda: m_full_pipeline(
            img, mcfg["full_pipeline"], psf=psf, all_methods_cfg=mcfg)),
        ("enhanced", "Enhanced (no resize)", lambda: m_enhance(
            img, mcfg["enhanced"], psf=psf, all_methods_cfg=mcfg, out_dir=out_dir,
            stem=stem, save_steps=gen.get("save_steps", False))),
    ]

    cells, rows = [], []
    for key, title, fn in jobs:
        if only is not None:
            if key != "original" and key not in only:
                continue
        elif not enabled.get(key, False):
            continue
        t0 = time.time()
        result = fn()
        dt = time.time() - t0
        if gen.get("save_individual", True):
            cv2.imwrite(str(out_dir / f"{stem}_{key}.png"), result)
        cells.append(make_cell(result, title, img.shape, comp))
        # score on a common size so upscaled results are comparable
        common = cv2.resize(result, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
        rows.append((title, sharpness_score(common), dt))
        print(f"  done: {title:<22} {dt:6.2f}s")

    if not cells:
        sys.exit("No methods enabled in the config.")

    sheet = stack_grid(cells, int(comp["columns"]))
    sheet_path = out_dir / f"{stem}_comparison.png"
    cv2.imwrite(str(sheet_path), sheet)

    print("\nSharpness score (variance of Laplacian, higher = sharper; noise also raises it)")
    print(f"  {'Method':<22} {'Score':>10} {'Time(s)':>9}")
    for title, score, dt in rows:
        print(f"  {title:<22} {score:>10.1f} {dt:>9.2f}")
    print(f"\nComparison sheet: {sheet_path}")


if __name__ == "__main__":
    main()