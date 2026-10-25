#!/usr/bin/env python3
"""
cas_sharpen.py - "pillow_unsharp, fixed": one sharpening method that targets the
three weaknesses seen with Pillow's UnsharpMask on the nuScenes frames.

    Pillow weakness                     Fix used here
    ---------------------------------   ------------------------------------------
    1. halos / clipping on strong edges Contrast Adaptive Sharpening (AMD FidelityFX
                                        CAS): sharpens less where contrast is high
    2. noise boosted in dark/flat areas light non-local-means denoise BEFORE sharpening
    3. colour fringes / colour noise    sharpen the brightness (Y) channel only

Pipeline per image:  denoise -> BGR to YCrCb -> CAS on Y -> back to BGR

For every input it saves a comparison sheet  Original | pillow_unsharp | cas_plus
and writes _report_cas.csv with sharpness, flat-area noise and clipping for both.

Usage:
    python 4.cas_sharpen.py
    python 4.cas_sharpen.py --input my_folder --out output --sharpness 0.8
    python 4.cas_sharpen.py --save-result          # also save the clean result images

Install:
    pip install opencv-python pillow numpy
"""

import argparse
import csv
import os
import time

import cv2
import numpy as np
from PIL import Image, ImageFilter

# ----------------------------------------------------------------------
# Settings (all can be overridden on the command line)
# ----------------------------------------------------------------------
DEFAULTS = {
    "input": r"F:\kpit\image-data-quality\sample2\resized_images",
    "out": r"F:\kpit\image-data-quality\output",
    "denoise_h": 3,             # brightness denoise strength (0 = off). 2-4 is light.
                                # Colour is not denoised: that washed out saturation by ~8%
    "sharpness": 1.0,           # CAS strength 0-1
    "passes": 1,                # repeat CAS. Compounds fast: 2 passes ~5x flat-area noise
    # Edge gate: sharpen object outlines, leave fine texture/flat areas alone.
    # Keeps busy backgrounds (glass, brick, trees) from turning grainy and
    # competing with the foreground. Set edge_high to 0 to sharpen everywhere.
    "edge_sigma": 1.5,          # blur (px) before measuring edges. Bigger = only larger shapes
    "edge_low": 0.15,           # edge strength below this -> no sharpening
    "edge_high": 0.45,          # edge strength above this -> full sharpening. Lower both
                                # to sharpen more of the image (and more texture)
    # pillow_unsharp settings, only used for the side-by-side reference column
    "pil_radius": 2,
    "pil_percent": 150,
    "pil_threshold": 3,
}

EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
LABEL_HEIGHT = 40


# ----------------------------------------------------------------------
# Method
# ----------------------------------------------------------------------
def cas(y, sharpness):
    """Contrast Adaptive Sharpening on one float [0,1] channel.
    Follows AMD FidelityFX CAS: a 3x3 cross filter with a negative lobe whose weight
    shrinks where the local neighbourhood is already close to black or white."""
    kernel = np.ones((3, 3), np.uint8)
    mn = cv2.erode(y, kernel, borderType=cv2.BORDER_REFLECT_101)    # local minimum
    mx = cv2.dilate(y, kernel, borderType=cv2.BORDER_REFLECT_101)   # local maximum

    # amp -> 0 near clipping / strong contrast, -> 1 in soft low-contrast areas
    amp = np.sqrt(np.clip(np.minimum(mn, 1.0 - mx) / np.maximum(mx, 1e-5), 0, 1))
    peak = -1.0 / (8.0 - 3.0 * sharpness)          # lerp(8, 5, sharpness)
    w = amp * peak

    p = cv2.copyMakeBorder(y, 1, 1, 1, 1, cv2.BORDER_REFLECT_101)
    cross = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    return np.clip((y + w * cross) / (1.0 + 4.0 * w), 0, 1)


def structure_mask(y, s):
    """0..1 weight: 1 on object outlines (people, cars, text), 0 on fine texture and
    flat areas. Edges are measured on a blurred copy, so small repeating patterns
    (glass dots, brick, foliage, sensor noise) fade out and only larger shapes remain."""
    if s["edge_high"] <= 0:
        return np.ones_like(y)
    g = cv2.GaussianBlur(y, (0, 0), s["edge_sigma"])
    mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    t = np.clip((mag - s["edge_low"]) / max(s["edge_high"] - s["edge_low"], 1e-6), 0, 1)
    t = t * t * (3 - 2 * t)                                   # smoothstep
    return cv2.GaussianBlur(t, (0, 0), 1.0)                   # soft transitions


def cas_plus(img, s):
    """uint8 BGR -> uint8 BGR.
    Everything happens on the brightness (Y) channel; colour is never touched:
    denoise Y -> CAS on Y -> keep the sharpening only where structure_mask says so."""
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y8 = ycc[..., 0]
    if s["denoise_h"] > 0:
        y8 = cv2.fastNlMeansDenoising(y8, None, s["denoise_h"], 7, 21)
    y = y8.astype(np.float32) / 255.0
    sharp = y
    for _ in range(s["passes"]):
        sharp = cas(sharp, s["sharpness"])
    y = y + structure_mask(y, s) * (sharp - y)
    ycc[..., 0] = (np.clip(y, 0, 1) * 255 + 0.5).astype(np.uint8)
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


def pillow_unsharp(img, s):
    """Same as m_pillow_unsharp in 2.deblur_compare.py (reference column)."""
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pil = pil.filter(ImageFilter.UnsharpMask(
        radius=s["pil_radius"], percent=s["pil_percent"], threshold=s["pil_threshold"]))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ----------------------------------------------------------------------
# Measurements (each compared with the original)
# ----------------------------------------------------------------------
def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)


def sharpness_score(img):
    """Variance of the Laplacian. Noise and halos also raise it - judge visually too."""
    return float(cv2.Laplacian(gray(img), cv2.CV_64F).var())


def flat_noise(img, flat_mask):
    """Fine-grain variation inside areas that are flat in the original (walls, road)."""
    g = gray(img)
    return float((g - cv2.GaussianBlur(g, (5, 5), 0))[flat_mask].std())


def flat_mask_of(original):
    g = gray(original)
    grad = np.hypot(cv2.Sobel(g, cv2.CV_64F, 1, 0), cv2.Sobel(g, cv2.CV_64F, 0, 1))
    return grad < np.percentile(grad, 30)


def clip_percent(img):
    """% of pixels at pure black or pure white."""
    g = gray(img)
    return float(((g <= 2) | (g >= 253)).mean() * 100)


# ----------------------------------------------------------------------
# Comparison sheet
# ----------------------------------------------------------------------
def labelled(img, title):
    band = np.full((LABEL_HEIGHT, img.shape[1], 3), 30, np.uint8)
    cv2.putText(band, title, (8, int(LABEL_HEIGHT * 0.68)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([band, img])


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    ap.add_argument("--save-result", action="store_true", help="also save each cas_plus image")
    s = vars(ap.parse_args())

    files = sorted(f for f in os.listdir(s["input"]) if f.lower().endswith(EXTENSIONS))
    if not files:
        print(f"No images found in {s['input']}")
        return

    comp_dir = os.path.join(s["out"], "_comparisons_cas_plus")
    os.makedirs(comp_dir, exist_ok=True)
    if s["save_result"]:
        os.makedirs(os.path.join(s["out"], "cas_plus"), exist_ok=True)

    rows = []
    for name in files:
        img = cv2.imread(os.path.join(s["input"], name))
        if img is None:
            print(f"Could not read: {name}")
            continue

        t = time.time()
        res = cas_plus(img, s)
        secs = time.time() - t
        pil = pillow_unsharp(img, s)

        sheet = np.hstack([labelled(img, "Original"),
                           labelled(res, "cas_plus")])
        cv2.imwrite(os.path.join(comp_dir, os.path.splitext(name)[0] + "_compare.png"), sheet)
        if s["save_result"]:
            cv2.imwrite(os.path.join(s["out"], "cas_plus", os.path.splitext(name)[0] + ".png"), res)

        flat = flat_mask_of(img)
        base_sharp, base_noise, base_clip = sharpness_score(img), flat_noise(img, flat), clip_percent(img)
        row = {"file": name, "width": img.shape[1], "height": img.shape[0], "seconds": round(secs, 3)}
        for tag, out in (("pillow", pil), ("cas_plus", res)):
            row[f"{tag}_sharpness_x"] = round(sharpness_score(out) / base_sharp, 2)
            row[f"{tag}_flat_noise_x"] = round(flat_noise(out, flat) / max(base_noise, 1e-6), 2)
            row[f"{tag}_clip_added_%"] = round(clip_percent(out) - base_clip, 2)
        rows.append(row)
        print(f"Done: {name}")

    if not rows:
        print("No images could be processed.")
        return

    with open(os.path.join(s["out"], "_report_cas.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{len(rows)} image(s). Average vs original:")
    print(f"{'':10}{'sharpness x':>13}{'flat noise x':>14}{'clip added %':>14}")
    for tag in ("pillow", "cas_plus"):
        avg = [np.mean([r[f"{tag}_{k}"] for r in rows]) for k in ("sharpness_x", "flat_noise_x", "clip_added_%")]
        print(f"{tag:10}{avg[0]:13.2f}{avg[1]:14.2f}{avg[2]:14.2f}")
    print(f"\nComparison sheets: {comp_dir}")


if __name__ == "__main__":
    main()
