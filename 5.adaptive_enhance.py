#!/usr/bin/env python3
"""
adaptive_enhance.py - image-specific enhancement for model input.

Per image:
    1. resize    to the model input size (configurable: size, stretch / pad / crop)
    2. diagnose  measure brightness, clipped highlights, noise and blur and decide
                 which issues it has: dark | bright | noisy | blurry | very blurry | ok
    3. fix       apply the fixes for the issues found, plus (fix.enhance_all) the same
                 base treatment for every frame, in a fixed safe order:
                 denoise -> brighten / highlights -> levels -> local contrast
                 -> sharpen -> colour

Usage:
    python 5.adaptive_enhance.py
    python 5.adaptive_enhance.py --config adaptive_config.yaml
    python 5.adaptive_enhance.py --input my_folder --limit 20
    python 5.adaptive_enhance.py --diagnose-only          # just write the report

Install:
    pip install opencv-python numpy pyyaml
"""

import argparse
import csv
import importlib.util
import os
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent

# Edge-gated CAS sharpening lives in 4.cas_sharpen.py (file name starts with a digit,
# so it is loaded by path instead of a normal import).
_spec = importlib.util.spec_from_file_location("cas_sharpen", HERE / "4.cas_sharpen.py")
cas_sharpen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cas_sharpen)

INTERP = {
    "area": cv2.INTER_AREA,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}
LABEL_HEIGHT = 40


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ----------------------------------------------------------------------
# Step 1: resize
# ----------------------------------------------------------------------
def resize(img, c):
    if not c["enabled"]:
        return img
    W, H = c["width"], c["height"]
    interp = INTERP[c["interpolation"]]
    h, w = img.shape[:2]
    if c["mode"] == "stretch":
        return cv2.resize(img, (W, H), interpolation=interp)

    if c["mode"] == "pad":
        scale = min(W / w, H / h)
        nw, nh = round(w * scale), round(h * scale)
        small = cv2.resize(img, (nw, nh), interpolation=interp)
        out = np.full((H, W, 3), c["pad_value"], np.uint8)
        x, y = (W - nw) // 2, (H - nh) // 2
        out[y:y + nh, x:x + nw] = small
        return out

    if c["mode"] == "crop":
        scale = max(W / w, H / h)
        nw, nh = round(w * scale), round(h * scale)
        big = cv2.resize(img, (nw, nh), interpolation=interp)
        x, y = (nw - W) // 2, (nh - H) // 2
        return big[y:y + H, x:x + W]

    raise ValueError(f"resize.mode must be stretch, pad or crop, not {c['mode']!r}")


# ----------------------------------------------------------------------
# Step 2: diagnose
# ----------------------------------------------------------------------
def noise_sigma(y, block=8, percentile=10):
    """Sensor-noise level. Immerkaer's fast noise estimator (1996), measured per 8x8
    block and taken from the flattest 10% of blocks (sky, road, walls), so busy
    texture like foliage is not mistaken for noise.  y: float32 luma 0-255."""
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    r = np.abs(cv2.filter2D(y, -1, k)) * np.sqrt(0.5 * np.pi) / 6.0
    h, w = (y.shape[0] // block) * block, (y.shape[1] // block) * block
    blocks = r[:h, :w].reshape(h // block, block, w // block, block).mean(axis=(1, 3))
    return float(np.percentile(blocks, percentile))


def blur_score(y):
    """Re-blur metric (Crete et al. 2007): 0 = sharp, 1 = very blurry.
    Blurs the image again and checks how much edge contrast that removes: a sharp
    image loses a lot, an already-blurry one hardly changes. Unlike Laplacian
    variance it does not depend much on how busy the scene is."""
    bv, bh = cv2.blur(y, (1, 9)), cv2.blur(y, (9, 1))
    dy, dx = np.abs(np.diff(y, axis=0)), np.abs(np.diff(y, axis=1))
    vy = np.maximum(0, dy - np.abs(np.diff(bv, axis=0)))
    vx = np.maximum(0, dx - np.abs(np.diff(bh, axis=1)))
    sy, sx = max(dy.sum(), 1e-6), max(dx.sum(), 1e-6)
    return float(max((sy - vy.sum()) / sy, (sx - vx.sum()) / sx))


def diagnose(img, c):
    y = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[..., 0].astype(np.float32)
    m = {
        "brightness": float(y.mean()),
        "clipped_white_pct": float((y >= 250).mean() * 100),
        "noise_sigma": noise_sigma(y),
        "blur_score": blur_score(y),
    }
    issues = []
    if m["brightness"] < c["dark_mean"]:
        issues.append("dark")
    elif m["brightness"] > c["bright_mean"] or m["clipped_white_pct"] > c["clipped_white_pct"]:
        issues.append("bright")
    if m["noise_sigma"] > c["noise_sigma"]:
        issues.append("noisy")
    if m["blur_score"] > c["strong_blur_score"]:
        issues.append("very blurry")
    elif m["blur_score"] > c["blur_score"]:
        issues.append("blurry")
    return issues or ["ok"], m


# ----------------------------------------------------------------------
# Step 3: fixes  (all on luma Y in [0,1] unless stated)
# ----------------------------------------------------------------------
def fix_denoise(img, sigma, c):
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    h = float(np.clip(c["h_per_sigma"] * sigma, c["h_min"], c["h_max"]))
    ycc[..., 0] = cv2.fastNlMeansDenoising(ycc[..., 0], None, h, 7, 21)
    if c["color_h"] > 0:
        for ch in (1, 2):
            ycc[..., ch] = cv2.fastNlMeansDenoising(ycc[..., ch], None, c["color_h"], 7, 21)
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR), f"denoise(h={h:.1f})"


# The luma functions below take and return float32 Y in [0, 1].

def shadow_lift(y, c):
    """Local tone mapping for dark frames: the brightening gain follows the smoothed
    local brightness, so dark regions are lifted while street lamps, headlights and
    the black sky stay put (a global gamma turns all of that into grey fog)."""
    base = cv2.GaussianBlur(y, (0, 0), c["radius"])
    mean = float(np.clip(y.mean(), 1e-3, 0.99))
    g = float(np.clip(np.log(c["target_mean"]) / np.log(mean), c["gamma_min"], 1.0))
    gain = np.clip(np.power(np.clip(base, 1e-3, 1), g - 1), 1.0, c["max_gain"])
    return np.clip(y * gain, 0, 1).astype(np.float32), float(gain.mean())


def highlight_rolloff(y, c):
    """For bright frames: compress tones above the knee to recover near-white detail."""
    k, s = c["highlight_knee"], c["highlight_strength"]
    over = np.clip(y - k, 0, None)
    return (y - s * over * over / max(1 - k, 1e-6) * 0.5).astype(np.float32)


def auto_levels(y, c):
    """Stretch so the darkest / brightest clip_low / clip_high % reach black / white.
    Removes the flat, hazy look. max_gain stops it from blowing up noise in
    frames that are genuinely low-contrast (fog, rain)."""
    lo, hi = np.percentile(y, c["clip_low"]), np.percentile(y, 100 - c["clip_high"])
    gain = min(1.0 / max(hi - lo, 1e-3), c["max_gain"])
    out = np.clip((y - lo) * gain, 0, 1)
    return ((1 - c["strength"]) * y + c["strength"] * out).astype(np.float32)


def local_contrast(y, c):
    """CLAHE, blended in: makes objects stand out from similar-toned surroundings."""
    clahe = cv2.createCLAHE(clipLimit=c["clip"], tileGridSize=(c["tile"],) * 2)
    eq = clahe.apply((y * 255 + 0.5).astype(np.uint8)).astype(np.float32) / 255.0
    return ((1 - c["strength"]) * y + c["strength"] * eq).astype(np.float32)


def sharpen(y, sharpness, c):
    """Edge-gated CAS from 4.cas_sharpen.py: object outlines only, no texture/noise."""
    sharp = cas_sharpen.cas(y, sharpness)
    return (y + cas_sharpen.structure_mask(y, c) * (sharp - y)).astype(np.float32)


def scale_chroma(ycc, factor):
    """Saturation in YCrCb: scale the colour channels around neutral grey (128)."""
    for ch in (1, 2):
        ycc[..., ch] = np.clip((ycc[..., ch].astype(np.float32) - 128) * factor + 128, 0, 255).astype(np.uint8)
    return ycc


def enhance(img, issues, m, c):
    """Fix the diagnosed issues, then (if c["enhance_all"]) give every frame the same
    base treatment: auto levels -> local contrast -> sharpen -> colour.
    Order: denoise -> exposure -> contrast -> sharpen -> colour, so no step amplifies
    an artifact that a later step can't undo. Returns (image, list of actions)."""
    actions = []
    if "noisy" in issues or "dark" in issues:      # brightening a dark frame lifts its noise
        img, a = fix_denoise(img, m["noise_sigma"], c["denoise"])
        actions.append(a)

    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y = ycc[..., 0].astype(np.float32) / 255.0
    saturation = 1.0
    ok_frame = issues == ["ok"]

    if "dark" in issues:
        y, gain = shadow_lift(y, c["brighten"])
        # lifting luma alone leaves colours washed out; give part of the gain back to chroma
        saturation = 1.0 + c["brighten"]["color_restore"] * (gain - 1.0)
        actions.append(f"brighten(x{gain:.2f})")
    elif "bright" in issues:
        y = highlight_rolloff(y, c["darken"])
        actions.append("highlights")

    if c["enhance_all"] or not ok_frame:
        if c["levels"]["enabled"]:
            y = auto_levels(y, c["levels"])
            actions.append("levels")
        if c["local_contrast"]["enabled"]:
            y = local_contrast(y, c["local_contrast"])
            actions.append("contrast")

    s = c["sharpen"]
    amount = (s["strong_sharpness"] if "very blurry" in issues else
              s["sharpness"] if "blurry" in issues else
              s["ok_sharpness"] if c["enhance_all"] else 0)
    if amount > 0:
        y = sharpen(y, amount, s)
        actions.append(f"sharpen({amount:.1f})")

    if c["enhance_all"] or not ok_frame:
        saturation *= c["color"]["saturation"]
    saturation = min(saturation, c["color"]["max_saturation"])
    ycc[..., 0] = (np.clip(y, 0, 1) * 255 + 0.5).astype(np.uint8)
    if abs(saturation - 1.0) > 0.01:
        ycc = scale_chroma(ycc, saturation)
        actions.append(f"color(x{saturation:.2f})")
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR), actions or ["none"]


# ----------------------------------------------------------------------
# Output helpers
# ----------------------------------------------------------------------
def labelled(img, title):
    """Copy of img with the title drawn in its top-left corner on a dark strip.
    The image size is not changed."""
    out = img.copy()
    fs = 0.5
    while fs > 0.3 and cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0][0] > img.shape[1] - 12:
        fs -= 0.05
    (tw, th), base = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
    cv2.rectangle(out, (0, 0), (tw + 10, th + base + 8), (30, 30, 30), -1)
    cv2.putText(out, title, (5, th + 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def save_image(path, img, g):
    fmt = g["output_format"]
    if fmt != "same":
        path = path.with_suffix("." + fmt)
    if path.suffix.lower() in (".jpg", ".jpeg"):
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, g["jpg_quality"]])
    else:
        cv2.imwrite(str(path), img)


def list_images(g):
    root = Path(g["input_dir"])
    pattern = "**/*" if g["recursive"] else "*"
    exts = tuple(e.lower() for e in g["extensions"])
    files = sorted(p for p in root.glob(pattern) if p.is_file() and p.suffix.lower() in exts)
    return files[:g["limit"]] if g["limit"] else files


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "adaptive_config.yaml"))
    ap.add_argument("--input", help="overrides general.input_dir")
    ap.add_argument("--out", help="overrides general.output_dir")
    ap.add_argument("--limit", type=int, help="overrides general.limit")
    ap.add_argument("--diagnose-only", action="store_true", help="only resize + diagnose, write the report")
    args = ap.parse_args()

    cfg = load_config(args.config)
    g = cfg["general"]
    if args.input:
        g["input_dir"] = args.input
    if args.out:
        g["output_dir"] = args.out
    if args.limit is not None:
        g["limit"] = args.limit

    files = list_images(g)
    if not files:
        print(f"No images found in {g['input_dir']}")
        return

    # Each run writes into a subfolder named after the input folder, e.g. output/sample2
    out_dir = Path(g["output_dir"]) / Path(g["input_dir"]).resolve().name
    out_dir.mkdir(parents=True, exist_ok=True)
    if not (g["write_original"] or g["write_enhanced"]):
        print("Note: write_original and write_enhanced are both false - frames that were "
              "enhanced will still be written as the original only.")

    rows = []
    for path in files:
        img = cv2.imread(str(path))
        if img is None:
            print(f"Could not read: {path.name}")
            continue
        t = time.time()

        small = resize(img, cfg["resize"])                                # step 1
        issues, m = diagnose(small, cfg["diagnose"])                       # step 2
        if args.diagnose_only:
            result, actions = small, ["(diagnose only)"]
        else:
            result, actions = enhance(small, issues, m, cfg["fix"])        # step 3

        rel = path.relative_to(g["input_dir"])
        if not args.diagnose_only:
            # One output file per input, same name. Which images go in it is set by
            # write_original / write_enhanced; both -> joined per "layout". A frame that
            # needed no enhancement has no enhanced image, so only the original is written.
            enhanced = actions != ["none"]
            left, right = small, result
            if g["labels"]:
                left = labelled(small, "Original: " + ", ".join(issues))
                right = labelled(result, "Enhanced: " + " + ".join(actions))
            parts = []
            if g["write_original"] or not enhanced:
                parts.append(left)
            if g["write_enhanced"] and enhanced:
                parts.append(right)
            stack = np.vstack if g["layout"] == "stacked" else np.hstack
            (out_dir / rel.parent).mkdir(parents=True, exist_ok=True)
            save_image(out_dir / rel, stack(parts), g)

        rows.append({"file": str(rel), "width": small.shape[1], "height": small.shape[0],
                     **{k: round(v, 3) for k, v in m.items()},
                     "issues": ";".join(issues), "actions": ";".join(actions),
                     "seconds": round(time.time() - t, 3)})
        print(f"{rel}: {', '.join(issues)} -> {', '.join(actions)}")

    if g["save_report"] and rows:
        with open(out_dir / "_report.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    counts = {}
    for r in rows:
        for i in r["issues"].split(";"):
            counts[i] = counts.get(i, 0) + 1
    print(f"\n{len(rows)} image(s). Issues found: " +
          ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])))
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
