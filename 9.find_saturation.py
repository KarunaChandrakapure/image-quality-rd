#!/usr/bin/env python3
"""
9.find_saturation_compare.py
Batch‑process a folder of images, compute a saturation score, optionally
fix the image so the score falls in the 3‑4 window, and (optionally) create a
side‑by‑side comparison image that shows the original next to the fixed one.
Usage
-----
    python3 9.find_saturation_compare.py <input_folder> <output_folder> [options]
Options
-------
    -f, --fix          Apply automatic correction to bring the score into 3‑4.
    -c, --compare      Save a side‑by‑side comparison (original | fixed) image.
    -v, --verbose     Print per‑file information (score, factor, etc.).
"""
# --------------------------------------------------------------
# Imports
# --------------------------------------------------------------
import argparse
import pathlib
import sys
from typing import List, Tuple
import cv2
import numpy as np
# --------------------------------------------------------------
# Helper – compute the YCrCb‑based saturation score
# --------------------------------------------------------------
def _ycrcb_saturation_score(ycc: np.ndarray) -> float:
    """Mean absolute distance of Cr/Cb from neutral (128). Returns 0‑127."""
    chroma = ycc[..., 1:].astype(np.int16)   # Cr, Cb channels
    distance = np.abs(chroma - 128)
    return float(distance.mean())
# --------------------------------------------------------------
# Helper – scale chroma (identical to the original routine)
# --------------------------------------------------------------
def _scale_chroma(ycc: np.ndarray, factor: float, *, inplace: bool = False) -> np.ndarray:
    """Multiply (Cr‑128) and (Cb‑128) by *factor* and clamp to 0‑255."""
    if factor < 0 or not np.isfinite(factor):
        raise ValueError("factor must be a non‑negative finite number")
    img = ycc if inplace else ycc.copy()
    chroma = img[..., 1:].astype(np.float32)
    chroma = np.clip((chroma - 128) * factor + 128, 0, 255)
    img[..., 1:] = chroma.astype(np.uint8)
    return img
# --------------------------------------------------------------
# New – compute a factor that drives the score into 3‑4
# --------------------------------------------------------------
def _calc_fix_factor(
    current_score: float,
    *,
    ideal_low: float = 5.0,
    ideal_high: float = 6.0,
    min_factor: float = 0.5,
    max_factor: float = 1.5,
) -> float:
    """
    Rough linear mapping: new_score ≈ current_score * factor.
    Returns a factor that pushes the score toward the centre of the ideal window.
    """
    if ideal_low <= current_score <= ideal_high:
        return 1.0
    target = (ideal_low + ideal_high) / 2.0
    factor_target = target / current_score
    return float(np.clip(factor_target, min_factor, max_factor))
# --------------------------------------------------------------
# Helper – draw the score overlay (used for both fixed and compare)
# --------------------------------------------------------------
def _draw_score_overlay(
    img: np.ndarray,
    score: float,
) -> np.ndarray:
    """Return a copy of *img* with the numeric score drawn in the top‑left."""
    overlay = img.copy()
    txt = f"Score: {score:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    # Choose colour according to the *original* score
    if 3.0 <= score <= 4.0:                # ideal → green
        txt_colour = (0, 255, 0)           # BGR
    elif score < 3.0:                      # flushed → orange/gold
        txt_colour = (0, 165, 255)
    else:                                  # weird → red
        txt_colour = (0, 0, 255)
    # Background rectangle for readability
    (txt_w, txt_h), _ = cv2.getTextSize(txt, font, font_scale, thickness)
    rect_x, rect_y = 10, 10
    cv2.rectangle(
        overlay,
        (rect_x - 5, rect_y - 5),
        (rect_x + txt_w + 5, rect_y + txt_h + 5),
        (0, 0, 0),
        cv2.FILLED,
    )
    cv2.putText(
        overlay,
        txt,
        (rect_x, rect_y + txt_h),
        font,
        font_scale,
        txt_colour,
        thickness,
        cv2.LINE_AA,
    )
    return overlay
# --------------------------------------------------------------
# Core – process a single image
# --------------------------------------------------------------
def _process_one(
    src_path: pathlib.Path,
    dst_path: pathlib.Path,
    *,
    fix: bool = False,
    compare: bool = False,
) -> Tuple[float, float, pathlib.Path, pathlib.Path | None]:
    """
    Returns
    -------
    (orig_score, factor, fixed_path, compare_path_or_None)
    * `factor` is 1.0 when fix=False.
    * `compare_path` is only set when compare=True.
    """
    # ----- read -------------------------------------------------
    orig_bgr = cv2.imread(str(src_path))
    if orig_bgr is None:
        raise IOError(f"Cannot read image {src_path}")
    # ----- compute original saturation score --------------------
    ycc = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2YCrCb)
    orig_score = _ycrcb_saturation_score(ycc)
    # ----- optionally fix ---------------------------------------
    factor = 1.0
    fixed_ycc = ycc
    if fix:
        factor = _calc_fix_factor(orig_score)
        if abs(factor - 1.0) > 0.01:               # apply only if noticeable
            fixed_ycc = _scale_chroma(ycc, factor)
    # ----- convert back to BGR (fixed version) -----------------
    fixed_bgr = cv2.cvtColor(fixed_ycc, cv2.COLOR_YCrCb2BGR)
    # ----- draw overlay on the fixed image ---------------------
    fixed_with_overlay = _draw_score_overlay(fixed_bgr, orig_score)
    # ----- ensure destination directory exists ------------------
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    # ----- write the fixed image --------------------------------
    cv2.imwrite(str(dst_path), fixed_with_overlay)
    # ----- create comparison image if requested -----------------
    compare_path = None
    if compare:
        # Stack original (no overlay) on the left, fixed+overlay on the right
        # Resize to the same height if they differ (should not happen, but be safe)
        if orig_bgr.shape != fixed_with_overlay.shape:
            # Preserve aspect ratio – resize the larger one to the smaller height
            h_min = min(orig_bgr.shape[0], fixed_with_overlay.shape[0])
            orig_bgr = cv2.resize(orig_bgr, (int(orig_bgr.shape[1] * h_min / orig_bgr.shape[0]), h_min))
            fixed_with_overlay = cv2.resize(fixed_with_overlay,
                                            (int(fixed_with_overlay.shape[1] * h_min / fixed_with_overlay.shape[0]), h_min))
        compare_img = np.hstack([orig_bgr, fixed_with_overlay])
        # Preserve original extension for the compare file
        compare_path = dst_path.with_name(dst_path.stem + "_compare" + dst_path.suffix)
        cv2.imwrite(str(compare_path), compare_img)
    return orig_score, factor, dst_path, compare_path

# --------------------------------------------------------------
# Helper – collect all image files recursively
# --------------------------------------------------------------
def _collect_images(root: pathlib.Path) -> List[pathlib.Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]

# --------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch compute saturation scores, optionally fix them, "
                    "and optionally produce side‑by‑side comparison images."
    )
    parser.add_argument(
        "input_folder",
        type=pathlib.Path,
        help="Folder with the source images (recursively scanned).",
    )
    parser.add_argument(
        "output_folder",
        type=pathlib.Path,
        help="Folder where the processed images (and optional compares) are saved.",
    )
    parser.add_argument(
        "-f",
        "--fix",
        action="store_true",
        help="Apply automatic colour correction to move the score into the 3‑4 window.",
    )
    parser.add_argument(
        "-c",
        "--compare",
        action="store_true",
        help="Create a side‑by‑side (original | fixed) comparison image for each file.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print per‑file details (score, factor, etc.).",
    )
    args = parser.parse_args()
    # ----- sanity checks ----------------------------------------
    if not args.input_folder.is_dir():
        sys.exit(f"❌ Input path {args.input_folder} is not a directory.")
    images = _collect_images(args.input_folder)
    if not images:
        sys.exit(f"❌ No supported image files found under {args.input_folder}")
    print(f"🔎 Found {len(images)} image(s). Processing…")
    print(f"   • Fix enabled   : {args.fix}")
    print(f"   • Compare mode  : {args.compare}")
    # ----- process each image ------------------------------------
    for src_path in images:
        rel_path = src_path.relative_to(args.input_folder)
        dst_path = args.output_folder / rel_path  # where the fixed image goes
        try:
            orig_score, factor, fixed_path, compare_path = _process_one(
                src_path,
                dst_path,
                fix=args.fix,
                compare=args.compare,
            )
            if args.verbose:
                msg = f"{src_path} → score={orig_score:.2f}"
                if args.fix:
                    msg += f", factor={factor:.3f}"
                msg += f" → {fixed_path}"
                if compare_path:
                    msg += f", compare={compare_path}"
                print(msg)
        except Exception as exc:
            print(f"⚠️  Failed on {src_path}: {exc}", file=sys.stderr)
    print(f"\n✅ Finished. Results are in: {args.output_folder}")
    if args.compare:
        print("   Comparison images have the suffix '_compare' before the extension.")

if __name__ == "__main__":
    main()