#!/usr/bin/env python3
"""
measure_saturation_hsv.py
-------------------------
Batch‑process a directory of images and compute an **average saturation**
value using the HSV colour space (the *S* channel).  The script can:
* write a CSV report:   <relative_path>, <average_saturation (0‑255)>
* optionally save a copy of the image with the numeric value over‑laid
* optionally ignore a typical clear‑sky region (useful for sunny‑day photos)
Usage
-----
    python3 measure_saturation_hsv.py <input_folder> <output_folder> [options]
Options
-------
    -o, --output-csv  PATH   Path of the CSV report (default: report.csv)
    -v, --verbose            Print per‑file stats to the console.
    -a, --annotate           Save an annotated copy of each image (original
                             image + overlayed score) in the output folder.
    -s, --skip-sky           Mask out a clear‑sky region before measuring
                             saturation (HSV‑based sky mask, can be tuned).
"""
# --------------------------------------------------------------
# Imports
# --------------------------------------------------------------
import argparse
import csv
import pathlib
import sys
from typing import List, Tuple
import cv2
import numpy as np
# --------------------------------------------------------------
# Helper – collect image paths recursively
# --------------------------------------------------------------
def _collect_images(root: pathlib.Path) -> List[pathlib.Path]:
    """Return a list of all image files under *root* (case‑insensitive)."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]
# --------------------------------------------------------------
# Helper – simple sky mask (HSV based)
# --------------------------------------------------------------
def _sky_mask(bgr: np.ndarray) -> np.ndarray:
    """
    Return a binary mask where 1 = sky, 0 = non‑sky.
    The thresholds work well for a clear blue sky on a sunny day.
    Adjust `lower` / `upper` if your images differ.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Hue for blue ≈ 90‑130, fairly saturated, relatively bright
    lower = np.array([90, 30, 80], dtype=np.uint8)
    upper = np.array([130, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)            # 255 = sky
    # Clean small holes / speckles
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return (mask > 0).astype(np.uint8)               # 1 / 0
# --------------------------------------------------------------
# Core – compute average saturation (0‑255) for a single image
# --------------------------------------------------------------
def _average_saturation(
    bgr: np.ndarray,
    *,
    ignore_sky: bool = False,
) -> float:
    """
    Convert *bgr* to HSV and return the mean of the S channel.
    If *ignore_sky* is True, pixels identified by `_sky_mask` are excluded
    from the average.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s_channel = hsv[..., 1].astype(np.float32)      # 0‑255
    if ignore_sky:
        sky = _sky_mask(bgr)                         # 1 = sky
        # Keep only non‑sky pixels
        non_sky = 1 - sky
        if non_sky.sum() == 0:                       # whole image is sky
            return 0.0
        return float((s_channel * non_sky).sum() / non_sky.sum())
    else:
        return float(s_channel.mean())
# --------------------------------------------------------------
# Helper – draw numeric overlay on an image (BGR)
# --------------------------------------------------------------
def _draw_overlay(bgr: np.ndarray, score: float) -> np.ndarray:
    """Return a copy of *bgr* with the numeric saturation score drawn top‑left."""
    overlay = bgr.copy()
    txt = f"Saturation: {score:.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    # Choose colour based on magnitude (just for visual cue)
    if score < 85:       # low   → orange
        colour = (0, 165, 255)
    elif score < 170:    # medium → yellow
        colour = (0, 255, 255)
    else:                # high   → green
        colour = (0, 255, 0)
    # Background rectangle for readability
    (w, h), _ = cv2.getTextSize(txt, font, scale, thickness)
    cv2.rectangle(overlay, (8, 8), (8 + w + 4, 8 + h + 4), (0, 0, 0), cv2.FILLED)
    cv2.putText(
        overlay,
        txt,
        (12, 12 + h),
        font,
        scale,
        colour,
        thickness,
        cv2.LINE_AA,
    )
    return overlay
# --------------------------------------------------------------
# Process one image (read → score → optional annotate → write)
# --------------------------------------------------------------
def _process_one(
    src_path: pathlib.Path,
    dst_root: pathlib.Path,
    *,
    ignore_sky: bool,
    annotate: bool,
) -> Tuple[float, pathlib.Path]:
    """
    Returns (average_saturation, destination_path_of_annotated_image|None).
    The annotated image is written under *dst_root* preserving the original
    directory layout.
    """
    img = cv2.imread(str(src_path))
    if img is None:
        raise IOError(f"Unable to read image {src_path}")
    sat = _average_saturation(img, ignore_sky=ignore_sky)
    annotated_path = None
    if annotate:
        annotated = _draw_overlay(img, sat)
        # Mirror the input folder structure inside *dst_root*
        rel = src_path.relative_to(src_path.anchor)   # relative to filesystem root
        annotated_path = dst_root / rel
        annotated_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(annotated_path), annotated)
    return sat, annotated_path
# --------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute average HSV‑S saturation for every image under a folder. "
            "Results are saved to a CSV file and optionally as annotated copies."
        )
    )
    parser.add_argument(
        "input_folder",
        type=pathlib.Path,
        help="Folder containing source images (recursively scanned).",
    )
    parser.add_argument(
        "output_folder",
        type=pathlib.Path,
        help="Folder where optional annotated images will be stored.",
    )
    parser.add_argument(
        "-o",
        "--output-csv",
        type=pathlib.Path,
        default=pathlib.Path("report.csv"),
        help="Path of the CSV report (default: report.csv in cwd).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Print per‑file stats.")
    parser.add_argument(
        "-a",
        "--annotate",
        action="store_true",
        help="Save a copy of each image with the saturation value over‑laid.",
    )
    parser.add_argument(
        "-s",
        "--skip-sky",
        action="store_true",
        help="Mask out clear‑sky pixels before measuring saturation.",
    )
    args = parser.parse_args()
    # ------------------------------------------------------------------
    # Basic sanity checks
    # ------------------------------------------------------------------
    if not args.input_folder.is_dir():
        sys.exit(f"❌ Input path {args.input_folder} is not a directory.")
    images = _collect_images(args.input_folder)
    if not images:
        sys.exit(f"❌ No supported image files found under {args.input_folder}")
    # ------------------------------------------------------------------
    # Prepare CSV writer
    # ------------------------------------------------------------------
    csv_path = args.output_csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["relative_path", "average_saturation_0_255"])
    # ------------------------------------------------------------------
    # Process each image
    # ------------------------------------------------------------------
    for src in images:
        rel_path = src.relative_to(args.input_folder)
        try:
            sat, ann_path = _process_one(
                src,
                args.output_folder,
                ignore_sky=args.skip_sky,
                annotate=args.annotate,
            )
            csv_writer.writerow([str(rel_path), f"{sat:.2f}"])
            if args.verbose:
                msg = f"{rel_path} → {sat:.2f}"
                if ann_path:
                    msg += f" (annotated saved to {ann_path})"
                print(msg)
        except Exception as exc:
            print(f"⚠️  Failed on {src}: {exc}", file=sys.stderr)
    csv_file.close()
    print("\n✅ Done.")
    print(f"   CSV report      : {csv_path}")
    if args.annotate:
        print(f"   Annotated images: {args.output_folder}")
if __name__ == "__main__":
    main()
