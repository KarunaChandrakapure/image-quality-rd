#!/usr/bin/env python3
"""
calc_scores_folder.py
Batch‑process all images inside a folder, compute a saturation score for each
image, overlay the score on a copy of the image and save the copy to an
output folder that preserves the original hierarchy.
Usage
-----
    python calc_scores_folder.py <input_folder> <output_folder>
    # optional: add a numeric threshold to colour‑code the text
    python calc_scores_folder.py <input_folder> <output_folder> -t 7.5 -v
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
# Helper – compute the saturation score (identical to the one used
# in the adaptive‑saturation function)
# --------------------------------------------------------------
def _ycrcb_saturation_score(ycc: np.ndarray) -> float:
    """
    Mean absolute distance of the Cr/Cb channels from neutral grey (128).
    Returns
    -------
    float
        Value in the range 0 … 127  (0 = completely gray, 127 = max chroma)
    """
    chroma = ycc[..., 1:].astype(np.int16)   # Cr, Cb channels
    distance = np.abs(chroma - 128)
    return float(distance.mean())
# --------------------------------------------------------------
# Core: process a single image → draw score → write result
# --------------------------------------------------------------
def _process_one(
    src_path: pathlib.Path,
    dst_path: pathlib.Path,
    *,
    threshold: float | None = None,
) -> Tuple[float, pathlib.Path]:
    """
    Reads src_path, computes its saturation score, writes the score on a copy
    and saves the copy to dst_path.
    Returns
    -------
    (score, dst_path)
    """
    # ----- read -------------------------------------------------
    img = cv2.imread(str(src_path))
    if img is None:
        raise IOError(f"Cannot read image {src_path}")
    # ----- compute score ----------------------------------------
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    score = _ycrcb_saturation_score(ycc)
    # ----- overlay score ----------------------------------------
    overlay = img.copy()
    txt = f"Score: {score:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    # colour based on optional threshold (green ≤ thr, red > thr)
    txt_colour = (255, 255, 255)  # default white
    if threshold is not None:
        txt_colour = (0, 255, 0) if score <= threshold else (0, 0, 255)
    # background rectangle for readability
    (txt_w, txt_h), _ = cv2.getTextSize(txt, font, font_scale, thickness)
    rect_x, rect_y = 10, 10
    cv2.rectangle(
        overlay,
        (rect_x - 5, rect_y - 5),
        (rect_x + txt_w + 5, rect_y + txt_h + 5),
        (0, 0, 0),
        cv2.FILLED,
    )
    # write the text
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
    # ----- make sure destination directory exists --------------
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    # ----- write ------------------------------------------------
    cv2.imwrite(str(dst_path), overlay)
    return score, dst_path

# --------------------------------------------------------------
# Helper – collect all image files under a root folder
# --------------------------------------------------------------
def _collect_images(root: pathlib.Path) -> List[pathlib.Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]

# --------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute a saturation score for every image in a folder, "
                    "draw the score on a copy, and save to an output folder."
    )
    parser.add_argument(
        "input_folder",
        type=pathlib.Path,
        help="Folder containing the source images (processed recursively).",
    )
    parser.add_argument(
        "output_folder",
        type=pathlib.Path,
        help="Folder where the annotated copies will be written.",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=None,
        help="Optional numeric threshold – scores > threshold are shown in red, "
             "≤ threshold in green.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print <src> → <score> for each processed image.",
    )
    args = parser.parse_args()
    # ----- validate input folder ---------------------------------
    if not args.input_folder.is_dir():
        sys.exit(f"❌ Input path {args.input_folder} is not a directory.")
    # ----- gather image files ------------------------------------
    images = _collect_images(args.input_folder)
    if not images:
        sys.exit(f"❌ No image files found under {args.input_folder}")
    print(f"🔎 Found {len(images)} image(s). Processing…")
    # ----- process each image ------------------------------------
    for src_path in images:
        # Preserve the relative path inside the output folder
        rel_path = src_path.relative_to(args.input_folder)
        dst_path = args.output_folder / rel_path
        try:
            score, out_path = _process_one(
                src_path,
                dst_path,
                threshold=args.threshold,
            )
            if args.verbose:
                print(f"{src_path} → {score:.2f} → {out_path}")
        except Exception as exc:
            print(f"⚠️  Failed on {src_path}: {exc}", file=sys.stderr)
    print(f"\n✅ Finished. Annotated images saved in: {args.output_folder}")

if __name__ == "__main__":
    main()
