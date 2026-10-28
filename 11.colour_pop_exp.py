#!/usr/bin/env python3
"""
colour_pop_balancer.py
-----------------------
Batch‑process a folder of images and automatically bring every picture
into a pre‑defined “sweet‑spot” of saturation and brightness.
Features
--------
* Adaptive gain on the HSV S channel (and optional V channel).
* Optional CLAHE for local contrast boost.
* Side‑by‑side before | after visual comparison.
* CSV report with original / adjusted statistics.
* Keeps original folder hierarchy.
Usage
-----
    python3 colour_pop_balancer.py <input_dir> <output_dir> [options]
Options
-------
    -s, --sat-range  LOW HIGH   Target saturation range (default: 80 130)
    -v, --val-range  LOW HIGH   Target brightness range (default: 100 180)
    -c, --clahe                   Apply CLAHE on the V channel (recommended).
    -p, --compare                 Save before|after comparison images.
    -r, --report <path>           CSV file for the report (default: report.csv)
    -V, --verbose                 Print per‑file statistics.
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
# Constants (you can tweak them as defaults)
# --------------------------------------------------------------
DEFAULT_SAT_RANGE = (80, 130)   # low, high (0‑255)
DEFAULT_VAL_RANGE = (100, 180)  # low, high (0‑255)
# --------------------------------------------------------------
# Utility – collect image files recursively
# --------------------------------------------------------------
def collect_images(root: pathlib.Path) -> List[pathlib.Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]
# --------------------------------------------------------------
# Core – compute mean (and robust percentiles) for S and V
# --------------------------------------------------------------
def channel_stats(channel: np.ndarray) -> Tuple[float, float, float]:
    """Return (mean, 5th‑pct, 95th‑pct) for a single‑channel image."""
    flat = channel.ravel()
    mean = float(flat.mean())
    p5 = float(np.percentile(flat, 5))
    p95 = float(np.percentile(flat, 95))
    return mean, p5, p95
# --------------------------------------------------------------
# Core – compute gain for a channel given target low/high bounds
# --------------------------------------------------------------
def compute_gain(current_mean: float, low_target: float, high_target: float) -> float:
    """
    Return a multiplicative factor that pushes `current_mean` into the
    interval [low_target, high_target].  The factor is clipped to a safe
    range (0.3 … 3.0) to avoid extreme clipping.
    """
    if current_mean < low_target:                      # too low → boost
        gain = low_target / max(current_mean, 1e-6)
    elif current_mean > high_target:                   # too high → reduce
        gain = high_target / current_mean
    else:                                              # already inside window
        gain = 1.0
    # safety clamp – you can edit these limits
    return np.clip(gain, 0.3, 3.0)
# --------------------------------------------------------------
# Optional – CLAHE on the V channel (local contrast)
# --------------------------------------------------------------
def apply_clahe(v_channel: np.ndarray) -> np.ndarray:
    """CLAHE (Contrast Limited Adaptive Histogram Equalisation) on V."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(v_channel)
# --------------------------------------------------------------
# Process a single image
# --------------------------------------------------------------
def process_one(
    src_path: pathlib.Path,
    dst_root: pathlib.Path,
    *,
    sat_range: Tuple[int, int],
    val_range: Tuple[int, int],
    use_clahe: bool,
    make_compare: bool,
) -> Tuple[dict, pathlib.Path, pathlib.Path | None]:
    """
    Returns:
        - stats dict (original & adjusted means / gains)
        - path of the adjusted image
        - path of the side‑by‑side comparison (or None)
    """
    # ---- read -------------------------------------------------
    bgr = cv2.imread(str(src_path))
    if bgr is None:
        raise IOError(f"Cannot read {src_path}")
    # ---- convert to HSV ---------------------------------------
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    s_chan = hsv[..., 1]
    v_chan = hsv[..., 2]
    # ---- original statistics ----------------------------------
    s_mean, s_p5, s_p95 = channel_stats(s_chan)
    v_mean, v_p5, v_p95 = channel_stats(v_chan)
    # ---- compute gains -----------------------------------------
    sat_gain = compute_gain(s_mean, sat_range[0], sat_range[1])
    val_gain = compute_gain(v_mean, val_range[0], val_range[1])
    # ---- apply gains -------------------------------------------
    s_adj = np.clip(s_chan * sat_gain, 0, 255)
    v_adj = np.clip(v_chan * val_gain, 0, 255)
    # optional CLAHE (after gain, because CLAHE expects 8‑bit)
    if use_clahe:
        v_adj = apply_clahe(v_adj.astype(np.uint8)).astype(np.float32)
    # ---- re‑assemble HSV and go back to BGR --------------------
    hsv_adj = hsv.copy()
    hsv_adj[..., 1] = s_adj
    hsv_adj[..., 2] = v_adj
    bgr_adj = cv2.cvtColor(hsv_adj.astype(np.uint8), cv2.COLOR_HSV2BGR)
    # ---- adjusted statistics -----------------------------------
    s_adj_mean, *_ = channel_stats(s_adj)
    v_adj_mean, *_ = channel_stats(v_adj)
    # ---- write adjusted image (preserve layout) ---------------
    rel = src_path.relative_to(src_path.anchor)        # keep hierarchy
    dst_path = dst_root / rel
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_path), bgr_adj)
    # ---- optional side‑by‑side comparison ----------------------
    compare_path = None
    if make_compare:
        # Resize to the same height if anything changed (unlikely)
        if bgr.shape != bgr_adj.shape:
            h = min(bgr.shape[0], bgr_adj.shape[0])
            bgr = cv2.resize(bgr, (int(bgr.shape[1] * h / bgr.shape[0]), h))
            bgr_adj = cv2.resize(bgr_adj, (int(bgr_adj.shape[1] * h / bgr_adj.shape[0]), h))
        comp = np.hstack([bgr, bgr_adj])
        compare_path = dst_path.with_name(dst_path.stem + "_compare" + dst_path.suffix)
        cv2.imwrite(str(compare_path), comp)
    # ---- collect stats for CSV --------------------------------
    stats = {
        "rel_path": str(rel),
        "s_mean_orig": f"{s_mean:.2f}",
        "s_gain": f"{sat_gain:.3f}",
        "s_mean_adj": f"{s_adj_mean:.2f}",
        "v_mean_orig": f"{v_mean:.2f}",
        "v_gain": f"{val_gain:.3f}",
        "v_mean_adj": f"{v_adj_mean:.2f}",
        "clahe": use_clahe,
    }
    return stats, dst_path, compare_path
# --------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto‑balance colour saturation & brightness across a mixed image set."
    )
    parser.add_argument("input_dir", type=pathlib.Path, help="Root folder with source images.")
    parser.add_argument("output_dir", type=pathlib.Path, help="Folder where adjusted images will be saved.")
    parser.add_argument("-s", "--sat-range", nargs=2, type=int, default=DEFAULT_SAT_RANGE,
                        metavar=("LOW", "HIGH"),
                        help="Target saturation range (0‑255).")
    parser.add_argument("-v", "--val-range", nargs=2, type=int, default=DEFAULT_VAL_RANGE,
                        metavar=("LOW", "HIGH"),
                        help="Target brightness (V) range (0‑255).")
    parser.add_argument("-c", "--clahe", action="store_true", help="Apply CLAHE on the V channel.")
    parser.add_argument("-p", "--compare", action="store_true", help="Save before|after comparison images.")
    parser.add_argument("-r", "--report", type=pathlib.Path, default=pathlib.Path("report.csv"),
                        help="Path to CSV report.")
    parser.add_argument("-V", "--verbose", action="store_true", help="Print per‑file statistics.")
    args = parser.parse_args()
    # ----------------------------------------------------------
    # Basic validation
    # ----------------------------------------------------------
    if not args.input_dir.is_dir():
        sys.exit(f"❌ Input folder {args.input_dir} does not exist or is not a directory.")
    images = collect_images(args.input_dir)
    if not images:
        sys.exit(f"❌ No supported image files found under {args.input_dir}")
    # ----------------------------------------------------------
    # Prepare CSV
    # ----------------------------------------------------------
    args.report.parent.mkdir(parents=True, exist_ok=True)
    csv_fp = open(args.report, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_fp)
    csv_writer.writerow([
        "relative_path",
        "sat_mean_original",
        "sat_gain",
        "sat_mean_adjusted",
        "val_mean_original",
        "val_gain",
        "val_mean_adjusted",
        "clahe_applied",
        "adjusted_path",
        "compare_path"
    ])
    # ----------------------------------------------------------
    # Process loop
    # ----------------------------------------------------------
    for src in images:
        try:
            stats, adj_path, cmp_path = process_one(
                src,
                args.output_dir,
                sat_range=tuple(args.sat_range),
                val_range=tuple(args.val_range),
                use_clahe=args.clahe,
                make_compare=args.compare,
            )
            csv_writer.writerow([
                stats["rel_path"],
                stats["s_mean_orig"],
                stats["s_gain"],
                stats["s_mean_adj"],
                stats["v_mean_orig"],
                stats["v_gain"],
                stats["v_mean_adj"],
                stats["clahe"],
                str(adj_path.relative_to(args.output_dir)),
                str(cmp_path.relative_to(args.output_dir)) if cmp_path else "",
            ])
            if args.verbose:
                print(
                    f"{stats['rel_path']}: S {stats['s_mean_orig']}→{stats['s_mean_adj']} (gain {stats['s_gain']}), "
                    f"V {stats['v_mean_orig']}→{stats['v_mean_adj']} (gain {stats['v_gain']})"
                )
        except Exception as exc:
            print(f"⚠️  Failed on {src}: {exc}", file=sys.stderr)
    csv_fp.close()
    print("\n✅ Processing complete.")
    print(f"   CSV report   : {args.report}")
    if args.compare:
        print("   Comparison images have the suffix '_compare' before the extension.")
if __name__ == "__main__":
    main()
