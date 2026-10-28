#!/usr/bin/env python3
"""
make_video_side_by_side.py
Create a side‑by‑side video from images that reside in several
sub‑folders (e.g. 1/, 2/, 3/, 4/).  Each sub‑folder contributes one column
to the final frame.
Usage
-----
    python3 make_video_side_by_side.py <root_folder> [options]
Example
-------
    python3 make_video_side_by_side.py ./fixed_saturation -v
"""
# --------------------------------------------------------------
# Imports
# --------------------------------------------------------------
import argparse
import pathlib
import re
import sys
from typing import List, Tuple, Dict
import cv2
import numpy as np
# --------------------------------------------------------------
# Helper – robust natural sort of file names
# --------------------------------------------------------------
def _collect_image_paths(folder: pathlib.Path) -> List[pathlib.Path]:
    """Return a naturally sorted list of image files inside *folder*."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    files = [p for p in folder.iterdir()
             if p.is_file() and p.suffix.lower() in exts]
    numeric_re = re.compile(r'(\d+)')   # split on digit groups
    def _nat_key(p: pathlib.Path):
        parts = numeric_re.split(p.stem)      # e.g. "img_12_a" → ['img_', '12', '_a']
        key = []
        for part in parts:
            if part.isdigit():
                key.append((0, int(part)))    # numbers sort before letters
            else:
                key.append((1, part.lower())) # case‑insensitive for text
        return key
    files.sort(key=_nat_key)
    return files

# --------------------------------------------------------------
# Helper – load + optional resize
# --------------------------------------------------------------
def _load_and_resize(img_path: pathlib.Path, size: Tuple[int, int]) -> np.ndarray:
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Cannot read image {img_path}")
    if size is not None:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img

# --------------------------------------------------------------
# Core – build the side‑by‑side video
# --------------------------------------------------------------
def make_side_by_side_video(
    root: pathlib.Path,
    output_path: pathlib.Path,
    fps: int = 30,
    forced_size: Tuple[int, int] | None = None,
    verbose: bool = False,
) -> None:
    # 1️⃣ discover sub‑folders
    subfolders = [p for p in root.iterdir() if p.is_dir()]
    if not subfolders:
        sys.exit(f"❌ No sub‑folders found under {root}")
    if verbose:
        print(f"🔎 Found {len(subfolders)} sub‑folders: {[p.name for p in subfolders]}")
    # 2️⃣ collect image lists per stream
    streams: Dict[str, List[pathlib.Path]] = {}
    for sf in subfolders:
        img_paths = _collect_image_paths(sf)
        if not img_paths:
            print(f"⚠️  Sub‑folder {sf} contains no images – will be padded with black frames.", file=sys.stderr)
        streams[sf.name] = img_paths
    # 3️⃣ decide on a common frame size
    if forced_size is not None:
        common_size = forced_size
    else:
        widths, heights = [], []
        for paths in streams.values():
            if paths:
                img = cv2.imread(str(paths[0]), cv2.IMREAD_COLOR)
                if img is not None:
                    h, w = img.shape[:2]
                    widths.append(w)
                    heights.append(h)
        if not widths:
            common_size = (640, 480)        # fallback
        else:
            common_size = (min(widths), min(heights))
    if verbose:
        print(f"📐 Common frame size = {common_size[0]}×{common_size[1]}")
    # 4️⃣ initialise VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (common_size[0] * len(subfolders), common_size[1]),
    )
    if not out_writer.isOpened():
        sys.exit(f"❌ Unable to create VideoWriter for {output_path}")
    # 5️⃣ length = longest stream
    max_frames = max(len(p) for p in streams.values())
    if verbose:
        print(f"🖼️  Video will contain {max_frames} frames (longest stream).")
    # 6️⃣ per‑frame loop
    for idx in range(max_frames):
        parts = []
        for name in sorted(streams.keys()):   # alphabetical order
            img_list = streams[name]
            if idx < len(img_list):
                img = _load_and_resize(img_list[idx], common_size)
            else:
                # black placeholder
                img = np.zeros((common_size[1], common_size[0], 3), dtype=np.uint8)
                cv2.putText(
                    img,
                    f"{name} (pad)",
                    (10, common_size[1] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            parts.append(img)
        side_by_side = np.hstack(parts)
        out_writer.write(side_by_side)
        if verbose and (idx % 50 == 0 or idx == max_frames - 1):
            print(f"   → wrote frame {idx + 1}/{max_frames}")
    # 7️⃣ cleanup
    out_writer.release()
    if verbose:
        print(f"\n✅ Video saved to: {output_path}")

# --------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a side‑by‑side video from images stored in sub‑folders."
    )
    parser.add_argument(
        "root_folder",
        type=pathlib.Path,
        help="Folder that contains the sub‑folders with images (e.g. ./fixed_saturation).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("output.mp4"),
        help="Output video file (default: output.mp4).",
    )
    parser.add_argument(
        "-r",
        "--fps",
        type=int,
        default=30,
        help="Frames per second (default: 30).",
    )
    parser.add_argument(
        "--size",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        help="Force frames to this size (WIDTH HEIGHT). If omitted the script uses the smallest common size.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print progress information.",
    )
    args = parser.parse_args()
    forced_sz = tuple(args.size) if args.size else None
    make_side_by_side_video(
        root=args.root_folder,
        output_path=args.output,
        fps=args.fps,
        forced_size=forced_sz,
        verbose=args.verbose,
    )

if __name__ == "__main__":
    main()
