#!/usr/bin/env python3
"""
enhance.py - enhance every image in a folder (all sub-folder levels).

    python enhance.py                     # input_dir from config.yaml
    python enhance.py path/to/folder      # any folder

Output goes to <output_dir>/<input folder name>/<same sub-folders>/<image>.
save_comp_frame in config.yaml: true = raw | enhanced side by side, false = enhanced only.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

import enhance_lib


def find_images(folder, extensions):
    exts = {e.lower() for e in extensions}
    return sorted(p for p in Path(folder).rglob("*") if p.is_file() and p.suffix.lower() in exts)


def output_path(src, in_dir, out_dir, fmt):
    dst = Path(out_dir) / in_dir.name / src.relative_to(in_dir)
    return dst if fmt == "same" else dst.with_suffix("." + fmt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?", help="input folder (default: input_dir in config.yaml)")
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    in_dir = Path(args.folder or cfg["input_dir"]).resolve()
    images = find_images(in_dir, cfg["extensions"])
    if not images:
        print(f"No images found in {in_dir}")
        return
    join = np.vstack if cfg["comp_layout"] == "stacked" else np.hstack
    print(f"{len(images)} images in {in_dir}")

    start = time.time()
    counts = {"day": 0, "night": 0, "failed": 0}
    for i, src in enumerate(images, 1):
        img = cv2.imread(str(src))
        if img is None:
            print(f"Could not read: {src}")
            counts["failed"] += 1
            continue
        raw, enhanced, kind = enhance_lib.enhance(img, cfg)
        counts[kind] += 1
        dst = output_path(src, in_dir, cfg["output_dir"], cfg["output_format"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), join([raw, enhanced]) if cfg["save_comp_frame"] else enhanced)
        if i % 100 == 0 or i == len(images):
            print(f"  {i}/{len(images)}  ({time.time() - start:.0f} s)")

    total = time.time() - start
    done = counts["day"] + counts["night"]
    print(f"\nDone: {done} images ({counts['day']} day, {counts['night']} night, {counts['failed']} failed)")
    print(f"Total time: {total:.1f} s  ({total / max(done, 1) * 1000:.0f} ms per image)")
    print(f"Output: {Path(cfg['output_dir']) / in_dir.name}")


if __name__ == "__main__":
    main()
