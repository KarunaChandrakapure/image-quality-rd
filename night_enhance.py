#!/usr/bin/env python3
"""
night_enhance.py - night images: less grain, sharper, softer glare.

    python night_enhance.py                     # night_enhance.input_dir from config.yaml
    python night_enhance.py path/to/folder      # any folder, all sub-folder levels

Per image:
    1. denoise   stronger in dark areas, gentler in lit ones
    2. sharpen   object outlines only (not noise, not the night sky)
    3. glare     headlights / street lamps keep their white core; the glow and road
                 reflections around them are turned down. Only the smooth glow is
                 compressed - edges and texture are kept, so nothing gets blurred.
Output: <night_enhance.output_dir>/<input folder name>/<same sub-folders>/<image>.png
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

import enhance_lib as lib
from enhance import find_images, output_path


def reduce_glare(img, zone, c, light_threshold):
    """Split luma into a smooth base (the glow, a wide blur) and detail (everything else:
    edges, texture). Inside the glare zone, the base is turned down above `knee`; the
    detail is added back untouched, so edges keep their full contrast (no blur).
    The lights themselves are left white."""
    ycc = lib.to_ycc(img)
    y = ycc[..., 0] / 255
    base = cv2.GaussianBlur(y, (0, 0), c["glow_radius"])
    weight = cv2.GaussianBlur(zone, (0, 0), c["glow_radius"] / 2)        # smooth -> no seams
    if c["keep_lights"]:
        core = (y >= light_threshold).astype(np.float32)
        weight *= 1 - np.clip(cv2.GaussianBlur(core, (0, 0), 2) * 2, 0, 1)
    glow = c["strength"] * weight * np.clip(base - c["knee"], 0, None)   # smooth, so edges stay
    y_new = np.maximum(y - glow, y * c["min_keep"])
    detail = y - cv2.GaussianBlur(y, (0, 0), 1.5)                        # optional edge boost
    y_new += (c["detail_gain"] - 1) * weight * detail
    ycc[..., 0] = np.clip(y_new, 0, 1) * 255
    return lib.from_ycc(ycc)


def enhance_night(img, cfg):
    """Returns (raw resized, enhanced, glare-zone share %)."""
    img = lib.resize(img, cfg["resize"])
    y = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    noise = lib.noise_level(y)
    n = cfg["night"]

    clean = lib.denoise(img, noise, {**cfg["denoise"], "h_per_sigma": n["denoise_h_per_sigma"]})   # 1
    ycc = lib.to_ycc(clean)
    ycc[..., 0] = lib.sharpen(ycc[..., 0] / 255, n["sharpen_amount"], cfg["sharpen"]) * 255         # 2
    clean = lib.from_ycc(ycc)

    zone = lib.glare_zone(y / 255, n["glare"])                                                     # 3
    out = (zone[..., None] * img + (1 - zone[..., None]) * clean + 0.5).astype(np.uint8)
    g = cfg["night_enhance"]["glare_reduction"]
    if g["enabled"] and zone.any():
        out = reduce_glare(out, zone, g, n["glare"]["light_threshold"])
    return img, out, float((zone > 0.5).mean() * 100)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?", help="input folder (default: night_enhance.input_dir)")
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    c = cfg["night_enhance"]
    in_dir = Path(args.folder or c["input_dir"]).resolve()
    images = find_images(in_dir, cfg["extensions"])
    if not images:
        print(f"No images found in {in_dir}")
        return
    join = np.vstack if cfg["comp_layout"] == "stacked" else np.hstack
    print(f"{len(images)} images in {in_dir}")

    start = time.time()
    done = skipped = failed = 0
    for i, src in enumerate(images, 1):
        img = cv2.imread(str(src))
        if img is None:
            print(f"Could not read: {src}")
            failed += 1
            continue
        if c["only_night_frames"]:
            small = lib.resize(img, cfg["resize"])
            noise = lib.noise_level(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32))
            if not lib.is_night(small, noise, cfg["night_check"]):
                raw = enhanced = small
                skipped += 1
            else:
                raw, enhanced, _ = enhance_night(img, cfg)
                done += 1
        else:
            raw, enhanced, _ = enhance_night(img, cfg)
            done += 1
        dst = output_path(src, in_dir, c["output_dir"], cfg["output_format"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), join([raw, enhanced]) if c["save_comp_frame"] else enhanced)
        if i % 100 == 0 or i == len(images):
            print(f"  {i}/{len(images)}  ({time.time() - start:.0f} s)")

    total = time.time() - start
    print(f"\nDone: {done} enhanced, {skipped} day frames copied unchanged, {failed} failed")
    print(f"Total time: {total:.1f} s  ({total / max(done + skipped, 1) * 1000:.0f} ms per image)")
    print(f"Output: {Path(c['output_dir']) / in_dir.name}")


if __name__ == "__main__":
    main()
