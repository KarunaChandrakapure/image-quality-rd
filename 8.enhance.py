#!/usr/bin/env python3
"""
enhance.py - one command for any folder: day and night frames each get their own pipeline.

Per image:
    1. resize       (adaptive_config.yaml -> resize)
    2. day / night  night_check from 7.night_clean.py: dark overall OR dark sky
    3. day   -> 5.adaptive_enhance.py:  diagnose -> levels, local contrast, sharpen, colour
                                        (plus fixes for any diagnosed issue)
       night -> 7.night_clean.py:       glare zone kept as is; elsewhere brightness-adaptive
                                        denoise + sharpen (not in flat dark areas)

Settings:
    adaptive_config.yaml     general (input/output, write flags, layout), resize, day pipeline
    night_clean_config.yaml  night detection, glare zone, night cleaning
                             (its general/resize sections are not used here)

Output (in output/<input folder>_enhanced/):
    <image>.png          original + result (per write_original / write_enhanced / layout)
    _glare_masks/        night glare zones, white = left untouched (if save_mask in night config)
    _report.csv          day/night, metrics, issues, actions per image

Usage:
    python 8.enhance.py
    python 8.enhance.py --input v1.0-mini/samples/CAM_FRONT --limit 50
"""

import argparse
import csv
import importlib.util
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ae = _load("adaptive_enhance", "5.adaptive_enhance.py")     # day pipeline + shared helpers
nc = _load("night_clean", "7.night_clean.py")               # night pipeline


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--day-config", default=str(HERE / "adaptive_config.yaml"))
    ap.add_argument("--night-config", default=str(HERE / "night_clean_config.yaml"))
    ap.add_argument("--input", help="overrides general.input_dir in the day config")
    ap.add_argument("--out", help="overrides general.output_dir in the day config")
    ap.add_argument("--limit", type=int, help="overrides general.limit in the day config")
    args = ap.parse_args()

    with open(args.day_config, "r", encoding="utf-8") as f:
        day = yaml.safe_load(f)
    with open(args.night_config, "r", encoding="utf-8") as f:
        night = yaml.safe_load(f)
    g = day["general"]
    if args.input:
        g["input_dir"] = args.input
    if args.out:
        g["output_dir"] = args.out
    if args.limit is not None:
        g["limit"] = args.limit

    files = ae.list_images(g)
    if not files:
        print(f"No images found in {g['input_dir']}")
        return

    out_dir = Path(g["output_dir"]) / (Path(g["input_dir"]).resolve().name + "_enhanced")
    out_dir.mkdir(parents=True, exist_ok=True)
    stack = np.vstack if g["layout"] == "stacked" else np.hstack

    rows = []
    for path in files:
        img = cv2.imread(str(path))
        if img is None:
            print(f"Could not read: {path.name}")
            continue
        t = time.time()
        small = ae.resize(img, day["resize"])                                       # 1
        is_night, brightness, sky = nc.night_check(small, night["night"])           # 2
        rel = path.relative_to(g["input_dir"])
        row = {"file": str(rel), "time_of_day": "night" if is_night else "day",
               "brightness": round(brightness, 1), "sky_brightness": round(sky, 1)}

        if is_night:                                                                # 3
            result, zone, info = nc.clean_night(small, night)
            actions = info["actions"]
            row.update(issues="night", noise_sigma=round(info["noise_sigma"], 3),
                       glare_pct=round(info["glare_pct"], 2))
            if night["general"]["save_mask"]:
                d = out_dir / "_glare_masks" / rel.parent
                d.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str((d / rel.name).with_suffix(".png")), (zone * 255 + 0.5).astype(np.uint8))
        else:
            issues, m = ae.diagnose(small, day["diagnose"])
            result, actions = ae.enhance(small, issues, m, day["fix"])
            row.update(issues=";".join(issues), noise_sigma=round(m["noise_sigma"], 3), glare_pct="")
        row.update(actions=";".join(actions), seconds=round(time.time() - t, 3))

        changed = actions != ["none"]
        left, right = small, result
        if g["labels"]:
            left = ae.labelled(small, f"Original ({row['time_of_day']}): {row['issues']}")
            right = ae.labelled(result, "Enhanced: " + " + ".join(actions))
        parts = []
        if g["write_original"] or not changed:
            parts.append(left)
        if g["write_enhanced"] and changed:
            parts.append(right)
        (out_dir / rel.parent).mkdir(parents=True, exist_ok=True)
        ae.save_image(out_dir / rel, stack(parts), g)

        rows.append(row)
        print(f"{rel}: {row['time_of_day']} -> {', '.join(actions)}")

    if g["save_report"] and rows:
        with open(out_dir / "_report.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    n_night = sum(r["time_of_day"] == "night" for r in rows)
    print(f"\n{len(rows)} image(s): {len(rows) - n_night} day, {n_night} night.  Output: {out_dir}")


if __name__ == "__main__":
    main()
