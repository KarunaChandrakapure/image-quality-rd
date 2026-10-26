#!/usr/bin/env python3
"""
night_clean.py - less grain + sharper night frames, without touching the glare.

Per image:
    1. resize          to the model input size (same as 5.adaptive_enhance.py)
    2. night check     dark overall OR dark sky (top of the frame) -> night frame;
                       the sky test catches brightly lit night streets
                       (day frames are passed through unchanged by default)
    3. glare zone      light sources (headlights, street lamps) plus the glow and
                       wet-road reflections connected to them, plus a safety margin
    4. clean           outside the glare zone: denoise -> edge-gated sharpen
                       inside the glare zone:  original pixels, untouched
                       (soft transition at the border, so there is no visible seam)

Output (in output/<input folder>_clean/):
    <image>.png               original | result   (or result only, see config)
    _glare_masks/<image>.png  the glare zone, white = left untouched
    _report.csv               night?, brightness, noise, glare %, actions

Usage:
    python 7.night_clean.py
    python 7.night_clean.py --input sample5 --limit 10

Install:
    pip install opencv-python numpy pyyaml
"""

import argparse
import csv
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent

# Shared helpers (resize, noise estimate, edge-gated CAS, file listing, saving) live in
# 5.adaptive_enhance.py; loaded by path because the file name starts with a digit.
_spec = importlib.util.spec_from_file_location("adaptive_enhance", HERE / "5.adaptive_enhance.py")
ae = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ae)


# ----------------------------------------------------------------------
# Step 3: glare zone
# ----------------------------------------------------------------------
def glare_zone(y, c):
    """Float mask 0..1 (1 = glare, leave untouched).  y: float32 luma in [0,1].

    a) light sources: pixels >= light_threshold, ignoring specks < min_light_px
    b) glow / reflections: pixels >= halo_threshold, no further than halo_max_px from
       a light, whose connected bright region touches a light source (within
       halo_search_px). A bright wall or white car that is NOT attached to a light
       is not included.
    c) grow by margin_px, then feather by feather_px."""
    light = (y >= c["light_threshold"]).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(light, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= c["min_light_px"]
    light = keep[labels].astype(np.uint8)
    if not light.any():
        return np.zeros_like(y), 0

    k = 2 * c["halo_search_px"] + 1
    near_light = cv2.dilate(light, np.ones((k, k), np.uint8))
    # glow can only be within halo_max_px of a light, so a bright region that happens to
    # touch a light (a white car next to a lit shopfront) is not swallowed whole
    dist = cv2.distanceTransform((light == 0).astype(np.uint8), cv2.DIST_L2, 5)
    bright = ((y >= c["halo_threshold"]) & (dist <= c["halo_max_px"])).astype(np.uint8)
    n, labels = cv2.connectedComponents(bright, connectivity=8)
    touching = np.unique(labels[(near_light > 0) & (bright > 0)])
    halo = np.isin(labels, touching[touching > 0])

    zone = (light > 0) | halo
    if c["margin_px"] > 0:
        k = 2 * c["margin_px"] + 1
        zone = cv2.dilate(zone.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    soft = zone.astype(np.float32)
    if c["feather_px"] > 0:
        soft = np.maximum(soft, cv2.GaussianBlur(soft, (0, 0), c["feather_px"]))
    n_lights = cv2.connectedComponents(light, connectivity=8)[0] - 1
    return np.clip(soft, 0, 1), n_lights


# ----------------------------------------------------------------------
# Step 4: clean (applied to the whole frame, then blended back outside the glare zone)
# ----------------------------------------------------------------------
def _ramp(x, lo, hi):
    """0 below lo, 1 above hi, smooth in between."""
    t = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
    return t * t * (3 - 2 * t)


def denoise(img, sigma, c):
    """Non-local means, stronger in dark areas (where night noise is worst) and
    gentler in lit areas (where there is real texture to keep). Two passes at the
    two strengths are blended per pixel by the local brightness."""
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    h = float(np.clip(c["h_per_sigma"] * sigma, c["h_min"], c["h_max"]))
    tw, sw = c["template_window"], c["search_window"]
    a = c["adaptive"]
    if a["enabled"]:
        local = cv2.GaussianBlur(ycc[..., 0].astype(np.float32) / 255.0, (0, 0), a["radius"])
        lit = _ramp(local, a["dark_level"], a["bright_level"])            # 0 = dark, 1 = lit
        strong = cv2.fastNlMeansDenoising(ycc[..., 0], None, h * a["dark_factor"], tw, sw)
        gentle = cv2.fastNlMeansDenoising(ycc[..., 0], None, h * a["bright_factor"], tw, sw)
        ycc[..., 0] = (lit * gentle + (1 - lit) * strong + 0.5).astype(np.uint8)
        label = f"denoise(h={h * a['dark_factor']:.1f}-{h * a['bright_factor']:.1f})"
    else:
        ycc[..., 0] = cv2.fastNlMeansDenoising(ycc[..., 0], None, h, tw, sw)
        label = f"denoise(h={h:.1f})"
    if c["color_h"] > 0:
        for ch in (1, 2):
            ycc[..., ch] = cv2.fastNlMeansDenoising(ycc[..., ch], None, c["color_h"], tw, sw)
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR), label


def flat_dark_mask(y, c):
    """1 where the image is both very dark and nearly featureless (night sky, unlit
    walls, deep shadow): there is nothing there to sharpen except leftover noise."""
    r = c["radius"]
    mean = cv2.GaussianBlur(y, (0, 0), r)
    std = np.sqrt(np.maximum(cv2.GaussianBlur(y * y, (0, 0), r) - mean * mean, 0))
    dark = 1 - _ramp(mean, c["dark_level"], c["dark_level"] * 1.5)
    flat = 1 - _ramp(std, c["flat_std"], c["flat_std"] * 2)
    return cv2.GaussianBlur((dark * flat).astype(np.float32), (0, 0), 2)


def sharpen(img, c):
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y = ycc[..., 0].astype(np.float32) / 255.0
    delta = ae.sharpen(y, c["amount"], c) - y
    if c["skip_flat_dark"]["enabled"]:
        delta *= 1 - flat_dark_mask(y, c["skip_flat_dark"])
    ycc[..., 0] = (np.clip(y + delta, 0, 1) * 255 + 0.5).astype(np.uint8)
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR), f"sharpen({c['amount']:.1f})"


def night_check(img, c):
    """Night if dark overall OR the sky (top of the frame) is dark. The sky test
    catches brightly lit night streets. Returns (is_night, brightness, sky)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    sky = float(gray[:max(1, int(gray.shape[0] * c["sky_fraction"]))].mean())
    return brightness < c["max_brightness"] or sky < c["max_sky_brightness"], brightness, sky


def clean_night(img, cfg):
    """Returns (result, glare mask, info dict)."""
    y = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[..., 0].astype(np.float32)
    sigma = ae.noise_sigma(y)
    zone, n_lights = glare_zone(y / 255.0, cfg["glare"])

    out, actions = img, []
    if cfg["clean"]["denoise"]["enabled"]:
        out, a = denoise(out, sigma, cfg["clean"]["denoise"])
        actions.append(a)
    if cfg["clean"]["sharpen"]["enabled"]:
        out, a = sharpen(out, cfg["clean"]["sharpen"])
        actions.append(a)

    m = zone[..., None]
    out = (m * img.astype(np.float32) + (1 - m) * out.astype(np.float32) + 0.5).astype(np.uint8)
    if n_lights:
        actions.append(f"glare kept ({n_lights} lights)")
    return out, zone, {"noise_sigma": sigma, "lights": n_lights,
                       "glare_pct": float((zone > 0.5).mean() * 100), "actions": actions}


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "night_clean_config.yaml"))
    ap.add_argument("--input", help="overrides general.input_dir")
    ap.add_argument("--out", help="overrides general.output_dir")
    ap.add_argument("--limit", type=int, help="overrides general.limit")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    g = cfg["general"]
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

    out_dir = Path(g["output_dir"]) / (Path(g["input_dir"]).resolve().name + "_clean")
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = out_dir / "_glare_masks"
    stack = np.vstack if g["layout"] == "stacked" else np.hstack

    rows = []
    for path in files:
        img = cv2.imread(str(path))
        if img is None:
            print(f"Could not read: {path.name}")
            continue
        small = ae.resize(img, cfg["resize"])                                         # 1
        is_night, brightness, sky = night_check(small, cfg["night"])                   # 2
        rel = path.relative_to(g["input_dir"])

        if is_night or cfg["night"]["process_day_frames"]:
            result, zone, info = clean_night(small, cfg)                               # 3 + 4
        else:
            result, zone, info = small, None, {"noise_sigma": float("nan"), "lights": 0,
                                               "glare_pct": 0.0, "actions": ["day frame - unchanged"]}
        changed = result is not small

        parts = [small] if (g["write_original"] or not changed or not g["write_result"]) else []
        if changed and g["write_result"]:
            parts.append(result)
        (out_dir / rel.parent).mkdir(parents=True, exist_ok=True)
        ae.save_image(out_dir / rel, stack(parts), g)
        if g["save_mask"] and zone is not None:
            (mask_dir / rel.parent).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str((mask_dir / rel).with_suffix(".png")), (zone * 255 + 0.5).astype(np.uint8))

        rows.append({"file": str(rel), "night": is_night, "brightness": round(brightness, 1),
                     "sky_brightness": round(sky, 1),
                     "noise_sigma": round(info["noise_sigma"], 3), "lights": info["lights"],
                     "glare_pct": round(info["glare_pct"], 2), "actions": ";".join(info["actions"])})
        print(f"{rel}: {'night' if is_night else 'day'} -> {', '.join(info['actions'])}")

    if g["save_report"] and rows:
        with open(out_dir / "_report.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    n_night = sum(r["night"] for r in rows)
    print(f"\n{len(rows)} image(s): {n_night} night, {len(rows) - n_night} day.  Output: {out_dir}")


if __name__ == "__main__":
    main()
