"""
enhance_lib.py - image processing used by enhance.py.

Day frames:   denoise (if noisy) -> white balance -> tone -> local contrast
              -> sharpen -> colour, each sized from the image's own measurements.
Night frames: denoise + sharpen everywhere except the glare zone (lights, glow,
              reflections), which is kept as in the original.
All images are uint8 BGR (OpenCV); luma functions work on float32 in [0, 1].
"""

import cv2
import numpy as np


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
def ramp(x, lo, hi):
    """0 below lo, 1 above hi, smooth in between."""
    t = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
    return t * t * (3 - 2 * t)


def resize(img, c):
    h, w = img.shape[:2]
    if c["mode"] == "width":
        return cv2.resize(img, (c["width"], max(1, round(c["width"] * h / w))), interpolation=cv2.INTER_AREA)
    if c["mode"] == "exact":
        return cv2.resize(img, (c["width"], c["height"]), interpolation=cv2.INTER_AREA)
    return img


def to_ycc(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(np.float32)


def from_ycc(ycc):
    return cv2.cvtColor(np.clip(ycc + 0.5, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR)


# ----------------------------------------------------------------------
# Measurements
# ----------------------------------------------------------------------
def noise_level(y):
    """Sensor noise (Immerkaer 1996) in the flattest 10% of 8x8 blocks. y: luma 0-255."""
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    r = np.abs(cv2.filter2D(y, -1, k)) * np.sqrt(0.5 * np.pi) / 6.0
    h, w = (y.shape[0] // 8) * 8, (y.shape[1] // 8) * 8
    blocks = r[:h, :w].reshape(h // 8, 8, w // 8, 8).mean(axis=(1, 3))
    return float(np.percentile(blocks, 10))


def blur_level(y):
    """Re-blur score (Crete 2007): 0 = sharp, 1 = very blurry. y: luma 0-255."""
    bv, bh = cv2.blur(y, (1, 9)), cv2.blur(y, (9, 1))
    dy, dx = np.abs(np.diff(y, axis=0)), np.abs(np.diff(y, axis=1))
    vy = np.maximum(0, dy - np.abs(np.diff(bv, axis=0)))
    vx = np.maximum(0, dx - np.abs(np.diff(bh, axis=1)))
    sy, sx = max(dy.sum(), 1e-6), max(dx.sum(), 1e-6)
    return float(max((sy - vy.sum()) / sy, (sx - vx.sum()) / sx))


def colour_tint(ycc, c):
    """Mean Cr / Cb offset of pixels that should be grey (blue tint: Cr < 0, Cb > 0)."""
    cr, cb, y = ycc[..., 1] - 128, ycc[..., 2] - 128, ycc[..., 0]
    grey = (np.hypot(cr, cb) < c["neutral_chroma"]) & (y > 30) & (y < 230)
    if grey.mean() < 0.02:
        grey = (y > 30) & (y < 230)
    return (float(cr[grey].mean()), float(cb[grey].mean())) if grey.any() else (0.0, 0.0)


def is_night(img, noise, c):
    """High sensor noise AND (dark overall OR dark sky)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sky = gray[:max(1, int(gray.shape[0] * c["sky_fraction"]))].mean()
    return noise > c["min_noise"] and (gray.mean() < c["max_brightness"] or sky < c["max_sky_brightness"])


# ----------------------------------------------------------------------
# Denoise and sharpen (shared)
# ----------------------------------------------------------------------
def denoise(img, noise, c):
    """Non-local means on luma, stronger in dark areas, gentler in lit ones; light on colour."""
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    h = float(np.clip(c["h_per_sigma"] * noise, c["h_min"], c["h_max"]))
    y = ycc[..., 0]
    lit = ramp(cv2.GaussianBlur(y.astype(np.float32) / 255, (0, 0), 8), c["dark_level"], c["bright_level"])
    strong = cv2.fastNlMeansDenoising(y, None, h * c["dark_factor"], 5, 15)
    gentle = cv2.fastNlMeansDenoising(y, None, h * c["bright_factor"], 5, 15)
    ycc[..., 0] = (lit * gentle + (1 - lit) * strong + 0.5).astype(np.uint8)
    for ch in (1, 2):
        ycc[..., ch] = cv2.fastNlMeansDenoising(ycc[..., ch], None, c["color_h"], 5, 15)
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


def cas(y, amount):
    """Contrast Adaptive Sharpening (AMD FidelityFX): sharpens less where contrast is high."""
    k = np.ones((3, 3), np.uint8)
    mn, mx = cv2.erode(y, k), cv2.dilate(y, k)
    amp = np.sqrt(np.clip(np.minimum(mn, 1 - mx) / np.maximum(mx, 1e-5), 0, 1))
    w = amp * (-1.0 / (8.0 - 3.0 * amount))
    p = cv2.copyMakeBorder(y, 1, 1, 1, 1, cv2.BORDER_REFLECT_101)
    cross = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    return np.clip((y + w * cross) / (1 + 4 * w), 0, 1)


def sharpen(y, amount, c):
    """CAS on object outlines only: not on fine texture / noise, not on flat dark areas."""
    g = cv2.GaussianBlur(y, (0, 0), c["edge_sigma"])
    edges = ramp(np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)),
                 c["edge_low"], c["edge_high"])
    mean = cv2.GaussianBlur(y, (0, 0), 6)
    std = np.sqrt(np.maximum(cv2.GaussianBlur(y * y, (0, 0), 6) - mean * mean, 0))
    flat_dark = (1 - ramp(mean, c["flat_dark_level"], c["flat_dark_level"] * 1.5)) * \
                (1 - ramp(std, c["flat_std"], c["flat_std"] * 2))
    weight = cv2.GaussianBlur(edges * (1 - flat_dark), (0, 0), 1.0)
    return (y + weight * (cas(y, amount) - y)).astype(np.float32)


# ----------------------------------------------------------------------
# Day
# ----------------------------------------------------------------------
def white_balance(ycc, c):
    cr, cb = colour_tint(ycc, c)
    ycc[..., 1] -= np.clip(c["strength"] * cr, -c["max_shift"], c["max_shift"])
    ycc[..., 2] -= np.clip(c["strength"] * cb, -c["max_shift"], c["max_shift"])


def tone(y, target, c):
    """Levels stretch, then a gamma moving mean brightness toward the target (0-255)."""
    lo, hi = np.percentile(y, c["levels_clip"]), np.percentile(y, 100 - c["levels_clip"])
    y = np.clip((y - lo) * min(1 / max(hi - lo, 1e-3), c["levels_max_gain"]), 0, 1)
    mean = float(np.clip(y.mean(), 1e-3, 0.999))
    want = np.clip(mean + c["strength"] * (target / 255 - mean), 1e-3, 0.999)
    g = float(np.clip(np.log(want) / np.log(mean), c["gamma_min"], c["gamma_max"]))
    return np.power(y, g).astype(np.float32)


def local_contrast(y, target, c):
    """CLAHE, blended in more when the image is flatter than the target contrast."""
    s = float(np.clip(c["base"] + c["gain"] * max(0.0, (target - y.std() * 255) / target), 0, c["max"]))
    clahe = cv2.createCLAHE(clipLimit=c["clip"], tileGridSize=(c["tile"], c["tile"]))
    eq = clahe.apply((y * 255 + 0.5).astype(np.uint8)).astype(np.float32) / 255
    return ((1 - s) * y + s * eq).astype(np.float32)


def colour(ycc, target, c):
    """Move the colour score toward the target. Boosts spare already-vivid and very dark pixels."""
    cr, cb = ycc[..., 1] - 128, ycc[..., 2] - 128
    current = max(float((np.abs(cr) + np.abs(cb)).mean() / 2), 0.3)
    factor = float(np.clip((target / current) ** c["strength"], c["min_factor"], c["max_factor"]))
    if factor > 1:
        w = (1 - np.clip(np.hypot(cr, cb) / c["protect"], 0, 1) ** 2) * \
            np.clip(ycc[..., 0] / 255 / c["dark_protect"], 0, 1)
        factor = 1 + (factor - 1) * w
    ycc[..., 1] = 128 + cr * factor
    ycc[..., 2] = 128 + cb * factor


def enhance_day(img, noise, blur, cfg):
    d = cfg["day"]
    if noise > d["denoise_min_noise"]:
        img = denoise(img, noise, cfg["denoise"])
    ycc = to_ycc(img)
    white_balance(ycc, d["white_balance"])
    y = tone(np.clip(ycc[..., 0] / 255, 0, 1), d["target_brightness"], d["tone"])
    y = local_contrast(y, d["target_contrast"], d["local_contrast"])
    s = d["sharpen"]
    amount = float(np.clip(s["base"] + s["blur_gain"] * max(0.0, blur - s["blur_ref"]), 0, s["max"]))
    ycc[..., 0] = sharpen(y, amount, cfg["sharpen"]) * 255
    colour(ycc, d["target_colour"], d["color"])
    return from_ycc(ycc)


# ----------------------------------------------------------------------
# Night
# ----------------------------------------------------------------------
def glare_zone(y, c):
    """Soft mask (1 = keep original): lights + the bright glow / reflections attached to
    them (within halo_max_px), grown by margin_px and feathered."""
    light = (y >= c["light_threshold"]).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(light, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= c["min_light_px"]       # ignore lone bright specks
    keep[0] = False                                                # label 0 = background
    light = keep[labels].astype(np.uint8)
    if not light.any():
        return np.zeros_like(y)
    k = 2 * c["halo_search_px"] + 1
    near = cv2.dilate(light, np.ones((k, k), np.uint8)) > 0
    dist = cv2.distanceTransform(1 - light, cv2.DIST_L2, 5)
    bright = ((y >= c["halo_threshold"]) & (dist <= c["halo_max_px"])).astype(np.uint8)
    _, labels = cv2.connectedComponents(bright, connectivity=8)
    touching = np.unique(labels[near & (bright > 0)])
    zone = (light > 0) | np.isin(labels, touching[touching > 0])
    k = 2 * c["margin_px"] + 1
    zone = cv2.dilate(zone.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(np.float32)
    return np.clip(np.maximum(zone, cv2.GaussianBlur(zone, (0, 0), c["feather_px"])), 0, 1)


def enhance_night(img, noise, cfg):
    n = cfg["night"]
    clean = denoise(img, noise, {**cfg["denoise"], "h_per_sigma": n["denoise_h_per_sigma"]})
    ycc = to_ycc(clean)
    ycc[..., 0] = sharpen(ycc[..., 0] / 255, n["sharpen_amount"], cfg["sharpen"]) * 255
    clean = from_ycc(ycc)
    zone = glare_zone(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255, n["glare"])[..., None]
    return (zone * img + (1 - zone) * clean + 0.5).astype(np.uint8)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def enhance(img, cfg):
    """Resized + enhanced image. Returns (raw resized, enhanced, 'day' | 'night')."""
    img = resize(img, cfg["resize"])
    y = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[..., 0].astype(np.float32)
    noise = noise_level(y)
    if is_night(img, noise, cfg["night_check"]):
        return img, enhance_night(img, noise, cfg), "night"
    return img, enhance_day(img, noise, blur_level(y), cfg), "day"
