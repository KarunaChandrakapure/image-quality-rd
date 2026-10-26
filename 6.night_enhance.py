#!/usr/bin/env python3
"""
night_enhance.py - compare night-time enhancement methods, one folder per method.

Targets the two night problems:
    low light  -> gamma, clahe, shadow_lift, lime, retinex, dehaze_inverse, zero_dce_pp*
    glare      -> glare_suppress, highlight_rolloff, flare7kpp*   (headlights, street lights)
    noise      -> scunet*
and "night_pipeline" / "night_pipeline_dl", which chain the chosen steps into one result.
* = learned model: needs PyTorch and the weights in models/ (see night_config.yaml).

Per image: resize (same as 5.adaptive_enhance.py) -> every enabled method.
Output (in output/<input folder>_night/):
    <method>/<image>.png        original | method result   (or result only)
    _all_methods/<image>.png    grid: original + every method, in config order
    _report.csv                 brightness, near-black %, clipped %, noise, glare halo, sharpness

Usage:
    python 6.night_enhance.py
    python 6.night_enhance.py --input sample5 --limit 10
    python 6.night_enhance.py --method lime            # run one method only

Install:
    pip install opencv-python numpy pyyaml
    learned methods also:  pip install torch spandrel timm einops
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

# Shared helpers (resize, noise estimate, levels, CLAHE, edge-gated CAS, saving)
# live in 5.adaptive_enhance.py; loaded by path because the name starts with a digit.
_spec = importlib.util.spec_from_file_location("adaptive_enhance", HERE / "5.adaptive_enhance.py")
ae = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ae)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def split_luma(img):
    """BGR uint8 -> (YCrCb uint8, Y float32 in [0,1])."""
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    return ycc, ycc[..., 0].astype(np.float32) / 255.0


def merge_luma(ycc, y, saturation=1.0):
    ycc = ycc.copy()
    ycc[..., 0] = (np.clip(y, 0, 1) * 255 + 0.5).astype(np.uint8)
    if abs(saturation - 1.0) > 0.01:
        ycc = ae.scale_chroma(ycc, saturation)
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


def light_mask(y, p):
    """Light sources: very bright pixels, ignoring lone specks."""
    m = (y >= p["light_threshold"]).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= p["min_light_px"]
    return keep[labels]


# ----------------------------------------------------------------------
# Low-light methods   (BGR uint8 -> BGR uint8)
# ----------------------------------------------------------------------
def m_denoise(img, p, ctx):
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    h = float(np.clip(p["h_per_sigma"] * ctx["noise_sigma"], p["h_min"], p["h_max"]))
    tw, sw = p["template_window"], p["search_window"]
    ycc[..., 0] = cv2.fastNlMeansDenoising(ycc[..., 0], None, h, tw, sw)
    if p["color_h"] > 0:
        for ch in (1, 2):
            ycc[..., ch] = cv2.fastNlMeansDenoising(ycc[..., ch], None, p["color_h"], tw, sw)
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


def m_gamma(img, p, ctx):
    """Global gamma: every pixel lifted by the same curve. Simple baseline - it also
    lifts the sky to grey and spreads glare."""
    ycc, y = split_luma(img)
    mean = float(np.clip(y.mean(), 1e-3, 0.99))
    g = float(np.clip(np.log(p["target_mean"]) / np.log(mean), p["gamma_min"], 1.0))
    return merge_luma(ycc, np.power(y, g))


def m_clahe(img, p, ctx):
    ycc, y = split_luma(img)
    clahe = cv2.createCLAHE(clipLimit=p["clip"], tileGridSize=(p["tile"],) * 2)
    return merge_luma(ycc, clahe.apply(ycc[..., 0]).astype(np.float32) / 255.0)


def m_shadow_lift(img, p, ctx):
    """Local tone mapping (same as 5.adaptive_enhance.py): gain follows the smoothed
    local brightness, so shadows are lifted and lights / black sky stay put."""
    ycc, y = split_luma(img)
    y, gain = ae.shadow_lift(y, p)
    return merge_luma(ycc, y, 1.0 + p["color_restore"] * (gain - 1.0))


def m_lime(img, p, ctx):
    """LIME-style (Guo et al. 2017): illumination = brightest colour channel, smoothed
    edge-aware; the image is divided by it. Works on all three colour channels, so
    colours brighten together and stay saturated."""
    f = img.astype(np.float32) / 255.0
    t = f.max(axis=2)
    t = cv2.bilateralFilter(t, p["smooth_d"], p["smooth_sigma_color"], p["smooth_sigma_space"])
    t = np.power(np.clip(t, p["t_min"], 1.0), p["gamma"])
    out = np.clip(f / t[..., None], 0, 1)
    out = (1 - p["strength"]) * f + p["strength"] * out
    return (out * 255 + 0.5).astype(np.uint8)


def m_retinex(img, p, ctx):
    """Multi-scale Retinex on brightness: log(image) - log(blurred image) at several
    scales removes the slowly varying illumination and keeps local detail."""
    ycc, y = split_luma(img)
    ly = np.log(y + 1e-3)
    r = np.zeros_like(y)
    for s in p["scales"]:
        r += ly - np.log(cv2.GaussianBlur(y, (0, 0), s) + 1e-3)
    r /= len(p["scales"])
    lo, hi = np.percentile(r, p["clip_low"]), np.percentile(r, p["clip_high"])
    r = np.clip((r - lo) / max(hi - lo, 1e-6), 0, 1)
    return merge_luma(ycc, (1 - p["strength"]) * y + p["strength"] * r)


def m_dehaze_inverse(img, p, ctx):
    """Dong et al. 2011: an inverted low-light image looks like a hazy image, so
    invert -> dark-channel dehaze -> invert back."""
    inv = 1.0 - img.astype(np.float32) / 255.0
    kernel = np.ones((p["patch"], p["patch"]), np.uint8)
    dark = cv2.erode(inv.min(axis=2), kernel)
    idx = np.argsort(dark.ravel())[-max(1, dark.size // 1000):]        # brightest 0.1%
    A = inv.reshape(-1, 3)[idx].max(axis=0)
    t = 1.0 - p["omega"] * cv2.erode((inv / np.maximum(A, 1e-3)).min(axis=2), kernel)
    t = np.where(t < 0.5, 2 * t * t, t)                                 # Dong's adjustment
    t = cv2.GaussianBlur(t, (0, 0), p["patch"])
    t = np.maximum(t, p["t_min"])[..., None]
    j = (inv - A) / t + A
    return ((1.0 - np.clip(j, 0, 1)) * 255 + 0.5).astype(np.uint8)


# ----------------------------------------------------------------------
# Glare methods
# ----------------------------------------------------------------------
def m_glare_suppress(img, p, ctx):
    """Veiling-glare removal: the glow around a light is (roughly) the light source
    blurred by the lens. Estimate it by blurring the light-source mask at a small and
    a wide radius, and subtract it - objects next to headlights / street lights
    reappear. The light itself (the clipped core) is left white."""
    ycc, y = split_luma(img)
    core = light_mask(y, p)
    src = y * core
    glow = np.zeros_like(y)
    for s, w in zip(p["glow_sigmas"], p["glow_weights"]):
        glow += w * cv2.GaussianBlur(src, (0, 0), s)
    # Scale so the glow just touches the pixels right next to the lights.
    ring = cv2.dilate(core.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~core
    if ring.any() and glow[ring].mean() > 1e-4:
        glow *= np.clip(np.median(y[ring] / np.maximum(glow[ring], 1e-4)), 0, 4) * 0.7
    out = np.clip(y - p["strength"] * glow, 0, 1)
    out = np.maximum(out, y * p["min_keep"])           # never darken a pixel below this share
    if p["keep_core"]:
        # feathered: fade from "light kept" to "glow removed" over a few px, so the
        # light does not end in a hard edge with a dark ring around it
        soft = cv2.GaussianBlur(core.astype(np.float32), (0, 0), p["core_feather"])
        soft = np.clip(soft * 2, 0, 1)
        out = soft * y + (1 - soft) * out
    return merge_luma(ycc, out)


def m_highlight_rolloff(img, p, ctx):
    ycc, y = split_luma(img)
    return merge_luma(ycc, ae.highlight_rolloff(y, p))


# ----------------------------------------------------------------------
# Learned (deep-learning) methods - need PyTorch and the weights in models_dir.
# Loaded on first use only, so the classical methods run without torch.
#   scunet       real-noise denoiser      cszn/KAIR v1.0 release   Apache-2.0
#   zero_dce_pp  low-light curve model    Li-Chongyi/Zero-DCE_extension   NON-COMMERCIAL
#   flare7kpp    night flare / glare removal   ykdai/Flare7K           NON-COMMERCIAL (S-Lab 1.0)
# ----------------------------------------------------------------------
_MODELS = {}


def _torch():
    import torch
    torch.set_grad_enabled(False)
    return torch


def _to_tensor(img):
    torch = _torch()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0


def _to_image(t):
    rgb = (t.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _pad_to(x, multiple):
    """Reflect-pad H and W up to a multiple (window-based transformers need it)."""
    torch = _torch()
    h, w = x.shape[-2:]
    ph, pw = (-h) % multiple, (-w) % multiple
    return torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect"), h, w


def _load_state(path):
    torch = _torch()
    sd = torch.load(path, map_location="cpu", weights_only=True)    # tensors only, no code
    return sd.get("params_ema", sd.get("params", sd))


def m_scunet(img, p, ctx):
    """SCUNet (Zhang et al. 2023), trained on realistic camera noise. Removes night
    grain and colour speckle without the smeared look of non-local means."""
    if "scunet" not in _MODELS:
        import spandrel
        _MODELS["scunet"] = spandrel.ModelLoader().load_from_file(
            str(Path(ctx["models_dir"]) / p["weights"])).eval()
    x, h, w = _pad_to(_to_tensor(img), 64)
    out = _to_image(_MODELS["scunet"](x)[..., :h, :w])
    return cv2.addWeighted(out, p["strength"], img, 1 - p["strength"], 0)


def _zero_dce_net():
    """Zero-DCE++ network (Li et al. 2021), re-implemented from the paper's model.py
    so its weights can be loaded: predicts a per-pixel curve, applied 8 times."""
    torch = _torch()
    nn = torch.nn

    class DSConv(nn.Module):                    # depthwise-separable conv ("CSDN_Tem")
        def __init__(self, cin, cout):
            super().__init__()
            self.depth_conv = nn.Conv2d(cin, cin, 3, 1, 1, groups=cin)
            self.point_conv = nn.Conv2d(cin, cout, 1)

        def forward(self, x):
            return self.point_conv(self.depth_conv(x))

    class ZeroDCEpp(nn.Module):
        def __init__(self, nf=32):
            super().__init__()
            self.e_conv1, self.e_conv2 = DSConv(3, nf), DSConv(nf, nf)
            self.e_conv3, self.e_conv4 = DSConv(nf, nf), DSConv(nf, nf)
            self.e_conv5, self.e_conv6 = DSConv(nf * 2, nf), DSConv(nf * 2, nf)
            self.e_conv7 = DSConv(nf * 2, 3)

        def forward(self, x):
            r = torch.relu
            x1 = r(self.e_conv1(x)); x2 = r(self.e_conv2(x1))
            x3 = r(self.e_conv3(x2)); x4 = r(self.e_conv4(x3))
            x5 = r(self.e_conv5(torch.cat([x3, x4], 1)))
            x6 = r(self.e_conv6(torch.cat([x2, x5], 1)))
            a = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
            for _ in range(8):
                x = x + a * (x * x - x)
            return x

    return ZeroDCEpp()


def m_zero_dce_pp(img, p, ctx):
    if "zero_dce_pp" not in _MODELS:
        net = _zero_dce_net()
        net.load_state_dict(_load_state(Path(ctx["models_dir"]) / p["weights"]))
        _MODELS["zero_dce_pp"] = net.eval()
    out = _to_image(_MODELS["zero_dce_pp"](_to_tensor(img)))
    return cv2.addWeighted(out, p["strength"], img, 1 - p["strength"], 0)


def _load_uformer(models_dir, p):
    """Flare7K++ uses a Uformer from its own repo (models/Flare7K). Only the network
    file is loaded; its one framework import (a class registry) is stubbed out."""
    import sys
    import types
    reg = types.ModuleType("basicsr.utils.registry")

    class _Registry:
        def register(self, obj=None):
            return obj if obj is not None else (lambda o: o)

    reg.ARCH_REGISTRY = _Registry()
    for name in ("basicsr", "basicsr.utils"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["basicsr.utils.registry"] = reg
    arch = Path(models_dir) / p["repo"] / "basicsr" / "archs" / "uformer_arch.py"
    spec = importlib.util.spec_from_file_location("flare7k_uformer_arch", arch)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    net = mod.Uformer(img_size=512, img_ch=3, output_ch=6)
    net.load_state_dict(_load_state(Path(models_dir) / p["weights"]))
    return net.eval()


def m_flare7kpp(img, p, ctx):
    """Flare7K++ (Dai et al., NeurIPS 2022 / TPAMI 2024): splits a night image into
    'scene' and 'flare' (glow, streaks, reflections of light sources) and keeps the
    scene. The network outputs 6 channels: deflared image + predicted flare."""
    if "flare7kpp" not in _MODELS:
        _MODELS["flare7kpp"] = _load_uformer(ctx["models_dir"], p)
    # This Uformer only accepts SQUARE inputs (it derives height = width = sqrt(pixels)).
    s = p["model_size"]
    h, w = img.shape[:2]
    if p["input_mode"] == "resize":         # what the official test.py does
        x = _to_tensor(cv2.resize(img, (s, s), interpolation=cv2.INTER_CUBIC))
        out = cv2.resize(_to_image(_MODELS["flare7kpp"](x)[:, :3]), (w, h), interpolation=cv2.INTER_AREA)
    else:                                   # "pad": keep proportions, pad to a square
        side = max(s, -(-max(h, w) // 128) * 128)
        sq = cv2.copyMakeBorder(img, 0, side - h, 0, side - w, cv2.BORDER_REPLICATE)
        out = _to_image(_MODELS["flare7kpp"](_to_tensor(sq))[:, :3])[:h, :w]
    if p["keep_light_source"]:
        # put the light itself back (the model removes its glow, the lamp is real)
        _, y = split_luma(img)
        core = light_mask(y, {"light_threshold": 0.97, "min_light_px": 3}).astype(np.float32)
        soft = np.clip(cv2.GaussianBlur(core, (0, 0), 3) * 2, 0, 1)[..., None]
        out = (soft * img + (1 - soft) * out + 0.5).astype(np.uint8)
    return cv2.addWeighted(out, p["strength"], img, 1 - p["strength"], 0)


LOW_LIGHT = {
    "gamma": m_gamma, "clahe": m_clahe, "shadow_lift": m_shadow_lift,
    "lime": m_lime, "retinex": m_retinex, "dehaze_inverse": m_dehaze_inverse,
    "zero_dce_pp": m_zero_dce_pp,
}
GLARE = {"glare_suppress": m_glare_suppress, "highlight_rolloff": m_highlight_rolloff,
         "flare7kpp": m_flare7kpp}
DENOISE = {"scunet": m_scunet}


# ----------------------------------------------------------------------
# Combined pipeline
# ----------------------------------------------------------------------
def m_night_pipeline(img, P, ctx, block="night_pipeline"):
    c = P[block]
    for step in c["steps"]:
        if step == "denoise":
            img = m_denoise(img, P["denoise"], ctx)
        elif step in DENOISE:
            img = DENOISE[step](img, P[step], ctx)
        elif step == "low_light":
            img = LOW_LIGHT[c["low_light"]](img, P[c["low_light"]], ctx)
        elif step in GLARE:
            img = GLARE[step](img, P[step], ctx)
        elif step in ("levels", "local_contrast", "sharpen"):
            ycc, y = split_luma(img)
            if step == "levels":
                y = ae.auto_levels(y, P["levels"])
            elif step == "local_contrast":
                y = ae.local_contrast(y, P["local_contrast"])
            else:
                y = ae.sharpen(y, P["sharpen"]["amount"], P["sharpen"])
            img = merge_luma(ycc, y)
        elif step == "color":
            ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            s = min(P["color"]["saturation"], P["color"]["max_saturation"])
            img = cv2.cvtColor(ae.scale_chroma(ycc, s), cv2.COLOR_YCrCb2BGR)
        else:
            raise ValueError(f"Unknown night_pipeline step: {step!r}")
    return img


def run_method(name, img, cfg, ctx):
    P = cfg["params"]
    if name in ("night_pipeline", "night_pipeline_dl"):
        return m_night_pipeline(img, P, ctx, block=name)
    if name in DENOISE:
        return DENOISE[name](img, P[name], ctx)
    if name in LOW_LIGHT:
        if cfg["denoise_before_low_light"]:
            img = m_denoise(img, P["denoise"], ctx)
        return LOW_LIGHT[name](img, P[name], ctx)
    if name in GLARE:
        return GLARE[name](img, P[name], ctx)
    raise ValueError(f"Unknown method: {name!r}")


# ----------------------------------------------------------------------
# Measurements
# ----------------------------------------------------------------------
def glare_ring(original, p):
    """Pixels 3-15 px around the light sources of the ORIGINAL frame - where glow hides
    things. Lower brightness there after a method = less glare."""
    _, y = split_luma(original)
    core = light_mask(y, p).astype(np.uint8)
    near = cv2.dilate(core, np.ones((7, 7), np.uint8))
    far = cv2.dilate(core, np.ones((31, 31), np.uint8))
    return (far > 0) & (near == 0)


def metrics(img, ring):
    y = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[..., 0].astype(np.float32)
    return {
        "brightness": float(y.mean()),
        "near_black_pct": float((y < 20).mean() * 100),
        "clipped_white_pct": float((y >= 250).mean() * 100),
        "noise": ae.noise_sigma(y),
        "glare_ring": float(y[ring].mean()) if ring.any() else float("nan"),
        "sharpness": float(cv2.Laplacian(y, cv2.CV_32F).var()),
    }


def grid(tiles, names, cols, labels):
    h, w = tiles[0].shape[:2]
    rows = []
    for i in range(0, len(tiles), cols):
        row = [ae.labelled(t, n) if labels else t for t, n in zip(tiles[i:i + cols], names[i:i + cols])]
        row += [np.zeros((h, w, 3), np.uint8)] * (cols - len(row))
        rows.append(np.hstack(row))
    return np.vstack(rows)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "night_config.yaml"))
    ap.add_argument("--input", help="overrides general.input_dir")
    ap.add_argument("--out", help="overrides general.output_dir")
    ap.add_argument("--limit", type=int, help="overrides general.limit")
    ap.add_argument("--method", help="run only this method")
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
    methods = [args.method] if args.method else cfg["methods"]

    files = ae.list_images(g)
    if not files:
        print(f"No images found in {g['input_dir']}")
        return

    out_dir = Path(g["output_dir"]) / (Path(g["input_dir"]).resolve().name + "_night")
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Grid order (left to right, top to bottom): original, " + ", ".join(methods))

    stack = np.vstack if g["layout"] == "stacked" else np.hstack
    rows, timing = [], {m: 0.0 for m in methods}
    for path in files:
        img = cv2.imread(str(path))
        if img is None:
            print(f"Could not read: {path.name}")
            continue
        small = ae.resize(img, cfg["resize"])
        _, y = split_luma(small)
        ctx = {"noise_sigma": ae.noise_sigma(y * 255.0), "models_dir": cfg["models_dir"]}
        ring = glare_ring(small, cfg["params"]["glare_suppress"])
        rel = path.relative_to(g["input_dir"])

        rows.append({"file": str(rel), "method": "original", **metrics(small, ring)})
        tiles = [small]
        for name in methods:
            t = time.time()
            res = run_method(name, small, cfg, ctx)
            timing[name] += time.time() - t
            tiles.append(res)
            rows.append({"file": str(rel), "method": name, **metrics(res, ring)})
            d = out_dir / name / rel.parent
            d.mkdir(parents=True, exist_ok=True)
            ae.save_image(d / rel.name, stack([small, res]) if g["write_original"] else res, g)

        if g["save_grid"]:
            d = out_dir / "_all_methods" / rel.parent
            d.mkdir(parents=True, exist_ok=True)
            ae.save_image(d / rel.name, grid(tiles, ["original"] + methods, g["grid_columns"], g["labels"]), g)
        print(f"Done: {rel}")

    if not rows:
        return
    if g["save_report"]:
        with open(out_dir / "_report.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in rows)

    n = len(files)
    keys = ["brightness", "near_black_pct", "clipped_white_pct", "noise", "glare_ring", "sharpness"]
    print(f"\nAverages over {n} image(s)  (glare_ring = brightness around lights, lower = less glow)")
    print(f"{'method':18}" + "".join(f"{k:>18}" for k in keys) + f"{'ms/img':>9}")
    for name in ["original"] + methods:
        sel = [r for r in rows if r["method"] == name]
        vals = [np.nanmean([r[k] for r in sel]) for k in keys]
        ms = timing.get(name, 0.0) / max(n, 1) * 1000
        print(f"{name:18}" + "".join(f"{v:18.2f}" for v in vals) + f"{ms:9.0f}")
    print(f"\nOutput: {out_dir}")


if __name__ == "__main__":
    main()
