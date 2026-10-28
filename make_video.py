#!/usr/bin/env python3
"""
make_video.py - one video per view, then all views of the scene side by side.

    python make_video.py path/to/scene

The scene folder holds one sub-folder per view (camera), e.g.
    scene/CAM_FRONT/*.png, scene/CAM_FRONT_LEFT/*.png, ...
Images inside a view are found at any depth and played in natural order
(frame_2 before frame_10).  Output in <video.output_dir>/<scene name>/:
    <view>.mp4         one video per view
    <scene>_all.mp4    every view side by side (video.columns > 0 wraps into rows)
"""

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import yaml


def natural_key(path):
    return [(0, int(t)) if t.isdigit() else (1, t.lower()) for t in re.split(r"(\d+)", path.stem)]


def find_images(folder, extensions):
    exts = {e.lower() for e in extensions}
    return sorted((p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts), key=natural_key)


def fit_height(img, height):
    h, w = img.shape[:2]
    return cv2.resize(img, (max(1, round(w * height / h)), height), interpolation=cv2.INTER_AREA)


def write_video(frames, path, fps):
    """frames: iterable of BGR images (all the same size). Returns the number written."""
    writer, n = None, 0
    for frame in frames:
        if writer is None:
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                     (frame.shape[1], frame.shape[0]))
        writer.write(frame)
        n += 1
    if writer:
        writer.release()
    return n


def view_video(images, path, fps, height):
    """One view: every image scaled to `height`, padded to the widest frame."""
    width = max(fit_height(cv2.imread(str(images[0])), height).shape[1], 1)
    def frames():
        for p in images:
            img = cv2.imread(str(p))
            if img is not None:
                img = fit_height(img, height)
                yield cv2.copyMakeBorder(img, 0, 0, 0, max(0, width - img.shape[1]),
                                         cv2.BORDER_CONSTANT)[:, :width]
    return write_video(frames(), path, fps)


def stack_videos(paths, out_path, fps, columns):
    """Play all view videos together; a view that ends early shows black."""
    caps = [cv2.VideoCapture(str(p)) for p in paths]
    sizes = [(int(c.get(cv2.CAP_PROP_FRAME_WIDTH)), int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))) for c in caps]
    cols = columns or len(caps)

    def frames():
        while True:
            tiles, any_left = [], False
            for cap, (w, h) in zip(caps, sizes):
                ok, frame = cap.read()
                any_left |= ok
                tiles.append(frame if ok else np.zeros((h, w, 3), np.uint8))
            if not any_left:
                return
            rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
            width = max(r.shape[1] for r in rows)
            yield np.vstack([cv2.copyMakeBorder(r, 0, 0, 0, width - r.shape[1], cv2.BORDER_CONSTANT)
                             for r in rows])

    n = write_video(frames(), out_path, fps)
    for c in caps:
        c.release()
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", help="folder with one sub-folder per view")
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    v = cfg["video"]
    scene = Path(args.scene).resolve()
    out_dir = Path(v["output_dir"]) / scene.name
    out_dir.mkdir(parents=True, exist_ok=True)

    views = [d for d in sorted(scene.iterdir()) if d.is_dir()]
    view_paths = []
    for view in views:
        images = find_images(view, cfg["extensions"])
        if not images:
            continue
        path = out_dir / f"{view.name}.mp4"
        n = view_video(images, path, v["fps"], v["height"])
        print(f"{view.name}: {n} frames -> {path.name}")
        view_paths.append(path)
    if not view_paths:
        print(f"No view sub-folders with images in {scene}")
        return

    all_path = out_dir / f"{scene.name}_all.mp4"
    n = stack_videos(view_paths, all_path, v["fps"], v["columns"])
    print(f"All {len(view_paths)} views side by side: {n} frames -> {all_path}")


if __name__ == "__main__":
    main()
