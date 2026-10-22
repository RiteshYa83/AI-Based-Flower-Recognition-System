"""Generate a small SYNTHETIC flower dataset (drawn shapes, not real photos).

Only for checking that the pipeline runs end-to-end without downloading
anything. Do not use the resulting model for real flowers.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

# name: (BGR petal colour, number of petals)
SPECIES = {
    "demo_red":    ((40, 40, 200), 8),
    "demo_yellow": ((40, 210, 240), 12),
    "demo_purple": ((180, 60, 150), 6),
    "demo_white":  ((235, 235, 235), 10),
    "demo_blue":   ((200, 120, 40), 5),
}


def draw_flower(rng, colour, petals, size=160):
    img = np.full((size, size, 3), rng.integers(60, 120, 3), np.uint8)
    img = cv2.GaussianBlur(img + rng.integers(0, 25, img.shape, dtype=np.uint8), (5, 5), 0)
    cx, cy = size // 2 + rng.integers(-12, 13, 2)
    length = int(rng.integers(30, 52))
    col = np.clip(np.array(colour) + rng.integers(-25, 26, 3), 0, 255).tolist()
    offset = rng.uniform(0, 360)
    for k in range(petals):
        ang = offset + k * 360 / petals
        mx = cx + int(np.cos(np.radians(ang)) * length * 0.6)
        my = cy + int(np.sin(np.radians(ang)) * length * 0.6)
        cv2.ellipse(img, (mx, my), (length // 2, max(6, length // 4)),
                    ang, 0, 360, col, -1)
    cv2.circle(img, (cx, cy), int(length * 0.25), (30, 160, 230), -1)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/demo_flowers")
    ap.add_argument("--per-class", type=int, default=60)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    for name, (colour, petals) in SPECIES.items():
        folder = Path(args.out) / name
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(args.per_class):
            cv2.imwrite(str(folder / f"{i:03d}.jpg"), draw_flower(rng, colour, petals))
    print(f"Wrote {args.per_class * len(SPECIES)} synthetic images to {args.out}")


if __name__ == "__main__":
    main()
