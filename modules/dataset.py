"""Dataset loading and collage construction.

Primary collage: 2x2 Brodatz grid of
    TL = D9  Grass       (1.1.01.tiff)
    TR = D12 Bark        (1.1.02.tiff)
    BL = D29 Beach sand  (1.1.07.tiff)
    BR = D94 Brick wall  (1.1.12.tiff)

Also supports loading USC texmos pre-built mosaics and EuroSAT patches
(for the satellite extension).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image


BRODATZ_COLLAGE_FILES = [
    ("1.1.01.tiff", "Grass (D9)"),
    ("1.1.02.tiff", "Bark (D12)"),
    ("1.1.07.tiff", "Beach sand (D29)"),
    ("1.1.12.tiff", "Brick wall (D94)"),
]


def load_brodatz(tex_dir: Path, filename: str) -> np.ndarray:
    """Load a single Brodatz .tiff as a uint8 grayscale array."""
    path = Path(tex_dir) / filename
    img = Image.open(path)
    if img.mode != "L":
        img = img.convert("L")
    return np.asarray(img, dtype=np.uint8)


def build_collage(tex_dir: Path, tile_size: int = 256) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build the 2x2 Brodatz collage and its ground-truth mask.

    Returns
    -------
    collage : (2*tile_size, 2*tile_size) uint8 image.
    gt_mask : same shape, values {0, 1, 2, 3} for TL, TR, BL, BR.
    labels  : length-4 list of human-readable texture names matching {0..3}.
    """
    tex_dir = Path(tex_dir)
    tiles = []
    labels = []
    for fname, name in BRODATZ_COLLAGE_FILES:
        img = load_brodatz(tex_dir, fname)
        if img.shape[0] < tile_size or img.shape[1] < tile_size:
            raise ValueError(f"{fname} is {img.shape}, smaller than tile_size={tile_size}")
        tiles.append(img[:tile_size, :tile_size])
        labels.append(name)

    tl, tr, bl, br = tiles
    top = np.concatenate([tl, tr], axis=1)
    bottom = np.concatenate([bl, br], axis=1)
    collage = np.concatenate([top, bottom], axis=0).astype(np.uint8)

    gt = np.zeros((2 * tile_size, 2 * tile_size), dtype=np.uint8)
    gt[:tile_size,  :tile_size]  = 0
    gt[:tile_size,  tile_size:]  = 1
    gt[tile_size:,  :tile_size]  = 2
    gt[tile_size:,  tile_size:]  = 3

    return collage, gt, labels


def load_texmos(tex_dir: Path, which: str = "texmos2") -> np.ndarray:
    """Load a USC pre-built mosaic (texmos1/2/3). Returns uint8 array."""
    tex_dir = Path(tex_dir)
    candidates = [f"{which}.p512.tiff", f"{which}.p512", f"{which}.tiff"]
    for c in candidates:
        p = tex_dir / c
        if p.exists():
            img = Image.open(p)
            if img.mode != "L":
                img = img.convert("L")
            return np.asarray(img, dtype=np.uint8)
    raise FileNotFoundError(f"No texmos file for '{which}' under {tex_dir}")


def load_eurosat_collage(
    eurosat_dir: Path,
    classes: list[str],
    tile: int = 256,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build a 2x2 collage from 4 EuroSAT classes.

    Each 64x64 patch is upsampled (nearest-neighbour) to `tile` so the final
    collage is 2*tile on a side with ground-truth label per quadrant.
    """
    if len(classes) != 4:
        raise ValueError("Need exactly 4 classes for a 2x2 collage.")
    eurosat_dir = Path(eurosat_dir)
    rng = np.random.default_rng(seed)
    tiles = []
    for cls in classes:
        folder = eurosat_dir / cls
        jpgs = sorted(folder.glob("*.jpg"))
        if not jpgs:
            raise FileNotFoundError(f"No .jpg files in {folder}")
        pick = jpgs[rng.integers(0, len(jpgs))]
        img = Image.open(pick).convert("L").resize((tile, tile), Image.NEAREST)
        tiles.append(np.asarray(img, dtype=np.uint8))

    tl, tr, bl, br = tiles
    collage = np.concatenate(
        [np.concatenate([tl, tr], axis=1),
         np.concatenate([bl, br], axis=1)], axis=0
    )
    gt = np.zeros((2 * tile, 2 * tile), dtype=np.uint8)
    gt[:tile,  :tile]  = 0
    gt[:tile,  tile:]  = 1
    gt[tile:,  :tile]  = 2
    gt[tile:,  tile:]  = 3
    return collage, gt, list(classes)
