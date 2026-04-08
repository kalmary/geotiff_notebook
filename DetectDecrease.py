"""
Detector 1 — Threshold Gate
=============================
Simplest possible detector: flag every pixel whose NDVI drops below
a single threshold derived from the image itself.

No patches, no models, no features. Just:
    threshold = mean(NDVI) - k * std(NDVI)
    damaged   = NDVI < threshold

k is fixed at 2.0 (flag pixels more than 2σ below the field mean).
That's the only "parameter" and it has a direct statistical meaning.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple
from skimage.morphology import remove_small_objects, closing, disk
import rasterio as rio


def detect_threshold(data: np.ndarray, k: float = -10.) -> dict:

    valid = data[~np.isnan(data)]
    threshold = float(valid.mean() - k * valid.std())

    mask = (data < threshold) & ~np.isnan(data)
    mask = closing(mask, disk(3))
    mask = remove_small_objects(mask, max_size=20)

    return mask


if __name__ == "__main__":
    # Example usage
    path = Path("data/processed/czarna 121 2025-04-22-ORTHO-NDVI.data_augmented.tif")
    with rio.open(path) as dataset:
        ndvi = dataset.read(1)
        nodata = dataset.nodata
        ndvi = np.where(ndvi == nodata, np.nan, ndvi)

    damage_mask = detect_threshold(ndvi, 6.)

    plt.figure(figsize=(10, 6))
    # plt.imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
    plt.imshow(damage_mask, cmap="Reds", alpha=0.5)
    plt.title("Detected Damage (Red Overlay)")
    plt.axis("off")
    plt.show()