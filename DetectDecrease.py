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
from typing import Optional, Tuple, Literal
from skimage.morphology import remove_small_objects, closing, disk
import rasterio as rio
from dataclasses import dataclass, field

@dataclass
class DetectorMethod:
    method: Literal["threshold"]
    cfg: dict = field(default_factory=dict)


class Detector:
    def __init__(self, method: DetectorMethod):
        self._method = method

    def _detect_threshold(self, data: np.ndarray, valid: np.ndarray, cfg: dict) -> np.ndarray:
        threshold = float(valid.mean() - float(cfg["k"]) * valid.std())
        mask = (data < threshold) & ~np.isnan(data)
        # mask = closing(mask, disk(3))
        # mask = remove_small_objects(mask, max_size=20)
        return mask

    def _generate_mask(self, data: np.ndarray, valid: np.ndarray) -> np.ndarray:
        dispatch = {
            "threshold": self._detect_threshold
        }
        return dispatch[self._method.method](data, valid, self._method.cfg)

    def apply(self, data: np.ndarray) -> np.ndarray:
        valid = data[~np.isnan(data)]
        return self._generate_mask(data, valid)
    


def test_detector():
    import pathlib as pth
    import matplotlib.pyplot as plt

    path = pth.Path("data/processed/czarna 121 2025-04-22-ORTHO-NDVI.data")
    detector = Detector(DetectorMethod(method="threshold", cfg={"k": 0.25}))

    for file in path.glob("*.tif"):
        if "_mod" not in file.name:
            continue

        with rio.open(file) as dataset:
            data = dataset.read(1).astype(float)

        data = np.clip(data, -1.0, 1.0)
        data[data == dataset.nodata] = np.nan  # proper nodata handling if applicable

        mask = detector.apply(data)
        ndvi_display = np.where(np.isnan(data), np.nan, data)
        masked_overlay = np.where(mask, 1.0, np.nan)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for ax in axes:
            ax.imshow(ndvi_display, cmap='RdYlGn', vmin=-1, vmax=1)

        axes[1].imshow(masked_overlay, cmap='Reds', vmin=0, vmax=1, alpha=0.6)
        axes[0].set_title("NDVI")
        axes[1].set_title("Detected damage")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    test_detector()
