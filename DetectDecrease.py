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
import matplotlib.pyplot as plt

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
    
def test_threshold_detector():

    # --- 1. Tworzymy sztuczne NDVI ---
    rng = np.random.default_rng(42)

    ndvi = rng.uniform(0.4, 0.9, (200, 200))  # zdrowa roślinność

    # --- 2. Dodajemy "uszkodzenia" ---
    ndvi[80:120, 80:120] -= 0.5   # silna degradacja
    ndvi[30:50, 150:180] -= 0.3   # mniejsza degradacja

    # --- 3. Dodajemy NaNy ---
    ndvi[0:20, 0:20] = np.nan

    # bezpieczeństwo zakresu NDVI
    ndvi = np.clip(ndvi, -1.0, 1.0)

    # --- 4. Uruchamiamy detector ---
    detector = Detector(
        DetectorMethod(method="threshold", cfg={"k": 2.0})
    )

    mask = detector.apply(ndvi)

    # --- 5. Debug info ---
    valid = ndvi[~np.isnan(ndvi)]
    threshold = valid.mean() - 2.0 * valid.std()

    print(f"Mean NDVI: {valid.mean():.3f}")
    print(f"Std NDVI: {valid.std():.3f}")
    print(f"Threshold: {threshold:.3f}")
    print(f"Detected pixels: {mask.sum()}")

    # --- 6. Wizualizacja ---
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("NDVI")
    plt.imshow(np.where(np.isnan(ndvi), -999, ndvi),
               cmap="RdYlGn", vmin=-1, vmax=1)

    plt.subplot(1, 3, 2)
    plt.title("Mask (threshold)")
    plt.imshow(mask, cmap="gray")

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    overlay = np.where(mask, 1, 0)
    plt.imshow(np.where(np.isnan(ndvi), -999, ndvi),
               cmap="RdYlGn", vmin=-1, vmax=1)
    plt.imshow(overlay, cmap="Reds", alpha=0.4)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_threshold_detector()