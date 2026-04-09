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