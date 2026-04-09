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
from typing import Optional, Tuple, Literal
from skimage.morphology import remove_small_objects, closing, disk
import rasterio as rio
from dataclasses import dataclass, field


class DetectorMethod:
    method: Literal["threshold"] # + other methods later
    cfg: dict


class Detector:
    def __init__(self, data: np.ndarray):
        self._data = data
        self._mask = None

    

    def _detect_threshold(self, cfg: dict) -> dict:

        valid = self._data[~np.isnan(self._data)]
        threshold = float(valid.mean() - float(cfg["k"]) * valid.std())

        mask = (self._data < threshold) & ~np.isnan(self._data)
        mask = closing(mask, disk(3))
        mask = remove_small_objects(mask, max_size=20)

        return mask

    def _generate_mask(self, method: DetectorMethod) -> np.ndarray:

        dispatch = {
            "threshold": self._detect_threshold
        }

        return dispatch[method.method](method.cfg)

    def apply(self, method: DetectorMethod) -> "Detector":
        self._mask = self._generate_mask(method)

    @property
    def get_mask(self):
        if self._mask is not None:
            return self._mask
        else:
            raise ValueError("No mask could be found. Most likely no detection method was applied.")
        
    @property
    def get_original_data(self):
        return self._data
    
    @property
    def get_masked_data(self):
        if self._mask is not None:
            return self._data[self._mask]
        else:
            raise ValueError("No mask could be found. Most likely no detection method was applied.")


