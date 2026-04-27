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
import pathlib as pth
from typing import Literal, Union
import rasterio as rio
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from utils import save_tiff
from tqdm import tqdm
from Evaluation import binarize_mask

@dataclass
class DetectorMethod:
    method: Literal["threshold", "threshold-dynamic", "sauvola"]
    cfg: dict = field(default_factory=dict)


class Detector:
    def __init__(self, method: DetectorMethod):
        self._method = method

    def _detect_threshold(self, data: np.ndarray, cfg: dict) -> np.ndarray:
        valid = data[~np.isnan(data)]

        return (data < cfg["threshold_val"]) & ~np.isnan(data)

    def _detect_threshold_dynamic(self, data: np.ndarray, cfg: dict) -> np.ndarray:
        from skimage.morphology import remove_small_objects, closing, disk

        valid = data[~np.isnan(data)]
        threshold = float(valid.mean() - float(cfg["k"]) * valid.std())
        mask = (data < threshold) & ~np.isnan(data)
        mask = closing(mask, disk(3))
        mask = remove_small_objects(mask, max_size=20)
        return mask
    
    def _detect_sauvola(self, data: np.ndarray, cfg: dict) -> np.ndarray:
        """
        Sauvola's local thresholding method.

        cfg keys:
            win_frac  (float, default 0.05)  window size as fraction of min(H, W)
            k         (float, default 0.2)   threshold offset
            r         (float, default 0.5)   dynamic range scaling
        """

        from skimage.filters import threshold_sauvola

        nan_mask = np.isnan(data)
        filled = np.where(nan_mask, np.nanmean(data), data)

        win = max(15, int(min(data.shape) * cfg["win_frac"]))
        if win % 2 == 0:
            win += 1

        thresh_map = threshold_sauvola(filled, window_size=win,
                                    k=cfg["k"], r=cfg["r"])
        return (filled < thresh_map) & ~nan_mask
    




    def _generate_mask(self, data: np.ndarray) -> np.ndarray:
        dispatch = {
            "threshold": self._detect_threshold,
            "threshold-dynamic": self._detect_threshold_dynamic,
            "sauvola":   self._detect_sauvola
        }
        return dispatch[self._method.method](data, self._method.cfg)

    def apply(self, data: np.ndarray) -> np.ndarray:
        
        return self._generate_mask(data)

def plot_mask(mask, ndvi, ax=None, path: Union[str, pth.Path] = None):
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(10, 10))

    display_ndvi = np.where(np.isnan(ndvi), -999, ndvi)
    im = ax.imshow(display_ndvi, cmap="RdYlGn", vmin=-1, vmax=1)

    overlay = np.where(mask, 1.0, np.nan)
    ax.imshow(overlay, cmap="Blues", alpha=0.75, vmin=0, vmax=1)

    plt.colorbar(im, ax=ax, label="NDVI")

    if own_fig:
        plt.tight_layout()
        if path is not None:
            fig.savefig(path, dpi=300, bbox_inches="tight")
        else:
            plt.show()
        plt.close(fig)


def detect_decrease(data: Union[str, pth.Path]):
    methods = {
        "threshold": {"threshold_val": 0.25}, # simple global threshold (not adaptive)
        "threshold-dynamic": {"k": 1.5}, # 2.2 does well (poorly as necessary))
        "sauvola": {"win_frac": 0.01, "k": 2.7, "r": 0.4}, # good enough
    }

    data = pth.Path(data).joinpath('processed')

    file_list = list(data.rglob('*.tif'))
    file_list = [f for f in file_list if '_augmented' in f.name and '_mask' not in f.name]

    pbar = tqdm(file_list, total=len(file_list), desc="Processing files")

    for file in pbar:
        tiff_dir = file.parent
        plots_dir = file.parent.parent.joinpath('plots')

        for method, cfg in methods.items():
            plots_dir_method = plots_dir / method
            plots_dir_method.mkdir(parents=True,exist_ok=True)

            detector = Detector(DetectorMethod(method=method, cfg=cfg))

            dataset = rio.open(file)
            ndvi = dataset.read(1)

            if dataset.nodata is not None:
                ndvi = np.where(ndvi == dataset.nodata, np.nan, ndvi).astype(np.float32)

            mask = detector.apply(ndvi)

            save_tiff(
                tiff_dir / f"{file.stem}_mask_{method}.tif",
                {"mask": mask.astype(np.uint8)},
                dataset.transform,
                dataset.crs
            )
            
            plot_mask(mask, ndvi, path=plots_dir_method / f"{file.stem}_mask_{method}.png")


def test_detector():
    # Simple test with single file loaded
    path = "data/processed/wrzaca 418 2025-06-26-ORTHO-NDVI.data/tiff/wrzaca 418 2025-06-26-ORTHO-NDVI.data_augmented.tif" # TODO: remember that file must be existing
    path = "data/processed/wrzaca 1404 2025-06-26-ORTHO-NDVI.data/tiff/wrzaca 1404 2025-06-26-ORTHO-NDVI.data_augmented.tif"
    path = pth.Path(path)
    dataset = rio.open(path)
    ndvi = dataset.read(1)

    if dataset.nodata is not None:
        ndvi = np.where(ndvi == dataset.nodata, np.nan, ndvi).astype(np.float32)

    curr_method_idx = 2
    methods = {
        "threshold": {"threshold_val": 0.25}, # simple global threshold (not adaptive)
        "threshold-dynamic": {"k": 1.5}, # 2.2 does well (poorly as necessary))
        "sauvola": {"win_frac": 0.005, "k": 2.7, "r": 0.25} # good enough
    }

    detector = Detector(DetectorMethod(method=list(methods.keys())[curr_method_idx], cfg=methods[list(methods.keys())[curr_method_idx]]))
    mask = detector.apply(ndvi)

    path_diff = str(path).replace("augmented.tif", "diff.tif")    
    path_diff = pth.Path(path_diff)
    
    dataset_diff = rio.open(path_diff)
    diff = dataset_diff.read(1)

    if dataset_diff.nodata is not None:
        diff = np.where(diff == dataset_diff.nodata, np.nan, diff).astype(np.float32)
    
    diff_binary = binarize_mask(diff, threshold=0.1)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("NDVI with Detected Mask Overlay")
    display_ndvi = np.where(np.isnan(ndvi), -999, ndvi)
    plt.imshow(display_ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
    overlay = np.where(mask, 1.0, np.nan)
    plt.imshow(overlay, cmap="Blues", alpha=0.6, vmin=0, vmax=1)
    plt.colorbar(label="NDVI")
    plt.subplot(1, 2, 2)
    plt.title("Augmented NDVI with Difference Mask Overlay")
    display_ndvi_aug = np.where(np.isnan(ndvi), -999, ndvi)
    plt.imshow(display_ndvi_aug, cmap="RdYlGn", vmin=-1, vmax=1)
    diff_overlay = np.where(diff_binary, 1.0, np.nan)
    plt.imshow(diff_overlay, cmap="Blues", alpha=0.6, vmin=0, vmax=1)
    plt.colorbar(label="NDVI with Difference Mask")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_detector()