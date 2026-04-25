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
    method: Literal["threshold", "sauvola", "msglof"]
    cfg: dict = field(default_factory=dict)


class Detector:
    def __init__(self, method: DetectorMethod):
        self._method = method

    def _detect_threshold(self, data: np.ndarray, cfg: dict) -> np.ndarray:
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
    
    def _detect_msglof(self, data: np.ndarray, cfg: dict) -> np.ndarray:
        """
        Multi-Scale Graph-refined Local Outlier Factor (MS-GLOF).

        Per-pixel feature vector: [mean, std] at 3 window scales + gradient
        magnitude + local NDVI rank. LOF run per tile (tile_size x tile_size
        with overlap), scores stitched via Gaussian-weighted blending to avoid
        tile boundary artefacts. Spatial graph post-processing suppresses
        isolated false positives by dropping connected components whose mean
        LOF score or pixel count falls below thresholds.

        Reference: Breunig et al. (2000) LOF: Identifying Density-Based
        Local Outliers. SIGMOD.

        cfg keys:
            tile_size        (int,   default 256)   tile side in pixels
            overlap          (int,   default 32)    overlap between tiles
            n_neighbors      (int,   default 20)    LOF k
            contamination    (float, default 0.1)   expected outlier fraction
            scales           (list,  default [0.02, 0.05, 0.10])
                                                    window sizes as fraction of
                                                    min(H, W)
            min_cluster_size (int,   default 50)    min px per retained component
            min_cluster_score(float, default 1.5)   min mean LOF score per component
        """
        from sklearn.neighbors import LocalOutlierFactor
        from scipy.ndimage import uniform_filter, generic_filter, label
        from scipy.ndimage import sobel


        nan_mask = np.isnan(data)
        filled = np.where(nan_mask, np.nanmean(data), data)

        H, W = filled.shape
        short = min(H, W)

        # --- 1. Multi-scale feature map (H, W, n_features) ---
        scales = cfg["scales"]
        feat_maps = []

        for s in scales:
            win = max(3, int(short * s))
            if win % 2 == 0:
                win += 1
            mu = uniform_filter(filled, size=win)
            # std via E[X²] - E[X]²
            sq_mu = uniform_filter(filled ** 2, size=win)
            std = np.sqrt(np.clip(sq_mu - mu ** 2, 0, None))
            feat_maps.extend([mu, std])

        # gradient magnitude
        gx = sobel(filled, axis=1)
        gy = sobel(filled, axis=0)
        grad = np.hypot(gx, gy)
        feat_maps.append(grad)

        # local rank: percentile of pixel within its neighbourhood
        win_rank = max(3, int(short * scales[1]))
        if win_rank % 2 == 0:
            win_rank += 1

        def _rank(patch):
            c = patch[len(patch) // 2]
            return np.mean(patch <= c)

        rank_map = generic_filter(filled, _rank, size=win_rank)
        feat_maps.append(rank_map)

        features = np.stack(feat_maps, axis=-1)  # (H, W, F)

        # --- 2. Tile-based LOF with Gaussian-weighted blending ---
        tile_size = cfg["tile_size"]
        overlap   = cfg["overlap"]
        n_neighbors   = cfg["n_neighbors"]
        contamination = cfg["contamination"]
        stride = tile_size - overlap

        score_acc  = np.zeros((H, W), dtype=np.float32)
        weight_acc = np.zeros((H, W), dtype=np.float32)

        # Gaussian kernel for overlap blending
        lin = np.linspace(-1, 1, tile_size)
        gx_k, gy_k = np.meshgrid(lin, lin)
        gauss = np.exp(-(gx_k ** 2 + gy_k ** 2) / (2 * 0.5 ** 2)).astype(np.float32)

        ys = list(range(0, H - tile_size + 1, stride))
        xs = list(range(0, W - tile_size + 1, stride))

        if not ys or ys[-1] + tile_size < H:
            ys.append(max(0, H - tile_size))
        if not xs or xs[-1] + tile_size < W:
            xs.append(max(0, W - tile_size))

        for y0 in ys:
            y1 = min(y0 + tile_size, H)
            for x0 in xs:
                x1 = min(x0 + tile_size, W)
                tile_feat = features[y0:y1, x0:x1]          # (th, tw, F)
                th, tw    = tile_feat.shape[:2]
                tile_nan  = nan_mask[y0:y1, x0:x1].ravel()

                X = tile_feat.reshape(-1, tile_feat.shape[-1])
                valid_idx = np.where(~tile_nan)[0]

                if len(valid_idx) < n_neighbors + 1:
                    continue

                X_valid = X[valid_idx]
                lof = LocalOutlierFactor(
                    n_neighbors=n_neighbors,
                    contamination=contamination,
                    novelty=False,
                )
                lof.fit(X_valid)
                # negative_outlier_factor_: more negative = more anomalous
                raw_scores = -lof.negative_outlier_factor_  # (n_valid,)

                tile_scores = np.zeros(th * tw, dtype=np.float32)
                tile_scores[valid_idx] = raw_scores.astype(np.float32)
                tile_scores = tile_scores.reshape(th, tw)

                g = gauss[:th, :tw]
                score_acc[y0:y1, x0:x1]  += tile_scores * g
                weight_acc[y0:y1, x0:x1] += g

        with np.errstate(invalid="ignore", divide="ignore"):
            score_map = np.where(weight_acc > 0, score_acc / weight_acc, 0.0)

        # LOF threshold: contamination quantile
        valid_scores = score_map[~nan_mask]
        threshold = np.quantile(valid_scores, 1.0 - contamination)
        raw_mask = (score_map > threshold) & ~nan_mask

        # --- 3. Spatial graph refinement ---
        min_cluster_size  = cfg["min_cluster_size"]
        min_cluster_score = cfg["min_cluster_score"]

        labeled, n_components = label(raw_mask)
        refined = np.zeros_like(raw_mask)

        for comp_id in range(1, n_components + 1):
            comp = labeled == comp_id
            if comp.sum() < min_cluster_size:
                continue
            mean_score = score_map[comp].mean()
            if mean_score < min_cluster_score:
                continue
            refined |= comp

        return refined


    def _generate_mask(self, data: np.ndarray) -> np.ndarray:
        dispatch = {
            "threshold": self._detect_threshold,
            "sauvola": self._detect_sauvola,
            "msglof": self._detect_msglof
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
        "threshold": {"k": 2.0},
        "sauvola": {"win_frac": 0.05, "k": 0.2, "r": 0.5},
        "msglof": {"tile_size": 256, "overlap": 32, "n_neighbors": 20,
                   "contamination": 0.1, "scales": [0.02, 0.05, 0.10],
                   "min_cluster_size": 50, "min_cluster_score": 1.5}
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
    path = "data/processed/wrzaca 2014 2025-11-05-ORTHO-NDVI.data/tiff/wrzaca 2014 2025-11-05-ORTHO-NDVI.data_augmented.tif"
    path = pth.Path(path)
    dataset = rio.open(path)
    ndvi = dataset.read(1)

    if dataset.nodata is not None:
        ndvi = np.where(ndvi == dataset.nodata, np.nan, ndvi).astype(np.float32)

    curr_method_idx = 2
    methods = {
        "threshold": {"k": 2.2}, # 2.2 does well (poorly as necessary))
        "sauvola": {"win_frac": 0.01, "k": 2.7, "r": 0.4}, # good enough
        "msglof": {"tile_size": 256, "overlap": 32, "n_neighbors": 20,
                   "contamination": 0.1, "scales": [0.02, 0.05, 0.10],
                   "min_cluster_size": 50, "min_cluster_score": 1.5}
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