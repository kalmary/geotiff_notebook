from dataclasses import dataclass, field
from typing import Literal
import numpy as np
import os
from typing import Optional, Union
import pathlib as pth
import rasterio as rio
from tqdm import tqdm

@dataclass
class ClusterMethod:
    method: Literal[
        "KMeans",
        "MeanShift",
        "AgglomerativeClustering",
        "SpectralClustering",
        "HDBSCAN"]
    cfg: dict = field(default_factory=dict)


class Detector:
    def __init__(self, method: ClusterMethod, n_jobs: int = -1, verbose: bool = False):
        self._method = method
        self.n_jobs = os.cpu_count() if (n_jobs == -1 or n_jobs > os.cpu_count()) else n_jobs


    def find_optimal_k_silhouette(self, data: np.ndarray, model_cls: object, k_range: range) -> int:
        """
        Automated search for the best k.
        Note: Only works for methods requiring n_clusters.
        """
        from sklearn.metrics import silhouette_score

        flat_data = data.reshape(-1, 1) if data.ndim == 1 else data

        def score_for_k(k: int) -> float:
            labels = model_cls(n_clusters=k, **self._method.cfg.get("kwargs", {})).fit_predict(flat_data)
            return silhouette_score(flat_data, labels) if len(set(labels)) > 1 else -1

        best_k = max(k_range, key=score_for_k)
        self._method.cfg["n_clusters"] = best_k

        return best_k

    def _kmeans(self, data: np.ndarray, cfg: dict) -> np.ndarray:
        from sklearn.cluster import KMeans
        cfg["n_clusters"] = self.find_optimal_k_silhouette(data, KMeans, range(1, 5))

        model = KMeans(n_clusters=cfg["n_clusters"], **cfg.get("kwargs", {}))
        labels = model.fit_predict(data)

        return labels
    
    def _meanshift(self, data: np.ndarray, cfg: dict) -> np.ndarray:
        from sklearn.cluster import MeanShift
        cfg["n_jobs"] = self.n_jobs
        model = MeanShift(**cfg.get("kwargs", {}))
        labels = model.fit_predict(data)
        return labels
    
    def _agglomerative(self, data: np.ndarray, cfg: dict) -> np.ndarray:
        from sklearn.cluster import AgglomerativeClustering
        cfg["n_clusters"] = self.find_optimal_k_silhouette(data, AgglomerativeClustering, range(1, 8))
        model = AgglomerativeClustering(n_clusters=cfg["n_clusters"], **cfg.get("kwargs", {}))
        labels = model.fit_predict(data)
        return labels
    
    def _spectral(self, data, cfg: dict) -> np.ndarray:
        from sklearn.cluster import SpectralClustering
        cfg["n_clusters"] = self.find_optimal_k_silhouette(data, SpectralClustering, range(1, 8))
        cfg["n_jobs"] = self.n_jobs
        model = SpectralClustering(n_clusters=cfg["n_clusters"], **cfg.get("kwargs", {}))
        labels = model.fit_predict(data)
        return labels
    
    def _hdbscan(self, data: np.ndarray, cfg: dict) -> np.ndarray:
        from sklearn.cluster import HDBSCAN
        model = HDBSCAN(**cfg.get("kwargs", {}))
        cfg["n_jobs"] = self.n_jobs
        labels = model.fit_predict(data)
        return labels



    def _get_bbox(self, labels: np.ndarray) -> dict[int, tuple]:
        bboxes = {}
        for label in np.unique(labels):
            if label == -1:
                continue
            rows, cols = np.where(labels == label)
            bboxes[label] = (cols.min(), rows.min(), cols.max() - cols.min(), rows.max() - rows.min())
        return bboxes




    def _generate_cluster(self, data: np.ndarray) -> np.ndarray:
        dispatch = {
                "KMeans": self._kmeans,
                "MeanShift": self._meanshift,
                "AgglomerativeClustering": self._agglomerative,
                "SpectralClustering": self._spectral,
                "HDBSCAN": self._hdbscan,
            }
        return dispatch[self._method.method](data, self._method.cfg)
    
    def apply(self, mask: np.ndarray) -> tuple[np.ndarray, dict]:
        rows, cols = np.where(mask)
        xy = np.column_stack([cols, rows])  # (N, 2) XY positions

        labels = np.full(mask.shape, -1, dtype=int)
        labels[rows, cols] = self._generate_cluster(xy).ravel()

        bboxes = self._get_bbox(labels)
        return labels, bboxes

    def apply_patches(self, mask: np.ndarray, patch_size: int = 32) -> tuple[np.ndarray, dict]:
        orig_h, orig_w = mask.shape
        ph = (orig_h + patch_size - 1) // patch_size
        pw = (orig_w + patch_size - 1) // patch_size

        padded = np.zeros((ph * patch_size, pw * patch_size), dtype=bool)
        padded[:orig_h, :orig_w] = mask

        patches = padded.reshape(ph, patch_size, pw, patch_size)
        patch_mask = patches.any(axis=(1, 3))  # (ph, pw) bool grid

        small_labels, _ = self.apply(patch_mask)

        labels = np.repeat(np.repeat(small_labels, patch_size, axis=0), patch_size, axis=1)
        labels = labels[:orig_h, :orig_w]
        labels[~mask] = -1

        bboxes = self._get_bbox(labels)
        return labels, bboxes

    def apply_downsampled(self, mask: np.ndarray, scale: float = 0.25) -> tuple[np.ndarray, dict]:
        import cv2
        orig_h, orig_w = mask.shape
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)

        small_mask = cv2.resize(
            mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST
        ).astype(bool)

        small_labels, _ = self.apply(small_mask)

        labels = cv2.resize(
            small_labels.astype(np.int32), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
        ).astype(int)
        labels[~mask] = -1

        bboxes = self._get_bbox(labels)
        return labels, bboxes

    
def _vis_patches(ndvi: np.ndarray, patch_size: int) -> None:
    import matplotlib.pyplot as plt
    h, w = ndvi.shape
    fig, ax = plt.subplots()
    ax.imshow(np.where(np.isnan(ndvi), -999, ndvi), cmap="RdYlGn", vmin=-1, vmax=1)
    for y in range(0, h, patch_size):
        ax.axhline(y - 0.5, color="blue", linewidth=0.5, alpha=0.6)
    for x in range(0, w, patch_size):
        ax.axvline(x - 0.5, color="blue", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_clusters(data: np.ndarray, labels: np.ndarray, ax=None,  path: Optional[Union[str, pth.Path]]=None) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm
    from matplotlib import colormaps

    fig, ax = plt.subplots() if ax is None else (None, ax)

    unique = [l for l in np.unique(labels) if l != -1]
    cmap = colormaps["tab20"].resampled(len(unique))
    norm = BoundaryNorm(range(len(unique) + 1), cmap.N)

    ax.imshow(np.where(np.isnan(data), -999, data), cmap="RdYlGn", vmin=-1, vmax=1, interpolation="none")

    display = np.full(data.shape, np.nan)
    for i, label in enumerate(unique):
        display[labels == label] = i

    im = ax.imshow(display, cmap=cmap, norm=norm, interpolation="none", alpha=0.6)
    plt.colorbar(im, ax=ax, ticks=range(len(unique)), label="Cluster")
    ax.set_title("NDVI Clusters")

    if fig is not None:
        plt.tight_layout()


        if path is None:
            plt.show()
        else:
            path = pth.Path(path)
            plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()

def plot_bbox(data: np.ndarray, bboxes: dict, ax=None, path: Optional[Union[str, pth.Path]]=None) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots() if ax is None else (None, ax)

    ax.imshow(data, cmap="RdYlGn", vmin=-1, vmax=1, interpolation="none")

    for label, (x, y, w, h) in bboxes.items():
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor="red", facecolor="none")
        ax.add_patch(rect)
        ax.text(x, y - 2, str(label), color="red", fontsize=8)

    ax.set_title("NDVI Bounding Boxes")

    if fig is not None:
        plt.tight_layout()

        if path is None:
            plt.show()
        else:
            path = pth.Path(path)
            plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()





def _load_ndvi(mask_file: pth.Path) -> np.ndarray:
    ndvi_stem = mask_file.stem.split("_mask_")[0]
    ndvi_file = mask_file.parent / f"{ndvi_stem}.tif"
    with rio.open(ndvi_file) as ds:
        ndvi = ds.read(1).astype(np.float32)
        if ds.nodata is not None:
            ndvi[ndvi == ds.nodata] = np.nan
    return ndvi


def cluster_data(path: Optional[Union[str, pth.Path]] = "data") -> None:
    import pathlib as pth
    from utils import load_data
    from tqdm import tqdm
    data_path = pth.Path(path)

    methods = {
        "KMeans": {
            "n_init": "auto", # int if not auto
            "max_iter": 300,
            "tol": 0.0001,
            "random_state": 42,
        },
        "MeanShift": {
            "min_bin_freq": 1,
            "cluster_all": True,
            "max_iter": 300
        },
        "AgglomerativeClustering": {
            "metric": "euclidean",
            "linkage": "ward",
        },
        "SpectralClustering": {
            "n_init": 10,
            "gamma": 1.0,
            "affinity": "rbf",
            "n_neighbors": 10,
        },
        "HDBSCAN": {
            "min_cluster_size": 20,
            "min_samples": None,
            "cluster_selection_epsilon": 0.0,
            "metric": "euclidean",
            "alpha": 1.0,
            "leaf_size": 40
        },
    }
    data_path = pth.Path(data_path).joinpath('processed')

    file_list = list(data_path.rglob('*.tif'))
    file_list = [f for f in file_list if '_mask' in f.name]

    pbar = tqdm(file_list, total=len(file_list), desc="Clustering changes")

    for file in pbar:
        tiff_dir = file.parent
        plots_dir = file.parent.parent.joinpath('plots')

        ndvi = _load_ndvi(file)

        for method, cfg in methods.items():
            detection_method = file.stem.rsplit("_")[-1]
            plots_dir_method = plots_dir / detection_method

            detector = Detector(ClusterMethod(method=method, cfg=cfg))

            with rio.open(file) as dataset:
                mask = dataset.read(1).astype(bool)
                
            labels, bboxes = detector.apply_patches(mask, patch_size=32)
            try:
                plot_clusters(ndvi, labels, path=plots_dir_method / f"{file.stem}_clusters_{method}.png")
                plot_bbox(ndvi, bboxes, path=plots_dir_method / f"{file.stem}_bbox_{method}.png")
            except Exception as e:
                print(f"Error during plotting: {e}")
                print("file:", file, '  method:', method, '  detection method:', detection_method)
                print("labels shape:", labels.shape, "unique labels:", np.unique(labels))
                raise e

def test_clustering():
    path = "data/processed/wrzaca 418 2025-06-26-ORTHO-NDVI.data/tiff/wrzaca 418 2025-06-26-ORTHO-NDVI.data_augmented_mask_threshold-dynamic.tif" # TODO: remember that file must be existing
    # TODO - worst testing, must be done for every detection method
    path = pth.Path(path)
    ndvi = _load_ndvi(path)

    _vis_patches(ndvi, patch_size=16) # check if patch size is satisfactory for the data, if not - change it and check again. for drought and flood huge patches are better, for boars - smaller ones. keep balance


    dataset = rio.open(path)
    mask = dataset.read(1).astype(bool)

    methods = {
        "KMeans": {
            "n_init": "auto", # int if not auto
            "max_iter": 500,
            "tol": 0.0001,
            "random_state": 42,
        },
        "MeanShift": {
            "min_bin_freq": 1,
            "cluster_all": True,
            "max_iter": 500
        },
        "AgglomerativeClustering": {
            "metric": "euclidean",
            "linkage": "average",
        },
        "SpectralClustering": {
            "n_init": 10,
            "affinity": "nearest_neighbors",
            "n_neighbors": 20,
        },
        "HDBSCAN": {
            "min_cluster_size": 3,
            "min_samples": 3,
            "cluster_selection_epsilon": 0.,
            "metric": "euclidean",
            "alpha": 0.5,
            "leaf_size": 40
        },
    }
    curr_method_idx = 4
    detector = Detector(ClusterMethod(method=list(methods.keys())[curr_method_idx], cfg=methods[list(methods.keys())[curr_method_idx]]))
    labels, bboxes = detector.apply_patches(mask.copy(), patch_size=32)

    try:
        plot_clusters(ndvi, labels)
        plot_bbox(ndvi, bboxes)
    except Exception as e:
        print(f"Error during plotting: {e}")
        print("file:", path)
        print("labels shape:", labels.shape, "unique labels:", np.unique(labels))

                

if __name__ == "__main__":
    test_clustering()


        
            


    
