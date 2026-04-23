from dataclasses import dataclass, field
from typing import Literal
import numpy as np
import os
from typing import Optional, Union
import pathlib as pth

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
        cfg["n_clusters"] = self.find_optimal_k_silhouette(data, KMeans, range(1, 25))

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
        cfg["n_clusters"] = self.find_optimal_k_silhouette(data, AgglomerativeClustering, range(1, 25))
        model = AgglomerativeClustering(n_clusters=cfg["n_clusters"], **cfg.get("kwargs", {}))
        labels = model.fit_predict(data)
        return labels
    
    def _spectral(self, data, cfg: dict) -> np.ndarray:
        from sklearn.cluster import SpectralClustering
        cfg["n_clusters"] = self.find_optimal_k_silhouette(data, SpectralClustering, range(1, 25))
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
    
    def apply(self, data: np.ndarray) -> tuple[np.ndarray, dict]:
        valid_mask = ~np.isnan(data)
        valid_data = data[valid_mask].reshape(-1, 1)

        labels = np.full(data.shape, -1, dtype=int)
        labels[valid_mask] = self._generate_cluster(valid_data).ravel()

        bboxes = self._get_bbox(labels)
        return labels, bboxes

    
def plot_clusters(data: np.ndarray, labels: np.ndarray, ax=None,  path: Optional[Union[str, pth.Path]]=None) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm
    from matplotlib import colormaps

    fig, ax = plt.subplots() if ax is None else (None, ax)

    unique = [l for l in np.unique(labels) if l != -1]
    cmap = colormaps["tab20"].resampled(len(unique))
    norm = BoundaryNorm(range(len(unique) + 1), cmap.N)

    display = np.full(data.shape, np.nan)
    for i, label in enumerate(unique):
        display[labels == label] = i

    im = ax.imshow(display, cmap=cmap, norm=norm, interpolation="none")
    plt.colorbar(im, ax=ax, ticks=range(len(unique)), label="Cluster")
    ax.set_title("NDVI Clusters")

    if fig is not None:
        plt.tight_layout()

        if path is None:
            plt.show()
        else:
            path = pth.Path(path)
            plt.savefig(path, dpi=300, bbox_inches='tight')

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





def main():
    import pathlib as pth
    from utils import load_data
    from tqdm import tqdm
    data_folder = pth.Path("data/processed")

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
    data = pth.Path(data).joinpath('processed')

    file_list = list(data.rglob('*.tif'))
    file_list = [f for f in file_list if '_mask' in f.name]

    pbar = tqdm(file_list, total=len(file_list), desc="Processing files")

    for file in pbar:
        tiff_dir = file.parent
        plots_dir = file.parent.parent.joinpath('plots')

        for method, cfg in methods.items():

            detector = Detector(ClusterMethod(method=method, cfg=cfg))

            dataset = rio.open(file)
            data = dataset.read(1)

            if dataset.nodata is not None:
                ndvi = np.where(ndvi == dataset.nodata, np.nan, ndvi).astype(np.float32)

            labels, bboxes = detector.apply(data)



            # plot clusters
            plot_clusters(data, labels, path=plots_dir / f"{file.stem}_clusters_{method}.png")
            plot_bbox(data, bboxes, path=plots_dir / f"{file.stem}_bbox_{method}.png")

                

if __name__ == "__main__":
    main()


        
            


    
