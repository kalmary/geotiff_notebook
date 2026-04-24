import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import pathlib as pth
from typing import Optional, Union
import rasterio as rio
from tqdm import tqdm

def binarize_mask(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert a probability mask to binary."""
    return mask > threshold

def get_confusion_matrix(mask: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    """Calculate confusion matrix of the mask compared to ground truth."""
    return confusion_matrix(ground_truth.flatten(), mask.flatten())

def plot_confusion_matrix(mask: np.ndarray, ground_truth: np.ndarray, path2save: Optional[Union[str, pth.Path]] = None) -> None:
    """Plot confusion matrix of the mask compared to ground truth."""
    cm = get_confusion_matrix(mask, ground_truth)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    if path2save is not None:
        path2save = pth.Path(path2save)
        plt.savefig(path2save)
    else:
        plt.show()

def get_miou(mask: np.ndarray, ground_truth: np.ndarray) -> float:
    """Calculate mean Intersection over Union (mIoU) of the mask compared to ground truth."""
    cm = get_confusion_matrix(mask, ground_truth)
    intersection = cm[1, 1]
    union = cm[1, 1] + cm[0, 1] + cm[1, 0]
    return intersection / union if union > 0 else 0.0


def evaluate_results(data_path: Union[pth.Path, str]) -> None:
    data_path = pth.Path(data_path)
    data_path = data_path.joinpath("processed")

    # load diff and binarize it
    # load mask
    # compare both and calculate metrics
    tif_files = list(data_path.rglob("*.tif"))
    tif_files = [f for f in tif_files if "_mask" in f.name]

    pbar = tqdm(tif_files, total=len(tif_files), desc="Evaluating files")

    for file in pbar: #iterate by mask files, open diff files too
        tiff_folder = file.parent
        plots_folder = tiff_folder.parent.joinpath("plots")
        method = file.stem.rsplit("_")[-1]

        plots_folder_method = plots_folder / method

        with rio.open(file) as dataset:
            mask = dataset.read(1).astype(bool)
            
        base = file.stem.replace(f"_augmented_mask_{method}", "")
        diff_file = tiff_folder / f"{base}_diff.tif"
        with rio.open(diff_file) as dataset:
            diff_ndvi = dataset.read(1).astype(np.float32)
            if dataset.nodata is not None:
                diff_ndvi[diff_ndvi == dataset.nodata] = np.nan

        diff_binary = binarize_mask(diff_ndvi)


        # binarized diff is ground truth, mask is binary prediction, calculate metrics
        miou = get_miou(diff_binary, mask)
        classification_report_str = classification_report(mask.flatten(), diff_binary.flatten(), zero_division=0)
        plot_confusion_matrix(diff_binary, mask, path2save=plots_folder_method / f"{file.stem}_confusion_matrix.png")

        with open(plots_folder_method / f"{file.stem}_classification_report.txt", "w") as f:
            f.write(classification_report_str)

if __name__ == "__main__":
    evaluate_results("data")





    

