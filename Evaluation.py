import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import pathlib as pth
from typing import Optional, Union
import rasterio as rio
from tqdm import tqdm

def binarize_mask(mask: np.ndarray, threshold: float = 0.) -> np.ndarray:
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
        plt.close()
    else:
        plt.show()

def prc_curve(mask: np.ndarray, ground_truth: np.ndarray, path2save: Optional[Union[str, pth.Path]] = None) -> None:
    """Plot Precision-Recall curve of the mask compared to ground truth."""
    from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay

    precision, recall, _ = precision_recall_curve(ground_truth.flatten(), mask.flatten())
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot()
    if path2save is not None:
        path2save = pth.Path(path2save)
        plt.savefig(path2save)
        plt.close()
    else:
        plt.show()

def roc_curve(mask: np.ndarray, ground_truth: np.ndarray, path2save: Optional[Union[str, pth.Path]] = None) -> None:
    """Plot ROC curve of the mask compared to ground truth."""
    from sklearn.metrics import roc_curve, RocCurveDisplay

    fpr, tpr, _ = roc_curve(ground_truth.flatten(), mask.flatten())
    disp = RocCurveDisplay(fpr=fpr, tpr=tpr)
    disp.plot()
    if path2save is not None:
        path2save = pth.Path(path2save)
        plt.savefig(path2save)
        plt.close()
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
        prc_curve(diff_binary, mask, path2save=plots_folder_method / f"{file.stem}_prc_curve.png")
        roc_curve(diff_binary, mask, path2save=plots_folder_method / f"{file.stem}_roc_curve.png")

        with open(plots_folder_method / f"{file.stem}_classification_report.txt", "w") as f:
            f.write(classification_report_str)

def summarize_results(data_path: Union[pth.Path, str]) -> None:
    from collections import defaultdict
    processed = pth.Path(data_path).joinpath("processed")

    reports = list(processed.rglob("*_classification_report.txt"))

    by_method: dict[str, list[dict]] = defaultdict(list)
    for report in reports:
        method = report.parent.name
        field = report.stem.split("_augmented_mask_")[0]

        tiff_dir = report.parent.parent.parent / "tiff"
        mask_file = tiff_dir / f"{field}_augmented_mask_{method}.tif"
        diff_file = tiff_dir / f"{field}_diff.tif"

        with rio.open(mask_file) as ds:
            mask = ds.read(1).astype(bool)
        with rio.open(diff_file) as ds:
            diff = ds.read(1).astype(np.float32)
            if ds.nodata is not None:
                diff[diff == ds.nodata] = np.nan
        gt = binarize_mask(diff)

        from sklearn.metrics import matthews_corrcoef, cohen_kappa_score, f1_score
        y_true, y_pred = gt.flatten(), mask.flatten()
        f1    = f1_score(y_true, y_pred, zero_division=0)
        mcc   = matthews_corrcoef(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        iou   = get_miou(mask, gt)

        by_method[method].append({"field": field, "f1": f1, "mcc": mcc, "kappa": kappa, "iou": iou})

    metrics = ["f1", "mcc", "kappa", "iou"]

    for method, rows in by_method.items():
        n = len(rows)
        xs = list(range(n))
        summary_lines = [f"Method: {method}"]

        for metric in metrics:
            vals = [r[metric] for r in rows]
            step = max(1, n // 10)

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.bar(xs, vals)
            ax.set_title(f"{method} — {metric}")
            ax.set_ylim(-1 if metric in ("mcc", "kappa") else 0, 1)
            ax.set_xlabel("num of files")
            ax.set_xticks(range(0, n, step))
            fig.tight_layout()
            fig.savefig(processed / f"summary_{method}_{metric}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

            summary_lines.append(f"\n{metric}:")
            for i, (field, val) in enumerate(zip([r["field"] for r in rows], vals)):
                summary_lines.append(f"  [{i}] {field}: {val:.3f}")
            summary_lines.append(f"  mean: {np.mean(vals):.3f}")

        (processed / f"summary_{method}.txt").write_text("\n".join(summary_lines))


if __name__ == "__main__":
    evaluate_results("data")
    summarize_results("data")





    

