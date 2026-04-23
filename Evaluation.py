import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import pathlib as pth
from typing import Optional, Union


def binarize_mask(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert a probability mask to binary."""
    return mask > threshold

def get_accuracy(mask: np.ndarray, ground_truth: np.ndarray) -> float:
    """Calculate accuracy of the mask compared to ground truth."""
    return np.mean(mask == ground_truth)

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


def get_precision(mask: np.ndarray, ground_truth: np.ndarray) -> float:
    """Calculate precision of the mask compared to ground truth."""
    cm = get_confusion_matrix(mask, ground_truth)
    return cm[1, 1] / (cm[1, 1] + cm[0, 1])

def get_recall(mask: np.ndarray, ground_truth: np.ndarray) -> float:
    """Calculate recall of the mask compared to ground truth."""
    cm = get_confusion_matrix(mask, ground_truth)
    return cm[1, 1] / (cm[1, 1] + cm[1, 0])



def evaluate_results(data_path: Union[pth.Path, str]) -> None:
    

