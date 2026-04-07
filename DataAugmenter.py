# DataAugmenter.py

from loader import load_data
from utils import downsample_image_nan_safe

import numpy as np
import matplotlib.pyplot as plt

from modifiers import NDVIdecreaseSimulator, DegradationEvent


def process_dataset(dataset, scale=0.5):
    """
    Single dataset pipeline:
    load -> downsample -> augment -> return
    """

    ndvi = dataset.read(1)

    # nodata -> NaN
    if dataset.nodata is not None:
        ndvi = np.where(ndvi == dataset.nodata, np.nan, ndvi)

    # downsampling
    ndvi_lowres = downsample_image_nan_safe(ndvi, scale=scale)

    return ndvi_lowres


def augment_ndvi(ndvi: np.ndarray):
    """
    Apply multiple augmentations to one NDVI image
    """

    results = []

    # przykład 1: dziki
    sim1 = NDVIdecreaseSimulator(ndvi.copy())
    sim1.apply(DegradationEvent(cause="boars", seed=42, count=10, intensity=1))
    results.append(("boars", sim1.result))

    # przykład 2: susza
    sim2 = NDVIdecreaseSimulator(ndvi.copy())
    sim2.apply(DegradationEvent(cause="drought", seed=42, count=1, intensity=0.7))
    results.append(("drought", sim2.result))

    # przykład 3: powodz
    sim3 = NDVIdecreaseSimulator(ndvi.copy())
    sim3.apply(DegradationEvent(cause="flood", seed=42, count=3, intensity=0.8))
    results.append(("flood", sim3.result))

    return results


def visualize(results, original):
    """
    Show original + augmented images
    """

    n = len(results) + 1

    plt.figure(figsize=(5 * n, 5))

    # oryginał
    plt.subplot(1, n, 1)
    plt.title("original (downsampled)")
    plt.imshow(np.where(np.isnan(original), -999, original),
               cmap='RdYlGn', vmin=-1, vmax=1)

    # augmentacje
    for i, (name, img) in enumerate(results, start=2):
        plt.subplot(1, n, i)
        plt.title(name)
        plt.imshow(np.where(np.isnan(img), -999, img),
                   cmap='RdYlGn', vmin=-1, vmax=1)

    plt.show()


def main():

    for dataset in load_data("data", ".tif"):

        ndvi_lowres = process_dataset(dataset, scale=0.5)

        results = augment_ndvi(ndvi_lowres)

        visualize(results, ndvi_lowres)


if __name__ == "__main__":
    main()