# DataAugmenter.py
import pathlib as pth
from loader import load_data
from utils import downsample_image_nan_safe
import tifffile as tiff

import numpy as np
import matplotlib.pyplot as plt

from modifiers import NDVIdecreaseSimulator, DegradationEvent
from tqdm import tqdm


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

    methods = {
        0: ("boars", [5, 10]),
        1: ("drought", [1, 2]),
        2: ("flood", [1, 3])
    }

    methods_keys = list(methods.keys())
    key = np.random.choice(methods_keys)
    count = np.random.randint(methods[key][1][0], methods[key][1][1])

    sim1 = NDVIdecreaseSimulator(ndvi.copy())
    sim1.apply(DegradationEvent(cause=methods[key][0], seed=42, count=count, intensity=1.0))


    return sim1.result

def save_augmented_image(image: np.ndarray, filename):
    """
    Save augmented image to file (e.g. for later use or to avoid showing all images at once)
    """

    tiff.imwrite(filename, image.astype(np.float32))

def visualize(results: np.ndarray):
    """
    Show original + augmented images
    """

    plt.figure(figsize=(8,5)) # adding this makes all the figures appear in separate windows, idk why but seems to be working xd
    plt.imshow(np.where(np.isnan(results), -999, results),
            cmap='RdYlGn',
            vmin=-1,
            vmax=1)
    # plt.savefig('ndvi.png', dpi=300) # zapisuje obrazek do pliku, zeby nie tracic jakosci

    plt.show()


def main():
    data_dir = pth.Path("data/")



    for dataset, path in load_data(data_dir / "raw", ".tif", verbose=True):
        data_dir_curr = data_dir.joinpath(f"processed/{path.stem}")
        data_dir_curr.mkdir(parents=True, exist_ok=True)

        ndvi_lowres = process_dataset(dataset, scale=0.5)
        results = augment_ndvi(ndvi_lowres)
        diff = ndvi_lowres - results

        file_name_org = data_dir_curr.joinpath(path.stem + "_org.tif")
        file_name_aug = data_dir_curr.joinpath(path.stem + "_mod.tif")
        file_name_diff = data_dir_curr.joinpath(path.stem + "_diff.tif")

        save_augmented_image(ndvi_lowres, file_name_org)
        save_augmented_image(results, file_name_aug)
        save_augmented_image(diff, file_name_diff)
# tak zeby miec podglad co sie dzieje
        visualize(ndvi_lowres)
        visualize(results)
        visualize(diff)










if __name__ == "__main__":
    main()