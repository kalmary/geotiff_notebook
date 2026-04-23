# 1. W utils.py (jednak tam, będzie porządek) napisz funkcję do obniżania rozdzielczości, wykorzystującą opencv Sama funkcja jest prosta, ale powinna zawierać logikę która:
# 1.1 Będzie obniżać rozdzielczość o tę samą skalę (np. 2x), którą uznasz za dostateczną,
# 1.2 Zachowa oryginalne proporcje obrazka
# 1.3 Napisz krótką funkcję testową

# utils.py

import numpy as np
import cv2
import pathlib as pth
import rasterio as rio
from tqdm import tqdm
import tifffile as tiff
import rasterio
from typing import Union



def downsample_image_nan_safe(image: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """
    Downsample image without NaN bias using weighted averaging.

    Args:
        image (np.ndarray): 2D NDVI array
        scale (float): scale factor (0 < scale < 1)

    Returns:
        np.ndarray: downsampled image
    """

    if image.ndim != 2:
        raise ValueError("Only 2D arrays supported")

    if not (0 < scale < 1):
        raise ValueError("Scale must be between 0 and 1")

    h, w = image.shape
    new_w = int(w * scale)
    new_h = int(h * scale)

    # maska validnych pikseli
    valid_mask = ~np.isnan(image)

    # wartości → NaN zastępujemy 0 (ale kontrolujemy wagą)
    values = np.where(valid_mask, image, 0).astype(np.float32)

    weights = valid_mask.astype(np.float32)

    # suma wartości
    value_sum = cv2.resize(
        values,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    # suma wag
    weight_sum = cv2.resize(
        weights,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    # unikamy dzielenia przez 0
    with np.errstate(divide='ignore', invalid='ignore'):
        result = value_sum / weight_sum

    # gdzie nie było żadnych danych → NaN
    result[weight_sum == 0] = np.nan

    # bezpieczeństwo NDVI
    result = np.clip(result, -1.0, 1.0)

    return result

# -------------------
# TEST
# -------------------

def test_downsample():
    import matplotlib.pyplot as plt

    img = np.random.uniform(-1, 1, (500, 500))
    img[100:150, 100:150] = np.nan

    down = downsample_image_nan_safe(img, scale=0.5)

    print("Original:", img.shape)
    print("Downsampled:", down.shape)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(np.where(np.isnan(img), -999, img), cmap="RdYlGn", vmin=-1, vmax=1)

    plt.subplot(1, 2, 2)
    plt.title("Downsampled")
    plt.imshow(np.where(np.isnan(down), -999, down), cmap="RdYlGn", vmin=-1, vmax=1)

    plt.show()

def load_data(source_dir: str, extension: str = "tif", verbose: bool = False):
    source_dir = pth.Path(source_dir)

    if not source_dir.exists():
        raise FileNotFoundError(f"Folder {source_dir} nie istnieje")
    
    pattern = f"*{extension}"

    path_list = list(source_dir.glob(pattern))

    if verbose:
        path_list = tqdm(path_list, total=len(path_list), desc="File iteration")

    # print(path)
    # print(path.resolve())
    # print(list(path.iterdir()))
    for path in path_list:
        dataset = rio.open(path)
        yield dataset, path

def save_tiff(path: Union[str, pth.Path], arrays: dict[str, np.ndarray], transform, crs):
    path = pth.Path(path)
    first = next(iter(arrays.values()))
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=first.shape[0],
        width=first.shape[1],
        count=len(arrays),
        dtype=first.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        for i, (name, arr) in enumerate(arrays.items(), start=1):
            dst.write(arr, i)
            dst.update_tags(i, name=name)