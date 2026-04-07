# 1. W utils.py (jednak tam, będzie porządek) napisz funkcję do obniżania rozdzielczości, wykorzystującą opencv Sama funkcja jest prosta, ale powinna zawierać logikę która:
# 1.1 Będzie obniżać rozdzielczość o tę samą skalę (np. 2x), którą uznasz za dostateczną,
# 1.2 Zachowa oryginalne proporcje obrazka
# 1.3 Napisz krótką funkcję testową

# utils.py

import numpy as np
import cv2


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


if __name__ == "__main__":
    test_downsample()

# import numpy as np
# import cv2 

# def downsample_img(data: np.ndarray, factor: int) -> np.ndarray:

#     # data: 2D -> data.shape == (height, width)
#     # data: 3D -> data.shape == (channels, height, width)
#     # data 3D -> data.shape == (n_imgs, height, width)

#     if data.ndim == 2:
#         return data[::factor, ::factor]
#     else:
#         return data[..., ::factor, ::factor]

# img0 = np.random.rand(100, 100)
# img1 = cv2.resize(img0, (50, 50))



