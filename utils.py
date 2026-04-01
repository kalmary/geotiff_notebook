import numpy as np
import cv2 

def downsample_img(data: np.ndarray, factor: int) -> np.ndarray:

    # data: 2D -> data.shape == (height, width)
    # data: 3D -> data.shape == (channels, height, width)
    # data 3D -> data.shape == (n_imgs, height, width)

    if data.ndim == 2:
        return data[::factor, ::factor]
    else:
        return data[..., ::factor, ::factor]

img0 = np.random.rand(100, 100)
img1 = cv2.resize(img0, (50, 50))



