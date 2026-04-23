# DataAugmenter.py
import pathlib as pth
from utils import load_data, save_tiff
from utils import downsample_image_nan_safe

import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from dataclasses import dataclass
from typing import Literal, Optional

def gauss(rng: np.random.Generator, mean: float, std: float,
          lo: float = -np.inf, hi: float = np.inf) -> float:
    """
    Draw from N(mean, std) but hard-clamp the result to [lo, hi].
    This lets us say "radius is typically 8px, std=2, never below 3"
    instead of uniform(3, 15) which treats all values equally likely.
    """
    return float(np.clip(rng.normal(mean, std), lo, hi))




@dataclass
class DegradationEvent:
    cause: Literal["boars", "storm", "drought", "flooding"] #praca domowa
    seed: Optional[int] = None
    count: int = 1 # number of occurecnes of this event, e.g. how many boars, how many storms etc.
    intensity: float = 0.3 # mean ndvi decrease caused by this event, between 0 and 1
    intensity_std: float = 0.05 # adds variation at pixel level




class NDVIdecreaseSimulator:

    ALTITUDE_SCALE = 0.5 # multiplier for radius/ size params. Higher we are -> smaller effect we can see

    def __init__ (self, ndvi_array: np.ndarray): 
        self.ndvi_array = ndvi_array

        self.ndvi_array = np.where(
            np.isnan(self.ndvi_array),
            np.nan,
            np.clip(self.ndvi_array, -1.0, 1.0)
        )

        assert self.ndvi_array.ndim == 2, "ndvi_array must be 2-dimensional"
        #assert np.all((self.ndvi_array >= -1) & (self.ndvi_array <= 1)), "ndvi_array values must be between -1 and 1"
        # chat gpt kazal mi zmienic na ponizsze ...
        valid = ~np.isnan(self.ndvi_array)
        assert np.all(
            (self.ndvi_array[valid] >= -1) & (self.ndvi_array[valid] <= 1)
        ), "ndvi_array values must be between -1 and 1"

        self.original_ndvi = self.ndvi_array.copy()
        self.shape = self.ndvi_array.shape

        self.S = min(self.shape[0], self.shape[1]) # to scale our parameters based on resolution (not metric dimensions)

        self.masks: dict[str, np.ndarray] = {}

    def apply(self, event: DegradationEvent) -> "NDVIdecreaseSimulator":
        """
        Apply a degradation event to the NDVI array.
        Args:
            event::DegradationEvent
                Event to apply

        Returns:
            self::NDVIdecreaseSimulator
                Returns itself so calls can be chained

        """

        rng = np.random.default_rng(event.seed) # random generator with fixed seed - guesses are deterministic in another runs
        total_mask = np.zeros(self.shape, dtype=float) # mask with full zeros

        for _ in range(event.count):
            raw = self._generate_mask(event.cause, rng)
            total_mask = np.clip(total_mask + raw, 0, 1)

        label = f"{event.cause} (x{event.count})"
        self.masks[label] = total_mask

        # Intensity with per-pixel Gaussian noise — makes the affected zone
        # non-uniform, mimicking real spectral variation within a damage patch
        intensity_map = rng.normal(event.intensity, event.intensity_std,
                                   size=self.shape)
        intensity_map = np.clip(intensity_map, 0.0, 1.0)

        reduction = total_mask * intensity_map
        self.ndvi_array = np.clip(self.ndvi_array - reduction, -1.0, 1.0)
        return self

    @property
    def result(self) -> np.ndarray:
        return self.ndvi_array

    def _generate_mask(self, cause: str, rng: np.random.Generator) -> np.ndarray:
        """
        Args:
            cause::[str]
                One of decrease ndvi options:boars, storm etc
            rng::Generator
                Random number generator with predefined seed -> returns deterministic results

        Returns:
            mask::ndarray
                2D array of floats between 0 and 1, with 1 indicating the affected area
        """
        dispatch = {
            "boars": self._mask_boars,
            "storm": self._mask_storm,
            "drought": self._mask_drought,
            "flood": self._mask_flooding
        }

        return dispatch[cause](rng)
    
    def _cx_cy(self, rng: np.random.Generator, margin: float = 0.15) -> tuple[int, int]:
        """
        Generate random center coordinates (cx, cy) for an event, ensuring they are not too close to the edges of the NDVI array. The margin parameter defines how far from the edges the center can be, as a fraction of the array dimensions. For example, with margin=0.15, the center will be at least 15% of the width/height away from the edges.


        Args:
            rng::Generator
                Random number generator with predefined seed
            margin::float
                Margin from the edges, as a fraction of the array dimensions (default: 0.15)
        Returns:
            cx, cy::tuple[int, int]
                Random center coordinates
        """

        cx = rng.integers(int(self.shape[1]*margin), int(self.shape[1]*(1-margin)))
        cy = rng.integers(int(self.shape[0]*margin), int(self.shape[0]*(1-margin)))
        return cx, cy

    def _mask_boars(self, rng):
        """
        One call = one boar group with 3–5 individual rooting patches.
        All patches clustered around a shared group centre.
        count in DegradationEvent controls how many independent groups appear.

        Shared noise fields across all patches in the group — same underlying
        soil/texture turbulence for the whole sounder's activity zone.
        """
        # Group centre — all individual patches scatter around this
        group_cx, group_cy = self._cx_cy(rng, margin=0.15)

        # How spread out the rooting spots are within the group
        scatter = gauss(rng, 0.02 * self.S, 0.01 * self.S,
                        lo=0.01 * self.S, hi=0.04 * self.S)

        # Shared noise — same spatial turbulence for all patches in this group
        noise_coarse = gaussian_filter(rng.standard_normal(self.shape),
                                    sigma=self.S * 0.04)
        noise_fine   = gaussian_filter(rng.standard_normal(self.shape),
                                    sigma=self.S * 0.01)

        n_boars = rng.integers(2, 6)  # 3 to 5 inclusive
        mask = np.zeros(self.shape, dtype=np.float32)
        Y, X = np.ogrid[:self.shape[0], :self.shape[1]]

        for _ in range(n_boars):
            cx = int(np.clip(rng.normal(group_cx, scatter),
                            self.shape[1] * 0.05, self.shape[1] * 0.95))
            cy = int(np.clip(rng.normal(group_cy, scatter),
                            self.shape[0] * 0.05, self.shape[0] * 0.95))

            r = gauss(rng, 0.03 * self.S * self.ALTITUDE_SCALE,
                        0.02 * self.S, lo=0.01 * self.S, hi=0.05 * self.S)

            dist = np.sqrt((X - cx)**2 + (Y - cy)**2)

            warp_strength = gauss(rng, 0.5, 0.1, lo=0.2, hi=0.8)
            noise = noise_coarse + 0.3 * noise_fine
            patch = np.exp(-((dist - noise * r * warp_strength) / (r * 0.4))**2)
            mask = np.clip(mask + patch, 0, 1)

        return mask

    # wydaje mi sie, ze w koncu ta burza niestety do wywalenia
    def _mask_storm(self, rng):

        angle = rng.uniform(0, 2*np.pi)

        cx, cy = self._cx_cy(rng, margin=0.1)

        Y, X = np.ogrid[:self.shape[0], :self.shape[1]]

        along = (X - cx) * np.cos(angle) + (Y - cy) * np.sin(angle)
        perp = -(X - cx) * np.sin(angle) + (Y - cy) * np.cos(angle)

        corners = np.array([
            [0 - cx,        0 - cy],
            [self.shape[1] - cx,   0 - cy],
            [0 - cx,        self.shape[0] - cy],
            [self.shape[1] - cx,   self.shape[0] - cy],
        ])

        corners_along = corners[:,0] * np.cos(angle) + corners[:,1] * np.sin(angle)
        diag = float(np.abs(corners_along).max())

        length = gauss(rng, 0.5 * diag, 0.1 * diag, lo=0.3 * diag, hi=0.8 * diag)
        along_weight = np.exp(-(along / (length * 0.55))**2)

        noise_coarse = gaussian_filter(rng.standard_normal(self.shape),
                            sigma=self.S * 0.10 * self.ALTITUDE_SCALE)

        noise_fine   = gaussian_filter(rng.standard_normal(self.shape),
                                    sigma=self.S * 0.025 * self.ALTITUDE_SCALE)

        n_stripes = int(gauss(rng, 3, 1, lo=2, hi=5))

        field_half = gauss(rng, 0.35 * self.S, 0.05 * self.S,
                        lo=0.2 * self.S, hi=0.75 * self.S)

        centers    = np.linspace(-field_half, field_half, n_stripes)
        centers   += rng.normal(0, 0.03 * self.S, size=n_stripes)

        mask = np.zeros(self.shape, dtype=np.float32)
        for c in centers:
            r = gauss(rng, 0.04 * self.S, 0.008 * self.S * self.ALTITUDE_SCALE, lo=0.012 * self.S, hi=0.07 * self.S)

            warp_coarse = gauss(rng, 0.6, 0.12, lo=0.2, hi=1.0)
            warp_fine = gauss(rng, 0.25, 0.08, lo=0.05, hi=0.5)

            perp_warped = (perp - c) - (noise_coarse * r * warp_coarse + noise_fine * r * warp_fine)

            stripe = np.exp(-((perp_warped) / (r * 0.4))**2)

            intensity_mod = np.clip(
                1. + 0.35 * gaussian_filter(rng.standard_normal(self.shape), sigma=self.S * 0.08) * self.ALTITUDE_SCALE,
                0.3, 1.5
            )

            mask += stripe * intensity_mod * gauss(rng, 1.0, 0.15, lo=0.5, hi=1.3)

        mask *= along_weight

        debris = gaussian_filter(
            (rng.random(self.shape) > gauss(rng, 0.82, 0.05,
                                                lo=0.7, hi=0.93)).astype(float),
            sigma=self.S * 0.008 * self.ALTITUDE_SCALE
        )

        mask += debris * (mask/ (mask.max() + 1e-9)) * gauss(rng, 0.25, 0.1, lo=0.08, hi=0.45)

        return np.clip(mask, 0, 1)

    def _mask_drought(self, rng):
        Y, X = np.ogrid[:self.shape[0], :self.shape[1]]
        n_epicentres = int(gauss(rng, 1.5, 0.6, lo=1, hi=3))
        mask = np.zeros((self.shape[0], self.shape[1]), dtype=np.float32)

        for _ in range(n_epicentres):
            cx, cy = self._cx_cy(rng, margin=0.0)

            rx    = gauss(rng, 0.35 * self.shape[1], 0.10 * self.shape[1], lo=0.15 * self.shape[1], hi=0.65 * self.shape[1])
            ry    = gauss(rng, 0.30 * self.shape[0], 0.10 * self.shape[0], lo=0.12 * self.shape[0], hi=0.60 * self.shape[0])
            angle = rng.uniform(0, np.pi)

            dx = (X - cx) * np.cos(angle) + (Y - cy) * np.sin(angle)
            dy = -(X - cx) * np.sin(angle) + (Y - cy) * np.cos(angle)

            blob = np.exp(-(dx**2 / (2 * rx**2) + dy**2 / (2 * ry**2))).astype(np.float32)

            noise_soil = gaussian_filter(rng.standard_normal((self.shape[0], self.shape[1])),
                                        sigma=self.S * 0.18 * self.ALTITUDE_SCALE)
            noise_fine = gaussian_filter(rng.standard_normal((self.shape[0], self.shape[1])),
                                        sigma=self.S * 0.04 * self.ALTITUDE_SCALE)
            modulation = np.clip(1.0 + 0.30 * noise_soil + 0.12 * noise_fine,
                                0.2, 1.8).astype(np.float32)

            mask += blob * modulation

        return np.clip(mask, 0, 1)


    def _mask_flooding(self, rng):
        Y, X = np.ogrid[:self.shape[0], :self.shape[1]]
        n_pools = int(gauss(rng, 1.3, 0.5, lo=1, hi=3))
        mask = np.zeros((self.shape[0], self.shape[1]), dtype=np.float32)

        for _ in range(n_pools):
            cx, cy = self._cx_cy(rng, margin=0.05)

            rx    = gauss(rng, 0.12 * self.shape[1], 0.04 * self.shape[1], lo=0.04 * self.shape[1], hi=0.25 * self.shape[1])
            ry    = gauss(rng, 0.10 * self.shape[0], 0.04 * self.shape[0], lo=0.03 * self.shape[0], hi=0.22 * self.shape[0])
            angle = rng.uniform(0, np.pi)

            dx = (X - cx) * np.cos(angle) + (Y - cy) * np.sin(angle)
            dy = -(X - cx) * np.sin(angle) + (Y - cy) * np.cos(angle)

            blob = np.exp(-(dx**2 / (2 * rx**2) + dy**2 / (2 * ry**2))).astype(np.float32)

            flat_threshold = gauss(rng, 0.35, 0.08, lo=0.15, hi=0.60)
            interior = np.clip((blob - flat_threshold) / (1.0 - flat_threshold), 0, 1)

            noise_shore = gaussian_filter(rng.standard_normal((self.shape[0], self.shape[1])),
                                        sigma=self.S * 0.06 * self.ALTITUDE_SCALE)
            warp_strength = gauss(rng, 0.4, 0.1, lo=0.1, hi=0.7)
            blob_warped = blob - noise_shore * warp_strength * 0.15
            edge_zone = np.clip(np.clip(blob_warped, 0, 1) - interior, 0, 1)

            pool = (interior * gauss(rng, 0.9, 0.05, lo=0.7, hi=1.0) +
                    edge_zone * gauss(rng, 0.55, 0.1, lo=0.3, hi=0.8))

            noise_fine = gaussian_filter(rng.standard_normal((self.shape[0], self.shape[1])),
                                        sigma=self.S * 0.03 * self.ALTITUDE_SCALE)
            mask += pool * np.clip(1.0 + 0.2 * noise_fine, 0.6, 1.4).astype(np.float32)

        return np.clip(mask, 0, 1)


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

    print(f"Applying {methods[key][0]} degradation, count: {count}")
    sim1 = NDVIdecreaseSimulator(ndvi.copy())
    sim1.apply(DegradationEvent(cause=methods[key][0], seed=42, count=count, intensity=1.0))


    return sim1.result, count, methods[key]

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
    import matplotlib.pyplot as plt

    data_dir = pth.Path("data/")



    for dataset, path in load_data(data_dir / "raw", ".tif", verbose=True):
        data_dir_curr = data_dir.joinpath(f"processed/{path.stem}")
        data_dir_curr.mkdir(parents=True, exist_ok=True)

        tiff_dir = data_dir_curr.joinpath("tiff")
        tiff_dir.mkdir(parents=True, exist_ok=True)

        plots_dir = data_dir_curr.joinpath("plots")
        plots_dir.mkdir(parents=True, exist_ok=True)

        ndvi_lowres = process_dataset(dataset, scale=0.5)
        plt.figure(figsize=(8,5)) # adding this makes all the figures appear in separate windows, idk why but seems to be working xd

        results, count, cause = augment_ndvi(ndvi_lowres)
        diff = ndvi_lowres - results

        results = {
            "ndvi": ndvi_lowres,
            "augmented": results,
            "diff": diff}

        
        
        for key, result in results.items():
            file_name = tiff_dir.joinpath(path.stem + f"_{key}.tif")

            save_tiff(file_name,
                  {key: result},
                  dataset.transform,
                  dataset.crs)

            plt_path = plots_dir.joinpath(path.stem + f"_{key}.png")
            plt.figure(figsize=(8,5))
            plt.title(key)
            plt.imshow(np.where(np.isnan(result), -999, result),
                    cmap='RdYlGn',
                    vmin=-1,
                    vmax=1)
            plt.colorbar()
            plt.savefig(plt_path, dpi=300, bbox_inches = "tight")


            # save key to txt
            with open(data_dir_curr.joinpath("degradation.txt"), "w") as f:
                f.write(f"cause: {cause[0]}, count: {count}")





if __name__ == "__main__":
    main()