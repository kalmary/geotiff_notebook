import numpy as np
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
        assert np.all((self.ndvi_array >= -1) & (self.ndvi_array <= 1)), "ndvi_array values must be between -1 and 1"

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

    def _generate_mask (self, cause: str, rng: np.random.Generator) -> np.ndarray:
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
            "boars": self._mask_boars
            # "storm": self._mask_storm,
            # "drought": self._mask_drought,
            # "flooding": self._mask_flooding
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
        1. pick random center point
        2. generate noise field with shape of full array
        3. blur the noise field - more regular look
        4. use gaussian bell:
        4.1 distance from circle center - noise modified by r and strength creates circle with 0-1 values with irregular edges 


        Args:
            rng::Generator
                Random number generator with predefined seed

        Returns:
            mask::ndarray
                2D array of floats between 0 and 1, with 1 indicating the affected area
        """


        cx, cy = self._cx_cy(rng, margin=0.15) # random circle center, margin avoids edges
        
        r = gauss(rng, 0.04 * self.S * self.ALTITUDE_SCALE,
                       0.01 * self.S, lo=0.02 * self.S, hi=0.07 * self.S) # TODO so far values are made up. lo must be large enough to make decrease visible, hi must be small enough to avoid being huge, 0.04* self.S = 4% of picture size. 0.01* self.S ~ std of radius, so we get some variation in size of holes.

        # Spatial noise field — smoothed so it warps at patch scale, not pixel
        noise = rng.standard_normal(self.shape)
        noise = gaussian_filter(noise, sigma=r * 0.4)

        Y, X = np.ogrid[:self.shape[0], :self.shape[1]]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)

        # Warp boundary: the effective radius varies with noise
        warp_strength = gauss(rng, 0.5, 0.1, lo=0.2, hi=0.8)
        mask = np.exp(-((dist - noise * r * warp_strength) / (r * 0.4))**2)
        return np.clip(mask, 0, 1)



    def boars (self):
        print(self.ndvi_array.ndim)
    # dziki ryjace w polu i niszczace je, mysle, ze na zasadzie pedzla (wielkosc do ustalenia) ktory znaczaco obniza NDVI
    # w danym miejscu + pewnie jakas sciezka od boku pola. do pomyslenia czy jakos symulujemy zachowania stada czy pojedyncze dziki

    def storm (self):
        print(self.ndvi_array.shape)
    # burza niszczaca pole poprzez mocne wiatry (?)
    # znisczenie poprzez rozmyty duzy pedzel chyba
    
    def drought (self):
        print(self.ndvi_array)
    # susza suszy 
    # obnizenie calego indeksu na polu, ale moze gradientowo tak, w czescie polnocnej bardziej niz

    
