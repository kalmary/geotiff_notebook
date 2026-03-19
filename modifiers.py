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

    def _mask_storm(self, rng):
        """
        Wind lodging: 2–5 parallel stripes across most of the field.
        Direction fully random (0–360°).

        Uses the same warp philosophy as _mask_wild_boar:
        - smoothed noise field shifts the effective stripe boundary per pixel
            → organic, meandering edges instead of perfect Gaussians
        - coarse noise (σ ≈ 10% S) handles large meanders
        - fine noise   (σ ≈ 2.5% S) adds local roughness
        - per-pixel intensity modulation inside each stripe (same as boar)
        - scattered debris pixels masked to the damaged zone
        """
        angle = rng.uniform(0, 2 * np.pi) # angle of wind direction - 0 - 360 deg

        Y, X = np.ogrid[:self.H, :self.W] 
        cx, cy = self._cx_cy(rng)

        along = (X - cx) * np.cos(angle) + (Y - cy) * np.sin(angle)
        perp  = -(X - cx) * np.sin(angle) + (Y - cy) * np.cos(angle)

        # Along-wind taper — soft ends, same Gaussian bell as boar boundary
        diag         = np.sqrt(self.H**2 + self.W**2) / 2
        length       = diag * gauss(rng, 0.85, 0.08, lo=0.65, hi=1.0)
        along_weight = np.exp(-(along / (length * 0.55))**2)

        # Two shared noise fields (built once, reused per stripe)
        # Mirrors boar: single noise field smoothed at patch scale
        noise_coarse = gaussian_filter(rng.standard_normal((self.H, self.W)),
                                    sigma=self.S * 0.10)
        noise_fine   = gaussian_filter(rng.standard_normal((self.H, self.W)),
                                    sigma=self.S * 0.025)

        n_stripes  = max(2, int(gauss(rng, 3, 1, lo=2, hi=5)))
        field_half = gauss(rng, 0.35 * self.S, 0.05 * self.S,
                        lo=0.2 * self.S, hi=0.48 * self.S)
        centers    = np.linspace(-field_half, field_half, n_stripes)
        centers   += rng.normal(0, 0.03 * self.S, size=n_stripes)

        mask = np.zeros((self.H, self.W))
        for c in centers:
            # Stripe half-width — Gaussian sampled, analogous to boar radius r
            r = gauss(rng, 0.04 * self.S, 0.008 * self.S,
                    lo=0.012 * self.S, hi=0.07 * self.S)

            # Warp strengths — same role as boar's warp_strength
            warp_coarse = gauss(rng, 0.6, 0.12, lo=0.2, hi=1.0)
            warp_fine   = gauss(rng, 0.25, 0.08, lo=0.05, hi=0.5)

            # Warped perpendicular distance — directly mirrors boar:
            #   boar:  dist  - noise * r * warp_strength
            #   storm: perp  - noise * r * warp_strength  (1D analogue)
            perp_warped = (perp - c) - (noise_coarse * r * warp_coarse
                                    + noise_fine   * r * warp_fine)

            # Gaussian bell on warped distance — identical formula to boar
            stripe = np.exp(-((perp_warped) / (r * 0.4))**2)

            # Per-pixel intensity modulation — same as boar's intensity_map
            intensity_mod = np.clip(
                1.0 + 0.35 * gaussian_filter(rng.standard_normal((self.H, self.W)),
                                            sigma=self.S * 0.08),
                0.3, 1.5
            )
            mask += stripe * intensity_mod * gauss(rng, 1.0, 0.15, lo=0.5, hi=1.3)

        mask *= along_weight

        # Debris: sparse soil/crop pixels inside the damage zone
        debris = gaussian_filter(
            (rng.random((self.H, self.W)) > gauss(rng, 0.82, 0.05,
                                                lo=0.7, hi=0.93)).astype(float),
            sigma=self.S * 0.008
        )
        mask += debris * (mask / (mask.max() + 1e-9)) * gauss(rng, 0.25, 0.08,
                                                            lo=0.05, hi=0.45)

        return np.clip(mask, 0, 1)

    def storm (self):
        print(self.ndvi_array.shape)
    # burza niszczaca pole poprzez mocne wiatry (?)
    # znisczenie poprzez rozmyty duzy pedzel chyba
    
    def drought (self):
        print(self.ndvi_array)
    # susza suszy 
    # obnizenie calego indeksu na polu, ale moze gradientowo tak, w czescie polnocnej bardziej niz

    
