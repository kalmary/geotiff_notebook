# 3. stworz folderze projektu plik Modifiers.py:
# 3.1 Stwórz w nim klasę NDVIdecrease - niech (póki co) argumentem w metodzie __init__ będzie ndvi_array (typ: np.ndarray)
# 3.2 Klasa ma posiadać metody, które zmodyfikują ndvi_array. Póki co nie będziemy tych metod rozpisywać dokładnie, ale:
# 3.2.1 Przygotuj kilka metod (same nazwy i opis metod, co robią, w komentarzach pod nimi), które będą modyfikować (symulować zniszczenia pól) macierz ndvi.
#       Żeby program nie zgłaszał błędu, każda metoda powinna printować rozmiar ndvi_array.
# 3.3 Zaimportuj klasę do main i po wczytaniu pliku tif wg. punktu 1., użyj każdej metody. 
import numpy as np
from typing import Literal, Optional

# wartosc dodana pracy - nie samo okreslenie spadku wskaznika ndvi ale na tej podstawie szacunek np. procentowego zniszczenia pola
# a co za tym idzie mniejszymi plonami?

@dataclass
class DegradationEvent:
    cause: Literal["boars", "storm", "drought", "flooding"] #praca domowa
    seed: Optional[int] = None
    count: int = 1
    intensity: float = 0.3



class NDVIdecreaseSimulater:
    def __init__ (self, ndvi_array: np.ndarray): 
        self.ndvi_array = ndvi_array

        assert self.ndvi_array.ndim == 2, "ndvi_array must be 2-dimensional"
        assert np.all((self.ndvi_array >= -1) & (self.ndvi_array <= 1)), "ndvi_array values must be between -1 and 1"

        self.original_ndvi = self.ndvi_array.copy()
        self.shape = self.ndvi_array.shape

        self.masks = {}

    def apply_mask (self, event: DegradationEvent):
        rng = np.random.default_rng(event.seed)
        total_mask = np.zeros(self.shape, dtype=float)

        for _ in range(event.count):
            mask = self.generate_mask(event.cause, rng)
            total_mask = np.clip(total_mask, 0, 1)

        reduction = total_mask * event.intensity
        self.ndvi_array = np.clip(self.ndvi_array - reduction, -1, 1)

        return self

    def _generate_mask (self, cause: str, rng: np.random.Generator) -> np.ndarray:
        distapch = {
            "boars": self._generate_boars_mask,
            "storm": self._generate_storm_mask,
            "drought": self._generate_drought_mask,
            "flooding": self._generate_flooding_mask
        }

        return distapch[cause](rng)
    
    def _cx_cy(self, rng: np.random.Generator, margin: float = 0.15) -> tuple[int, int]:
        cx = rng.integers(int(self.shape[1]*margin), int(3*self.shape[1]*(1-margin)))
        cy = rng.integers(int(self.shape[0]*margin), int(3*self.shape[0]*(1-margin)))
        return cx, cy

    def _mask_boars(self, rng: np.random.Generator) -> np.ndarray:
        mask = np.zeros(self.shape, dtype=float)
        cx, cy = self._cx_cy(rng, margin=0.15)

        noise = rng.standard_normal(self.shape)
        noise = gaussian_filter(noise, sigma=min(self.shape)*0.05)

        Y, X = np.ogrid[:self.shape[0], :self.shape[1]]
        r = min(self.shape) * rng.uniform(0.05, 0.15)
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)

        mask = (dist < r + noise * r * 0.5).astype(float)

        return np.clip(gaussian_filter(mask, sigma=1.5), 0, 1)



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

if __name__ == "__main__":
    decrease_simulator = NDVIdecreaseSimulater(np.array([[1, 0.5], [0.5, 0]]))
    decrease_simulator.boars()
    decrease_simulator.storm()
