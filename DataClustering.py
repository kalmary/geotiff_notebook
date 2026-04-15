from dataclasses import dataclass, field
from typing import Literal
import numpy as np

@dataclass
class DetectorMethod:
    method: Literal[""]
    cfg: dict = field(default_factory=dict)


class Detector:
    def __init__(self, method: DetectorMethod):
        self._method = method



    def _generate_bbox(self, data: np.ndarray, valid: np.ndarray) -> np.ndarray:
        dispatch = {
        }
        return dispatch[self._method.method](data, valid, self._method.cfg)

    def apply(self, data: np.ndarray) -> np.ndarray:
        valid = data[~np.isnan(data)]
        return self._generate_bbox(data, valid)
    
