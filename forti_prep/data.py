from dataclasses import dataclass

import numpy as np


@dataclass
class Data:
    name: str
    values: np.ndarray
    times: np.ndarray
    units: str


def zeros(based_on: Data, name: str = "anonymous", units: str = "") -> Data:
    return Data(
        "anonymous", np.zeros(based_on.values.shape, np.float32), based_on.times, ""
    )
