import numpy as np


def ReLu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def dReLu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, 1, 0)


