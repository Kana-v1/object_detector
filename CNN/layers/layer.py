import numpy as np
from abc import abstractmethod


class Layer:
    @abstractmethod
    def forward(self, input: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, input: np.ndarray) -> np.ndarray:
        pass
