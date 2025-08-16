import numpy as np
from abc import ABC, abstractmethod

class FourierOperator(ABC):
    def __init__(self, dx):
        self.dx = dx

    @abstractmethod
    def operator(self, k: np.ndarray) -> np.ndarray:
        """
        Return the spectral multiplier for a given wavevector array k.
        """
        pass

    def apply(self, u: np.ndarray) -> np.ndarray:
        n = u.size
        k = 2 * np.pi * np.fft.fftfreq(n, d=self.dx)
        op = self.operator(k)
        arr_hat = np.fft.fft(u)
        arr_hat_op = arr_hat * op
        return np.fft.ifft(arr_hat_op)


class Derivative(FourierOperator):
    def __init__(self, dx, order=1):
        super().__init__(dx)
        self.order = order

    def operator(self, k):
        return (1j * k) ** self.order


class HilbertTransform(FourierOperator):
    def operator(self, k):
        return -1j * np.sign(k)
    
class Translation(FourierOperator):
    def __init__(self, dx, offset = 0):
        super().__init__(dx)
        self.offset = offset
    def operator(self, k: np.ndarray) -> np.ndarray:
        return np.exp(-1j * k * self.offset)
    