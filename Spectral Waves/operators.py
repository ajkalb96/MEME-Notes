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
        if not (hasattr(self, 'n') and u.size == self.n):
            self.n = u.size
            self.k = 2 * np.pi * np.fft.fftfreq(self.n, d=self.dx)
            self.op = self.operator(self.k)
        arr_hat = np.fft.fft(u)
        arr_hat_op = arr_hat * self.op
        return np.fft.ifft(arr_hat_op)


class Derivative(FourierOperator):
    def __init__(self, dx, order=1):
        super().__init__(dx)
        self.order = order

    def operator(self, k):
        return (1j * k) ** self.order

class FilteredDerivative(FourierOperator):
    def __init__(self, dx, order=1, filter=1.0/3):
        super().__init__(dx)
        self.order = order
        self.unfiltered = 1.0 - filter
        self._n_cached = None   # track last n used
        self._op_cached = None  # cached operator

    def operator(self, k: np.ndarray) -> np.ndarray:
        n = k.size
        if self._n_cached != n or self._op_cached is None:
            self._n_cached = n
            kmax = np.max(np.abs(k))
            eta = np.abs(k) / kmax

            # exponential spectral filter
            p = 8  # steepness of filter
            sigma = np.exp(- (eta / self.unfiltered) ** p)

            # cache the operator
            self._op_cached = (1j * k) ** self.order * sigma

        return self._op_cached


class HilbertTransform(FourierOperator):
    def operator(self, k):
        return -1j * np.sign(k)
    
class Translation(FourierOperator):
    def __init__(self, dx, offset = 0):
        super().__init__(dx)
        self.offset = offset
    def operator(self, k: np.ndarray) -> np.ndarray:
        return np.exp(-1j * k * self.offset)
    