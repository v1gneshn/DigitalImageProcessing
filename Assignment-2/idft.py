import numpy as np
from fft import fft


def idft(x):
    x = np.asarray(x)
    N = x.shape[0]
    x = np.conjugate(np.multiply(x, -1j))
    x = fft(x)
    x = np.conjugate(np.multiply(x, -1j))
    return x/N
