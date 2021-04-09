import numpy as np
from fft import fft


def fft2(x):
    x = np.asarray(x, dtype='complex128')
    rows, cols = x.shape

    for i in range(rows):
        x[i, 0:] = fft(x[i, 0:])
    for i in range(cols):
        x[0:, i] = fft(x[0:, i])

    return x
