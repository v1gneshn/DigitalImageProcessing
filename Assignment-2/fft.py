import numpy as np


def fft(x):
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]
    twiddle = np.exp(-2j*np.pi/N)
    if N % 2 > 0:
        raise ValueError("must be a power of 2")
    elif N <= 2:
        n = np.arange(N)
        k = n.reshape((N, 1))
        return np.dot(twiddle**(n*k), x)
    else:
        X_even = fft(x[::2])
        X_odd = fft(x[1::2])
        terms = twiddle**np.arange(N)
        return np.concatenate([X_even + terms[:int(N/2)] * X_odd, X_even + terms[int(N/2):] * X_odd])
