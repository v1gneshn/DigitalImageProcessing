import numpy as np
from fft import fft


x = [1, 2, 3, 4, 5, 6, 7, 8]
print(fft(x))
print(np.fft.fft(x))
print(np.allclose(fft(x), np.fft.fft(x)))
