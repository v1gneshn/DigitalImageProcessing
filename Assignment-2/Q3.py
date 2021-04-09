import numpy as np
from fft import fft


x = [1, 2, 3, 4, 5, 6, 7, 8]

npFFT = np.fft.fft(x)
myFFT = fft(x)

print(myFFT)
print(npFFT)
print(np.allclose(myFFT, npFFT))
