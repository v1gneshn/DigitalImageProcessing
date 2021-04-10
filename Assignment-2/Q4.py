import numpy as np
import cv2 as cv
from fft2 import fft2
from helpers import plotSpectrum

lena = cv.imread('./images/Lena.png', cv.IMREAD_GRAYSCALE)

npFFT2 = np.fft.fft2(lena)
myFFT2 = fft2(lena)

plotSpectrum(npFFT2, 'Numpy FFT on Lena', True)
plotSpectrum(myFFT2, 'Our FFT implementation on Lena', True)

cv.waitKey(0)

print(np.allclose(myFFT2, npFFT2))
