import numpy as np
import cv2 as cv
from fft2 import fft2

lena = cv.imread('./images/Lena.png', cv.IMREAD_GRAYSCALE)

npFFT2 = np.fft.fft2(lena)
npFFT2 = np.fft.fftshift(np.absolute(npFFT2))
npFFT2 = 255*(npFFT2/np.amax(npFFT2))
npFFT2 = 10*np.log10(1+npFFT2)

myFFT2 = fft2(lena)
myFFT2 = np.fft.fftshift(np.absolute(myFFT2))
myFFT2 = 255*(myFFT2/np.amax(myFFT2))
myFFT2 = 10*np.log10(1+myFFT2)

cv.namedWindow('Numpy FFT on Lena (Magnitude Spectrum)', cv.WINDOW_AUTOSIZE)
cv.imshow('Numpy FFT on Lena (Magnitude Spectrum)', npFFT2)
cv.namedWindow('Our FFT implementation on Lena (Magnitude Spectrum)', cv.WINDOW_AUTOSIZE)
cv.imshow('Our FFT implementation on Lena (Magnitude Spectrum)', myFFT2)
cv.waitKey(0)

print(np.allclose(myFFT2, npFFT2))
