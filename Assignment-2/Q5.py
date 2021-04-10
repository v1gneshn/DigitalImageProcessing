import numpy as np
import cv2 as cv
from fft2 import fft2
from helpers import plotSpectrum


lena = cv.imread('./images/Lena.png', cv.IMREAD_GRAYSCALE)
dog = cv.imread('./images/dog.tif', cv.IMREAD_GRAYSCALE)
(rows, cols) = dog.shape

dog = cv.resize(dog, None, fx=0.5689, fy=0.5689)
dog = np.pad(dog, [(0, 0), (0, 512-dog.shape[1])], mode='constant')

lenaFFT = fft2(lena)
dogFFT = fft2(dog)

plotSpectrum(lenaFFT, 'Lena', True)
plotSpectrum(dogFFT, 'Dog', True)
