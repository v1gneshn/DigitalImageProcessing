import numpy as np
import cv2 as cv
from fft2 import fft2
from idft2 import idft2
from helpers import plotSpectrum


lena = cv.imread('./images/Lena.png', cv.IMREAD_GRAYSCALE)
dog = cv.imread('./images/dog.tif', cv.IMREAD_GRAYSCALE)
dog = cv.resize(dog, None, fx=0.5689, fy=0.5689)
dog = np.pad(dog, [(0, 0), (0, 512-dog.shape[1])], mode='constant')


lenaFFT = fft2(lena)
mag = np.absolute(lenaFFT)
phase = np.angle(lenaFFT)

fft = np.multiply(mag, np.exp(1j*phase))
img = idft2(fft)
# img = np.fft.ifft2(fft)
cv.namedWindow('Lena', cv.WINDOW_AUTOSIZE)
cv.imshow('Lena', img/255)

cv.waitKey(0)
