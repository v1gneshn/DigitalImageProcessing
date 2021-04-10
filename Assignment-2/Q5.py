import numpy as np
import cv2 as cv
from fft2 import fft2


lena = cv.imread('./images/Lena.png', cv.IMREAD_GRAYSCALE)
dog = cv.imread('./images/dog.tif', cv.IMREAD_GRAYSCALE)
(rows, cols) = dog.shape

dog = cv.resize(dog, None, fx=0.5689, fy=0.5689)
dog = np.pad(dog, [(0, 0), (0, 512-dog.shape[1])], mode='constant')

lenaFFT = np.fft.fftshift(fft2(lena))
dogFFT = np.fft.fftshift(fft2(dog))

lenaMag = np.absolute(lenaFFT)
lenaMag = 255*(lenaMag/np.amax(lenaMag))
lenaMag = 10*np.log10(1+lenaMag)
lenaPha = np.angle(lenaFFT, deg=True)

dogMag = np.absolute(dogFFT)
dogMag = 255*(dogMag/np.amax(dogMag))
dogMag = 10*np.log10(1+dogMag)
dogPha = np.angle(dogFFT, deg=True)

cv.namedWindow('Lena (Magnitude Spectrum)', cv.WINDOW_AUTOSIZE)
cv.imshow('Lena (Magnitude Spectrum)', lenaMag)
cv.namedWindow('Lena (Phase Spectrum)', cv.WINDOW_AUTOSIZE)
cv.imshow('Lena (Phase Spectrum)', lenaPha)

cv.namedWindow('Dog (Magnitude Spectrum)', cv.WINDOW_AUTOSIZE)
cv.imshow('Dog (Magnitude Spectrum)', dogMag)
cv.namedWindow('Dog (Phase Spectrum)', cv.WINDOW_AUTOSIZE)
cv.imshow('Dog (Phase Spectrum)', dogPha)

cv.waitKey(0)
