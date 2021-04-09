import numpy as np
import cv2 as cv
from fft2 import fft2


lena = cv.imread('./images/Lena.png', cv.IMREAD_GRAYSCALE)
dog = cv.imread('./images/dog.tif', cv.IMREAD_GRAYSCALE)

lenaFFT = np.fft.fftshift(fft2(lena))

lenaMag = np.absolute(lenaFFT)
lenaMag = 255*(lenaMag/np.amax(lenaMag))
lenaMag = 10*np.log10(1+lenaMag)

lenaPha = np.angle(lenaFFT, deg=True)

cv.namedWindow('Lena (Magnitude Spectrum)', cv.WINDOW_AUTOSIZE)
cv.imshow('Lena (Magnitude Spectrum)', lenaMag)
cv.namedWindow('Lena (Phase Spectrum)', cv.WINDOW_AUTOSIZE)
cv.imshow('Lena (Phase Spectrum)', lenaPha)

cv.waitKey(0)
