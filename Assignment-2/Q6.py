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
lenaMag = np.absolute(lenaFFT)
lenaPhase = np.angle(lenaFFT)
dogFFT = fft2(dog)
dogMag = np.absolute(dogFFT)
dogPhase = np.angle(dogFFT)
fft = np.multiply(lenaMag, np.exp(1j*dogPhase))

lenaFFT_builtin = np.fft.fft2(lena)
lenaMag_builtin = np.absolute(lenaFFT)
lenaPhase_builtin = np.angle(lenaFFT)
dogFFT_builtin = np.fft.fft2(dog)
dogMag_builtin = np.absolute(dogFFT)
dogPhase_builtin = np.angle(dogFFT)
fft_builtin = np.multiply(lenaMag, np.exp(1j*dogPhase_builtin))

img = idft2(fft)
img_builtin = np.absolute(np.fft.ifft2(fft_builtin))

cv.namedWindow('User defined', cv.WINDOW_AUTOSIZE)
cv.imshow('User defined', img/255)

cv.namedWindow('Builtin', cv.WINDOW_AUTOSIZE)
cv.imshow('Builtin', img_builtin/255)

cv.waitKey(0)
