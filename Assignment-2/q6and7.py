# -*- coding: utf-8 -*-
"""Q6and7.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fYPYLlTvwEDTvL3F0abH0pY5Hhf_kmCd
"""

import numpy as np
import cv2 as cv
from fft2 import fft2
from idft import idft
from idft2 import idft2

# Driver Code for Questions 6 and 7 

lena = cv.imread('Lena.png', cv.IMREAD_GRAYSCALE)
dog = cv.imread('dog.tif', cv.IMREAD_GRAYSCALE)

# Resizing Dog to 512x512
(rows, cols) = dog.shape
dog = cv.resize(dog, None, fx=0.5689, fy=0.5689)
dog = np.pad(dog, [(0, 0), (0, 512-dog.shape[1])], mode='constant')

# Performing Fourier transform on dog and lena using user-defined function fft2
lenaFFT = fft2(lena)
dogFFT = fft2(dog)

# exchanging lenaFFT and dogFFT in the follwing command we get different combinations
swapped_img = np.multiply(np.abs(lenaFFT), np.exp(1j*np.angle(dogFFT)))

# Performing inverse fourier transform on swapped_img
# np.fft.ifft for in-built and idft2 for user-defined
imshow(idft2(swapped_img))


cv.waitKey(0)