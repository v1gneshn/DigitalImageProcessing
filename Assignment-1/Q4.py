# Download Lena image and scale it by factors of 1, 2, 0.5 using bilinear interpolation and display the scaled images.
# Also, display the output of built-in functions for doing scaling by factors of 0.5, 2. Compare the results.

import cv2 as cv
import numpy as np
from math import floor
from helpers import *

X_RESIZE_FACTOR = 0.4
Y_RESIZE_FACTOR = 1

resizedImg = []
img = cv.imread('lena.png', cv.IMREAD_GRAYSCALE)
w, h = img.shape
scale = (floor(w*X_RESIZE_FACTOR), floor(h*Y_RESIZE_FACTOR))

cv.namedWindow(f"Original image{img.shape}", cv.WINDOW_AUTOSIZE)
cv.namedWindow(f"Resized image{scale}", cv.WINDOW_AUTOSIZE)
cv.imshow(f"Original image{img.shape}", img)
cv.imshow(f"Resized image{scale}", np.uint8(bilinearInterpolate(img, scale)))
cv.waitKey(0)
