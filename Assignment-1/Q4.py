import cv2 as cv
import numpy as np
from math import floor
from helpers import *

X_RESIZE_FACTOR = 2
Y_RESIZE_FACTOR = 2

img = cv.imread('lena.png', cv.IMREAD_GRAYSCALE)
resized = cv.resize(img, None, fx=X_RESIZE_FACTOR, fy=Y_RESIZE_FACTOR)
w, h = img.shape
scale = (floor(w*X_RESIZE_FACTOR), floor(h*Y_RESIZE_FACTOR))

cv.namedWindow(f"Original image{img.shape}", cv.WINDOW_AUTOSIZE)
cv.namedWindow(f"Resized image{scale}", cv.WINDOW_AUTOSIZE)
cv.namedWindow(f"Using resize method{resized.shape}", cv.WINDOW_AUTOSIZE)
cv.imshow(f"Original image{img.shape}", img)
cv.imshow(f"Resized image{scale}", np.uint8(bilinearInterpolate(img, scale)))
cv.imshow(f"Using resize method{resized.shape}", resized)
cv.waitKey(0)
