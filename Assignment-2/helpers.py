import numpy as np
import cv2 as cv


def plotSpectrum(fft, fileName, magOnly=False):
    fft = np.fft.fftshift(fft)

    mag = np.absolute(fft)
    mag = 255*(mag/np.amax(mag))
    mag = 10*np.log10(1+mag)
    cv.namedWindow(f'{fileName} (Magnitude Spectrum)', cv.WINDOW_AUTOSIZE)
    cv.imshow(f'{fileName} (Magnitude Spectrum)', mag)
    if(not(magOnly)):
        pha=np.angle(fft, deg=True)
        cv.namedWindow(f'{fileName} (Phase Spectrum)', cv.WINDOW_AUTOSIZE)
        cv.imshow(f'{fileName} (Phase Spectrum)', pha)
