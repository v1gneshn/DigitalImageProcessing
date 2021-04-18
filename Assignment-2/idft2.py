# -*- coding: utf-8 -*-
"""idft2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fYPYLlTvwEDTvL3F0abH0pY5Hhf_kmCd
"""

import numpy as np

def idft2(x):
    x = np.array(x)
    x_1 = x.T
    fft2_1 = []
    for i in x_1:
      fft2_1.append(idft(i))
    
    fft2_1 = np.array(fft2_1)
    fft2_2_input = np.array(fft2_1.T)
    fft2_2 = []
    for i in fft2_2_input:
      fft2_2.append(idft(i))

    fft2_2 = np.array(fft2_2)

    return fft2_2.real