import numpy as np
from idft import idft


def idft2(x):
    x = np.asarray(x)
    rows, cols = x.shape

    for i in range(cols):
        x[:, i] = idft(x[:, i])
    for j in range(rows):
        x[j, :] = idft(x[j, :])

    return np.absolute(x)
