from math import *


def bilinearInterpolate(arr, size):
    w, h = size
    resizedArr = []
    intervalX = (len(arr) - 1)/(w - 1)
    intervalY = (len(arr) - 1)/(h - 1)
    for i in range(w):
        row = []
        for j in range(h):
            x = intervalX*i
            y = intervalY*j
            x1, y1 = floor(x), floor(y)
            x2, y2 = ceil(x), ceil(y)
            a, b = arr[x1][y1], arr[x2][y1]
            c, d = arr[x1][y2], arr[x2][y2]
            weightX = x2-x
            weightY = y2-y
            pixel = a*weightX*weightY + b * \
                (1-weightX)*weightY + c*weightX * \
                (1-weightY) + d*(1-weightX)*(1-weightY)
            row.append(floor(pixel))
        resizedArr.append(row)

    return resizedArr
