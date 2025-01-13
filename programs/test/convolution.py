import numpy as np
from scipy.signal import convolve2d

arr = np.array([
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8]
])

kernel = np.array([
    [0, 1],
    [2, 3]
])

print(kernel[::-1, ::-1])
print(convolve2d(arr, kernel[::-1, ::-1], mode="valid"))