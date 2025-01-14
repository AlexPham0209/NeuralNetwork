import sys
sys.path.insert(1, '../NeuralNetwork')

import numpy as np
from src.layers.flatten import Flatten
from scipy.signal import convolve

arr = np.array([[
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8]
], 
[
    [5, 1, 2],
    [3, 4, 5],
    [6, 7, 8]
]])

kernel = np.array([[
    [0, 1],
    [2, 3]
],
[
    [2, 1],
    [2, 5]
]])
print(arr[:, :, 0])
print(arr[:, :, 1])
arr[:, :, 0] = arr[:, :, 1]
print(arr[:, :, 0])