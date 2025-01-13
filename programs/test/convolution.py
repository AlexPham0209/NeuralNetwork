import sys
sys.path.insert(1, '../NeuralNetwork')

import numpy as np
from src.layers.flatten import Flatten

arr = np.array([
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8]
])

kernel = np.array([
    [0, 1],
    [2, 3]
])

flatten = Flatten()
flatten.input_size = kernel.shape
out = flatten.feed_forward(kernel)
print(out)
print(flatten.backpropagation(out))