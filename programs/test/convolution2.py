import sys
sys.path.insert(1, '../NeuralNetwork')

from src.layers.conv2d import Conv2D
from src.layers.pooling import MaxPooling
import src.activation as act
import numpy as np

layer = Conv2D(5, (2, 2), act.Sigmoid())
layer.input_size = (2, 3, 3)

input = np.array([np.arange(9).reshape(3, 3), np.arange(1, 10).reshape(3, 3)])
res = layer.feed_forward(input)

layer.backpropagation(res)
layer.update_gradient()


max_pooling = MaxPooling((2, 2))
max_pooling.input_size = (2, 4, 3)

input = np.array([np.arange(1, 13).reshape(4, 3), np.arange(12).reshape(4, 3)])
print(input)
print()
print(max_pooling.feed_forward(input))
print()
input = np.array([np.arange(1, 5).reshape(2, 2), np.arange(4).reshape(2, 2)])
print(max_pooling.backpropagation(input))