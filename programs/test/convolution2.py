import sys
sys.path.insert(1, '../NeuralNetwork')

from src.layers.conv2d import Conv2D
import src.activation as act
import numpy as np

layer = Conv2D(5, (2, 2), act.Sigmoid())
layer.input_size = (2, 3, 3)

input = np.array([np.arange(9).reshape(3, 3), np.arange(1, 10).reshape(3, 3)])
res = layer.feed_forward(input)

layer.backpropagation(res)
layer.update_gradient()