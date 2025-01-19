import numpy as np
import cupy as cp
from src.layers.layer import Layer

class Flatten(Layer):
    def feed_forward(self, a):
        self.input = a

        b, c, h, w = a.shape
        return a.reshape(b, self.output_size).T

    def backpropagation(self, prev, eta, size = 1):
        s, b = prev.shape
        c, w, h = self.input_size
        return prev.T.reshape(b, c, w, h)
        
    @Layer.input_size.setter
    def input_size(self, value):
        self._input_size = value
        self.output_size = np.prod(value)
        # print(self.output_size)
    