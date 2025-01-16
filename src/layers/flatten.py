import numpy as np
from src.layers.layer import Layer

class Flatten(Layer):
    def feed_forward(self, a):
        self.input = a
        return a.flatten()

    def backpropagation(self, prev):
        return prev.reshape(self.input_size)
    
    @Layer.input_size.setter
    def input_size(self, value):
        self._input_size = value
        self.output_size = np.prod(value)
    