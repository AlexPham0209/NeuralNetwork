import numpy as np
import cupy as cp
from src.layers.layer import Layer

class Flatten(Layer):
    def __init__(self, data = None):
        super().__init__()

        if data:
            self.load_data(data)

    def feed_forward(self, a):
        self.input = a

        b, c, h, w = a.shape
        return a.reshape(b, self.output_size)

    def backpropagation(self, prev, eta, size = 1, clipping = (-1, 1)):
        b, s = prev.shape
        c, w, h = self.input_size
        return prev.reshape(b, c, w, h)
            
    def save_data(self):
        data = dict()
        data["type"] = "flatten"
        data["input_size"] = self.input_size
        data["output_size"] = self.output_size

        return data

    def load_data(self, data):
        self._input_size = tuple(data["input_size"])
        self.output_size = data["output_size"]

    @Layer.input_size.setter
    def input_size(self, value):
        self._input_size = value
        self.output_size = np.prod(value)