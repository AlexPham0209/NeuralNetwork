import numpy as np
import cupy as cp
from src.layers.layer import Layer

class Flatten(Layer):
    def feed_forward(self, a):
        self.input = a

        b, c, h, w = a.shape
        return a.reshape(b, self.output_size)

    def backpropagation(self, prev, eta, size = 1):
        b, s = prev.shape
        c, w, h = self.input_size
        return prev.reshape(b, c, w, h)
        
    @Layer.input_size.setter
    def input_size(self, value):
        self._input_size = value
        self.output_size = np.prod(value)
    
    def save_data(self):
        data = dict()
        data["input_size"] = self.input_size
        data["output_size"] = self.output_size

        return data
    
    def load_data(self, data):
        self.input_size = data["input_size"]
        self.output_size = data["output_size"]