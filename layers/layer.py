from enum import Enum
import random
import numpy as np
import activation as act

class Layer:
    def __init__(self, output_size, activation = act.Sigmoid()):
        self.activation = activation
        self.output_size = output_size
        
        #Before and after layers
        self.prev_layer = None
        self.next_layer = None

    def feed_forward(self, a):
        pass
    
    def backpropagation(self, prev):
        pass
    
    def update_gradient(self, prev):
        pass

    def apply_gradient(self, eta, size = 1):
        pass
    
    @property
    def input_size(self):
        return self._input_size

    @input_size.setter
    def input_size(self, value):
        self._input_size = value

