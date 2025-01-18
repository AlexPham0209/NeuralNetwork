import random
import numpy as np
import cupy as cp
from src.layers.layer import Layer

class Dense(Layer):
    def __init__(self, output_size, activation):
        super().__init__()
        self.output_size = output_size
        self.activation = activation
        
        self.error = cp.zeros(self.output_size)
        self.out = cp.zeros(self.output_size)

    def feed_forward(self, a):
        self.input = a
        self.out = self.weights @ a + self.biases[:, cp.newaxis]
        return self.activation.activate(self.out)

    def backpropagation(self, prev, eta, size = 1):
        # Calculate dC/dA for output
        self.error = 2 * (self.activation.activate(self.out) - prev) if not self.next_layer else prev
        
        o = self.activation.derivative(self.out) * self.error
        
        a = cp.repeat(o.T[:, :, cp.newaxis], self.input_size, 2)
        b = cp.repeat(self.input.T[:, cp.newaxis, :], self.output_size, 1)
            
        self.weights -= (eta / size) * (a * b).sum(0)
        self.biases -= o.sum(1)
        
        return self.weights.T @ o
        
    # Randomizes all weights from 0 to 1 
    def randomize_weights(self):
        self.weights = cp.array([[random.uniform(-1.0, 1.0) for j in range(self.input_size)] for i in range(self.output_size)])
            
    # Randomizes all biases from 0 to 1 
    def randomize_biases(self):
        self.biases = cp.array([random.uniform(-1.0, 1.0) for i in range(self.output_size)])
    
    # Setter function that is ran when the 
    @Layer.input_size.setter
    def input_size(self, value):
        if cp.ndim(value) > 0:
            raise Exception("Not a one dimensional input")
        
        self._input_size = value

        # Once we know what the input size is, we create the weights and biases
        self.randomize_weights()
        self.randomize_biases()
