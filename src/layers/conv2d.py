import random
import numpy as np
from scipy.signal import convolve2d
from src.layers.layer import Layer

class Conv2D(Layer):
    def __init__(self, output_size, kernel_size, activation):
        super().__init__(output_size, activation)

        self.kernel_size = kernel_size
        self.randomize_kernel()
        self.kernel_gradient = np.zeros(self.kernel_size)
        
    def feed_forward(self, a):
        self.out = convolve2d(a, self.kernel[::-1, ::-1], mode='valid')
        return self.activation.activate(self.out)
    
    def backpropagation(self, prev):
        self.error = prev * self.activation.derivative(self.out)  
        return convolve2d(self.filter, self.error, mode="full")

    def update_gradient(self, prev):
        self.kernel_gradient += convolve2d(prev, self.error, mode="valid")
        
    def apply_gradient(self, eta, size = 1):
        self.kernel -= (eta / size) * self.kernel_gradient

        # Resets the error after applying gradient vector
        self.kernel_gradient = np.zeros(self.kernel_size)
    
    def randomize_kernel(self):
        height, width = self.kernel_size
        self.kernel = np.array([[random.uniform(-1.0, 1.0) for j in range(height)] for i in range(width)])

