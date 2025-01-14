import random
import numpy as np
from scipy.signal import convolve2d
from src.layers.layer import Layer

class Conv2D(Layer):
    def __init__(self, filters, filter_size, activation):
        super().__init__(activation=activation)

        self.filter_size = filter_size
        self.filters = filters
        
        self.randomize_kernel()
        self.kernel_gradient = np.zeros(self.kernel_size)
        
    def feed_forward(self, a):
        height, width, channel = self.kernel_size
        self.out = np.array((height, width, self.kernels))
        
        # Goes through all feature maps/filters
        for i in range(self.kernels):
            res = np.array((height, width))

            # Applies convolution on all channels and adds them together
            for c in range(len(channel)):
                res += convolve2d(a[::, ::, c], self.kernel[i, ::-1, ::-1, c], mode='valid')
                
            self.out[::, ::, i] = res
        
        return self.out
    
    def backpropagation(self, prev):
        height, width, channel = self.filter_size
        self.error = prev * self.activation.derivative(self.out)
        
        delta = np.array((height, width, self.filters))
        # Using the error dC/dA, we can calculate dC/dA by calculating the full convolution between 
        # the input matrix flipped 180 degrees and the error matrix
        # We do this for all filters and all channels 
        for i in range(self.filters):
            res = np.array((height, width))

            for c in range(len(channel)):
                res += convolve2d(self.filter[i, ::, ::, c], self.error[i, ::, ::, c], mode='full')

            delta[::, ::, i] = res
    
        return delta
    
    def update_gradient(self, prev):
        height, width, channel = self.filter_size

        # Using error dC/dA, we can calculate dC/dF by getting the valid convolution between 
        # the error matrix and the input matrix (No 180 degree rotation)
        for i in range(self.filters):
            res = np.array((height, width))
            for c in range(len(channel)):
                self.kernel_gradient[i, :, :, c] += convolve2d(self.prev[i, ::-1, ::-1, c], self.error[i, ::, ::, c], mode='valid')

    def apply_gradient(self, eta, size = 1):
        self.kernel -= (eta / size) * self.kernel_gradient

        # Resets the error after applying gradient vector
        self.kernel_gradient = np.zeros(self.kernel_size)
    
    def randomize_kernel(self):
        height, width, channel = self.kernel_size
        self.filter = np.random.uniform(low = -1.0, high = 1.0, size = (self.kernel, height, width, channel))

    @Layer.input_size.setter
    def input_size(self, value):
        self._input_size = value
        self.output_size = value