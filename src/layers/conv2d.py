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
    
    def convolve(a, b, mode = 'valid'):
        num, k_c, k_h, k_w = b.shape

        if mode == 'full':
            a = np.pad(a, ((0,0), (k_h - 1, k_h - 1), (k_w - 1, k_w - 1)), 'constant', constant_values=0)
        
        channel_stride, r_stride, c_stride = a.strides
        c, h, w = a.shape
        
        out_h, out_w = (h - k_h) + 1, (w - k_w) + 1
        new_shape = (c, out_h, out_w, k_h, k_w)
        new_stride = (channel_stride, r_stride, c_stride, r_stride, c_stride)
        
        out = np.lib.stride_tricks.as_strided(a, new_shape, new_stride)
        return np.einsum("chwkt,nckt->nhw", out, b)

    def feed_forward(self, a):
        self.input = a
        height, width, channel = self.kernel_size
        self.out = self.convolve(a, self.kernel, mode="valid")
        return self.activation.activate(self.out)
    
    def backpropagation(self, prev):
        height, width, channel = self.filter_size
        self.error = prev * self.activation.derivative(self.out)
        return self.convolve(self.filter, self.error[::, ::, -1, -1], mode='full')

    def update_gradient(self):
        height, width, channel = self.filter_size

        # Using error dC/dA, we can calculate dC/dF by getting the valid convolution between 
        # the error matrix and the input matrix (No 180 degree rotation)
        self.kernel_gradient += self.convolve(self.input, self.error)
        
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