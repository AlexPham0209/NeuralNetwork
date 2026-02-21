import numpy as np
from cupyx.scipy.signal import convolve2d
from cupyx.scipy.signal import correlate2d
import cupy as cp
from opt_einsum import contract

from ph.layers.layer import Layer
import ph.layers.activation as act

import random

class Conv2D(Layer):
    def __init__(self, kernels = 0, kernel_size = (0, 0), data = None):
        super().__init__()
        if data:
            self.load_data(data)
            return
        self.kernel_size = kernel_size
        self.kernels = kernels

    def feed_forward(self, a):
        self.input = a
        
        batch_stride, channel_stride, r_stride, c_stride = a.strides
        b, c, h, w = a.shape
        num, k_c, k_h, k_w = self.kernel.shape
        
        out_h, out_w = (h - k_h) + 1, (w - k_w) + 1
        new_shape = (b, c, out_h, out_w, k_h, k_w)
        new_stride = (batch_stride, channel_stride, r_stride, c_stride, r_stride, c_stride)
        
        out = cp.lib.stride_tricks.as_strided(a, new_shape, new_stride)
        self.out = contract("bchwkt,nckt->bnhw", out, self.kernel) + self.biases
        
        return self.out

    def backpropagation(self, prev, eta, size):
        self.error = prev
            
        self.update_gradients(eta, size)

        # Calculate error for next layer
        flipped_error = self.error[:, :, ::-1, ::-1]
        k_b, k_c, k_h, k_w = flipped_error.shape

        kernel = cp.pad(self.kernel, ((0, 0), (0, 0), (k_h - 1, k_h - 1), (k_w - 1, k_w - 1)), 'constant', constant_values=0)

        n, c, h, w = kernel.shape
        num_stride, channel_stride, r_stride, c_stride = kernel.strides
        
        out_h, out_w = (h - k_h) + 1, (w - k_w) + 1
        new_shape = (n, c, out_h, out_w, k_h, k_w)
        new_stride = (num_stride, channel_stride, r_stride, c_stride, r_stride, c_stride)
        delta = cp.lib.stride_tricks.as_strided(kernel, new_shape, new_stride)
        
        return contract("nchwkt,bnkt->bchw", delta, flipped_error)

    def update_gradients(self, eta, size = 1):
        b, c, h, w = self.input.shape
        batch_stride, channel_stride, r_stride, c_stride = self.input.strides
        k_b, k_c, k_h, k_w = self.error.shape
            
        out_h, out_w = (h - k_h) + 1, (w - k_w) + 1
        new_shape = (b, c, out_h, out_w, k_h, k_w)
        new_stride = (batch_stride, channel_stride, r_stride, c_stride, r_stride, c_stride)
        
        out = cp.lib.stride_tricks.as_strided(self.input, new_shape, new_stride)
        delta = contract("bnhwkt,bckt->cnhw", out, self.error)
        
        self.kernel -= (eta / size) * delta
        self.biases -= (eta / size) * self.error.sum(0)
        
    def save_data(self):
        data = dict()
        data["type"] = "conv2d"
        data["activation"] = str(self.activation)
        data["input_size"] = self.input_size
        data["output_size"] = self.output_size

        data["kernel_amount"] = int(self.kernels)
        data["kernel_size"] = self.kernel_size

        data["kernel"] = self.kernel.tolist()
        data["biases"] = self.biases.tolist()

        return data
    
    def load_data(self, data):
        self.activation = act.create_activation(data["activation"])
        self._input_size = tuple(data["input_size"])
        self.output_size = tuple(data["output_size"])

        self.kernels = data["kernel_amount"]
        self.kernel_size = tuple(data["kernel_size"])

        self.kernel = cp.array(data["kernel"])
        self.biases = cp.array(data["biases"])
    
    @Layer.input_size.setter
    def input_size(self, value):
        if len(value) != 3:
            raise Exception("Not a 3 dimensional input")
        
        self._input_size = value

        c, h, w = value
        height, width = self.kernel_size
        self.kernel = cp.random.uniform(low = -1.0, high = 1.0, size = (self.kernels, c, height, width))

        k_n, k_c, k_h, k_w = self.kernel.shape
        out_h, out_w = (h - k_h) + 1, (w - k_w) + 1
        self.output_size = (self.kernels, out_h, out_w)

        self.biases = cp.random.uniform(low = -1.0, high = 1.0, size = self.output_size)