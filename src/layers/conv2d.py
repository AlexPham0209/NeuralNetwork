import random
import numpy as np
from cupyx.scipy.signal import convolve2d
from cupyx.scipy.signal import correlate2d
from opt_einsum import contract
import cupy as cp
from src.layers.layer import Layer

class Conv2D(Layer):
    def __init__(self, kernels, kernel_size, activation):
        super().__init__()
        self.activation = activation
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

        # res = cp.zeros((b, self.kernels, self.biases.shape[1], self.biases.shape[2]))
        # for k in range(b):
        #     t = cp.array(self.biases)
        #     for i in range(self.kernels):
        #         for j in range(self._input_size[0]):
        #             t[i, ...] += correlate2d(a[k, j, ...], self.kernel[i, j, ...], mode="valid") 
        #     res[k] = t

        # print("dC/dI: ")
        # print(np.allclose(self.out, t))
        # print()

        return self.activation.activate(self.out)

    def backpropagation(self, prev, eta, size):
        self.error = prev * self.activation.derivative(self.out)
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

        # res = cp.zeros((k_b, self.input_size[0], self.input_size[1], self.input_size[2]))
        # for k in range(k_b):
        #     t = cp.zeros(self.input_size)
        #     for i in range(self.kernels):
        #         for j in range(self.input_size[0]):
        #             t[j] += convolve2d(self.kernel[i, j], self.error[k, i], "full")
        #     res[k] = t

        # other = cp.zeros((k_b, self.input_size[0], self.input_size[1], self.input_size[2]))
        # # for k in range(k_b):
        # #     other[k] = cp.einsum("nchwkt,nkt->chw", delta, flipped_error[k])

        # print("dC/dI: ")
        # print(cp.allclose(res, cp.einsum("nchwkt,bnkt->bchw", delta, flipped_error)))
        # print()
        
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

        # res = cp.zeros(self.kernel.shape)
        # for k in range(b):
        #     o = cp.zeros(self.kernel.shape)
        #     for i in range(self.kernels):
        #         for j in range(self.input_size[0]):
        #             o[i, j] = correlate2d(self.input[k, j], self.error[k, i], "valid")
        #     res += o
            
        # print("dC/dK")
        # print(cp.allclose(delta, res))
        # print()

        self.kernel -= (eta / size) * delta
        self.biases -= (eta / size) * self.error.sum(0)
        
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
        self.output_size = (k_n, out_h, out_w)

        self.biases = cp.random.uniform(low = -1.0, high = 1.0, size = self.output_size)
