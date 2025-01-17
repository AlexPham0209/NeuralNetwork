import random
import numpy as np
from scipy.signal import convolve2d
from scipy.signal import correlate2d
from src.layers.layer import Layer

class Conv2D(Layer):
    def __init__(self, kernels, kernel_size, activation):
        super().__init__(activation=activation)

        self.kernel_size = kernel_size
        self.kernels = kernels

    def feed_forward(self, a):
        self.input = a
        
        channel_stride, r_stride, c_stride = a.strides
        c, h, w = a.shape
        num, k_c, k_h, k_w = self.kernel.shape
        
        out_h, out_w = (h - k_h) + 1, (w - k_w) + 1
        new_shape = (c, out_h, out_w, k_h, k_w)
        new_stride = (channel_stride, r_stride, c_stride, r_stride, c_stride)
        
        out = np.lib.stride_tricks.as_strided(a, new_shape, new_stride)
        self.out = np.einsum("chwkt,nckt->nhw", out, self.kernel) + self.biases

        # Naive Implementation

        # res = np.array(self.biases)
        # for i in range(self.kernels):
        #     for j in range(self._input_size[0]):
        #         res[i, ...] += correlate2d(a[j, ...], self.kernel[i, j, ...], mode="valid") 

        # print("dC/dI: ")
        # print(np.allclose(self.out, res))
        # print()

        return self.activation.activate(self.out)
    
    def backpropagation(self, prev):
        self.error = prev * self.activation.derivative(self.out)

        # dC/dA is the convolution between the kernel and the error flipped 180 degrees
        kernel = np.pad(self.kernel, ((0, 0), (0, 0), (1, 1), (1, 1)), 'constant', constant_values=0)
        flipped_error = self.error[:, ::-1, ::-1]

        n, c, h, w = kernel.shape
        num_stride, channel_stride, r_stride, c_stride = kernel.strides
        k_c, k_h, k_w = flipped_error.shape
        
        out_h, out_w = (h - k_h) + 1, (w - k_w) + 1
        new_shape = (n, c, out_h, out_w, k_h, k_w)
        new_stride = (num_stride, channel_stride, r_stride, c_stride, r_stride, c_stride)
        delta = np.lib.stride_tricks.as_strided(kernel, new_shape, new_stride)
        
        # Naive implementation
        # All errors(i) in errors are used to convolve kernel[i] and those get sum together.  For example, if kernel is (3, 2, 3, 3) and 
        # error is (2, 2, 2) then we convolve kernel[i][0], kernel[i][1], and kernel[i][2] by error[i] then add them all together.

        # res = np.zeros(self.input_size)
        # for i in range(self.kernels):
        #     for j in range(self.input_size[0]):
        #         res[j] += convolve2d(self.error[i], self.kernel[i, j], "full")

        # print("dC/dI: ")
        # print(np.allclose(res, np.einsum("nchwkt,nkt->chw", delta, flipped_error)))
        # print()
        
        return np.einsum("nchwkt,nkt->chw", delta, flipped_error)

    def update_gradient(self):
        c, h, w = self.input_size
        channel_stride, r_stride, c_stride = self.input.strides
        k_c, k_h, k_w = self.error.shape
            
        out_h, out_w = (h - k_h) + 1, (w - k_w) + 1
        new_shape = (c, out_h, out_w, k_h, k_w)
        new_stride = (channel_stride, r_stride, c_stride, r_stride, c_stride)
        
        out = np.lib.stride_tricks.as_strided(self.input, new_shape, new_stride)
        delta = np.einsum("nhwkt,ckt->cnhw", out, self.error)

        # Naive implementation
        # We correlate all matrices in input to each error in the errors.  For example, we correlate each matrix in tensor by the first error. 
        # Then, it creates a row where each element is the correlation between input[i, j]/input[j] and error[i].
        
        # res = np.zeros(self.output_size)
        # for i in range(self.kernels):
        #     for j in range(self.input_size[0]):
        #         res[i, j] = correlate2d(self.input[j], self.error[i], "valid")

        # print("dC/dK")
        # print(np.allclose(res, delta))
        # print()
        self.kernel_gradient += delta
        
    def apply_gradient(self, eta, size = 1):
        self.kernel -= (eta / size) * self.kernel_gradient
        self.biases -= (eta / size) * self.error

        # Resets the error after applying gradient vector
        self.kernel_gradient = np.zeros(self.kernel_size)
        self.biases_gradient = np.zeros(self.output_size)
    
    @Layer.input_size.setter
    def input_size(self, value):
        self._input_size = value

        c, h, w = value
        height, width = self.kernel_size
        self.kernel = np.random.uniform(low = -1.0, high = 1.0, size = (self.kernels, c, height, width))
        self.kernel_gradient = np.zeros(self.kernel.shape)

        k_n, k_c, k_h, k_w = self.kernel.shape
        out_h, out_w = (h - k_h) + 1, (w - k_w) + 1
        self.output_size = (k_n, out_h, out_w)

        self.biases = np.random.uniform(low = -1.0, high = 1.0, size = self.output_size)
        self.biases_gradient = np.zeros(self.output_size)
