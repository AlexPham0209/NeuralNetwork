import numpy as np
from layers.layer import Layer


class Conv2D(Layer):
    def __init__(self, kernel):
        self.kernel = np.array(kernel)
        
    def feed_forward(self, a):
        #a = np.pad(a, 1, mode='constant')
        a = np.array(a)

        input_height, input_width = np.shape(a)
        kernel_height, kernel_width = np.shape(self.kernel)

        width = input_width - kernel_width + 1
        height = input_height - kernel_height + 1

        out = np.zeros((height, width))
        for i in range(width):
            for j in range(height):
                window = a[i : i + kernel_height, j : j + kernel_width]
                out[i][j] = (window * self.kernel).sum()
            
        return out