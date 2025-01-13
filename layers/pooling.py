import numpy as np
from skimage.measure import block_reduce
from layers.layer import Layer


class Pooling(Layer):
    def __init__(self, output_size, kernel_size, activation):
        super.__init__(output_size, activation)
        self.kernel_size = kernel_size

    def feed_forward(self, a):
        self.out = block_reduce(a, self.kernel_size, np.max)
        return self.activation.activate(self.out)

    