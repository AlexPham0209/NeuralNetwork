import numpy as np
from layers.layer import Layer


class Pooling(Layer):
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride

    def feed_forward(self, a):
        a = np.array(a)
        height, width = np.shape(a)
        q_y, q_x = self.kernel_size
        out = np.zeros((height // q_y, width // q_x))

        for i in range(0, height, q_y):
            for j in range(0, width, q_x):
                window = a[i : i + q_y, j : j + q_x]
                out[i//q_y][j//q_x] = window.max()

        return out