import numpy as np

import activation


class Layer:
    def __init__(self, activation):
        self.activation = activation
        self.error = np.zeros(self.neuron_size)
        self.out = np.zeros(self.neuron_size)

    def feed_forward(self, a):
        pass
    
    def output_backpropagation(self, expected):
        pass
    
    def backpropagation(self, prev):
        pass
    
    def update_gradient(self, prev):
        pass

    def apply_gradient(self, eta, size):
        pass

    
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

class Flatten:
    def __init__(self):
        pass

class Dense:
    def __init__(self):
        pass

arr = [
    [1, 1, 1, 0, 0, 1],
    [0, 1, 1, 0, 1, 1],
    [0, 0, 1, 0, 0, 1],
    [0, 0, 1, 1, 1, 0],
    [1, 0, 1, 1, 1, 1],
    [0, 0, 1, 0, 1, 1]
]

kernel = [
    [1, 1, 1],
    [0, 1, 1],
    [0, 0, 1]
]

# conv = Conv2D(kernel)
# out = conv.feed_forward(arr)
# print(out)

# pool = Pooling((2, 2), 1)
# print(pool.feed_forward(out))

