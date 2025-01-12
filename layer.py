import numpy as np


class Layer:
    def __init__(self):
        pass

    def feed_forward(self, a):
        return a
    
    def output_backpropagation(self, actual, expected):
        return None
    
    def backpropagation():
        return 0
    
    
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

class Pooling:
    def __init__(self):
        pass

class Flatten:
    def __init__(self):
        pass

class Dense:
    def __init__(self):
        pass

arr = [
    [3, 0, 1, 2, 7, 4],
    [1, 5, 8, 9, 3, 1],
    [2, 7, 2, 5, 1, 3],
    [0, 1, 3, 1, 7, 8],
    [4, 2, 1, 6, 2, 8],
    [2, 4, 5, 2, 3, 9]
]

kernel = [
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
]

conv = Conv2D(kernel)
print(conv.feed_forward(arr))