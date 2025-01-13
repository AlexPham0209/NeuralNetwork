from enum import Enum
import random
import numpy as np
import activation as act

class Layer:
    def __init__(self, output_size, activation = act.Sigmoid()):
        self.activation = activation
        self.output_size = output_size
        
        #Before and after layers
        self.prev_layer = None
        self.next_layer = None

    def feed_forward(self, a):
        pass
    
    def backpropagation(self, prev):
        pass
    
    def update_gradient(self, prev):
        pass

    def apply_gradient(self, eta, size = 1):
        pass
    
    @property
    def input_size(self):
        return self._input_size

    @input_size.setter
    def input_size(self, value):
        self._input_size = value

class Dense(Layer):
    def __init__(self, output_size, activation):
        super().__init__(output_size, activation)

        self.error = np.zeros(self.output_size)
        self.out = np.zeros(self.output_size)

    def feed_forward(self, a):
        self.out = self.weights.dot(a) + self.biases 
        return self.out
    
    def backpropagation(self, prev):
        activate = self.activation.activate
        derivative = self.activation.derivative
        
        # Calculate dC/dA for output
        self.error = 2 * (activate(self.out) - prev) if not self.next_layer else prev

        if not self.prev_layer:
            return None
        
        return self.weights.T.dot(derivative(self.out) * self.error)

    def update_gradient(self, prev):
        activate = self.activation.activate
        derivative = self.activation.derivative

        p = activate(prev)
        o = derivative(self.out)

        self.weights_gradient += np.tile((self.error * o), (self.input_size,1)).T * p
        self.biases_gradient += o * self.error

    def apply_gradient(self, eta, size = 1):
        self.weights -= (eta / size) * self.weights_gradient
        self.biases -= (eta / size) * self.biases_gradient
        
        # Resets the error after applying gradient vector
        self.weights_gradient = np.zeros((self.output_size, self.input_size))
        self.biases_gradient = np.zeros(self.output_size)

    # Randomizes all weights from 0 to 1 
    def randomize_weights(self):
        self.weights = np.array([[random.uniform(-1.0, 1.0) for j in range(self.input_size)] for i in range(self.output_size)])
        
    # Randomizes all biases from 0 to 1 
    def randomize_biases(self):
        self.biases = np.array([random.uniform(-1.0, 1.0) for i in range(self.output_size)])

    @Layer.input_size.setter
    def input_size(self, value):
        self._input_size = value

        self.randomize_weights()
        self.randomize_biases()

        self.weights_gradient = np.zeros((self.output_size, self.input_size))
        self.biases_gradient = np.zeros(self.output_size)

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
