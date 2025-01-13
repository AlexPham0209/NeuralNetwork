import random
import numpy as np
from src.layers.layer import Layer

class Dense(Layer):
    def __init__(self, output_size, activation):
        super().__init__(output_size, activation)
        
        self.error = np.zeros(self.output_size)
        self.out = np.zeros(self.output_size)

    def feed_forward(self, a):
        self.out = self.weights.dot(a) + self.biases 
        return self.activation.activate(self.out)
    
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
    
    # Setter function that is ran when the 
    @Layer.input_size.setter
    def input_size(self, value):
        if np.ndim(value) > 0:
            raise Exception("Invalid input shape")
        
        self._input_size = value

        # Once we know what the input size is, we create the weights and biases
        self.randomize_weights()
        self.randomize_biases()

        # Sets the weight and biases gradients as well
        self.weights_gradient = np.zeros((self.output_size, self.input_size))
        self.biases_gradient = np.zeros(self.output_size)
