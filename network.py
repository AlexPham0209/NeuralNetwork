import random
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.e ** (-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def cost(actual, expected):
    return (actual - expected) ** 2

def cost_derivative(actual, expected):
    return 2 * (actual - expected)


class Layer:
    def __init__(self, neuron_size, weight_size):
        self.neuron_size = neuron_size
        self.weight_size = weight_size

        self.activation = [0] * self.neuron_size
        self.error = [0] * self.neuron_size
        
        self.randomize_weights()
        self.randomize_biases()

    def feed_forward(self, a):
        self.activation = [np.dot(a, w) + b for w, b in zip(self.weights, self.biases)]        
        return self.activation
    
    def output_backpropagation(self, expected):
        self.error = [0] * self.neuron_size

        for i in range(self.neuron_size):
            cost_derivative = 2 * (sigmoid(self.activation[i]) - expected[i])
            activation_derivative = sigmoid_derivative(self.activation[i])
            self.error[i] = cost_derivative * activation_derivative

        return self.error

    def backpropagation(self, prev):
        self.error = [0] * self.neuron_size

        for i in range(self.neuron_size):
            res = 0 
            for j in range(prev.neuron_size):
                res += prev.error[j] * prev.weights[j][i]

            res *= sigmoid_derivative(self.activation[i])
            self.error[i] = res

        return self.error


    def apply_gradient(self, eta, prev):
        for i in range(self.neuron_size):
            for j in range(self.weight_size):
                self.weights[i][j] -= eta * sigmoid(prev[j]) * self.error[i]

        for i in range(self.neuron_size):
            self.biases[i] -= eta * self.error[i]

    def randomize_weights(self):
        self.weights = [[random.uniform(0.0, 1.0) for j in range(self.weight_size)] for i in range(self.neuron_size)]

    def randomize_biases(self):
        self.biases = [random.uniform(0.0, 1.0) for i in range(self.neuron_size)]


class NeuralNetwork:
    def __init__(self, layer_size):
        self.layer_size = layer_size
        self.layers = [Layer(curr, prev) for curr, prev in zip(layer_size[1:], layer_size)]

    def feed_forward(self, a, activ = sigmoid):
        if len(a) != self.layer_size[0]:
            return None
        
        for layer in self.layers:
            a = list(map(activ, layer.feed_forward(a)))

        return a

    def backpropagation(self, a, expected, eta = 0.5):
        actual = self.feed_forward(a)

        error = self.layers[-1].output_backpropagation(expected)

        for i in range(len(self.layers) - 2, -1, -1):
            curr, prev = self.layers[i], self.layers[i + 1]
            curr.backpropagation(prev)
            
        self.layers[0].apply_gradient(eta, a)
        for i in range(1, len(self.layers)):
            curr, prev = self.layers[i], self.layers[i - 1]
            curr.apply_gradient(eta, prev.activation)
