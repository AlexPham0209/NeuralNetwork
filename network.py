import random
import numpy as np

class Activation:
    def activate(self, x):
        pass

    def derivative(self, x):
        pass

class Sigmoid(Activation):
    def activate(self, x):
        return 1 / (1 + np.e ** (-x))

    def derivative(self, x):
        return self.activate(x) * (1 - self.activate(x))

class Layer:
    def __init__(self, neuron_size, weight_size, activation):
        self.activation = activation
        self.neuron_size = neuron_size
        self.weight_size = weight_size

        self.values = [0] * self.neuron_size
        self.error = [0] * self.neuron_size
        
        self.randomize_weights()
        self.randomize_biases()

    def feed_forward(self, a):
        self.values = [np.dot(a, w) + b for w, b in zip(self.weights, self.biases)]        
        return self.values
    
    def output_backpropagation(self, expected):
        self.error = [0] * self.neuron_size
        activate = self.activation.activate
        derivative = self.activation.derivative

        for i in range(self.neuron_size):
            cost_derivative = 2 * (activate(self.values[i]) - expected[i])
            activation_derivative = derivative(self.values[i])
            self.error[i] = cost_derivative * activation_derivative

        return self.error

    def backpropagation(self, prev):
        self.error = [0] * self.neuron_size
        output = self.activation.activate
        derivative = self.activation.derivative

        for i in range(self.neuron_size):
            res = 0 
            for j in range(prev.neuron_size):
                res += prev.error[j] * prev.weights[j][i]

            res *= derivative(self.values[i])
            self.error[i] = res

        return self.error


    def apply_gradient(self, eta, prev):
        activate = self.activation.activate

        for i in range(self.neuron_size):
            for j in range(self.weight_size):
                self.weights[i][j] -= eta * activate(prev[j]) * self.error[i]

        for i in range(self.neuron_size):
            self.biases[i] -= eta * self.error[i]

    def randomize_weights(self):
        self.weights = [[random.uniform(0.0, 1.0) for j in range(self.weight_size)] for i in range(self.neuron_size)]

    def randomize_biases(self):
        self.biases = [random.uniform(0.0, 1.0) for i in range(self.neuron_size)]


class NeuralNetwork:
    def __init__(self, layer_size, activation = Sigmoid()):
        self.layer_size = layer_size
        self.activation = activation
        self.layers = [Layer(curr, prev, activation) for curr, prev in zip(layer_size[1:], layer_size)]

    def feed_forward(self, a):
        if len(a) != self.layer_size[0]:
            return None
        
        for layer in self.layers:
            a = list(map(self.activation.activate, layer.feed_forward(a)))

        return a

    def learn(self, dataset, iterations, eta):
        for i in range(iterations):
            for data in dataset:
                input, expected = data
                self.backpropagation(input, expected, eta)


    def backpropagation(self, a, expected, eta = 0.5):
        actual = self.feed_forward(a)

        error = self.layers[-1].output_backpropagation(expected)
        for i in range(len(self.layers) - 2, -1, -1):
            curr, prev = self.layers[i], self.layers[i + 1]
            curr.backpropagation(prev)
            
        self.layers[0].apply_gradient(eta, a)
        for i in range(1, len(self.layers)):
            curr, prev = self.layers[i], self.layers[i - 1]
            curr.apply_gradient(eta, prev.values)
