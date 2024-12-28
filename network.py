import random
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.e ** (-x))

def sigmoid_derivative():
    return 

class Layer:
    def __init__(self, neuron_size, weight_size):
        self.neuron_size = neuron_size
        self.weight_size = weight_size
        
        self.randomize_weights()
        self.randomize_biases()

    def feed_forward(self, a):
        return [np.dot(a, w) + b for w, b in zip(self.weights, self.biases)]
        
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

    def backpropagation(self):
        pass

    def backpropagation_output(self):
        pass
    