import random
import numpy as np

class Layer:
    def __init__(self, neuron_size, weight_size):
        self.neuron_size = neuron_size
        self.weight_size = weight_size

        self.randomize_weights()
        self.randomize_biases()

    def feed_forward(self, a):
        output = [0] * self.neuron_size

        for i, l in enumerate(zip(self.weights, self.biases)):
            w, b = l
            output[i] = np.dot(a, w) + b

        return output
        
    def randomize_weights(self):
        self.weights = [[random.uniform(0.0, 1.0)] * self.weight_size for _ in range(self.neuron_size)]

    def randomize_biases(self):
        self.biases = [random.uniform(0.0, 1.0)] * self.neuron_size


class NeuralNetwork:
    def __init__(self, layer_size):
        self.layer_size = layer_size
        self.layers = [Layer(curr, prev) for curr, prev in zip(layer_size[1:], layer_size)]

    def feed_forward(self, a):
        print(self.layers)
        for layer in self.layers:
            print(a)
            a = layer.feed_forward(a)
        
        return a

    def backpropagation(self):
        pass

    def backpropagation_output(self):
        pass
    