import random
import numpy as np
import cupy as cp
from ph.layers.layer import Layer


class Dense(Layer):
    def __init__(self, output_size=(), data=None):
        super().__init__()
        if data:
            self.load_data(data)
            return

        self.output_size = output_size
        self.error = cp.zeros(self.output_size)
        self.out = cp.zeros(self.output_size)

    def feed_forward(self, a):
        self.input = a
        self.out = a @ self.weights.T + self.biases[cp.newaxis, :]
        return self.out

    def backpropagation(self, prev, eta, size=1):
        # Calculate dC/dA for output
        self.error = prev

        # dz = self.activation.derivative(self.out, self.error)
        self.weights -= (eta / size) * (self.error.T @ self.input)
        self.biases -= (eta / size) * self.error.sum(0)

        return self.error @ self.weights

    # Randomizes all weights from 0 to 1
    def randomize_weights(self):
        self.weights = cp.random.uniform(
            low=-1.0, high=1.0, size=(self.output_size, self.input_size)
        )

    # Randomizes all biases from 0 to 1
    def randomize_biases(self):
        self.biases = cp.random.uniform(low=-1.0, high=1.0, size=(self.output_size))

    def save_data(self):
        data = dict()
        data["type"] = "dense"
        data["input_size"] = self.input_size
        data["output_size"] = self.output_size

        data["weights"] = self.weights.tolist()
        data["biases"] = self.biases.tolist()

        return data

    def load_data(self, data):
        self._input_size = data["input_size"]
        self.output_size = data["output_size"]

        self.weights = cp.array(data["weights"])
        self.biases = cp.array(data["biases"])

    # Setter function that is ran when the
    @Layer.input_size.setter
    def input_size(self, value):
        if cp.ndim(value) > 0:
            raise Exception("Not a scalar")

        self._input_size = value

        # Once we know what the input size is, we create the weights and biases
        self.randomize_weights()
        self.randomize_biases()
