from scipy.special import expit
from scipy.special import softmax
from opt_einsum import contract
import numpy as np
import numpy as cp

from ph.layers.layer import Layer


class Activation(Layer):
    def __init__(self, data=None):
        super().__init__()
        if data:
            self.load_data(data)

    @Layer.input_size.setter
    def input_size(self, value):
        self._input_size = value
        self.output_size = value

    def save_data(self):
        data = dict()
        data["type"] = str(self)
        data["input_size"] = self.input_size
        data["output_size"] = self.output_size

        return data

    def load_data(self, data):
        self._input_size = data["input_size"]
        self.output_size = data["output_size"]

    def __repr__(self):
        return ""


class Sigmoid(Activation):
    def feed_forward(self, a):
        self.input = a
        self.out = expit(a)
        return self.out

    def backpropagation(self, prev, eta, size=1):
        return (self.out * (1 - self.out)) * prev

    def __repr__(self):
        return "sigmoid"


class ReLU(Activation):
    def feed_forward(self, a):
        self.input = a
        self.out = cp.maximum(0.01 * a, a)
        return self.out

    def backpropagation(self, prev, eta, size=1):
        return cp.where(self.input > 0, 1, 0.01) * prev

    def __repr__(self):
        return "relu"


class SoftMax(Activation):
    def feed_forward(self, a):
        self.input = a
        max_x = np.amax(a, 1).reshape(a.shape[0], 1)
        e_x = cp.exp(a - max_x)
        self.out = e_x / e_x.sum(axis=1, keepdims=True)
        return self.out

    def backpropagation(self, prev, eta, size=1):
        if self.out.ndim != 2 or prev.ndim != 2:
            raise Exception("Softmax derivative only accepts 2D matrices")

        # Calculate Jacobian matrix for all samples in batch
        s = self.out
        a = cp.eye(s.shape[-1])
        temp1 = cp.zeros((s.shape[0], s.shape[1], s.shape[1]), dtype=cp.float32)
        temp2 = cp.zeros((s.shape[0], s.shape[1], s.shape[1]), dtype=cp.float32)
        temp1 = contract("ij,jk->ijk", s, a)
        temp2 = contract("ij,ik->ijk", s, s)
        J = temp1 - temp2

        # Calculate the dot product between each sample and its subsequent Jacobian
        return contract("ijk,ki->ij", J, prev.T)

    def __repr__(self):
        return "softmax"


class Tanh(Activation):
    def feed_forward(self, a):
        self.input = a
        self.out = np.tanh(a)
        return self.out

    def backpropagation(self, prev, eta, size=1):
        return (1 - self.out**2) * prev

    def __repr__(self):
        return "tanh"
