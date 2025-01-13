from src.layers.layer import Layer

class Flatten(Layer):
    def __init__(self):
        super().__init__((0))

    def feed_forward(self, a):
        return a.flatten()

    def backpropagation(self, prev):
        return prev.reshape(self.input_size)
    