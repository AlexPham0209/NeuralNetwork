class Layer:
    def __init__(self):
        self._input_size = None
        self.output_size = None

    def feed_forward(self, a):
        pass

    def backpropagation(self, prev, eta, size=1):
        pass

    def save_data(self):
        return dict()

    def load_data(self, data):
        pass

    @property
    def input_size(self):
        return self._input_size

    @input_size.setter
    def input_size(self, value):
        self._input_size = value
