import src.activation as act

class Layer:
    def __init__(self):
        self.output_size = (0)
        
        #Before and after layers
        self.prev_layer = None
        self.next_layer = None

    def feed_forward(self, a):
        pass
    
    def backpropagation(self, prev, eta, size = 1):
        pass
    
    @property
    def input_size(self):
        return self._input_size

    @input_size.setter
    def input_size(self, value):
        self._input_size = value

