import src.activation as act

class Layer:
    def __init__(self):
        self.output_size = (0)
        
        #Before and after layers
        self.prev_layer = None
        self.next_layer = None

    def feed_forward(self, a):
        pass
    
    def backpropagation(self, prev):
        pass
    
    def update_gradient(self):
        pass

    def apply_gradient(self, eta, size = 1):
        pass
    
    @property
    def input_size(self):
        return self._input_size

    @input_size.setter
    def input_size(self, value):
        if np.ndim(value) != 3:
            raise Exception("Not a 3 dimensional input")
        
        self._input_size = value

