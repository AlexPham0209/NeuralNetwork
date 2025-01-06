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

class ReLU(Activation):
    def activate(self, x):
        return max(0, x)
    
    def derivative(self, x):
        return 0 if x <= 0 else 1