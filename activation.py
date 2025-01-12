import numpy as np
from scipy.special import expit

class Activation:
    def activate(self, x):
        pass

    def derivative(self, x):
        pass

    def __repr__(self): 
        return ""

class Sigmoid(Activation):
    def activate(self, x):
        return expit(x)

    def derivative(self, x):
        return self.activate(x) * (1 - self.activate(x))
    
    def __repr__(self): 
        return "Sigmoid"

class ReLU(Activation):
    def activate(self, x):
        return x * (x > 0)
    
    def derivative(self, x):
        return 1. * (x > 0)
    
    def __repr__(self): 
        return "ReLU"
    