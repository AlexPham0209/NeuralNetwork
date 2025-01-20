from scipy.special import expit
from scipy.special import softmax
from opt_einsum import contract
import numpy as np
import numpy as cp

class Activation:
    def activate(self, x):
        return x

    def derivative(self, x):
        return x

    def __repr__(self): 
        return ""

class Sigmoid(Activation):
    def activate(self, x):
        return expit(x)

    def derivative(self, x):
        return self.activate(x) * (1 - self.activate(x))
    
    def __repr__(self): 
        return "sigmoid"

class ReLU(Activation):
    def activate(self, x):
        return x * (x > 0)
    
    def derivative(self, x):
        return 1. * (x > 0)
    
    def __repr__(self): 
        return "relu"

class SoftMax(Activation):
    def activate(self, x):
        e_x = cp.exp(x) 
        return e_x / e_x.sum(axis=1, keepdims=True) 
    
    def derivative(self, x): 
        J = - x[..., None] * x[:, None, :] # off-diagonal Jacobian
        iy, ix = cp.diag_indices_from(J[0])
        J[:, iy, ix] = x * (1. - x) # diagonal
        return J.sum(axis=1) # sum across-rows for each sample
        
    def __repr__(self): 
        return "softmax"
    
class Tanh(Activation):
    def activate(self, x):
        return np.tanh(x)
    
    def derivative(self, x):
        return 1 - np.tanh(x)**2 
    
    def __repr__(self): 
        return "tanh"
    

activations = {
    "sigmoid" : Sigmoid(),
    "rel" : ReLU(),
    "softmax" : SoftMax(),
    "tanh" : Tanh()
}


def create_activation(type):
    return activations[type]