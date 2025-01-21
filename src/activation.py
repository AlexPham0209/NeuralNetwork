from scipy.special import expit
from scipy.special import softmax
from opt_einsum import contract
import numpy as np
import numpy as cp

class Activation:
    def activate(self, x):
        return x

    def derivative(self, x, error):
        return x

    def __repr__(self): 
        return ""

class Sigmoid(Activation):
    def activate(self, x):
        return expit(x)

    def derivative(self, x, error):
        return (self.activate(x) * (1 - self.activate(x))) * error
        
    def __repr__(self): 
        return "sigmoid"

class ReLU(Activation):
    def activate(self, x):
        return x * (x > 0)
    
    def derivative(self, x, error):
        return (1. * (x > 0)) * error
    
    def __repr__(self): 
        return "relu"

class SoftMax(Activation):
    def activate(self, x):
        e_x = cp.exp(x) 
        return e_x / e_x.sum(axis=1, keepdims=True) 

    def derivative(self, x, error): 
        if x.ndim != 2 or error.ndim != 2:
            raise Exception("Softmax derivative only accepts 2D matrices")

        # Calculate Jacobian matrix for all samples in batch
        s = self.activate(x)
        a = cp.eye(s.shape[-1])
        temp1 = cp.zeros((s.shape[0], s.shape[1], s.shape[1]),dtype=cp.float32)
        temp2 = cp.zeros((s.shape[0], s.shape[1], s.shape[1]),dtype=cp.float32)
        temp1 = contract('ij,jk->ijk',s,a)
        temp2 = contract('ij,ik->ijk',s,s)
        J = temp1 - temp2

        # Calculate the dot product between each sample and its subsequent Jacobian
        return contract("ijk,ki->ij", J, error.T)
        
    def __repr__(self): 
        return "softmax"
    
class Tanh(Activation):
    def activate(self, x):
        return np.tanh(x)
    
    def derivative(self, x, error):
        return (1 - np.tanh(x)**2) * error 
    
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