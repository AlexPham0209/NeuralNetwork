import sys
sys.path.insert(1, '../NeuralNetwork')

import cupy as cp
from src.activation import SoftMax
from opt_einsum import contract
from src.activation import ReLU

a = cp.array([
    [1, 2, 3, 5, 6],
    [4, 5, 6, 5, 6]
])

b = cp.array([
    [1, 2, 3, 4, 5],
    [2, 4, 5, 4, 5],
])

c = cp.array([
    [4, 2, 3, 4, 5],
    [2, 4, 5, 4, 5],
])

# print(cp.array([a, b]).reshape(2, 10))

# Jacobian matrix
e = cp.array([a, b, c])

# Batch error matrix
k = cp.array([e, e, e])

relu = ReLU()
print(relu.derivative(k, k))
# print(cp.array([-2, -1, 0, 1, 2]) * (cp.array([-2, -1, 0, 1, 2]) > 0))

# print(e)
# print()
# print(k)
# print()
# print(contract("ijk,ki->ij", e, k))
