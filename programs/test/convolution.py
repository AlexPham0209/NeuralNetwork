import sys
sys.path.insert(1, '../NeuralNetwork')

import numpy as np
from src.layers.flatten import Flatten
from scipy.signal import convolve

def convolve(a, b, stride = 1, mode = 'valid'):
    if mode == 'full':
        a = np.pad(arr, ((0,0), (1, 1), (1, 1)), 'constant', constant_values=0)
    
    c, h, w = a.shape
    channel_stride, r_stride, c_stride = a.strides
    num, k_c, k_h, k_w = b.shape
        
    out_h, out_w = (h - k_h) // stride + 1, (w - k_w) // stride + 1
    new_shape = (c, out_h, out_w, k_h, k_w)
    new_stride = (channel_stride, r_stride * stride, c_stride * stride, r_stride, c_stride)
    
    out = np.lib.stride_tricks.as_strided(a, new_shape, new_stride)
    return np.einsum("chwkt,nckt->nhw", out, b)

def pooling(a, kernel_size):
    c, h, w = a.shape
    stride = kernel_size
    channel_stride, r_stride, c_stride = a.strides

    out_h, out_w = (h - kernel_size) // stride + 1, (w - kernel_size) // stride + 1
    new_shape = (c, out_h, out_w, kernel_size, kernel_size)
    new_stride = (channel_stride, r_stride * stride, c_stride * stride, r_stride, c_stride)

    out = np.lib.stride_tricks.as_strided(a, new_shape, new_stride)
    return out.max(axis=(3, 4))
    
    
arr = np.array([[
    [0, 1, 2, 4],
    [3, 4, 5, 10],
    [6, 7, 8, 10],
    [6, 7, 8, 10],
], 
[
    [0, 1, 2, 4],
    [3, 4, 5, 10],
    [6, 7, 8, 10],
    [6, 7, 8, 10],
]])

k1 = np.array([[
    [0, 1],
    [2, 3]
],
[
    [2, 1],
    [2, 5]
]])

k2 = np.array([[
    [4, 1],
    [2, 5]
],
[
    [2, 1],
    [2, 5]
]])

res = np.array([k1, k2])

# print(np.pad(arr, ((0,0), (1, 1), (1, 1)), 'constant', constant_values=0))

# arr = np.array([
#     [0, 1, 2],
#     [3, 4, 5],
#     [6, 7, 8]
# ])


# kernel = np.array([
#     [0, 1],
#     [2, 3]
# ])

print(convolve(arr, res))
print(convolve(arr, res, mode="full"))
# print(pooling(arr, 2))