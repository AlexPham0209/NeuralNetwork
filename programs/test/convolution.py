import math
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

def pooling(a, kernel_size, pad = True):
    o, j, k = a.shape
    c, h, w = a.shape
    k_h, k_w = kernel_size

    #Depending on pad, it either pads the outside with 0 or it reduces the sides so that its 
    if pad:
        n_h = int((np.ceil(h/k_h) * k_h) - h) if h % k_h != 0 else 0
        n_w = int((np.ceil(h/k_h) * k_w) - w) if w % k_w != 0 else 0
        a = np.pad(a, ((0, 0), (0, n_h), (0, n_w)), 'constant', constant_values=0)
    else:
        a = a[:, :(h//k_h) * k_h, :(w//k_w) * k_w]
    
    channel_stride, r_stride, c_stride = a.strides
    c, h, w = a.shape
    out_h, out_w = (h - k_h) // k_h + 1, (w - k_w) // k_w + 1
    new_shape = (c, out_h, out_w, k_h, k_w)
    new_stride = (channel_stride, r_stride * k_h, c_stride * k_w, r_stride, c_stride)

    out = np.lib.stride_tricks.as_strided(a, new_shape, new_stride)
    res = out.max(axis=(3, 4))

    maxs = res.repeat(k_h, axis=1).repeat(k_w, axis=2)
    x_window = a[:, :out_h * k_h, :out_w * k_w]
    mask = np.equal(x_window, maxs).astype(int)[:, :j, :k]
    print(mask)

    return out.max(axis=(3, 4))
    
    
# arr = np.array([[
#     [0, 1],
#     [3, 4],
#     [6, 7],
# ], 
# [
#     [0, 1],
#     [3, 4],
#     [6, 7],
# ]])

# k1 = np.array([[
#     [0, 1],
#     [2, 3]
# ],
# [
#     [2, 1],
#     [2, 5]
# ]])

# k2 = np.array([[
#     [4, 1],
#     [2, 5]
# ],
# [
#     [2, 1],
#     [2, 5]
# ]])

arr = np.array([[0, 1], [2, 3]])
w, h = arr.shape
print(arr.repeat(2, axis=0).repeat(2, axis=1))

# print(arr)
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

# print(convolve(arr, res))
# print(convolve(arr, res, mode="full"))
print(pooling(arr, (2, 2)))