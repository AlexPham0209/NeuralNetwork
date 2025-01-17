import math
import sys
sys.path.insert(1, '../NeuralNetwork')

import numpy as np
from src.layers.flatten import Flatten
from scipy.signal import convolve2d
from scipy.signal import correlate2d

# dC/dF code
def convolve(a, b):
    c, h, w = a.shape
    channel_stride, r_stride, c_stride = a.strides
    k_c, k_h, k_w = b.shape
        
    out_h, out_w = (h - k_h) + 1, (w - k_w) + 1
    new_shape = (c, out_h, out_w, k_h, k_w)
    new_stride = (channel_stride, r_stride, c_stride, r_stride, c_stride)
    
    out = np.lib.stride_tricks.as_strided(a, new_shape, new_stride)
    return np.einsum("nhwkt,ckt->cnhw", out, b)

# dC/dA code
def convolve2(a, b, stride = 1):
    a = np.pad(a, ((0, 0), (0, 0), (1, 1), (1, 1)), 'constant', constant_values=0)
    b = b[:, ::-1, ::-1]
    n, c, h, w = a.shape
    num_stride, channel_stride, r_stride, c_stride = a.strides
    k_c, k_h, k_w = b.shape
        
    out_h, out_w = (h - k_h) + 1, (w - k_w) + 1
    new_shape = (n, c, out_h, out_w, k_h, k_w)
    new_stride = (num_stride, channel_stride, r_stride * stride, c_stride * stride, r_stride, c_stride)
    
    out = np.lib.stride_tricks.as_strided(a, new_shape, new_stride)

    return np.einsum("nchwkt,nkt->chw", out, b)

def test(a, b):
    c, h, w = a.shape
    k_c, k_h, k_w = b.shape
        
    out_h, out_w = (h - k_h) + 1, (w - k_w) + 1
    res = np.zeros((c, k_c, out_h, out_w))

    for i in range(k_c):
        for j in range(k_c):
            res[i, j] += correlate2d(a[j], b[i], "valid")

    return res

def test2(a, b):
    n, c, h, w = a.shape
    k_c, k_h, k_w = b.shape
    out_h, out_w = (h - k_h) + 1, (w - k_w) + 1
    new_shape = (c, 3)
        
    res = np.zeros(new_shape)
    print(res.shape)
    for i in range(c):
        for j in range(c):
            print(convolve2d(a[i, j], b[i], "full"))
            res[j] += convolve2d(a[i, j], b[i], "full")
    
    return res

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
    
    
arr = np.array([[
    [0, 1, 4],
    [3, 4, 5],
    [6, 7, 6],
], 
[
    [0, 1, 1],
    [3, 4, 2],
    [6, 7, 3],
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

print(convolve(arr, k1))
print("next")
print(test(arr, k1))
print("\n\n")
print(convolve2(np.array([k1, k2]), k1))
print("next")
print(test2(np.array([k1, k2]), k1))

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

#print(convolve(np.array(arr), k1))
# print()
# print(np.array([k1, k2]))
# print(convolve(arr, res, mode="full"))
# print(pooling(arr, (2, 2)))