import numpy as np
from skimage.measure import block_reduce
from src.layers.layer import Layer

class Pooling(Layer):
    def __init__(self, output_size, kernel_size, activation):
        super.__init__(output_size, activation)
        self.kernel_size = kernel_size

    def pooling(a, kernel_size, pad = True):
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
        return out.max(axis=(3, 4))

    def feed_forward(self, a):
        self.input = a
        self.out = self.pooling(a, self.kernel_size, pad = True)
        return self.activation.activate(self.out)
    

    def backpropagation(self, prev):
        return super().backpropagation(prev)

    def update_gradient(self):
        return super().update_gradient()
    
    def apply_gradient(self, eta, size=1):
        return super().apply_gradient(eta, size)
    
    