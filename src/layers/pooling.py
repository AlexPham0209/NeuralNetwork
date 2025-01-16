import numpy as np
from skimage.measure import block_reduce
from src.layers.layer import Layer

class Pooling(Layer):
    def __init__(self, output_size, kernel_size, activation):
        super.__init__(output_size, activation)
        self.kernel_size = kernel_size

    def pooling(a, kernel_size):
        c, h, w = a.shape
        k_h, k_w = kernel_size

        #Depending on pad, it either pads the outside with 0 or it reduces the sides so that its 
        n_h = int((np.ceil(h/k_h) * k_h) - h) if h % k_h != 0 else 0
        n_w = int((np.ceil(h/k_h) * k_w) - w) if w % k_w != 0 else 0
        a = np.pad(a, ((0, 0), (0, n_h), (0, n_w)), 'constant', constant_values=0)

        # else:
        #     a = a[:, :(h//k_h) * k_h, :(w//k_w) * k_w]
        
        channel_stride, r_stride, c_stride = a.strides
        c, h, w = a.shape
        out_h, out_w = (h - k_h) // k_h + 1, (w - k_w) // k_w + 1
        new_shape = (c, out_h, out_w, k_h, k_w)
        new_stride = (channel_stride, r_stride * k_h, c_stride * k_w, r_stride, c_stride)
        
        out = np.lib.stride_tricks.as_strided(a, new_shape, new_stride)
        return out.max(axis=(3, 4))

    def feed_forward(self, a):
        self.input = np.array(a, copy=True)
        self.out = self.pooling(a, self.kernel_size)

        c, h, w = self.input.shape
        out_c, out_h, out_w = self.out.shape
        k_h, k_w = self.kernel_size

        maxs = self.out.repeat(k_h, axis=1).repeat(k_w, axis=2)
        x_window = a[:, :out_h * k_h, :out_w * k_w]
        self.mask = np.equal(x_window, maxs).astype(int)

        return self.activation.activate(self.out)
    
    def backpropagation(self, prev):
        self.error = prev * self.activation.derivative(self.out)

        w, h = self.kernel_size.shape
        res = prev.repeat(h, axis=1).repeat(w, axis=2)
        
        pad = np.zeros(self.input.shape)
        pad[:, :res.shape[1], :res.shape[2]] = res

        return pad
    
    