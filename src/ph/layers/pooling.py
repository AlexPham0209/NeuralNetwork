import numpy as np
import cupy as cp
from skimage.measure import block_reduce
from ph.layers.layer import Layer


class MaxPooling(Layer):
    def __init__(self, kernel_size=(), data=None):
        super().__init__()
        if data:
            self.load_data(data)
            return

        self.kernel_size = kernel_size

    def pooling(self, a, kernel_size):
        b, c, h, w = a.shape
        k_h, k_w = kernel_size

        batch_stride, channel_stride, r_stride, c_stride = a.strides

        out_c, out_h, out_w = self.output_size
        new_shape = (b, out_c, out_h, out_w, k_h, k_w)
        new_stride = (
            batch_stride,
            channel_stride,
            r_stride * k_h,
            c_stride * k_w,
            r_stride,
            c_stride,
        )

        out = cp.lib.stride_tricks.as_strided(a, new_shape, new_stride)
        return out.max(axis=(4, 5))

    def feed_forward(self, a):
        # Pads the input so that its divisible by the kernel size
        a = cp.pad(
            a,
            ((0, 0), (0, 0), (0, self.pad_h), (0, self.pad_w)),
            "constant",
            constant_values=0,
        )
        self.input = a

        self.out = self.pooling(a, self.kernel_size)

        out_c, out_h, out_w = self.output_size
        k_h, k_w = self.kernel_size

        # Creates a matrix where all the maxes in each window are equal to 1 and everything else is 0
        maxs = self.out.repeat(k_h, axis=2).repeat(k_w, axis=3)
        x_window = a[:, :, : out_h * k_h, : out_w * k_w]
        self.mask = cp.equal(x_window, maxs).astype(int)

        return self.out

    def backpropagation(self, prev, eta, size=1):
        self.error = prev
        w, h = self.kernel_size
        scaled = prev.repeat(h, axis=2).repeat(w, axis=3)

        return (self.mask * scaled)[:, :, : self.input_size[1], : self.input_size[2]]

    def save_data(self):
        data = dict()
        data["type"] = "pooling"
        data["input_size"] = self.input_size
        data["output_size"] = self.output_size
        data["kernel_size"] = self.kernel_size

        data["pad_height"] = int(self.pad_h)
        data["pad_width"] = int(self.pad_w)

        return data

    def load_data(self, data):
        self._input_size = tuple(data["input_size"])
        self.output_size = tuple(data["output_size"])
        self.kernel_size = tuple(data["kernel_size"])

        self.pad_h = data["pad_height"]
        self.pad_w = data["pad_width"]

    @Layer.input_size.setter
    def input_size(self, value):
        if len(value) != 3:
            raise Exception("Not a 3 dimensional input")

        self._input_size = value

        c, h, w = self.input_size
        k_h, k_w = self.kernel_size

        # Creates padding for the array in cases where the height and width are not divisible by the kernel size
        self.pad_h = int((np.ceil(h / k_h) * k_h) - h) if h % k_h != 0 else 0
        self.pad_w = int((np.ceil(h / k_h) * k_w) - w) if w % k_w != 0 else 0

        # Sets the output size for the max pooling layer
        out_h, out_w = (
            ((h + self.pad_h) - k_h) // k_h + 1,
            ((w + self.pad_w) - k_w) // k_w + 1,
        )
        self.output_size = (c, out_h, out_w)
