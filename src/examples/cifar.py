from sklearn.preprocessing import OneHotEncoder

from ph.layers.activation import ReLU, Sigmoid, SoftMax
from ph.layers.conv2d import Conv2D
from ph.layers.dense import Dense
from ph.layers.flatten import Flatten
from ph.layers.pooling import MaxPooling
from ph.loss import CrossEntropy
from ph.metric import Accuracy
from ph.network import Model
from keras.datasets import cifar10
import cupy as cp

(train_X, train_y), (test_X, test_y) = cifar10.load_data()

# Flatten images
train_X = cp.transpose(train_X, axes=(0, 3, 1, 2))
test_X = cp.transpose(test_X, axes=(0, 3, 1, 2))

train_X = train_X / 255.0
test_X = test_X / 255.0

# Convert labels to one-hot encoded vectors
encoder = OneHotEncoder().fit(train_y.reshape(-1, 1))
train_y = encoder.transform(train_y.reshape(-1, 1)).toarray()
test_y = encoder.transform(test_y.reshape(-1, 1)).toarray()

# Convert to GPU arrays
train_X = cp.array(train_X)
train_y = cp.array(train_y)
test_X = cp.array(test_X)
test_y = cp.array(test_y)

architecture = [
    Conv2D(kernels=16, kernel_size=(3, 3)),
    MaxPooling(kernel_size=(3, 3)),
    Sigmoid(),

    Conv2D(kernels=32, kernel_size=(3, 3)),
    MaxPooling(kernel_size=(3, 3)),
    Sigmoid(),

    Flatten(),

    Dense(1024),
    Sigmoid(),

    Dense(train_y.shape[-1]),
    SoftMax(),
]

network = Model(architecture, input_size = (3, 32, 32), output_size = train_y.shape[-1], loss = CrossEntropy(), metric=Accuracy())
network.learn(
    x = train_X,
    y = train_y,
    valid_set=(test_X, test_y),
    epoch=30,
    eta=0.1,
    batch_size=32
)