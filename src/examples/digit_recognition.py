from sklearn.preprocessing import OneHotEncoder

from ph.layers.activation import Sigmoid, SoftMax
from ph.layers.dense import Dense
from ph.loss import CrossEntropy
from ph.metrics import Accuracy
from ph.network import Model
from keras.datasets import mnist
import cupy as cp

(train_X, train_y), (test_X, test_y) = mnist.load_data()

# Flatten images
train_X = train_X.reshape(train_X.shape[0], -1)
test_X = test_X.reshape(test_X.shape[0], -1)

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
    Dense(train_X.shape[-1]),
    Sigmoid(),

    Dense(1000),
    Sigmoid(),

    Dense(1000),
    Sigmoid(),

    Dense(train_y.shape[-1]),
    SoftMax(),
]

network = Model(architecture, input_size = train_X.shape[-1], output_size = train_y.shape[-1], loss = CrossEntropy(), metric=Accuracy())
network.learn(
    x = train_X,
    y = train_y,
    valid_set=(test_X, test_y),
    epoch=100,
    eta=0.01,
    batch_size=16
)