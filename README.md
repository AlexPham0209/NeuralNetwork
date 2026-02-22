# Ph: Convolutional and Feedforward Neural Network from Scratch
Ph is a deep learning library written from scratch using only Python and CuPy (a GPU-accelerated version of NumPy). The purpose of the project is to learn about major concepts behind convolutional neural networks: convolutions, pooling, feed forward and networks.

Requires CUDA.

## Usage
First, install the project and execute the following command to build the project and download all dependencies.

```
pip install -e .
```

Then, to run the MNIST demo, run this command from the root: 
```
python src/examples/mnist.py
```

Then after running, you should see the image below: 
![MNIST visual](images/mnist.png)

## Layer Types
You are able to build your model using Convolution, Max Pooling, Dense, and the Flatten layer.

## Activation Functions
There are four activations functions in this library: Sigmoid, ReLU, SoftMax, and Tanh. 

## Loss Functions
This library contains implementations for two loss functions, Mean Squared Error and Cross Entropy, for regression and classification tasks respectively. 


## Getting Started
To get started, after importing our weights, we first define our networks architecture using a list of Layer objects. Afterwards, we create a Network object, passing in our architecture list, the shape of our input and output, the loss function, and the validation metric. Finally, to train, we call network.learn(). 

```python
architecture = [
    Conv2D(kernels=16, kernel_size=(3, 3)),
    MaxPooling(kernel_size=(3, 3)),
    Sigmoid(),
    Conv2D(kernels=8, kernel_size=(3, 3)),
    MaxPooling(kernel_size=(3, 3)),
    Sigmoid(),
    Flatten(),
    Dense(512),
    Sigmoid(),
    Dense(train_y.shape[-1]),
    SoftMax(),
]

network = Model(
    layers=architecture,
    input_size=(1, 28, 28),
    output_size=train_y.shape[-1],
    loss=CrossEntropy(),
    metric=Accuracy(),
)

network.learn(
    x=train_X, y=train_y, valid_set=(test_X, test_y), epoch=10, eta=1e-1, batch_size=32
)
```
