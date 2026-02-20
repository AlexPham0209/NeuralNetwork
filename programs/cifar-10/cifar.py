import sys

import keras
import cupy as cp
import numpy as np

import json
import random

from encoder import NumpyEncoder
from layers.dense import Dense
from layers.conv2d import Conv2D
from layers.pooling import MaxPooling
from layers.flatten import Flatten
from network import Model

import src.activation as act
import src.loss as ls

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_train = x_train.transpose(0, 3, 1, 2) / 255
x_test = x_test.transpose(0, 3, 1, 2) / 255

architecture = [
    Conv2D(32, (3, 3), act.Sigmoid()),
    MaxPooling((2, 2)),

    Conv2D(64, (3, 3), act.Sigmoid()),
    MaxPooling((2, 2)),

    Conv2D(128, (3, 3), act.Sigmoid()),
    MaxPooling((2, 2)),
    
    Conv2D(128, (2, 2), act.Sigmoid()),
    Flatten(),
        
    Dense(64, act.Sigmoid()), 
    Dense(10, act.SoftMax())
]

network = None
match input("Select Mode (Train, Load, or Continue): ").lower():
    case "train":
        network = Model(architecture, input_size = (3, 32, 32), output_size = 10, loss=ls.MeanSquaredError())
        network.learn(x_train, y_train, 15, 0.5, 64, debug=True)

        save_data = network.save_data()
        with open("programs/cifar-10/network.json", "w") as file:
            json.dump(save_data, file, indent = 1, cls=NumpyEncoder)

    case "load":
        network = Model(path = "programs/cifar-10/network.json")

    case "continue":
        network = Model(path = "programs/cifar-10/network.json")
        network.learn(x_train, y_train, 15, 0.75, 64, debug=True)

        save_data = network.save_data()
        with open("programs/cifar-10/network.json", "w") as file:
            json.dump(save_data, file, indent = 1, cls=NumpyEncoder)

test_data = list(zip(x_test.tolist(), y_test.tolist()))
correct = 0
wrong = 0
    
for data in test_data:
    input, label = data

    input = cp.array(input)
    label = label[0]

    index, actual = network.evaluate(input)

    if index == label:
        correct += 1 
    else:
        wrong += 1 

print(f"Correct: {correct}")
print(f"Wrong: {wrong}")
print(f"Percentage: {correct / (correct + wrong)}")