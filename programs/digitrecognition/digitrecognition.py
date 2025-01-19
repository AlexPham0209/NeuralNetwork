import sys
sys.path.insert(1, '../NeuralNetwork')

import json
import random
import numpy as np

import src.activation as act
from src.layers.dense import Dense
from src.layers.conv2d import Conv2D
from src.layers.pooling import MaxPooling
from src.layers.flatten import Flatten
from src.network import Model
import cupy as cp

ROW = 28
COL = 28

def read_digits(path, array = False):
    dataset = []
    with open(path) as file:
        for data in file.read().strip().split("\n"):
            data = list(map(int, data.split(",")))

            expected = data[0]
            if array:
                expected = cp.zeros(10)
                expected[data[0]] = 1 

            dataset.append((cp.array(data[1:]).reshape((1, 28, 28))/255, expected))

    return dataset


def train_network():
    train_data = read_digits("C:/Users/RedAP/Desktop/mnist_train.csv", True)
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
        Dense(10, act.Sigmoid())
    ]
    
    network = Model(architecture, input_size = (1, 28, 28))
    network.learn(train_data, 3, 0.5, 64, debug=True)
    test(network)


def load_network():
    architecture = [
        Dense(64, act.Sigmoid()), 
        Dense(64, act.Sigmoid()), 
        Dense(10, act.Sigmoid())
    ]
        
    network = Model(architecture, input_size = ROW * COL)
    test(network)

def test(network):
    test_data = read_digits("C:/Users/RedAP/Desktop/mnist_test.csv", False)
    correct = 0
    wrong = 0
    
    output = open("programs/digitrecognition/output/out.txt", "w")
    for data in test_data:
        input, expected = data
        actual, test = network.evaluate(input)
        
        output.write(f"Actual: {actual}\n")
        output.write(f"Expected: {expected}\n")
        output.write(f"Array: {test}\n\n")

        if actual == expected:
            correct += 1 
        else:
            wrong += 1 

    print(f"Correct: {correct}")
    print(f"Wrong: {wrong}")
    print(f"Percentage: {correct / (correct + wrong)}")


# if __name__ == "__main__":
#     match input("Pick Mode (Train or Load): ").lower():
#         case "train":
#             train_network()
#         case "load":
#             load_network()

train_network()