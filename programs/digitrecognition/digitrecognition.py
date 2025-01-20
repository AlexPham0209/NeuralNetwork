import sys
sys.path.insert(1, '../NeuralNetwork')

import json
import random
import numpy as np
from src.encoder import NumpyEncoder

import src.activation as act
from src.layers.dense import Dense
from src.layers.conv2d import Conv2D
from src.layers.pooling import MaxPooling
from src.layers.flatten import Flatten
from src.network import Model

import cupy as cp
import src.loss as ls

ROW = 28
COL = 28

def read_digits(path, array = False):
    dataset = cp.genfromtxt(path, delimiter=",", usemask=True)
    labels = dataset[:, :1]
    data = dataset[:, 1:]

    size = data.shape[0]
    data = data.reshape(size, 1, 28, 28)/255

    return labels, data


def train_network():
    labels, data = read_digits("C:/Users/RedAP/Desktop/mnist_train.csv")
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
    
    network = Model(architecture, input_size = (1, 28, 28), output_size = 10, loss=ls.CrossEntropy())
    network.learn(data, labels, 3, 0.5, 64, debug=True)

    # save_data = network.save_data()
    # with open("programs/digitrecognition/output/network.json", "w") as file:
    #     json.dump(save_data, file, indent = 1, cls=NumpyEncoder)

    test(network)


def load_network(path):        
    network = Model(path = path)
    test(network)

def test(network = None):
    labels, data = read_digits("C:/Users/RedAP/Desktop/mnist_test.csv")
    test_data = list(zip(labels.tolist(), data.tolist()))
    correct = 0
    wrong = 0
    
    for data in test_data:
        label, input = data

        label = label[0]
        input = cp.array(input).reshape(1, 28, 28)
        index, actual = network.evaluate(input)

        if index == label:
            correct += 1 
        else:
            wrong += 1 

    print(f"Correct: {correct}")
    print(f"Wrong: {wrong}")
    print(f"Percentage: {correct / (correct + wrong)}")


if __name__ == "__main__":
    match input("Pick Mode (Train or Load): ").lower():
        case "train":
            train_network()
        case "load":
            load_network("programs/digitrecognition/output/network.json")
