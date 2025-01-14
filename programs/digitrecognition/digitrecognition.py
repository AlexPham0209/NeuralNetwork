import sys
sys.path.insert(1, '../NeuralNetwork')

import json
import random
import numpy as np
import src.activation as act
from src.layers.dense import Dense
from src.network import Model

ROW = 28
COL = 28

def read_digits(path, array = False):
    dataset = []
    with open(path) as file:
        for data in file.read().strip().split("\n"):
            data = list(map(int, data.split(",")))

            expected = data[0]
            if array:
                expected = np.zeros(10)
                expected[data[0]] = 1 

            dataset.append((np.array(data[1:])/255, expected))

    return dataset


def train_network():
    train_data = read_digits("C:/Users/RedAP/Desktop/mnist_train.csv", True)
    architecture = [
        Dense(64, act.Sigmoid()), 
        Dense(64, act.Sigmoid()), 
        Dense(10, act.Sigmoid())
    ]

    network = Model(architecture, input_size = ROW * COL)
    network.learn(train_data, 20, 0.5, 25, debug=True)
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


if __name__ == "__main__":
    match input("Pick Mode (Train or Load): ").lower():
        case "train":
            train_network()
        case "load":
            load_network()

