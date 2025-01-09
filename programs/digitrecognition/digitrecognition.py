import sys
sys.path.insert(1, '../NeuralNetwork')

import json
import random
import numpy as np
import activation as act
import network as nw

ROW = 28
COL = 28

def read_digits(path):
    dataset = []
    with open(path) as file:
        for data in file.read().strip().split("\n"):
            data = list(map(int, data.split(",")))
            expected = data[0]
            input = [val/255 for val in data[1:]]

            expected = [0] * 10
            expected[data[0]] = 1 
            dataset.append((input, expected))

    return dataset

train_data = read_digits("programs\digitrecognition\mnist_train.csv")
test_data = read_digits("programs\digitrecognition\mnist_test.csv")

def train_network():
    network = nw.NeuralNetwork([ROW * COL, 30, 10])
    network.learn(train_data, 1, 0.5, 1)

    save_data = network.save_data()
    with open("network.json", "w") as file:
        json.dump(save_data, file, indent = 3)
    
    test(network)

    
def load_network():
    network = nw.NeuralNetwork([ROW * COL, 30, 30, 10], path = "network.json")
    test(network)

def test(network):
    global test_data
    correct = 0
    wrong = 0
    
    output = open("output.txt", "w")
    for data in test_data:
        input, expected = data
        actual = [val for val in list(map(float, network.feed_forward(input)))] 

        max_index = 0 
        for i in range(1, len(actual)):
            if actual[i] >= actual[max_index]:
                max_index = i
        
        actual = [int(i == max_index) for i in range(len(actual))]
        output.write(f"Actual: {actual}\n")
        output.write(f"Expected: {expected}\n\n")

        if actual == expected:
            correct += 1 
        else:
            wrong += 1 

    print(f"Correct: {correct}")
    print(f"Wrong: {wrong}")
    print(f"Percentage: {correct / (correct + wrong)}")


match input("Pick Mode (Train or Load): ").lower():
    case "train":
        train_network()
    case "load":
        load_network()
