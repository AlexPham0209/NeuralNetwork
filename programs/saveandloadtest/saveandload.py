import sys
sys.path.insert(1, '../NeuralNetwork')

import json
import random
import numpy as np
import activation as act
import network as nw

dataset = [
    ([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    ([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
    ([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
    ([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
    ([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
    ([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
    ([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
    ([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
    ([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]), 
    ([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
]

def print_output(network, data):
        for data in dataset:
            index, vector = network.evaluate(data[0])
            print(index)
            print(vector)
            print()
        print()
    
def train():
    global dataset 
    network = nw.NeuralNetwork([3, 10, 10, 10], act.Sigmoid())

    print_output(network, dataset)
    network.learn(dataset, 10000, 0.5, 10, multithreading = False)
    print()
    print_output(network, dataset)

    save_data = network.save_data()
    with open("test.json", "w") as file:
        json.dump(save_data, file, indent = 3)

def load():
    global dataset
    network = nw.NeuralNetwork([3, 5, 5, 5, 5], act.Sigmoid(), "test.json")
    print_output(network, dataset)


while True:
    mode = input("Train or Load: ")
    match mode.lower().strip():
        case "train":
            train()
        case "load":
            load()
        case _:
            break

if __name__ == "__main__":
    train()