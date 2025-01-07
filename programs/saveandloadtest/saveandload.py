import sys
sys.path.insert(1, '../NeuralNetwork')

import json
import random
import numpy as np
import activation as act
import network as nw

dataset = [
    ([0.125, 0.52124, 0.623], [0, 0, 0, 1, 0]), 
    ([0.923, 0.17, 0.345], [0, 0, 1, 0, 0]),
    ([0.012, 0.29, 0.854], [0, 1, 0, 0, 0])
]

def print_output(network, data):
        for data in dataset:
            for val in list(map(float, network.feed_forward(data[0]))):
                print(f"{val:.5f}", end = ", ")
            print()
    
def train():
    global dataset 
    network = nw.NeuralNetwork([3, 5, 5, 5, 5], act.Sigmoid())

    print_output(network, dataset)
    network.learn(dataset, 10000, 0.5, 2)
    print()
    print_output(network, dataset)

    save_data = network.save_data()
    with open("test.json", "w") as file:
        json.dump(save_data, file, indent = 3)

def load():
    global dataset
    network = nw.NeuralNetwork([3, 5, 5, 5, 5], act.Sigmoid())
    network.load_data("test.json")
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