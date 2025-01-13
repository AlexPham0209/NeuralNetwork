import sys
sys.path.insert(1, '../NeuralNetwork')

import json
import random
import numpy as np
import src.activation as act
import src.network as nw
from src.layers.dense import Dense

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
    architecture = [
        Dense(5, act.Sigmoid()),
        Dense(5, act.Sigmoid()),
        Dense(10, act.Sigmoid())
    ]
    network = nw.Model(architecture, input_size = 3)

    print_output(network, dataset)
    network.learn(dataset, 10000, 0.5, 10)
    print()
    print_output(network, dataset)

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

