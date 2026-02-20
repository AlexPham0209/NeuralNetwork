import sys

import json
import random
import numpy as np
import src.activation as act
import src.network as nw
from src.layers.dense import Dense
from src.loss import CrossEntropy, MeanSquaredError
import cupy as cp

dataset = [
    ([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)], 0),
    ([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)], 1),
    ([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)], 2),
    ([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)], 3),
    ([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)], 4),
    ([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)], 5),
    ([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)], 6),
    ([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)], 7),
    ([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)], 8), 
    ([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)], 9),
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
        Dense(5, act.ReLU()),
        Dense(5, act.ReLU()),
        Dense(10, act.SoftMax())
    ]
    network = nw.Model(architecture, input_size = 3, output_size = 10, loss = CrossEntropy())
    
    input, expected = zip(*dataset)
    input = cp.array(list(input))
    expected = cp.array(list(expected))
    print_output(network, dataset)
    
    network.learn(input, expected, 10000, 0.5, 10)
    print()
    print_output(network, dataset)
    
def load():
    global dataset
    network = nw.NeuralNetwork(data = "test.json")
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
