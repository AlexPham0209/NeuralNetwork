import sys

import json
import random
import numpy as np
import ph.layers.activation as act
import ph.network as nw
from ph.layers.dense import Dense
from ph.loss import CrossEntropy, MeanSquaredError
import cupy as cp

dataset = [
    ([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)], [1, 0, 0]),
    ([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)], [0, 1, 0]),
    ([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)], [0, 0, 1]),
]

def print_output(network, data):
    for data in dataset:
        index, vector = network.evaluate(data[0])

def train():
    global dataset 
    architecture = [
        Dense(5, act.ReLU()),
        Dense(5, act.ReLU()),
        Dense(3, act.SoftMax())
    ]
    network = nw.Model(architecture, input_size = 3, output_size = 3, loss = CrossEntropy())

    input, expected = zip(*dataset)
    input = cp.array(list(input))
    expected = cp.array(list(expected))
    print_output(network, dataset)

    network.learn(input, expected, (input, expected), 10000, 0.01, 2)
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
