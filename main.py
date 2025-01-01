import random
import numpy as np
import network as nw


network = nw.NeuralNetwork([3, 5, 5, 5, 5], nw.Sigmoid())

dataset = [
    ([0.125, 0.52124, 0.623], [0, 0, 0, 1, 0]), 
    ([0.923, 0.17, 0.345], [0, 0, 1, 0, 0]),
    ([0.012, 0.29, 0.854], [0, 1, 0, 0, 0])
]

for data in dataset:
    for val in list(map(float, network.feed_forward(data[0]))):
        print(f"{val:.3f}", end = ", ")
    print()

print()

network.learn(dataset, 10000, 0.5)

for data in dataset:
    for val in list(map(float, network.feed_forward(data[0]))):
        print(f"{val:.3f}", end = ", ")
    print()