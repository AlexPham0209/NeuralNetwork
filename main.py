import random
import numpy as np
import network as nw


network = nw.NeuralNetwork([3, 5, 5, 5, 5], nw.Sigmoid())


# print(f"Before Training: {list(map(float, network.feed_forward(input)))}")
print(f"Before Training: {list(map(float, network.feed_forward([0.125, 0.52124, 0.623])))}")
print(f"Before Training: {list(map(float, network.feed_forward([0.923, 0.17, 0.345])))}")
print(f"Before Training: {list(map(float, network.feed_forward([0.012, 0.29, 0.854])))}")
print()

for i in range(10000):
    network.backpropagation([0.125, 0.52124, 0.623], [0, 0, 0, 1, 0], 0.5)
    network.backpropagation([0.923, 0.17, 0.345], [0, 0, 1, 0, 0], 0.5)
    network.backpropagation([0.012, 0.29, 0.854], [0, 1, 0, 0, 0], 0.5)


print(f"After Training: {list(map(float, network.feed_forward([0.125, 0.52124, 0.623])))}")
print(f"After Training: {list(map(float, network.feed_forward([0.923, 0.17, 0.345])))}")
print(f"After Training: {list(map(float, network.feed_forward([0.012, 0.29, 0.854])))}")