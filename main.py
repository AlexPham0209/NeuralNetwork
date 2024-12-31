import random
import numpy as np
import network as nw


network = nw.NeuralNetwork([5, 2, 3, 2, 5])

input = [0.2512, 0.12, 0.01, 0.54, 0.6]
print(network.feed_forward(input))

for i in range(1000):
    network.backpropagation(input, [0, 1, 0, 0, 0])

print(network.feed_forward(input))