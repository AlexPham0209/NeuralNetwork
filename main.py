import random
import numpy as np
import network as nw


network = nw.NeuralNetwork([2, 2, 3, 2, 2])

print(network.feed_forward([random.uniform(0, 1), random.uniform(0, 1)]))