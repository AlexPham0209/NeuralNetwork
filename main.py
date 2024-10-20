import numpy as np
import network as nw


network = nw.NeuralNetwork([2, 2, 3, 2, 2])

network.feed_forward([1, 2])