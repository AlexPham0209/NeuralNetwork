import cupy as cp

import json
import random

import src.activation as act
import src.loss as ls

from src.layers.layer import Layer
from src.layers.conv2d import Conv2D
from src.layers.dense import Dense
from src.layers.flatten import Flatten
from src.layers.pooling import MaxPooling

class Model:
    def __init__(self, layers = [], input_size = (), output_size = (), loss = ls.Loss(), path = ""):
        if len(path) > 0:
            self.load_data(path)
            return
        
        self.input_size = input_size
        self.output_size = output_size
        self.loss = loss

        self.layers = []
        self.add_layers(layers)

    def feed_forward(self, a):
        # If size of the input vector does not match the amount of neurons in the input layer, return None
        # Go through each layer and feed forward
        for layer in self.layers:
            a = layer.feed_forward(a)

        return a
    
    def learn(self, data, labels, epoch, eta, batch_size = 1, debug = False):
        data = cp.array(data)
        labels = cp.array(list(map(self.create_expected, labels)))

        dataset = list(zip(data.tolist(), labels.tolist()))
        for i in range(epoch):
            if debug:
                print(f"Iteration {i + 1}\n")

            # Randomly shuffles the dataset and partitions it into mini batches
            random.shuffle(dataset)
            batches = self.get_batches(dataset, batch_size)
            
            # Go through each mini-batch and train the neural network using each sample
            for i, batch in enumerate(batches):    
                if debug:
                    print(f"Batch {i + 1}")
                
                input, expected = [cp.array(t) for t in zip(*batch)]
                self.backpropagation(input, expected, eta, len(batch))

    # Trains the neural network using gradient descent
    def backpropagation(self, input, expected, eta, size):
        # Apply backpropagation algorithm to generate error for each layer
        error = self.calculate_loss_derivative(input, expected)

        for i in range(len(self.layers) - 1, -1, -1):
            curr = self.layers[i]
            error = curr.backpropagation(error, eta, size)

    def calculate_loss_derivative(self, input, expected):
        actual = self.feed_forward(input)
        
        if actual.shape != expected.shape:
            raise Exception("Size of neural network's output does not match size of expected")
        
        return self.loss.derivative(actual, expected)

    def calculate_loss(self, input, expected):
        actual = self.feed_forward(input)

        if actual.shape != expected.shape:
            raise Exception("Size of neural network's output does not match size of expected")
        
        return self.loss.loss(actual, expected)

    def evaluate(self, a):
        output = self.feed_forward(cp.array([a]))
        return cp.argmax(output), output
            
    def add_layers(self, layers):
        self.layers = []
        for i in range(len(layers)):
            self.add(layers[i])
        
    def add(self, layer):
        if len(self.layers) == 0:
            layer.input_size = self.input_size
            self.layers.append(layer)
            return
        
        prev = self.layers[-1]
        layer.input_size = prev.output_size
        self.layers.append(layer)

    def save_data(self):
        data = dict()

        #Overall neural network data
        data["layer_size"] = len(self.layers)
        data["input_size"] = self.input_size
        data["output_size"] = self.output_size

        data["loss"] = str(self.loss)
        
        #Create new JSON key for each layer
        for i, layer in enumerate(self.layers):
            curr = self.layers[i]
            data[i] = curr.save_data()

        return data
    
    def load_data(self, path):
        #Read create dictionary from string with JSON data in it
        data = json.loads(open(path).read())
        
        self.layer_size = data["layer_size"]
        self.input_size = data["input_size"]
        self.output_size = data["output_size"]

        self.loss = ls.create_loss(data["loss"])
    
        #Creates layers for neural network
        self.layers = []
        for i in range(self.layer_size):
            layer_data = data[str(i)]
            type = layer_data["type"]

            layer = None
            match type:
                case "dense":
                    layer = Dense(data = layer_data)
                case "conv2d":
                    layer = Conv2D(data = layer_data)
                case "pooling":
                    layer = MaxPooling(data = layer_data)
                case "flatten":
                    layer = Flatten(data = layer_data)
                case _:
                    raise Exception("Unknown layer type during deserialization")
                
            self.layers.append(layer)

    def create_expected(self, x):
        res = cp.zeros(self.output_size)
        res[int(x)] = 1
        return res
            
    def get_batches(self, set, batch_size):
        return [set[j : j + batch_size] for j in range(0, len(set), batch_size)]

