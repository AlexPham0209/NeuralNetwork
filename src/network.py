import src.activation as act
from src.layers.layer import Layer
import numpy as np
import cupy as cp

import json
import random

class Model:
    def __init__(self, layers = [], input_size = (), path = ""):
        if len(path) > 0:
            self.load_data(path)
            return
        
        self.input_size = input_size
        self.layers = []
        self.add_layers(layers)

    def feed_forward(self, a):
        # If size of the input vector does not match the amount of neurons in the input layer, return None
        # Go through each layer and feed forward
        for layer in self.layers:
            a = layer.feed_forward(a)

        return a
    
    def learn(self, dataset, epoch, eta, batch_size = 1, debug = False):
        temp = list(dataset)
        for i in range(epoch):
            if debug:
                print(f"Iteration {i + 1}\n")
            # Randomly shuffles the dataset and partitions it into mini batches
            random.shuffle(temp)
            batches = [temp[j : j + batch_size] for j in range(0, len(temp), batch_size)]
                
            # Go through each mini-batch and train the neural network using each sample
            for i, batch in enumerate(batches):    
                if debug:
                    print(f"Batch {i + 1}")
                
                input, expected = [list(t) for t in zip(*batch)]
                input = cp.array(input)
                expected = cp.array(expected)

                self.backpropagation(input, expected, eta, len(batch))

    # Trains the neural network using gradient descent
    def backpropagation(self, input, expected, eta, size):
        # Feed forward algorithm to generate output values so we can evaluate the cost 
        actual, activations = self.feed_forward(input)

        # Apply backpropagation algorithm to generate error for each layer
        error = expected
        for i in range(len(self.layers) - 1, -1, -1):
            curr = self.layers[i]
            error = curr.backpropagation(error, eta, size)
    
    def calculate_cost(self, input, expected):
        actual, activations = self.feed_forward(input)
        

        return 0

    def evaluate(self, a):
        output = self.feed_forward(cp.array([a]))[0]
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
        data["layers"] = self.layer_size
        data["cost"] = self.c
        
        #Create new JSON key for each layer
        for i, layer in enumerate(self.layers):
            layer_data = dict()
            
            layer_data["type"] = "Dense"
            layer_data["weights_size"] = layer.weight_size
            layer_data["biases_size"] = layer.neuron_size
            layer_data["weights"] = layer.weights.tolist()
            layer_data["biases"] = layer.biases.tolist()

            data[i] = layer_data

        return data
        
    def load_data(self, path):
        #Read create dictionary from string with JSON data in it
        data = json.loads(open(path).read())
        self.layer_size = data["layers"]
        
        #Based on the activation_type parameter, create the activation object
        match data["activation_type"]:
            case "Sigmoid":
                self.activation = act.Sigmoid()
            case "ReLU":
                self.activation = act.ReLU()
        
        #Creates layers for neural network
        self.layers = []
        for i in range(len(self.layer_size) - 1):
            layer_data = data[str(i)]
            layer = Layer(layer_data["biases_size"], layer_data["weights_size"], self.activation)
            layer.weights = cp.array(layer_data["weights"])
            layer.biases = cp.array(layer_data["biases"])

            self.layers.append(layer)

