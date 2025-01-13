from collections import deque
import multiprocessing
import json
import random

import concurrent
import activation as act
import numpy as np

import layers.layer as Layer

class Model:
    def __init__(self, layers, input_size, path = ""):
        if len(path) > 0:
            self.load_data(path)
            return
        
        self.input_size = input_size
        self.layers = []
        self.add_layers(layers)

    def feed_forward(self, a):
        # If size of the input vector does not match the amount of neurons in the input layer, return None
        # Go through each layer and feed forward
        activations = []
        for layer in self.layers:
            a = layer.feed_forward(a)
            activations.append(a) 

        return a, activations
    
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

                for data in batch:
                    input, expected = data                    
                    self.backpropagation(input, expected)

                # Applies gradient vectors for weights and biases on the neural netowrk  
                self.apply_gradient(input, eta, len(batch))
    
    # Trains the neural network using gradient descent
    def backpropagation(self, a, expected):
        # Feed forward algorithm to generate output values so we can evaluate the cost 
        actual, activations = self.feed_forward(a)

        # Apply backpropagation algorithm to generate error for each layer
        error = expected
        for i in range(len(self.layers) - 1, -1, -1):
            curr = self.layers[i]
            error = curr.backpropagation(error)
        
        # Update gradient vector for biases and weights
        self.layers[0].update_gradient(a)
        for i in range(1, len(self.layers)):
            curr, prev = self.layers[i], self.layers[i - 1]
            curr.update_gradient(prev.out)

        
    def apply_gradient(self, a, eta, size):
        # Apply the gradient for each layer using the error of the previous layer
        for i in range(len(self.layers)):
            self.layers[i].apply_gradient(eta, size)

    def evaluate(self, a):
        output = self.feed_forward(a)[0]
        return np.argmax(output), output

    def add_layers(self, layers):
        self.layers = []
        for i in range(len(layers)):
            self.add(layers[i])
        
    def add(self, layer):
        if len(self.layers) == 0:
            print(self.input_size)
            layer.input_size = self.input_size
            self.layers.append(layer)
            return
        
        prev = self.layers[-1]
        prev.next_layer = layer
        layer.prev_layer = prev
        layer.input_size = prev.output_size
        self.layers.append(layer)

    def save_data(self):
        data = dict()

        #Overall neural network data
        data["layers"] = self.layer_size
        data["activation_type"] = str(self.activation)

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
            layer.weights = np.array(layer_data["weights"])
            layer.biases = np.array(layer_data["biases"])

            self.layers.append(layer)

        self.layers[-1].is_output = True
        self.layers[0].is_input = True

