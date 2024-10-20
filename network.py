import random
import numpy as np

class Layer:
    def __init__(self, neuron_size, weight_size):
        self.size = neuron_size
        self.neurons = np.array([Neuron(weight_size) for i in range(neuron_size)])

    def feed_forward(self, prev_layer):
        #Get previous layer neurons' values
        values = np.array([neuron.value for neuron in prev_layer.neurons])

        for neuron in self.neurons:
            neuron.feed_forward(values)
            print(neuron.value)
        print()
        
    def randomize_weights(self):
        for neuron in self.neurons:
            neuron.randomize_weights() 

class Neuron:
    def __init__(self, weight_size):
        self.value = 0
        self.bias = 0

        self.weight_size = weight_size
        self.weights = np.array([0] * weight_size)
        self.randomize_weights()
        
    def feed_forward(self, values):
        #Get previous layer neurons' values
        print(values)
        print(self.weights)
        self.value = np.dot(values, self.weights) + self.bias

    def randomize_weights(self):
        self.weights = np.array([random.uniform(0.0, 1.0)] * self.weight_size)
        
        
class NeuralNetwork:
    def __init__(self, layer_size):
        self.layer_size = layer_size
        self.layers = np.array([None] * len(layer_size))

        #Add layer objects into layer array 
        self.create_input_layer()
        self.create_hidden_layers()
        self.create_output_layer()

    #Create the starting layer or the input layer
    def create_input_layer(self):
        input_layer_size = self.layer_size[0]
        self.layers[0] = Layer(input_layer_size, 0)

    #Create the middle hidden layers
    def create_hidden_layers(self):
        for i in range(1, len(self.layers) - 1):
            curr_layer_size = self.layer_size[i]
            prev_layer_size = self.layer_size[i - 1]
            self.layers[i] = Layer(curr_layer_size, prev_layer_size)

    #Create the output layer, the final layer of the network
    def create_output_layer(self):
        curr_layer_size = self.layer_size[-1]
        prev_layer_size = self.layer_size[-2]
        self.layers[-1] = Layer(curr_layer_size, prev_layer_size)

    def feed_forward(self, inputs):
        self.set_input(inputs)

        for i in range(1, len(self.layers)):
            prev_layer = self.layers[i - 1]
            self.layers[i].feed_forward(prev_layer)

    def set_input(self, inputs): 
        input_layer = self.layers[0]
        
        #Set the neuron's values in the input layer to the input values
        for i, neuron in enumerate(input_layer.neurons):
            neuron.value = inputs[i]
    
    def randomize_weights(self):
        for layer in self.layers:
            layer.randomize_weights()