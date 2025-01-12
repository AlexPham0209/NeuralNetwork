from collections import deque
import multiprocessing
import json
import random

import concurrent
import activation as act
import numpy as np

class Layer:
    def __init__(self, neuron_size, weight_size, activation):
        self.activation = activation
        self.neuron_size = neuron_size
        self.weight_size = weight_size
        
        self.weights_gradient = np.zeros((self.neuron_size, self.weight_size))
        self.biases_gradient = np.zeros(self.neuron_size)
        
        self.randomize_weights()
        self.randomize_biases()

    def feed_forward(self, a):
        # Zip the weights and biases like this -> [(w1, b1), (w2, b1), ...]
        # Then calculate the dot product between the weights and activation + bias 
        # Matrix vector multiplication essentially
        out = self.weights.dot(a) + self.biases 
        return out
    
    def output_backpropagation(self, actual, expected):
        activate = self.activation.activate
        derivative = self.activation.derivative
        
        delta = np.zeros(self.neuron_size)

        # Calculate error for output layer:
        # err = dC/dZ = dA/dZ * dC/dA
        for i in range(self.neuron_size):
            cost_derivative = 2 * (activate(actual[i]) - expected[i])
            activation_derivative = derivative(actual[i])
            delta[i] = cost_derivative * activation_derivative

        # Apply the error calculated for current input to the total error
        return delta

    def backpropagation(self, error, prev, out):
        derivative = self.activation.derivative
        delta = np.zeros(self.neuron_size)

        # Calculate the error terms for the error derivative (For 1 node in each layer)
        # err(i) = dC/dZ(i) = dA(i)/dZ(i) * dZ(i + 1)/dA(i) * err(i + 1)
        for i in range(self.neuron_size):
            # Calculate dZ(i + 1)/dA(i)
            res = sum([error[j] * prev.weights[j][i] for j in range(prev.neuron_size)])

            # Multiply dA(i)/dZ(i) due to chain rule
            res *= derivative(out[i])

            # Adds all possible errors for every input in sample so we can average them layer 
            delta[i] = res

        return delta

    def update_gradient(self, prev, error):
        activate = self.activation.activate
        # Applies gradient on the weights
        # dC/dW(i) = dZ(i)/dW(i) * dC/dZ(i)
        # The error is dC/dZ(i)
        for i in range(self.neuron_size):
            for j in range(self.weight_size):
                self.weights_gradient[i][j] += (activate(prev[j]) * error[i])

        # Applies gradient on the biases
        # dC/dB(i) = dZ(i)/dW(i) * dC/dB(i)
        # The error is dC/dZ(i)
        for i in range(self.neuron_size):
            self.biases_gradient[i] += error[i]


    def apply_gradient(self, eta, size = 1):
        activate = self.activation.activate

        # Applies gradient on the weights
        # dC/dW(i) = dZ(i)/dW(i) * dC/dZ(i)
        # The error is dC/dZ(i)
        # Averages the gradient component
        for i in range(self.neuron_size):
            for j in range(self.weight_size):
                self.weights[i][j] -= (eta / size) * (self.weights_gradient[i][j])

        # Applies gradient on the biases
        # dC/dB(i) = dZ(i)/dW(i) * dC/dB(i)
        # The error is dC/dZ(i)
        for i in range(self.neuron_size):
            self.biases[i] -= (eta / size) * (self.biases_gradient[i])
        
        #Resets the error after applying gradient vector
        self.weights_gradient = [[0] * self.weight_size for i in range(self.neuron_size)]
        self.biases_gradient = [0] * self.neuron_size

    # Randomizes all weights from 0 to 1 
    def randomize_weights(self):
        self.weights = np.array([[random.uniform(-1.0, 1.0) for j in range(self.weight_size)] for i in range(self.neuron_size)])

    # Randomizes all biases from 0 to 1 
    def randomize_biases(self):
        self.biases = np.array([random.uniform(-1.0, 1.0) for i in range(self.neuron_size)])


class NeuralNetwork:
    def __init__(self, layer_size, activation = act.Sigmoid(), path = ""):
        if len(path) > 0:
            self.load_data(path)
            return
        
        self.layer_size = layer_size
        self.activation = activation
        self.layers = np.array([Layer(curr, prev, activation) for curr, prev in zip(layer_size[1:], layer_size)])

    def feed_forward(self, a):
        # If size of the input vector does not match the amount of neurons in the input layer, return None
        if len(a) != self.layer_size[0]:
            return None
        
        # Go through each layer and feed forward
        activations = []
        for layer in self.layers:
            out = layer.feed_forward(a)
            a = list(map(self.activation.activate, out))
            activations.append(out)

        return a, activations
    
    def learn(self, dataset, epoch, eta, batch_size = 1, multithreading = False, debug = False):
        temp = np.array(dataset)
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

                # Based on mulithreading parameter, it either processes each input in batch individually
                # Or it processes all input in the data at the same time
                if not multithreading:
                    for data in batch:
                        input, expected = data

                        if len(input) != self.layer_size[0]:
                            return
                    
                        self.backpropagation(input, expected)
                else:
                    pool = multiprocessing.Pool(processes=(multiprocessing.cpu_count() - 1))

                    for data in batch:
                        input, expected = data
                        pool.apply_async(self.backpropagation, args=(input,expected))

                    pool.close()
                    pool.join()

                # Applies gradient vectors for weights and biases on the neural netowrk  
                self.apply_gradient(input, eta, len(batch))
    
    # Trains the neural network using gradient descent
    def backpropagation(self, a, expected):
        # Feed forward algorithm to generate output values so we can evaluate the cost 
        actual, activations = self.feed_forward(a)

        # Apply backpropagation algorithm to generate error for each layer
        error = self.layers[-1].output_backpropagation(activations[-1], expected)
        errors = [0] * len(self.layers)
        errors[-1] = error

        for i in range(len(self.layers) - 2, -1, -1):
            curr, prev = self.layers[i], self.layers[i + 1]
            error = curr.backpropagation(errors[i + 1], prev, activations[i])
            errors[i] = error
        
        # Update gradient vector for biases and weights
        self.layers[0].update_gradient(a, errors[0])
        for i in range(1, len(self.layers)):
            self.layers[i].update_gradient(activations[i - 1], errors[i])

        
    def apply_gradient(self, a, eta, size):
        # Apply the gradient for each layer using the error of the previous layer
        for i in range(len(self.layers)):
            self.layers[i].apply_gradient(eta, size)

    def evaluate(self, a):
        output = self.feed_forward(a)[0]
        
        # Get the index of the component in the output vector that has the largest value
        index = 0
        for i in range(len(output)):
            if output[i] >= output[index]:
                index = i

        # Returns the max index and the values of the output layer
        return index, output
        
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

