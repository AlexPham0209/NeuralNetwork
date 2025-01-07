import json
import random
import activation as act
import numpy as np

class Layer:
    def __init__(self, neuron_size, weight_size, activation):
        self.activation = activation
        self.neuron_size = neuron_size
        self.weight_size = weight_size

        self.values = [0] * self.neuron_size
        self.error = [0] * self.neuron_size
        
        self.randomize_weights()
        self.randomize_biases()

    def feed_forward(self, a):
        # Zip the weights and biases like this -> [(w1, b1), (w2, b1), ...]
        # Then calculate the dot product between the weights and activation + bias 
        # Matrix vector multiplication essentially
        self.values = [np.dot(a, w) + b for w, b in zip(self.weights, self.biases)]        
        return self.values
    
    def output_backpropagation(self, expected):
        # Reset error vector for layer
        self.error = [0] * self.neuron_size
        activate = self.activation.activate
        derivative = self.activation.derivative

        # Calculate error for output layer:
        # err = dA/dZ * dC/dA
        for i in range(self.neuron_size):
            cost_derivative = 2 * (activate(self.values[i]) - expected[i])
            activation_derivative = derivative(self.values[i])
            self.error[i] = cost_derivative * activation_derivative

        return self.error

    def backpropagation(self, prev):
        self.error = [0] * self.neuron_size
        derivative = self.activation.derivative

        # Calculate the error terms for the error derivative (For 1 node in each layer)
        # err(i) = dC/dZ(i) = dA(i)/dZ(i) * dZ(i + 1)/dA(i) * err(i + 1)
        for i in range(self.neuron_size):
            #Calculate dZ(i + 1)/dA(i)
            res = sum([prev.error[j] * prev.weights[j][i] for j in range(prev.neuron_size)])

            #Multiply dA(i)/dZ(i) due to chain rule
            res *= derivative(self.values[i])
            self.error[i] = res

        return self.error

    def apply_gradient(self, eta, prev):
        activate = self.activation.activate

        # Applies gradient on the weights
        # dC/dW(i) = dZ(i)/dW(i) * dC/dZ(i)
        # The error is dC/dZ(i)
        for i in range(self.neuron_size):
            for j in range(self.weight_size):
                self.weights[i][j] -= eta * activate(prev[j]) * self.error[i]

        # Applies gradient on the biases
        # dC/dB(i) = dZ(i)/dW(i) * dC/dB(i)
        # The error is dC/dZ(i)
        for i in range(self.neuron_size):
            self.biases[i] -= eta * self.error[i]

    # Randomizes all weights from 0 to 1 
    def randomize_weights(self):
        self.weights = [[random.uniform(0.0, 1.0) for j in range(self.weight_size)] for i in range(self.neuron_size)]

    # Randomizes all biases from 0 to 1 
    def randomize_biases(self):
        self.biases = [random.uniform(0.0, 1.0) for i in range(self.neuron_size)]


class NeuralNetwork:
    def __init__(self, layer_size, activation = act.Sigmoid()):
        self.layer_size = layer_size
        self.activation = activation
        self.layers = [Layer(curr, prev, activation) for curr, prev in zip(layer_size[1:], layer_size)]

    def feed_forward(self, a):
        # If size of the input vector does not match the amount of neurons in the input layer, return None
        if len(a) != self.layer_size[0]:
            return None
        
        # Go through each layer and feed fot
        for layer in self.layers:
            a = list(map(self.activation.activate, layer.feed_forward(a)))

        return a
    
    def learn(self, dataset, iterations, eta, batch_size = -1):
        temp = list(dataset)
        for i in range(iterations):
            # Randomly shuffles the dataset and partitions it into mini batches
            random.shuffle(temp)
            batches = [temp[j : j + batch_size] for j in range(0, len(temp), batch_size)]

            #Go through each mini-batch and train the neural network using each sample
            for batch in batches:
                for data in batch:
                    input, expected = data
                    self.backpropagation(input, expected, eta)
    
    # Trains the neural network using gradient descent
    def backpropagation(self, a, expected, eta = 0.5):
        # Feed forward algorithm to generate output values so we can evaluate the cost 
        actual = self.feed_forward(a)

        # Apply backpropagation algorithm to generate error for each layer
        error = self.layers[-1].output_backpropagation(expected)
        for i in range(len(self.layers) - 2, -1, -1):
            curr, prev = self.layers[i], self.layers[i + 1]
            curr.backpropagation(prev)
        
        # Apply the gradient for each layer using the error of the previous layer
        self.layers[0].apply_gradient(eta, a)
        for i in range(1, len(self.layers)):
            curr, prev = self.layers[i], self.layers[i - 1]
            curr.apply_gradient(eta, prev.values)

    def save_data(self):
        data = dict()

        data["layers"] = self.layer_size
        data["activation_type"] = str(self.activation)

        for i, layer in enumerate(self.layers):
            layer_data = dict()

            layer_data["type"] = "Dense"
            layer_data["weights_size"] = layer.weight_size
            layer_data["biases_size"] = layer.neuron_size
            layer_data["weights"] = layer.weights
            layer_data["biases"] = layer.biases

            data[i] = layer_data

        return data


    def load_data(self, path):
        data = json.loads(open(path).read())
        self.layer_size = data["layers"]

        match data["activation_type"]:
            case "Sigmoid":
                self.activation = act.Sigmoid()
            case "ReLU":
                self.activation = act.ReLU()

        self.layers = []
        for i in range(len(self.layer_size) - 1):
            layer_data = data[str(i)]
            layer = Layer(layer_data["biases_size"], layer_data["weights_size"], self.activation)
            layer.weights = layer_data["weights"]
            layer.biases = layer_data["biases"]

            self.layers.append(layer)

